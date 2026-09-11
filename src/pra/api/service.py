"""调查执行器 —— 全链路唯一执行入口（HTTP 路由调用，未来 worker 复用）。

``run_review`` 接收 ``ProductReviewCase``，走完整图（hypothesize → plan → tools → reevaluate
×N → decide），返回 ``ReviewRunResult``；保持纯执行、无传输语义：不感知 HTTP/MQ、不落库
（落库闭环在 ``pra.infra.persist_service``）。

run_id 语义：案件身份不进 AgentState，接入层映射为 LangGraph 线程维度 ``thread_id = run_id``；
缺省自动生成 ``uuid4().hex`` —— 每请求独立线程，InMemorySaver 线程状态互不串扰（也可传可读
形式如 ``RUN_{case_id}``）。图装配为懒加载 + 模块级缓存（``get_graph``），首次调用才
``build_agent_graph(tools=build_production_tools(), checkpointer=make_memory_checkpointer())``
（生产工具世界：商品与商家读 MySQL、案例与政策读真实 RAG；scripted LLM 桩，无需 API key），
图不挂在 FastAPI app 上。单测不连库/不连 Chroma —— ``tests/conftest.py`` 把生产装配钉回
InMemory 世界。

并发：build_agent_graph / InMemorySaver 均为同步、无 I/O（不 await），同一事件循环内检查与赋值
之间无协程切换点，故「先查缓存再构建」天然原子，无需加锁；未来换异步 Checkpointer
（AsyncSqlite/自研 MySQL saver）需在此加锁或改 asyncio 单飞模式。约束：不注入 llm → 保持
scripted 桩；注入真实 LLM 属未来配置化（调用方先 ``set_llm_backend`` 或传 ``llm=`` 并重建缓存）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from pra.agent.checkpointer import make_memory_checkpointer
from pra.agent.state import build_initial_state
from pra.api.schemas import ReviewRunResult
from pra.domain.models import ProductReviewCase, ReviewDecision
from pra.observability.tracing import (
    TraceContext,
    experiment_name,
    get_tracer,
    trace_id_from_run_id,
)

if TYPE_CHECKING:  # 仅类型：运行时不需要 CompiledStateGraph（future annotations 惰性求值）
    from langgraph.graph.state import CompiledStateGraph

__all__ = ["run_review", "get_graph"]


# 模块级懒加载缓存：首次调用 get_graph 时装配编译，之后复用（见模块 docstring 并发说明）。
_graph: CompiledStateGraph | None = None


def get_graph() -> CompiledStateGraph:
    """懒加载返回编译图单例（scripted LLM 桩 + 生产工具世界，无 API key）。

    工具世界 = ``build_production_tools()``（商品/商家读 MySQL，案例/政策读真实 RAG）——
    HTTP/生产入口读真库是刻意的：单测走 ``build_tools()`` 的 InMemory 世界，
    ``tests/conftest.py`` 的 autouse fixture 把生产装配钉回 InMemory，故测试/CI 不连库。

    图实例与 FastAPI app 生命周期解耦 —— app 重建/热重载不影响已编译图；接真实 LLM / 外部
    Checkpointer（MySQL saver）时改这里即可，调用方零改动。
    """
    global _graph
    if _graph is None:
        from pra.agent.graph import (
            build_agent_graph,  # 延迟 import：pra.api 不被 agent 反向依赖
        )
        from pra.tools import build_production_tools  # 函数内 import：便于测试替换装配

        # checkpointer 默认 None = 不持久化仅调试；接入层一律注入 InMemorySaver，
        # 使 thread_id=run_id 的线程状态可查询/可断点续跑。
        _graph = build_agent_graph(
            tools=build_production_tools(), checkpointer=make_memory_checkpointer()
        )
    return _graph


async def run_review(
    case: ProductReviewCase,
    *,
    run_id: str | None = None,
) -> ReviewRunResult:
    """执行一次完整风险调查 —— HTTP 路由调用，未来 MQ worker 复用同一入口。

    :param case: 审核案件（domain 输入 DTO：商品快照/商家/事件类型/机审信号）。
    :param run_id: 本次运行 ID（= LangGraph thread_id）；None → 自动 ``uuid4().hex``。
    :return: ``ReviewRunResult{run_id, review_decision}``，review_decision 为图终态 decision
        （三分类 + 风险等级/类型 + 置信度 + 证据链 + 假设轨迹 + 预算快照）。

    :raises RuntimeError: 图执行完成但终态缺少 decision（理论不可达 —— decide 是图唯一终态
        出口；HTTP 层捕获转 500，worker 捕获转失败重试/死信）。
    """
    resolved_run_id = run_id or uuid4().hex
    config = {"configurable": {"thread_id": resolved_run_id}}
    app = get_graph()

    # Root trace：trace_id = run_id 映射（32-hex 原样，否则确定性 uuid5）→ Langfuse trace
    # 可与 MySQL review_run.run_id 硬对齐。常驻服务不 per-request flush（SDK 后台批量上报）。
    root_ctx = TraceContext(
        trace_id=trace_id_from_run_id(resolved_run_id),
        name="review",
        session_id=None,
        version=experiment_name(),
        metadata={
            "case_id": case.case_id,
            "run_id": resolved_run_id,
            "event_type": case.event_type,
            "source": "http",
        },
        tags=["env:local", "source:http"],
        input={"case_id": case.case_id},
    )
    with get_tracer().trace_root(root_ctx) as root:
        final_state = await app.ainvoke(build_initial_state(case), config)
        decision: ReviewDecision | None = final_state.get("decision")
        if decision is not None:
            root.update(
                output={
                    "decision": decision.decision.value,
                    "risk_level": decision.risk_level.value,
                }
            )
    if decision is None:
        raise RuntimeError(
            f"调查图执行完成但终态缺少 decision（run_id={resolved_run_id}, "
            f"case_id={case.case_id}）—— 违反 'decide 为图唯一终态出口' 契约"
        )
    return ReviewRunResult(run_id=resolved_run_id, review_decision=decision)
