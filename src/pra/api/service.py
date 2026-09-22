"""调查执行器 —— 纯执行、不落库的图入口（无 DB 场景与测试/演示用）。

``run_review`` 接收 ``ProductReviewCase``，走完整图（hypothesize → plan → tools → reevaluate
×N → decide），返回 ``ReviewRunResult``；保持纯执行、无传输语义：不感知 HTTP/MQ、不落库。
HTTP 路由走 ``pra.infra.persist_service.process_review``（受理分流 + 落库闭环），**不调本函数**；
``run_review`` 的真实调用方是 2 个 demo 脚本（``scripts/demo_api.py`` /
``scripts/demo_langfuse_trace.py``）与测试。

run_id 语义：案件身份不进 AgentState，接入层映射为 LangGraph 线程维度 ``thread_id = run_id``；
缺省自动生成 ``uuid4().hex`` —— 每请求独立线程，InMemorySaver 线程状态互不串扰（也可传可读
形式如 ``RUN_{case_id}``）。图装配由组合根 ``pra.wiring.get_production_graph`` 提供（模块级单例，
首次调用才装配生产工具世界：商品与商家读 MySQL、案例与政策读真实 RAG；LLM 后端由
``pra.wiring.build_llm_backend`` 读 ``Settings`` 构造并经 ``build_agent_graph(llm=...)`` 显式注入，
缺 ``DEEPSEEK_API_KEY`` 抛 ``RuntimeError``），图不挂在 FastAPI app 上。单测不连库/不连 Chroma、
不调真实模型 —— ``tests/conftest.py`` 把生产装配钉回 InMemory 世界与确定性 LLM 桩。
"""

from __future__ import annotations

from uuid import uuid4

from pra.agent.state import build_initial_state
from pra.api.schemas import ReviewRunResult
from pra.domain.models import ProductReviewCase, ReviewDecision
from pra.observability.tracing import (
    TraceContext,
    experiment_name,
    get_tracer,
    trace_id_from_run_id,
)
from pra.wiring import get_production_graph

__all__ = ["run_review"]


async def run_review(
    case: ProductReviewCase,
    *,
    run_id: str | None = None,
) -> ReviewRunResult:
    """执行一次完整风险调查 —— 纯执行、不落库（无 DB 场景与测试/演示用）。

    :param case: 审核案件（domain 输入 DTO：商品快照/商家/事件类型/机审信号）。
    :param run_id: 本次运行 ID（= LangGraph thread_id）；None → 自动 ``uuid4().hex``。
    :return: ``ReviewRunResult{run_id, review_decision}``，review_decision 为图终态 decision
        （三分类 + 风险等级/类型 + 置信度 + 证据链 + 假设轨迹 + 预算快照）。

    :raises RuntimeError: 图装配期缺 LLM 凭据（``pra.wiring.build_llm_backend`` 缺
        ``DEEPSEEK_API_KEY``）；或图执行完成但终态缺少 decision（理论不可达 —— decide 是图唯一
        终态出口）。
    """
    resolved_run_id = run_id or uuid4().hex
    config = {"configurable": {"thread_id": resolved_run_id}}
    app = get_production_graph()

    # Root trace：trace_id = run_id 映射（32-hex 原样，否则确定性 uuid5）→ Langfuse trace
    # 可与 MySQL review_run.run_id 硬对齐。本入口不在 HTTP 路径上（source=local）；
    # 带落库的 HTTP 路径见 ``pra.infra.persist_service.run_and_persist``（source=http）。
    root_ctx = TraceContext(
        trace_id=trace_id_from_run_id(resolved_run_id),
        name="review",
        session_id=None,
        version=experiment_name(),
        metadata={
            "case_id": case.case_id,
            "run_id": resolved_run_id,
            "event_type": case.event_type,
            "source": "local",
        },
        tags=["env:local", "source:local"],
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
