"""API 服务层 —— **调查执行器**（总链路 A·1「HTTP 接入」的核心；未来 worker 复用入口）。

职责与设计：
- ``run_review`` 是**全链路唯一执行入口**：接收一个 ``ProductReviewCase``，走完整图
  （hypothesize → plan → tools → reevaluate ×N → decide），返回 ``ReviewRunResult``。
  本轮由 HTTP 路由**同步**调用（请求内 await 完成）；演进到 infra 阶段后，**同一个
  函数**被 MQ worker 消费 ``product_review_request`` topic 时调用（docs/00-system-design.md
  §9.3 —— Redis 幂等/限流/分布式锁、落 MySQL 属 worker 侧职责，均不进本函数），
  故本函数保持"纯执行、无传输语义"：不感知 HTTP、不感知 MQ、不落库。
- run_id 语义（O-6 拍板）：案件身份不进 AgentState，接入层映射为 LangGraph **线程维度
  thread_id = run_id**。本函数把 run_id 同时用作 thread_id 写进
  ``config = {"configurable": {"thread_id": run_id}}``；调用方未显式给 run_id 时自动生成
  ``uuid4().hex`` —— 每请求一条独立线程，天然隔离多次调查（InMemorySaver 线程状态互不
  串扰），也与 demo 脚本"RUN_{case_id}"的可读风格并存（两者都只是字符串，语义等价）。
- 图装配**懒加载 + 模块级缓存**（``get_graph``）：首次调用才 build + compile
  （``build_agent_graph(checkpointer=make_memory_checkpointer())``，默认 6 个 InMemory
  Tool + 默认 scripted LLM 桩，**无需 API key**）；之后复用同一编译图实例。图不挂在
  FastAPI app 对象上（app.py 只做路由装配），避免 app 持有重对象、也便于未来 worker
  进程直接 import 本模块复用同一份缓存。

并发说明：build_agent_graph / InMemorySaver 均为**同步、无 I/O**（不 await），同一事件
循环内检查与赋值之间不存在协程切换点，故 ``get_graph`` 的"先查缓存再构建"天然原子，
无需加锁；未来若换异步 Checkpointer（AsyncSqlite/自研 MySQL saver）需在此加锁或改
asyncio 单飞（once）模式，届时见 04 §7 选型 A 的迁移注释。

约束：build_agent_graph 不注入 llm → 保持 scripted 桩；注入真实 LLM 属未来配置化
（调用方先 ``set_llm_backend`` 或传 ``llm=`` 改此处缓存重建），本轮不引入。
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from pra.agent.checkpointer import make_memory_checkpointer
from pra.agent.state import build_initial_state
from pra.api.schemas import ReviewRunResult
from pra.domain.models import ProductReviewCase, ReviewDecision

if TYPE_CHECKING:  # 仅类型：运行时不需要 CompiledStateGraph（future annotations 惰性求值）
    from langgraph.graph.state import CompiledStateGraph

__all__ = ["run_review", "get_graph"]


# 模块级懒加载缓存：首次调用 get_graph 时装配编译，之后复用（见模块 docstring 并发说明）。
_graph: CompiledStateGraph | None = None


def get_graph() -> CompiledStateGraph:
    """懒加载返回编译图单例（默认 6 InMemory Tools + scripted LLM 桩，无 API key）。

    图实例与 FastAPI app 生命周期解耦 —— app 重建/热重载不影响已编译图（反之亦然）；
    未来真实 LLM / 外部 Checkpointer（MySQL saver）接入时改这里即可，调用方零改动。
    """
    global _graph
    if _graph is None:
        from pra.agent.graph import build_agent_graph  # 延迟 import：pra.api 不被 agent 反向依赖

        # checkpointer 默认 None = 不持久化仅调试；接入层一律注入 InMemorySaver，
        # 使 thread_id=run_id 的线程状态可查询/可断点续跑（04 §7.3 选型 A / §2.3 invoke 约定）。
        _graph = build_agent_graph(checkpointer=make_memory_checkpointer())
    return _graph


async def run_review(
    case: ProductReviewCase,
    *,
    run_id: str | None = None,
) -> ReviewRunResult:
    """执行一次完整风险调查 —— **本轮 HTTP 路由调用，未来 MQ worker 复用同一入口**。

    :param case: 审核案件（domain 输入 DTO：商品快照/商家/事件类型/机审信号）。
    :param run_id: 本次运行 ID（= LangGraph thread_id，O-6）。None → 自动生成
        ``uuid4().hex``（每请求独立线程）；调用方也可传可读形式（如 ``RUN_{case_id}``）。
    :return: ``ReviewRunResult{run_id, review_decision}``，review_decision 为图终态
        decision（三分类 + 风险等级/类型 + 置信度 + 证据链 + 假设轨迹 + 预算快照）。

    :raises RuntimeError: 图执行完成但终态缺少 decision（理论不可达 —— decide 是图唯一
        终态出口；HTTP 层捕获转 500，未来 worker 捕获转失败重试/死信）。

    执行流程：解析 run_id → 构造 config（thread_id=run_id）→ 取缓存编译图 →
    ``await app.ainvoke(build_initial_state(case), config)`` 同步跑完整图 →
    读取终态 decision 打包返回。每次调用都传 ``build_initial_state(case)`` 全量初始态
    （04 §2.3 invoke 约定），多次调查间的隔离由唯一 thread_id 保证。
    """
    resolved_run_id = run_id or uuid4().hex
    config = {"configurable": {"thread_id": resolved_run_id}}
    app = get_graph()

    final_state = await app.ainvoke(build_initial_state(case), config)
    decision: ReviewDecision | None = final_state.get("decision")
    if decision is None:
        raise RuntimeError(
            f"调查图执行完成但终态缺少 decision（run_id={resolved_run_id}, "
            f"case_id={case.case_id}）—— 违反 'decide 为图唯一终态出口' 契约"
        )
    return ReviewRunResult(run_id=resolved_run_id, review_decision=decision)
