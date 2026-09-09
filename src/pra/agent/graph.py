"""StateGraph 装配 —— 复杂风险调查子图（5 节点 7 边单回环；docs/04-graph-design.md §2.2 落地）。

拓扑（节点名 = 本模块 N_* 常量）：:

    START ──> hypothesize ──> plan ──(route_after_plan)──> tools ──> reevaluate
                                 │                          ▲         │
                                 └─> decide ──> END          │   (route_after_reevaluate)
                                                            └── continue ──> plan

- 5 个节点：hypothesize / plan / tools / reevaluate / decide；5 节点间的有向弧恰 7 条
  （不含 START 入口边）：hypothesize→plan、plan→tools、plan→decide、tools→reevaluate、
  reevaluate→continue(→plan)、reevaluate→decide、decide→END。
- 唯一回环是 ``plan → tools → reevaluate → plan``；离开回环只有条件出口
  （converged / conclude / dedup 清空 / 预算超限 / degraded），见 04 §6 终止性论证。
- **DECIDED 是图唯一终态**：decide 之后仅剩 ``decide → END``，无任何出边可继续
  （03 T-8：decide 产出 decision 即结束）。
- 路由/预算/收敛全为**确定性纯函数**（无随机、无 LLM），同 state 必同后继
  （04 §6.5：不存在"同样的 state 反复走不同分支"的非确定性死循环）。
- 本模块只负责装配；节点逻辑在各自节点模块，tools 节点经 ``make_tools_node(tools)``
  闭包工厂注入 Tool 集（ToolRegistry 注册在闭包内，04 §9 tools_node）。
- checkpointer 默认 None = 不持久化（仅调试）；demo/联调用 ``make_memory_checkpointer()``
  注入 InMemorySaver（04 §7.3 选型 A）。``llm`` 参数用于注入自定义 LLMBackend
  （``set_llm_backend``，进程级全局后端 —— 见 llm_shell 契约 §4.1；缺省用 scripted 桩）。
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from langgraph.graph import END, START, StateGraph

from pra.agent.guardrails.budget import budget_exceeded
from pra.agent.guardrails.converge import is_converged
from pra.agent.guardrails.llm_shell import set_llm_backend
from pra.agent.nodes.decide import decide_node
from pra.agent.nodes.hypothesize import hypothesize_node
from pra.agent.nodes.plan import plan_node
from pra.agent.nodes.reevaluate import reevaluate_node
from pra.agent.state import AgentState
from pra.agent.tools_node import make_tools_node
from pra.observability.tracing import get_tracer
from pra.tools import build_tools

if TYPE_CHECKING:  # 仅类型（from __future__ import annotations：注解惰性求值，无需运行时）
    from langgraph.graph.state import CompiledStateGraph

# 节点名常量 —— 与 add_node 名、条件边 map 的 key/target 一一对应（路由表唯一事实源）
N_HYPOTHESIZE = "hypothesize"
N_PLAN = "plan"
N_TOOLS = "tools"
N_REEVALUATE = "reevaluate"
N_DECIDE = "decide"

__all__ = [
    "N_HYPOTHESIZE",
    "N_PLAN",
    "N_TOOLS",
    "N_REEVALUATE",
    "N_DECIDE",
    "route_after_plan",
    "route_after_reevaluate",
    "build_agent_graph",
]


# --------------------------------------------------------------------------------------
# Node span 包装（docs/09 §4.2）—— 只包节点外层，不改节点实现 / 不改拓扑
# --------------------------------------------------------------------------------------


def _enum_value(value: Any) -> Any:
    """Enum → ``.value``（普通值原样返回）；仅用于 span 摘要的 JSON 标量化。"""
    return getattr(value, "value", value)


def _node_input_summary(state: dict) -> dict:
    """节点入参**轻量**摘要（绝不把整个 state 序列化进 span）。

    对齐 ``persist_service._node_input_summary`` 的口径思路（计数 + 关键标识）。
    """
    case = state.get("case")
    return {
        "case_id": getattr(case, "case_id", None),
        "hypotheses": len(state.get("hypotheses") or []),
        "evidence": len(state.get("evidence") or []),
        "pending_tool_calls": len(state.get("pending_tool_calls") or []),
        "degraded": bool(state.get("degraded")),
    }


def _node_output_summary(update: dict) -> dict:
    """节点产物轻量摘要（计数 / 终裁标量；不塞全量对象）。"""
    summary: dict[str, Any] = {"keys": sorted(update.keys())}
    for key in ("hypotheses", "evidence", "pending_tool_calls", "tool_call_history"):
        if key in update:
            summary[key] = len(update.get(key) or [])
    decision = update.get("decision")
    if decision is not None:
        summary["decision"] = _enum_value(decision.decision)
        summary["risk_level"] = _enum_value(decision.risk_level)
    return summary


def _wrap_node(name: str, action: Callable) -> Callable:
    """把节点 action 包进 ``node_span(name=...)``（闭包工厂，保持 graph.py 简洁）。

    契约：**不改节点内部实现、不改拓扑/边/路由、不改返回值** —— 只多一层旁路观测；
    ``functools.wraps`` 保留原签名（LangGraph 依签名判定是否传 config）。
    tools 节点同样包装，其内部 tool span 自然成为本 span 的子观测。
    """

    @functools.wraps(action)
    async def _node_with_span(state: dict, config) -> dict:
        with get_tracer().node_span(name=name, input=_node_input_summary(state)) as span:
            update = await action(state, config)
            span.update(output=_node_output_summary(update))
        return update

    return _node_with_span


def route_after_plan(state: AgentState) -> Literal["tools", "decide"]:
    """``plan`` 之后的条件路由（04 §2.2 / graph-mvp-contracts §9.1；确定性纯函数）。

    分支顺序固定：
    ① ``degraded``（上一 LLM 步 schema 校验失败，§5 降级）→ decide；
    ② ``budget_exceeded(state["budget"])`` 非 None → decide（预算护栏）；
    ③ ``pending_tool_calls`` 非空（已过 dedup 的真实动作）→ tools；
    ④ 否则（conclude / 空计划 / dedup 清空视同 conclude）→ decide。

    注：langgraph 条件边在 plan 写入 state 后执行，故能读到 plan 刚写的
    ``pending_tool_calls``；``.get`` 为纯函数防御（部分 state 直测安全）。
    """
    if state.get("degraded"):
        return N_DECIDE
    budget = state.get("budget")
    if budget is not None and budget_exceeded(budget) is not None:
        return N_DECIDE
    if state.get("pending_tool_calls"):
        return N_TOOLS
    return N_DECIDE


def route_after_reevaluate(state: AgentState) -> Literal["continue", "decide"]:
    """``reevaluate`` 之后的条件路由（04 §2.2 / graph-mvp-contracts §9.1；确定性纯函数）。

    分支顺序固定：
    ① ``degraded`` → decide；
    ② ``budget_exceeded(state["budget"])`` 非 None → decide（预算护栏）；
    ③ ``is_converged(state)`` → decide（证据链已可交 decide：无 PENDING/UNRESOLVED
       假设 且 SUPPORTED 假设均有带 ref_id 的可引用依据，converge.py）；
    ④ 否则 → "continue"（map 到 plan，进入下一轮 plan→tools→reevaluate 迭代）。
    """
    if state.get("degraded"):
        return N_DECIDE
    budget = state.get("budget")
    if budget is not None and budget_exceeded(budget) is not None:
        return N_DECIDE
    if is_converged(state):
        return N_DECIDE
    return "continue"


def build_agent_graph(*, tools: list | None = None, checkpointer=None, llm=None) -> CompiledStateGraph:
    """装配并编译复杂风险调查子图（graph-mvp-contracts §9.1 / 04 §2.2 落地）。

    :param tools: 6 个 ``Tool`` 列表；默认 None → ``pra.tools.build_tools()``
        （6 个 InMemory 工具，开箱可测）。经 ``make_tools_node(tools)`` 闭包工厂注入
        tools 节点。
    :param checkpointer: LangGraph checkpointer（demo 用 ``make_memory_checkpointer()``
        的 InMemorySaver，04 §7.3）；None = 不持久化，仅调试。
    :param llm: 可选 ``LLMBackend`` 注入；非 None → ``llm_shell.set_llm_backend(llm)``
        （进程级全局后端，见 llm_shell 契约 §4.1 —— 调用方负责适时
        ``set_llm_backend(None)`` 恢复默认 scripted 桩）；None/缺省 = 保持当前后端
        （默认 scripted 桩，无需 API key）。

    装配：``StateGraph(AgentState)`` + 5 个 add_node（hypothesize/plan/reevaluate/decide
    为模块级节点函数直接 import；"tools" 用闭包 ``tools_action``）+
    边：START→hypothesize→plan；plan 条件边（route_after_plan，
    map {tools: tools, decide: decide}）；tools→reevaluate；reevaluate 条件边
    （route_after_reevaluate，map {continue: plan, decide: decide}）；decide→END。
    返回 ``builder.compile(checkpointer=checkpointer)``。

    **观测（docs/09 §4.2）**：5 个 add_node 一律经 ``_wrap_node`` 包一层 node span
    （name = 节点名常量，input = 轻量状态摘要）—— 只多一层旁路观测，节点内部实现 /
    拓扑 / 边 / 路由 / 返回值全部不变（图编译产物与 ``ainvoke`` / ``astream`` 语义不变）。
    """
    if tools is None:
        tools = build_tools()
    if llm is not None:
        set_llm_backend(llm)

    tools_action = make_tools_node(tools)

    builder = StateGraph(AgentState)
    builder.add_node(N_HYPOTHESIZE, _wrap_node(N_HYPOTHESIZE, hypothesize_node))
    builder.add_node(N_PLAN, _wrap_node(N_PLAN, plan_node))
    builder.add_node(N_TOOLS, _wrap_node(N_TOOLS, tools_action))
    builder.add_node(N_REEVALUATE, _wrap_node(N_REEVALUATE, reevaluate_node))
    builder.add_node(N_DECIDE, _wrap_node(N_DECIDE, decide_node))

    # 静态边：START→hypothesize→plan；tools→reevaluate；decide→END（唯一终态出口）
    builder.add_edge(START, N_HYPOTHESIZE)
    builder.add_edge(N_HYPOTHESIZE, N_PLAN)
    # plan 条件边：真实动作 → tools；降级/超限/conclude/空计划 → decide
    builder.add_conditional_edges(
        N_PLAN,
        route_after_plan,
        {N_TOOLS: N_TOOLS, N_DECIDE: N_DECIDE},
    )
    builder.add_edge(N_TOOLS, N_REEVALUATE)
    # reevaluate 条件边：收敛/降级/超限 → decide；否则 continue 回 plan（唯一回环）
    builder.add_conditional_edges(
        N_REEVALUATE,
        route_after_reevaluate,
        {"continue": N_PLAN, N_DECIDE: N_DECIDE},
    )
    builder.add_edge(N_DECIDE, END)

    return builder.compile(checkpointer=checkpointer)
