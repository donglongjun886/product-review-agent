"""StateGraph 装配：复杂风险调查子图，5 节点 7 条边、单回环（节点名 = 本模块 ``N_*``）。

拓扑::

    START ──> hypothesize ──> plan ──(route_after_plan)──> tools ──> reevaluate
                                 │                          ▲         │
                                 └─> decide ──> END          │   (route_after_reevaluate)
                                                            └── continue ──> plan

- 唯一回环是 ``plan → tools → reevaluate → plan``；离开回环只有条件出口。
- DECIDED 是图唯一终态：``decide → END`` 之后无出边可继续。
- 路由/预算/收敛全为确定性纯函数（无随机、无 LLM），同 state 必同后继。
- 本模块只负责装配：``tools`` 与 ``llm`` 都是**必填的显式依赖**（无缺省值、无进程级全局、
  不回落任何桩），由组合根 ``pra.wiring``（生产）/ 评测 harness / 测试各自注入；
  ``checkpointer`` 默认 None = 不持久化。
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from langgraph.graph import END, START, StateGraph

from pra.agent.guardrails.budget import budget_exceeded
from pra.agent.guardrails.converge import is_converged
from pra.agent.guardrails.measurements import capabilities_from_tools
from pra.agent.nodes.decide import decide_node
from pra.agent.nodes.hypothesize import hypothesize_node
from pra.agent.nodes.plan import plan_node
from pra.agent.nodes.reevaluate import reevaluate_node
from pra.agent.state import AgentState
from pra.agent.tools_node import make_tools_node
from pra.observability.tracing import get_tracer

if TYPE_CHECKING:  # 仅类型（注解惰性求值，无需运行时）
    from langgraph.graph.state import CompiledStateGraph

    from pra.agent.guardrails.llm_shell import LLMBackend

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


# Node span 包装 —— 只包节点外层，不改节点实现 / 拓扑


def _enum_value(value: Any) -> Any:
    """Enum → ``.value``（普通值原样返回）；用于 span 摘要的 JSON 标量化。"""
    return getattr(value, "value", value)


def _node_input_summary(state: dict) -> dict:
    """节点入参**轻量**摘要（计数 + 关键标识；绝不把整个 state 序列化进 span）。"""
    case = state.get("case")
    return {
        "case_id": getattr(case, "case_id", None),
        "hypotheses": len(state.get("hypotheses") or []),
        "evidence": len(state.get("evidence") or []),
        "pending_tool_calls": len(state.get("pending_tool_calls") or []),
        "degraded": bool(state.get("degraded")),
    }


def _node_output_summary(update: dict) -> dict:
    """节点产物轻量摘要（计数 / 终裁标量，不塞全量对象）。"""
    summary: dict[str, Any] = {"keys": sorted(update.keys())}
    for key in ("hypotheses", "evidence", "pending_tool_calls", "tool_call_history"):
        if key in update:
            summary[key] = len(update.get(key) or [])
    decision = update.get("decision")
    if decision is not None:
        summary["decision"] = _enum_value(decision.decision)
        summary["risk_level"] = _enum_value(decision.risk_level)
    return summary


def _wrap_node(
    name: str,
    action: Callable,
    *,
    llm: LLMBackend | None = None,
    state_update: dict | None = None,
) -> Callable:
    """把节点 action 包成图节点：node span + **装配期依赖注入**（闭包工厂，图内唯一注入点）。

    - ``llm`` 非 None → 以关键字参数交给 action（LLM 节点签名 ``(state, config, *, llm)``）；
    - ``state_update`` 非 None → 并入该节点的 state 更新（图入口把测量环境能力写进 state）。

    契约：只多一层旁路观测与装配期常量注入，不改节点实现 / 拓扑 / 边 / 路由判定；
    ``functools.wraps`` 保留原签名（LangGraph 依签名判定是否传 config）。
    """

    @functools.wraps(action)
    async def _node_with_span(state: dict, config) -> dict:
        with get_tracer().node_span(name=name, input=_node_input_summary(state)) as span:
            update = (
                await action(state, config, llm=llm)
                if llm is not None
                else await action(state, config)
            )
            span.update(output=_node_output_summary(update))
        if state_update and isinstance(update, dict):
            return {**update, **state_update}
        return update

    return _node_with_span


def route_after_plan(state: AgentState) -> Literal["tools", "decide"]:
    """``plan`` 之后的条件路由（确定性纯函数）。分支顺序固定：

    ① ``degraded``（上一 LLM 步 schema 校验失败）→ decide；
    ② ``budget_exceeded(state["budget"])`` 非 None → decide（预算护栏）；
    ③ ``pending_tool_calls`` 非空（已过 dedup 的真实动作）→ tools；
    ④ 否则（conclude / 空计划 / dedup 清空视同 conclude）→ decide。

    注：条件边在 plan 写入 state 后执行，故能读到刚写的 ``pending_tool_calls``；
    ``.get`` 为纯函数防御（部分 state 直测安全）。
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
    """``reevaluate`` 之后的条件路由（确定性纯函数）。分支顺序固定：

    ① ``degraded`` → decide；② 预算超限 → decide（预算护栏）；
    ③ ``is_converged(state)`` → decide（无 PENDING/UNRESOLVED 假设，且 SUPPORTED 假设
       均有带 ref_id 的可引用依据）；④ 否则 → "continue"（map 到 plan，进入下一轮）。
    """
    if state.get("degraded"):
        return N_DECIDE
    budget = state.get("budget")
    if budget is not None and budget_exceeded(budget) is not None:
        return N_DECIDE
    if is_converged(state):
        return N_DECIDE
    return "continue"


def build_agent_graph(
    *, tools: list, llm: LLMBackend, checkpointer=None
) -> CompiledStateGraph:
    """装配并编译复杂风险调查子图。

    ``tools`` 与 ``llm`` 均为**必填的显式依赖**：没有缺省值、没有进程级全局、不回落任何桩
    —— 生产（组合根 ``pra.wiring``）、评测（入口 ``scripts/run_evaluation.py`` 与
    ``pra/evaluation/``）、测试各自注入自己的世界。

    :param tools: ``Tool`` 列表（生产 ``pra.tools.build_production_tools()``；评测/测试显式
        装配自己的工具世界），经 ``make_tools_node(tools)`` 闭包工厂注入 tools 节点。
    :param llm: 本次运行使用的 ``LLMBackend``：生产 = ``pra.wiring.build_llm_backend()`` 的
        真实网关后端；评测 = 审查员桩 / 真实后端；测试 = 替身。经 ``_wrap_node`` 以关键字
        参数注入 4 个 LLM 节点。
    :param checkpointer: LangGraph checkpointer（demo 用 ``make_memory_checkpointer()``
        的 InMemorySaver）；None = 不持久化，仅调试。
    :raises TypeError: ``tools`` 或 ``llm`` 为 None —— 装配缺陷显式失败，不静默降级。

    5 个 add_node 一律经 ``_wrap_node`` 包一层 node span（只多一层旁路观测）；测量环境能力
    由入口节点写下（见下方 ``state_update``），其后节点与条件路由直接读 state。
    """
    if tools is None:
        raise TypeError(
            "build_agent_graph 需要显式注入工具列表（tools=None）—— "
            "不再回落 InMemory 默认工具世界"
        )
    if llm is None:
        raise TypeError(
            "build_agent_graph 需要显式注入 LLM 后端（llm=None）—— 不再回落 scripted 桩"
        )

    tools_action = make_tools_node(tools)

    builder = StateGraph(AgentState)
    # 测量环境能力：唯一权威来源 = 实际装配的工具集。
    # 由入口节点写入 state（唯一写入点）：plan 的缺口提示、decide 的 Gate、reevaluate 后的
    # 收敛路由一律直接读 state，不再各自包一层注入闭包。
    capabilities = capabilities_from_tools(tools)
    builder.add_node(
        N_HYPOTHESIZE,
        _wrap_node(
            N_HYPOTHESIZE,
            hypothesize_node,
            llm=llm,
            state_update={"measurement_capabilities": capabilities},
        ),
    )
    builder.add_node(N_PLAN, _wrap_node(N_PLAN, plan_node, llm=llm))
    builder.add_node(N_TOOLS, _wrap_node(N_TOOLS, tools_action))
    builder.add_node(N_REEVALUATE, _wrap_node(N_REEVALUATE, reevaluate_node, llm=llm))
    builder.add_node(N_DECIDE, _wrap_node(N_DECIDE, decide_node, llm=llm))

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
    # reevaluate 条件边：收敛/降级/超限 → decide；否则 continue 回 plan
    # （能力表已由图入口写入 state，路由直接读 state）。
    builder.add_conditional_edges(
        N_REEVALUATE,
        route_after_reevaluate,
        {"continue": N_PLAN, N_DECIDE: N_DECIDE},
    )
    builder.add_edge(N_DECIDE, END)

    return builder.compile(checkpointer=checkpointer)
