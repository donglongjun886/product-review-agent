"""StateGraph 装配：复杂风险调查子图（节点名 = 本模块 ``N_*``）。

拓扑::

    START ──> hypothesize ──> plan ──(route_after_plan)──> tools ──> reevaluate
                                 │                          ▲         │
                                 └─> decide ──> END          │   (route_after_reevaluate)
                                                            └── continue ──> plan
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

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

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from pra.agent.guardrails.llm_shell import LLMBackend

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


def _wrap_node(
    action: Callable,
    *,
    llm: LLMBackend | None = None,
    state_update: dict | None = None,
) -> Callable:
    """把节点 action 包成图节点（装配期依赖注入）。

    ``llm`` 非 None → 以关键字参数传给 action；``state_update`` 非 None → 并入节点返回的 state 更新。
    """

    @functools.wraps(action)
    async def _node(state: dict, config) -> dict:
        update = (
            await action(state, config, llm=llm)
            if llm is not None
            else await action(state, config)
        )
        if state_update and isinstance(update, dict):
            return {**update, **state_update}
        return update

    return _node


def route_after_plan(state: AgentState) -> Literal["tools", "decide"]:
    """``plan`` 之后的条件路由：``degraded`` 或预算超限 → ``"decide"``；
    ``pending_tool_calls`` 非空 → ``"tools"``；否则 ``"decide"``。
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
    """``reevaluate`` 之后的条件路由：``degraded``、预算超限或 ``is_converged(state)`` →
    ``"decide"``；否则 ``"continue"``。
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

    :param tools: ``Tool`` 列表，经 ``make_tools_node(tools)`` 注入 tools 节点。
    :param llm: 本次运行使用的 ``LLMBackend``，经 ``_wrap_node`` 注入 4 个 LLM 节点。
    :param checkpointer: LangGraph checkpointer；None = 不持久化。
    :raises TypeError: ``tools`` 或 ``llm`` 为 None。
    """
    if tools is None:
        raise TypeError(
            "build_agent_graph 需要显式注入工具列表（tools=None）—— "
            "生产/评测/测试各自装配自己的工具世界"
        )
    if llm is None:
        raise TypeError(
            "build_agent_graph 需要显式注入 LLM 后端（llm=None）—— 请在装配处注入 LLMBackend"
        )

    tools_action = make_tools_node(tools)

    builder = StateGraph(AgentState)
    capabilities = capabilities_from_tools(tools)
    builder.add_node(
        N_HYPOTHESIZE,
        _wrap_node(
            hypothesize_node,
            llm=llm,
            state_update={"measurement_capabilities": capabilities},
        ),
    )
    builder.add_node(N_PLAN, _wrap_node(plan_node, llm=llm))
    builder.add_node(N_TOOLS, _wrap_node(tools_action))
    builder.add_node(N_REEVALUATE, _wrap_node(reevaluate_node, llm=llm))
    builder.add_node(N_DECIDE, _wrap_node(decide_node, llm=llm))

    builder.add_edge(START, N_HYPOTHESIZE)
    builder.add_edge(N_HYPOTHESIZE, N_PLAN)
    builder.add_conditional_edges(
        N_PLAN,
        route_after_plan,
        {N_TOOLS: N_TOOLS, N_DECIDE: N_DECIDE},
    )
    builder.add_edge(N_TOOLS, N_REEVALUATE)
    builder.add_conditional_edges(
        N_REEVALUATE,
        route_after_reevaluate,
        {"continue": N_PLAN, N_DECIDE: N_DECIDE},
    )
    builder.add_edge(N_DECIDE, END)

    return builder.compile(checkpointer=checkpointer)
