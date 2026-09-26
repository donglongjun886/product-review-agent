"""decide 节点：LLM 产 ``DecisionProposal``，``run_decision_overlay`` 确定性收口成 ``ReviewDecision``。

``can_llm`` = not degraded ∧ 预算未超限；不调或调用失败时 ``proposal=None``。返回
``{"decision", "degraded": False, "budget", "failures"}``（failures 只含本次新增）。
"""

from __future__ import annotations

from pra.agent.guardrails.budget import budget_exceeded, bump_llm_usage
from pra.agent.guardrails.errors import SEV_CRITICAL, STEP_DECIDE, make_failure
from pra.agent.guardrails.gate import run_decision_overlay
from pra.agent.guardrails.llm_shell import LLMBackend, call_structured_llm
from pra.agent.guardrails.schemas import DecisionProposal

__all__ = ["decide_node"]


def _state_payload(state: dict) -> dict:
    """LLM 入参 state 子集：hypotheses / evidence 全量 + degraded + failures + budget 摘要。"""
    budget = state.get("budget")
    if budget is not None:
        budget_summary = {
            "llm_calls": budget.llm_calls,
            "tool_calls": budget.tool_calls,
            "tokens": budget.tokens,
            "limits": {
                "max_llm_calls": budget.limits.max_llm_calls,
                "max_tool_calls": budget.limits.max_tool_calls,
            },
        }
    else:
        budget_summary = None
    return {
        "hypotheses": [
            h.model_dump(mode="json") for h in state.get("hypotheses") or []
        ],
        "evidence": [e.model_dump(mode="json") for e in state.get("evidence") or []],
        "degraded": bool(state.get("degraded")),
        "failures": state.get("failures") or [],
        "budget": budget_summary,
    }


async def decide_node(state: dict, config, *, llm: LLMBackend) -> dict:
    """decide 图节点；``llm`` 由装配期显式注入，``config`` 为 LangGraph 运行时配置（本节点不消费）。"""
    can_llm = (
        not state["degraded"]
        and budget_exceeded(state["budget"]) is None
    )
    if can_llm:
        outcome = await call_structured_llm(
            OutputModel=DecisionProposal,
            node="decide",
            state=_state_payload(state),
            llm=llm,
        )
        budget = bump_llm_usage(
            state["budget"], llm_calls=outcome.attempts, tokens=outcome.tokens
        )
        proposal = outcome.model
        decide_llm_failed = outcome.model is None
    else:
        proposal = None
        budget = state["budget"]
        decide_llm_failed = False

    # overlay 只读快照：换入记账后的 budget，失败时注入 degraded + critical failure
    overlay_state = dict(state)
    overlay_state["budget"] = budget
    failure = None
    if decide_llm_failed:
        failure = make_failure(
            step_type=STEP_DECIDE,
            severity=SEV_CRITICAL,
            reason="decide LLM schema 校验重试仍失败",
        )
        overlay_state["degraded"] = True
        overlay_state["failures"] = [*state["failures"], failure]

    final = run_decision_overlay(overlay_state, proposal)
    return {
        "decision": final,
        "degraded": False,
        "budget": budget,
        "failures": [failure] if decide_llm_failed else [],
    }
