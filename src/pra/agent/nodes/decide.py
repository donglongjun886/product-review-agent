"""decide 节点：最终裁决 —— 图中唯一"LLM 提案 + 确定性 overlay"双层节点。

职责：LLM 只产 **DecisionProposal 提案**；``run_decision_overlay`` 做确定性收口 ——
硬规则优先、预算超限 / 证据矛盾 / 政策不确定 / 假设不可区分 / degraded
等条件一律弃权转人工（归因码写进 overrides），PASS/REJECT 只有过 Gate 才被采纳 →
产出唯一终态 ``ReviewDecision``。DECIDED 是图内唯一终态：worker 在 invoke 返回后把
decision 落 DB（本节点不落库、不写额外状态字段）。

契约要点：
1. ``can_llm`` = not degraded ∧ 预算未超限；否则不调 LLM：
   proposal=None、budget=state["budget"]、decide_llm_failed=False。
2. can_llm 时调 ``call_structured_llm(OutputModel=DecisionProposal, node="decide")``，
   按 outcome.attempts/tokens 记账。
3. LLM 失败 → proposal=None，且 overlay 只读快照置 degraded=True + 追加一条 critical
   failure（reason="decide LLM schema 校验重试仍失败"），让降级归因码落进
   decision.overrides（可审计"谁降级了"）。budget_used 快照由 overlay 自己完成。
4. **decide 恒返回 degraded=False**：降级/失败被消费进 decision / overrides / failures
   审计，不再透传。

``_state_payload`` 以结构化 dict 注入 state 子集：hypotheses/evidence 全量
+ degraded + failures + budget 摘要（``model_dump(mode="json")``）。
"""

from __future__ import annotations

from pra.agent.guardrails.budget import budget_exceeded, bump_llm_usage
from pra.agent.guardrails.errors import SEV_CRITICAL, STEP_DECIDE, make_failure
from pra.agent.guardrails.gate import run_decision_overlay
from pra.agent.guardrails.llm_shell import LLMBackend, call_structured_llm
from pra.agent.guardrails.schemas import DecisionProposal
from pra.observability.tracing import get_tracer

__all__ = ["decide_node"]


def _state_payload(state: dict) -> dict:
    """LLM 入参 state 子集：hypotheses / evidence 全量（``model_dump(mode="json")``）
    + degraded + failures + budget 摘要（llm_calls/tool_calls/tokens + 两维限额）。"""
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


def _gate_input_summary(proposal: DecisionProposal | None) -> dict:
    """Gate 子 span 的入参摘要（decision / risk_level / confidence；只读、不参与判定）。"""
    if proposal is None:
        return {"proposal": None}
    return {
        "proposal": {
            "decision": proposal.decision,
            "risk_level": proposal.risk_level,
            "confidence": proposal.confidence,
            "risk_type": [getattr(t, "value", t) for t in proposal.risk_type],
        }
    }


def _gate_output_summary(final) -> dict:
    """Gate 子 span 的出参摘要（overlay 后终裁 + overrides 原因码，只读）。"""
    return {
        "decision": getattr(final.decision, "value", final.decision),
        "risk_level": getattr(final.risk_level, "value", final.risk_level),
        "decision_confidence": final.decision_confidence,
        "overrides": list(final.overrides),
    }


async def decide_node(state: dict, config, *, llm: LLMBackend) -> dict:
    """decide 图节点 action（graph.py 按 ``pra.agent.nodes.decide`` import）。

    ``llm`` 由 ``build_agent_graph`` 装配期显式注入（本节点不持有/不查找任何默认后端）；
    ``config`` 为 LangGraph 运行时配置（thread_id 本节点不消费，预留签名）。返回
    {"decision", "degraded": False, "budget", "failures"}（failures 只含本次新增）。
    """
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
        # 记账：按实际尝试次数（含 schema 重试）bump。
        budget = bump_llm_usage(
            state["budget"], llm_calls=outcome.attempts, tokens=outcome.tokens
        )
        proposal = outcome.model
        decide_llm_failed = outcome.model is None
    else:
        # 预算/降级 → 不再烧 LLM；overlay 用 proposal=None 兜底。
        proposal = None
        budget = state["budget"]
        decide_llm_failed = False

    # overlay 只读的本地快照（浅拷贝）：预算换为记账后对象；decide LLM 失败时把
    # degraded=True + critical failure 注进去，让降级归因码落进 decision.overrides
    # （节点自身返回仍为 degraded=False）。
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

    # Gate 子 span：Gate 不是图节点，这里只给它一层**只读**子观测 ——
    # 不新增 Graph Node、不改路由、不改 gate.py 的判定顺序与结果。
    with get_tracer().node_span("gate", input=_gate_input_summary(proposal)) as gate_span:
        final = run_decision_overlay(overlay_state, proposal)
        gate_span.update(output=_gate_output_summary(final))
    return {
        "decision": final,
        "degraded": False,
        "budget": budget,
        "failures": [failure] if decide_llm_failed else [],
    }
