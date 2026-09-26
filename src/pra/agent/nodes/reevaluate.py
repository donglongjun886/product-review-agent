"""reevaluate 节点：把新证据综合进假设状态（LLM 提案 + 确定性 apply）。

``hypotheses`` 为覆盖写，成功路径返回全集；degraded 或预算超限时不调 LLM，返回 ``{}``。
"""

from __future__ import annotations

import re

from pra.agent.guardrails.budget import budget_exceeded, bump_llm_usage
from pra.agent.guardrails.errors import SEV_CRITICAL, STEP_REEVALUATE, make_failure
from pra.agent.guardrails.llm_shell import LLMBackend, call_structured_llm
from pra.agent.guardrails.schemas import ReevaluateOutput
from pra.domain.models import Hypothesis, HypothesisStatus

__all__ = ["reevaluate_node"]

# 假设 id 序号匹配：H1..Hn
_H_ID_RE = re.compile(r"^H(\d+)$")

# LLM 步降级 failure 文案（固定字面值）
_DEGRADE_REASON = "schema 校验重试仍失败"


def _state_payload(state: dict) -> dict:
    """LLM 入参 state 子集：hypotheses / evidence 全量 + 上一轮 pending_tool_calls 摘要。"""
    return {
        "hypotheses": [h.model_dump(mode="json") for h in state.get("hypotheses") or []],
        "evidence": [e.model_dump(mode="json") for e in state.get("evidence") or []],
        "pending_tool_calls": [
            {
                "tool": c.get("tool"),
                "priority": c.get("priority"),
                "reason": c.get("reason"),
            }
            for c in (state.get("pending_tool_calls") or [])
        ],
    }


def _max_h_seq(hypotheses: list[Hypothesis]) -> int:
    """现用假设的最大 H 序号；无匹配时返回 0。"""
    seq = 0
    for h in hypotheses:
        m = _H_ID_RE.match(h.id)
        if m:
            seq = max(seq, int(m.group(1)))
    return seq


def _apply(state: dict, out: ReevaluateOutput) -> dict:
    """确定性应用 ReevaluateOutput → hypotheses **全集**（覆盖写）。"""
    hypotheses = list(state.get("hypotheses") or [])
    by_id = {h.id: h for h in hypotheses}
    updated: dict[str, Hypothesis] = {}
    for u in out.hypothesis_updates:
        # 原对象 model_copy；id 未命中则跳过
        base = updated.get(u.id) if u.id in updated else by_id.get(u.id)
        if base is None:
            continue
        updated[u.id] = base.model_copy(
            update={
                "posterior": u.posterior,
                "status": HypothesisStatus(u.status),
                "evidence_for": list(u.evidence_for),
                "evidence_against": list(u.evidence_against),
            }
        )
    result: list[Hypothesis] = [updated.get(h.id, h) for h in hypotheses]
    seq = _max_h_seq(hypotheses)
    for proposal in out.new_hypotheses:
        seq += 1
        result.append(
            Hypothesis(
                id=f"H{seq}",
                statement=proposal.statement,
                prior=proposal.prior,
                posterior=None,
                status=HypothesisStatus.PENDING,
                evidence_for=[],
                evidence_against=[],
            )
        )
    return {"hypotheses": result}


async def reevaluate_node(state: dict, config, *, llm: LLMBackend) -> dict:
    """reevaluate 图节点；``llm`` 由装配期显式注入。"""
    if state.get("degraded") or budget_exceeded(state["budget"]) is not None:
        return {}
    outcome = await call_structured_llm(
        OutputModel=ReevaluateOutput,
        node="reevaluate",
        state=_state_payload(state),
        llm=llm,
    )
    budget = bump_llm_usage(
        state["budget"], llm_calls=outcome.attempts, tokens=outcome.tokens
    )
    if outcome.model is None:
        return {
            "degraded": True,
            "failures": [
                make_failure(
                    step_type=STEP_REEVALUATE,
                    severity=SEV_CRITICAL,
                    reason=_DEGRADE_REASON,
                )
            ],
            "budget": budget,
        }
    updates = _apply(state, outcome.model)
    updates["degraded"] = False
    updates["budget"] = budget
    return updates
