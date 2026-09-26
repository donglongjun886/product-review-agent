"""hypothesize 节点：初始风险假设生成（LLM 语义步）。

把 ``ProductReviewCase`` 转为初始假设集（``prior`` 为未经调查的怀疑度 0..1）；
返回 hypotheses / degraded / failures / budget（failures 只含本次新增）。
"""

from __future__ import annotations

from pra.agent.guardrails.budget import budget_exceeded, bump_llm_usage
from pra.agent.guardrails.errors import SEV_CRITICAL, STEP_HYPOTHESIZE, make_failure
from pra.agent.guardrails.llm_shell import LLMBackend, call_structured_llm
from pra.agent.guardrails.schemas import HypothesizeOutput
from pra.domain.models import Hypothesis, HypothesisStatus

__all__ = ["hypothesize_node"]

# LLM 步降级 failure 文案（固定字面值）
_DEGRADE_REASON = "schema 校验重试仍失败"


def _state_payload(state: dict) -> dict:
    """LLM 入参 state 子集：仅 case 全量（domain 对象已 ``model_dump(mode="json")``）。"""
    case = state["case"]
    return {"case": case.model_dump(mode="json")}


def _apply_hypotheses(out: HypothesizeOutput) -> list[Hypothesis]:
    """LLM 提案 → Hypothesis 列表：按序编号 H1..Hn，``status=PENDING``、``posterior=None``；``evidence_hint`` / ``rationale`` 丢弃。"""
    return [
        Hypothesis(
            id=f"H{idx + 1}",
            statement=proposal.statement,
            prior=proposal.prior,
            posterior=None,
            status=HypothesisStatus.PENDING,
            evidence_for=[],
            evidence_against=[],
        )
        for idx, proposal in enumerate(out.hypotheses)
    ]


async def hypothesize_node(state: dict, config, *, llm: LLMBackend) -> dict:
    """hypothesize 图节点：生成初始假设集；``llm`` 由装配期显式注入。"""
    if budget_exceeded(state["budget"]) is not None:
        return {
            "hypotheses": [],
            "degraded": True,
            "failures": [
                make_failure(
                    step_type=STEP_HYPOTHESIZE,
                    severity=SEV_CRITICAL,
                    reason="预算已超限，hypothesize 无法生成假设（止损转人工）",
                )
            ],
            "budget": state["budget"],
        }

    outcome = await call_structured_llm(
        OutputModel=HypothesizeOutput,
        node="hypothesize",
        state=_state_payload(state),
        llm=llm,
    )
    budget = bump_llm_usage(
        state["budget"], llm_calls=outcome.attempts, tokens=outcome.tokens
    )
    if outcome.model is None:
        return {
            "hypotheses": [],
            "degraded": True,
            "failures": [
                make_failure(
                    step_type=STEP_HYPOTHESIZE,
                    severity=SEV_CRITICAL,
                    reason=_DEGRADE_REASON,
                )
            ],
            "budget": budget,
        }
    return {
        "hypotheses": _apply_hypotheses(outcome.model),
        "degraded": False,
        "budget": budget,
    }
