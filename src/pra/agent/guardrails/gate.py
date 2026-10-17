"""决策收口：LLM 只出提案，最终结论由本模块按事实确定性算出。

输入 ``DecisionProposal`` + state 的事实通道 → ``ReviewDecision``；改判原因写入
``overrides``（空 = 未改判）。顺序：硬规则 → 弃权归因 → 无提案补 R5 → PASS/REJECT Gate
→ 采纳。
"""

from __future__ import annotations

from pra.agent.guardrails.budget import budget_exceeded, snapshot_budget
from pra.agent.guardrails.hard_rules import hard_rule_hit
from pra.agent.guardrails.measurements import (
    RULE_BRAND_WORD,
    RULE_EVASION_WORD,
    CoverageReport,
    coverage_report,
    has_citable_evidence,
    rule_hit_ids,
)
from pra.agent.state import merge_evidence
from pra.domain.measurement import (
    DEFAULT_EVIDENCE_WEIGHT,
    DIM_MERCHANT_PROFILE,
    DIM_TEXT_COMPLIANCE,
)
from pra.domain.models import (
    Budget,
    Decision,
    Evidence,
    ReviewDecision,
    RiskLevel,
    RiskType,
)
from pra.screening.engine import RULE_EVIDENCE_SOURCE, triage
from pra.screening.rule_engine import terms
from pra.tools.policy_search.tool import POLICY_REF_TYPE

DECISION_CONFIDENCE = 1.0

# overrides 原因码
R1_HARD_RULE = "R1_HARD_RULE"
R2_REJECT_GATE_FAIL = "R2_REJECT_GATE_FAIL"
R3_BUDGET_EXHAUSTED = "R3_BUDGET_EXHAUSTED"
R3_MEASUREMENT_MISSING = "R3_MEASUREMENT_MISSING"
R3_DIMENSION_UNMEASURABLE = "R3_DIMENSION_UNMEASURABLE"
R3_POSITIVE_INSUFFICIENT = "R3_POSITIVE_INSUFFICIENT"
R4_PASS_GATE_FAIL = "R4_PASS_GATE_FAIL"
R5_DEGRADED_OR_FAILED_STEP = "R5_DEGRADED_OR_FAILED_STEP"
R6_INFRA_UNAVAILABLE = "R6_INFRA_UNAVAILABLE"


# 谓词


def coverage_of(state) -> CoverageReport:
    """从 state 的事实通道构造覆盖报告。"""
    return coverage_report(
        state.get("case"),
        state.get("evidence") or [],
        state.get("measurement_capabilities"),
    )


def _rule_positive_dims(case) -> frozenset[str]:
    """规则侧阳性维度：R-102 / R-302 命中 → ``text_compliance``。"""
    hits = rule_hit_ids(case)
    return (
        frozenset({DIM_TEXT_COMPLIANCE})
        if hits & {RULE_BRAND_WORD, RULE_EVASION_WORD}
        else frozenset()
    )


def pass_gate(state) -> bool:
    """PASS Gate：无阳性证据、无规则侧阳性、required 维度全部覆盖时返回 True；``case`` 缺失返回 False。"""
    if state.get("case") is None:
        return False
    cov = coverage_of(state)
    if cov.positive:
        return False
    if _rule_positive_dims(state.get("case")):
        return False
    return not (cov.missing or cov.unmeasurable)


def reject_gate(state) -> bool:
    """REJECT Gate：命中 R-302 规避词且存在带 ``ref_id`` 的可引用依据时返回 True；``case`` 缺失返回 False。"""
    if state.get("case") is None:
        return False
    if RULE_EVASION_WORD not in rule_hit_ids(state.get("case")):
        return False
    return has_citable_evidence(state.get("evidence") or [])


def _finalize_risk_level(state, proposal) -> RiskLevel:
    """风险等级：提案声明 ``risk_level`` 则用之，否则按阳性证据的最高权重派生。"""
    if proposal is not None and proposal.risk_level:
        return RiskLevel(proposal.risk_level)
    top = max(
        (float(e.weight or 0.0) for items in coverage_of(state).positive.values() for e in items),
        default=0.0,
    )
    if top >= 0.8:
        return RiskLevel.HIGH
    if top >= 0.5:
        return RiskLevel.MEDIUM
    if top >= 0.2:
        return RiskLevel.LOW
    return RiskLevel.NONE


def _finalize_risk_type(state, proposal) -> list:
    """风险类型：提案有非空 ``risk_type`` 则用之，否则按阳性证据所在维度派生；无命中返回 ``[]``。"""
    if proposal is not None and proposal.risk_type:
        return list(proposal.risk_type)
    positive_dims = set(coverage_of(state).positive)
    risk: list = []
    if DIM_MERCHANT_PROFILE in positive_dims:
        risk.append(RiskType.EVASION_PATTERN)
    return risk


# 组装与 overlay 收口


def _build_decision(
    state: dict,
    *,
    decision: Decision,
    risk_level: RiskLevel,
    risk_type: list,
    overrides: list,
) -> ReviewDecision:
    """组装 ``ReviewDecision``，取值来自确定性参数与 state 的事实通道。"""
    evidence = list(state.get("evidence") or [])
    policy = sorted(
        {
            str(e.extra["policy_id"])
            for e in evidence
            if e.type == "POLICY_REF"
            and (e.extra or {}).get("policy_id") is not None
        }
    )
    budget = state.get("budget")
    return ReviewDecision(
        decision=decision,
        risk_level=risk_level,
        risk_type=list(risk_type),
        decision_confidence=DECISION_CONFIDENCE,
        evidence=evidence,
        policy=policy,
        hypothesis_trace=list(state.get("hypotheses") or []),
        budget_used=snapshot_budget(budget) if budget is not None else Budget(),
        overrides=list(overrides),
    )


def abstention_codes(state, cov: CoverageReport) -> list:
    """HUMAN_REVIEW 弃权原因码：预算超限 / 测量缺口 / 不可测维度 / ``state["degraded"]``，命中全部收集。"""
    codes: list = []
    budget = state.get("budget")
    if budget is not None and budget_exceeded(budget) is not None:
        codes.append(R3_BUDGET_EXHAUSTED)
    if cov.missing:
        codes.append(R3_MEASUREMENT_MISSING)
    if cov.unmeasurable:
        codes.append(R3_DIMENSION_UNMEASURABLE)
    if state.get("degraded"):
        codes.append(R5_DEGRADED_OR_FAILED_STEP)
    return codes


def _human_review(state, proposal, overrides: list) -> ReviewDecision:
    """转人工的统一组装。"""
    return _build_decision(
        state,
        decision=Decision.HUMAN_REVIEW,
        risk_level=_finalize_risk_level(state, proposal),
        risk_type=_finalize_risk_type(state, proposal),
        overrides=overrides,
    )


def _policy_citation(*, policy_id: str, value: str) -> Evidence:
    """规则层确定性引用的 POLICY_REF 证据（source=ScreeningRuleEngine）。"""
    return Evidence(
        type=POLICY_REF_TYPE,
        source=RULE_EVIDENCE_SOURCE,
        value=value,
        weight=DEFAULT_EVIDENCE_WEIGHT,
        ref_id=policy_id,
        extra={"policy_id": policy_id},
    )


def _hard_rule_citation(case) -> Evidence:
    """R1 命中的可引用依据：黑名单命中 → ``terms.BLACKLIST_POLICY_ID`` 的 POLICY_REF 证据。"""
    brand = getattr(getattr(case, "product", None), "brand", None)
    policy_id = terms.BLACKLIST_POLICY_ID
    return _policy_citation(
        policy_id=policy_id, value=f"品牌黑名单命中：brand={brand!r}，依据 {policy_id}"
    )


def _evasion_citations(state) -> list[Evidence]:
    """R-302 规避词命中的可引用依据：``terms.EVASION_POLICY_ID``；merchant_profile 阳性叠加从重条款。"""
    case = state.get("case")
    hits = [h for h in triage(case).hits if h.rule_id == RULE_EVASION_WORD] if case else []
    if not hits:
        return []
    policy_id = terms.EVASION_POLICY_ID
    citations = [
        _policy_citation(
            policy_id=policy_id,
            value=f"R-302 规避模糊词命中：{hits[0].detail}，依据 {policy_id}",
        )
    ]
    if DIM_MERCHANT_PROFILE in coverage_of(state).positive:
        aggravated = terms.EVASION_AGGRAVATED_POLICY_ID
        citations.append(
            _policy_citation(
                policy_id=aggravated,
                value=f"R-302 规避词命中且商家历史画像阳性（叠加从重），依据 {aggravated}",
            )
        )
    return citations


def run_decision_overlay(state: dict, proposal) -> ReviewDecision:
    """decide 收口主函数：``proposal`` 为 None 表示无提案（预算耗尽 / 降级 / LLM 失败）。"""
    hit = hard_rule_hit(state)
    if hit is not None:
        citations = [_hard_rule_citation(state.get("case")), *_evasion_citations(state)]
        evidence = merge_evidence(list(state.get("evidence") or []), citations)
        return _build_decision(
            {**state, "evidence": evidence},
            decision=Decision.REJECT,
            risk_level=RiskLevel.HIGH,
            risk_type=list(hit.risk_types),
            overrides=[R1_HARD_RULE],
        )

    cov = coverage_of(state)
    overrides = abstention_codes(state, cov)

    if proposal is None and not overrides:
        overrides.append(R5_DEGRADED_OR_FAILED_STEP)

    if overrides:
        return _human_review(state, proposal, overrides)

    if proposal.decision == "PASS" and not pass_gate(state):
        return _human_review(state, proposal, [R4_PASS_GATE_FAIL])
    if proposal.decision == "REJECT" and not reject_gate(state):
        codes = [R2_REJECT_GATE_FAIL]
        if cov.positive:
            codes.append(R3_POSITIVE_INSUFFICIENT)
        return _human_review(state, proposal, codes)

    if proposal.decision == "REJECT":
        state = {
            **state,
            "evidence": merge_evidence(
                list(state.get("evidence") or []), _evasion_citations(state)
            ),
        }
    return _build_decision(
        state,
        decision=Decision(proposal.decision),
        risk_level=RiskLevel(proposal.risk_level),
        risk_type=list(proposal.risk_type),
        overrides=[],
    )


__all__ = [
    "R1_HARD_RULE",
    "R2_REJECT_GATE_FAIL",
    "R3_BUDGET_EXHAUSTED",
    "R3_DIMENSION_UNMEASURABLE",
    "R3_MEASUREMENT_MISSING",
    "R3_POSITIVE_INSUFFICIENT",
    "R4_PASS_GATE_FAIL",
    "R5_DEGRADED_OR_FAILED_STEP",
    "R6_INFRA_UNAVAILABLE",
    "abstention_codes",
    "coverage_of",
    "pass_gate",
    "reject_gate",
    "run_decision_overlay",
]
