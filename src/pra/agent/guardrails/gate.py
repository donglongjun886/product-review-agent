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
    rule_hit_ids,
)
from pra.domain.measurement import (
    CITABLE_TYPES,
    DIM_MERCHANT_PROFILE,
    DIM_TEXT_COMPLIANCE,
)
from pra.domain.models import (
    Budget,
    Decision,
    ReviewDecision,
    RiskLevel,
    RiskType,
)

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


def _has_citable(evidence) -> bool:
    """是否存在带 ``ref_id`` 的可引用依据（POLICY_REF / CASE_PRECEDENT）。"""
    return any(e.type in CITABLE_TYPES and e.ref_id for e in evidence)


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
    return _has_citable(state.get("evidence") or [])


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
    if DIM_TEXT_COMPLIANCE in positive_dims:
        risk.append(RiskType.FALSE_CLAIM)
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


def run_decision_overlay(state: dict, proposal) -> ReviewDecision:
    """decide 收口主函数：``proposal`` 为 None 表示无提案（预算耗尽 / 降级 / LLM 失败）。"""
    hit = hard_rule_hit(state)
    if hit is not None:
        return _build_decision(
            state,
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
