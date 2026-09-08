"""确定性决策 Gate / overlay（guardrails/gate.py）单测 —— code review 拍板语义固化。

覆盖清单：
- dc 公式锚点：5 证据 + H2 SUPPORTED posterior=0.91 → 0.87；空态 → 0.10 基线；
- run_decision_overlay 各分支 overrides 码：R1 / R3_BUDGET_EXHAUSTED /
  R5_DEGRADED_OR_FAILED_STEP / R3_CRITICAL_CONFLICT /
  R3_HYPOTHESES_INDISTINGUISHABLE / R3_KEY_TOOL_FAILED（warn 不触发）/
  R4_PASS_GATE_FAIL / R2_REJECT_GATE_FAIL；
- policy_indeterminate 只拦 REJECT 侧（干净案 False 且 pass_gate=True；风险案无
  POLICY_REF True；有 POLICY_REF False；CASE_PRECEDENT 不替代政策）；
- 空 hypotheses/evidence 全谓词安全不抛、无 vacuous PASS。
"""

from __future__ import annotations

import pytest

from pra.agent.guardrails import gate, hard_rules
from pra.agent.guardrails.errors import make_failure
from pra.agent.guardrails.gate import (
    R1_HARD_RULE,
    R2_REJECT_GATE_FAIL,
    R3_BUDGET_EXHAUSTED,
    R3_CRITICAL_CONFLICT,
    R3_HYPOTHESES_INDISTINGUISHABLE,
    R3_KEY_TOOL_FAILED,
    R3_POLICY_UNCERTAIN,
    R4_PASS_GATE_FAIL,
    R5_DEGRADED_OR_FAILED_STEP,
    contradiction_detect,
    finalize_decision_confidence,
    high_priority,
    pass_gate,
    policy_indeterminate,
    run_decision_overlay,
)
from pra.agent.guardrails.schemas import DecisionProposal
from pra.domain.models import Budget, Decision, HypothesisStatus, RiskLevel, RiskType
from helpers import (
    budget_exhausted_state,
    dc_anchor_state,
    ev,
    hp,
    make_case,
)

_BRAND = "某违禁品牌"


def _proposal(
    decision: str = "HUMAN_REVIEW",
    risk_level: str = "LOW",
    risk_type: list | None = None,
    confidence: float = 0.8,
) -> DecisionProposal:
    return DecisionProposal(
        decision=decision,  # type: ignore[arg-type]
        risk_level=risk_level,  # type: ignore[arg-type]
        risk_type=risk_type or [],
        confidence=confidence,
        evidence_ids=[],
        policy=[],
        rationale="test",
    )


# ---------------------------------------------------------------------------
# dc 公式锚点
# ---------------------------------------------------------------------------


def test_dc_anchor_five_evidence_posterior_091():
    """走查锚点：5 证据 + H2 SUPPORTED posterior=0.91 → 0.87（0.45*0.91+0.25*5/8
    +0.20*1+0.10，round 2，无矛盾扣分）。"""
    assert finalize_decision_confidence(dc_anchor_state()) == 0.87


def test_dc_empty_state_baseline():
    """hypotheses/evidence 空/缺失 → top=0/completeness=0/citation=0 → 0.10 基线。"""
    assert finalize_decision_confidence({}) == 0.10
    assert finalize_decision_confidence({"hypotheses": [], "evidence": []}) == 0.10


def test_dc_clips_to_one_and_penalizes_conflict():
    """公式 clip 到 [0,1] 且矛盾扣 0.2（此处不细算，只钉边界行为）。"""
    conflict_state = {
        "hypotheses": [],
        "evidence": [
            ev("IMAGE_SIMILARITY", value="similarity=0.99", weight=0.99, ref_id="img1"),
            ev("MERCHANT_HISTORY", value="0 similar / 0 removals / 0 title-relisting, credit=90",
               weight=0.85, ref_id="M1", extra={"removals": 0, "title": 0}),
        ],
    }
    assert contradiction_detect(conflict_state) is True
    assert finalize_decision_confidence(conflict_state) < finalize_decision_confidence(
        {"hypotheses": [], "evidence": []}
    )


# ---------------------------------------------------------------------------
# 谓词层
# ---------------------------------------------------------------------------


def test_high_priority_threshold_and_none_prior():
    """prior>=0.3 才算高优先；0.29 与 None 排除（T-1 口径，仅 Gate 用）。"""
    hs = [
        hp("H1", prior=0.3, status=HypothesisStatus.PENDING),
        hp("H2", prior=0.29, status=HypothesisStatus.PENDING),
        hp("H3", prior=None, status=HypothesisStatus.PENDING),
    ]
    got = [h.id for h in high_priority(hs)]
    assert got == ["H1"]
    assert high_priority([]) == []
    assert high_priority(None) == []


def test_contradiction_requires_strong_sim_and_clean_merchant():
    """矛盾：强相似(>=0.85) ∧ MERCHANT 干净(removals==0 & title==0)。"""
    strong_clean = {
        "evidence": [
            ev("IMAGE_SIMILARITY", value="similarity=0.91", weight=0.91, ref_id="img1"),
            ev("MERCHANT_HISTORY", value="0 similar / 0 removals / 0 title-relisting, credit=90",
               weight=0.85, ref_id="M1", extra={"removals": 0, "title": 0}),
        ]
    }
    dirty_merchant = {
        "evidence": [
            ev("IMAGE_SIMILARITY", value="similarity=0.91", weight=0.91, ref_id="img1"),
            ev("MERCHANT_HISTORY", value="23 similar / 5 removals / 3 title-relisting, credit=62",
               weight=0.85, ref_id="M1", extra={"removals": 5, "title": 3}),
        ]
    }
    weak_sim = {
        "evidence": [
            ev("IMAGE_SIMILARITY", value="similarity=0.42", weight=0.42, ref_id="img1"),
            ev("MERCHANT_HISTORY", value="0 similar / 0 removals / 0 title-relisting, credit=90",
               weight=0.85, ref_id="M1", extra={"removals": 0, "title": 0}),
        ]
    }
    assert contradiction_detect(strong_clean) is True
    assert contradiction_detect(dirty_merchant) is False
    assert contradiction_detect(weak_sim) is False
    assert contradiction_detect({"evidence": []}) is False


def test_contradiction_merchant_extra_missing_treated_not_clean():
    """extra 缺失/未回填的 MERCHANT_HISTORY 按"不干净" → 不触发矛盾（不误转人工）。"""
    st = {
        "evidence": [
            ev("IMAGE_SIMILARITY", value="similarity=0.91", weight=0.91, ref_id="img1"),
            ev("MERCHANT_HISTORY", value="干净商家？无 extra", weight=0.85, ref_id="M1"),
        ]
    }
    assert contradiction_detect(st) is False


def test_pass_gate_clean_case():
    """PASS Gate：高优先非空 ∧ 全部 REFUTED 且有 evidence_against ∧ 无关键失败/矛盾。"""
    st = {
        "hypotheses": [
            hp("H1", prior=0.5, status=HypothesisStatus.REFUTED, evidence_against=["e1"]),
            hp("H2", prior=0.4, status=HypothesisStatus.REFUTED, evidence_against=["e2"]),
        ],
        "evidence": [],
        "failures": [],
        "tool_call_history": [],
    }
    assert pass_gate(st) is True


def test_pass_gate_refuted_without_evidence_against_is_false():
    """REFUTED 但 evidence_against 为空 = "没查到"而非"证伪" → 不 PASS。"""
    st = {
        "hypotheses": [hp("H1", prior=0.5, status=HypothesisStatus.REFUTED)],
        "evidence": [],
        "failures": [],
        "tool_call_history": [],
    }
    assert pass_gate(st) is False


def test_pass_gate_empty_hypotheses_false():
    """空 hypotheses 防御：不误 PASS。"""
    assert pass_gate({"hypotheses": [], "evidence": [], "failures": [], "tool_call_history": []}) is False


# ---------------------------------------------------------------------------
# policy_indeterminate —— 只拦 REJECT 侧
# ---------------------------------------------------------------------------


def _supported_state(*, prior: float, evidence_types: list[str]) -> dict:
    evidence = []
    if "POLICY_REF" in evidence_types:
        evidence.append(ev("POLICY_REF", value="POLICY_3.2 v2 条款：x", weight=0.9,
                           ref_id="POLICY_3.2_v2_c1", extra={"policy_id": "POLICY_3.2"}))
    if "CASE_PRECEDENT" in evidence_types:
        evidence.append(ev("CASE_PRECEDENT", value="case_1001", weight=0.8, ref_id="case_1001"))
    if "IMAGE_SIMILARITY" in evidence_types:
        evidence.append(ev("IMAGE_SIMILARITY", value="similarity=0.91", weight=0.91,
                           ref_id="img1"))
    return {
        "hypotheses": [hp("H1", prior=prior, status=HypothesisStatus.SUPPORTED,
                          posterior=0.91, evidence_for=["x"])],
        "evidence": evidence,
    }


def test_policy_indeterminate_clean_all_refuted_false():
    """干净案（全 REFUTED、无 SUPPORTED）无 POLICY_REF → False（不拦 PASS 候选）。"""
    st = {
        "hypotheses": [
            hp("H1", prior=0.5, status=HypothesisStatus.REFUTED, evidence_against=["e"]),
        ],
        "evidence": [],
    }
    assert policy_indeterminate(st) is False
    assert pass_gate({**st, "failures": [], "tool_call_history": []}) is True


def test_policy_indeterminate_supported_without_policy_ref_true():
    """风险案：SUPPORTED 高优先假设但无 POLICY_REF → True（政策缺失拦 REJECT 侧）。"""
    assert policy_indeterminate(_supported_state(prior=0.5, evidence_types=["IMAGE_SIMILARITY"])) is True


def test_policy_indeterminate_case_precedent_does_not_substitute_policy():
    """CASE_PRECEDENT 不替代政策 —— 只有先例无政策仍 True。"""
    assert policy_indeterminate(_supported_state(prior=0.5, evidence_types=["CASE_PRECEDENT"])) is True


def test_policy_indeterminate_supported_with_policy_ref_false():
    """SUPPORTED + 带 ref_id 的 POLICY_REF → False。"""
    assert policy_indeterminate(_supported_state(prior=0.5, evidence_types=["POLICY_REF"])) is False


def test_policy_indeterminate_low_prior_supported_false():
    """低 prior（0.2）SUPPORTED 非高优先 → 不拦（policy 只约束拟 REJECT 的高优先案）。"""
    assert policy_indeterminate(_supported_state(prior=0.2, evidence_types=[])) is False


# ---------------------------------------------------------------------------
# run_decision_overlay —— overrides 码
# ---------------------------------------------------------------------------


def test_overlay_r1_hard_rule_forces_reject(monkeypatch):
    """R1 硬规则命中 → 强制 REJECT / HIGH / 1.0 / [R1_HARD_RULE]，覆盖 PASS 提案。"""
    monkeypatch.setattr(hard_rules, "BLACKLISTED_BRANDS", frozenset({_BRAND}))
    st = {
        "case": make_case(brand=_BRAND),
        "hypotheses": [],
        "evidence": [],
        "budget": Budget(),
        "failures": [],
    }
    final = run_decision_overlay(st, _proposal(decision="PASS", risk_level="NONE"))
    assert final.decision == Decision.REJECT
    assert final.risk_level == RiskLevel.HIGH
    assert final.decision_confidence == 1.0
    assert final.overrides == [R1_HARD_RULE]
    assert final.risk_type == [RiskType.POTENTIAL_IP_RISK]


def test_overlay_budget_exhausted_r3():
    """预算超限 → HUMAN + [R3_BUDGET_EXHAUSTED]（proposal=None 也不补 R5）。"""
    final = run_decision_overlay(budget_exhausted_state(), None)
    assert final.decision == Decision.HUMAN_REVIEW
    assert final.overrides == [R3_BUDGET_EXHAUSTED]
    assert final.decision_confidence == 0.10  # 空证据基线


def test_overlay_degraded_r5():
    """degraded=True → HUMAN + [R5_DEGRADED_OR_FAILED_STEP]。"""
    st = dc_anchor_state()
    st["degraded"] = True
    final = run_decision_overlay(st, None)
    assert final.decision == Decision.HUMAN_REVIEW
    assert final.overrides == [R5_DEGRADED_OR_FAILED_STEP]
    assert final.risk_level == RiskLevel.HIGH  # 派生：最高 SUPPORTED posterior 0.91>=0.8


def test_overlay_critical_conflict_r3():
    """关键矛盾（强相似 + 干净商家）→ HUMAN + [R3_CRITICAL_CONFLICT]。"""
    st = {
        "hypotheses": [],
        "evidence": [
            ev("IMAGE_SIMILARITY", value="similarity=0.91", weight=0.91, ref_id="img1"),
            ev("MERCHANT_HISTORY", value="0 similar / 0 removals / 0 title-relisting, credit=90",
               weight=0.85, ref_id="M1", extra={"removals": 0, "title": 0}),
        ],
        "budget": Budget(),
        "failures": [],
    }
    final = run_decision_overlay(st, None)
    assert final.decision == Decision.HUMAN_REVIEW
    assert final.overrides == [R3_CRITICAL_CONFLICT]
    assert final.risk_type == [RiskType.POTENTIAL_IP_RISK]  # 由强相似派生


def test_overlay_indistinguishable_hypotheses_r3():
    """多假设不可分（互斥假设同组证据）→ HUMAN + [R3_HYPOTHESES_INDISTINGUISHABLE]。"""
    st = {
        "hypotheses": [
            hp("H1", prior=0.5, status=HypothesisStatus.SUPPORTED, posterior=0.8,
               evidence_for=["a", "b"]),
            hp("H2", prior=0.4, status=HypothesisStatus.SUPPORTED, posterior=0.7,
               evidence_for=["b", "a"]),  # 集合相等
        ],
        "evidence": [ev("POLICY_REF", value="POLICY_3.2 v2 条款：x", weight=0.9,
                        ref_id="POLICY_3.2_v2_c1")],  # 避免同时触发 policy_indeterminate
        "budget": Budget(),
        "failures": [],
    }
    final = run_decision_overlay(st, None)
    assert final.decision == Decision.HUMAN_REVIEW
    assert final.overrides == [R3_HYPOTHESES_INDISTINGUISHABLE]


def test_overlay_key_tool_failed_r3():
    """未解决的 critical Tool 失败 → HUMAN + [R3_KEY_TOOL_FAILED]。"""
    st = {
        "hypotheses": [],
        "evidence": [],
        "budget": Budget(),
        "degraded": False,
        "failures": [
            make_failure(step_type="TOOL_CALL", severity="critical", reason="boom",
                         tool="ImageAnalysisTool", seq=1),
        ],
        "tool_call_history": [
            {"tool": "ProductTool", "status": "ok", "seq": 1, "args": {}, "tokens": 0},
        ],
    }
    final = run_decision_overlay(st, None)
    assert final.decision == Decision.HUMAN_REVIEW
    assert final.overrides == [R3_KEY_TOOL_FAILED]


def test_overlay_warn_failure_does_not_trigger():
    """severity=warn 的工具失败不触发 R3/R5 —— REJECT 提案正常过 Gate 被采纳。"""
    st = dc_anchor_state()
    st["failures"] = [
        make_failure(step_type="TOOL_CALL", severity="warn", reason="非关键抖动",
                     tool="ImageAnalysisTool", seq=1),
    ]
    final = run_decision_overlay(st, _proposal(decision="REJECT", risk_level="HIGH"))
    assert final.decision == Decision.REJECT
    assert final.overrides == []


def test_overlay_reject_accepted_when_gate_passes():
    """REJECT Gate 全满足（dc=0.87 等）→ 采纳 REJECT，overrides=[]、policy 确定性提取。"""
    final = run_decision_overlay(
        dc_anchor_state(), _proposal(decision="REJECT", risk_level="HIGH")
    )
    assert final.decision == Decision.REJECT
    assert final.overrides == []
    assert final.decision_confidence == 0.87
    assert final.policy == ["POLICY_3.2"]  # 从 POLICY_REF.extra["policy_id"] 提取
    assert len(final.evidence) == 5  # 全量证据链
    assert len(final.hypothesis_trace) == 2


def test_overlay_reject_gate_fail_dc_below_threshold_r2():
    """REJECT 提案但 dc<0.7 → HUMAN + [R2_REJECT_GATE_FAIL]（dc 只约束 REJECT 侧）。"""
    st = {
        "hypotheses": [
            hp("H1", prior=0.4, status=HypothesisStatus.SUPPORTED, posterior=0.5,
               evidence_for=["e1"]),
        ],
        "evidence": [ev("POLICY_REF", value="POLICY_3.2 v2 条款：x", weight=0.9,
                        ref_id="POLICY_3.2_v2_c1", extra={"policy_id": "POLICY_3.2"})],
        "budget": Budget(),
        "failures": [],
        "tool_call_history": [],
    }
    assert finalize_decision_confidence(st) == 0.56  # 0.45*0.5+0.25*1/8+0.2*1+0.1
    final = run_decision_overlay(st, _proposal(decision="REJECT", risk_level="HIGH"))
    assert final.decision == Decision.HUMAN_REVIEW
    assert final.overrides == [R2_REJECT_GATE_FAIL]
    assert final.decision_confidence == 0.56


def test_overlay_supported_without_policy_r3_policy_uncertain():
    """风险案：SUPPORTED 高优先 + 无 POLICY_REF（政策缺失）→ abstention 先于 Gate，
    HUMAN + [R3_POLICY_UNCERTAIN]（R2 的"无可引用"分支被该 R3 吸收 —— 设计如此）。"""
    st = {
        "hypotheses": [
            hp("H1", prior=0.5, status=HypothesisStatus.SUPPORTED, posterior=0.91,
               evidence_for=["e1"]),
        ],
        "evidence": [ev("IMAGE_SIMILARITY", value="similarity=0.91", weight=0.91, ref_id="img1")],
        "budget": Budget(),
        "failures": [],
        "tool_call_history": [],
    }
    assert policy_indeterminate(st) is True
    final = run_decision_overlay(st, _proposal(decision="REJECT", risk_level="HIGH"))
    assert final.decision == Decision.HUMAN_REVIEW
    assert final.overrides == [R3_POLICY_UNCERTAIN]


def test_overlay_pass_gate_fail_when_supported_exists_r4():
    """PASS 提案但存在 SUPPORTED 高优先假设 → HUMAN + [R4_PASS_GATE_FAIL]。"""
    st = {
        "hypotheses": [
            hp("H1", prior=0.5, status=HypothesisStatus.SUPPORTED, posterior=0.8,
               evidence_for=["e1"]),
        ],
        "evidence": [ev("POLICY_REF", value="POLICY_3.2 v2 条款：x", weight=0.9,
                        ref_id="POLICY_3.2_v2_c1")],  # 隔离：避免 policy_indeterminate
        "budget": Budget(),
        "failures": [],
        "tool_call_history": [],
    }
    final = run_decision_overlay(st, _proposal(decision="PASS", risk_level="NONE"))
    assert final.decision == Decision.HUMAN_REVIEW
    assert final.overrides == [R4_PASS_GATE_FAIL]


def test_overlay_pass_accepted_on_clean_case():
    """干净案（全部高优先 REFUTED 有据）→ PASS 被采纳；政策缺失不拦 PASS 侧。"""
    st = {
        "hypotheses": [
            hp("H1", prior=0.5, status=HypothesisStatus.REFUTED, evidence_against=["e1"]),
            hp("H2", prior=0.4, status=HypothesisStatus.REFUTED, evidence_against=["e2"]),
        ],
        "evidence": [],
        "budget": Budget(),
        "failures": [],
        "tool_call_history": [],
    }
    assert policy_indeterminate(st) is False
    final = run_decision_overlay(st, _proposal(decision="PASS", risk_level="NONE", confidence=0.9))
    assert final.decision == Decision.PASS
    assert final.overrides == []
    assert final.decision_confidence == 0.10  # PASS 不受低 dc 约束（确定性重算值）


def test_overlay_human_proposal_adopted_without_overrides():
    """HUMAN 提案无需过 Gate → 直接采纳（overrides=[]），dc 为确定性重算值。"""
    final = run_decision_overlay(dc_anchor_state(), _proposal(decision="HUMAN_REVIEW"))
    assert final.decision == Decision.HUMAN_REVIEW
    assert final.overrides == []
    assert final.decision_confidence == 0.87


# ---------------------------------------------------------------------------
# 空 hypotheses/evidence —— 全谓词安全、无 vacuous PASS
# ---------------------------------------------------------------------------


def test_empty_state_proposal_none_r5_fallback():
    """空态 + proposal=None → HUMAN + [R5_DEGRADED_OR_FAILED_STEP]（兜底不抛）。"""
    final = run_decision_overlay({}, None)
    assert final.decision == Decision.HUMAN_REVIEW
    assert final.overrides == [R5_DEGRADED_OR_FAILED_STEP]
    assert final.decision_confidence == 0.10


def test_empty_state_pass_proposal_not_vacuous():
    """空态 + PASS 提案 → 不误 PASS：pass_gate=False → HUMAN + [R4_PASS_GATE_FAIL]。"""
    final = run_decision_overlay({}, _proposal(decision="PASS", risk_level="NONE"))
    assert final.decision == Decision.HUMAN_REVIEW
    assert final.overrides == [R4_PASS_GATE_FAIL]
    assert final.overrides != []


def test_empty_state_reject_proposal_r2():
    """空态 + REJECT 提案 → reject_gate 前提不满足 → HUMAN + [R2_REJECT_GATE_FAIL]。"""
    final = run_decision_overlay({}, _proposal(decision="REJECT", risk_level="HIGH"))
    assert final.decision == Decision.HUMAN_REVIEW
    assert final.overrides == [R2_REJECT_GATE_FAIL]


def test_overlay_combined_budget_and_degraded_order():
    """abstention 码全量收集且顺序固定：预算码在前、R5 在后。"""
    st = budget_exhausted_state()
    st["degraded"] = True
    final = run_decision_overlay(st, None)
    assert final.decision == Decision.HUMAN_REVIEW
    assert final.overrides == [R3_BUDGET_EXHAUSTED, R5_DEGRADED_OR_FAILED_STEP]


def test_abstention_codes_empty_state_no_raise():
    """_abstention_codes 空/缺键 state 安全返回 []。"""
    assert gate._abstention_codes({}) == []
    assert gate._abstention_codes({"budget": None}) == []
    assert gate._abstention_codes({"budget": Budget(), "hypotheses": None, "evidence": None,
                                   "failures": None, "degraded": False}) == []


@pytest.mark.parametrize(
    ("pred", "expected"),
    [
        (lambda s: gate.high_priority(s.get("hypotheses")), []),
        (lambda s: gate.pass_gate(s), False),
        (lambda s: gate.contradiction_detect(s), False),
        (lambda s: gate.policy_indeterminate(s), False),
        (lambda s: gate.indistinguishable_hypotheses(s), False),
        (lambda s: gate.key_evidence_complete(s), True),  # 无关键失败 → 完整（空真）
        (lambda s: gate.evidence_sufficient(s), True),  # 无 SUPPORTED 高优先 → 空真
        (lambda s: gate.finalize_decision_confidence(s), 0.10),
        (lambda s: gate.reject_gate(s, gate.finalize_decision_confidence(s)), False),
    ],
)
def test_predicates_safe_on_empty_state(pred, expected):
    """全部确定性谓词对空 state 安全不抛且取值确定（无 vacuous PASS/REJECT）。"""
    assert pred({}) == expected
