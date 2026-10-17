"""确定性决策 Gate / overlay（``guardrails/gate.py``）单测：锁住事实侧判定与归因码。"""

from __future__ import annotations

import pytest
from helpers import (
    all_measureable_caps,
    budget_exhausted_state,
    covered_evidence,
    ev,
    evasion_case,
    hp,
    make_case,
    risk_anchor_state,
)

from pra.agent.guardrails import gate
from pra.agent.guardrails.errors import make_failure
from pra.agent.guardrails.gate import (
    R1_HARD_RULE,
    R2_REJECT_GATE_FAIL,
    R3_BUDGET_EXHAUSTED,
    R3_DIMENSION_UNMEASURABLE,
    R3_MEASUREMENT_MISSING,
    R3_POSITIVE_INSUFFICIENT,
    R4_PASS_GATE_FAIL,
    R5_DEGRADED_OR_FAILED_STEP,
    pass_gate,
    reject_gate,
    run_decision_overlay,
)
from pra.agent.guardrails.hard_rules import hard_rule_hit
from pra.agent.guardrails.schemas import DecisionProposal
from pra.domain.measurement import (
    DIM_LISTING_REGISTRY,
    DIM_MERCHANT_PROFILE,
)
from pra.domain.models import Budget, Decision, HypothesisStatus, RiskLevel
from pra.screening.engine import triage
from pra.screening.rule_engine import terms


def _pass_ready_state(**overrides) -> dict:
    """满足 PASS Gate 条件的干净 state（required 覆盖完整、无阳性）。"""
    state = {
        "case": make_case(),
        "hypotheses": [],
        "evidence": covered_evidence(),
        "budget": Budget(),
        "failures": [],
        "tool_call_history": [],
        "degraded": False,
        "measurement_capabilities": all_measureable_caps(),
    }
    state.update(overrides)
    return state


def _proposal(
    *,
    decision: str = "PASS",
    risk_level: str = "NONE",
    risk_type: list | None = None,
    confidence: float = 0.8,
) -> DecisionProposal:
    return DecisionProposal(
        decision=decision,
        risk_level=risk_level,
        risk_type=list(risk_type or []),
        confidence=confidence,
        evidence_ids=[],
        policy=[],
        rationale="test",
    )


def test_pass_gate_clean_covered_case_true():
    assert pass_gate(_pass_ready_state()) is True


def test_pass_gate_true_even_with_empty_hypotheses():
    assert pass_gate(_pass_ready_state(hypotheses=[])) is True


def test_pass_gate_true_even_with_high_prior_normal_hypotheses():
    st = _pass_ready_state(
        hypotheses=[
            hp("H1", prior=0.35, posterior=0.9, status=HypothesisStatus.SUPPORTED,
               statement="商品正常上架，不存在品牌规避", evidence_for=["PRODUCT_FACT x"]),
            hp("H2", prior=0.3, posterior=0.8, status=HypothesisStatus.SUPPORTED),
        ]
    )
    assert pass_gate(st) is True


def test_pass_gate_false_on_any_positive_evidence():
    st = _pass_ready_state()
    st["evidence"] = covered_evidence(merchant_removals=5)
    assert pass_gate(st) is False


def test_pass_gate_false_when_required_measurement_missing():
    st = _pass_ready_state()
    st["evidence"] = [
        e for e in st["evidence"]
        if not (e.type == "MEASUREMENT" and e.ref_id.startswith(DIM_LISTING_REGISTRY))
    ]
    assert pass_gate(st) is False


def test_pass_gate_false_when_rule_positive():
    st = _pass_ready_state()
    st["case"].product.title = "NIKE 联名风格宽松卫衣"
    assert pass_gate(st) is False


def test_pass_gate_false_when_no_case():
    assert pass_gate({"evidence": [], "failures": []}) is False


def test_reject_gate_true_with_evasion_word_and_citation():
    st = risk_anchor_state()
    assert reject_gate(st) is True


def test_reject_gate_false_without_citable():
    """R-302 命中但无可引用依据 ⇒ 不授权自动拒绝（阳性不足）。"""
    st = _pass_ready_state()
    st["case"] = evasion_case()
    st["evidence"] = covered_evidence(merchant_removals=5)
    assert reject_gate(st) is False


def test_reject_gate_false_without_case():
    """无 case 时即便命中规避词 + 有可引用依据，也不得自动拒绝。"""
    st = risk_anchor_state()
    st["case"] = None
    assert reject_gate(st) is False


@pytest.mark.parametrize(
    "mutation",
    [
        {"prior": 0.99},
        {"prior": 0.0},
        {"posterior": 1.0},
        {"posterior": None},
        {"status": HypothesisStatus.SUPPORTED},
        {"status": HypothesisStatus.REFUTED},
        {"status": HypothesisStatus.UNRESOLVED},
        {"evidence_for": ["MERCHANT_HISTORY 5 removals, match=某商家"]},
        {"evidence_against": []},
    ],
)
def test_gate_is_invariant_to_llm_hypothesis_fields(mutation):
    """篡改假设的 prior/posterior/status/evidence_for 后，两道事实侧判定必须逐字不变。"""
    base = risk_anchor_state()
    before = (pass_gate(base), reject_gate(base))
    mutated = risk_anchor_state()
    for key, value in mutation.items():
        setattr(mutated["hypotheses"][1], key, value)
    after = (pass_gate(mutated), reject_gate(mutated))
    assert before == after


def test_gate_is_invariant_to_hypotheses_being_removed_entirely():
    base = risk_anchor_state()
    stripped = risk_anchor_state()
    stripped["hypotheses"] = []
    assert pass_gate(base) == pass_gate(stripped)
    assert reject_gate(base) == reject_gate(stripped)


def test_overlay_pass_accepted_on_clean_case():
    final = run_decision_overlay(_pass_ready_state(), _proposal(decision="PASS"))
    assert final.decision is Decision.PASS and final.overrides == []


def test_overlay_reject_accepted_when_gate_passes():
    final = run_decision_overlay(
        risk_anchor_state(),
        _proposal(decision="REJECT", risk_level="HIGH", risk_type=["POTENTIAL_IP_RISK"]),
    )
    assert final.decision is Decision.REJECT and final.overrides == []
    assert final.policy == ["POLICY_1.2", "POLICY_3.2", "POLICY_4.5"]
    assert final.decision_confidence == 1.0


def test_overlay_pass_gate_fail_gets_r4():
    st = _pass_ready_state()
    st["case"].product.title = "高仿 1:1 复古跑鞋"
    final = run_decision_overlay(st, _proposal(decision="PASS"))
    assert final.decision is Decision.HUMAN_REVIEW
    assert final.overrides == [R4_PASS_GATE_FAIL]


def test_overlay_positive_without_citation_gets_measurement_missing():
    """有阳性但无可引用依据 ⇒ ``policy_citation`` 必需且未覆盖 ⇒ 按测量缺口归因。"""
    st = _pass_ready_state()
    st["case"] = evasion_case()
    st["evidence"] = covered_evidence(merchant_removals=5)
    final = run_decision_overlay(
        st, _proposal(decision="REJECT", risk_level="HIGH", risk_type=["POTENTIAL_IP_RISK"])
    )
    assert final.decision is Decision.HUMAN_REVIEW
    assert final.overrides == [R3_MEASUREMENT_MISSING]


def test_overlay_reject_gate_fail_with_citation_but_no_evasion_hit():
    """有阳性且已有可引用依据、但未命中 R-302 ⇒ 仍不得自动拒绝（R2 + 阳性不足）。"""
    st = _pass_ready_state()
    st["evidence"] = [
        *covered_evidence(merchant_removals=5),
        ev("POLICY_REF", value="POLICY_3.2 v2 条款：x", weight=0.9, ref_id="POLICY_3.2_v2_c1"),
    ]
    final = run_decision_overlay(
        st, _proposal(decision="REJECT", risk_level="HIGH", risk_type=["POTENTIAL_IP_RISK"])
    )
    assert final.decision is Decision.HUMAN_REVIEW
    assert final.overrides == [R2_REJECT_GATE_FAIL, R3_POSITIVE_INSUFFICIENT]


def test_overlay_r1_hard_rule_forces_reject():
    st = _pass_ready_state()
    st["case"].product.brand = "山寨"
    final = run_decision_overlay(st, _proposal(decision="PASS"))
    assert final.decision is Decision.REJECT
    assert final.overrides == [R1_HARD_RULE]
    assert final.decision_confidence == 1.0
    assert final.risk_level is RiskLevel.HIGH


def test_overlay_r1_hard_rule_carries_policy_citation():
    """R1 强制拒绝必须自动携带规则书条款引用（policy + 对应 POLICY_REF 证据）。"""
    st = _pass_ready_state()
    st["case"].product.brand = "山寨"
    final = run_decision_overlay(st, _proposal(decision="PASS"))
    assert final.decision is Decision.REJECT
    assert "POLICY_1.1" in final.policy
    refs = [
        e for e in final.evidence
        if e.type == "POLICY_REF" and (e.extra or {}).get("policy_id") == "POLICY_1.1"
    ]
    assert len(refs) == 1
    assert refs[0].ref_id == "POLICY_1.1"
    assert "山寨" in refs[0].value


def test_overlay_r1_citation_is_idempotent():
    """证据通道已有该引用时不得重复追加。"""
    st = _pass_ready_state()
    st["case"].product.brand = "山寨"
    first = run_decision_overlay(st, _proposal(decision="PASS"))
    second = run_decision_overlay({**st, "evidence": first.evidence}, _proposal(decision="PASS"))
    refs = [
        e for e in second.evidence
        if e.type == "POLICY_REF" and (e.extra or {}).get("policy_id") == "POLICY_1.1"
    ]
    assert len(refs) == 1


def _evasion_reject_ready_state(*, merchant_removals: int = 0) -> dict:
    """R-302 命中 + 可引用先例 + 覆盖完整的 state（reject_gate 通过，无模型侧 POLICY_REF）。"""
    st = _pass_ready_state()
    st["case"] = evasion_case()
    st["evidence"] = [
        *covered_evidence(
            product_id="P_88231", merchant_id="M_5512", merchant_removals=merchant_removals
        ),
        ev("CASE_PRECEDENT", source="CaseSearchTool",
           value="case_1001 无品牌标识+外观高度模仿", weight=0.8, ref_id="case_1001"),
    ]
    return st


def _policy_refs(final, policy_id: str) -> list:
    return [
        e for e in final.evidence
        if e.type == "POLICY_REF" and (e.extra or {}).get("policy_id") == policy_id
    ]


def test_overlay_adopted_reject_carries_evasion_policy_citation():
    """模型自主 REJECT（过 reject_gate）+ R-302 命中 → 确定性补 POLICY_1.2 引用。"""
    final = run_decision_overlay(
        _evasion_reject_ready_state(),
        _proposal(decision="REJECT", risk_level="HIGH", risk_type=["POTENTIAL_IP_RISK"]),
    )
    assert final.decision is Decision.REJECT and final.overrides == []
    assert "POLICY_1.2" in final.policy
    refs = _policy_refs(final, "POLICY_1.2")
    assert len(refs) == 1
    assert refs[0].ref_id == "POLICY_1.2"
    assert "高仿" in refs[0].value
    assert "POLICY_4.5" not in final.policy


def test_overlay_adopted_reject_with_dirty_merchant_also_cites_aggravated():
    """R-302 命中 + merchant_profile 阳性 → 从重叠加 POLICY_4.5。"""
    final = run_decision_overlay(
        _evasion_reject_ready_state(merchant_removals=5),
        _proposal(decision="REJECT", risk_level="HIGH", risk_type=["POTENTIAL_IP_RISK"]),
    )
    assert final.decision is Decision.REJECT and final.overrides == []
    assert "POLICY_1.2" in final.policy
    assert "POLICY_4.5" in final.policy


def test_overlay_human_review_does_not_append_evasion_citation():
    """只在最终 REJECT 时注入：HUMAN_REVIEW 不追加规避词引用。"""
    final = run_decision_overlay(
        _evasion_reject_ready_state(merchant_removals=5),
        _proposal(decision="HUMAN_REVIEW", risk_level="MEDIUM"),
    )
    assert final.decision is Decision.HUMAN_REVIEW
    assert "POLICY_1.2" not in final.policy
    assert "POLICY_4.5" not in final.policy


def test_overlay_r1_with_evasion_hit_carries_both_citations():
    """R1 分支同时命中 R-302（blacklist_plus_evasion）→ POLICY_1.1 + POLICY_1.2。"""
    st = _pass_ready_state()
    st["case"] = make_case(brand="山寨")
    st["case"].product.title = "高仿 1:1 复古跑鞋"
    final = run_decision_overlay(st, _proposal(decision="PASS"))
    assert final.decision is Decision.REJECT and final.overrides == [R1_HARD_RULE]
    assert "POLICY_1.1" in final.policy
    assert "POLICY_1.2" in final.policy


def test_overlay_evasion_citation_is_idempotent():
    """证据通道已有规避词引用时不得重复追加。"""
    st = _evasion_reject_ready_state()
    proposal = _proposal(decision="REJECT", risk_level="HIGH", risk_type=["POTENTIAL_IP_RISK"])
    first = run_decision_overlay(st, proposal)
    second = run_decision_overlay({**st, "evidence": first.evidence}, proposal)
    assert len(_policy_refs(second, "POLICY_1.2")) == 1


def test_r1_and_r101_share_one_blacklist_source():
    """R1 硬规则与 R-101 必须消费同一份内置黑名单：同一 brand 两者同时命中。"""
    brand = "山寨"
    assert brand in terms.BLACKLISTED_BRANDS
    assert hard_rule_hit({"case": make_case(brand=brand), "evidence": []}) is not None
    assert [h.rule_id for h in triage(make_case(brand=brand)).hits] == ["R-101"]


def test_overlay_budget_exhausted_r3():
    final = run_decision_overlay(budget_exhausted_state(), None)
    assert final.decision is Decision.HUMAN_REVIEW
    assert final.overrides == [R3_BUDGET_EXHAUSTED]


def test_overlay_degraded_r5():
    final = run_decision_overlay(_pass_ready_state(degraded=True), None)
    assert final.overrides == [R5_DEGRADED_OR_FAILED_STEP]


def test_overlay_missing_measurement_gets_dedicated_code():
    """NOT_MEASURED：可补救的取证缺口 → 专用归因（与"证据不足"区分开）。"""
    st = _pass_ready_state()
    st["evidence"] = [
        e for e in st["evidence"]
        if not (e.type == "MEASUREMENT" and e.ref_id.startswith(DIM_MERCHANT_PROFILE))
    ]
    final = run_decision_overlay(st, _proposal(decision="PASS"))
    assert final.overrides == [R3_MEASUREMENT_MISSING]


def test_overlay_unmeasurable_dimension_gets_dedicated_code():
    """UNMEASURABLE：环境缺失 → 与"没测"分开归因。"""
    st = _pass_ready_state()
    caps = all_measureable_caps()
    caps[DIM_MERCHANT_PROFILE] = False
    st["measurement_capabilities"] = caps
    st["evidence"] = [
        e for e in st["evidence"]
        if not (e.type == "MEASUREMENT" and e.ref_id.startswith(DIM_MERCHANT_PROFILE))
    ]
    final = run_decision_overlay(st, _proposal(decision="PASS"))
    assert final.overrides == [R3_DIMENSION_UNMEASURABLE]


def test_overlay_warn_failure_does_not_trigger():
    st = _pass_ready_state()
    st["failures"] = [
        make_failure(step_type="TOOL_CALL", severity="warn", tool="PolicySearchTool",
                     reason="args 校验失败", seq=1)
    ]
    final = run_decision_overlay(st, _proposal(decision="PASS"))
    assert final.decision is Decision.PASS and final.overrides == []


def test_overlay_human_proposal_adopted_without_overrides():
    final = run_decision_overlay(_pass_ready_state(), _proposal(decision="HUMAN_REVIEW"))
    assert final.decision is Decision.HUMAN_REVIEW and final.overrides == []


def test_overlay_combined_budget_and_missing_order():
    """多码并存：顺序固定（预算在前），全量收集。"""
    st = budget_exhausted_state()
    st["case"] = make_case()
    st["measurement_capabilities"] = all_measureable_caps()
    final = run_decision_overlay(st, None)
    assert final.overrides == [R3_BUDGET_EXHAUSTED, R3_MEASUREMENT_MISSING]


def test_overlay_empty_state_proposal_none_r5_fallback():
    final = run_decision_overlay({"evidence": [], "failures": []}, None)
    assert final.overrides == [R5_DEGRADED_OR_FAILED_STEP]


@pytest.mark.parametrize(
    "state",
    [
        {"evidence": [], "failures": []},
        {"hypotheses": [], "evidence": [], "failures": []},
        budget_exhausted_state(),
    ],
)
def test_predicates_safe_on_empty_state(state):
    """空 state 下全部谓词安全不抛、且不 vacuous PASS / REJECT。"""
    cov = gate.coverage_of(state)
    gate.abstention_codes(state, cov)
    assert pass_gate(state) is False
    assert reject_gate(state) is False
