"""确定性决策 Gate / overlay（``guardrails/gate.py``）单测：锁住事实侧判定与归因码。

本次语义重构后的核心不变量（本文件的主要目的）：

1. **终裁不读任何 LLM 生成量** —— 篡改 ``prior`` / ``posterior`` / ``status`` /
   ``evidence_for`` 后 ``pass_gate`` / ``reject_gate`` / ``dc`` 结果必须逐字不变；
2. PASS = 无阳性 ∧ 无规则阳性 ∧ required 全覆盖 ∧ 无关键失败 ∧ 无冲突；
3. REJECT = 维度匹配的**硬阳性** ∧ 可引用依据 ∧ dc>=0.7 ∧ 无冲突（弱相似单独不成立）；
4. HUMAN 归因区分：关键测量未取（可补救） / 维度不可测（环境缺失） / 阳性不足 / 证据冲突。
"""

from __future__ import annotations

import pytest
from helpers import (
    all_measureable_caps,
    budget_exhausted_state,
    covered_evidence,
    dc_anchor_state,
    ev,
    hp,
    make_case,
)

from pra.agent.guardrails import gate, hard_rules
from pra.agent.guardrails.errors import make_failure
from pra.agent.guardrails.gate import (
    R1_HARD_RULE,
    R2_REJECT_GATE_FAIL,
    R3_BUDGET_EXHAUSTED,
    R3_DIMENSION_UNMEASURABLE,
    R3_EVIDENCE_CONFLICT,
    R3_KEY_TOOL_FAILED,
    R3_MEASUREMENT_MISSING,
    R3_POSITIVE_INSUFFICIENT,
    R4_PASS_GATE_FAIL,
    R5_DEGRADED_OR_FAILED_STEP,
    contradiction_detect,
    finalize_decision_confidence,
    high_priority,
    pass_gate,
    reject_gate,
    run_decision_overlay,
)
from pra.agent.guardrails.schemas import DecisionProposal
from pra.domain.measurement import (
    DIM_IMAGE_APPEARANCE,
    DIM_LISTING_REGISTRY,
    DIM_MERCHANT_PROFILE,
)
from pra.domain.models import Budget, Decision, HypothesisStatus, RiskLevel


def _pass_ready_state(**overrides) -> dict:
    """满足新 PASS Gate 五项条件的 state（干净案 + required 覆盖完整 + 无阳性）。"""
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


# ---- dc（事实侧公式） ----


def test_dc_anchor_state_is_fact_side_high():
    """锚点：required 四维全覆盖 + 两个阳性 + 可引用 ⇒ coverage/strength/citation 三项拉满。"""
    # coverage=1.0(0.40) + strength=mean(0.91,0.85,0.6,1.0)=0.84(0.252) + citation(0.20) + 0.10
    assert finalize_decision_confidence(dc_anchor_state()) == 0.95


def test_dc_empty_state_is_base_line():
    assert finalize_decision_confidence({"evidence": [], "failures": []}) == 0.10


def test_dc_counts_only_covered_required_dimensions():
    """少一个 required 测量 ⇒ coverage 3/4，强度项只按已覆盖维度平均。"""
    st = _pass_ready_state()
    st["evidence"] = [
        e for e in st["evidence"]
        if not (e.type == "MEASUREMENT" and e.ref_id.startswith(DIM_IMAGE_APPEARANCE))
    ]
    # coverage=3/4(0.30) + strength=mean(0.6,0.85,1.0)=0.8167(0.245) + 0 + 0.10 = 0.645 → 0.65
    assert finalize_decision_confidence(st) == 0.65


def test_dc_penalizes_conflict_and_clips():
    st = _pass_ready_state()
    st["evidence"] = [
        *covered_evidence(merchant_removals=0),
        ev("IMAGE_SIMILARITY", weight=0.93,
           ref_id="https://cdn.example.com/products/P_TEST/img1.jpg"),
    ]
    assert contradiction_detect(st) is True
    assert 0.0 <= finalize_decision_confidence(st) <= 1.0


# ---- 谓词：证据冲突 / 高优先展示 ----


def test_contradiction_requires_strong_sim_and_clean_merchant():
    strong = ev("IMAGE_SIMILARITY", weight=0.91, ref_id="img")
    clean = ev("MERCHANT_HISTORY", value="x", ref_id="M_1", extra={"removals": 0, "title": 0})
    dirty = ev("MERCHANT_HISTORY", value="x", ref_id="M_2", extra={"removals": 5, "title": 3})
    assert contradiction_detect({"evidence": [strong, clean]}) is True
    assert contradiction_detect({"evidence": [strong, dirty]}) is False
    assert contradiction_detect({"evidence": [ev("IMAGE_SIMILARITY", weight=0.72), clean]}) is False


def test_high_priority_is_display_only_threshold():
    hyps = [hp("H1", prior=0.5), hp("H2", prior=0.2), hp("H3", prior=0.3)]
    assert [h.id for h in high_priority(hyps)] == ["H1", "H3"]
    assert high_priority([]) == []


# ---- PASS Gate ----


def test_pass_gate_clean_covered_case_true():
    assert pass_gate(_pass_ready_state()) is True


def test_pass_gate_true_even_with_empty_hypotheses():
    """旧链路死在 `hp==[]`；新链路里假设为空**不再阻塞** PASS。"""
    assert pass_gate(_pass_ready_state(hypotheses=[])) is True


def test_pass_gate_true_even_with_high_prior_normal_hypotheses():
    """旧链路死在"正常假设进 hp 后必须 REFUTED"；新链路完全不读假设。"""
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
    st["evidence"] = [*st["evidence"], ev("IMAGE_LOGO", weight=0.7, ref_id="img")]
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
    """空 state 防御：无 case ⇒ required 为空，若不显式拦截会 vacuous PASS。"""
    assert pass_gate({"evidence": [], "failures": []}) is False


def test_pass_gate_false_on_conflict():
    st = _pass_ready_state()
    st["evidence"] = [
        *covered_evidence(merchant_removals=0),
        ev("IMAGE_SIMILARITY", weight=0.93,
           ref_id="https://cdn.example.com/products/P_TEST/img1.jpg"),
    ]
    assert pass_gate(st) is False


# ---- REJECT Gate ----


def test_reject_gate_true_with_hard_positive_and_citation():
    st = dc_anchor_state()
    assert reject_gate(st, finalize_decision_confidence(st)) is True


def test_reject_gate_false_on_weak_similarity_only():
    """弱相似 0.70~0.85 **不是**硬阳性 ⇒ 不得授权自动拒绝（真实跑的 2 例误杀即此）。"""
    st = _pass_ready_state()
    st["evidence"] = [
        *covered_evidence(similarity=0.73),
        ev("POLICY_REF", value="POLICY_3.2 v2 条款：x", weight=0.9, ref_id="POLICY_3.2_v2_c1",
           extra={"policy_id": "POLICY_3.2"}),
    ]
    assert gate.weak_similarity(st["evidence"]) is True
    assert not gate.coverage_of(st).positive
    assert reject_gate(st, finalize_decision_confidence(st)) is False


def test_reject_gate_false_without_citable():
    st = _pass_ready_state()
    st["evidence"] = [
        e for e in st["evidence"] if e.type not in {"POLICY_REF", "CASE_PRECEDENT"}
    ]
    st["evidence"] = [*st["evidence"], ev("IMAGE_SIMILARITY", weight=0.93, ref_id="img")]
    assert reject_gate(st, 0.99) is False


def test_reject_gate_false_below_confidence_threshold():
    assert reject_gate(dc_anchor_state(), 0.69) is False


def test_reject_gate_false_on_conflict():
    st = _pass_ready_state()
    st["evidence"] = [
        *covered_evidence(merchant_removals=0, similarity=0.93),
        ev("POLICY_REF", value="p", weight=0.9, ref_id="c1", extra={"policy_id": "POLICY_3.2"}),
    ]
    assert reject_gate(st, 0.99) is False


# ---- 核心不变量：终裁不读 LLM 生成量 ----


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
        {"evidence_for": ["IMAGE_SIMILARITY similarity=0.99, match=某品牌"]},
        {"evidence_against": []},
    ],
)
def test_gate_is_invariant_to_llm_hypothesis_fields(mutation):
    """篡改假设的 prior/posterior/status/evidence_for 后，三道事实侧判定必须逐字不变。"""
    base = dc_anchor_state()
    before = (
        pass_gate(base),
        reject_gate(base, finalize_decision_confidence(base)),
        finalize_decision_confidence(base),
    )
    mutated = dc_anchor_state()
    for key, value in mutation.items():
        setattr(mutated["hypotheses"][1], key, value)
    after = (
        pass_gate(mutated),
        reject_gate(mutated, finalize_decision_confidence(mutated)),
        finalize_decision_confidence(mutated),
    )
    assert before == after


def test_gate_is_invariant_to_hypotheses_being_removed_entirely():
    base = dc_anchor_state()
    stripped = dc_anchor_state()
    stripped["hypotheses"] = []
    assert pass_gate(base) == pass_gate(stripped)
    assert reject_gate(base, 0.9) == reject_gate(stripped, 0.9)
    assert finalize_decision_confidence(base) == finalize_decision_confidence(stripped)


# ---- overlay：归因码 ----


def test_overlay_pass_accepted_on_clean_case():
    final = run_decision_overlay(_pass_ready_state(), _proposal(decision="PASS"))
    assert final.decision is Decision.PASS and final.overrides == []


def test_overlay_reject_accepted_when_gate_passes():
    final = run_decision_overlay(
        dc_anchor_state(),
        _proposal(decision="REJECT", risk_level="HIGH", risk_type=["POTENTIAL_IP_RISK"]),
    )
    assert final.decision is Decision.REJECT and final.overrides == []
    assert final.policy == ["POLICY_3.2"]  # 只从 POLICY_REF 证据读，不采信提案
    assert final.decision_confidence == 0.95


def test_overlay_pass_gate_fail_gets_r4():
    st = _pass_ready_state()
    st["case"].product.title = "高仿 1:1 复古跑鞋"  # 规则侧阳性 ⇒ PASS 不得放行
    final = run_decision_overlay(st, _proposal(decision="PASS"))
    assert final.decision is Decision.HUMAN_REVIEW
    assert final.overrides == [R4_PASS_GATE_FAIL]


def test_overlay_reject_gate_fail_on_weak_sim_gets_r2_and_positive_insufficient():
    st = _pass_ready_state()
    st["evidence"] = [
        *covered_evidence(similarity=0.72),
        ev("POLICY_REF", value="p", weight=0.9, ref_id="c1", extra={"policy_id": "POLICY_3.2"}),
    ]
    final = run_decision_overlay(
        st, _proposal(decision="REJECT", risk_level="HIGH", risk_type=["POTENTIAL_IP_RISK"])
    )
    assert final.decision is Decision.HUMAN_REVIEW
    assert final.overrides == [R2_REJECT_GATE_FAIL, R3_POSITIVE_INSUFFICIENT]


def test_overlay_r1_hard_rule_forces_reject(monkeypatch):
    monkeypatch.setattr(hard_rules, "BLACKLISTED_BRANDS", frozenset({"山寨"}))
    st = _pass_ready_state()
    st["case"].product.brand = "山寨"
    final = run_decision_overlay(st, _proposal(decision="PASS"))
    assert final.decision is Decision.REJECT
    assert final.overrides == [R1_HARD_RULE]
    assert final.decision_confidence == 1.0
    assert final.risk_level is RiskLevel.HIGH


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
    caps[DIM_IMAGE_APPEARANCE] = False
    st["measurement_capabilities"] = caps
    st["evidence"] = [
        e for e in st["evidence"]
        if not (e.type == "MEASUREMENT" and e.ref_id.startswith(DIM_IMAGE_APPEARANCE))
    ]
    final = run_decision_overlay(st, _proposal(decision="PASS"))
    assert final.overrides == [R3_DIMENSION_UNMEASURABLE]


def test_overlay_evidence_conflict_code():
    st = _pass_ready_state()
    st["evidence"] = [
        *covered_evidence(merchant_removals=0),
        ev("IMAGE_SIMILARITY", weight=0.93,
           ref_id="https://cdn.example.com/products/P_TEST/img1.jpg"),
    ]
    final = run_decision_overlay(st, _proposal(decision="PASS"))
    assert final.overrides == [R3_EVIDENCE_CONFLICT]


def test_overlay_key_tool_failed_r3():
    st = _pass_ready_state()
    st["failures"] = [
        make_failure(step_type="TOOL_CALL", severity="critical", tool="ProductTool",
                     reason="x", seq=1)
    ]
    st["tool_call_history"] = [{"seq": 1, "tool": "ProductTool", "status": "error"}]
    final = run_decision_overlay(st, _proposal(decision="PASS"))
    assert final.overrides == [R3_KEY_TOOL_FAILED]


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
    assert reject_gate(state, 1.0) is False
