"""边际增益审计探针（guardrails/metrics.py）单测：钉死 gate_probe 与 overlay 语义。"""

from __future__ import annotations

from helpers import (
    all_measureable_caps,
    budget_exhausted_state,
    covered_evidence,
    evasion_case,
    make_case,
    risk_anchor_state,
)

from pra.agent.guardrails.metrics import gate_probe
from pra.domain.models import Budget


def _clean_pass_state() -> dict:
    """干净 PASS 案：required 覆盖完整 + 无阳性 + 无规则命中。"""
    return {
        "case": make_case(),
        "hypotheses": [],
        "evidence": covered_evidence(),
        "budget": Budget(),
        "failures": [],
        "tool_call_history": [],
        "degraded": False,
        "measurement_capabilities": all_measureable_caps(),
    }


def _risk_no_policy_state() -> dict:
    """风险案：命中 R-302 + 商家脏（阳性）但**无可引用依据**（政策/先例缺失）。"""
    return {
        "case": evasion_case(),
        "hypotheses": [],
        "evidence": covered_evidence(merchant_removals=5),
        "budget": Budget(),
        "failures": [],
        "tool_call_history": [],
        "degraded": False,
        "measurement_capabilities": all_measureable_caps(),
    }


def test_gate_probe_empty_state_undecided():
    """空态（证据不足，无自动 Gate 通过）→ UNDECIDED。"""
    assert gate_probe({}) == "UNDECIDED"
    assert gate_probe({"hypotheses": [], "evidence": [], "failures": [],
                       "tool_call_history": [], "degraded": False}) == "UNDECIDED"


def test_gate_probe_budget_exceeded_human():
    assert gate_probe(budget_exhausted_state()) == "HUMAN"


def test_gate_probe_degraded_human():
    st = risk_anchor_state()
    st["degraded"] = True
    assert gate_probe(st) == "HUMAN"


def test_gate_probe_r1_hard_rule_reject():
    st = {
        "case": make_case(brand="某违禁品牌"),
        "hypotheses": [],
        "evidence": [],
        "budget": Budget(),
        "failures": [],
        "tool_call_history": [],
    }
    assert gate_probe(st) == "REJECT"


def test_gate_probe_agrees_with_overlay_outcome():
    """probe 与真实 overlay 结论一致（clean→PASS；风险无政策→HUMAN；risk anchor→REJECT）。"""
    from pra.agent.guardrails.gate import run_decision_overlay
    from pra.agent.guardrails.schemas import DecisionProposal

    pass_prop = DecisionProposal(decision="PASS", risk_level="NONE", confidence=0.9)  # type: ignore[arg-type]
    assert run_decision_overlay(_clean_pass_state(), pass_prop).decision.value == "PASS"
    assert gate_probe(_clean_pass_state()) == "PASS"

    risk = _risk_no_policy_state()
    reject_prop = DecisionProposal(decision="REJECT", risk_level="HIGH", confidence=0.9)  # type: ignore[arg-type]
    risk_final = run_decision_overlay(risk, reject_prop)
    assert risk_final.decision.value == "HUMAN_REVIEW"
    assert risk_final.overrides == ["R2_REJECT_GATE_FAIL", "R3_POSITIVE_INSUFFICIENT"]
    assert gate_probe(risk) == "UNDECIDED"

    assert run_decision_overlay(risk_anchor_state(), reject_prop).decision.value == "REJECT"
    assert gate_probe(risk_anchor_state()) == "REJECT"
