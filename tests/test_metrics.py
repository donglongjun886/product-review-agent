"""边际增益审计探针（guardrails/metrics.py）单测：钉死 gate_probe 与 overlay 语义。

``decision_conf_probe`` 恒等于 ``gate.finalize_decision_confidence``；``gate_probe``
是 ``run_decision_overlay``（无提案输入）的轻量代理，顺序一致：R1 → abstention 清单
→ PASS/REJECT Gate → UNDECIDED。探针只供审计、不驱动路由。
"""

from __future__ import annotations

from pra.agent.guardrails import gate
from pra.agent.guardrails.metrics import decision_conf_probe, gate_probe
from pra.domain.models import Budget, HypothesisStatus
from helpers import (
    budget_exhausted_state,
    dc_anchor_state,
    ev,
    hp,
    make_case,
)
from pra.agent.guardrails import hard_rules


def _clean_pass_state() -> dict:
    """干净 PASS 案：全部高优先 REFUTED 有反驳证据，无政策证据。"""
    return {
        "hypotheses": [
            hp("H1", prior=0.5, status=HypothesisStatus.REFUTED, evidence_against=["e1"]),
            hp("H2", prior=0.4, status=HypothesisStatus.REFUTED, evidence_against=["e2"]),
        ],
        "evidence": [],
        "budget": Budget(),
        "failures": [],
        "tool_call_history": [],
        "degraded": False,
    }


def _risk_no_policy_state() -> dict:
    """风险案：SUPPORTED 高优先假设但证据链无 POLICY_REF（政策缺失）。"""
    return {
        "hypotheses": [
            hp("H1", prior=0.5, status=HypothesisStatus.SUPPORTED, posterior=0.91,
               evidence_for=["e1"]),
        ],
        "evidence": [
            ev("IMAGE_SIMILARITY", value="similarity=0.91", weight=0.91, ref_id="img1"),
        ],
        "budget": Budget(),
        "failures": [],
        "tool_call_history": [],
        "degraded": False,
    }


def test_decision_conf_probe_equals_gate_finalize():
    assert decision_conf_probe(dc_anchor_state()) == gate.finalize_decision_confidence(
        dc_anchor_state()
    )
    assert decision_conf_probe({}) == gate.finalize_decision_confidence({}) == 0.10


def test_gate_probe_clean_pass():
    assert gate_probe(_clean_pass_state()) == "PASS"


def test_gate_probe_risk_without_policy_human():
    assert gate_probe(_risk_no_policy_state()) == "HUMAN"


def test_gate_probe_empty_state_undecided():
    """空态（证据不足，无自动 Gate 通过）→ UNDECIDED。"""
    assert gate_probe({}) == "UNDECIDED"
    assert gate_probe({"hypotheses": [], "evidence": [], "failures": [],
                       "tool_call_history": [], "degraded": False}) == "UNDECIDED"


def test_gate_probe_reject_when_reject_gate_passes():
    assert gate_probe(dc_anchor_state()) == "REJECT"


def test_gate_probe_budget_exceeded_human():
    assert gate_probe(budget_exhausted_state()) == "HUMAN"


def test_gate_probe_degraded_human():
    st = dc_anchor_state()
    st["degraded"] = True
    assert gate_probe(st) == "HUMAN"


def test_gate_probe_r1_hard_rule_reject(monkeypatch):
    monkeypatch.setattr(hard_rules, "BLACKLISTED_BRANDS", frozenset({"某违禁品牌"}))
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
    """probe 与真实 overlay 结论一致（clean→PASS；风险无政策→HUMAN；dc anchor→REJECT）。"""
    from pra.agent.guardrails.gate import run_decision_overlay
    from pra.agent.guardrails.schemas import DecisionProposal

    pass_prop = DecisionProposal(decision="PASS", risk_level="NONE", confidence=0.9)  # type: ignore[arg-type]
    assert run_decision_overlay(_clean_pass_state(), pass_prop).decision.value == "PASS"
    assert gate_probe(_clean_pass_state()) == "PASS"

    # 风险无政策 → HUMAN：空提案也经 abstention（R3_POLICY_UNCERTAIN）转人工
    assert run_decision_overlay(_risk_no_policy_state(), None).decision.value == "HUMAN_REVIEW"
    assert gate_probe(_risk_no_policy_state()) == "HUMAN"

    reject_prop = DecisionProposal(decision="REJECT", risk_level="HIGH", confidence=0.9)  # type: ignore[arg-type]
    assert run_decision_overlay(dc_anchor_state(), reject_prop).decision.value == "REJECT"
    assert gate_probe(dc_anchor_state()) == "REJECT"
