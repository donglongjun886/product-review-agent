"""边际增益审计探针（guardrails/metrics.py）单测：钉死 gate_probe 与 overlay 语义。

``decision_conf_probe`` 恒等于 ``gate.finalize_decision_confidence``；``gate_probe``
是 ``run_decision_overlay``（无提案输入）的轻量代理，顺序一致：R1 → abstention 清单
→ PASS/REJECT Gate → UNDECIDED。探针只供审计、不驱动路由。
"""

from __future__ import annotations

from pra.agent.guardrails import gate
from pra.agent.guardrails.metrics import decision_conf_probe, gate_probe
from pra.domain.models import Budget
from helpers import (
    all_measureable_caps,
    budget_exhausted_state,
    covered_evidence,
    dc_anchor_state,
    make_case,
)
from pra.agent.guardrails import hard_rules


def _clean_pass_state() -> dict:
    """干净 PASS 案（事实侧口径）：required 覆盖完整 + 无阳性 + 无规则命中。"""
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
    """风险案：证据链有硬阳性但**无可引用依据**（政策/先例缺失）。

    商家取"脏"（removals>=3）以避免落入"强相似 ∧ 商家干净"的证据冲突形态。
    """
    return {
        "case": make_case(),
        "hypotheses": [],
        "evidence": covered_evidence(merchant_removals=5, similarity=0.93),
        "budget": Budget(),
        "failures": [],
        "tool_call_history": [],
        "degraded": False,
        "measurement_capabilities": all_measureable_caps(),
    }


def test_decision_conf_probe_equals_gate_finalize():
    assert decision_conf_probe(dc_anchor_state()) == gate.finalize_decision_confidence(
        dc_anchor_state()
    )
    assert decision_conf_probe({}) == gate.finalize_decision_confidence({}) == 0.10


def test_gate_probe_clean_pass():
    assert gate_probe(_clean_pass_state()) == "PASS"


def test_gate_probe_risk_without_policy_is_undecided_without_proposal():
    """有硬阳性但无可引用依据：**探针无提案输入** ⇒ UNDECIDED（是否转人工取决于提案）。

    对照 ``test_gate_probe_agrees_with_overlay_outcome``：同一 state 配上 REJECT 提案时，
    overlay 走 REJECT Gate 失败分支 → HUMAN + R2/R3_POSITIVE_INSUFFICIENT。
    """
    assert gate_probe(_risk_no_policy_state()) == "UNDECIDED"


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

    # 风险无政策 + REJECT 提案 → HUMAN：REJECT Gate 缺可引用依据（R2 + 阳性不足）
    risk = _risk_no_policy_state()
    reject_prop = DecisionProposal(decision="REJECT", risk_level="HIGH", confidence=0.9)  # type: ignore[arg-type]
    risk_final = run_decision_overlay(risk, reject_prop)
    assert risk_final.decision.value == "HUMAN_REVIEW"
    assert risk_final.overrides == ["R2_REJECT_GATE_FAIL", "R3_POSITIVE_INSUFFICIENT"]
    # 无提案时探针只能给"待定"（是否转人工取决于提案走哪道 Gate）
    assert gate_probe(risk) == "UNDECIDED"

    assert run_decision_overlay(dc_anchor_state(), reject_prop).decision.value == "REJECT"
    assert gate_probe(dc_anchor_state()) == "REJECT"
