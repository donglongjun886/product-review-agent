"""边际增益审计探针：``run_decision_overlay`` 的轻量代理（无提案输入），产出 ``gate_probe``。"""

from __future__ import annotations

from pra.agent.guardrails import gate, hard_rules


def gate_probe(state) -> str:
    """Gate 探测结论：返回 "REJECT" / "HUMAN" / "PASS" / "UNDECIDED" 之一。"""
    if hard_rules.hard_rule_hit(state) is not None:
        return "REJECT"

    cov = gate.coverage_of(state)
    if gate.abstention_codes(state, cov):
        return "HUMAN"

    if gate.pass_gate(state):
        return "PASS"
    if gate.reject_gate(state):
        return "REJECT"
    return "UNDECIDED"
