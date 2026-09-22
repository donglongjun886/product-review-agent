"""边际增益审计探针：轻量确定性纯函数，tools_node 每条工具调用前后各调一次。

产出 ``gate_probe``（JSON 承载，不进 DB 列）—— ``run_decision_overlay`` 的轻量代理
（无提案输入）。

**仅供审计，绝不参与任何判定分支或路由** —— 路由只认 ``budget_exceeded`` /
``is_converged`` 等确定性谓词。只 import gate / hard_rules（无环），state 缺字段时安全返回
确定性取值。
"""

from __future__ import annotations

from pra.agent.guardrails import gate, hard_rules


def gate_probe(state) -> str:
    """Gate 探测结论（overlay 的轻量代理，无提案输入）。

    与 ``run_decision_overlay`` 同序：硬规则 → "REJECT"；弃权清单任一命中 → "HUMAN"；
    pass_gate → "PASS"；reject_gate → "REJECT"；否则 "UNDECIDED"。
    ``decision_changed = gate_probe(before) != gate_probe(after)``，**不驱动路由**。

    弃权清单与 overlay 共用 ``gate.abstention_codes``（含关键测量缺口 / 不可测维度两类），
    故探针与真实终裁不会因清单漂移而分叉。
    """
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
