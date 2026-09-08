"""边际增益审计探针（guardrails/metrics.py）—— 轻量确定性纯函数，tools_node 消费。

tools_node 主循环（04-graph-design §4）在每条工具调用成功前后调用本模块两个探针，
产出边际增益 4 字段中的 before/after_confidence 与 decision_changed（04 §4.1 / O-10，
JSON 承载，不进 DB 列）：

- ``decision_conf_probe(state) -> float``：= ``gate.finalize_decision_confidence``
  （01 §7.5 确定性 decision_confidence 公式）—— 当前证据集 + 假设下“自动判安全把握”
  的代理值（确定性，非 LLM 输出）；
- ``gate_probe(state) -> str``：对 ``gate.run_decision_overlay`` 的**轻量代理**
  （无 LLM 提案输入）—— 按 overlay 固定顺序（R1 → abstention 清单 → PASS/REJECT
  Gate）给出当前快照若交给 overlay 处理最可能的结论：R1 命中 → "REJECT"；abstention
  任一命中 → "HUMAN"；elif pass_gate → "PASS"；elif reject_gate(dc) → "REJECT"；
  否则 "UNDECIDED"（证据不足，尚无法通过任何自动 Gate）。

**仅供边际增益审计（before/after/decision_changed），不驱动路由**（04 §4.2）：图路由
只认 budget_exceeded / is_converged 等确定性谓词，探针绝不参与任何判定分支或路由。

import 约束：只 import gate / budget / errors / hard_rules（无环），**不 import
nodes / tools_node**；state 缺 hypotheses/evidence/budget 时安全（gate 谓词与
budget_exceeded 的 None 守卫），返回确定性 False/[]/0.1/UNDECIDED 类取值。
"""

from __future__ import annotations

from pra.agent.guardrails import errors, gate, hard_rules
from pra.agent.guardrails.budget import budget_exceeded


def decision_conf_probe(state) -> float:
    """decision_confidence 代理值 = gate.finalize_decision_confidence(state)。

    确定性重算（01 §7.5 / T-4）：before/after 两次调用之间的差异度量单条工具调用对
    “自动判安全把握”的边际增益。空/缺失 evidence/hypotheses 时返回 0.10 基线。
    """
    return gate.finalize_decision_confidence(state)


def gate_probe(state) -> str:
    """Gate 探测结论（overlay 的轻量代理；无提案输入）。

    顺序与 ``gate.run_decision_overlay`` 一致（00 §7.2 / 01 §7.2）：
    1. R1 硬规则命中 → "REJECT"（强制，不可被 LLM 覆盖）；
    2. abstention 清单任一命中（预算/关键矛盾/关键 Tool 失败/政策不确定/多假设
       不可分/降级）→ "HUMAN"；
    3. pass_gate → "PASS"；reject_gate(dc) → "REJECT"；
    4. 否则 "UNDECIDED"（证据不足，尚无法通过任何自动 Gate）。

    返回字符串仅供边际增益审计（decision_changed = gate_probe(before) !=
    gate_probe(after)，04 §4），**不驱动路由**。
    """
    # R1 硬规则优先（overlay 步骤 1）
    if hard_rules.hard_rule_hit(state) is not None:
        return "REJECT"

    # abstention 清单（overlay 步骤 3）：任一命中 → HUMAN
    budget = state.get("budget")
    if budget is not None and budget_exceeded(budget) is not None:
        return "HUMAN"
    if gate.contradiction_detect(state):
        return "HUMAN"
    if errors.key_tool_failure(state, state.get("failures") or []):
        return "HUMAN"
    if gate.policy_indeterminate(state):
        return "HUMAN"
    if gate.indistinguishable_hypotheses(state):
        return "HUMAN"
    if state.get("degraded"):
        return "HUMAN"

    # PASS / REJECT Gate（overlay 步骤 5 的代理；两者都不满足 = UNDECIDED）
    if gate.pass_gate(state):
        return "PASS"
    dc = decision_conf_probe(state)
    if gate.reject_gate(state, dc):
        return "REJECT"
    return "UNDECIDED"
