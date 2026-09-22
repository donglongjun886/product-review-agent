"""真值索引：EvalCase 列表 → 指标层共享的 expected 字典。"""

from __future__ import annotations

from pra.evaluation.dataset.schema import EvalCase

__all__ = ["expected_index"]


def expected_index(cases: list[EvalCase]) -> dict[str, dict]:
    """由数据集构造 expected 索引：真值字段的**唯一**读取入口，指标层共享。

    键：``decision`` / ``scene`` / ``abstain_label``（业务指标）+
    ``expected_tools`` / ``evidence`` / ``risk_type`` / ``risk_level``（Agent 级指标：
    工具选择 / 推理正确性）。``abstain_label`` 经 ``getattr`` 兼容读取（schema 未升级时
    → None，等价全 AUTO_DECIDABLE）；缺字段取空列表 / None，指标层按"空真值不进分母"处理。

    :param cases: 数据集（EvalCase 列表）。
    :return: ``{eval_case_id: {decision, scene, abstain_label, expected_tools, evidence,
        risk_type, risk_level}}``。
    """
    return {
        c.eval_case_id: {
            "decision": c.expected.decision,
            "scene": c.scene,
            "abstain_label": getattr(c.expected, "abstain_label", None),
            "expected_tools": list(getattr(c.expected, "expected_tools", None) or []),
            "evidence": list(getattr(c.expected, "evidence", None) or []),
            "risk_type": list(getattr(c.expected, "risk_type", None) or []),
            "risk_level": getattr(c.expected, "risk_level", None),
        }
        for c in cases
    }
