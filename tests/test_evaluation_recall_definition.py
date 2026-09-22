"""Recall 定义守护：分母 = 全部真值 REJECT 案；pred HUMAN_REVIEW 入分母、不入 TP/FN。

用户已裁定的 recall 语义（标准定义）：真违规被自动拦下的比例。转人工是安全的保守行为，
但不算"拦下"，因此它必须拉低 recall，而不能像"只在自动裁决子集上算"那样把分母缩小。
"""

from __future__ import annotations

import pytest

from pra.evaluation.harness.base import EvalRecord
from pra.evaluation.metrics.business import DecisionEvaluator

_AUTO = "AUTO_DECIDABLE"
_SHOULD = "SHOULD_ABSTAIN"


def _rec(case_id: str, decision: str) -> EvalRecord:
    return EvalRecord(eval_case_id=case_id, scheme="rule", decision=decision)  # type: ignore[arg-type]


def test_recall_denominator_is_all_true_reject_cases() -> None:
    """真值 REJECT 的三个互斥去向（自动拒绝 / 转人工 / 自动放行）都进 recall 分母。"""
    records = [
        _rec("r1", "REJECT"),  # tp：自动拦下
        _rec("r2", "PASS"),  # fn：自动放行（危险侧）
        _rec("r3", "HUMAN_REVIEW"),  # 转人工：入分母、不入 tp/fn
        _rec("r4", "REJECT"),  # fp（真值 PASS）
        _rec("r5", "PASS"),  # tn（真值 PASS）
        _rec("h1", "HUMAN_REVIEW"),  # 真值 HUMAN_REVIEW：与 recall 无关
    ]
    expected = {
        "r1": {"decision": "REJECT", "abstain_label": _AUTO},
        "r2": {"decision": "REJECT", "abstain_label": _AUTO},
        "r3": {"decision": "REJECT", "abstain_label": _AUTO},
        "r4": {"decision": "PASS", "abstain_label": _AUTO},
        "r5": {"decision": "PASS", "abstain_label": _AUTO},
        "h1": {"decision": "HUMAN_REVIEW", "abstain_label": _SHOULD},
    }
    m = DecisionEvaluator.evaluate(records, expected)

    assert m.reject_truth == 3  # r1/r2/r3：真值 REJECT 全体
    assert (m.tp, m.fn, m.reject_human) == (1, 1, 1)
    # 转人工既不进 TP 也不进 FN（否则分母会退化成"只在自动裁决子集上算"）
    assert m.tp + m.fn == 2
    assert m.recall == pytest.approx(1 / 3)
    # 与"条件分母"旧口径（1/2）可区分：这是本用例的核心断言
    assert m.recall != pytest.approx(1 / 2)
    # 三分恒等：自动拒绝 + 转人工 + 自动放行 = 全部真值 REJECT
    assert m.reject_unhandled == pytest.approx(1 / 3)
    assert m.recall + m.reject_unhandled + m.fn / m.reject_truth == pytest.approx(1.0)
