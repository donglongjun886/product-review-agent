"""Recall 定义守护：分母 = 全部真值 REJECT 案；pred HUMAN_REVIEW 入分母、不入 TP/FN。"""

from __future__ import annotations

import pytest
from evaluation.metrics.business import DecisionEvaluator
from evaluation.record import EvalRecord

_AUTO = "AUTO_DECIDABLE"
_SHOULD = "SHOULD_ABSTAIN"


def _rec(case_id: str, decision: str) -> EvalRecord:
    return EvalRecord(eval_case_id=case_id, scheme="rule", decision=decision)  # type: ignore[arg-type]


def test_recall_denominator_is_all_true_reject_cases() -> None:
    """真值 REJECT 的三个互斥去向（自动拒绝 / 转人工 / 自动放行）都进 recall 分母。"""
    records = [
        _rec("r1", "REJECT"),
        _rec("r2", "PASS"),
        _rec("r3", "HUMAN_REVIEW"),
        _rec("r4", "REJECT"),
        _rec("r5", "PASS"),
        _rec("h1", "HUMAN_REVIEW"),
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

    assert m.reject_truth == 3
    assert (m.tp, m.fn, m.reject_human) == (1, 1, 1)
    assert m.tp + m.fn == 2
    assert m.recall == pytest.approx(1 / 3)
    assert m.recall != pytest.approx(1 / 2)
    assert m.reject_unhandled == pytest.approx(1 / 3)
    assert m.recall + m.reject_unhandled + m.fn / m.reject_truth == pytest.approx(1.0)
