"""头条 selective prediction 指标 seam：coverage / selective risk / 错误拒绝率的分母纪律。

期望值全部手算字面量（场景见 test_headline_three_metrics 注释），不得重算实现逻辑。
"""

from __future__ import annotations

import pytest
from evaluation.grading.grader import CaseGrading
from evaluation.metrics.selective import evaluate_selective


def _grading(case_id: str, truth: str, pred: str) -> CaseGrading:
    """造一条只带头条所需字段的判卷行（其余比对格置 None/空，不进头条）。"""
    return CaseGrading(
        eval_case_id=case_id,
        scheme="agent",
        truth_decision=truth,
        truth_abstain_label="SHOULD_ABSTAIN" if truth == "HUMAN_REVIEW" else "AUTO_DECIDABLE",
        predicted_decision=pred,
        decision_match=pred == truth,
        truth_risk_type=[],
        predicted_risk_type=[],
        risk_type_match=None,
        expected_tools=[],
        actual_tools=[],
        tools_missing=[],
        tools_extra=[],
        tools_match=None,
        expected_policies=[],
        actual_policies=[],
        policies_missing=[],
        policies_extra=[],
        policies_match=None,
    )


def _sample() -> list[CaseGrading]:
    """8 案样本：c1 自动判对(PASS) / c2 误拒 / c3 真值PASS转人工 / c4 自动判对(REJECT) /
    c5 漏拒(真值REJECT判PASS) / c6 真值REJECT转人工 / c7 应人工被自动PASS / c8 应人工且转人工。
    """
    return [
        _grading("c1", "PASS", "PASS"),
        _grading("c2", "PASS", "REJECT"),
        _grading("c3", "PASS", "HUMAN_REVIEW"),
        _grading("c4", "REJECT", "REJECT"),
        _grading("c5", "REJECT", "PASS"),
        _grading("c6", "REJECT", "HUMAN_REVIEW"),
        _grading("c7", "HUMAN_REVIEW", "PASS"),
        _grading("c8", "HUMAN_REVIEW", "HUMAN_REVIEW"),
    ]


def test_headline_three_metrics() -> None:
    """手算：total=8；auto=5(c1,c2,c4,c5,c7)；错 3(c2,c5,c7)；真值PASS=3；误拒 1(c2)。"""
    m = evaluate_selective(_sample())

    assert m.total == 8
    assert m.auto_decided == 5
    assert m.auto_wrong == 3
    assert m.truth_pass_total == 3
    assert m.false_rejects == 1
    assert m.coverage == pytest.approx(5 / 8)
    assert m.selective_risk == pytest.approx(3 / 5)
    assert m.false_reject_rate == pytest.approx(1 / 3)


def test_human_review_is_restraint_not_error() -> None:
    """转人工不进 selective risk 分母（只压低 coverage）：c6/c8 不在 auto_decided。"""
    m = evaluate_selective(_sample())
    # 若转人工被错算进 selective risk 分母，值会变成 3/8 而非 3/5
    assert m.selective_risk == pytest.approx(3 / 5)
    assert m.selective_risk != pytest.approx(3 / 8)


def test_exclusion_set_removed_from_all_denominators() -> None:
    """疑问篮子剔除：c2（误拒）与 c8（转人工）被排除后，三指标分母同步收缩。"""
    m = evaluate_selective(_sample(), exclude={"c2", "c8"})

    assert m.total == 6
    assert m.auto_decided == 4
    assert m.auto_wrong == 2
    assert m.truth_pass_total == 2
    assert m.false_rejects == 0
    assert m.coverage == pytest.approx(4 / 6)
    assert m.selective_risk == pytest.approx(2 / 4)
    assert m.false_reject_rate == 0.0


def test_empty_denominators_are_none_not_zero() -> None:
    """分母为 0 → None（不硬造 0/∞）；coverage 分母恒非空时照常出数。"""
    all_human = [_grading("h1", "PASS", "HUMAN_REVIEW")]
    m = evaluate_selective(all_human)
    assert m.coverage == 0.0
    assert m.selective_risk is None  # 无被自动决策的案

    no_truth_pass = [_grading("r1", "REJECT", "REJECT")]
    m2 = evaluate_selective(no_truth_pass)
    assert m2.coverage == 1.0
    assert m2.selective_risk == 0.0
    assert m2.false_reject_rate is None  # 无真值 PASS 案

    m3 = evaluate_selective([], exclude=set())
    assert m3.total == 0
    assert m3.coverage is None
