"""诊断层指标 seam：工具选择准确率 / 多余调用率 / 风险类型命中率 / 有效取证率的分母纪律。

期望为空的案不进对应分母（docs/02 §5.2）；指标只进报告诊断区，不进头条。
"""

from __future__ import annotations

import pytest
from evaluation.grading.grader import CaseGrading
from evaluation.metrics.diagnostics import evaluate_diagnostics
from evaluation.record import EvalRecord


def _grading(
    case_id: str,
    *,
    expected_tools: list[str] | None = None,
    actual_tools: list[str] | None = None,
    truth_risk_type: list[str] | None = None,
    predicted_risk_type: list[str] | None = None,
) -> CaseGrading:
    """造一条判卷行：模拟 grade_case 的空期望语义（期望空 → 差异空、match None）。"""
    exp_tools = list(expected_tools or [])
    act_tools = list(actual_tools or [])
    exp_rt = list(truth_risk_type or [])
    pred_rt = list(predicted_risk_type or [])
    if exp_tools:
        missing = [t for t in exp_tools if t not in act_tools]
        extra = [t for t in act_tools if t not in exp_tools]
        tools_match: bool | None = not missing and not extra
    else:
        missing, extra, tools_match = [], [], None
    return CaseGrading(
        eval_case_id=case_id,
        scheme="agent",
        truth_decision="PASS",
        predicted_decision="PASS",
        decision_match=True,
        truth_risk_type=exp_rt,
        predicted_risk_type=pred_rt,
        risk_type_match=(set(pred_rt) == set(exp_rt)) if exp_rt else None,
        expected_evidence=[],
        actual_evidence=[],
        evidence_missing=[],
        evidence_extra=[],
        evidence_match=None,
        expected_tools=exp_tools,
        actual_tools=act_tools,
        tools_missing=missing,
        tools_extra=extra,
        tools_match=tools_match,
        expected_policies=[],
        actual_policies=[],
        policies_missing=[],
        policies_extra=[],
        policies_match=None,
    )


def _record(case_id: str, tool_history: list[dict] | None = None) -> EvalRecord:
    return EvalRecord(
        eval_case_id=case_id,
        scheme="agent",
        decision="PASS",
        detail={"tool_history": tool_history or []},
    )


class TestToolSelectionDenominator:
    """工具选择准确率 / 多余调用率：分母 = 标注了期望工具的案。"""

    def test_counts_hand_computed(self) -> None:
        """手算：标注期望 3 案 —— t1 恰好（covered 不冗余）/ t2 少调 / t3 多调（covered 但冗余）；
        t4 未标注不进分母。"""
        gradings = [
            _grading("t1", expected_tools=["ProductTool"], actual_tools=["ProductTool"]),
            _grading("t2", expected_tools=["ProductTool", "MerchantTool"], actual_tools=["ProductTool"]),
            _grading(
                "t3",
                expected_tools=["ProductTool"],
                actual_tools=["ProductTool", "CaseSearchTool"],
            ),
            _grading("t4", expected_tools=[], actual_tools=["ProductTool"]),
        ]
        d = evaluate_diagnostics(gradings, [])

        assert d.tool_expected_cases == 3
        assert d.tool_covered_cases == 2  # t1, t3（该调的都调了）
        assert d.tool_redundant_cases == 1  # t3
        assert d.tool_selection_accuracy == pytest.approx(2 / 3)
        assert d.redundant_tool_rate == pytest.approx(1 / 3)

    def test_unlabeled_cases_never_enter_denominator(self) -> None:
        """未标注期望工具的案即使调了工具也不算冗余（分母不含它）。"""
        gradings = [_grading("t4", expected_tools=[], actual_tools=["ProductTool"])]
        d = evaluate_diagnostics(gradings, [])
        assert d.tool_expected_cases == 0
        assert d.tool_selection_accuracy is None
        assert d.redundant_tool_rate is None


class TestRiskTypeHitRate:
    """风险类型命中率：分母 = 标注了期望风险类型的案。"""

    def test_counts_hand_computed(self) -> None:
        """手算：标注 2 案 —— r1 命中 / r2 未命中；r3 未标注不进分母。"""
        gradings = [
            _grading("r1", truth_risk_type=["POTENTIAL_IP_RISK"], predicted_risk_type=["POTENTIAL_IP_RISK"]),
            _grading("r2", truth_risk_type=["POTENTIAL_IP_RISK"], predicted_risk_type=["EVASION_PATTERN"]),
            _grading("r3", truth_risk_type=[], predicted_risk_type=["POTENTIAL_IP_RISK"]),
        ]
        d = evaluate_diagnostics(gradings, [])

        assert d.risk_type_labeled_cases == 2
        assert d.risk_type_hit_cases == 1
        assert d.risk_type_hit_rate == pytest.approx(1 / 2)

    def test_unlabeled_is_none(self) -> None:
        d = evaluate_diagnostics([_grading("r3")], [])
        assert d.risk_type_hit_rate is None


class TestEvidenceGainRate:
    """有效取证率：分母 = 成功的工具调用（来自 tool_history）。"""

    def test_counts_hand_computed(self) -> None:
        """手算：4 次 ok 调用中 2 次带来新证据；失败/跳过调用不进分母。"""
        records = [
            _record(
                "e1",
                tool_history=[
                    {"tool": "ProductTool", "status": "ok", "evidence_added": ["PRODUCT_FACT:x"]},
                    {"tool": "MerchantTool", "status": "ok", "evidence_added": []},
                    {"tool": "CaseSearchTool", "status": "error", "evidence_added": []},
                ],
            ),
            _record(
                "e2",
                tool_history=[
                    {"tool": "PolicySearchTool", "status": "ok", "evidence_added": ["POLICY_REF:y"]},
                    {"tool": "ProductTool", "status": "skipped", "evidence_added": []},
                ],
            ),
            _record("e3"),  # 无 tool_history（rule/single 臂形态）
        ]
        d = evaluate_diagnostics([], records)

        assert d.ok_tool_calls == 3
        assert d.calls_with_new_evidence == 2
        assert d.evidence_gain_rate == pytest.approx(2 / 3)

    def test_no_tool_calls_is_none(self) -> None:
        d = evaluate_diagnostics([], [_record("e3")])
        assert d.ok_tool_calls == 0
        assert d.evidence_gain_rate is None
