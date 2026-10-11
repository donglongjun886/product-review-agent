"""报告 seam：头条三指标（带分子/分母）、诊断区、过程断言失败清单都出现在报告中；
头条区不得出现 Agent 诊断指标。"""

from __future__ import annotations

import pytest
from evaluation.grading.grader import CaseGrading
from evaluation.metrics.assertions import AssertionFailure
from evaluation.metrics.diagnostics import evaluate_diagnostics
from evaluation.metrics.engineering import EngineeringEvaluator
from evaluation.metrics.selective import evaluate_selective
from helpers import SCRIPTS_DIR, load_script

SCRIPT = SCRIPTS_DIR / "run_evaluation.py"


@pytest.fixture
def script():
    return load_script(SCRIPT)


def _grading(case_id: str, truth: str, pred: str, **over) -> CaseGrading:
    base = {
        "eval_case_id": case_id,
        "scheme": "agent",
        "truth_decision": truth,
        "predicted_decision": pred,
        "decision_match": pred == truth,
        "truth_risk_type": [],
        "predicted_risk_type": [],
        "risk_type_match": None,
        "expected_evidence": [],
        "actual_evidence": [],
        "evidence_missing": [],
        "evidence_extra": [],
        "evidence_match": None,
        "expected_tools": [],
        "actual_tools": [],
        "tools_missing": [],
        "tools_extra": [],
        "tools_match": None,
        "expected_policies": [],
        "actual_policies": [],
        "policies_missing": [],
        "policies_extra": [],
        "policies_match": None,
    }
    base.update(over)
    return CaseGrading(**base)


def _render(script, gradings: list[CaseGrading], assertions: dict) -> str:
    selective = {arm: evaluate_selective(gradings) for arm in script.ARMS}
    engineering = {arm: EngineeringEvaluator.evaluate([]) for arm in script.ARMS}
    return script._render_report(
        data_path="tests-injected",
        model="test-model",
        cases=[],
        selective=selective,
        diagnostics=evaluate_diagnostics(gradings, []),
        assertions=assertions,
        engineering=engineering,
        failures={arm: [] for arm in script.ARMS},
        excluded=set(),
        metamorphic=None,
        pass_k=None,
        pass_k_runs=1,
        llm_config={"thinking": None, "reasoning_effort": None},
        out_path=None,
    )


def test_headline_shows_three_metrics_with_fractions(script) -> None:
    """手算：4 案 —— 2 自动判对 / 1 误拒 / 1 转人工 → coverage 3/4、risk 1/3、误拒 1/2。"""
    gradings = [
        _grading("r1", "PASS", "PASS"),
        _grading("r2", "PASS", "REJECT"),
        _grading("r3", "REJECT", "HUMAN_REVIEW"),
        _grading("r4", "REJECT", "REJECT"),
    ]
    report = _render(script, gradings, {arm: [] for arm in script.ARMS})

    assert "coverage" in report and "selective_risk" in report and "false_reject_rate" in report
    assert "0.750(3/4)" in report  # coverage
    assert "0.333(1/3)" in report  # selective_risk
    assert "0.500(1/2)" in report  # false_reject_rate
    assert "命中过程断言" in report  # 判错口径注记（裁决不一致 ∨ 过程断言）


def test_diagnostics_and_assertions_in_their_own_sections(script) -> None:
    gradings = [
        _grading(
            "d1",
            "REJECT",
            "REJECT",
            truth_risk_type=["POTENTIAL_IP_RISK"],
            predicted_risk_type=["POTENTIAL_IP_RISK"],
            risk_type_match=True,
            expected_tools=["MerchantTool"],
            actual_tools=["MerchantTool"],
            tools_match=True,
        )
    ]
    assertions = {
        arm: [
            AssertionFailure(
                eval_case_id="d1",
                scheme=arm,
                code="REJECT_WITHOUT_CITATION",
                detail="自动拒绝缺可引用依据",
            )
        ]
        if arm == "agent"
        else []
        for arm in script.ARMS
    }
    report = _render(script, gradings, assertions)

    headline, _, rest = report.partition("诊断区")
    assert "工具选择准确率" not in headline, "诊断指标不得出现在头条区"
    assert "有效取证率" not in headline
    assert "工具选择准确率 1.000(1/1)" in rest
    assert "风险类型命中率 1.000(1/1)" in rest
    assert "过程断言" in rest
    assert "REJECT_WITHOUT_CITATION×1" in rest
    assert "d1" in rest
