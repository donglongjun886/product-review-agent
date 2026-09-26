"""评测逐案保护：单案基础设施异常只废该案，整臂仍产出可读数据。"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from helpers import make_case

from pra.evaluation.dataset.schema import EvalCase, EvalExpected
from pra.evaluation.metrics.business import DecisionEvaluator
from pra.evaluation.metrics.engineering import EngineeringEvaluator
from pra.evaluation.record import EvalRecord

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_evaluation.py"
_FAIL_CASE_ID = "EC_T2"
_FAIL_TYPE = "RuntimeError"
_FAIL_MESSAGE = "injected infra failure"


def _load_script():
    """按路径加载跑分脚本（它不是包）。"""
    spec = importlib.util.spec_from_file_location("run_evaluation_mod", SCRIPT)
    assert spec and spec.loader, f"无法定位脚本: {SCRIPT}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _eval_case(eval_case_id: str, case_id: str) -> EvalCase:
    """造一条真 ``EvalCase``（真值 PASS）。"""
    return EvalCase(
        eval_case_id=eval_case_id,
        scene="normal",
        input=make_case(case_id=case_id),
        expected=EvalExpected(decision="PASS"),
    )


@pytest.fixture
def script() -> Iterator[Any]:
    """按路径加载的跑分脚本模块（``scripts/`` 下的独立入口，不是包）。"""
    yield _load_script()


async def test_single_case_failure_keeps_arm_alive(script) -> None:
    """中间一案抛异常 → 该臂仍出数据，失败只废该案且不进指标分母。"""
    cases = [
        _eval_case("EC_T1", "CASE_T1"),
        _eval_case(_FAIL_CASE_ID, "CASE_T2"),
        _eval_case("EC_T3", "CASE_T3"),
    ]

    async def run_one(case: EvalCase) -> EvalRecord:
        if case.eval_case_id == _FAIL_CASE_ID:
            raise RuntimeError(_FAIL_MESSAGE)
        return EvalRecord(
            eval_case_id=case.eval_case_id,
            scheme="rule",
            decision="PASS",
            risk_level="NONE",
        )

    results, failures = await script._run_arm(cases, run_one, concurrency=1, progress=False)

    assert len(results) == 2, f"失败案不得中断整臂：{results}"
    assert [rec.eval_case_id for rec, _ in results] == ["EC_T1", "EC_T3"]
    assert len(failures) == 1

    failure = failures[0]
    assert failure.eval_case_id == _FAIL_CASE_ID
    assert failure.error_type == _FAIL_TYPE
    assert _FAIL_MESSAGE in failure.reason

    records = [rec for rec, _ in results]
    assert _FAIL_CASE_ID not in {rec.eval_case_id for rec in records}
    assert not any(rec.decision == "HUMAN_REVIEW" for rec in records), (
        "失败案不得被伪装成 HUMAN_REVIEW（会污染混淆矩阵与 human_review_rate）"
    )

    exp = script._expected_index(cases)
    metrics = DecisionEvaluator.evaluate(records, exp)
    assert (metrics.total, metrics.human_pred_total) == (2, 0)
    assert metrics.human_review_rate == 0.0
    latencies = [ms for _, ms in results]
    engineering = EngineeringEvaluator.evaluate(records, latency_ms=latencies)
    assert engineering.total_records == 2


async def test_failed_count_and_reason_reach_report_text(script) -> None:
    """报告必须显式带 ``failed=1`` 与失败原因（含异常类型短名）且注明分母不含失败案。"""
    cases = [
        _eval_case("EC_T1", "CASE_T1"),
        _eval_case(_FAIL_CASE_ID, "CASE_T2"),
        _eval_case("EC_T3", "CASE_T3"),
    ]

    async def run_one(case: EvalCase) -> EvalRecord:
        if case.eval_case_id == _FAIL_CASE_ID:
            raise RuntimeError(_FAIL_MESSAGE)
        return EvalRecord(
            eval_case_id=case.eval_case_id,
            scheme="rule",
            decision="PASS",
            risk_level="NONE",
        )

    results, failures = await script._run_arm(cases, run_one, concurrency=1, progress=False)

    exp = script._expected_index(cases)
    records = [rec for rec, _ in results]
    latencies = [ms for _, ms in results]
    metrics = {arm: DecisionEvaluator.evaluate(records, exp) for arm in script.ARMS}
    engineering = {
        arm: EngineeringEvaluator.evaluate(records, latency_ms=latencies) for arm in script.ARMS
    }
    failures_by_arm = {arm: (failures if arm == "rule" else []) for arm in script.ARMS}

    report = script._render_report(
        data_path="tests-injected",
        model="test-model",
        cases=cases,
        metrics=metrics,
        engineering=engineering,
        failures=failures_by_arm,
        llm_config={"thinking": None, "reasoning_effort": None},
        out_path=None,
    )

    assert "failed=1" in report
    assert any(
        _FAIL_CASE_ID in line and _FAIL_TYPE in line and _FAIL_MESSAGE in line
        for line in report.splitlines()
    ), "失败案 id / 异常类型短名 / 异常消息必须在报告里同现"
    assert "不含失败案" in report
