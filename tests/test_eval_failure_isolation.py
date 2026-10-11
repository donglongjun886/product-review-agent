"""评测逐案保护：单案基础设施异常只废该案，整臂仍产出可读数据。"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from evaluation.dataset.schema import (
    CaseManifestV3,
    EvalCaseV3,
    EvalConstructionV3,
    EvalExpectedV3,
    EvalTruthV3,
    MerchantFactsV3,
    RegistryFactsV3,
)
from evaluation.grading.grader import grade_records
from evaluation.metrics.assertions import load_citation_index
from evaluation.metrics.diagnostics import evaluate_diagnostics
from evaluation.metrics.engineering import EngineeringEvaluator
from evaluation.metrics.selective import evaluate_selective
from evaluation.record import EvalRecord
from helpers import REPO_ROOT, SCRIPTS_DIR, load_script, make_case

SCRIPT = SCRIPTS_DIR / "run_evaluation.py"
CORPUS_DIR = REPO_ROOT / "src" / "pra" / "rag" / "corpus"
_FAIL_CASE_ID = "EC_T2"
_FAIL_TYPE = "RuntimeError"
_FAIL_MESSAGE = "injected infra failure"


def _eval_case(eval_case_id: str, case_id: str) -> EvalCaseV3:
    """造一条 v3 案件行（干净案输入）。"""
    return EvalCaseV3(
        eval_case_id=eval_case_id,
        scene="normal",
        construction=EvalConstructionV3(pattern="clean", params={}),
        input=make_case(case_id=case_id),
    )


def _eval_truth(eval_case_id: str) -> EvalTruthV3:
    """造一条 v3 真值行（PASS / AUTO_DECIDABLE）。"""
    return EvalTruthV3(
        eval_case_id=eval_case_id,
        manifest=CaseManifestV3(
            text_hits=[],
            snapshot_brand="潮动",
            snapshot_gaps=[],
            registry_mismatch=False,
            merchant=MerchantFactsV3(
                merchant_id="M_TEST",
                removals=0,
                title_relisting_count=0,
                similar_product_count=0,
                credit_score=90,
            ),
            registry=RegistryFactsV3(
                product_id="P_TEST",
                category="女鞋/运动鞋",
                brand="潮动",
                version=3,
                status="ON_SALE",
            ),
        ),
        expected=EvalExpectedV3(
            decision="PASS",
            abstain_label="AUTO_DECIDABLE",
            risk_type=[],
            expected_tools=[],
            expected_policies=[],
        ),
    )


def _cases_and_truth() -> tuple[list[EvalCaseV3], list[EvalTruthV3]]:
    """三案数据集：中间一案将注入基础设施异常。"""
    ids = [("EC_T1", "CASE_T1"), (_FAIL_CASE_ID, "CASE_T2"), ("EC_T3", "CASE_T3")]
    return (
        [_eval_case(eid, cid) for eid, cid in ids],
        [_eval_truth(eid) for eid, _ in ids],
    )


@pytest.fixture
def script() -> Iterator[Any]:
    """按路径加载的跑分脚本模块（``scripts/`` 下的独立入口，不是包）。"""
    yield load_script(SCRIPT)


async def test_single_case_failure_keeps_arm_alive(script) -> None:
    """中间一案抛异常 → 该臂仍出数据，失败只废该案且不进指标分母。"""
    cases, truth = _cases_and_truth()

    async def run_one(case: EvalCaseV3) -> EvalRecord:
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
        "失败案不得被伪装成 HUMAN_REVIEW（会压低 coverage 并污染分母）"
    )

    gradings = grade_records(records, truth)
    metrics = evaluate_selective(gradings)
    assert (metrics.total, metrics.auto_decided) == (2, 2)
    assert metrics.coverage == 1.0
    latencies = [ms for _, ms in results]
    engineering = EngineeringEvaluator.evaluate(records, latency_ms=latencies)
    assert engineering.total_records == 2


async def test_failed_count_and_reason_reach_report_text(script) -> None:
    """报告必须显式带 ``failed=1`` 与失败原因（含异常类型短名）且注明分母不含失败案。"""
    cases, truth = _cases_and_truth()

    async def run_one(case: EvalCaseV3) -> EvalRecord:
        if case.eval_case_id == _FAIL_CASE_ID:
            raise RuntimeError(_FAIL_MESSAGE)
        return EvalRecord(
            eval_case_id=case.eval_case_id,
            scheme="rule",
            decision="PASS",
            risk_level="NONE",
        )

    results, failures = await script._run_arm(cases, run_one, concurrency=1, progress=False)

    records = [rec for rec, _ in results]
    latencies = [ms for _, ms in results]
    gradings = {arm: grade_records(records, truth) for arm in script.ARMS}
    selective = {arm: evaluate_selective(gradings[arm]) for arm in script.ARMS}
    engineering = {
        arm: EngineeringEvaluator.evaluate(records, latency_ms=latencies) for arm in script.ARMS
    }
    failures_by_arm = {arm: (failures if arm == "rule" else []) for arm in script.ARMS}
    citations = load_citation_index(CORPUS_DIR)
    assertions = {arm: [] for arm in script.ARMS}

    report = script._render_report(
        data_path="tests-injected",
        model="test-model",
        cases=cases,
        selective=selective,
        diagnostics=evaluate_diagnostics(gradings["agent"], records),
        assertions=assertions,
        engineering=engineering,
        failures=failures_by_arm,
        excluded=set(),
        metamorphic=None,
        pass_k=None,
        pass_k_runs=1,
        llm_config={"thinking": None, "reasoning_effort": None},
        out_path=None,
    )

    assert "failed=1" in report
    assert any(
        _FAIL_CASE_ID in line and _FAIL_TYPE in line and _FAIL_MESSAGE in line
        for line in report.splitlines()
    ), "失败案 id / 异常类型短名 / 异常消息必须在报告里同现"
    assert "不含失败案" in report
    assert citations.policy_refs  # 语料索引非空（防呆：断言失败不应来自空索引）
