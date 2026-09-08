"""Evaluation Phase 1 测试 —— 确定性重放 + 指标正确性 + 三方案结构。

覆盖（对齐任务书"确定性重放断言"）：
1. ``test_dataset_integrity``：eval_data/v1 数据集可加载、≥30 条、五类 scene 均有、
   真值全为二值（PASS/REJECT）—— loader/schema 校验即线上 DTO 校验；
2. ``test_deterministic_replay``：同一数据 + 同一 runner 跑两遍 → EvalRecord 序列
   逐字节一致（全字段 model_dump 序列化比对；含 decision 序列 hash，任务书口径）；
3. ``test_decision_metrics_hand_calculated``：DecisionEvaluator 在手工可算小样本上
   数值正确（Recall=0.75 等）；
4. ``test_smoke_runs_all_schemes``：smoke（≤10 条）跑通、三方案都产出 EvalRecord、
   decision ∈ {PASS, REJECT, HUMAN_REVIEW}、成本确定性为 0/无墙钟字段；
5. ``test_three_scheme_contrast_flagship``：几条旗舰案的三方案对比（框架设计意图的
   回归护栏：rule 直漏案 / Agent 增量案 / 干净案 / 文本明示案）。

全程离线：无网络、无真 LLM；被测对象全部为确定性代码（mock/桩）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pra.evaluation.dataset.loader import load_dataset
from pra.evaluation.harness.base import EvalRecord
from pra.evaluation.metrics.business import DecisionEvaluator
from pra.evaluation.runner import ALL_SCHEMES, EvaluationRunner

DATA_PATH = Path(__file__).resolve().parents[1] / "eval_data" / "v1" / "cases_v1.jsonl"

_DECISION_SET = {"PASS", "REJECT", "HUMAN_REVIEW"}


def _records_hash(records: list[EvalRecord]) -> str:
    """全字段 EvalRecord 的确定性序列化指纹（逐字节重放比对口径）。"""
    payload = json.dumps(
        [r.model_dump(mode="json") for r in records],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _make_record(case_id: str, decision: str) -> EvalRecord:
    """构造最小 EvalRecord（未指定字段走默认，确定性）。"""
    return EvalRecord(
        eval_case_id=case_id,
        scheme="rule",
        decision=decision,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# 1) 数据集完整性
# ---------------------------------------------------------------------------


def test_dataset_integrity() -> None:
    cases = load_dataset(DATA_PATH)
    assert len(cases) >= 30, "Phase 1 数据集至少 30 条"
    scenes = {c.scene for c in cases}
    assert scenes == {
        "normal",
        "violation",
        "boundary",
        "multi-signal",
        "evasion",
    }, f"五类 scene 都应覆盖: {sorted(scenes)}"
    truths = {c.expected.decision for c in cases}
    assert truths <= {"PASS", "REJECT"}, f"Phase 1 真值仅二值: {truths}"
    # input 是 ProductReviewCase（loader 校验即线上 DTO 校验）；抽查必填字段
    sample = cases[0]
    assert sample.input.product.product_id and sample.input.merchant_id


# ---------------------------------------------------------------------------
# 2) 确定性重放
# ---------------------------------------------------------------------------


async def test_deterministic_replay() -> None:
    runner = EvaluationRunner(data_path=str(DATA_PATH))
    result1 = await runner.run(smoke=True, smoke_limit=8)
    result2 = await runner.run(smoke=True, smoke_limit=8)

    recs1, recs2 = result1.all_records, result2.all_records
    assert len(recs1) == len(recs2) == 8 * len(ALL_SCHEMES)

    # EvalRecord 全字段逐字节一致（确定性重放，任务书"decision 序列 hash"的超集）
    assert _records_hash(recs1) == _records_hash(recs2)
    # 显式保留任务书口径：decision 序列 hash 一致
    d1 = hashlib.sha256(
        json.dumps([r.decision for r in recs1]).encode()
    ).hexdigest()
    d2 = hashlib.sha256(
        json.dumps([r.decision for r in recs2]).encode()
    ).hexdigest()
    assert d1 == d2


# ---------------------------------------------------------------------------
# 3) DecisionEvaluator 手工可算小样本
# ---------------------------------------------------------------------------


def test_decision_metrics_hand_calculated() -> None:
    # 手工构造：truth REJECT×4（pred REJECT×3 + pred PASS×1 → FN）、
    # truth PASS×2（pred PASS×2 → TN）、truth PASS×1→HUMAN、truth REJECT×1→HUMAN
    records = [
        _make_record("r1", "REJECT"),
        _make_record("r2", "REJECT"),
        _make_record("r3", "REJECT"),
        _make_record("r4", "PASS"),  # FN
        _make_record("p1", "PASS"),
        _make_record("p2", "PASS"),
        _make_record("p3", "HUMAN_REVIEW"),  # pass_unhandled
        _make_record("r5", "HUMAN_REVIEW"),  # reject_unhandled
    ]
    expected = {
        "r1": {"decision": "REJECT", "scene": "x"},
        "r2": {"decision": "REJECT", "scene": "x"},
        "r3": {"decision": "REJECT", "scene": "x"},
        "r4": {"decision": "REJECT", "scene": "x"},
        "p1": {"decision": "PASS", "scene": "x"},
        "p2": {"decision": "PASS", "scene": "x"},
        "p3": {"decision": "PASS", "scene": "x"},
        "r5": {"decision": "REJECT", "scene": "x"},
    }
    m = DecisionEvaluator.evaluate(records, expected)
    assert m.total == 8
    assert (m.tp, m.fp, m.tn, m.fn) == (3, 0, 2, 1)
    assert m.human_pred == 2 and m.human_reject == 1 and m.human_pass == 1
    assert m.accuracy == pytest.approx(5 / 8)
    assert m.precision == pytest.approx(1.0)
    assert m.recall == pytest.approx(3 / 4)  # 任务书示例：4 REJECT 中预测对 3 → 0.75
    assert m.fpr == pytest.approx(0.0)
    assert m.fnr == pytest.approx(1 / 4)
    assert m.human_rate == pytest.approx(2 / 8)
    assert m.automation == pytest.approx(6 / 8)
    # REJECT 真值共 5 条（r1..r5），PASS 真值共 3 条（p1..p3）
    assert m.reject_unhandled == pytest.approx(1 / 5)
    assert m.pass_unhandled == pytest.approx(1 / 3)


def test_decision_metrics_zero_denominator_is_none() -> None:
    # 无任何自动 REJECT 预测 → precision/recall/fnr 未定义（None），不硬造 0
    records = [_make_record("p1", "PASS"), _make_record("p2", "HUMAN_REVIEW")]
    expected = {
        "p1": {"decision": "PASS", "scene": "x"},
        "p2": {"decision": "PASS", "scene": "x"},
    }
    m = DecisionEvaluator.evaluate(records, expected)
    assert m.precision is None and m.recall is None and m.fnr is None
    assert m.fpr == pytest.approx(0.0)
    assert m.accuracy == pytest.approx(0.5)  # (1 TN + 0)/2（HUMAN 判错口径）


# ---------------------------------------------------------------------------
# 4) smoke 跑通 + 三方案结构
# ---------------------------------------------------------------------------


async def test_smoke_runs_all_schemes() -> None:
    runner = EvaluationRunner(data_path=str(DATA_PATH))
    result = await runner.run(smoke=True, smoke_limit=8)
    assert result.smoke and result.total_cases <= 10
    assert set(result.records.keys()) == set(ALL_SCHEMES)
    assert set(result.overall.keys()) == set(ALL_SCHEMES)
    for scheme in ALL_SCHEMES:
        recs = result.records[scheme]
        assert len(recs) == result.total_cases
        for rec in recs:
            assert rec.scheme == scheme
            assert rec.decision in _DECISION_SET
            assert set(rec.cost.keys()) <= {"llm_calls", "tool_calls", "tokens"}
            assert "latency_ms" not in rec.cost, "确定性重放要求 EvalRecord 不含墙钟字段"


# ---------------------------------------------------------------------------
# 5) 三方案对比旗舰案（设计意图回归护栏；数据确定性 → 断言稳定）
# ---------------------------------------------------------------------------


async def test_three_scheme_contrast_flagship() -> None:
    runner = EvaluationRunner(data_path=str(DATA_PATH))
    result = await runner.run()  # 全量 35 条
    by_case = {scheme: {r.eval_case_id: r.decision for r in recs}
               for scheme, recs in result.records.items()}
    g = lambda s, cid: by_case[s][cid]

    # EC_0401 核心对抗（brand 空缺 + 强相似 + 脏商家）→ 仅 Agent 能 REJECT
    assert g("agent", "EC_0401") == "REJECT"
    assert g("rule", "EC_0401") == "HUMAN_REVIEW"
    assert g("single_call_llm", "EC_0401") == "HUMAN_REVIEW"

    # EC_0402 自有品牌对抗（Rule 直漏 / Single 漏放 / Agent REJECT）
    assert g("rule", "EC_0402") == "PASS"
    assert g("single_call_llm", "EC_0402") == "PASS"
    assert g("agent", "EC_0402") == "REJECT"

    # EC_0201 边界 Agent 增量案（brand 空缺但在库可查 → Agent PASS）
    assert g("agent", "EC_0201") == "PASS"
    assert g("rule", "EC_0201") == "HUMAN_REVIEW"

    # EC_0001 干净自有品牌 → 三方案一致 PASS；EC_0101 文本明示 → single/agent REJECT
    assert g("rule", "EC_0001") == g("single_call_llm", "EC_0001") == g("agent", "EC_0001") == "PASS"
    assert g("single_call_llm", "EC_0101") == "REJECT"
    assert g("agent", "EC_0101") == "REJECT"

    # Agent 无 FN 也无 FP（Phase 1 确定性审查员模型与数据集同口径构造 —— 详见数据集
    # annotation.notes；该"满分"是设计耦合结果，非真实模型结论，报告已注明）
    m = result.overall["agent"]
    assert m.fn == 0 and m.fp == 0 and m.total == result.total_cases
