"""Phase 1 评测护栏：数据集完整性、确定性重放、指标数值、三方案结构。

1. 数据集 ≥30 条、五类 scene 齐全、真值仅二值（loader 校验即线上 DTO 校验）；
2. 同数据同 runner 跑两遍 → EvalRecord 全字段序列化逐字节一致；
3. DecisionEvaluator 在手工可算小样本上数值正确；
4. smoke 跑通且三方案都产出 EvalRecord，成本无墙钟字段；
5. 旗舰案三方案对比（rule 直漏 / Agent 增量 / 干净 / 文本明示）。

全程离线：无网络、无真 LLM，被测对象全是确定性代码。
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
    payload = json.dumps(
        [r.model_dump(mode="json") for r in records],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _make_record(case_id: str, decision: str) -> EvalRecord:
    return EvalRecord(
        eval_case_id=case_id,
        scheme="rule",
        decision=decision,  # type: ignore[arg-type]
    )


# --- 1) 数据集完整性


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


# --- 2) 确定性重放


async def test_deterministic_replay() -> None:
    runner = EvaluationRunner(data_path=str(DATA_PATH))
    result1 = await runner.run(smoke=True, smoke_limit=8)
    result2 = await runner.run(smoke=True, smoke_limit=8)

    recs1, recs2 = result1.all_records, result2.all_records
    assert len(recs1) == len(recs2) == 8 * len(ALL_SCHEMES)

    # 全字段一致是 decision 序列 hash 口径的超集
    assert _records_hash(recs1) == _records_hash(recs2)
    # decision 序列 hash 同口径
    d1 = hashlib.sha256(
        json.dumps([r.decision for r in recs1]).encode()
    ).hexdigest()
    d2 = hashlib.sha256(
        json.dumps([r.decision for r in recs2]).encode()
    ).hexdigest()
    assert d1 == d2


# --- 3) DecisionEvaluator 手工可算小样本


def test_decision_metrics_hand_calculated() -> None:
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
    assert m.recall == pytest.approx(3 / 4)  # 4 条 REJECT 真值预测对 3
    assert m.fpr == pytest.approx(0.0)
    assert m.fnr == pytest.approx(1 / 4)
    assert m.human_rate == pytest.approx(2 / 8)
    assert m.automation == pytest.approx(6 / 8)
    assert m.reject_unhandled == pytest.approx(1 / 5)
    assert m.pass_unhandled == pytest.approx(1 / 3)


def test_decision_metrics_zero_denominator_is_none() -> None:
    records = [_make_record("p1", "PASS"), _make_record("p2", "HUMAN_REVIEW")]
    expected = {
        "p1": {"decision": "PASS", "scene": "x"},
        "p2": {"decision": "PASS", "scene": "x"},
    }
    m = DecisionEvaluator.evaluate(records, expected)
    assert m.precision is None and m.recall is None and m.fnr is None
    assert m.fpr == pytest.approx(0.0)
    assert m.accuracy == pytest.approx(0.5)  # (1 TN + 0)/2（HUMAN 判错口径）


# --- 4) smoke 跑通 + 三方案结构


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


# --- 5) 三方案对比旗舰案（设计意图回归护栏；数据确定性 → 断言稳定）


async def test_three_scheme_contrast_flagship() -> None:
    runner = EvaluationRunner(data_path=str(DATA_PATH))
    result = await runner.run()  # 全量 35 条
    by_case = {scheme: {r.eval_case_id: r.decision for r in recs}
               for scheme, recs in result.records.items()}
    g = lambda s, cid: by_case[s][cid]

    # EC_0401（brand 空缺 + 强相似 + 脏商家）→ 仅 Agent 能 REJECT
    assert g("agent", "EC_0401") == "REJECT"
    assert g("rule", "EC_0401") == "HUMAN_REVIEW"
    assert g("single_call_llm", "EC_0401") == "HUMAN_REVIEW"

    # EC_0402 自有品牌对抗：Rule 直漏 / Single 漏放 / Agent REJECT
    assert g("rule", "EC_0402") == "PASS"
    assert g("single_call_llm", "EC_0402") == "PASS"
    assert g("agent", "EC_0402") == "REJECT"

    # EC_0201 边界 Agent 增量案（brand 空缺但在库可查 → Agent PASS）
    assert g("agent", "EC_0201") == "PASS"
    assert g("rule", "EC_0201") == "HUMAN_REVIEW"

    # EC_0001 三方案一致 PASS；EC_0101 文本明示 → single/agent REJECT
    assert g("rule", "EC_0001") == g("single_call_llm", "EC_0001") == g("agent", "EC_0001") == "PASS"
    assert g("single_call_llm", "EC_0101") == "REJECT"
    assert g("agent", "EC_0101") == "REJECT"

    # Agent 无 FN/FP：审查员模型与数据集同口径构造，"满分"是耦合结果而非能力
    m = result.overall["agent"]
    assert m.fn == 0 and m.fp == 0 and m.total == result.total_cases
