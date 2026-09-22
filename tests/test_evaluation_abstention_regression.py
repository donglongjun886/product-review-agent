"""评测抽象层测试：Abstention / Regression 两组。

- ``test_abstention_*``：五指标在手工可算小样本上数值正确（SHOULD_ABSTAIN 被自动
  只表现为 recall 缺口、不进 wrong_auto；分母 0 → None；老数据无 abstain_label 兼容）；
- ``test_regression_*``：篡改记录后比对失败，真实跑评测集两次 digest 一致。

全程离线确定性；测试数据取 ``eval_data/v2`` 的子集（只读），基线快照一律写 tmp_path。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pra.evaluation.dataset.loader import load_dataset
from pra.evaluation.harness.base import EvalRecord
from pra.evaluation.metrics.abstention import AbstentionEvaluator
from pra.evaluation.regression import (
    canonical_digest,
    compare_snapshots,
    compute_current_snapshot,
    snapshot_from_records,
    write_baseline,
)

DATA_PATH = Path(__file__).resolve().parents[1] / "eval_data" / "v2" / "cases_v2.jsonl"


def _subset_path(tmp_path: Path, n: int = 4) -> str:
    """把 v2 前 ``n`` 条写成临时 JSONL（回归机制用例只需一个小而确定的集）。"""
    lines = DATA_PATH.read_text(encoding="utf-8").splitlines()[:n]
    path = tmp_path / "cases_subset.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def _make_record(case_id: str, decision: str, scheme: str = "rule") -> EvalRecord:
    return EvalRecord(eval_case_id=case_id, scheme=scheme, decision=decision)  # type: ignore[arg-type]


def test_abstention_five_metrics_hand_calculated() -> None:
    # a1 正确自动 / a2 自动但判错(wrong_auto) / a3 过度保守转人工(abstention_rate)
    # s1 正确转人工(abstention_recall) / s2 SHOULD_ABSTAIN 被自动(危险误自动 → recall 缺口)
    records = [
        _make_record("a1", "REJECT"),
        _make_record("a2", "PASS"),
        _make_record("a3", "HUMAN_REVIEW"),
        _make_record("s1", "HUMAN_REVIEW"),
        _make_record("s2", "REJECT"),
    ]
    expected = {
        "a1": {"decision": "REJECT", "scene": "x", "abstain_label": "AUTO_DECIDABLE"},
        "a2": {"decision": "REJECT", "scene": "x", "abstain_label": "AUTO_DECIDABLE"},
        "a3": {"decision": "PASS", "scene": "x", "abstain_label": "AUTO_DECIDABLE"},
        "s1": {"decision": "HUMAN_REVIEW", "scene": "x", "abstain_label": "SHOULD_ABSTAIN"},
        "s2": {"decision": "HUMAN_REVIEW", "scene": "x", "abstain_label": "SHOULD_ABSTAIN"},
    }
    m = AbstentionEvaluator.evaluate(records, expected)
    assert m.total == 5
    assert m.auto_decidable_total == 3 and m.should_abstain_total == 2
    assert m.human_pred_total == 2 and m.auto_pred_total == 3
    # 互斥口径：s2（SHOULD_ABSTAIN 被自动）不进 wrong_auto —— 其危险由 recall 缺口承接
    assert m.should_abstain_auto == 1
    assert m.auto_decidable_auto_wrong == 1  # 只有 a2
    assert m.human_review_rate == pytest.approx(2 / 5)
    assert m.automation_coverage == pytest.approx(3 / 5)
    assert m.abstention_rate == pytest.approx(1 / 3)  # a3 / (a1..a3)
    assert m.abstention_recall == pytest.approx(1 / 2)  # s1 / (s1,s2)
    assert m.wrong_auto_decision_rate == pytest.approx(1 / 2)  # a2 / (a1,a2) 自动判


def test_abstention_sho_auto_not_counted_in_wrong_auto() -> None:
    """SHOULD_ABSTAIN 被自动终裁只表现为 abstention_recall < 1，wrong_auto 分母不含它。"""
    records = [_make_record("s1", "PASS"), _make_record("s2", "PASS")]
    expected = {
        "s1": {"decision": "HUMAN_REVIEW", "scene": "x", "abstain_label": "SHOULD_ABSTAIN"},
        "s2": {"decision": "HUMAN_REVIEW", "scene": "x", "abstain_label": "SHOULD_ABSTAIN"},
    }
    m = AbstentionEvaluator.evaluate(records, expected)
    assert m.should_abstain_auto == 2
    assert m.abstention_recall == pytest.approx(0.0)
    # 无 AUTO_DECIDABLE 案 → 两指标 None（分母 0），不是 0
    assert m.wrong_auto_decision_rate is None
    assert m.abstention_rate is None


def test_abstention_zero_denominator_none_and_phase1_compat() -> None:
    """Phase 1 老数据（无 abstain_label、真值仅 PASS/REJECT）→ 全按 AUTO_DECIDABLE；SHOULD 子集为空 → abstention_recall None（报告 '-'）；wrong_auto 退化为自动终裁错误率。"""
    records = [
        _make_record("p1", "PASS"),
        _make_record("r1", "REJECT"),
        _make_record("p2", "HUMAN_REVIEW"),  # 过度保守
    ]
    expected = {
        "p1": {"decision": "PASS", "scene": "x"},  # 无 abstain_label（Phase 1 形态）
        "r1": {"decision": "REJECT", "scene": "x"},
        "p2": {"decision": "PASS", "scene": "x"},
    }
    m = AbstentionEvaluator.evaluate(records, expected)
    assert m.auto_decidable_total == 3 and m.should_abstain_total == 0
    assert m.abstention_recall is None  # 分母 0
    assert m.human_review_rate == pytest.approx(1 / 3)
    assert m.automation_coverage == pytest.approx(2 / 3)
    assert m.abstention_rate == pytest.approx(1 / 3)
    assert m.wrong_auto_decision_rate == pytest.approx(0.0)  # 自动判两案都对


def test_abstention_label_missing_but_human_truth_inferred() -> None:
    """A 面 schema 未升级（无 abstain_label）但真值已是 HUMAN_REVIEW → 推断 SHOULD_ABSTAIN。"""
    records = [_make_record("s1", "HUMAN_REVIEW"), _make_record("s2", "PASS")]
    expected = {
        "s1": {"decision": "HUMAN_REVIEW", "scene": "x"},
        "s2": {"decision": "HUMAN_REVIEW", "scene": "x"},
    }
    m = AbstentionEvaluator.evaluate(records, expected)
    assert m.should_abstain_total == 2
    assert m.abstention_recall == pytest.approx(1 / 2)
    assert m.wrong_auto_decision_rate is None  # AUTO 案数 0


def test_regression_tampered_record_fails() -> None:
    """篡改一条记录后快照 digest/序列不匹配 → FAIL（regression 能抓到漂移）。"""
    cases = [c for c in load_dataset(DATA_PATH)][:4]
    records = {
        "rule": [_make_record(c.eval_case_id, "PASS") for c in cases],
        "single_call_llm": [_make_record(c.eval_case_id, "PASS") for c in cases],
        "agent": [_make_record(c.eval_case_id, "PASS") for c in cases],
    }
    baseline = snapshot_from_records(cases, records)
    tampered = {
        "format_version": 1,
        "data_hint": "x",
        "total_cases": len(cases),
        "scheme_order": ["rule", "single_call_llm", "agent"],
        "per_case_ids": [c.eval_case_id for c in cases],
        "decisions": {
            "rule": ["PASS"] * 4,
            "single_call_llm": ["PASS"] * 4,
            "agent": ["PASS", "HUMAN_REVIEW", "PASS", "PASS"],
        },
    }
    tampered["digest"] = canonical_digest(tampered["per_case_ids"], tampered["decisions"])
    report = compare_snapshots(tampered, baseline)
    assert report.ok is False and report.status == "FAIL"
    assert "agent" in report.mismatches
    assert report.mismatches["agent"][0][2] == "PASS"  # 基线值
    assert report.mismatches["agent"][0][3] == "HUMAN_REVIEW"  # 当前值


def test_regression_subset_compare_and_missing_scheme(tmp_path: Path) -> None:
    """基线只录两方案、当前只跑子集 → 同决策 PASS（子集比对不误报）；当前含基线未录的方案 → FAIL（提示重录）。"""
    cases = [c for c in load_dataset(_subset_path(tmp_path))][:3]
    base_records = {
        "rule": [_make_record(c.eval_case_id, "PASS") for c in cases],
        "single_call_llm": [_make_record(c.eval_case_id, "PASS") for c in cases],
    }
    baseline = snapshot_from_records(cases, base_records, scheme_order=("rule", "single_call_llm"))
    subset_records = {
        "single_call_llm": [_make_record(c.eval_case_id, "PASS") for c in cases]
    }
    current = snapshot_from_records(cases, subset_records, scheme_order=("single_call_llm",))
    assert compare_snapshots(current, baseline).ok is True
    extra_records = {
        "agent": [_make_record(c.eval_case_id, "PASS") for c in cases],
        "single_call_llm": [_make_record(c.eval_case_id, "PASS") for c in cases],
    }
    current_extra = snapshot_from_records(
        cases, extra_records, scheme_order=("single_call_llm", "agent")
    )
    report = compare_snapshots(current_extra, baseline)
    assert report.ok is False
    assert any("不在基线快照" in line for line in report.summary)


async def test_regression_real_run_twice_passes(tmp_path: Path) -> None:
    """真实跑同一评测集两次 → 快照 digest 一致 → PASS（确定性重放；基线走 tmp_path 不碰数据目录）。"""
    data = _subset_path(tmp_path)
    baseline_file = tmp_path / "regression_baseline.json"
    snap1 = await compute_current_snapshot(data, schemes=("single_call_llm", "rule"))
    write_baseline(snap1, baseline_file)
    from pra.evaluation.regression import run_regression

    report = await run_regression(data, baseline_file, schemes=("single_call_llm", "rule"))
    assert report.ok is True and report.status == "PASS"
    # 再跑一次当前快照（不落盘）仍一致 —— 逐字节确定性
    snap2 = await compute_current_snapshot(data, schemes=("single_call_llm", "rule"))
    assert snap2["digest"] == snap1["digest"]
