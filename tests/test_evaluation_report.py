"""评测报告与 runner 接线护栏：abstention 五指标、Console Report 区与数字行。

1. 数据含 SHOULD_ABSTAIN 真值时 runner 并行算出五指标，与直接调用
   AbstentionEvaluator 逐字段一致，AUTO/SHOULD 子集计数与数据吻合；
2. 正式集（320 案）五指标与固化数字对齐（rule hrr 全量分母 0.606 vs 决策指标分母
   0.540；agent abstention_recall=1.0 / wrong_auto=0 / abstention_rate=10/274）；
3. 渲染含三值真值分布、两套分母注记与五指标区（**数值存在性**断言，不锁装饰性文案）；
4. 老格式（无 abstain 标签）路径无五指标区，报告走 Phase 1 兼容口径。

全程离线、被测对象确定性；只读 eval_data/v2。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pra.evaluation.dataset.loader import load_dataset
from pra.evaluation.metrics.abstention import AbstentionEvaluator
from pra.evaluation.report import render_report
from pra.evaluation.runner import ALL_SCHEMES, EvaluationRunner

DATA_V2 = Path(__file__).resolve().parents[1] / "eval_data" / "v2" / "cases_v2.jsonl"

# v2 确定性子集：5 条 AUTO_DECIDABLE（PASS×2/REJECT×3，覆盖五 scene）+ 3 条 SHOULD_ABSTAIN
# （evasion / multi-signal / boundary 各 1，decision=HUMAN_REVIEW）—— 用于轻量接线测试。
_V2_SUBSET_IDS = (
    "EC_V2_0001",  # normal     PASS   AUTO_DECIDABLE
    "EC_V2_0002",  # violation  REJECT AUTO_DECIDABLE
    "EC_V2_0003",  # boundary   PASS   AUTO_DECIDABLE
    "EC_V2_0004",  # multi-signal REJECT AUTO_DECIDABLE
    "EC_V2_0005",  # evasion    REJECT AUTO_DECIDABLE
    "EC_V2_0105",  # evasion    HUMAN  SHOULD_ABSTAIN
    "EC_V2_0260",  # multi-signal HUMAN SHOULD_ABSTAIN
    "EC_V2_0295",  # boundary   HUMAN  SHOULD_ABSTAIN
)
_V2_SUBSET_N_AUTO = 5
_V2_SUBSET_N_SHOULD = 3


def _load_subset(ids: tuple[str, ...]) -> list:
    cases = load_dataset(DATA_V2)
    by_id = {c.eval_case_id: c for c in cases}
    missing = [cid for cid in ids if cid not in by_id]
    assert not missing, f"v2 数据缺测试用 case: {missing}"
    return [by_id[cid] for cid in ids]


# --- 1) runner 接线：SHOULD 真值存在 → AbstentionEvaluator 五指标并行计算


async def test_runner_v2_wires_abstention_five_metrics() -> None:
    subset = _load_subset(_V2_SUBSET_IDS)
    result = await EvaluationRunner().run(cases=subset, include=ALL_SCHEMES)
    assert result.has_should_abstain is True
    assert set(result.abstention.keys()) == set(ALL_SCHEMES)
    for scheme in ALL_SCHEMES:
        a = result.abstention[scheme]
        # 接线正确性：runner 结果 == 用同 records/expected 直接调用 AbstentionEvaluator
        recomputed = AbstentionEvaluator.evaluate(result.records[scheme], result.expected)
        assert a == recomputed, f"{scheme}: runner abstention 与直接计算不一致"
        assert a.total == 8
        assert a.auto_decidable_total == _V2_SUBSET_N_AUTO
        assert a.should_abstain_total == _V2_SUBSET_N_SHOULD
        assert a.human_pred_total + a.auto_pred_total == a.total
        assert a.abstention_recall is not None  # SHOULD 分母非 0 → 已定义
    # DecisionEvaluator 二值分母不变（HUMAN 真值不进入其计数）
    assert result.overall["agent"].total == _V2_SUBSET_N_AUTO


def _legacy_rows_jsonl(tmp_path: Path, n: int = 3) -> Path:
    """老格式（无 abstain_label / 无 lineage、真值仅二值）最小评测集。"""
    lines: list[str] = []
    for c in load_dataset(DATA_V2)[:n]:
        row = c.model_dump(mode="json")
        row["schema_version"] = 1
        row["lineage"] = None
        row["expected"].pop("abstain_label", None)
        if row["expected"]["decision"] == "HUMAN_REVIEW":
            row["expected"]["decision"] = "PASS"  # 老格式没有三值真值
        lines.append(json.dumps(row, ensure_ascii=False))
    path = tmp_path / "legacy.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


async def test_runner_legacy_no_abstain_labels_runs_phase1_path(tmp_path: Path) -> None:
    # 老格式：真值仅 PASS/REJECT → abstention 区为空，走 Phase 1 兼容口径
    result = await EvaluationRunner(data_path=str(_legacy_rows_jsonl(tmp_path))).run()
    assert result.has_should_abstain is False
    assert result.abstention == {}
    assert result.abstention_grouped == {}
    text = render_report(result)
    assert "Phase 1 二值" in text
    assert "无 abstain 标签" in text
    assert "abstention 五指标不适用" in text
    assert "行含义: hrr/autom=human_review_rate" not in text


# --- 2) 正式集（320 案）五指标与固化数字对齐


async def test_v2_full_run_abstention_matches_docs() -> None:
    result = await EvaluationRunner(data_path=str(DATA_V2)).run()
    assert result.total_cases == 320 and result.has_should_abstain is True

    assert result.overall["rule"].human_rate == pytest.approx(148 / 274)
    assert result.overall["single_call_llm"].human_rate == pytest.approx(110 / 274)
    assert result.overall["agent"].human_rate == pytest.approx(10 / 274)
    assert result.overall["agent"].accuracy == pytest.approx(264 / 274)  # 显示值 0.964（3 位四舍五入）
    assert result.overall["rule"].accuracy == pytest.approx(116 / 274)  # 显示值 0.423
    # Rule 唯一的自动 REJECT 路径 = R-101 黑名单（v2 blackbrand_field 12 案）；prec 1.0 = 零误杀
    assert result.overall["rule"].precision == pytest.approx(1.0)
    assert result.overall["rule"].tp == 12
    for scheme in ALL_SCHEMES:
        a = result.abstention[scheme]
        assert a.total == 320
        assert a.auto_decidable_total == 274
        assert a.should_abstain_total == 46
        assert a.human_pred_total + a.auto_pred_total == 320
    rule_a = result.abstention["rule"]
    assert rule_a.human_review_rate == pytest.approx(194 / 320)  # (148 过度 + 46 SHOULD) / 320
    assert rule_a.automation_coverage == pytest.approx(126 / 320)
    assert rule_a.abstention_rate == pytest.approx(148 / 274)
    assert rule_a.abstention_recall == pytest.approx(1.0)
    assert rule_a.wrong_auto_decision_rate == pytest.approx(10 / 126)  # 漏放 22→10（R-101 直判 12 案）
    agent_a = result.abstention["agent"]
    assert agent_a.human_review_rate == pytest.approx(56 / 320)
    assert agent_a.automation_coverage == pytest.approx(264 / 320)
    assert agent_a.abstention_rate == pytest.approx(10 / 274)
    assert agent_a.abstention_recall == pytest.approx(1.0)
    assert agent_a.wrong_auto_decision_rate == pytest.approx(0.0)
    assert agent_a.human_review_rate > result.overall["agent"].human_rate


# --- 3) Console Report 渲染内容


async def test_render_report_v2_sections() -> None:
    result = await EvaluationRunner(data_path=str(DATA_V2)).run()
    text = render_report(result)
    # 数值断言（评测即产品，这些数字是主要产出）：三值真值分布、两套分母、五指标区、
    # agent accuracy 显示值。装饰性文案（标题、口径说明措辞）不逐字锁定。
    assert "PASS=134 / REJECT=140 / HUMAN_REVIEW=46" in text  # 三值真值分布
    assert "274" in text and "320" in text  # 二值真值分母 / 全量分母
    for marker in (
        "abstention 五指标",
        "human_review_rate",
        "automation_coverage",
        "abstention_rate",
        "abstention_recall",
        "wrong_auto_decision_rate",
        "1.000",
    ):
        assert marker in text, f"v2 Console Report 缺渲染要素: {marker}"
    assert "0.964" in text  # agent accuracy 显示值（264/274，见上方全量对齐用例）
