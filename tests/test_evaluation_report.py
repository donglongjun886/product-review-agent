"""评测报告与 runner 接线护栏：abstention 五指标、Console Report 区与数字行。

1. 数据含 SHOULD_ABSTAIN 真值时 runner 并行算出五指标，与直接调用
   AbstentionEvaluator 逐字段一致，AUTO/SHOULD 子集计数与数据吻合；
2. 正式集（320 案）五指标与固化数字对齐（rule hrr 全量分母 0.606 vs 决策指标分母
   0.540；agent abstention_recall=1.0 / wrong_auto=0 / abstention_rate=10/274）；
3. 渲染含三值真值分布、两套分母注记、五指标区、scene 分层注记与同口径耦合边界声明；
4. v1（无 abstain 标签）路径无五指标区，既有数字行逐字不变。

全程离线、被测对象确定性；只读 eval_data/v1、v2。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pra.evaluation.dataset.loader import load_dataset
from pra.evaluation.metrics.abstention import AbstentionEvaluator
from pra.evaluation.report import render_report
from pra.evaluation.runner import ALL_SCHEMES, EvaluationRunner

DATA_V1 = Path(__file__).resolve().parents[1] / "eval_data" / "v1" / "cases_v1.jsonl"
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


async def test_runner_v1_no_should_abstention_empty_abstention() -> None:
    # v1：真值仅 PASS/REJECT → abstention 区为空，走 Phase 1 兼容口径
    result = await EvaluationRunner(data_path=str(DATA_V1)).run(smoke=True, smoke_limit=10)
    assert result.has_should_abstain is False
    assert result.abstention == {}
    assert result.abstention_grouped == {}
    text = render_report(result)
    assert "Phase 1 二值" in text
    assert "无 abstain 标签" in text
    assert "abstention 五指标不适用" in text
    assert "行含义: hrr/autom=human_review_rate" not in text


# --- 2) 正式集（320 案）五指标与固化数字对齐


async def test_v2_full_run_abstention_matches_docs_s44() -> None:
    result = await EvaluationRunner(data_path=str(DATA_V2)).run()
    assert result.total_cases == 320 and result.has_should_abstain is True

    assert result.overall["rule"].human_rate == pytest.approx(148 / 274)
    assert result.overall["single_call_llm"].human_rate == pytest.approx(110 / 274)
    assert result.overall["agent"].human_rate == pytest.approx(10 / 274)
    assert result.overall["agent"].accuracy == pytest.approx(264 / 274)  # 显示值 0.964（3 位四舍五入）
    assert result.overall["rule"].accuracy == pytest.approx(104 / 274)  # 显示值 0.380
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
    assert rule_a.wrong_auto_decision_rate == pytest.approx(22 / 126)
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
    assert "Evaluation 三方案对比 Console Report" in text
    assert "Phase 1 三方案" not in text
    assert "真值案: 共 320 条（PASS=134 / REJECT=140 / HUMAN_REVIEW=46）" in text
    assert "二值真值 274" in text and "全量 320" in text
    assert "AUTO_DECIDABLE=274 / SHOULD_ABSTAIN=46" in text
    assert "boundary=96(二值70)" in text
    assert "HUMAN 真值案不计入下列 acc/prec/recall 数字" in text
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
    assert "标注-审查员同口径" in text
    assert "同口径耦合只会高估一致性" in text
    assert "real 对照" in text and "0.200" in text and "0.771" in text
    assert "留 Phase 2" not in text  # 陈旧文案已删（Phase 3 real 已执行）


# --- 4) v1 路径：既有数字行零变化（逐字比对）


async def test_render_report_v1_numeric_rows_unchanged() -> None:
    result = await EvaluationRunner(data_path=str(DATA_V1)).run()  # 全量 35 条
    text = render_report(result)
    assert "Phase 1 二值" in text
    # 决策指标数字行逐字固化（三方案行 + 成本列）
    assert (
        "rule              0.343  -  0.000  0.000  1.000  0.629  0.371  0/0/12/1  "
        "llm=0.0 tool=0.0 tok=0.0" in text
    )
    assert (
        "single_call_llm   0.514  1.000  0.857  0.000  0.143  0.457  0.543  6/0/12/1  "
        "llm=1.0 tool=0.0 tok=0.0" in text
    )
    assert (
        "agent             1.000  1.000  1.000  0.000  0.000  0.000  1.000  20/0/15/0  "
        "llm=6.37 tool=5.0 tok=0.0" in text
    )
    assert "真值案: 共 35 条（PASS=15 / REJECT=20 / HUMAN_REVIEW=0）" in text
