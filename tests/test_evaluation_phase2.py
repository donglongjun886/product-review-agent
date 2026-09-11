"""Evaluation Phase 2 测试：Abstention / Ablation / Sweep / Regression 四组。

- ``test_abstention_*``：五指标在手工可算小样本上数值正确（SHOULD_ABSTAIN 被自动
  只表现为 recall 缺口、不进 wrong_auto；分母 0 → None；老数据无 abstain_label 兼容）；
- ``test_ablation_*``：方案级 2b vs 2a 在"预塞政策文本能命中"的案上决策不同；组件级
  裁剪 allowed_tools 后 tool_calls_actual 不含被裁工具且强视觉案决策改变；
- ``test_sweep_*``：阈值参数确实穿透评估路径（min_sim 0.70 vs 0.85 改变 EC_0303 决策）；
- ``test_regression_*``：篡改记录后比对失败，真实跑 v1 集两次 digest 一致。

全程离线确定性；测试数据来自 eval_data/v1（只读），基线快照一律写 tmp_path。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pra.evaluation.ablation import build_rag_context
from pra.evaluation.dataset.loader import load_dataset
from pra.evaluation.dataset.schema import EvalCase
from pra.evaluation.harness.agent_scheme import AgentScheme
from pra.evaluation.harness.base import EvalContext, EvalRecord
from pra.evaluation.harness.single_call_scheme import SingleCallScheme
from pra.evaluation.metrics.abstention import AbstentionEvaluator
from pra.evaluation.regression import (
    compare_snapshots,
    compute_current_snapshot,
    snapshot_from_records,
    write_baseline,
)
from pra.evaluation.runner import EvaluationRunner
from pra.evaluation.sweep import EVIDENCE_GRID, ThresholdSweepRunner

DATA_PATH = Path(__file__).resolve().parents[1] / "eval_data" / "v1" / "cases_v1.jsonl"

_DECISION_SET = {"PASS", "REJECT", "HUMAN_REVIEW"}


def _make_record(case_id: str, decision: str, scheme: str = "rule") -> EvalRecord:
    return EvalRecord(eval_case_id=case_id, scheme=scheme, decision=decision)  # type: ignore[arg-type]


def _cases_by_id() -> dict[str, EvalCase]:
    return {c.eval_case_id: c for c in load_dataset(DATA_PATH)}


async def _run_scheme(scheme, case: EvalCase, ctx: EvalContext) -> EvalRecord:
    """await 单个 scheme.run —— 供 async 测试复用（pytest-asyncio auto 模式驱动）。"""
    return await scheme.run(case, ctx)


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


async def test_single_call_2b_beats_2a_on_decisive_policy() -> None:
    """OCR 弱证据案：2a（Raw Input）→ HUMAN（置信不足转人工）；2b（预塞命中本案的自动拒绝判例文本）→ REJECT —— 仅当 RAG 文本命中时决策不同。"""
    from pra.domain.models import ProductImage, ProductInfo

    # 复用 EC_0406 的真实基础输入形状（标题无规避词），但把图片 OCR 换成命中"复刻"
    # （弱证据 REJECT 候选）→ base 规则② REJECT conf 0.60 → 2a 确定性转人工
    base = _cases_by_id()["EC_0406"]
    product: ProductInfo = base.input.product
    images = [
        ProductImage(
            url="https://cdn.example.com/eval/bound_bag/img1.jpg",
            ocr_text="复刻经典托特包图案",  # OCR 仿冒词 → base 规则② REJECT conf 0.60
            source="主图",
        )
    ]
    new_input = base.input.model_copy(
        update={"product": product.model_copy(update={"images": images})}
    )
    case = base.model_copy(update={"input": new_input})

    ctx = EvalContext()  # abstain threshold 0.7
    rec_2a = await _run_scheme(SingleCallScheme(), case, ctx)
    assert rec_2a.decision == "HUMAN_REVIEW", "2a：OCR 弱证据 REJECT 候选 conf<0.7 → 转人工"

    # 2b：预塞一条"类目同型、含复刻词、结论 → REJECT"的判例 digest（静态事实文本）
    context = ["判例: CASE_TEST_9 类目[箱包/女包] 复刻 仿冒确证 判定拒绝 → REJECT"]
    rec_2b = await _run_scheme(SingleCallScheme(extra_context=context), case, ctx)
    assert rec_2b.decision == "REJECT", "2b：RAG 命中同型自动拒绝判例 → 不再转人工"

    # 预塞的是转人工口径/无命中词 → 维持原样（机制不"见 REJECT 就升级"）
    context_human = ["判例: CASE_TEST_8 类目[箱包/女包] 复刻 判定转人工 → HUMAN_REVIEW"]
    rec_noop = await _run_scheme(SingleCallScheme(extra_context=context_human), case, ctx)
    assert rec_noop.decision == "HUMAN_REVIEW"

    # 无 extra_context 的默认构造 = Phase 1 行为（llm_fn 直接消费 case_json，无注入键）
    scheme_plain = SingleCallScheme()
    assert scheme_plain.extra_context is None
    assert (await _run_scheme(scheme_plain, case, ctx)).decision == "HUMAN_REVIEW"


def test_rag_context_from_eval_world_has_no_expected_answer() -> None:
    """build_rag_context 只含评测世界静态判例/政策文本 —— 不含任何 expected 字段值。"""
    cases = _cases_by_id()
    for cid in ("EC_0105", "EC_0401", "EC_0001"):
        case = cases[cid]
        for line in build_rag_context(case):
            # 只允许真实先例/政策来源的素材字段出现在 digest 行里
            assert not any(
                marker in line
                for marker in ("expected", "abstain_label", "SHOULD_ABSTAIN", "AUTO_DECIDABLE")
            )


async def test_component_ablation_cuts_tool_registration_and_calls() -> None:
    """组件级：allowed_tools 装配裁剪 → 图工具注册与 plan 都不含被裁工具，tool_calls_actual 不含被裁工具；强视觉案去掉 ImageTool 后决策改变（差异归因）。"""
    cases = _cases_by_id()
    ctx = EvalContext()
    ec0401 = cases["EC_0401"]

    full = await _run_scheme(AgentScheme(), ec0401, ctx)
    assert full.decision == "REJECT"
    assert "ImageAnalysisTool" in full.tool_calls_actual

    # 正式裁剪路径：allowed_tools = 全工具 − 被裁（工具注册 + plan schema 都只给该子集）
    all_tools = {"ProductTool", "ImageAnalysisTool", "MerchantTool",
                 "CaseSearchTool", "PolicySearchTool"}
    no_image = AgentScheme(allowed_tools=all_tools - {"ImageAnalysisTool"})
    rec_no_image = await _run_scheme(no_image, ec0401, ctx)
    assert "ImageAnalysisTool" not in rec_no_image.tool_calls_actual
    assert rec_no_image.decision == "HUMAN_REVIEW", "强视觉案无 ImageTool → 视觉未决 → 转人工"

    # −CaseTool：注册层去掉 CaseSearchTool → tool_calls_actual 不含它（PolicySearch 仍在）
    no_case = AgentScheme(allowed_tools=all_tools - {"CaseSearchTool"})
    rec_no_case = await _run_scheme(no_case, ec0401, ctx)
    assert "CaseSearchTool" not in rec_no_case.tool_calls_actual
    assert "PolicySearchTool" in rec_no_case.tool_calls_actual
    assert rec_no_case.decision in _DECISION_SET


def test_sweep_threshold_penetrates_sim_stats() -> None:
    """白盒：相似度 0.72 在 min_sim=0.70 时可见、0.75 时不可见；strong=0.70 时算强相似、0.85 时不算 —— sweep 档位直达相似度分档读取路径。"""
    evs = [{"type": "IMAGE_SIMILARITY", "weight": 0.72}]
    from pra.evaluation.harness import agent_scheme as A

    sim_max, strong70, any70 = A._sim_stats(evs, strong=0.70, min_sim=0.70)
    assert sim_max == pytest.approx(0.72) and strong70 is True and any70 is True
    sim_max, strong85, any85 = A._sim_stats(evs, strong=0.85, min_sim=0.70)
    assert strong85 is False and any85 is True  # 弱相似（0.70~0.85）
    sim_max, _s, any75 = A._sim_stats(evs, strong=0.85, min_sim=0.75)
    assert sim_max == pytest.approx(0.0) and any75 is False  # 低于证据下限 → 不可见


async def test_sweep_min_sim_changes_agent_decision_and_metrics() -> None:
    """同一（真实）数据两档阈值跑出不同 agent metrics：min_sim 0.85 抬证据下限 → EC_0303（弱相似 0.73 + 脏商家）由 REJECT 转 HUMAN（弱视觉证据被挡）—— 阈值穿透到决策。"""
    cases = _cases_by_id()
    weak_case = cases["EC_0303"]
    ctx_lo = EvalContext(evidence_thresholds={"min_sim": 0.70, "strong": 0.85})
    ctx_hi = EvalContext(evidence_thresholds={"min_sim": 0.85, "strong": 0.85})
    r_lo = await _run_scheme(AgentScheme(), weak_case, ctx_lo)
    r_hi = await _run_scheme(AgentScheme(), weak_case, ctx_hi)
    assert r_lo.decision == "REJECT"
    assert r_hi.decision == "HUMAN_REVIEW"

    trio = [cases["EC_0303"], cases["EC_0202"], cases["EC_0001"]]
    res_lo = await EvaluationRunner(ctx=ctx_lo).run(cases=trio, include=("agent",))
    res_hi = await EvaluationRunner(ctx=ctx_hi).run(cases=trio, include=("agent",))
    m_lo, m_hi = res_lo.overall["agent"], res_hi.overall["agent"]
    assert m_lo.human_pred == 0 and m_hi.human_pred == 1
    assert m_lo.human_rate == pytest.approx(0.0)
    assert m_hi.human_rate == pytest.approx(1 / 3)


async def test_sweep_runner_grid_rows_rule_single_flat_agent_moves() -> None:
    """ThresholdSweepRunner：min_sim 全网格跑同一个小数据集 —— rule / single_call_llm 对阈值不敏感（行恒定）；agent 在 min_sim≥0.75 后变。"""
    cases = [c for c in load_dataset(DATA_PATH) if c.eval_case_id in ("EC_0303", "EC_0202", "EC_0001")]
    runner = ThresholdSweepRunner(data_path=str(DATA_PATH))
    result = await runner.run(
        cases=cases,
        variables=("EVIDENCE_MIN_SIM",),
        schemes=("rule", "single_call_llm", "agent"),
    )
    assert [p.value for p in result.points] == list(EVIDENCE_GRID)
    # rule / single_call 行恒定（metrics dict 的 metric_row 序列跨档全等）
    rule_rows = {p.value: p.metrics_by_scheme["rule"].metric_row() for p in result.points}
    single_rows = {p.value: p.metrics_by_scheme["single_call_llm"].metric_row() for p in result.points}
    assert len({tuple(sorted(r.items())) for r in rule_rows.values()}) == 1
    assert len({tuple(sorted(r.items())) for r in single_rows.values()}) == 1
    agent_humans = [p.metrics_by_scheme["agent"].human_pred for p in result.points]
    # 0.60~0.70 弱相似可见（EC_0303 REJECT）；≥0.75 被挡（EC_0303 HUMAN）
    assert agent_humans[:3] == [0, 0, 0]
    assert agent_humans[3:] == [1, 1, 1, 1]
    assert any(p.thresholds["min_sim"] == pytest.approx(0.75) for p in result.points)


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
    from pra.evaluation.regression import canonical_digest

    tampered["digest"] = canonical_digest(tampered["per_case_ids"], tampered["decisions"])
    report = compare_snapshots(tampered, baseline)
    assert report.ok is False and report.status == "FAIL"
    assert "agent" in report.mismatches
    assert report.mismatches["agent"][0][2] == "PASS"  # 基线值
    assert report.mismatches["agent"][0][3] == "HUMAN_REVIEW"  # 当前值


def test_regression_subset_compare_and_missing_scheme() -> None:
    """基线只录两方案、当前只跑子集 → 同决策 PASS（子集比对不误报）；当前含基线未录的方案 → FAIL（提示重录）。"""
    cases = [c for c in load_dataset(DATA_PATH)][:3]
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


async def test_regression_real_v1_run_twice_passes(tmp_path: Path) -> None:
    """真实跑 v1 集两次 → 快照 digest 一致 → PASS（确定性重放；基线走 tmp_path 不碰数据目录）。"""
    baseline_file = tmp_path / "regression_baseline.json"
    snap1 = await compute_current_snapshot(str(DATA_PATH), schemes=("single_call_llm", "rule"))
    write_baseline(snap1, baseline_file)
    from pra.evaluation.regression import run_regression

    report = await run_regression(str(DATA_PATH), baseline_file, schemes=("single_call_llm", "rule"))
    assert report.ok is True and report.status == "PASS"
    # 再跑一次当前快照（不落盘）仍一致 —— 逐字节确定性
    snap2 = await compute_current_snapshot(str(DATA_PATH), schemes=("single_call_llm", "rule"))
    assert snap2["digest"] == snap1["digest"]
