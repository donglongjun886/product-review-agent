"""Agent 级指标（metrics/agent.py）与工程指标（metrics/engineering.py）单测。

口径不变量（本文件即契约）：
- 空真值不进分母：``expected_tools=[]`` / ``expected.evidence=[]`` / ``expected.risk_type=[]``
  的案一律排除，且被排除的案数在结果里可见；
- 证据类型覆盖按**可映射标签**算，无法映射的标签只计数、不硬猜；
- 边际增益只统计 ``status=="ok"`` 的调用，无 ``tool_history`` → 比率为 None；
- 分位数为 nearest-rank（无插值、无随机）。
"""

from __future__ import annotations

from pra.evaluation.harness.base import EvalRecord
from pra.evaluation.metrics.agent import (
    EvidenceSufficiencyEvaluator,
    MarginalEvidenceGainEvaluator,
    ReasoningCorrectnessEvaluator,
    ToolSelectionEvaluator,
)
from pra.evaluation.metrics.engineering import EngineeringEvaluator, percentile


def _rec(
    case_id: str,
    *,
    decision: str = "PASS",
    tools: list[str] | None = None,
    evidence: list[dict] | None = None,
    policy: list[str] | None = None,
    risk_type: list[str] | None = None,
    risk_level: str = "NONE",
    cost: dict | None = None,
    tool_history: list[dict] | None = None,
) -> EvalRecord:
    detail = {"tool_history": tool_history} if tool_history is not None else {}
    return EvalRecord(
        eval_case_id=case_id,
        scheme="agent",
        decision=decision,
        risk_level=risk_level,
        risk_type=list(risk_type or []),
        evidence=list(evidence or []),
        policy=list(policy or []),
        tool_calls_actual=list(tools or []),
        cost=dict(cost or {}),
        detail=detail,
    )


def _exp(**kwargs) -> dict:
    base = {
        "decision": "PASS",
        "scene": "normal",
        "abstain_label": "AUTO_DECIDABLE",
        "expected_tools": [],
        "evidence": [],
        "risk_type": [],
        "risk_level": None,
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# 1. Tool Selection
# ---------------------------------------------------------------------------


def test_tool_selection_coverage_and_empty_truth_excluded():
    records = [
        _rec("c1", tools=["ProductTool", "MerchantTool"]),  # 覆盖命中
        _rec("c2", tools=["ProductTool"]),  # 缺 MerchantTool → 未覆盖
        _rec("c3", tools=["ProductTool", "ImageAnalysisTool"]),  # 覆盖 + 期望外调用
        _rec("c4", tools=["ProductTool"]),  # 未标注 → 不进分母
    ]
    expected = {
        "c1": _exp(expected_tools=["ProductTool", "MerchantTool"]),
        "c2": _exp(expected_tools=["ProductTool", "MerchantTool"]),
        "c3": _exp(expected_tools=["ProductTool"]),
        "c4": _exp(),  # expected_tools 空 = 未标注
    }
    m = ToolSelectionEvaluator.evaluate(records, expected)
    assert m.cases_with_expectation == 3  # c4 被排除
    assert m.covered_cases == 2  # c1/c3
    assert m.tool_selection_accuracy == 2 / 3
    assert m.redundant_cases == 1  # c3 多调了 ImageAnalysisTool
    assert m.redundant_tool_calls == 1
    assert m.actual_tool_calls == 5  # 2+1+2，去重后工具名合计
    assert m.redundant_tool_rate == 1 / 3


def test_tool_selection_no_expectation_anywhere_returns_none_rate():
    m = ToolSelectionEvaluator.evaluate([_rec("c1", tools=["ProductTool"])], {"c1": _exp()})
    assert m.cases_with_expectation == 0
    assert m.tool_selection_accuracy is None and m.redundant_tool_rate is None


def test_tool_selection_missing_expected_record_skipped():
    m = ToolSelectionEvaluator.evaluate([_rec("c1", tools=["ProductTool"])], {})
    assert m.total_records == 0 and m.tool_selection_accuracy is None


# ---------------------------------------------------------------------------
# 2. Evidence Sufficiency
# ---------------------------------------------------------------------------


def test_evidence_type_coverage_micro_and_unmapped_visible():
    records = [
        _rec("c1", evidence=[{"type": "IMAGE_SIMILARITY"}, {"type": "MERCHANT_HISTORY"}]),
        _rec("c2", evidence=[{"type": "IMAGE_SIMILARITY"}]),
        _rec("c3"),  # 真值只有无法映射的标签 → 剔除出分母
    ]
    expected = {
        "c1": _exp(evidence=["image_similarity>=0.85", "merchant_history>=5_removals"]),
        "c2": _exp(evidence=["image_similarity>=0.85", "merchant_history>=5_removals"]),
        "c3": _exp(evidence=["brand_missing"]),  # 无映射
    }
    m = EvidenceSufficiencyEvaluator.evaluate(records, expected)
    assert m.cases_with_expected_evidence == 2
    assert (m.covered_expected_types, m.total_expected_types) == (3, 4)
    assert m.evidence_type_coverage == 3 / 4
    assert m.fully_covered_cases == 1  # 仅 c1 全命中
    assert m.unmapped_label_instances == 1
    assert m.cases_excluded_unmapped_only == 1


def test_evidence_type_coverage_prefix_match_variant_labels():
    """带阈值/区间的标签按前缀映射（``image_similarity 0.70~0.85`` 也算 IMAGE_SIMILARITY）。"""
    records = [_rec("c1", evidence=[{"type": "IMAGE_SIMILARITY"}])]
    expected = {"c1": _exp(evidence=["image_similarity 0.70~0.85"])}
    m = EvidenceSufficiencyEvaluator.evaluate(records, expected)
    assert m.evidence_type_coverage == 1.0


def test_reject_gate_pass_rate_requires_citable_ground():
    records = [
        _rec("c1", decision="REJECT", policy=["POLICY_5.2"]),  # 有政策 → 通过
        _rec("c2", decision="REJECT", evidence=[{"type": "POLICY_REF", "ref_id": "p1"}]),  # 可引用
        _rec("c3", decision="REJECT", evidence=[{"type": "IMAGE_SIMILARITY", "ref_id": "img"}]),  # 不算
        _rec("c4", decision="REJECT", evidence=[{"type": "POLICY_REF", "ref_id": None}]),  # 无 ref → 不算
        _rec("c5", decision="PASS"),  # 非 REJECT → 不进分母
    ]
    expected = {f"c{i}": _exp(decision="REJECT" if i < 5 else "PASS") for i in range(1, 6)}
    m = EvidenceSufficiencyEvaluator.evaluate(records, expected)
    assert m.reject_cases_pred_reject == 4
    assert m.reject_cases_with_citable == 2
    assert m.reject_evidence_gate_pass_rate == 0.5


# ---------------------------------------------------------------------------
# 3. Reasoning Correctness
# ---------------------------------------------------------------------------


def test_reasoning_correctness_risk_type_and_level():
    records = [
        _rec("c1", risk_type=["POTENTIAL_IP_RISK"], risk_level="HIGH"),
        _rec("c2", risk_type=[], risk_level="NONE"),  # risk_type 未覆盖 + level 不一致
        _rec("c3", risk_type=["EVASION_PATTERN"], risk_level="LOW"),  # 无期望 risk_type
    ]
    expected = {
        "c1": _exp(risk_type=["POTENTIAL_IP_RISK"], risk_level="HIGH"),
        "c2": _exp(risk_type=["POTENTIAL_IP_RISK"], risk_level="HIGH"),
        "c3": _exp(risk_type=[], risk_level="LOW"),
    }
    m = ReasoningCorrectnessEvaluator.evaluate(records, expected)
    assert m.cases_with_expected_risk_type == 2
    assert m.risk_type_coverage == 0.5
    assert m.cases_with_expected_risk_level == 3
    assert m.risk_level_agreement == 2 / 3


def test_reasoning_correctness_empty_truth_returns_none():
    m = ReasoningCorrectnessEvaluator.evaluate([_rec("c1")], {"c1": _exp(risk_level=None)})
    assert m.risk_type_coverage is None and m.risk_level_agreement is None


# ---------------------------------------------------------------------------
# 4. Marginal Evidence Gain
# ---------------------------------------------------------------------------


def test_marginal_gain_counts_only_ok_calls():
    history = [
        {"tool": "ProductTool", "status": "ok", "evidence_added": ["e1"], "decision_changed": False},
        {"tool": "MerchantTool", "status": "ok", "evidence_added": [], "decision_changed": True},
        {"tool": "OCRTool", "status": "error", "evidence_added": [], "decision_changed": False},
        {"tool": "OCRTool", "status": "skipped", "evidence_added": [], "decision_changed": False},
    ]
    m = MarginalEvidenceGainEvaluator.evaluate([_rec("c1", tool_history=history)])
    assert m.records_with_history == 1
    assert m.audited_tool_calls == 4  # 含失败/跳过
    assert m.ok_tool_calls == 2  # 增益分母只算 ok
    assert m.calls_with_new_evidence == 1
    assert m.no_gain_calls == 1
    assert m.calls_decision_changed == 1
    assert m.evidence_added_total == 1
    assert m.evidence_gain_rate == 0.5 and m.decision_changed_rate == 0.5
    assert m.avg_evidence_added == 0.5


def test_marginal_gain_without_history_returns_none_rates():
    m = MarginalEvidenceGainEvaluator.evaluate([_rec("c1")])
    assert m.records_with_history == 0 and m.audited_tool_calls == 0
    assert m.evidence_gain_rate is None and m.avg_evidence_added is None


# ---------------------------------------------------------------------------
# 5. Engineering
# ---------------------------------------------------------------------------


def test_percentile_nearest_rank():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert percentile(values, 50) == 5  # ceil(0.5*10)=5 → 第 5 个
    assert percentile(values, 95) == 10  # ceil(9.5)=10
    assert percentile([3, 1, 2], 50) == 2  # 先排序
    assert percentile([], 50) is None
    assert percentile([7], 95) == 7


def test_engineering_distributions_and_optional_latency():
    records = [
        _rec("c1", cost={"llm_calls": 1, "tool_calls": 2, "tokens": 10}),
        _rec("c2", cost={"llm_calls": 3, "tool_calls": 4, "tokens": 30}),
    ]
    m = EngineeringEvaluator.evaluate(records)
    assert m.total_records == 2
    assert (m.llm_calls.mean, m.llm_calls.p50, m.llm_calls.p95) == (2.0, 1, 3)
    assert m.tool_calls.max == 4 and m.tokens.mean == 20.0
    assert m.latency_ms is None  # scripted 路径不传延迟

    with_latency = EngineeringEvaluator.evaluate(records, latency_ms=[100.0, 300.0])
    assert with_latency.latency_ms is not None
    assert (with_latency.latency_ms.mean, with_latency.latency_ms.p50) == (200.0, 100)


def test_engineering_empty_records():
    m = EngineeringEvaluator.evaluate([])
    assert m.total_records == 0
    assert m.llm_calls.mean is None and m.llm_calls.count == 0


# ---------------------------------------------------------------------------
# 6. Runner 接线（真跑 v1 35 案的确定性冒烟）
# ---------------------------------------------------------------------------


async def test_runner_wires_agent_metrics_and_engineering():
    """runner 只在 agent 方案上算 Agent 级指标；工程指标按方案各有一行。"""
    from pathlib import Path

    from pra.evaluation.runner import EvaluationRunner

    data = Path(__file__).resolve().parents[1] / "eval_data" / "v1" / "cases_v1.jsonl"
    result = await EvaluationRunner(data_path=data).run(include=("agent",))

    assert result.agent_metrics is not None
    am = result.agent_metrics
    # v1 有 15 条未标注 expected_tools → 不进 Tool Selection 分母
    assert am.tool_selection.cases_with_expectation == 20
    assert am.tool_selection.tool_selection_accuracy == 1.0
    # agent 每案都有 tool_history 透传（边际增益分母非空）
    assert am.marginal_gain.records_with_history == result.total_cases
    assert am.marginal_gain.evidence_gain_rate is not None
    # 工程指标：只跑 agent → 只有一行；scripted 桩 token 恒 0（真实值，不伪造）
    assert set(result.engineering) == {"agent"}
    assert result.engineering["agent"].tokens.mean == 0.0
    assert result.engineering["agent"].latency_ms is None
