"""Agent 级指标：工具选择 / 推理正确性 / 边际证据增益。

只消费 ``EvalRecord`` × ``expected`` 真值（指标层唯一输入），**不参与任何判定** —— Agent
决策、Gate 与工具调度零依赖本模块。

三条口径不变量：

- **空真值不进分母**：``expected_tools=[]``（未标注工具期望）/ ``expected.risk_type=[]``
  一律排除，并在计数里显式给出被排除的案数 —— 避免把「未标注」读成「应调 0 个工具」；
- **不引入人为权重**：边际增益只做可核对的计数与比率，不做「新增证据数 × 翻转权重」这类
  合成分（权重无法自证，越合成越不可解释）；
- **分母为 0 的比率返回 None**（报告显示 "-"），不硬造 0/∞。

**自动代理声明**：推理正确性只实现可自动计算的两项（risk_type 覆盖 + risk_level 一致）；
"结论对但理由错"的语义层面判定无法自动化，本仓库也没有第二标注者 —— 不假装能测。
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, Field

from pra.evaluation.record import EvalRecord

__all__ = [
    "AgentMetricsBundle",
    "MarginalEvidenceGainEvaluator",
    "MarginalEvidenceGainMetrics",
    "ReasoningCorrectnessEvaluator",
    "ReasoningCorrectnessMetrics",
    "ToolSelectionEvaluator",
    "ToolSelectionMetrics",
]


def _ratio(numer: int, denom: int) -> float | None:
    return numer / denom if denom else None


# ---------------------------------------------------------------------------
# 1. Tool Selection
# ---------------------------------------------------------------------------


class ToolSelectionMetrics(BaseModel):
    """工具选择准确性（覆盖口径）+ 冗余调用观测。仅 agent scheme 有意义。

    ``tool_selection_accuracy``：分母 = 标注了工具期望的案；分子 = ``expected_tools ⊆
    实际调用工具集合``（覆盖口径，**不把"调用少"混进准确性**）。``redundant_tool_rate``
    单独看"期望之外还调了什么"（案级；``tool_calls_actual`` 已按工具名去重，故不做调用次数
    层面的浪费统计）。
    """

    total_records: int = Field(description="纳入统计的 record 数")
    cases_with_expectation: int = Field(description="expected_tools 非空的案数（指标分母）")
    covered_cases: int = Field(description="expected_tools ⊆ actual_tools 的案数")
    redundant_cases: int = Field(description="调用了期望外工具的案数")
    actual_tool_calls: int = Field(description="实际调用工具数合计（按工具名去重）")
    redundant_tool_calls: int = Field(description="期望外工具数合计 |actual − expected|")
    tool_selection_accuracy: float | None = None
    redundant_tool_rate: float | None = None


class ToolSelectionEvaluator:
    @staticmethod
    def evaluate(records: list[EvalRecord], expected: Mapping[str, Mapping]) -> ToolSelectionMetrics:
        total = with_exp = covered = redundant_cases = actual_n = redundant_n = 0
        for rec in records:
            exp = expected.get(rec.eval_case_id)
            if exp is None:
                continue
            total += 1
            wanted = {str(t) for t in (exp.get("expected_tools") or [])}
            if not wanted:  # 未标注工具期望 → 不进分母
                continue
            with_exp += 1
            actual = {str(t) for t in rec.tool_calls_actual}
            actual_n += len(actual)
            extra = actual - wanted
            redundant_n += len(extra)
            if extra:
                redundant_cases += 1
            if wanted <= actual:
                covered += 1
        return ToolSelectionMetrics(
            total_records=total,
            cases_with_expectation=with_exp,
            covered_cases=covered,
            redundant_cases=redundant_cases,
            actual_tool_calls=actual_n,
            redundant_tool_calls=redundant_n,
            tool_selection_accuracy=_ratio(covered, with_exp),
            redundant_tool_rate=_ratio(redundant_cases, with_exp),
        )


# ---------------------------------------------------------------------------
# 2. Reasoning Correctness（自动代理）
# ---------------------------------------------------------------------------


class ReasoningCorrectnessMetrics(BaseModel):
    """推理正确性**自动代理**：risk_type 覆盖 + risk_level 一致。

    只覆盖可自动判定的部分：结论与真值一致但风险类型/等级错，会被这两项抓到；语义层面的
    "理由是否成立"不在本指标内（无第二标注者，不做人工抽样复核）。
    """

    total_records: int
    cases_with_expected_risk_type: int = Field(description="expected.risk_type 非空的案数")
    risk_type_covered_cases: int = Field(description="expected.risk_type ⊆ 输出 risk_type 的案数")
    risk_type_coverage: float | None = None
    cases_with_expected_risk_level: int = Field(description="expected.risk_level 非空的案数")
    risk_level_agreement_cases: int = Field(description="输出 risk_level 与真值一致的案数")
    risk_level_agreement: float | None = None


class ReasoningCorrectnessEvaluator:
    @staticmethod
    def evaluate(records: list[EvalRecord], expected: Mapping[str, Mapping]) -> ReasoningCorrectnessMetrics:
        total = rt_n = rt_ok = rl_n = rl_ok = 0
        for rec in records:
            exp = expected.get(rec.eval_case_id)
            if exp is None:
                continue
            total += 1
            wanted_rt = {str(t) for t in (exp.get("risk_type") or [])}
            if wanted_rt:
                rt_n += 1
                if wanted_rt <= {str(t) for t in rec.risk_type}:
                    rt_ok += 1
            wanted_rl = exp.get("risk_level")
            if isinstance(wanted_rl, str) and wanted_rl:
                rl_n += 1
                if rec.risk_level == wanted_rl:
                    rl_ok += 1
        return ReasoningCorrectnessMetrics(
            total_records=total,
            cases_with_expected_risk_type=rt_n,
            risk_type_covered_cases=rt_ok,
            risk_type_coverage=_ratio(rt_ok, rt_n),
            cases_with_expected_risk_level=rl_n,
            risk_level_agreement_cases=rl_ok,
            risk_level_agreement=_ratio(rl_ok, rl_n),
        )


# ---------------------------------------------------------------------------
# 3. Marginal Evidence Gain
# ---------------------------------------------------------------------------


class MarginalEvidenceGainMetrics(BaseModel):
    """逐次工具调用的边际增益（**计数与比率，无加权合成**）。

    数据来自 ``EvalRecord.detail["tool_history"]``（agent 臂的审计透传，见
    ``scripts/run_evaluation.py``）：每次调用的 ``evidence_added``（新增证据引用）与
    ``decision_changed``（本轮 Gate 判定是否翻转）。增益统计只算 ``status=="ok"`` 的调用 ——
    失败/跳过的调用不可能带来证据，混进分母会稀释信号。``no_gain_calls`` 即"调了但没带来
    新证据"的直接计数（回答'为调查而调查'）。
    """

    records_with_history: int = Field(description="带 tool_history 的 record 数（仅 agent 有）")
    audited_tool_calls: int = Field(description="审计到的工具调用数（含失败/跳过）")
    ok_tool_calls: int = Field(description="status==ok 的工具调用数（增益分母）")
    calls_with_new_evidence: int = Field(description="新增证据非空的调用数")
    no_gain_calls: int = Field(description="ok 但没有新增证据的调用数")
    calls_decision_changed: int = Field(description="触发 Gate 判定翻转的调用数")
    evidence_added_total: int = Field(description="新增证据引用合计")
    evidence_gain_rate: float | None = None
    decision_changed_rate: float | None = None
    avg_evidence_added: float | None = None


class MarginalEvidenceGainEvaluator:
    @staticmethod
    def evaluate(records: list[EvalRecord]) -> MarginalEvidenceGainMetrics:
        with_hist = audited = ok_n = gained = changed = added_total = 0
        for rec in records:
            history = (rec.detail or {}).get("tool_history")
            if not isinstance(history, list) or not history:
                continue
            with_hist += 1
            for item in history:
                if not isinstance(item, dict):
                    continue
                audited += 1
                if item.get("status") != "ok":
                    continue
                ok_n += 1
                added = item.get("evidence_added") or []
                count = len(added) if isinstance(added, list) else 0
                added_total += count
                if count:
                    gained += 1
                if item.get("decision_changed"):
                    changed += 1
        return MarginalEvidenceGainMetrics(
            records_with_history=with_hist,
            audited_tool_calls=audited,
            ok_tool_calls=ok_n,
            calls_with_new_evidence=gained,
            no_gain_calls=ok_n - gained,
            calls_decision_changed=changed,
            evidence_added_total=added_total,
            evidence_gain_rate=_ratio(gained, ok_n),
            decision_changed_rate=_ratio(changed, ok_n),
            avg_evidence_added=(added_total / ok_n) if ok_n else None,
        )


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------


class AgentMetricsBundle(BaseModel):
    """Agent 级指标三件套（仅 agent scheme 适用；其余方案为 None）。"""

    tool_selection: ToolSelectionMetrics
    reasoning_correctness: ReasoningCorrectnessMetrics
    marginal_gain: MarginalEvidenceGainMetrics

    @staticmethod
    def evaluate(
        records: list[EvalRecord], expected: Mapping[str, Mapping]
    ) -> AgentMetricsBundle:
        return AgentMetricsBundle(
            tool_selection=ToolSelectionEvaluator.evaluate(records, expected),
            reasoning_correctness=ReasoningCorrectnessEvaluator.evaluate(records, expected),
            marginal_gain=MarginalEvidenceGainEvaluator.evaluate(records),
        )
