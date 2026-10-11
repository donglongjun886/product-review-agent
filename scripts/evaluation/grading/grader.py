"""逐案判卷器：EvalRecord × v3 真值的纯代码确定性比对（零 LLM，见 docs/02 §4.1）。"""

from __future__ import annotations

from evaluation.dataset.schema import EvalExpectedV3, EvalTruthV3
from evaluation.record import EvalRecord
from pydantic import BaseModel, ConfigDict, Field

__all__ = ["CaseGrading", "grade_case", "grade_records"]


class CaseGrading(BaseModel):
    """单案判卷结果：裁决 / 风险类型 / 工具调用 / 政策引用四项比对 + 真值侧聚合字段。"""

    model_config = ConfigDict(extra="forbid")

    eval_case_id: str
    scheme: str = Field(description="产出被判记录的臂名（rule / single / agent）")
    truth_decision: str = Field(description="真值裁决（指标层聚合用）")
    truth_abstain_label: str = Field(description="真值 abstention 标签（指标层聚合用）")
    predicted_decision: str
    decision_match: bool
    truth_risk_type: list[str]
    predicted_risk_type: list[str]
    risk_type_match: bool | None = Field(
        description="risk_type 集合是否一致；None = 真值未标注风险类型（不进比对）"
    )
    expected_tools: list[str]
    actual_tools: list[str]
    tools_missing: list[str] = Field(description="应调未调的工具")
    tools_extra: list[str] = Field(description="调了期望外的工具")
    tools_match: bool | None = Field(
        description="工具集合是否一致；None = expected_tools 为空（规则层可判，不进工具分母）"
    )
    expected_policies: list[str]
    actual_policies: list[str]
    policies_missing: list[str] = Field(description="应引未引的政策条款")
    policies_extra: list[str] = Field(description="引了期望外的政策条款")
    policies_match: bool | None = Field(
        description="政策引用集合是否一致；None = expected_policies 为空（不进比对）"
    )


def _set_diff(expected: list[str], actual: list[str]) -> tuple[list[str], list[str]]:
    """保序集合差：``(应调未调, 期望外)``。"""
    exp, act = set(expected), set(actual)
    missing = [x for x in dict.fromkeys(expected) if x not in act]
    extra = [x for x in dict.fromkeys(actual) if x not in exp]
    return missing, extra


def grade_case(record: EvalRecord, expected: EvalExpectedV3) -> CaseGrading:
    """逐格比对一条记录与 v3 真值；空期望项 → 该格 None 且差异列表置空（不进对应分母）。

    :param record: 某一臂产出的预测记录；
    :param expected: 同案 v3 真值（打标函数产物）。
    """
    tools_missing, tools_extra = (
        _set_diff(list(expected.expected_tools), record.tool_calls_actual)
        if expected.expected_tools
        else ([], [])
    )
    policies_missing, policies_extra = (
        _set_diff(list(expected.expected_policies), record.policy)
        if expected.expected_policies
        else ([], [])
    )
    return CaseGrading(
        eval_case_id=record.eval_case_id,
        scheme=record.scheme,
        truth_decision=expected.decision,
        truth_abstain_label=expected.abstain_label,
        predicted_decision=record.decision,
        decision_match=record.decision == expected.decision,
        truth_risk_type=list(expected.risk_type),
        predicted_risk_type=list(record.risk_type),
        risk_type_match=(
            set(record.risk_type) == set(expected.risk_type) if expected.risk_type else None
        ),
        expected_tools=list(expected.expected_tools),
        actual_tools=list(record.tool_calls_actual),
        tools_missing=tools_missing,
        tools_extra=tools_extra,
        tools_match=(not tools_missing and not tools_extra) if expected.expected_tools else None,
        expected_policies=list(expected.expected_policies),
        actual_policies=list(record.policy),
        policies_missing=policies_missing,
        policies_extra=policies_extra,
        policies_match=(
            (not policies_missing and not policies_extra) if expected.expected_policies else None
        ),
    )


def grade_records(records: list[EvalRecord], truth: list[EvalTruthV3]) -> list[CaseGrading]:
    """对一臂记录逐案判卷，返回与 records 同序的比对结果。

    :raises ValueError: 记录的 eval_case_id 不在真值中（契约不一致，显式失败不静默跳过）。
    """
    index = {t.eval_case_id: t.expected for t in truth}
    out: list[CaseGrading] = []
    for rec in records:
        expected = index.get(rec.eval_case_id)
        if expected is None:
            raise ValueError(
                f"判卷失败：记录 {rec.eval_case_id} 在真值中不存在（数据契约不一致，不静默跳过）"
            )
        out.append(grade_case(rec, expected))
    return out
