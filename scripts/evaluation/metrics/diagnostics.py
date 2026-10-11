"""诊断层：Agent 专用指标（docs/02 §5.2）—— 不进头条，回答"结论对了，理由对不对"。

四项：工具选择准确率 / 多余调用率（分母 = 标注了期望工具的案）、风险类型命中率
（分母 = 标注了期望风险类型的案）、有效取证率（分母 = 成功的工具调用，来自调查轨迹）。
前两项直接消费判卷结果（``CaseGrading`` 的集合差已由判卷器算出，此处只聚合不重现）。
"""

from __future__ import annotations

from evaluation.grading.grader import CaseGrading
from evaluation.record import EvalRecord
from pydantic import BaseModel, Field

__all__ = ["AgentDiagnostics", "evaluate_diagnostics"]


def _ratio(numer: int, denom: int) -> float | None:
    return numer / denom if denom else None


class AgentDiagnostics(BaseModel):
    """Agent 臂诊断四指标及各自分子/分母计数（比率为 None = 分母为 0，未定义）。"""

    tool_expected_cases: int = Field(description="标注了期望工具的案数（工具两项分母）")
    tool_covered_cases: int = Field(description="应调工具全部调到的案数（工具选择准确率分子）")
    tool_redundant_cases: int = Field(description="调了期望外工具的案数（多余调用率分子）")
    tool_selection_accuracy: float | None = None
    redundant_tool_rate: float | None = None
    risk_type_labeled_cases: int = Field(description="标注了期望风险类型的案数（风险类型命中率分母）")
    risk_type_hit_cases: int = Field(description="风险类型与真值一致的案数（风险类型命中率分子）")
    risk_type_hit_rate: float | None = None
    ok_tool_calls: int = Field(description="status==ok 的工具调用数（有效取证率分母）")
    calls_with_new_evidence: int = Field(description="带来新证据的 ok 调用数（有效取证率分子）")
    evidence_gain_rate: float | None = None


def evaluate_diagnostics(
    gradings: list[CaseGrading], records: list[EvalRecord]
) -> AgentDiagnostics:
    """聚合 agent 臂诊断指标。

    :param gradings: agent 臂逐案判卷结果（工具/风险类型比对来源）；
    :param records: agent 臂原始记录（tool_history 取证增益来源）。
    """
    tool_n = tool_covered = tool_redundant = 0
    rt_n = rt_hit = 0
    for g in gradings:
        if g.tools_match is not None:  # tools_match 非 None ⇔ 标注了期望工具
            tool_n += 1
            if not g.tools_missing:
                tool_covered += 1
            if g.tools_extra:
                tool_redundant += 1
        if g.risk_type_match is not None:
            rt_n += 1
            if g.risk_type_match:
                rt_hit += 1

    ok_calls = gained = 0
    for rec in records:
        history = (rec.detail or {}).get("tool_history")
        if not isinstance(history, list):
            continue
        for item in history:
            if not isinstance(item, dict) or item.get("status") != "ok":
                continue
            ok_calls += 1
            added = item.get("evidence_added") or []
            if isinstance(added, list) and added:
                gained += 1

    return AgentDiagnostics(
        tool_expected_cases=tool_n,
        tool_covered_cases=tool_covered,
        tool_redundant_cases=tool_redundant,
        tool_selection_accuracy=_ratio(tool_covered, tool_n),
        redundant_tool_rate=_ratio(tool_redundant, tool_n),
        risk_type_labeled_cases=rt_n,
        risk_type_hit_cases=rt_hit,
        risk_type_hit_rate=_ratio(rt_hit, rt_n),
        ok_tool_calls=ok_calls,
        calls_with_new_evidence=gained,
        evidence_gain_rate=_ratio(gained, ok_calls),
    )
