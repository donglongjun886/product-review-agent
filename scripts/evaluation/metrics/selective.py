"""头条指标：selective prediction 口径（coverage / selective risk / 错误拒绝率，docs/02 §5.1）。

三条分母纪律：coverage 分母 = 全部案（剔除疑问篮子）；selective risk 分母 = 被自动决策的案
（转人工不进分母，只压低 coverage）；错误拒绝率分母 = 真值 PASS 案。输入为判卷结果
（``CaseGrading`` 已携带真值侧 decision），失败案不产出记录、天然不进任何分母。
"""

from __future__ import annotations

from collections.abc import Collection

from evaluation.grading.grader import CaseGrading
from pydantic import BaseModel, Field

__all__ = ["SelectiveMetrics", "evaluate_selective"]

_AUTO_DECISIONS = frozenset({"PASS", "REJECT"})


class SelectiveMetrics(BaseModel):
    """一臂的头条三指标及各自分子/分母计数（比率为 None = 分母为 0，未定义）。"""

    total: int = Field(description="coverage 分母：全部案（已剔除疑问篮子）")
    auto_decided: int = Field(description="自动决策（PASS/REJECT）案数：coverage 分子、selective risk 分母")
    auto_wrong: int = Field(description="自动决策中与真值不一致的案数：selective risk 分子")
    truth_pass_total: int = Field(description="真值 PASS 案数：错误拒绝率分母")
    false_rejects: int = Field(description="真值 PASS 被自动 REJECT 的案数：错误拒绝率分子")
    coverage: float | None = None
    selective_risk: float | None = None
    false_reject_rate: float | None = None


def _ratio(numer: int, denom: int) -> float | None:
    return numer / denom if denom else None


def evaluate_selective(
    gradings: list[CaseGrading], *, exclude: Collection[str] = ()
) -> SelectiveMetrics:
    """从一臂的逐案判卷结果聚合同头条三指标。

    :param gradings: 一臂的判卷结果；
    :param exclude: 疑问篮子 eval_case_id 集合，从全部分母剔除。
    """
    total = auto_decided = auto_wrong = truth_pass_total = false_rejects = 0
    excluded = set(exclude)
    for g in gradings:
        if g.eval_case_id in excluded:
            continue
        total += 1
        if g.truth_decision == "PASS":
            truth_pass_total += 1
            if g.predicted_decision == "REJECT":
                false_rejects += 1
        if g.predicted_decision in _AUTO_DECISIONS:
            auto_decided += 1
            if not g.decision_match:
                auto_wrong += 1
    return SelectiveMetrics(
        total=total,
        auto_decided=auto_decided,
        auto_wrong=auto_wrong,
        truth_pass_total=truth_pass_total,
        false_rejects=false_rejects,
        coverage=_ratio(auto_decided, total),
        selective_risk=_ratio(auto_wrong, auto_decided),
        false_reject_rate=_ratio(false_rejects, truth_pass_total),
    )
