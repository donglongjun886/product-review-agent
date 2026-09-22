"""决策业务指标：DecisionEvaluator —— 全量分母与 AUTO_DECIDABLE 分母一次遍历同算。

真值子集由 expected 的 (decision, abstain_label) 推导：``abstain_label`` 优先；缺失时
``decision == HUMAN_REVIEW`` → SHOULD_ABSTAIN，其余 → AUTO_DECIDABLE。

指标口径（REJECT 为正类）：

- 全量分母（全部案，含 HUMAN_REVIEW 真值）：
  ``automation_coverage`` = pred ∈ {PASS,REJECT} / total；
  ``human_review_rate`` = pred HUMAN_REVIEW / total；
- AUTO_DECIDABLE 分母（本可自动判的案）：
  ``accuracy`` = (tp+tn) / auto_decidable_total（pred HUMAN 入分母、不入分子）；
  ``precision`` = tp/(tp+fp)；
  ``wrong_auto_decision_rate`` = (fp+fn)/(tp+fp+tn+fn)（pred HUMAN 不入该分母）；
- 真值 REJECT 分母（全部真违规案）：
  ``recall`` = tp / reject_truth（pred HUMAN_REVIEW 记为该真违规未被自动拦下，入分母不入 TP/FN）；
  ``reject_unhandled`` = reject_human / reject_truth。

真值 REJECT 案被 ``tp`` / ``reject_human`` / ``fn`` 三个互斥桶完全划分，故恒有
``recall + reject_unhandled + fn/reject_truth = 1`` —— 缺口据此区分"安全转人工"与"漏放"。

分母为 0 的比率返回 None（报告显示 "-"，不硬造 0/∞）。
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, Field

from pra.evaluation.harness.base import EvalRecord

__all__ = ["DecisionEvaluator", "DecisionMetrics"]

_AUTO = "AUTO_DECIDABLE"
_SHOULD = "SHOULD_ABSTAIN"


class DecisionMetrics(BaseModel):
    """两套分母的业务指标 + 各自分子分母计数（None = 分母为 0，未定义）。"""

    total: int = Field(description="全部案数（全量分母）")
    human_pred_total: int = Field(description="pred HUMAN_REVIEW 的案数（human_review_rate 分子）")
    auto_decidable_total: int = Field(description="AUTO_DECIDABLE 案数（accuracy 分母）")
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    reject_truth: int = Field(default=0, description="truth REJECT 案数（recall 与 reject_unhandled 分母）")
    reject_human: int = Field(default=0, description="truth REJECT ∧ pred HUMAN（reject_unhandled 分子）")

    automation_coverage: float | None = None  # (total − human_pred_total) / total
    human_review_rate: float | None = None  # human_pred_total / total
    accuracy: float | None = None  # (tp+tn) / auto_decidable_total
    precision: float | None = None  # tp / (tp+fp)
    recall: float | None = None  # tp / reject_truth（转人工入分母、不入 TP/FN）
    wrong_auto_decision_rate: float | None = None  # (fp+fn) / (tp+fp+tn+fn)
    reject_unhandled: float | None = None  # reject_human / reject_truth


def _ratio(numer: int, denom: int) -> float | None:
    return numer / denom if denom else None


def _abstain_subset_of(expected: Mapping) -> str:
    """由 expected 的 (decision, abstain_label) 推导真值子集（AUTO_DECIDABLE / SHOULD_ABSTAIN）。

    :param expected: ``{"decision": ..., "abstain_label": ...}``；abstain_label 可为 None。
    """
    label = expected.get("abstain_label")
    if label == _SHOULD:
        return _SHOULD
    if label == _AUTO:
        return _AUTO
    if expected.get("decision") == "HUMAN_REVIEW":
        return _SHOULD
    return _AUTO


class DecisionEvaluator:
    """DecisionEvaluator —— 只吃 EvalRecord.decision × expected 真值。

    expected 索引由调用方构造：``{eval_case_id: {"decision": ..., "abstain_label": ...}}``。
    """

    @staticmethod
    def evaluate(records: list[EvalRecord], expected: Mapping[str, Mapping]) -> DecisionMetrics:
        """统计一组 EvalRecord 的决策指标。

        :param records: 同一 scheme 的记录；eval_case_id 不在 expected 中的记录跳过。
        :param expected: 真值索引。SHOULD_ABSTAIN 案只进全量分母，不进 AUTO_DECIDABLE 分母。
        """
        total = human_pred = auto_decidable_total = 0
        tp = fp = tn = fn = 0
        reject_truth = reject_human = 0
        for rec in records:
            exp = expected.get(rec.eval_case_id)
            if exp is None:
                continue
            pred = rec.decision
            truth = exp.get("decision")
            total += 1
            if pred == "HUMAN_REVIEW":
                human_pred += 1
            if truth == "REJECT":
                reject_truth += 1
                if pred == "HUMAN_REVIEW":
                    reject_human += 1
            if _abstain_subset_of(exp) == _SHOULD:
                continue
            auto_decidable_total += 1
            if pred == "REJECT":
                if truth == "REJECT":
                    tp += 1
                else:
                    fp += 1
            elif pred == "PASS":
                if truth == "PASS":
                    tn += 1
                else:
                    fn += 1

        return DecisionMetrics(
            total=total,
            human_pred_total=human_pred,
            auto_decidable_total=auto_decidable_total,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            reject_truth=reject_truth,
            reject_human=reject_human,
            automation_coverage=_ratio(total - human_pred, total),
            human_review_rate=_ratio(human_pred, total),
            accuracy=_ratio(tp + tn, auto_decidable_total),
            precision=_ratio(tp, tp + fp),
            recall=_ratio(tp, reject_truth),
            wrong_auto_decision_rate=_ratio(fp + fn, tp + fp + tn + fn),
            reject_unhandled=_ratio(reject_human, reject_truth),
        )
