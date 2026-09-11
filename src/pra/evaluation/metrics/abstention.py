"""HUMAN_REVIEW / abstention 五指标：AbstentionEvaluator。

命名权威：**全代码统一用五个全名，不用缩写**：``human_review_rate`` /
``automation_coverage`` / ``abstention_rate`` / ``abstention_recall`` /
``wrong_auto_decision_rate``。

消费每 case 一对：
- expected：``decision`` + ``abstain_label``（None = 等价 AUTO_DECIDABLE）；
- EvalRecord.decision ∈ {PASS, REJECT, HUMAN_REVIEW}。

真值子集语义：``AUTO_DECIDABLE``（本可自动判）评价"自动决策是否正确安全"；
``SHOULD_ABSTAIN``（应转人工）评价"该转人工的克制地转了"。``abstain_label``
缺失时按 decision == HUMAN_REVIEW 推断 SHOULD_ABSTAIN（防 schema 升级滞后），
其余一律 AUTO_DECIDABLE。

五指标：
- ``human_review_rate`` = pred HUMAN / 全部（人工占用）；
- ``automation_coverage`` = pred ∈ {PASS,REJECT} / 全部 = 1 − human_review_rate；
- ``abstention_rate`` = AUTO_DECIDABLE 案中 pred HUMAN 占比（过度保守，越高越保守）；
- ``abstention_recall`` = SHOULD_ABSTAIN 案中 pred HUMAN 占比；被自动终裁的
  SHOULD_ABSTAIN 案即**危险误自动**，是它的分子缺口；
- ``wrong_auto_decision_rate`` = AUTO_DECIDABLE 案中自动终裁且与真值不符 / 同案中
  自动终裁数。

**口径互斥（勿漂移）**：SHOULD_ABSTAIN 被自动终裁**不进**
``wrong_auto_decision_rate``（其真值是 HUMAN_REVIEW，无从谈自动判对/判错，该危险由
``abstention_recall`` 分子缺口承接）—— 一个 (case, record) 对只落一个指标桶。

分母为 0 的比率返回 None（报告显示 "-"，不硬造 0/∞）。v1 老数据全按
AUTO_DECIDABLE：``abstention_recall`` 分母 0 → None，另两项退化为 v1 口径的
自动终裁错误率 / 过度转人工率。
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, Field

from pra.evaluation.harness.base import EvalRecord

__all__ = ["AbstentionEvaluator", "AbstentionMetrics", "abstain_subset_of"]

_AUTO = "AUTO_DECIDABLE"
_SHOULD = "SHOULD_ABSTAIN"
_DECISION_SET = {"PASS", "REJECT", "HUMAN_REVIEW"}


class AbstentionMetrics(BaseModel):
    """Abstention 五指标 + 各子集计数（None = 分母为 0，未定义）。

    计数与指标桶一一对应，报告/单测可直接核对分子分母。
    """

    total: int = Field(description="全部案数")
    human_pred_total: int = Field(description="pred HUMAN 总数（人工占用）")
    auto_pred_total: int = Field(description="pred ∈ {PASS,REJECT} 总数（自动终裁）")

    auto_decidable_total: int = Field(description="AUTO_DECIDABLE 案数")
    auto_decidable_human: int = Field(description="AUTO_DECIDABLE ∧ pred HUMAN（过度保守）")
    auto_decidable_auto: int = Field(description="AUTO_DECIDABLE ∧ 自动终裁")
    auto_decidable_auto_wrong: int = Field(description="AUTO_DECIDABLE ∧ 自动终裁 ∧ 与真值不符")

    should_abstain_total: int = Field(description="SHOULD_ABSTAIN 案数")
    should_abstain_human: int = Field(description="SHOULD_ABSTAIN ∧ pred HUMAN（正确转人工）")
    should_abstain_auto: int = Field(description="SHOULD_ABSTAIN ∧ 自动终裁（危险误自动）")

    # 五指标（命名权威；None = 分母 0）
    human_review_rate: float | None = None
    automation_coverage: float | None = None
    abstention_rate: float | None = None  # 过度保守：AUTO..-HUMAN / AUTO..
    abstention_recall: float | None = None  # 正确转人工召回：SHOULD..-HUMAN / SHOULD..
    wrong_auto_decision_rate: float | None = None  # AUTO.. 自动终裁中错误占比

    def metric_row(self) -> dict:
        """报告用一行摘要（None → "-"；键 = 五指标全名 + 计数）。"""
        fmt = lambda v: "-" if v is None else f"{v:.3f}"
        return {
            "total": self.total,
            "human_review_rate": fmt(self.human_review_rate),
            "automation_coverage": fmt(self.automation_coverage),
            "abstention_rate": fmt(self.abstention_rate),
            "abstention_recall": fmt(self.abstention_recall),
            "wrong_auto_decision_rate": fmt(self.wrong_auto_decision_rate),
            "n_auto_decidable": self.auto_decidable_total,
            "n_should_abstain": self.should_abstain_total,
        }


def _ratio(numer: int, denom: int) -> float | None:
    return numer / denom if denom else None


def abstain_subset_of(expected: Mapping) -> str:
    """由 expected 的 (decision, abstain_label) 推导 abstention 真值子集。

    - ``abstain_label`` 存在（AUTO_DECIDABLE / SHOULD_ABSTAIN）→ 以它为准；
    - 缺失但 decision == HUMAN_REVIEW → SHOULD_ABSTAIN（Phase 2 真值语义推断）；
    - 其余（含 Phase 1 老数据：无 abstain_label 且 decision ∈ {PASS,REJECT}）→
      AUTO_DECIDABLE（兼容形态，等价 Phase 1）。
    """
    label = expected.get("abstain_label")
    if label == _SHOULD:
        return _SHOULD
    if label == _AUTO:
        return _AUTO
    if expected.get("decision") == "HUMAN_REVIEW":
        return _SHOULD
    return _AUTO


class AbstentionEvaluator:
    """AbstentionEvaluator —— 只吃 (EvalRecord.decision × expected truth subset)。

    expected 字典由调用方从数据集构造（``{eval_case_id: {"decision": …,
    "abstain_label": …}}``；abstain_label 可缺省，Phase 1 兼容）。与
    ``DecisionEvaluator`` 同风格：records 里查不到 expected 的防御性跳过。
    """

    @staticmethod
    def evaluate(records: list[EvalRecord], expected: Mapping[str, Mapping]) -> AbstentionMetrics:
        total = auto_n = should_n = 0
        human_total = auto_pred_total = 0
        auto_dec_human = auto_dec_auto = auto_dec_wrong = 0
        should_human = should_auto = 0

        for rec in records:
            exp = expected.get(rec.eval_case_id)
            if exp is None:
                continue  # 防御：expected 缺失的 record 不计
            pred = rec.decision
            if pred not in _DECISION_SET:
                continue  # 防御：record.decision 越界（schema 已约束，双保险）
            subset = abstain_subset_of(exp)

            total += 1
            if pred == "HUMAN_REVIEW":
                human_total += 1
            else:
                auto_pred_total += 1

            if subset == _SHOULD:
                should_n += 1
                if pred == "HUMAN_REVIEW":
                    should_human += 1
                else:
                    should_auto += 1
            else:  # AUTO_DECIDABLE
                auto_n += 1
                if pred == "HUMAN_REVIEW":
                    auto_dec_human += 1
                else:
                    auto_dec_auto += 1
                    if pred != exp.get("decision"):
                        auto_dec_wrong += 1

        return AbstentionMetrics(
            total=total,
            human_pred_total=human_total,
            auto_pred_total=auto_pred_total,
            auto_decidable_total=auto_n,
            auto_decidable_human=auto_dec_human,
            auto_decidable_auto=auto_dec_auto,
            auto_decidable_auto_wrong=auto_dec_wrong,
            should_abstain_total=should_n,
            should_abstain_human=should_human,
            should_abstain_auto=should_auto,
            human_review_rate=_ratio(human_total, total),
            automation_coverage=_ratio(auto_pred_total, total),
            abstention_rate=_ratio(auto_dec_human, auto_n),
            abstention_recall=_ratio(should_human, should_n),
            wrong_auto_decision_rate=_ratio(auto_dec_wrong, auto_dec_auto),
        )

    @staticmethod
    def evaluate_grouped(
        records: list[EvalRecord], expected: Mapping[str, Mapping]
    ) -> dict[str, AbstentionMetrics]:
        """按 scene 分层复用同口径（scene 取自 expected 索引，缺失归 _unknown）。"""
        by_scene: dict[str, list[EvalRecord]] = {}
        for rec in records:
            scene = expected.get(rec.eval_case_id, {}).get("scene")
            by_scene.setdefault(scene if isinstance(scene, str) else "_unknown", []).append(rec)
        grouped: dict[str, AbstentionMetrics] = {}
        for scene, group in sorted(by_scene.items()):
            grouped[scene] = AbstentionEvaluator.evaluate(group, expected)
        return grouped
