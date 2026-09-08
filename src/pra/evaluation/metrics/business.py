"""决策业务指标（metrics/business.py）—— DecisionEvaluator：三方案可比口径。

在 **expected.decision ∈ {PASS, REJECT}** 的案上计算（Phase 1 全量真值都是二值）：
REJECT 为**正类**。EvalRecord 的 HUMAN_REVIEW 输出 Phase 1 计为"未自动判"
（不计入二分类混淆，见下口径说明），单独以 human_rate / automation 观察。

口径（代码注释即文档；report.py 会复述）：
- 二分类混淆（只统计**自动判出**的案，即 pred ∈ {PASS, REJECT}）：
  TP=truth REJECT ∧ pred REJECT；FP=truth PASS ∧ pred REJECT；
  TN=truth PASS ∧ pred PASS；FN=truth REJECT ∧ pred PASS；
- Precision = TP/(TP+FP)（自动拒绝中真违规占比）；
- Recall（违规召回）= TP/(TP+FN)；
- FPR = FP/(FP+TN) —— **误杀红线**（真 PASS 被自动 REJECT 的比例）；
- FNR = FN/(TP+FN) —— 漏放（真 REJECT 被自动 PASS）；
- Accuracy = (TP+TN) / 全部真值案（PASS+REJECT）—— **口径选择：预测 HUMAN_REVIEW
  视为"未命中业务真值"（判错），故不入 (TP+TN) 分子但计入分母** —— 对保守转人工的
  方案 Accuracy 更严格；human_rate / automation 另行报出解释差异（避免"全转人工
  刷高 Precision"的假象；与 docs/02-evaluation.md §4.1 Decision Accuracy 一致：
  期望 ∈ {PASS,REJECT} 上逐案精确匹配）。
- human_rate = pred HUMAN / 全部（HRR_total）；automation = 1 - human_rate；
- reject_unhandled = (truth REJECT ∧ pred HUMAN) / truth REJECT 数（漏放的上限视角
  —— 本该自动判却转人工）；pass_unhandled 同理。

分母为 0 的比率返回 None（报告显示 "-"，不硬造 0/∞）。按 scene 分层由
``evaluate_grouped`` 复用同一实现（scene 取自 EvalRecord 关联的 case，由调用方
以 expected_by_case 传入 scene）。
"""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, Field

from pra.evaluation.harness.base import EvalRecord

__all__ = ["DecisionEvaluator", "DecisionMetrics"]

_BINARY = {"PASS", "REJECT"}


class DecisionMetrics(BaseModel):
    """一组二分类业务指标 + 覆盖率口径（None = 分母为 0，未定义）。"""

    total: int = Field(description="真值案总数（expected ∈ {PASS, REJECT}）")
    auto_decided: int = Field(description="自动判出案数（pred ∈ {PASS, REJECT}）")
    human_pred: int = Field(description="输出 HUMAN_REVIEW 的案数（未自动判）")
    human_reject: int = Field(default=0, description="truth REJECT ∧ pred HUMAN（该转拒却转人工）")
    human_pass: int = Field(default=0, description="truth PASS ∧ pred HUMAN（该放行却转人工）")
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    accuracy: float | None = None  # (TP+TN)/total（HUMAN 判错口径，见模块 docstring）
    precision: float | None = None  # TP/(TP+FP)
    recall: float | None = None  # TP/(TP+FN) —— 违规召回（自动判子集）
    fpr: float | None = None  # FP/(FP+TN) —— 误杀红线
    fnr: float | None = None  # FN/(TP+FN) —— 漏放（自动判子集）
    human_rate: float | None = None  # human_pred / total
    automation: float | None = None  # auto_decided / total
    reject_unhandled: float | None = None  # (truth REJECT ∧ pred HUMAN)/truth REJECT
    pass_unhandled: float | None = None  # (truth PASS ∧ pred HUMAN)/truth PASS

    def metric_row(self) -> dict:
        """报告用一行摘要（None → "-"）。"""
        fmt = lambda v: "-" if v is None else f"{v:.3f}"
        return {
            "total": self.total,
            "auto": self.auto_decided,
            "human": self.human_pred,
            "acc": fmt(self.accuracy),
            "prec": fmt(self.precision),
            "recall": fmt(self.recall),
            "fpr": fmt(self.fpr),
            "fnr": fmt(self.fnr),
            "hrr": fmt(self.human_rate),
            "auto_rate": fmt(self.automation),
        }


def _ratio(numer: int, denom: int) -> float | None:
    return numer / denom if denom else None


class DecisionEvaluator:
    """DecisionEvaluator —— 只吃 EvalRecord.decision × expected.decision。

    Phase 1 不含 AbstentionEvaluator（expected=HUMAN 真值）；expected 字典由调用方
    从数据集构造（``{eval_case_id: {"decision": ..., "scene": ...}}``）。
    """

    @staticmethod
    def evaluate(records: list[EvalRecord], expected: Mapping[str, Mapping]) -> DecisionMetrics:
        truth = {cid: exp.get("decision") for cid, exp in expected.items()}
        total = tp = fp = tn = fn = human = 0
        reject_truth = pass_truth = 0
        reject_human = pass_human = 0
        for rec in records:
            exp_decision = truth.get(rec.eval_case_id)
            if exp_decision not in _BINARY:
                continue  # 防御：非二值真值不计（Phase 1 全量真值均二值）
            total += 1
            if exp_decision == "REJECT":
                reject_truth += 1
            else:
                pass_truth += 1
            pred = rec.decision
            if pred not in _BINARY:  # HUMAN_REVIEW → 未自动判
                human += 1
                if exp_decision == "REJECT":
                    reject_human += 1
                else:
                    pass_human += 1
                continue
            if exp_decision == "REJECT" and pred == "REJECT":
                tp += 1
            elif exp_decision == "PASS" and pred == "REJECT":
                fp += 1
            elif exp_decision == "PASS" and pred == "PASS":
                tn += 1
            else:  # exp REJECT ∧ pred PASS
                fn += 1

        auto_decided = tp + fp + tn + fn
        return DecisionMetrics(
            total=total,
            auto_decided=auto_decided,
            human_pred=human,
            human_reject=reject_human,
            human_pass=pass_human,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            accuracy=_ratio(tp + tn, total),
            precision=_ratio(tp, tp + fp),
            recall=_ratio(tp, tp + fn),
            fpr=_ratio(fp, fp + tn),
            fnr=_ratio(fn, tp + fn),
            human_rate=_ratio(human, total),
            automation=_ratio(auto_decided, total),
            reject_unhandled=_ratio(reject_human, reject_truth),
            pass_unhandled=_ratio(pass_human, pass_truth),
        )

    @staticmethod
    def evaluate_grouped(
        records: list[EvalRecord], expected: Mapping[str, Mapping]
    ) -> dict[str, DecisionMetrics]:
        """按 scene 分层（normal/violation/boundary/multi-signal/evasion）复用同口径。"""
        by_scene: dict[str, list[EvalRecord]] = {}
        for rec in records:
            scene = expected.get(rec.eval_case_id, {}).get("scene")
            by_scene.setdefault(scene if isinstance(scene, str) else "_unknown", []).append(rec)
        grouped: dict[str, DecisionMetrics] = {}
        for scene, group in sorted(by_scene.items()):
            grouped[scene] = DecisionEvaluator.evaluate(group, expected)
        return grouped
