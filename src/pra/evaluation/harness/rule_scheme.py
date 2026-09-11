"""Baseline 1：复用 ``pra.screening`` 的确定性三分流。

对 ``case.input`` 跑 ``pra.screening.engine.triage``（DEFAULT_RULES）后映射：
PASS → PASS；REJECT → REJECT（R-101 黑名单直判）；**COMPLEX → HUMAN_REVIEW**
（评测语义 = 不可自动判：线上 COMPLEX 进 Agent，而本 baseline 没有 Agent，
等价只能转人工）。

本模块是薄封装，不含任何规则逻辑（规则在 screening 层，单一来源）；不落库、
不 import persist_service 的 DB 路径。``detail.hits`` 转录每条 RuleHit
（rule_id / name / detail），供审计与"哪条规则导致 COMPLEX/REJECT"回溯。

成本恒零（不耗 LLM/工具）；``decision_confidence`` 恒 None —— 确定性规则是
命中即满权重终裁，没有"自动判安全把握"的置信概念。

v1 词表现状（报告口径，非缺陷）：``terms.BLACKLISTED_BRANDS`` 为空 → R-101 不命中，
而 R-102（品牌词）/ R-301（brand/类目空缺）/ R-302（规避词）均为 COMPLEX →
v1 Rule baseline 只输出 PASS 或 HUMAN_REVIEW，REJECT 直判率恒 0。
"""

from __future__ import annotations

from pra.evaluation.dataset.schema import EvalCase
from pra.evaluation.harness.base import EvalContext, EvalRecord, SchemeRunner
from pra.screening.engine import Verdict, triage

__all__ = ["VERDICT_TO_DECISION", "RuleBaseline"]

# 三分流 verdict → EvalRecord 三分类（COMPLEX 的评测语义 = 转人工）
VERDICT_TO_DECISION: dict[Verdict, str] = {
    "PASS": "PASS",
    "REJECT": "REJECT",
    "COMPLEX": "HUMAN_REVIEW",
}


class RuleBaseline(SchemeRunner):
    """Baseline 1 —— 确定性规则三分流（复用 pra.screening，薄封装、零成本）。"""

    name = "rule"

    async def run(self, case: EvalCase, ctx: EvalContext) -> EvalRecord:
        result = triage(case.input)  # DEFAULT_RULES（R-101/102/301/302）；空规则集会抛 ValueError
        hits = [
            {
                "rule_id": hit.rule_id,
                "name": hit.name,
                "detail": hit.detail,
            }
            for hit in result.hits
        ]
        evidence = [
            {
                "type": "RULE_HIT",
                "value": f"{hit.rule_id} {hit.name}: {hit.detail}",
                "extra": {"rule_id": hit.rule_id},
            }
            for hit in result.hits
        ]
        decision = VERDICT_TO_DECISION[result.verdict]
        return EvalRecord(
            eval_case_id=case.eval_case_id,
            scheme=self.name,
            decision=decision,
            risk_level="NONE" if decision == "PASS" else None,
            risk_type=[],
            decision_confidence=None,  # 确定性规则直判无置信概念
            evidence=evidence,
            policy=[],  # 规则层不产出政策引用（R-101 黑名单属词表命中，非条款引用）
            tool_calls_actual=[],
            cost={"llm_calls": 0, "tool_calls": 0, "tokens": 0},
            detail={"verdict": result.verdict, "hits": hits},
        )
