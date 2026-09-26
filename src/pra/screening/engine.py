"""Screening 三分流 Triage 引擎 —— 纯函数层（无 IO）。

判定：任一 REJECT 命中即 REJECT，否则任一 COMPLEX 命中 → COMPLEX，都无命中 → PASS；
hits 按 ``DEFAULT_RULES`` 声明序全收集。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pra.domain.models import Evidence, ProductReviewCase
from pra.screening.rule_engine.rules import DEFAULT_RULES

Verdict = Literal["PASS", "REJECT", "COMPLEX"]

# RULE_HIT 证据常量。
RULE_EVIDENCE_TYPE = "RULE_HIT"
RULE_EVIDENCE_SOURCE = "ScreeningRuleEngine"
RULE_EVIDENCE_WEIGHT = 1.0


@dataclass(frozen=True)
class RuleHit:
    """单条规则命中记录（``rule_id`` 稳定、``detail`` 为人读摘要）。"""

    rule_id: str
    name: str
    detail: str


@dataclass(frozen=True)
class TriageResult:
    """一次三分流结果：verdict（PASS/REJECT/COMPLEX）+ 全部命中（按规则序）。"""

    verdict: Verdict
    hits: list[RuleHit] = field(default_factory=list)


def triage(case: ProductReviewCase) -> TriageResult:
    """对 case 快照做三分流（纯函数）。

    :return: TriageResult{verdict, hits}；hits 按 ``DEFAULT_RULES`` 声明序全收集。
    """
    hits: list[RuleHit] = []
    has_reject = False
    for rule in DEFAULT_RULES:
        detail = rule.match(case)
        if detail is None:
            continue
        hits.append(RuleHit(rule_id=rule.rule_id, name=rule.name, detail=detail))
        if rule.kind == "REJECT":
            has_reject = True

    if has_reject:
        verdict: Verdict = "REJECT"
    elif hits:
        verdict = "COMPLEX"
    else:
        verdict = "PASS"
    return TriageResult(verdict=verdict, hits=hits)


def rule_evidence(hit: RuleHit) -> Evidence:
    """把一条规则命中构造成 Evidence（只构造，不落库）。"""
    return Evidence(
        type=RULE_EVIDENCE_TYPE,
        source=RULE_EVIDENCE_SOURCE,
        value=f"{hit.rule_id} {hit.name}: {hit.detail}",
        weight=RULE_EVIDENCE_WEIGHT,
        ref_id=None,
        extra={"rule_id": hit.rule_id},
    )


__all__ = [
    "RULE_EVIDENCE_SOURCE",
    "RULE_EVIDENCE_TYPE",
    "RULE_EVIDENCE_WEIGHT",
    "RuleHit",
    "TriageResult",
    "Verdict",
    "rule_evidence",
    "triage",
]
