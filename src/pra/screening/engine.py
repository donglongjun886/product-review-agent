"""Screening 三分流 Triage 引擎 —— 纯函数层（无 IO，可注入 rules 测试）。

三分流语义（任务书拍板，勿改）：
- **PASS / REJECT = 确定性直接终裁**（规则直判，不再进 Agent；落库走
  ``persist_service.run_screening_direct``，trigger_type="SCREENING_DIRECT"）；
- **COMPLEX = 进 Agent 调查**（落库走 ``persist_service.run_and_persist``，Agent
  trigger_type=INITIAL/RE_REVIEW 是另一种执行方式）。
- 判定原则：**能确定才直判，不能确定一律 COMPLEX** —— 只允许确定性规则直判。

规则命中顺序：任一 REJECT 命中即终裁 REJECT（REJECT 优先于 COMPLEX）；无 REJECT 但
任一 COMPLEX 命中 → COMPLEX；都无命中 → PASS —— PASS 只在**能确定性放行**时给出
（brand/类目明确；空缺场景由 R-301 兜成 COMPLEX，不会零命中直放）。**规则集为空时抛
ValueError**：无规则 = "什么都不能确定"，绝不静默全量 PASS（配置加载失败/策略库为空的
最危险失败模式）。

同轮多命中**都收集**（hits 按传入规则序）。

证据：``rule_evidence`` 只**构造** Evidence 对象（type=RULE_HIT /
source=ScreeningRuleEngine / weight=1.0 / extra={"rule_id"}），落库由 persist 层做
（review_evidence.run_id NOT NULL —— 直判 run 也建行，保证 result → run → evidence
审计链统一成立）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pra.domain.models import Evidence, ProductReviewCase
from pra.screening.rule_engine.rules import DEFAULT_RULES, Rule

Verdict = Literal["PASS", "REJECT", "COMPLEX"]

# RULE_HIT 证据常量（screening 规则命中的统一证据来源；source_tool 落库同名）。
RULE_EVIDENCE_TYPE = "RULE_HIT"
RULE_EVIDENCE_SOURCE = "ScreeningRuleEngine"
RULE_EVIDENCE_WEIGHT = 1.0  # 确定性规则命中 → 满权重


@dataclass(frozen=True)
class RuleHit:
    """单条规则命中记录（审计/评测：rule_id 稳定；detail 人读摘要进证据 value）。"""

    rule_id: str
    name: str
    detail: str


@dataclass(frozen=True)
class TriageResult:
    """一次三分流结果：verdict（PASS/REJECT/COMPLEX）+ 全部命中（按规则序）。"""

    verdict: Verdict
    hits: list[RuleHit] = field(default_factory=list)


def triage(
    case: ProductReviewCase,
    *,
    rules: list[Rule] | tuple[Rule, ...] | None = None,
) -> TriageResult:
    """对 case 快照做三分流 —— **纯函数**（只读 case，无 IO，可注入 rules 测）。

    :param case: 审核案件（只读 ``product.brand/title/description/category`` 等）。
    :param rules: 规则集；None → ``rules.DEFAULT_RULES``（R-101/102/301/302）。
        **空规则集（[]/()）抛 ValueError** —— 无规则 = 什么都无法确定，绝不静默
        全量 PASS（配置加载失败/策略库为空的显式失败）。
    :return: TriageResult{verdict, hits}。verdict 收敛：REJECT（任一 REJECT 命中）>
        COMPLEX（无 REJECT 但任一命中）> PASS（零命中；brand/类目空缺已由 R-301
        兜成 COMPLEX，故零命中 PASS 即确定性放行）；hits 按规则序全收集。
    """
    active: tuple[Rule, ...] = tuple(rules) if rules is not None else DEFAULT_RULES
    if not active:
        raise ValueError(
            "规则集为空：三分流无法确定任何裁决（什么都不能确定），拒绝静默全量放行"
            "—— 请检查策略库/规则配置加载"
        )

    hits: list[RuleHit] = []
    has_reject = False
    for rule in active:
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


def rule_evidence(case: ProductReviewCase, hit: RuleHit) -> Evidence:
    """把一条规则命中构造成 Evidence 对象（只构造，不落库）。

    :param case: 触发命中的案件（预留上下文；当前 evidence 不引用 case 字段）。
    :param hit: 命中的 RuleHit。
    :return: Evidence(type="RULE_HIT", source="ScreeningRuleEngine",
        value=f"{rule_id} {name}: {detail}", weight=1.0, ref_id=None,
        extra={"rule_id": rule_id}) —— 确定性直判证据，供落库与决策快照引用。
    """
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
