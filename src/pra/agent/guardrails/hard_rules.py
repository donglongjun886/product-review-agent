"""R1 硬规则（blacklist / 硬违禁）：命中强制 REJECT，不可被 LLM 覆盖。

一条确定性扫描，**不用 LLM**：``case.product.brand`` 命中品牌黑名单。

品牌黑名单取 ``pra.screening.rule_engine.terms.BLACKLISTED_BRANDS``（内置固定 demo 词表），与
Screening 规则层共享同一份；本文件不另建词表，也没有运行时替换机制。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pra.domain.models import RiskType
from pra.screening.rule_engine import terms

# 品牌黑名单：经 ``terms`` 模块属性读取，与 Screening 规则层同一份（本文件不另建词表）。


@dataclass(frozen=True)
class HardRuleHit:
    """R1 命中信息：强制 REJECT 时的风险类型（写进 ``ReviewDecision.risk_type``）。"""

    risk_types: list[RiskType] = field(default_factory=list)


def hard_rule_hit(state: dict[str, Any]) -> HardRuleHit | None:
    """R1 扫描；命中返回 ``HardRuleHit``（overlay 强制 REJECT），否则 None。

    即便 LLM 提案 PASS，也以 REJECT 为准（防漏放）。
    """
    case = state.get("case")
    if case is None:
        return None

    # 商品 brand 黑名单
    brand = getattr(getattr(case, "product", None), "brand", None)
    if brand and brand in terms.BLACKLISTED_BRANDS:
        return HardRuleHit(risk_types=[RiskType.POTENTIAL_IP_RISK])

    return None
