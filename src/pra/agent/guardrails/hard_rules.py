"""R1 硬规则：``case.product.brand`` 命中品牌黑名单即强制 REJECT。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pra.domain.models import RiskType
from pra.screening.rule_engine import terms


@dataclass(frozen=True)
class HardRuleHit:
    """R1 命中信息：风险类型列表。"""

    risk_types: list[RiskType] = field(default_factory=list)


def hard_rule_hit(state: dict[str, Any]) -> HardRuleHit | None:
    """R1 扫描：命中返回 ``HardRuleHit``，否则 None。"""
    case = state.get("case")
    if case is None:
        return None

    brand = getattr(getattr(case, "product", None), "brand", None)
    if brand and brand in terms.BLACKLISTED_BRANDS:
        return HardRuleHit(risk_types=[RiskType.POTENTIAL_IP_RISK])

    return None
