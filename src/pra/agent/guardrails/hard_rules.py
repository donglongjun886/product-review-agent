"""R1 硬规则（blacklist / 硬违禁）：命中强制 REJECT，不可被 LLM 覆盖。

三条确定性扫描，**不用 LLM**：

1. ``case.product.brand`` 命中品牌黑名单；
2. 调查新证据 ``IMAGE_LOGO`` 识别到的品牌在黑名单内；
3. 机审信号带硬违禁语义（``name`` 前缀 ``HARD_`` 且 ``result != PASS``，防御性保留）。

默认黑名单为空集，故走查场景恒不命中 —— 本模块保证 overlay 的 R1 分支存在且可测
（单测注入黑名单即可覆盖）。词表单一来源：``pra.screening.rule_engine.terms`` 的
``BLACKLISTED_BRANDS``，与 Screening 规则层共享，不要在本文件另建词表。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pra.domain.models import RiskType
from pra.screening.rule_engine.terms import BLACKLISTED_BRANDS

# 品牌黑名单：引用 terms 侧的 BLACKLISTED_BRANDS，与 Screening 规则层共享同一份，
# 不要在本文件另建词表。单测/策略库接入时替换 terms 侧常量（或 monkeypatch 本名）。

# 机审信号中带"硬违禁"语义的 name 前缀（防御性兜底）。
_HARD_SIGNAL_PREFIX = "HARD_"


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

    # ① 商品 brand 黑名单
    brand = getattr(getattr(case, "product", None), "brand", None)
    if brand and brand in BLACKLISTED_BRANDS:
        return HardRuleHit(risk_types=[RiskType.POTENTIAL_IP_RISK])

    # ② 调查新证据：IMAGE_LOGO 命中黑名单品牌（黑名单非空才扫）
    if BLACKLISTED_BRANDS:
        for e in state.get("evidence") or []:
            if e.type == "IMAGE_LOGO":
                logo_brand = (e.extra or {}).get("brand") or _extract_logo_brand(e.value)
                if logo_brand in BLACKLISTED_BRANDS:
                    return HardRuleHit(risk_types=[RiskType.POTENTIAL_IP_RISK])

    # ③ 机审硬违禁信号
    for sig in getattr(case, "screening_signals", None) or []:
        if sig.name.startswith(_HARD_SIGNAL_PREFIX) and sig.result != "PASS":
            return HardRuleHit(risk_types=[RiskType.POTENTIAL_IP_RISK])

    return None


def _extract_logo_brand(value: str) -> str:
    """从 IMAGE_LOGO 的 value（``logo=<brand>, conf=0.93``）尽力还原品牌名。"""
    for part in value.split(","):
        part = part.strip()
        if part.startswith("logo="):
            return part[len("logo="):]
    return ""
