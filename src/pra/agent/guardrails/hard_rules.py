"""R1 硬规则（blacklist / 硬违禁）—— 强制 REJECT，不可被 LLM 覆盖（《00》§7.2-1 /《01》§7.4）。

MVP 最小黑名单（04 §8 MVP 裁剪）：只实现"确定性来源"扫描，**不用 LLM**：
1. ``case.product.brand`` 命中品牌黑名单；
2. 调查新证据 ``IMAGE_LOGO``（识别到的品牌在黑名单内）；
3. 机审信号带"硬违禁"语义（``name`` 前缀 HARD_ 且 ``result != PASS`` —— 正常
   分流层不会把硬违规投进来，但防御性保留）。

默认黑名单为空集（词表单一来源：见 ``pra.screening.rule_engine.terms`` 的
``BLACKLISTED_BRANDS`` —— Screening 规则层与 R1 硬规则共同引用同一份，杜绝双处手抄
漂移；真实词表经规则层注入/二期策略库接入），故走查场景恒不命中 —— 本模块保证 overlay
的 R1 分支存在且可测（单测注入黑名单即可覆盖，注入位点仍是本模块的
``BLACKLISTED_BRANDS`` 引用，见 tests/test_gate.py / test_metrics.py 用法）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pra.domain.models import RiskType
from pra.screening.rule_engine.terms import BLACKLISTED_BRANDS

# 品牌黑名单（v1 空 —— **词表单一来源**：本名引用 pra.screening.rule_engine.terms 的
# BLACKLISTED_BRANDS，与 Screening 规则 R-101 共享；真实数据经规则层注入/二期策略库，
# 不要在本文件另建词表）。单测/策略库接入时替换 terms 侧常量（或 monkeypatch 本名，
# 见模块 docstring）。

# 机审信号中带"硬违禁"语义的 name 前缀（防御性兜底）。
_HARD_SIGNAL_PREFIX = "HARD_"


@dataclass(frozen=True)
class HardRuleHit:
    """R1 命中信息：强制 REJECT 时的风险类型（写进 ReviewDecision.risk_type）。"""

    risk_types: list[RiskType] = field(default_factory=list)


def hard_rule_hit(state: dict[str, Any]) -> HardRuleHit | None:
    """R1 扫描；命中返回 HardRuleHit（overlay 强制 REJECT），否则 None。

    命中即防漏放 —— 即便 LLM 提案 PASS，也以 REJECT 为准。
    """
    case = state.get("case")
    if case is None:
        return None

    # ① 商品 brand 黑名单
    brand = getattr(getattr(case, "product", None), "brand", None)
    if brand and brand in BLACKLISTED_BRANDS:
        return HardRuleHit(risk_types=[RiskType.POTENTIAL_IP_RISK])

    # ② 调查新证据：IMAGE_LOGO 命中黑名单品牌（brand 在 value/extra；黑名单非空才扫）
    if BLACKLISTED_BRANDS:
        for e in state.get("evidence") or []:
            if e.type == "IMAGE_LOGO":
                logo_brand = (e.extra or {}).get("brand") or _extract_logo_brand(e.value)
                if logo_brand in BLACKLISTED_BRANDS:
                    return HardRuleHit(risk_types=[RiskType.POTENTIAL_IP_RISK])

    # ③ 机审硬违禁信号（防御性）
    for sig in getattr(case, "screening_signals", None) or []:
        if sig.name.startswith(_HARD_SIGNAL_PREFIX) and sig.result != "PASS":
            return HardRuleHit(risk_types=[RiskType.POTENTIAL_IP_RISK])

    return None


def _extract_logo_brand(value: str) -> str:
    """从 IMAGE_LOGO 的 value（"logo=<brand>, conf=0.93"）尽力还原品牌名（供黑名单比对）。"""
    for part in value.split(","):
        part = part.strip()
        if part.startswith("logo="):
            return part[len("logo="):]
    return ""
