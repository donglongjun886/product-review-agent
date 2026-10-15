"""Screening 规则词表（供 screening 规则层与 agent guardrails 引用）。"""

from __future__ import annotations

# 品牌黑名单（内置 demo/test 数据）。
BLACKLISTED_BRANDS: frozenset[str] = frozenset({"某违禁品牌", "山寨"})

# 品牌黑名单命中对应的政策条款 ID（R1 硬规则拒绝的可引用依据）。
BLACKLIST_POLICY_ID = "POLICY_1.1"

# 规避词命中（R-302）对应的政策条款 ID；``EVASION_AGGRAVATED_POLICY_ID`` 为叠加商家脏的从重条款。
EVASION_POLICY_ID = "POLICY_1.2"
EVASION_AGGRAVATED_POLICY_ID = "POLICY_4.5"

# 明确品牌词/商标词。
BRAND_TERMS: frozenset[str] = frozenset(
    {"NIKE", "ADIDAS", "GUCCI", "LOUIS VUITTON", "LV"}
)

# 规避模糊词。
EVASION_TERMS: frozenset[str] = frozenset(
    {"同款", "复刻", "高仿", "1:1", "原单"}
)

HIGH_RISK_PREFIXES: tuple[str, ...] = ("女鞋", "男鞋", "运动鞋", "箱包", "鞋")

__all__ = [
    "BLACKLISTED_BRANDS",
    "BLACKLIST_POLICY_ID",
    "BRAND_TERMS",
    "EVASION_AGGRAVATED_POLICY_ID",
    "EVASION_POLICY_ID",
    "EVASION_TERMS",
    "HIGH_RISK_PREFIXES",
]
