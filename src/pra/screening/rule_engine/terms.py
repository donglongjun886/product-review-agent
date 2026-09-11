"""Screening 规则词表 —— **单一来源**（供 screening 规则层与 agent guardrails.hard_rules 共同引用）。

只声明词表/常量，**不含匹配逻辑**（逻辑在 ``rules.py`` / ``hard_rules.py``）。两层共同引用
此处词表，杜绝手抄漂移；v1 是最小集合，真实品牌黑名单经规则层注入（monkeypatch/构建期替换
模块常量）或二期策略库接入，不在本模块内置。
"""

from __future__ import annotations

# 品牌黑名单（v1 默认**空集**）。测试注入黑名单时 monkeypatch 本常量
# （如 ``monkeypatch.setattr(terms, "BLACKLISTED_BRANDS", frozenset({...}))``），不要改写
# hard_rules 侧的引用（那会破坏两层的单一来源同步）。
BLACKLISTED_BRANDS: frozenset[str] = frozenset()

# 明确品牌词/商标词：标题/描述命中即 R-102 依据（大小写不敏感）。R-102 为 COMPLEX —— 品牌词
# 命中交 Agent 上下文调查后终裁，不再直接 REJECT，防官方店/适配词/授权产品被误杀；黑名单
# 品牌走 R-101 才是确定性 REJECT。
BRAND_TERMS: frozenset[str] = frozenset(
    {"NIKE", "ADIDAS", "GUCCI", "LOUIS VUITTON", "LV"}
)

# 规避模糊词：命中判 COMPLEX 信号（模糊仿冒暗示 → 规则无法确定性直判 → Agent 调查）。
EVASION_TERMS: frozenset[str] = frozenset(
    {"同款", "复刻", "高仿", "1:1", "原单"}
)

# 品牌空缺高危类目前缀（R-301）：类目命中时在命中 detail 中标注高危前缀 —— 前缀只增强
# detail 的人读信息，不作为 COMPLEX 的触发门槛。顺序无匹配优先级（任意前缀命中即高危）。
HIGH_RISK_PREFIXES: tuple[str, ...] = ("女鞋", "男鞋", "运动鞋", "箱包", "鞋")

__all__ = [
    "BLACKLISTED_BRANDS",
    "BRAND_TERMS",
    "EVASION_TERMS",
    "HIGH_RISK_PREFIXES",
]
