"""Screening 规则词表 —— **单一来源**（供 screening 规则层与 agent guardrails.hard_rules 共同引用）。

本模块只声明词表/常量，**不含匹配逻辑**（逻辑在 ``rules.py`` / ``hard_rules.py``）。
设计约束（任务书拍板）：
- Screening 规则引擎与 Agent 侧 R1 硬规则共同引用这里的词表，杜绝两处手抄漂移；
- v1 默认词表为确定性直判的最小集合，**真实品牌黑名单数据**经规则层注入
  （monkeypatch/构建期替换模块常量）或二期策略库接入，不在本模块内置。
"""

from __future__ import annotations

# 品牌黑名单（v1 默认**空集** —— 词表单一来源；真实数据经规则层注入/二期策略库；
# agent/guardrails/hard_rules.py 引用此处，见该文件 docstring）。
# 注意：与 screening 规则 R-101 共享同一份 —— 测试注入黑名单时 monkeypatch 本常量
# （如 ``monkeypatch.setattr(terms, "BLACKLISTED_BRANDS", frozenset({...}))``），
# 不要改写 hard_rules 侧的引用（那会破坏两层的单一来源同步）。
BLACKLISTED_BRANDS: frozenset[str] = frozenset()

# 明确品牌词/商标词：标题/描述命中即 R-102 REJECT 依据候选（大小写不敏感）。
BRAND_TERMS: frozenset[str] = frozenset(
    {"NIKE", "ADIDAS", "GUCCI", "LOUIS VUITTON", "LV"}
)

# 规避模糊词：命中判 COMPLEX 信号（模糊仿冒暗示 → 规则无法确定性直判 → Agent 调查），不直判。
EVASION_TERMS: frozenset[str] = frozenset(
    {"同款", "复刻", "高仿", "1:1", "原单"}
)

# 品牌空缺高危类目前缀（R-301：brand=None + 类目命中此前缀 → COMPLEX）。
# 注意顺序无匹配优先级（任意前缀命中即高危），元组仅为可读确定性。
HIGH_RISK_PREFIXES: tuple[str, ...] = ("女鞋", "男鞋", "运动鞋", "箱包", "鞋")

__all__ = [
    "BLACKLISTED_BRANDS",
    "BRAND_TERMS",
    "EVASION_TERMS",
    "HIGH_RISK_PREFIXES",
]
