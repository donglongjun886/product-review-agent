"""Screening 规则词表 —— **单一来源**（供 screening 规则层与 agent guardrails.hard_rules 共同引用）。

只声明词表/常量，**不含匹配逻辑**（逻辑在 ``rules.py`` / ``hard_rules.py``）。全部为项目内置的
**固定数据**：不接外部数据源 / 配置中心 / 动态加载，也没有运行时替换机制 —— 改词表即改本文件。
两层共同引用此处常量，杜绝手抄漂移。
"""

from __future__ import annotations

# 品牌黑名单：内置固定 demo/test 数据（**不是真实业务黑名单**）。
# 不变量：必须与 ``BRAND_TERMS`` 不相交 —— 命中 BRAND_TERMS 的词走 R-102（COMPLEX，交 Agent
# 上下文调查，防官方店/适配词被误杀）；同一词若同时进黑名单，R-101 会抢先自动 REJECT。
BLACKLISTED_BRANDS: frozenset[str] = frozenset({"某违禁品牌", "山寨"})

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
