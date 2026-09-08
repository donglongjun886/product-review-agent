"""Screening 规则定义 —— 声明式规则集（确定性直判；单一来源词表见 ``terms.py``）。

规则形态：每条 ``Rule{rule_id, name, kind, match}``：
- ``rule_id``：稳定规则 ID（审计/评测/证据引用），**R-1xx = REJECT、R-3xx = COMPLEX**；
- ``kind``：命中后的裁决方向（REJECT 直接终裁 REJECT；COMPLEX 才进 Agent 调查）；
- ``match(case) -> str | None``：纯函数（无 IO）；命中返回**人读 detail 描述**（进
  RULE_HIT 证据 value），未命中返回 None。

判定原则（任务书拍板，勿改）：**能确定才直判，不能确定一律 COMPLEX** —— 不搞软规则
打分猜测；只允许确定性规则直判（REJECT / PASS），模糊暗示只导向 COMPLEX（进 Agent）。

v1 默认规则集（三分演示可调）：
- R-101 REJECT：``case.product.brand ∈ BLACKLISTED_BRANDS``（词表来自 terms，v1 空集）；
- R-102 REJECT：title/description 含 BRAND_TERMS 词（大小写不敏感）；
- R-301 COMPLEX：brand=None 且类目命中 HIGH_RISK_PREFIXES（品牌空缺 + 高危类目 =
  规则无法确定是否规避 → Agent）；
- R-302 COMPLEX：title/description 含 EVASION_TERMS（模糊仿冒暗示，不直判）；
- 无任何命中 → PASS（确定性放行：brand 明确且干净）。

命中语义：同轮多命中**都收集**（hits 按本模块规则声明序）；最终裁决由 engine 按
「任一 REJECT 命中即 REJECT；否则任一 COMPLEX 命中即 COMPLEX；否则 PASS」收敛。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from pra.domain.models import ProductReviewCase
from pra.screening.rule_engine import terms

# kind 词汇：REJECT（确定性终裁 REJECT）/ COMPLEX（进 Agent 调查）。
RuleKind = Literal["REJECT", "COMPLEX"]


@dataclass(frozen=True)
class Rule:
    """一条声明式筛选规则。

    ``match`` 为纯函数（只读 case 快照，无 IO/无副作用）；命中返回人读 detail
    （进 RULE_HIT 证据 value），未命中返回 None。规则引用 terms 词表常量走
    **模块属性动态读取**（``terms.BLACKLISTED_BRANDS`` 而非 import 绑定），使词表
    注入（monkeypatch/二期策略库替换模块常量）对已构造规则即刻生效。
    """

    rule_id: str
    name: str
    kind: RuleKind
    match: Callable[[ProductReviewCase], str | None]


# ---------------------------------------------------------------------------
# 匹配辅助
# ---------------------------------------------------------------------------


def _fold(text: str) -> str:
    """匹配归一化：小写（BRAND_TERMS 大小写不敏感）+ 全角冒号→半角（兜 1：1 形态）。

    仅做确定性文本折叠，不做分词/词干（词表为精确子串语义，避免猜测）。
    """
    return (text or "").lower().replace("：", ":")


def _title_desc_text(case: ProductReviewCase) -> str:
    """R-102/R-302 的扫描文本：标题 + 描述（折叠后逐词子串命中）。"""
    return _fold(f"{case.product.title}\n{case.product.description}")


# ---------------------------------------------------------------------------
# R-1xx —— REJECT（确定性直接终裁）
# ---------------------------------------------------------------------------


def _match_r101(case: ProductReviewCase) -> str | None:
    """R-101 REJECT：brand 命中品牌黑名单（词表 = terms.BLACKLISTED_BRANDS，v1 空）。"""
    brand = case.product.brand
    if brand and brand in terms.BLACKLISTED_BRANDS:
        return f"brand={brand!r} 命中品牌黑名单 BLACKLISTED_BRANDS"
    return None


def _match_r102(case: ProductReviewCase) -> str | None:
    """R-102 REJECT：标题/描述含明确品牌词/商标词（BRAND_TERMS，大小写不敏感）。"""
    hay = _title_desc_text(case)
    matched = sorted(t for t in terms.BRAND_TERMS if _fold(t) in hay)
    if not matched:
        return None
    return f"标题/描述命中品牌词: {', '.join(matched)}"


# ---------------------------------------------------------------------------
# R-3xx —— COMPLEX（规则无法确定 → 进 Agent 调查，不直判）
# ---------------------------------------------------------------------------


def _match_r301(case: ProductReviewCase) -> str | None:
    """R-301 COMPLEX：brand 空缺（None）+ 类目命中高危前缀 —— 是否规避无法由规则确定。"""
    brand = case.product.brand
    if brand is not None:
        return None
    category = case.product.category or ""
    for prefix in terms.HIGH_RISK_PREFIXES:
        if category.startswith(prefix):
            return f"brand 空缺(None) 且类目 '{category}' 命中高危类目前缀 '{prefix}'"
    return None


def _match_r302(case: ProductReviewCase) -> str | None:
    """R-302 COMPLEX：标题/描述含规避模糊词（EVASION_TERMS）—— 模糊仿冒暗示，不直判。"""
    hay = _title_desc_text(case)
    matched = sorted(t for t in terms.EVASION_TERMS if _fold(t) in hay)
    if not matched:
        return None
    return f"标题/描述命中规避模糊词: {', '.join(matched)}"


# ---------------------------------------------------------------------------
# 默认规则集（规则序 = 声明序；engine 按规则序收集 hits，再按 REJECT>COMPLEX>PASS 定裁决）
# ---------------------------------------------------------------------------

DEFAULT_RULES: tuple[Rule, ...] = (
    Rule("R-101", "品牌黑名单命中", "REJECT", _match_r101),
    Rule("R-102", "品牌词命中", "REJECT", _match_r102),
    Rule("R-301", "品牌空缺+高危类目", "COMPLEX", _match_r301),
    Rule("R-302", "规避模糊词命中", "COMPLEX", _match_r302),
)

__all__ = ["DEFAULT_RULES", "Rule", "RuleKind"]
