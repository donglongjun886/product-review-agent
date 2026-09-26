"""Screening 规则定义 —— 声明式规则集。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Literal

from pra.domain.models import ProductReviewCase
from pra.screening.rule_engine import terms

# kind 词汇：REJECT / COMPLEX。
RuleKind = Literal["REJECT", "COMPLEX"]


@dataclass(frozen=True)
class Rule:
    """一条声明式筛选规则。

    ``match`` 为纯函数：命中返回人读 detail，未命中返回 ``None``。
    """

    rule_id: str
    name: str
    kind: RuleKind
    match: Callable[[ProductReviewCase], str | None]


# ---- 匹配辅助 ----


def _fold(text: str) -> str:
    """匹配归一化：转小写 + 全角冒号→半角。"""
    return (text or "").lower().replace("：", ":")


def _title_desc_text(case: ProductReviewCase) -> str:
    """R-102/R-302 的扫描文本：标题 + 描述（折叠后）。"""
    return _fold(f"{case.product.title}\n{case.product.description}")


def _brand_missing(brand: str | None) -> bool:
    """品牌空缺判定：``None`` 与纯空白串都算空缺。"""
    return brand is None or (isinstance(brand, str) and not brand.strip())


_ASCII_LATIN = re.compile(r"[a-z ]+\Z")  # 折叠后的纯拉丁字母串（可含空格）


def _term_hits(hay: str, term_list: frozenset[str]) -> list[str]:
    """在折叠文本 ``hay`` 中探测词表命中，返回命中的原词（排序后）。"""
    matched: list[str] = []
    for term in term_list:
        folded = _fold(term)
        if _ASCII_LATIN.fullmatch(folded):
            pattern = re.compile(rf"(?<![a-z0-9_]){re.escape(folded)}(?![a-z0-9_])")
        else:
            pattern = re.compile(re.escape(folded))
        if pattern.search(hay):
            matched.append(term)
    return sorted(matched)


# ---- R-1xx：品牌命中规则 ----


def _match_r101(case: ProductReviewCase) -> str | None:
    """R-101 REJECT：brand 明确且命中品牌黑名单。"""
    brand = case.product.brand
    if _brand_missing(brand):
        return None
    if brand in terms.BLACKLISTED_BRANDS:
        return f"brand={brand!r} 命中品牌黑名单 BLACKLISTED_BRANDS"
    return None


def _match_r102(case: ProductReviewCase) -> str | None:
    """R-102 COMPLEX：标题/描述含品牌词（``BRAND_TERMS``）。"""
    matched = _term_hits(_title_desc_text(case), terms.BRAND_TERMS)
    if not matched:
        return None
    return f"标题/描述命中品牌词: {', '.join(matched)}"


# ---- R-3xx：品牌/类目空缺与规避词 ----


def _match_r301(case: ProductReviewCase) -> str | None:
    """R-301 COMPLEX：brand 或 category 空缺。"""
    brand = case.product.brand
    category = case.product.category or ""
    brand_blank = _brand_missing(brand)
    category_blank = not category.strip()
    if not brand_blank and not category_blank:
        return None
    brand_disp = "None" if brand is None else repr(brand)
    if brand_blank and not category_blank:
        for prefix in terms.HIGH_RISK_PREFIXES:
            if category.startswith(prefix):
                return (
                    f"brand 空缺({brand_disp}) 且类目 '{category}' 命中高危类目前缀 "
                    f"'{prefix}'"
                )
        return (
            f"brand 空缺({brand_disp}) 且类目 '{category}' 未命中高危前缀 —— "
            f"品牌空缺无法确定是否规避，转 Agent 调查"
        )
    if not brand_blank:  # 类目空缺（brand 明确）
        return f"brand={brand!r} 但类目空缺 —— 无法确定审查域，转 Agent 调查"
    return f"brand 空缺({brand_disp}) 且类目空缺 —— 关键事实缺失，转 Agent 调查"


def _match_r302(case: ProductReviewCase) -> str | None:
    """R-302 COMPLEX：标题/描述含规避模糊词（``EVASION_TERMS``）。"""
    matched = _term_hits(_title_desc_text(case), terms.EVASION_TERMS)
    if not matched:
        return None
    return f"标题/描述命中规避模糊词: {', '.join(matched)}"


# ---- 默认规则集 ----

DEFAULT_RULES: tuple[Rule, ...] = (
    Rule("R-101", "品牌黑名单命中", "REJECT", _match_r101),
    Rule("R-102", "品牌词命中", "COMPLEX", _match_r102),
    Rule("R-301", "品牌/类目空缺", "COMPLEX", _match_r301),
    Rule("R-302", "规避模糊词命中", "COMPLEX", _match_r302),
)

__all__ = ["DEFAULT_RULES", "Rule", "RuleKind"]
