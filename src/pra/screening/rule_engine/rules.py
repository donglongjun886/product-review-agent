"""Screening 规则定义 —— 声明式规则集（确定性直判；单一来源词表见 ``terms.py``）。

规则形态：每条 ``Rule{rule_id, name, kind, match}``：
- ``rule_id``：稳定规则 ID（审计/评测/证据引用）；**动作一律看 ``kind``** —— v1：
  R-101 黑名单 = REJECT；R-102 品牌词 = COMPLEX（Q-1 拍板 B，见下）；R-301/302 =
  COMPLEX；R-1xx/R-3xx 前缀仅作分组，不绑定动作；
- ``kind``：命中后的裁决方向（REJECT 直接终裁 REJECT；COMPLEX 才进 Agent 调查）；
- ``match(case) -> str | None``：纯函数（无 IO）；命中返回**人读 detail 描述**（进
  RULE_HIT 证据 value），未命中返回 None。

判定原则（任务书拍板，勿改）：**能确定才直判，不能确定一律 COMPLEX** —— 不搞软规则
打分猜测；只允许确定性规则直判（REJECT / PASS），模糊暗示只导向 COMPLEX（进 Agent）。

v1 默认规则集（三分演示可调）：
- R-101 REJECT：``case.product.brand`` 明确（非空缺）且 ∈ BLACKLISTED_BRANDS
  （词表来自 terms，v1 空集）；
- R-102 COMPLEX：title/description 含 BRAND_TERMS 词 —— **Q-1 拍板 B：命中不再
  REJECT 直判**（REJECT 会误杀合法场景：NIKE 官方店 / 适配 NIKE 鞋带 / 授权产品；
  且 brand 与标题一致也不能证明真品，brand=NIKE + "高仿/复刻"仍违规），一律交
  Agent 上下文调查后终裁。大小写不敏感；纯拉丁词按 **词边界** 匹配（防 "LV" 命中
  silver/valve 等英文词内部子串触发无谓 COMPLEX；中文旁不构成 ASCII 词边界，真实
  品牌仍命中）；
- R-301 COMPLEX：brand 或 category **空缺**（None 与空串统一判定）—— 关键事实
  无法确定时规则不能直判，一律进 Agent 调查（类目命中 HIGH_RISK_PREFIXES 时
  detail 标注高危前缀）；
- R-302 COMPLEX：title/description 含 EVASION_TERMS（模糊仿冒暗示，不直判）；
- 无任何命中 → PASS（确定性放行：brand 与 category 均明确（空缺已由 R-301 兜住）
  且文本干净）。

命中语义：同轮多命中**都收集**（hits 按本模块规则声明序）；最终裁决由 engine 按
「任一 REJECT 命中即 REJECT；否则任一 COMPLEX 命中即 COMPLEX；否则 PASS」收敛。
"""

from __future__ import annotations

import re
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
    """匹配归一化：小写（词表大小写不敏感）+ 全角冒号→半角（兜 1：1 形态）。

    仅做确定性文本折叠，不做分词/词干；词表命中语义（拉丁词词边界 / 中文子串）
    见 ``_term_hits``。
    """
    return (text or "").lower().replace("：", ":")


def _title_desc_text(case: ProductReviewCase) -> str:
    """R-102/R-302 的扫描文本：标题 + 描述（折叠后统一探测，命中语义见 _term_hits）。"""
    return _fold(f"{case.product.title}\n{case.product.description}")


def _brand_missing(brand: str | None) -> bool:
    """品牌空缺统一判定：None 与空串/纯空白串都算空缺（R-101/R-301 共用同一口径）。

    原实现口径不一致：R-101 用 truthy 判空（"" 视为无）、R-301 用 ``is not None``
    判空（"" 视为有）；统一后两处对 None / "" 行为一致（都判空缺）。
    """
    return brand is None or (isinstance(brand, str) and not brand.strip())


_ASCII_LATIN = re.compile(r"[a-z ]+\Z")  # 折叠后纯拉丁字母（可含空格 → 多词短语）


def _term_hits(hay: str, term_list: frozenset[str]) -> list[str]:
    """在折叠文本 ``hay`` 中探测词表命中（返回命中的原词，排序后给人读 detail）。

    - **纯拉丁词/短语**（如 LV / LOUIS VUITTON）：按 **词边界** 匹配 —— 两侧要求
      非 ASCII 词字符（``(?<![a-z0-9_]) … (?![a-z0-9_])``）。短词 "LV" 不再命中
      silver/valve/solve 等英文词**内部子串**（误杀直判 REJECT 且无复议路径）；
      中文不属 ASCII 词字符、不构成边界，故 "新款LV手袋" 这类真实品牌仍命中。
    - **含中文/数字/符号的词**（如 复刻 / 1:1）：中文无空格分词，词边界语义失真，
      维持整段子串匹配（v1 行为不变）。
    """
    matched: list[str] = []
    for term in term_list:
        folded = _fold(term)
        if _ASCII_LATIN.fullmatch(folded):  # 纯拉丁（可含空格）→ 词边界
            pattern = re.compile(rf"(?<![a-z0-9_]){re.escape(folded)}(?![a-z0-9_])")
        else:  # 中文/符号词 → 子串（中文无空格分词，边界无意义）
            pattern = re.compile(re.escape(folded))
        if pattern.search(hay):
            matched.append(term)
    return sorted(matched)


# ---------------------------------------------------------------------------
# R-1xx —— 品牌命中规则（动作按 kind：R-101 黑名单 = REJECT 直接终裁；R-102 品牌词 =
# COMPLEX —— Q-1 拍板 B，品牌词不再直接终裁，交 Agent 上下文调查）
# ---------------------------------------------------------------------------


def _match_r101(case: ProductReviewCase) -> str | None:
    """R-101 REJECT：brand 明确（非空缺，None/"" 统一判空见 _brand_missing）且命中
    品牌黑名单（词表 = terms.BLACKLISTED_BRANDS，v1 空）。"""
    brand = case.product.brand
    if _brand_missing(brand):
        return None
    if brand in terms.BLACKLISTED_BRANDS:
        return f"brand={brand!r} 命中品牌黑名单 BLACKLISTED_BRANDS"
    return None


def _match_r102(case: ProductReviewCase) -> str | None:
    """R-102 COMPLEX：标题/描述含明确品牌词/商标词（BRAND_TERMS，大小写不敏感）。

    **Q-1 拍板 B：命中不再 REJECT 直判** —— REJECT 会误杀合法场景（NIKE 官方店 /
    适配 NIKE 鞋带 / 授权产品），且 brand 与标题一致也不能证明真品（brand=NIKE +
    "高仿/复刻"仍违规），故一律 COMPLEX 交 Agent 上下文调查后终裁。纯拉丁词按
    **词边界** 命中（见 _term_hits）—— 防 "LV" 命中 silver/valve 等英文词内部
    子串触发无谓 COMPLEX；中文旁不构成 ASCII 词边界，真实品牌（如"新款LV手袋"）
    仍命中。"""
    matched = _term_hits(_title_desc_text(case), terms.BRAND_TERMS)
    if not matched:
        return None
    return f"标题/描述命中品牌词: {', '.join(matched)}"


# ---------------------------------------------------------------------------
# R-3xx —— COMPLEX（规则无法确定 → 进 Agent 调查，不直判）
# ---------------------------------------------------------------------------


def _match_r301(case: ProductReviewCase) -> str | None:
    """R-301 COMPLEX：brand 或 category 空缺（None 与空串统一判定）→ 不 PASS 直放。

    brand 空缺 = "规避品牌"调查的起点信号（见 domain ``ProductInfo.brand`` 注记），
    category 空缺 = 无法判断商品属哪个审查域 —— 关键事实缺失时规则**无法确定**，
    一律进 Agent 调查（"能确定才直判，不能确定一律 COMPLEX"）。类目命中
    HIGH_RISK_PREFIXES 时 detail 标注高危前缀（保留原语义的人读信息）。
    """
    brand = case.product.brand
    category = case.product.category or ""
    brand_blank = _brand_missing(brand)
    category_blank = not category.strip()
    if not brand_blank and not category_blank:
        return None
    brand_disp = "None" if brand is None else repr(brand)  # None / ''
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
    """R-302 COMPLEX：标题/描述含规避模糊词（EVASION_TERMS）—— 模糊仿冒暗示，不直判。

    词表含中文词/符号词（同款/复刻/1:1）：中文无空格分词，维持子串命中（见
    _term_hits）。"""
    matched = _term_hits(_title_desc_text(case), terms.EVASION_TERMS)
    if not matched:
        return None
    return f"标题/描述命中规避模糊词: {', '.join(matched)}"


# ---------------------------------------------------------------------------
# 默认规则集（规则序 = 声明序；engine 按规则序收集 hits，再按 REJECT>COMPLEX>PASS 定裁决）
# ---------------------------------------------------------------------------

DEFAULT_RULES: tuple[Rule, ...] = (
    Rule("R-101", "品牌黑名单命中", "REJECT", _match_r101),
    Rule("R-102", "品牌词命中", "COMPLEX", _match_r102),
    Rule("R-301", "品牌/类目空缺", "COMPLEX", _match_r301),
    Rule("R-302", "规避模糊词命中", "COMPLEX", _match_r302),
)

__all__ = ["DEFAULT_RULES", "Rule", "RuleKind"]
