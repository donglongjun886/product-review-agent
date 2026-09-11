"""证据质量过滤 + extra 派生数值回填（tools_node 的两步确定性 Python，不调 LLM）。

``quality_filter``：丢弃 ``IMAGE_SIMILARITY`` 且 ``weight < EVIDENCE_MIN_SIM``
（0.70）的弱相似命中，其余类型全部保留（含低相似度的 ``IMAGE_LOGO`` —— logo 是
「检出即事实」，留给业务层判断）。工具层 ``to_evidence`` 只产原始事实、不设下限，
阈值裁决在本层。

``backfill_extra``：证据的派生数值统一在本层按类型回填，工具转换器不写 extra。
``IMAGE_SIMILARITY`` 按 ``weight`` 派生 ``similarity`` 与 ``strong``；
``MERCHANT_HISTORY`` / ``POLICY_REF`` / ``IMAGE_LOGO`` 按 value 固定格式正则解析；
``PRODUCT_FACT`` 仅在与库中版本不等时标注 ``version_drift``。解析失败保留原
extra、不报错。

不变式：两函数都不改写入参 ``Evidence``；``backfill_extra`` 逐条
``model_copy(update={"extra": merged})``，merged 恒为**新 dict**（先浅拷贝原
extra 再叠加），不与原对象共享 extra 引用。
"""

from __future__ import annotations

import re

from pra.domain.models import Evidence, ProductReviewCase
from pra.tools.image_analysis.tool import (
    EVIDENCE_MIN_SIM,
    EVIDENCE_STRONG,
    IMAGE_LOGO_TYPE,
    IMAGE_SIMILARITY_TYPE,
)
from pra.tools.merchant.tool import MERCHANT_HISTORY_TYPE
from pra.tools.policy_search.tool import POLICY_REF_TYPE
from pra.tools.product.tool import PRODUCT_FACT_TYPE

# 与各 tool.py::to_evidence 的 f-string 格式一一对应（改 tool 格式须同步改此处）。

# merchant/tool.py: f"{similar_product_count} similar / {removals} removals / "
#                   f"{title_relisting_count} title-relisting, credit={credit_score}"
_MERCHANT_HISTORY_RE = re.compile(
    r"(?P<similar>\d+)\s+similar\s*/\s*(?P<removals>\d+)\s+removals\s*/\s*"
    r"(?P<title>\d+)\s+title-relisting\s*,\s*credit=(?P<credit>-?\d+)"
)

# policy_search/tool.py: f"{h.policy_id} v{h.version} 条款：{text}"；policy_id 形如 POLICY_3.2
_POLICY_REF_RE = re.compile(r"^\s*(?P<policy_id>POLICY_[0-9]+(?:\.[0-9]+)*)\s+v(?P<version>\d+)")

# product/tool.py: value 内含 f"version={p.version}（库中最新）"
_PRODUCT_VERSION_RE = re.compile(r"version=(?P<version>\d+)（库中最新）")

# image_analysis/tool.py: f"logo={logo.brand}, conf={logo.confidence:.2f}"
_IMAGE_LOGO_RE = re.compile(r"logo=(?P<brand>[^,]+?)\s*,\s*conf=(?P<confidence>[0-9]+(?:\.[0-9]+)?)")


def quality_filter(raw: list[Evidence], *, evid_min_sim: float = EVIDENCE_MIN_SIM) -> list[Evidence]:
    """丢弃弱相似命中（``weight < evid_min_sim``）；返回新 list、元素复用原引用，入参空 → ``[]``。"""
    kept: list[Evidence] = []
    for e in raw or []:
        if e.type == IMAGE_SIMILARITY_TYPE and e.weight < evid_min_sim:
            continue  # 弱相似命中 —— 噪声，不进证据链
        kept.append(e)
    return kept


def _parse_merchant_history(value: str) -> dict | None:
    """按 value 固定格式解析 ``MERCHANT_HISTORY`` 数值；失败返回 None。"""
    m = _MERCHANT_HISTORY_RE.search(value)
    if m is None:
        return None
    try:
        return {
            "similar": int(m.group("similar")),
            "removals": int(m.group("removals")),
            "title": int(m.group("title")),
            "credit": int(m.group("credit")),
        }
    except ValueError:
        return None


def _parse_policy_ref(value: str) -> dict | None:
    """按 value 前缀 ``POLICY_x.y vN`` 解析；失败返回 None。"""
    m = _POLICY_REF_RE.match(value)
    if m is None:
        return None
    try:
        return {
            "policy_id": m.group("policy_id"),
            "policy_version": int(m.group("version")),
        }
    except ValueError:
        return None


def _parse_image_logo(value: str) -> dict | None:
    """按 value 固定格式 ``logo=<brand>, conf=<float>`` 解析；失败返回 None。"""
    m = _IMAGE_LOGO_RE.search(value)
    if m is None:
        return None
    try:
        return {"logo_brand": m.group("brand").strip(), "confidence": float(m.group("confidence"))}
    except ValueError:
        return None


def _product_version_drift(e: Evidence, case: ProductReviewCase | None) -> bool | None:
    """``PRODUCT_FACT`` 版本漂移判定：True=漂移；False/None=不标注（仅不等时标注）。"""
    if case is None:
        return None
    product = getattr(case, "product", None)
    if product is None or product.version is None:
        return None
    m = _PRODUCT_VERSION_RE.search(e.value)
    if m is None:
        return None
    try:
        return int(m.group("version")) != int(product.version)
    except ValueError:
        return None


def backfill_extra(evs: list[Evidence], *, case: ProductReviewCase | None = None) -> list[Evidence]:
    """extra 派生数值回填：逐条输出 ``model_copy(update={"extra": merged})``。

    merged 恒为新 dict 且**绝不改写原对象**；解析失败保留原 extra、不报错。
    """
    out: list[Evidence] = []
    for e in evs or []:
        merged = dict(e.extra)  # 新 dict：保留工具/前序已填键，叠加派生键
        if e.type == IMAGE_SIMILARITY_TYPE:
            # similarity 即 weight（工具侧 weight 就是相似度）
            merged["similarity"] = round(e.weight, 3)
            merged["strong"] = bool(e.weight >= EVIDENCE_STRONG)
        elif e.type == MERCHANT_HISTORY_TYPE:
            parsed = _parse_merchant_history(e.value)
            if parsed is not None:
                merged.update(parsed)
        elif e.type == POLICY_REF_TYPE:
            parsed = _parse_policy_ref(e.value)
            if parsed is not None:
                merged.update(parsed)
        elif e.type == PRODUCT_FACT_TYPE:
            if _product_version_drift(e, case) is True:
                merged["version_drift"] = True
        elif e.type == IMAGE_LOGO_TYPE:
            parsed = _parse_image_logo(e.value)
            if parsed is not None:
                merged.update(parsed)
        out.append(e.model_copy(update={"extra": merged}))
    return out
