"""确定性证据质量过滤 + O-8 extra 回填（04 §9 / 01 §5.0，tools_node 消费）。

本模块是 tools_node evidence processing 的两步确定性 Python（不调 LLM）：

1. ``quality_filter`` —— T-11 已拍板 B 的证据质量下限：IMAGE_SIMILARITY 且
   ``weight < EVIDENCE_MIN_SIM(0.70)`` 的弱命中丢弃（避免低相似噪声进证据链）；
   其余类型全保留。工具层 to_evidence 只产原始事实、不设下限（任务边界，
   见 ``pra.tools.image_analysis.tool`` 模块 docstring），阈值裁决在本层落位。

2. ``backfill_extra`` —— O-8 已拍板（04 §4/§9、03-decisions §4.4 行 5）：Evidence.extra
   的**派生数值**统一在本层回填，工具转换器不写 extra。回填目标字段以
   graph-mvp-contracts §2.2 定死：

   - ``IMAGE_SIMILARITY`` → ``{"similarity": round(weight,3), "strong": weight>=EVIDENCE_STRONG}``
     （确定性，不解析 value；``similarity`` 直接由 ``weight`` 派生 —— weight 本就是
     相似度，T-11；``strong`` 供矛盾检测/风险派生读，EVIDENCE_STRONG=0.85 与
     EVIDENCE_MIN_SIM 同源 import 自 ``pra.tools.image_analysis.tool``）。
   - ``MERCHANT_HISTORY`` → 按 value 固定格式 ``"N similar / N removals / N
     title-relisting, credit=N"``（merchant/tool.py to_evidence 拼装）正则解析
     ``{"similar", "removals", "title", "credit"}``。
   - ``POLICY_REF`` → 按 value 前缀 ``"POLICY_x.y vN 条款：…"``（policy_search/tool.py
     拼装）解析 ``{"policy_id", "policy_version"}``。
   - ``PRODUCT_FACT`` → value 含 ``"version=<int>（库中最新）"``（product/tool.py
     拼装）；当 ``case`` 给出且库中 version != ``case.product.version`` → 标注
     ``{"version_drift": True}``（§2.2 语义：仅在不等时标注；相等/无法解析/无 case
     时**不写该键** —— 确定性消费者一律用 ``extra.get("version_drift") is True`` 判定，
     缺失 == 无漂移，与"只在不等时标注"等价且无来回翻转风险，因为证据一旦收集不可篡改）。
   - ``IMAGE_LOGO`` → 按 value 固定格式 ``"logo=<brand>, conf=<float>"``
     （image_analysis/tool.py 拼装）解析 ``{"logo_brand", "confidence"}``。

   **尽力而为**：任何解析失败保留原 extra、不报错（不中断证据流）；回填是纯派生，
   确定性逻辑只读 extra/weight/ref_id（§2.2），value 字符串仅供人读与审计。

不变式：两个函数都不改动入参 Evidence（pydantic 模型默认不可变语义）—— 返回
新 list；``backfill_extra`` 对每条输出 ``model_copy(update={"extra": merged})``，
且 merged 恒为**新 dict**（先 ``dict(e.extra)`` 再叠加），避免与原对象共享 extra
引用。实现对齐 docs/04-graph-design.md §9 evidence.py 行（表 679 行）。
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

# ---------------------------------------------------------------------------
# value 解析正则 —— 与各 tool.py::to_evidence 的 f-string 固定格式一一对应
# （改动 tool 格式前必须先改此处；解析失败走"保留原 extra"兜底，不回退工具）
# ---------------------------------------------------------------------------

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
    """确定性证据质量过滤（T-11 B：EVIDENCE_MIN_SIM=0.70）。

    丢弃 IMAGE_SIMILARITY 且 ``weight < evid_min_sim`` 的弱命中；其余类型/强度全保留
    （MERCHANT_HISTORY/POLICY_REF 等默认权重 0.85/0.9 本就不低于下限；低相似度 Logo
    命中不在此过滤，Logo 是"检出即事实"，保留给业务层判断）。返回新 list、
    元素复用原 Evidence 引用（函数不改写任何对象）；入参为 None/空 → 返回 []。
    """
    kept: list[Evidence] = []
    for e in raw or []:
        if e.type == IMAGE_SIMILARITY_TYPE and e.weight < evid_min_sim:
            continue  # 弱相似命中 —— 噪声，不进证据链
        kept.append(e)
    return kept


def _parse_merchant_history(value: str) -> dict | None:
    """按 value 固定格式解析 MERCHANT_HISTORY 数值；失败返回 None（尽力而为）。"""
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
    """按 value 前缀 ``POLICY_x.y vN`` 解析；失败返回 None（尽力而为）。"""
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
    """PRODUCT_FACT 版本漂移判定：返回 True=漂移；False/None=不标注。

    None 表示无法判定（value 无 ``version=<int>（库中最新）`` 或 case 缺失/
    case.product.version 缺失）—— 与"相等"同样不写 extra 键（§2.2 仅在不等时标注）。
    """
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
    """O-8 extra 派生数值回填：逐条输出 ``model_copy(update={"extra": merged})``。

    按 graph-mvp-contracts §2.2 定死字段回填（见模块 docstring 对照表）；merged 恒为
    新 dict（``dict(e.extra)`` 起步再叠加派生键），**绝不改写原对象**、不共享 extra
    引用。解析失败 → merged 保持原 extra 内容，不报错（尽力而为）。每条都返回
    model_copy（即使无键新增），保证调用方拿到的与输入无对象别名。入参 None/空 → []。
    """
    out: list[Evidence] = []
    for e in evs or []:
        merged = dict(e.extra)  # 新 dict：保留工具/前序已填键，叠加派生键
        if e.type == IMAGE_SIMILARITY_TYPE:
            # 确定性派生：similarity 即 weight（工具 weight=相似度），不解析 value
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
        # 其余类型（OCR_TEXT / CASE_PRECEDENT 等）无 §2.2 回填字段 → 原样保留 extra
        out.append(e.model_copy(update={"extra": merged}))
    return out
