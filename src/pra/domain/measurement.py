"""测量维度与「已测」事实的领域契约（无依赖叶子模块）。

- **dimension**：平台风险面的**受控词表**（封闭枚举）；
- **verdict**：一次**已完成测量**的结论 —— ``POSITIVE``（发现达处置阈值的风险）/
  ``NEGATIVE``（未发现）。
"""

from __future__ import annotations

from .models import Evidence

__all__ = [
    "ALL_DIMENSIONS",
    "CITABLE_TYPES",
    "DEFAULT_EVIDENCE_WEIGHT",
    "DIM_LISTING_REGISTRY",
    "DIM_MERCHANT_PROFILE",
    "DIM_POLICY_CITATION",
    "DIM_TEXT_COMPLIANCE",
    "MEASUREMENT_TYPE",
    "MERCHANT_DIRTY_MIN",
    "VERDICTS",
    "VERDICT_NEGATIVE",
    "VERDICT_POSITIVE",
    "is_measurement",
    "make_measurement",
    "measurement_dimension",
    "measurement_ref_id",
    "measurement_verdict",
]

# ---- 受控维度词表（封闭枚举）----

DIM_LISTING_REGISTRY = "listing_registry"  # 商品在库可核验（品牌/类目/版本事实）
DIM_MERCHANT_PROFILE = "merchant_profile"  # 商家行为模式（removals / title-relisting）
DIM_TEXT_COMPLIANCE = "text_compliance"  # 标题/描述合规（品牌词、规避词、绝对化用语）
DIM_POLICY_CITATION = "policy_citation"  # 可引用依据（政策条款 / 人工先例）

ALL_DIMENSIONS: tuple[str, ...] = (
    DIM_LISTING_REGISTRY,
    DIM_MERCHANT_PROFILE,
    DIM_TEXT_COMPLIANCE,
    DIM_POLICY_CITATION,
)

# ---- 测量结论 ----

VERDICT_POSITIVE = "POSITIVE"
VERDICT_NEGATIVE = "NEGATIVE"
VERDICTS: frozenset[str] = frozenset({VERDICT_POSITIVE, VERDICT_NEGATIVE})

# ---- 测量证据类型 ----

MEASUREMENT_TYPE = "MEASUREMENT"

# ---- 证据确定性阈值 ----

# 无风险量纲证据类型的默认权重（PRODUCT_FACT / POLICY_REF）。
DEFAULT_EVIDENCE_WEIGHT = 0.5

# 商家历史「系统性规避行为」阈值：removals 或 title-relisting 达到该值即成立。
MERCHANT_DIRTY_MIN = 3

# 可引用依据类型。
CITABLE_TYPES = frozenset({"CASE_PRECEDENT", "POLICY_REF"})


def measurement_ref_id(dimension: str, source_ref: str) -> str:
    """测量证据的 ``ref_id`` = ``"{dimension}:{source_ref}"``。"""
    return f"{dimension}:{source_ref}"


def make_measurement(
    *,
    dimension: str,
    source: str,
    source_ref: str,
    verdict: str,
    weight: float,
    value: str,
    extra: dict | None = None,
) -> Evidence:
    """构造 1 条 ``MEASUREMENT`` 证据（一次已完成测量的结论）。

    :param dimension: ``ALL_DIMENSIONS`` 之一；
    :param source: 产出该测量的工具名；
    :param source_ref: 被测量的源对象稳定标识（商品 ID / 商家 ID / 条款 ID …）；
    :param verdict: ``POSITIVE`` / ``NEGATIVE``（相对**处置硬阈值**的结论）；
    :param weight: 测量本身的**可信度**（不是风险强度；``NEGATIVE`` 同样给高值）；
    :param value: 人读摘要（如"商家行为核验完成：历史无移除记录"）。
    """
    merged = dict(extra or {})
    merged["dimension"] = dimension
    merged["verdict"] = verdict
    merged["measured_by"] = source
    return Evidence(
        type=MEASUREMENT_TYPE,
        source=source,
        value=value,
        weight=weight,
        ref_id=measurement_ref_id(dimension, source_ref),
        extra=merged,
    )


def is_measurement(e: Evidence) -> bool:
    """是否为本模块定义的测量证据。"""
    return e.type == MEASUREMENT_TYPE


def measurement_dimension(e: Evidence) -> str | None:
    """取测量证据的维度；缺失/非测量证据返回 None。"""
    if not is_measurement(e):
        return None
    dim = (e.extra or {}).get("dimension")
    return dim if isinstance(dim, str) and dim else None


def measurement_verdict(e: Evidence) -> str | None:
    """取测量证据的结论；非法值返回 None。"""
    if not is_measurement(e):
        return None
    verdict = (e.extra or {}).get("verdict")
    return verdict if verdict in VERDICTS else None
