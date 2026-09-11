"""测量维度与「已测」事实的领域契约（无依赖叶子模块）。

**为什么单独一个模块**：工具层（``pra.tools.*``）需要产出测量证据，而 gate 层需要读取它；
若把类型常量放在 guardrails 侧，就会形成 ``tools → guardrails → tools`` 的循环 import。
本模块只依赖 ``domain/models``，两层都可安全引用。

三个概念必须分清（缺一不可）：

- **dimension**：平台风险面的**受控词表**（封闭枚举），不是 LLM 自由文本；
- **verdict**：一次**已完成测量**的结论 —— ``POSITIVE``（发现达处置阈值的风险）/
  ``NEGATIVE``（未发现）。注意 verdict 是相对**硬阈值**而言的：外观维度上弱相似
  （0.70~0.85）算 ``NEGATIVE``（未达处置阈值），它仍作为 ``IMAGE_SIMILARITY`` 证据留在
  证据链里供 LLM 参考，但不构成"风险阳性"。
- **coverage**（``COVERED`` / ``NOT_MEASURED`` / ``UNMEASURABLE``）：**不是证据**，
  是"测量是否发生"的状态，由 gate 层用 ``required × 证据存在性 × 环境能力`` 推导 ——
  绝不能为"没测"造一个证据对象（那会把"缺席"变成"事实"，
  正是"查不到 ≠ 证明无"被破坏的根源）。

``MEASUREMENT`` 是承载「已测」的**唯一**证据类型；风险**阳性**仍由各工具既有的证据类型
承载（``IMAGE_SIMILARITY`` / ``IMAGE_LOGO`` / 商家脏 …），本模块不为阳性另造表示，
避免同一事实两处表示而漂移。
"""

from __future__ import annotations

from .models import Evidence

__all__ = [
    "ALL_DIMENSIONS",
    "DIM_IMAGE_APPEARANCE",
    "DIM_LISTING_REGISTRY",
    "DIM_MERCHANT_PROFILE",
    "DIM_POLICY_CITATION",
    "DIM_TEXT_COMPLIANCE",
    "MEASUREMENT_TYPE",
    "VERDICTS",
    "VERDICT_NEGATIVE",
    "VERDICT_POSITIVE",
    "is_measurement",
    "make_measurement",
    "measurement_dimension",
    "measurement_ref_id",
    "measurement_verdict",
]

# ---- 受控维度词表（封闭枚举；新增维度须同时给出：必需性来源、可测性来源、阳性类型映射）----

DIM_LISTING_REGISTRY = "listing_registry"  # 商品在库可核验（品牌/类目/版本事实）
DIM_MERCHANT_PROFILE = "merchant_profile"  # 商家行为模式（removals / title-relisting）
DIM_TEXT_COMPLIANCE = "text_compliance"  # 标题/描述合规（品牌词、规避词、绝对化用语）
DIM_IMAGE_APPEARANCE = "image_appearance"  # 外观/IP（相似度、Logo 检出）
DIM_POLICY_CITATION = "policy_citation"  # 可引用依据（政策条款 / 人工先例）

ALL_DIMENSIONS: tuple[str, ...] = (
    DIM_LISTING_REGISTRY,
    DIM_MERCHANT_PROFILE,
    DIM_TEXT_COMPLIANCE,
    DIM_IMAGE_APPEARANCE,
    DIM_POLICY_CITATION,
)

# ---- 测量结论 ----

VERDICT_POSITIVE = "POSITIVE"
VERDICT_NEGATIVE = "NEGATIVE"
VERDICTS: frozenset[str] = frozenset({VERDICT_POSITIVE, VERDICT_NEGATIVE})

# ---- 测量证据类型 ----

MEASUREMENT_TYPE = "MEASUREMENT"


def measurement_ref_id(dimension: str, source_ref: str) -> str:
    """测量证据的 ``ref_id`` = ``"{dimension}:{source_ref}"``。

    **必须包含 dimension**：证据去重指纹是 ``(type, source, ref_id or value)``
    （``agent/state.py::_evidence_key``），语义为"证据不可篡改、重放/重试幂等"。
    若 ``ref_id`` 只取源对象（如图片 URL），同一对象在不同维度上的测量会互相吞并；
    测到阳性时改由**阳性证据类型**承载（类型不同 ⇒ key 不同），故阳性不会被阴性覆盖。
    """
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
    :param source_ref: 被测量的源对象稳定标识（图 URL / 商品 ID / 商家 ID …）；
    :param verdict: ``POSITIVE`` / ``NEGATIVE``（相对**处置硬阈值**的结论）；
    :param weight: 测量本身的**可信度**（不是风险强度；``NEGATIVE`` 同样给高值）；
    :param value: 人读摘要（如"外观比对完成：无 Logo 命中，最高相似度 0.12"）。
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
    """取测量证据的维度；缺失/非测量证据返回 None（防御：extra 由工具填写）。"""
    if not is_measurement(e):
        return None
    dim = (e.extra or {}).get("dimension")
    return dim if isinstance(dim, str) and dim else None


def measurement_verdict(e: Evidence) -> str | None:
    """取测量证据的结论；非法值返回 None（gate 侧按"未证实"保守处理）。"""
    if not is_measurement(e):
        return None
    verdict = (e.extra or {}).get("verdict")
    return verdict if verdict in VERDICTS else None
