"""维度覆盖判定：``required × 证据存在性 × 环境能力`` —— gate 的事实侧输入。

本模块是**确定性纯函数集合**，只读三类输入，绝不读真值/标注/假设：

1. ``case``（案件输入快照）：决定"本案需要哪些关键测量"；
2. ``evidence``（该次运行的证据链）：决定"哪些维度真的测过了、结论是什么"；
3. ``capabilities``（装配期声明的环境能力）：决定"某维度在本环境是否可测"。

三态由此推导（``domain/measurement.py`` 定义了 dimension/verdict 契约）：

- ``COVERED``：该维度有 ``MEASUREMENT`` 证据（任意 verdict），或有映射到它的阳性证据；
- ``NOT_MEASURED``：在 required 内、未覆盖、但本环境**可测** —— 属**可补救**的取证缺口；
- ``UNMEASURABLE``：在 required 内、未覆盖、且本环境**不可测** —— 环境缺失，重跑无用。

``required`` 的**唯一**来源是"平台风险分类 ⊗ 案件可观测事实"（见 ``required_dimensions``），
不由 Agent 的 plan 决定 —— 否则判定会随 LLM 的计划波动；也不读 ``expected``/``annotation``。
新增维度只需在 ``domain/measurement.py`` 加常量、在 ``required_dimensions`` 给出必需性来源、
在 ``positive_dimensions`` 给出阳性映射三处登记，**不写死成永久规则**。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pra.domain.measurement import (
    ALL_DIMENSIONS,
    DIM_IMAGE_APPEARANCE,
    DIM_LISTING_REGISTRY,
    DIM_MERCHANT_PROFILE,
    DIM_POLICY_CITATION,
    DIM_TEXT_COMPLIANCE,
    MEASUREMENT_TYPE,
    VERDICT_NEGATIVE,
    VERDICT_POSITIVE,
    VERDICTS,
    measurement_dimension,
    measurement_verdict,
)
from pra.domain.models import Evidence, ProductReviewCase
from pra.tools.image_analysis.tool import (
    EVIDENCE_MIN_SIM,
    EVIDENCE_STRONG,
    IMAGE_LOGO_TYPE,
    IMAGE_SIMILARITY_TYPE,
)
from pra.tools.merchant.tool import MERCHANT_DIRTY_MIN, MERCHANT_HISTORY_TYPE

from .evidence import _parse_merchant_history

__all__ = [
    "ALWAYS_COVERED_DIMENSIONS",
    "CoverageReport",
    "capabilities_from_tools",
    "coverage_report",
    "dimension_strength",
    "listing_signal_present",
    "positive_dimensions",
    "reject_positive_dims",
    "required_dimensions",
    "required_dimensions_for_reject",
    "text_brand_word_hit",
    "text_compliance_positive",
    "text_evasion_hit",
]

# 由**确定性纯函数**直接求值的维度：不需要任何工具/数据源参与 ⇒ 覆盖恒成立。
# ``text_compliance`` 由 screening 规则层在 gate 内即时求值（见 ``text_compliance_positive``）。
ALWAYS_COVERED_DIMENSIONS: frozenset[str] = frozenset({DIM_TEXT_COMPLIANCE})

# 提供"可引用依据"的工具名（决定 policy_citation 维度在本环境是否可测）。
_CITATION_TOOLS: frozenset[str] = frozenset({"CaseSearchTool", "PolicySearchTool"})


def _coerce_case(case: Any) -> ProductReviewCase:
    """把 state 里的 case 归一成 ``ProductReviewCase``（dict 输入按契约严格校验）。"""
    if isinstance(case, ProductReviewCase):
        return case
    return ProductReviewCase.model_validate(case)


def _merchant_is_dirty(e: Evidence) -> bool:
    """``MERCHANT_HISTORY`` 是否达到"系统性规避"硬阈值（extra 优先，缺则解析 value）。"""
    extra = e.extra or {}
    removals = extra.get("removals")
    title = extra.get("title")
    if removals is None or title is None:
        parsed = _parse_merchant_history(e.value)
        if parsed is None:
            return False
        removals, title = parsed["removals"], parsed["title"]
    return bool(removals >= MERCHANT_DIRTY_MIN or title >= MERCHANT_DIRTY_MIN)


def positive_dimensions(evidence: Iterable[Evidence]) -> dict[str, list[Evidence]]:
    """风险阳性证据按维度归类（确定性；只读证据类型/权重/extra）。

    阳性的**判定语义**（与 gate 的"硬阳性"要求同源，单一事实源）：

    - ``IMAGE_SIMILARITY`` 且 ``weight >= EVIDENCE_STRONG``（0.85）→ ``image_appearance``；
      **弱相似 0.70~0.85 不算阳性** —— 它未达处置阈值，应然处置是"与商品事实交叉"
      （由 required set 的 ``listing_registry`` 覆盖承担），不能单独撑起 REJECT。
    - ``IMAGE_LOGO`` → ``image_appearance``（检出即事实，信任工具置信度）；
    - ``MERCHANT_HISTORY`` 达 ``MERCHANT_DIRTY_MIN`` → ``merchant_profile``；
    - ``MEASUREMENT`` 且 ``verdict=POSITIVE`` → 该测量维度（与上面几条互为冗余校验）。
    """
    out: dict[str, list[Evidence]] = defaultdict(list)
    for e in evidence or []:
        if e.type == IMAGE_SIMILARITY_TYPE:
            if e.weight >= EVIDENCE_STRONG:
                out[DIM_IMAGE_APPEARANCE].append(e)
        elif e.type == IMAGE_LOGO_TYPE:
            out[DIM_IMAGE_APPEARANCE].append(e)
        elif e.type == MERCHANT_HISTORY_TYPE:
            if _merchant_is_dirty(e):
                out[DIM_MERCHANT_PROFILE].append(e)
        elif e.type == MEASUREMENT_TYPE:
            dim = measurement_dimension(e)
            if measurement_verdict(e) == VERDICT_POSITIVE and dim:
                out[dim].append(e)
    return dict(out)


def listing_signal_present(evidence: Iterable[Evidence]) -> bool:
    """本案商品**自身**的外观信号是否存在（弱相似 >= 0.70 或 Logo 检出即算）。

    用途：商家行为维度需要本 listing 的旁证才够格自动拒绝 —— 弱相似虽未达处置阈值
    （不算阳性），却足以把"商家脏"从"仅商家画像"提升为"本 listing 也有疑点"。
    """
    for e in evidence or []:
        if e.type == IMAGE_LOGO_TYPE:
            return True
        if e.type == IMAGE_SIMILARITY_TYPE and (e.weight or 0.0) >= EVIDENCE_MIN_SIM:
            return True
    return False


def reject_positive_dims(evidence: Iterable[Evidence]) -> frozenset[str]:
    """**足以授权自动 REJECT** 的阳性维度（比 ``positive_dimensions`` 更严）。

    - ``image_appearance``：本 listing 的直接测量（强相似 / Logo）→ 直接授权；
    - ``merchant_profile``：**需本 listing 的外观信号佐证**（``listing_signal_present``）——
      仅"商家历史脏"不足以對本 listing 定案（reviewer 语义：疑似规避但图/文本无确证
      → 克制转人工，见 GT 家族 ``dirty_brand_missing_cleanimg``）；反之
      "商家脏 + 弱相似"（``wsim_dirty``）或"商家脏 + 强相似"则成立。

    ``text_compliance`` 的规则级命中不在这里 —— 它由 gate 侧对确定性规则求值
    （``text_evasion_hit`` 授权 REJECT、``text_brand_word_hit`` 只阻塞 PASS）。
    """
    dims = set(positive_dimensions(evidence))
    out: set[str] = set()
    if DIM_IMAGE_APPEARANCE in dims:
        out.add(DIM_IMAGE_APPEARANCE)
    if DIM_MERCHANT_PROFILE in dims and listing_signal_present(evidence):
        out.add(DIM_MERCHANT_PROFILE)
    return frozenset(out)


def _negative_strengths(evidence: Iterable[Evidence]) -> dict[str, float]:
    """维度 → 阴性测量的可信度（同维度多条取最大；非测量证据不参与）。"""
    out: dict[str, float] = {}
    for e in evidence or []:
        if e.type != MEASUREMENT_TYPE or measurement_verdict(e) != VERDICT_NEGATIVE:
            continue
        dim = measurement_dimension(e)
        if dim:
            out[dim] = max(out.get(dim, 0.0), float(e.weight or 0.0))
    return out


def dimension_strength(evidence: Iterable[Evidence]) -> dict[str, float]:
    """维度 → 该维度**决定性证据**的强度（有阳性取阳性最大，否则取阴性测量可信度）。

    供 ``finalize_decision_confidence`` 使用 —— 置信度的主项来自真实证据强度，
    而不是 LLM 的 posterior。
    """
    evs = list(evidence or [])
    pos = positive_dimensions(evs)
    neg = _negative_strengths(evs)
    strength: dict[str, float] = dict(neg)
    for dim, items in pos.items():
        strength[dim] = max(float(e.weight or 0.0) for e in items)
    return strength


def required_dimensions(case: Any) -> tuple[str, ...]:
    """本案 PASS 前**必须完成**的关键测量维度（可观测事实导出）。

    - ``listing_registry`` / ``merchant_profile``：恒必需（商品与商家是两处可核验事实源）；
    - ``text_compliance``：恒必需（确定性规则层，无需数据源，覆盖恒成立）；
    - ``image_appearance``：**仅当案件带图时**必需（无图案件不存在外观风险面）。

    ``policy_citation`` **不在 PASS 必需集内** —— 它是 REJECT 候选的必要条件
    （见 ``required_dimensions_for_reject``），与"证明无风险"无关。

    ``case`` 缺失（空 state 防御）→ 返回空元组：**不臆断任何必需维度**，由 gate 侧
    的"无 case 不得 PASS"守住不 vacuous 放行。
    """
    if case is None:
        return ()
    product = _coerce_case(case).product
    dims = [DIM_LISTING_REGISTRY, DIM_MERCHANT_PROFILE, DIM_TEXT_COMPLIANCE]
    if product.images:
        dims.append(DIM_IMAGE_APPEARANCE)
    return tuple(dims)


def required_dimensions_for_reject(case: Any) -> tuple[str, ...]:
    """REJECT 候选额外要求的维度：可引用依据必须可得。"""
    return (*required_dimensions(case), DIM_POLICY_CITATION)


def capabilities_from_tools(tools: Sequence[Any]) -> dict[str, bool]:
    """从**实际装配的工具集**导出环境能力（唯一权威来源）。

    ``measured_dimensions`` / ``measurement_available`` 由各 Tool 声明；同一维度被任一
    "可测"的工具覆盖即为可测。确定性规则维度恒可测；``policy_citation`` 由检索工具提供。
    """
    caps: dict[str, bool] = {dim: False for dim in ALL_DIMENSIONS}
    for tool in tools or []:
        available = bool(getattr(tool, "measurement_available", True))
        for dim in getattr(tool, "measured_dimensions", frozenset()) or ():
            if dim in caps:
                caps[dim] = caps[dim] or available
    caps[DIM_TEXT_COMPLIANCE] = True
    if any(getattr(tool, "name", None) in _CITATION_TOOLS for tool in tools or []):
        caps[DIM_POLICY_CITATION] = True
    return caps


@dataclass(frozen=True)
class CoverageReport:
    """一次判定所需的全部事实侧输入（gate 只消费本对象 + dc + 规则命中）。

    ``positive`` 与 ``reject_positive`` 是**两个不同强度的概念**，不可混用：

    - ``positive``：任意维度的风险阳性 —— 用于**阻塞 PASS**（哪怕只是商家行为画像）；
    - ``reject_positive``：**足以授权自动 REJECT** 的阳性 —— 额外要求"本 listing 级信号"，
      因为商家行为是**针对该商家**的画像，不能单独当作本 listing 违规的确证
      （reviewer 语义：疑似规避但图/文本无确证 → 克制转人工）。
    """

    required: tuple[str, ...]
    covered: frozenset[str]
    missing: tuple[str, ...]  # required ∧ 未覆盖 ∧ 本环境可测 → NOT_MEASURED（可补救）
    unmeasurable: tuple[str, ...]  # required ∧ 未覆盖 ∧ 本环境不可测 → UNMEASURABLE
    positive: Mapping[str, list[Evidence]]
    reject_positive: frozenset[str]
    strength: Mapping[str, float]
    capabilities: Mapping[str, bool]

    @property
    def complete(self) -> bool:
        """required 全部覆盖（无论阴性/阳性）。"""
        return not self.missing and not self.unmeasurable

    @property
    def negative_only(self) -> bool:
        """required 全部覆盖且**无阳性** —— PASS 的事实前提。"""
        return self.complete and not self.positive

    @property
    def negative_dims(self) -> frozenset[str]:
        """已覆盖且结论为阴性的维度。"""
        return frozenset(d for d in self.covered if d not in self.positive)


def coverage_report(
    case: Any,
    evidence: Iterable[Evidence],
    capabilities: Mapping[str, bool] | None = None,
) -> CoverageReport:
    """构造覆盖报告（纯函数）。``capabilities=None`` → 视为全部可测（保守：不掩盖未测）。"""
    evs = list(evidence or [])
    caps: Mapping[str, bool] = capabilities if capabilities is not None else {
        dim: True for dim in ALL_DIMENSIONS
    }
    required = required_dimensions(case)
    covered: set[str] = set(ALWAYS_COVERED_DIMENSIONS)
    covered |= set(positive_dimensions(evs))
    for e in evs:
        if e.type == MEASUREMENT_TYPE and measurement_verdict(e) in VERDICTS:
            dim = measurement_dimension(e)
            if dim:
                covered.add(dim)
    missing = tuple(d for d in required if d not in covered and caps.get(d, True))
    unmeasurable = tuple(d for d in required if d not in covered and not caps.get(d, True))
    return CoverageReport(
        required=required,
        covered=frozenset(covered),
        missing=missing,
        unmeasurable=unmeasurable,
        positive=positive_dimensions(evs),
        reject_positive=reject_positive_dims(evs),
        strength=dimension_strength(evs),
        capabilities=caps,
    )


def text_compliance_positive(case: Any) -> bool:
    """文本合规维度的**阳性**判定：平台规则层命中品牌词（R-102）或规避词（R-302）。

    刻意**不含 R-301（brand/category 空缺）**：那是"关键事实缺失 ⇒ 需核验"的
    **可核验性**信号，由 required set 的 ``listing_registry`` 承担（在库可查即核验通过），
    若把它当文本阳性，会把"品牌空缺但在库可验证"的正常案一并挡死。
    """
    from pra.screening.engine import triage  # 延迟 import：避免模块导入期拉起筛选层

    result = triage(_coerce_case(case))
    return any(hit.rule_id in ("R-102", "R-302") for hit in result.hits)


def text_evasion_hit(case: Any) -> bool:
    """平台规则层命中**规避词**（R-302）—— 本 listing 文本自证，可直接授权 REJECT。

    R-302（同款/复刻/高仿/1:1/原单）是平台既定的违规信号（与 R-301/R-102 的
    "COMPLEX → 交调查"不同档），故它与"证据链里的硬阳性"并列，构成 REJECT 的事实依据。
    """
    if case is None:
        return False
    from pra.screening.engine import triage

    return any(hit.rule_id == "R-302" for hit in triage(_coerce_case(case)).hits)


def text_brand_word_hit(case: Any) -> bool:
    """平台规则层命中**第三方品牌词**（R-102）—— 只**阻塞 PASS**，不授权 REJECT。

    R-102 的既定语义是"交 Agent 上下文调查"（官方店/适配词/授权产品会被误杀），
    故它不能单独撑起自动拒绝（GT 家族 ``adapter_brandword`` 即此）。
    """
    if case is None:
        return False
    from pra.screening.engine import triage

    return any(hit.rule_id == "R-102" for hit in triage(_coerce_case(case)).hits)
