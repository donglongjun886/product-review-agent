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
在 ``positive_dimensions`` 给出阳性映射两处登记，**不写死成永久规则**。
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pra.domain.measurement import (
    ALL_DIMENSIONS,
    DIM_LISTING_REGISTRY,
    DIM_MERCHANT_PROFILE,
    DIM_POLICY_CITATION,
    DIM_TEXT_COMPLIANCE,
    MEASUREMENT_TYPE,
    MERCHANT_DIRTY_MIN,
    VERDICT_POSITIVE,
    VERDICTS,
    measurement_dimension,
    measurement_verdict,
)
from pra.domain.models import Evidence, ProductReviewCase
from pra.tools.merchant.tool import MERCHANT_HISTORY_TYPE

__all__ = [
    "ALWAYS_COVERED_DIMENSIONS",
    "RULE_BRAND_WORD",
    "RULE_EVASION_WORD",
    "CoverageReport",
    "capabilities_from_tools",
    "coverage_report",
    "positive_dimensions",
    "required_dimensions",
    "rule_hit_ids",
]

# 由**确定性纯函数**直接求值的维度：不需要任何工具/数据源参与 ⇒ 覆盖恒成立。
# ``text_compliance`` 由 screening 规则层在 gate 内即时求值（见 ``rule_hit_ids``）。
ALWAYS_COVERED_DIMENSIONS: frozenset[str] = frozenset({DIM_TEXT_COMPLIANCE})

# 平台规则层两个既定档位：R-102 品牌词命中只阻塞 PASS、不授权 REJECT（官方店/适配词/
# 授权产品会被误杀）；R-302 规避词命中是本 listing 文本自证，可授权 REJECT。
RULE_BRAND_WORD = "R-102"
RULE_EVASION_WORD = "R-302"

# 提供"可引用依据"的工具名（决定 policy_citation 维度在本环境是否可测）。
_CITATION_TOOLS: frozenset[str] = frozenset({"CaseSearchTool", "PolicySearchTool"})


def _coerce_case(case: Any) -> ProductReviewCase:
    """把 state 里的 case 归一成 ``ProductReviewCase``（dict 输入按契约严格校验）。"""
    if isinstance(case, ProductReviewCase):
        return case
    return ProductReviewCase.model_validate(case)


def _merchant_is_dirty(e: Evidence) -> bool:
    """``MERCHANT_HISTORY`` 是否达到"系统性规避"硬阈值（读 extra 的 removals / title）。"""
    extra = e.extra or {}
    removals = extra.get("removals")
    title = extra.get("title")
    if removals is None or title is None:
        return False
    return bool(removals >= MERCHANT_DIRTY_MIN or title >= MERCHANT_DIRTY_MIN)


def positive_dimensions(evidence: Iterable[Evidence]) -> dict[str, list[Evidence]]:
    """风险阳性证据按维度归类（确定性；只读证据类型/权重/extra）。

    阳性的**判定语义**（与 gate 的"硬阳性"要求同源，单一事实源）：

    - ``MERCHANT_HISTORY`` 达 ``MERCHANT_DIRTY_MIN`` → ``merchant_profile``；
    - ``MEASUREMENT`` 且 ``verdict=POSITIVE`` → 该测量维度（与上面几条互为冗余校验）。
    """
    out: dict[str, list[Evidence]] = defaultdict(list)
    for e in evidence or []:
        if e.type == MERCHANT_HISTORY_TYPE:
            if _merchant_is_dirty(e):
                out[DIM_MERCHANT_PROFILE].append(e)
        elif e.type == MEASUREMENT_TYPE:
            dim = measurement_dimension(e)
            if measurement_verdict(e) == VERDICT_POSITIVE and dim:
                out[dim].append(e)
    return dict(out)


def required_dimensions(case: Any) -> tuple[str, ...]:
    """本案 PASS 前**必须完成**的关键测量维度（可观测事实导出）。

    - ``listing_registry`` / ``merchant_profile``：恒必需（商品与商家是两处可核验事实源）；
    - ``text_compliance``：恒必需（确定性规则层，无需数据源，覆盖恒成立）。

    ``policy_citation`` **不在 PASS 必需集内** —— 它是「可引用依据可得」的 REJECT 候选
    条件，与"证明无风险"无关。

    ``case`` 缺失（空 state 防御）→ 返回空元组：**不臆断任何必需维度**，由 gate 侧
    的"无 case 不得 PASS"守住不 vacuous 放行。
    """
    if case is None:
        return ()
    _coerce_case(case)  # 契约校验：dict 输入非法即抛，不静默放行
    return (DIM_LISTING_REGISTRY, DIM_MERCHANT_PROFILE, DIM_TEXT_COMPLIANCE)


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
    """一次判定所需的全部事实侧输入（gate 只消费本对象 + 规则命中）。

    ``positive`` 是任意维度的风险阳性 —— 用于**阻塞 PASS**（哪怕只是商家行为画像）。
    """

    required: tuple[str, ...]
    covered: frozenset[str]
    missing: tuple[str, ...]  # required ∧ 未覆盖 ∧ 本环境可测 → NOT_MEASURED（可补救）
    unmeasurable: tuple[str, ...]  # required ∧ 未覆盖 ∧ 本环境不可测 → UNMEASURABLE
    positive: Mapping[str, list[Evidence]]
    capabilities: Mapping[str, bool]


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
        capabilities=caps,
    )


def rule_hit_ids(case: Any) -> frozenset[str]:
    """平台规则层命中集合（``triage(case).hits`` 的 rule_id；纯函数，一次求值）。

    ``case`` 缺失 → 空集。gate 据 ``RULE_EVASION_WORD``（R-302）授权 REJECT、
    据 ``RULE_BRAND_WORD``（R-102）阻塞 PASS —— 两者语义不同档，见模块常量注释。
    """
    if case is None:
        return frozenset()
    from pra.screening.engine import triage  # 延迟 import：避免模块导入期拉起筛选层

    return frozenset(hit.rule_id for hit in triage(_coerce_case(case)).hits)
