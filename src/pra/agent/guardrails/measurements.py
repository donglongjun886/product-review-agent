"""维度覆盖判定：按 ``required × 证据存在性 × 环境能力`` 推导各维度三态。"""

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

ALWAYS_COVERED_DIMENSIONS: frozenset[str] = frozenset({DIM_TEXT_COMPLIANCE})

# 平台规则层档位（R-102 品牌词 / R-302 规避词）
RULE_BRAND_WORD = "R-102"
RULE_EVASION_WORD = "R-302"

_CITATION_TOOLS: frozenset[str] = frozenset({"CaseSearchTool", "PolicySearchTool"})


def _coerce_case(case: Any) -> ProductReviewCase:
    """把 case 归一成 ``ProductReviewCase``（dict 输入严格校验）。"""
    if isinstance(case, ProductReviewCase):
        return case
    return ProductReviewCase.model_validate(case)


def _merchant_is_dirty(e: Evidence) -> bool:
    """``MERCHANT_HISTORY`` 是否达到 ``MERCHANT_DIRTY_MIN``（读 extra 的 removals / title）。"""
    extra = e.extra or {}
    removals = extra.get("removals")
    title = extra.get("title")
    if removals is None or title is None:
        return False
    return bool(removals >= MERCHANT_DIRTY_MIN or title >= MERCHANT_DIRTY_MIN)


def positive_dimensions(evidence: Iterable[Evidence]) -> dict[str, list[Evidence]]:
    """风险阳性证据按维度归类：``MERCHANT_HISTORY`` 达 ``MERCHANT_DIRTY_MIN`` → ``merchant_profile``，``MEASUREMENT`` 且 verdict 阳性 → 该测量维度。"""
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
    """本案必需的测量维度；``case`` 缺失返回空元组。"""
    if case is None:
        return ()
    _coerce_case(case)  # 契约校验：非法输入即抛
    return (DIM_LISTING_REGISTRY, DIM_MERCHANT_PROFILE, DIM_TEXT_COMPLIANCE)


def capabilities_from_tools(tools: Sequence[Any]) -> dict[str, bool]:
    """从装配的工具集导出各维度是否可测：``measured_dimensions`` / ``measurement_available`` 由工具声明，任一工具可测即为可测。"""
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
    """一次判定的事实侧输入汇总。"""

    required: tuple[str, ...]
    covered: frozenset[str]
    missing: tuple[str, ...]  # required ∧ 未覆盖 ∧ 本环境可测 → NOT_MEASURED
    unmeasurable: tuple[str, ...]  # required ∧ 未覆盖 ∧ 本环境不可测 → UNMEASURABLE
    positive: Mapping[str, list[Evidence]]
    capabilities: Mapping[str, bool]


def coverage_report(
    case: Any,
    evidence: Iterable[Evidence],
    capabilities: Mapping[str, bool] | None = None,
) -> CoverageReport:
    """构造覆盖报告；``capabilities=None`` 视为全部可测。"""
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
    """平台规则层命中的 rule_id 集合；``case`` 缺失返回空集。"""
    if case is None:
        return frozenset()
    from pra.screening.engine import triage  # 延迟 import

    return frozenset(hit.rule_id for hit in triage(_coerce_case(case)).hits)
