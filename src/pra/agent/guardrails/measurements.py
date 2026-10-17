"""维度覆盖判定：按 ``required × 证据存在性 × 环境能力`` 推导各维度三态。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pra.domain.measurement import (
    ALL_DIMENSIONS,
    CITABLE_TYPES,
    DIM_LISTING_REGISTRY,
    DIM_MERCHANT_PROFILE,
    DIM_POLICY_CITATION,
    DIM_TEXT_COMPLIANCE,
    MEASUREMENT_TYPE,
    VERDICT_POSITIVE,
    VERDICTS,
    make_measurement,
    measurement_dimension,
    measurement_verdict,
)
from pra.domain.models import Evidence, ProductReviewCase

__all__ = [
    "ALWAYS_COVERED_DIMENSIONS",
    "RULE_BRAND_WORD",
    "RULE_EVASION_WORD",
    "CoverageReport",
    "capabilities_from_tools",
    "coverage_gap_lines",
    "coverage_report",
    "has_citable_evidence",
    "positive_dimensions",
    "required_dimensions",
    "rule_hit_ids",
    "rule_hit_measurement",
]

ALWAYS_COVERED_DIMENSIONS: frozenset[str] = frozenset({DIM_TEXT_COMPLIANCE})

# 平台规则层档位（R-102 品牌词 / R-302 规避词）
RULE_BRAND_WORD = "R-102"
RULE_EVASION_WORD = "R-302"

_CITATION_TOOLS: frozenset[str] = frozenset({"CaseSearchTool", "PolicySearchTool"})


def has_citable_evidence(evidence: Iterable[Evidence]) -> bool:
    """是否存在带 ``ref_id`` 的可引用依据（POLICY_REF / CASE_PRECEDENT）。"""
    return any(e.type in CITABLE_TYPES and e.ref_id for e in evidence)


def positive_dimensions(evidence: Iterable[Evidence]) -> dict[str, list[Evidence]]:
    """阳性测量证据按维度归类：``MEASUREMENT`` 且 verdict 阳性 → 该测量维度（原始统计不直接构成阳性）。"""
    out: dict[str, list[Evidence]] = defaultdict(list)
    for e in evidence or []:
        if e.type == MEASUREMENT_TYPE:
            dim = measurement_dimension(e)
            if measurement_verdict(e) == VERDICT_POSITIVE and dim:
                out[dim].append(e)
    return dict(out)


def required_dimensions(
    case: ProductReviewCase | None, evidence: Iterable[Evidence]
) -> tuple[str, ...]:
    """本案必需的测量维度；``case`` 缺失返回空元组。存在阳性证据时追加 ``policy_citation``（自动拒绝必须可引用）。"""
    if case is None:
        return ()
    required = [DIM_LISTING_REGISTRY, DIM_MERCHANT_PROFILE, DIM_TEXT_COMPLIANCE]
    if positive_dimensions(evidence):
        required.append(DIM_POLICY_CITATION)
    return tuple(required)


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
    case: ProductReviewCase | None,
    evidence: Iterable[Evidence],
    capabilities: Mapping[str, bool] | None = None,
) -> CoverageReport:
    """构造覆盖报告；``capabilities=None`` 视为全部可测。"""
    evs = list(evidence or [])
    caps: Mapping[str, bool] = capabilities if capabilities is not None else {
        dim: True for dim in ALL_DIMENSIONS
    }
    positive = positive_dimensions(evs)
    required = required_dimensions(case, evs)
    covered: set[str] = set(ALWAYS_COVERED_DIMENSIONS)
    covered |= set(positive)
    for e in evs:
        if e.type == MEASUREMENT_TYPE and measurement_verdict(e) in VERDICTS:
            dim = measurement_dimension(e)
            if dim:
                covered.add(dim)
    if has_citable_evidence(evs):
        covered.add(DIM_POLICY_CITATION)
    missing = tuple(d for d in required if d not in covered and caps.get(d, True))
    unmeasurable = tuple(d for d in required if d not in covered and not caps.get(d, True))
    return CoverageReport(
        required=required,
        covered=frozenset(covered),
        missing=missing,
        unmeasurable=unmeasurable,
        positive=positive,
        capabilities=caps,
    )


def coverage_gap_lines(state: dict) -> list[str]:
    """本案必需测量的覆盖清单（人读行，注入 plan / decide 上下文）。"""
    evidence = list(state.get("evidence") or [])
    cov = coverage_report(
        state.get("case"),
        evidence,
        state.get("measurement_capabilities"),
    )
    covered = [d for d in cov.required if d in cov.covered]
    lines = [f"- 必需测量覆盖：{len(covered)}/{len(cov.required)}"]
    for dim in cov.required:
        if dim in cov.covered:
            if dim == DIM_POLICY_CITATION:
                lines.append(f"  - {dim}：已取得可引用依据")
            else:
                verdict = "阳性" if dim in cov.positive else "阴性"
                lines.append(f"  - {dim}：已测（{verdict}）")
        elif dim in cov.unmeasurable:
            lines.append(f"  - {dim}：本环境不可测（不要再安排该类工具，重跑无用）")
        else:
            lines.append(f"  - {dim}：**尚未取得** —— 优先安排能补齐它的工具")
    return lines


def rule_hit_ids(case: ProductReviewCase | None) -> frozenset[str]:
    """平台规则层命中的 rule_id 集合；``case`` 缺失返回空集。"""
    if case is None:
        return frozenset()
    from pra.screening.engine import triage  # 延迟 import

    return frozenset(hit.rule_id for hit in triage(case).hits)


def rule_hit_measurement(case: ProductReviewCase | None) -> Evidence | None:
    """品牌词 / 规避词（R-102 / R-302）命中 → ``text_compliance`` 阳性测量证据（value 含命中词与规则明细）；无命中或 ``case`` 缺失返回 None。"""
    if case is None:
        return None
    from pra.screening.engine import (  # 延迟 import
        RULE_EVIDENCE_SOURCE,
        RULE_EVIDENCE_WEIGHT,
        triage,
    )

    hits = [
        h for h in triage(case).hits if h.rule_id in (RULE_BRAND_WORD, RULE_EVASION_WORD)
    ]
    if not hits:
        return None
    detail = "；".join(f"{h.rule_id} {h.name}: {h.detail}" for h in hits)
    return make_measurement(
        dimension=DIM_TEXT_COMPLIANCE,
        source=RULE_EVIDENCE_SOURCE,
        source_ref=case.product.product_id,
        verdict=VERDICT_POSITIVE,
        weight=RULE_EVIDENCE_WEIGHT,
        value=f"规则层文本扫描命中：{detail}",
        extra={"rule_ids": [h.rule_id for h in hits]},
    )
