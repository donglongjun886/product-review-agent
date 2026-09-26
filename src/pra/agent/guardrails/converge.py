"""收敛判定：调查是否已达成"可以交给 decide"的事实状态。"""

from __future__ import annotations

from typing import Any

from pra.agent.guardrails.measurements import coverage_report
from pra.domain.measurement import CITABLE_TYPES, DIM_POLICY_CITATION


def is_converged(state: dict[str, Any]) -> bool:
    """是否收敛：required 维度无缺口，且出现阳性时已有可引用依据或该维度不可测。"""
    evidence = state.get("evidence") or []
    cov = coverage_report(
        state.get("case"),
        evidence,
        state.get("measurement_capabilities"),
    )
    if cov.missing:
        return False
    if cov.positive:
        has_citable = any(e.type in CITABLE_TYPES and e.ref_id for e in evidence)
        if not has_citable and cov.capabilities.get(DIM_POLICY_CITATION, False):
            return False
    return True
