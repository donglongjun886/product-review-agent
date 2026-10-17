"""收敛判定：调查是否已达成"可以交给 decide"的事实状态。"""

from __future__ import annotations

from typing import Any

from pra.agent.guardrails.measurements import coverage_report


def is_converged(state: dict[str, Any]) -> bool:
    """是否收敛：required 维度无可补测缺口（存在阳性时 ``policy_citation`` 同为必需，见 ``required_dimensions``）。"""
    cov = coverage_report(
        state.get("case"),
        state.get("evidence") or [],
        state.get("measurement_capabilities"),
    )
    return not cov.missing
