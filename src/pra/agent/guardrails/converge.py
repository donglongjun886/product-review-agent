"""收敛判定：调查是否已达成"可以交给 decide"的事实状态（由 route_after_reevaluate 消费）。

收敛 = 事实侧目标达成，**完全不看 hypothesis 状态**：

① 本案 required 维度**没有可补救的缺口**（``missing`` 为空）——
   还有"本环境可测却没测"的维度 → 不收敛，继续回环去补测；
   已确定 ``UNMEASURABLE`` 的维度**不算缺口**（重跑也测不出，继续循环只是烧预算）；
② 若证据链已出现阳性：必须已有可引用依据（带 ``ref_id`` 的 POLICY_REF/CASE_PRECEDENT），
   或该维度在本环境不可测 —— 否则继续回环找政策/先例（REJECT 的必要条件）。

本函数**不驱动**假设后验更新（那是 reevaluate LLM 的职责），也不读 ``prior``/``status``。

预期结局：有阳性但检索不到政策/先例时，plan 经 dedup 清空重复动作或无新工具可查而
``conclude`` → 路由 decide → REJECT Gate 因无可引用依据不放行 → HUMAN_REVIEW
（``R3_POSITIVE_INSUFFICIENT``），是预期克制行为而非缺陷。
"""

from __future__ import annotations

from typing import Any

from pra.agent.guardrails.measurements import coverage_report
from pra.domain.measurement import DIM_POLICY_CITATION

# 可引用依据类型（与 gate.py 的 CITABLE_TYPES 同义，本地重声明）
CITABLE_TYPES = {"CASE_PRECEDENT", "POLICY_REF"}


def is_converged(state: dict[str, Any]) -> bool:
    """是否收敛（纯确定性，可单测）。判据见模块 docstring。"""
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
