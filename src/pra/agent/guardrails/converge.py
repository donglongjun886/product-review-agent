"""收敛判定：证据链是否已到可以交给 decide 的程度（由 route_after_reevaluate 消费）。

收敛 = ① 无 PENDING/UNRESOLVED 假设（**全部假设，不过滤 prior** —— 低 prior
假设若被过滤掉会提前 decide）∧ ② 不存在「SUPPORTED 但证据链无任何带 ref_id 的
可引用依据」（可引用依据是**证据级**判断）。

本函数**不驱动**假设后验更新（那是 reevaluate LLM 的职责）。

预期结局：有 SUPPORTED 假设但检索不到政策/先例时，plan 经 dedup 清空重复动作或
无新工具可查而 ``conclude`` → 路由 decide → REJECT Gate 因无可引用依据不放行 →
HUMAN_REVIEW（新型风险），是预期克制行为而非缺陷。
"""

from __future__ import annotations

from typing import Any

from pra.domain.models import HypothesisStatus

# 可引用依据类型（REJECT/HUMAN 判定的 citable 集合）
CITABLE_TYPES = {"CASE_PRECEDENT", "POLICY_REF"}


def is_converged(state: dict[str, Any]) -> bool:
    """是否收敛（纯确定性，可单测）。

    ① 无 PENDING/UNRESOLVED 假设（含低 prior —— 不过滤 prior）；② 不存在「SUPPORTED
    假设但证据链无任何带 ref_id 的 POLICY_REF/CASE_PRECEDENT」。
    """
    hypotheses = state.get("hypotheses") or []
    evidence = state.get("evidence") or []
    open_hp = any(
        h.status in {HypothesisStatus.PENDING, HypothesisStatus.UNRESOLVED}
        for h in hypotheses
    )
    supported_wo_citation = (
        any(h.status == HypothesisStatus.SUPPORTED for h in hypotheses)
        and not any(e.type in CITABLE_TYPES and e.ref_id for e in evidence)
    )
    return not open_hp and not supported_wo_citation
