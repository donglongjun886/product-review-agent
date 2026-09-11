"""收敛判定 ``is_converged``（guardrails/converge.py）单测。

本次语义重构后：收敛由**事实侧的 required 覆盖**决定，**完全不看 hypothesis 状态**。

- 还有"本环境可测却没测"的维度 → 未收敛（继续回环补测）；
- 已确定 UNMEASURABLE 的维度**不算缺口**（重跑也测不出，继续循环只是烧预算）；
- 已出现阳性时，必须已有可引用依据（除非该维度在本环境不可测）。
"""

from __future__ import annotations

import pytest
from helpers import (
    all_measureable_caps,
    covered_evidence,
    ev,
    hp,
    make_case,
)

from pra.agent.guardrails.converge import is_converged
from pra.domain.measurement import DIM_IMAGE_APPEARANCE, DIM_POLICY_CITATION
from pra.domain.models import HypothesisStatus


def _state(*, evidence=None, hypotheses=None, caps=None) -> dict:
    return {
        "case": make_case(),
        "hypotheses": hypotheses or [],
        "evidence": evidence if evidence is not None else [],
        "measurement_capabilities": all_measureable_caps() if caps is None else caps,
    }


def test_converged_when_all_required_covered():
    assert is_converged(_state(evidence=covered_evidence())) is True


def test_not_converged_when_required_measurement_missing():
    """外观维度本可测却没测（N1）→ 继续回环补测。"""
    evs = [
        e for e in covered_evidence()
        if not (e.type == "MEASUREMENT" and e.ref_id.startswith(DIM_IMAGE_APPEARANCE))
    ]
    assert is_converged(_state(evidence=evs)) is False


def test_converged_when_required_dimension_is_unmeasurable():
    """环境测不了 ⇒ 不算缺口，不必再循环。"""
    caps = all_measureable_caps()
    caps[DIM_IMAGE_APPEARANCE] = False
    evs = [
        e for e in covered_evidence()
        if not (e.type == "MEASUREMENT" and e.ref_id.startswith(DIM_IMAGE_APPEARANCE))
    ]
    assert is_converged(_state(evidence=evs, caps=caps)) is True


def test_converged_when_positive_has_citable():
    evs = [
        *covered_evidence(similarity=0.93),
        ev("POLICY_REF", value="POLICY_3.2 v2 条款：x", weight=0.9, ref_id="POLICY_3.2_v2_c1"),
    ]
    assert is_converged(_state(evidence=evs)) is True


def test_not_converged_when_positive_without_citable_but_citation_measurable():
    assert is_converged(_state(evidence=covered_evidence(similarity=0.93))) is False


def test_converged_when_positive_and_citation_unmeasurable():
    caps = all_measureable_caps()
    caps[DIM_POLICY_CITATION] = False
    evs = covered_evidence(similarity=0.93)
    assert is_converged(_state(evidence=evs, caps=caps)) is True


def test_citable_without_ref_id_does_not_count():
    evs = [
        *covered_evidence(similarity=0.93),
        ev("POLICY_REF", value="POLICY_3.2 v2 条款：x", weight=0.9, ref_id=None),
    ]
    assert is_converged(_state(evidence=evs)) is False


@pytest.mark.parametrize(
    "status",
    [
        HypothesisStatus.PENDING,
        HypothesisStatus.UNRESOLVED,
        HypothesisStatus.SUPPORTED,
        HypothesisStatus.REFUTED,
    ],
)
def test_hypothesis_status_no_longer_affects_convergence(status):
    """收敛不再看假设状态：四种状态结果一致（旧链路里 PENDING/UNRESOLVED 会阻塞）。"""
    hyps = [hp("H1", prior=0.9, posterior=0.9, status=status, evidence_for=["x"])]
    assert is_converged(_state(evidence=covered_evidence(), hypotheses=hyps)) is True


def test_empty_state_converged_without_raise():
    """无 case ⇒ required 为空 ⇒ 收敛（但 gate 侧另有"无 case 不得 PASS"守卫）。"""
    assert is_converged({"hypotheses": [], "evidence": []}) is True
