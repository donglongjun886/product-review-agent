"""收敛判定 ``is_converged``（guardrails/converge.py）单测。

① 全定论（无 PENDING/UNRESOLVED）∧ ② SUPPORTED 均有可引用依据 → True；任
PENDING/UNRESOLVED → False；SUPPORTED 无带 ``ref_id`` 的 POLICY_REF/CASE_PRECEDENT
→ False。**不过滤 prior**：低 prior 的 PENDING 也算未收敛。
"""

from __future__ import annotations

from pra.agent.guardrails.converge import is_converged
from pra.domain.models import HypothesisStatus
from helpers import ev, hp


def _state(hypotheses=None, evidence=None) -> dict:
    return {"hypotheses": hypotheses or [], "evidence": evidence or []}


def test_converged_all_refuted_no_supported():
    hs = [
        hp("H1", prior=0.5, status=HypothesisStatus.REFUTED, evidence_against=["e"]),
        hp("H2", prior=0.4, status=HypothesisStatus.REFUTED, evidence_against=["e"]),
    ]
    assert is_converged(_state(hs, [])) is True


def test_converged_supported_with_citable():
    hs = [hp("H1", prior=0.4, status=HypothesisStatus.SUPPORTED, posterior=0.91,
             evidence_for=["x"])]
    evs = [ev("POLICY_REF", value="POLICY_3.2 v2 条款：x", weight=0.9, ref_id="POLICY_3.2_v2_c1")]
    assert is_converged(_state(hs, evs)) is True


def test_not_converged_pending_hypothesis():
    hs = [
        hp("H1", prior=0.4, status=HypothesisStatus.SUPPORTED, posterior=0.9, evidence_for=["x"]),
        hp("H2", prior=0.4, status=HypothesisStatus.PENDING),
    ]
    evs = [ev("POLICY_REF", value="p", weight=0.9, ref_id="c1")]
    assert is_converged(_state(hs, evs)) is False


def test_not_converged_unresolved_hypothesis():
    hs = [hp("H1", prior=0.5, status=HypothesisStatus.UNRESOLVED)]
    assert is_converged(_state(hs, [])) is False


def test_not_converged_supported_without_citable():
    hs = [hp("H1", prior=0.5, status=HypothesisStatus.SUPPORTED, posterior=0.91,
             evidence_for=["x"])]
    evs = [ev("IMAGE_SIMILARITY", value="similarity=0.91", weight=0.91, ref_id="img1")]
    assert is_converged(_state(hs, evs)) is False


def test_supported_with_case_precedent_ref_is_converged():
    hs = [hp("H1", prior=0.5, status=HypothesisStatus.SUPPORTED, posterior=0.9,
             evidence_for=["x"])]
    evs = [ev("CASE_PRECEDENT", value="case_1001", weight=0.8, ref_id="case_1001")]
    assert is_converged(_state(hs, evs)) is True


def test_citable_without_ref_id_does_not_count():
    hs = [hp("H1", prior=0.5, status=HypothesisStatus.SUPPORTED, posterior=0.9,
             evidence_for=["x"])]
    evs = [ev("POLICY_REF", value="POLICY_3.2", weight=0.9, ref_id=None)]
    assert is_converged(_state(hs, evs)) is False


def test_low_prior_pending_still_not_converged():
    """不过滤 prior：低 prior（0.15）PENDING 也算未收敛，不得因 prior 低提前 decide。"""
    hs = [hp("H3", prior=0.15, status=HypothesisStatus.PENDING)]
    assert is_converged(_state(hs, [])) is False


def test_empty_hypotheses_evidence_vacuous_true_no_raise():
    """空 hypotheses/evidence 不抛；无假设时收敛条件为空真（交给 decide 兜底）。"""
    assert is_converged({}) is True
    assert is_converged(_state([], [])) is True
