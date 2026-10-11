"""过程断言 seam：无证据引用的拒绝 / 应触发的 Gate 未触发 / 超预算 / 引用不可解析 / 静默降级。

各断言命中与正常案不命中（docs/02 §6 过程断言；引用解析直接读政策库/先例库 JSON）。
臂范围：REJECT_WITHOUT_CITATION / GATE_NOT_TRIGGERED 只评 agent 臂
（rule 臂的可引用依据是规则命中、single 臂无证据通道是基线设计）；
BUDGET_EXCEEDED / CITATION_UNRESOLVABLE / SILENT_DEGRADATION 三臂皆评。
"""

from __future__ import annotations

from evaluation.dataset.schema import (
    CaseManifestV3,
    EvalExpectedV3,
    EvalTruthV3,
    MerchantFactsV3,
    RegistryFactsV3,
)
from evaluation.labeling.labeler import load_rulebook
from evaluation.metrics.assertions import (
    BUDGET_EXCEEDED,
    CITATION_UNRESOLVABLE,
    GATE_NOT_TRIGGERED,
    REJECT_WITHOUT_CITATION,
    SILENT_DEGRADATION,
    check_process_assertions,
    load_citation_index,
)
from evaluation.record import EvalRecord
from helpers import REPO_ROOT

CORPUS_DIR = REPO_ROOT / "src" / "pra" / "rag" / "corpus"

_MANIFEST = {
    "text_hits": [],
    "snapshot_brand": "潮动",
    "snapshot_gaps": [],
    "registry_mismatch": False,
    "merchant": MerchantFactsV3(
        merchant_id="M_T1", removals=0, title_relisting_count=0,
        similar_product_count=0, credit_score=90,
    ),
    "registry": RegistryFactsV3(
        product_id="P_T1", category="女鞋/运动鞋", brand="潮动", version=1, status="ON_SALE",
    ),
}

_EXPECTED_PASS = {
    "decision": "PASS", "abstain_label": "AUTO_DECIDABLE",
    "risk_type": [], "expected_tools": [], "expected_policies": [],
}


def _truth(case_id: str, *, snapshot_brand: str | None = "潮动") -> EvalTruthV3:
    return EvalTruthV3(
        eval_case_id=case_id,
        manifest=CaseManifestV3(**{**_MANIFEST, "snapshot_brand": snapshot_brand}),
        expected=EvalExpectedV3(**_EXPECTED_PASS),
    )


def _record(case_id: str, **over) -> EvalRecord:
    base = {
        "eval_case_id": case_id,
        "scheme": "agent",
        "decision": "PASS",
        "evidence": [{"type": "PRODUCT_FACT", "ref_id": "P_1"}],
        "policy": [],
        "cost": {"llm_calls": 2, "tool_calls": 1, "tokens": 100},
        "detail": {},
    }
    base.update(over)
    return EvalRecord(**base)


def _check(records: list[EvalRecord], truth: list[EvalTruthV3], *, scheme: str = "agent"):
    return check_process_assertions(
        records,
        truth,
        scheme=scheme,
        citations=load_citation_index(CORPUS_DIR),
        blacklisted_brands=load_rulebook()["blacklisted_brands"],
        max_llm_calls=10,
        max_tool_calls=15,
    )


def _codes(failures, case_id: str) -> set[str]:
    return {f.code for f in failures if f.eval_case_id == case_id}


class TestRejectWithoutCitation:
    """只评 agent 臂：rule 臂的可引用依据是规则命中，single 臂无证据通道是基线设计。"""

    def test_reject_with_empty_policy_and_no_citable_evidence_fails(self) -> None:
        failures = _check([_record("a1", decision="REJECT")], [_truth("a1")])
        assert REJECT_WITHOUT_CITATION in _codes(failures, "a1")

    def test_reject_with_policy_but_no_citable_evidence_fails(self) -> None:
        """policy 非空但证据链没有可引用依据（无 ref_id / rule_id）→ 仍命中。"""
        rec = _record(
            "a2",
            decision="REJECT",
            policy=["POLICY_1.1"],
            evidence=[{"type": "MERCHANT_HISTORY", "value": "removals=5"}],
        )
        assert REJECT_WITHOUT_CITATION in _codes(_check([rec], [_truth("a2")]), "a2")

    def test_reject_with_citable_evidence_but_empty_policy_fails(self) -> None:
        rec = _record(
            "a3",
            decision="REJECT",
            evidence=[{"type": "POLICY_REF", "ref_id": "POLICY_1.1_v2_c1"}],
        )
        assert REJECT_WITHOUT_CITATION in _codes(_check([rec], [_truth("a3")]), "a3")

    def test_reject_with_policy_and_citable_evidence_passes(self) -> None:
        rec = _record(
            "a4",
            decision="REJECT",
            policy=["POLICY_1.1"],
            evidence=[{"type": "POLICY_REF", "ref_id": "POLICY_1.1_v2_c1"}],
        )
        assert REJECT_WITHOUT_CITATION not in _codes(_check([rec], [_truth("a4")]), "a4")

    def test_rule_hit_counts_as_citable_evidence(self) -> None:
        """agent 臂 REJECT：规则命中的 rule_id 是可引用依据（还差 policy 才过断言）。"""
        rec = _record(
            "a5",
            decision="REJECT",
            policy=["POLICY_1.1"],
            evidence=[{"type": "RULE_HIT", "extra": {"rule_id": "R-101"}}],
        )
        assert REJECT_WITHOUT_CITATION not in _codes(_check([rec], [_truth("a5")]), "a5")

    def test_rule_arm_reject_not_checked(self) -> None:
        """rule 臂 REJECT 无可引用政策是口径设计（依据=规则命中），不评此断言。"""
        rec = _record("a6", scheme="rule", decision="REJECT", evidence=[{"type": "RULE_HIT", "extra": {"rule_id": "R-101"}}])
        failures = _check([rec], [_truth("a6")], scheme="rule")
        assert REJECT_WITHOUT_CITATION not in _codes(failures, "a6")

    def test_single_arm_reject_not_checked(self) -> None:
        """single 臂无证据通道是基线设计，不评此断言。"""
        rec = _record("a7", scheme="single", decision="REJECT")
        failures = _check([rec], [_truth("a7")], scheme="single")
        assert REJECT_WITHOUT_CITATION not in _codes(failures, "a7")

    def test_pass_never_triggers(self) -> None:
        assert _check([_record("a8")], [_truth("a8")]) == []


class TestGateNotTriggered:
    def test_blacklist_brand_without_r1_override_fails(self) -> None:
        rec = _record("g1", decision="REJECT", detail={"overrides": ["R2_REJECT_GATE_FAIL"]})
        failures = _check([rec], [_truth("g1", snapshot_brand="某违禁品牌")])
        assert GATE_NOT_TRIGGERED in _codes(failures, "g1")

    def test_blacklist_brand_with_r1_override_passes(self) -> None:
        rec = _record("g2", decision="REJECT", detail={"overrides": ["R1_HARD_RULE"]})
        failures = _check([rec], [_truth("g2", snapshot_brand="某违禁品牌")])
        assert GATE_NOT_TRIGGERED not in _codes(failures, "g2")

    def test_non_blacklist_brand_not_checked(self) -> None:
        rec = _record("g3", detail={"overrides": []})
        assert _check([rec], [_truth("g3", snapshot_brand="潮动")]) == []

    def test_only_agent_scheme_checked(self) -> None:
        """rule / single 臂无 overrides 通道，不适用 Gate 断言。"""
        rec = _record("g4", scheme="rule", decision="REJECT")
        failures = _check([rec], [_truth("g4", snapshot_brand="某违禁品牌")], scheme="rule")
        assert GATE_NOT_TRIGGERED not in _codes(failures, "g4")


class TestSilentDegradation:
    """无静默降级（docs/02 §6）：auto 决策带 error，或 agent 臂 PASS 但证据链为空。"""

    def test_auto_decision_with_error_detail_fails(self) -> None:
        rec = _record("s1", decision="PASS", detail={"error": "litellm timeout", "attempts": 2})
        assert SILENT_DEGRADATION in _codes(_check([rec], [_truth("s1")]), "s1")

    def test_reject_with_error_detail_fails(self) -> None:
        """REJECT 同为 auto 决策：detail 带 error 即命中（三臂皆评）。"""
        rec = _record("s2", scheme="single", decision="REJECT", detail={"error": "boom"})
        failures = _check([rec], [_truth("s2")], scheme="single")
        assert SILENT_DEGRADATION in _codes(failures, "s2")

    def test_human_review_with_error_is_abstention_not_degradation(self) -> None:
        """调用失败转人工是显式弃权（带 error 是记录不是冒充），不命中。"""
        rec = _record("s3", decision="HUMAN_REVIEW", detail={"error": "boom"})
        assert SILENT_DEGRADATION not in _codes(_check([rec], [_truth("s3")]), "s3")

    def test_agent_pass_with_empty_evidence_fails(self) -> None:
        """agent 臂 PASS 但证据链为空 → 空结果冒充正常，命中。"""
        rec = _record("s4", decision="PASS", evidence=[])
        assert SILENT_DEGRADATION in _codes(_check([rec], [_truth("s4")]), "s4")

    def test_agent_pass_with_evidence_passes(self) -> None:
        rec = _record("s5", decision="PASS", evidence=[{"type": "PRODUCT_FACT", "ref_id": "P_1"}])
        assert SILENT_DEGRADATION not in _codes(_check([rec], [_truth("s5")]), "s5")

    def test_single_pass_with_empty_evidence_not_checked(self) -> None:
        """single 臂无证据通道是基线设计，空证据链不命中。"""
        rec = _record("s6", scheme="single", decision="PASS")
        failures = _check([rec], [_truth("s6")], scheme="single")
        assert SILENT_DEGRADATION not in _codes(failures, "s6")


class TestBudgetKept:
    def test_llm_calls_over_limit_fails(self) -> None:
        rec = _record("b1", cost={"llm_calls": 11, "tool_calls": 0, "tokens": 0})
        assert BUDGET_EXCEEDED in _codes(_check([rec], [_truth("b1")]), "b1")

    def test_tool_calls_over_limit_fails(self) -> None:
        rec = _record("b2", cost={"llm_calls": 0, "tool_calls": 16, "tokens": 0})
        assert BUDGET_EXCEEDED in _codes(_check([rec], [_truth("b2")]), "b2")

    def test_at_limit_passes(self) -> None:
        """贴着上限（==）不算超限运行：超限才违反预算守住。"""
        rec = _record("b3", cost={"llm_calls": 10, "tool_calls": 15, "tokens": 99999})
        assert BUDGET_EXCEEDED not in _codes(_check([rec], [_truth("b3")]), "b3")


class TestCitationResolvable:
    def test_unknown_policy_id_fails(self) -> None:
        rec = _record("c1", policy=["POLICY_9.9"])
        assert CITATION_UNRESOLVABLE in _codes(_check([rec], [_truth("c1")]), "c1")

    def test_policy_id_and_clause_id_both_resolve(self) -> None:
        rec = _record("c2", policy=["POLICY_1.1", "POLICY_1.1_v2_c1"])
        assert _check([rec], [_truth("c2")]) == []

    def test_unknown_precedent_ref_fails(self) -> None:
        rec = _record(
            "c3",
            evidence=[{"type": "CASE_PRECEDENT", "ref_id": "case_9999"}],
        )
        assert CITATION_UNRESOLVABLE in _codes(_check([rec], [_truth("c3")]), "c3")

    def test_known_precedent_ref_passes(self) -> None:
        rec = _record(
            "c4",
            evidence=[{"type": "CASE_PRECEDENT", "ref_id": "RAG_CASE_0001"}],
        )
        assert _check([rec], [_truth("c4")]) == []
