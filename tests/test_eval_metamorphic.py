"""蜕变测试与 pass^k seam：扰动确定性、裁决翻转/变松判定、重复运行一致率。

保语义扰动不得改变初筛规则命中（用 pra.screening 验证）；真实模型运行不在此做
（CLI 接线 + 此处守护逻辑）。
"""

from __future__ import annotations

import pytest
from evaluation.dataset.schema import EvalCaseV3, EvalConstructionV3
from evaluation.metamorphic import (
    check_metamorphic,
    pass_k_consistency,
    perturb_risk_amplifying,
    perturb_semantics_preserving,
    strictness,
)
from evaluation.record import EvalRecord
from helpers import evasion_case, make_case

from pra.screening.engine import triage


def _case(case_id: str = "CASE_M1", **kwargs) -> EvalCaseV3:
    return EvalCaseV3(
        eval_case_id="EC_M1",
        scene="normal",
        construction=EvalConstructionV3(pattern="clean", params={}),
        input=make_case(case_id=case_id, **kwargs),
    )


def _record(case_id: str, decision: str) -> EvalRecord:
    return EvalRecord(eval_case_id=case_id, scheme="agent", decision=decision)  # type: ignore[arg-type]


class TestPerturbationDeterminism:
    def test_same_input_same_output(self) -> None:
        case = _case()
        assert perturb_semantics_preserving(case) == perturb_semantics_preserving(case)
        assert perturb_risk_amplifying(case) == perturb_risk_amplifying(case)

    def test_neutral_perturbation_keeps_rule_hits_on_clean_case(self) -> None:
        """保语义扰动（价格/SKU/上架时间/图片地址）不动语义：初筛规则命中不变。"""
        case = _case()
        before = [(h.rule_id, h.detail) for h in triage(case.input).hits]
        after = [(h.rule_id, h.detail) for h in triage(perturb_semantics_preserving(case).input).hits]
        assert before == after

    def test_neutral_perturbation_keeps_rule_hits_on_evasion_case(self) -> None:
        case = EvalCaseV3(
            eval_case_id="EC_M2",
            scene="evasion",
            construction=EvalConstructionV3(pattern="evasion", params={}),
            input=evasion_case(),
        )
        before = [h.rule_id for h in triage(case.input).hits]
        after = [h.rule_id for h in triage(perturb_semantics_preserving(case).input).hits]
        assert before == after
        assert before  # 规避案原本就有命中，防止空通过

    def test_risk_amplifying_only_adds_hits(self) -> None:
        """风险增强扰动（标题植入规避词）：原有命中保留，且新增 R-302 规避词命中。"""
        case = _case()
        before = {h.rule_id for h in triage(case.input).hits}
        after = {h.rule_id for h in triage(perturb_risk_amplifying(case).input).hits}
        assert before <= after
        assert "R-302" in after - before

    def test_perturbations_do_not_touch_semantic_fields(self) -> None:
        case = _case()
        neutral = perturb_semantics_preserving(case)
        assert neutral.input.product.title == case.input.product.title
        assert neutral.input.product.description == case.input.product.description
        assert neutral.input.product.brand == case.input.product.brand
        assert neutral.input.product.category == case.input.product.category
        assert neutral.input.product.sku_list[0].price != case.input.product.sku_list[0].price
        assert neutral.input.product.listing_time != case.input.product.listing_time


class TestStrictnessOrder:
    def test_ordering(self) -> None:
        assert strictness("REJECT") > strictness("HUMAN_REVIEW") > strictness("PASS")


class TestCheckMetamorphic:
    def test_stable_and_stricter_pass(self) -> None:
        """保语义不翻转 + 风险增强不变松 → 无失败。"""
        failures = check_metamorphic(
            _record("m1", "HUMAN_REVIEW"),
            _record("m1", "HUMAN_REVIEW"),
            _record("m1", "REJECT"),
        )
        assert failures == []

    def test_neutral_flip_fails(self) -> None:
        failures = check_metamorphic(
            _record("m2", "PASS"),
            _record("m2", "REJECT"),
            _record("m2", "REJECT"),
        )
        assert [f.kind for f in failures] == ["NEUTRAL_FLIP"]
        assert failures[0].original == "PASS" and failures[0].perturbed == "REJECT"

    def test_risk_amplified_loosened_fails(self) -> None:
        """风险增强后裁决变松（REJECT→HUMAN_REVIEW / HUMAN_REVIEW→PASS / REJECT→PASS）。"""
        for original, perturbed in (("REJECT", "HUMAN_REVIEW"), ("HUMAN_REVIEW", "PASS"), ("REJECT", "PASS")):
            failures = check_metamorphic(
                _record("m3", original),
                _record("m3", original),
                _record("m3", perturbed),
            )
            assert [f.kind for f in failures] == ["RISK_AMPLIFIED_LOOSENED"], (original, perturbed)

    def test_both_failures_can_coexist(self) -> None:
        failures = check_metamorphic(
            _record("m4", "REJECT"),
            _record("m4", "PASS"),
            _record("m4", "PASS"),
        )
        assert {f.kind for f in failures} == {"NEUTRAL_FLIP", "RISK_AMPLIFIED_LOOSENED"}


class TestPassKConsistency:
    def test_hand_computed(self) -> None:
        """手算：c1 三次一致 / c2 三次不一致 / c3 只跑到一次不参与比对。"""
        metrics = pass_k_consistency(
            {
                "c1": ["PASS", "PASS", "PASS"],
                "c2": ["PASS", "REJECT", "PASS"],
                "c3": ["PASS"],
            }
        )
        assert metrics.compared_cases == 2
        assert metrics.consistent_cases == 1
        assert metrics.consistency_rate == pytest.approx(1 / 2)

    def test_no_repeated_case_is_none(self) -> None:
        metrics = pass_k_consistency({"c1": ["PASS"], "c2": ["REJECT"]})
        assert metrics.compared_cases == 0
        assert metrics.consistency_rate is None

    def test_human_review_counts_as_a_decision(self) -> None:
        """HUMAN_REVIEW 是一致性比对的有效裁决值（一致就算一致）。"""
        metrics = pass_k_consistency({"c1": ["HUMAN_REVIEW", "HUMAN_REVIEW"]})
        assert metrics.consistency_rate == 1.0
