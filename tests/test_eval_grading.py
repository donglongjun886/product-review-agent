"""判卷器 seam：EvalRecord × EvalExpectedV3 纯代码逐格比对；判卷路径零 LLM（静态守护）。"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from evaluation.dataset.schema import EvalExpectedV3, EvalTruthV3
from evaluation.grading.grader import CaseGrading, grade_case, grade_records
from evaluation.record import EvalRecord

GRADING_DIR = Path(__file__).resolve().parents[1] / "scripts" / "evaluation" / "grading"


def _record(**over) -> EvalRecord:
    """造一条被判记录（默认全对 PASS 案的预测）。"""
    base = {
        "eval_case_id": "EC_G1",
        "scheme": "agent",
        "decision": "PASS",
        "risk_level": "NONE",
        "risk_type": [],
        "policy": [],
        "tool_calls_actual": [],
    }
    base.update(over)
    return EvalRecord(**base)


def _expected(**over) -> EvalExpectedV3:
    """造一份 v3 真值（默认干净 PASS 案）。"""
    base = {
        "decision": "PASS",
        "abstain_label": "AUTO_DECIDABLE",
        "risk_type": [],
        "expected_evidence": [],
        "expected_tools": [],
        "expected_policies": [],
    }
    base.update(over)
    return EvalExpectedV3(**base)


def _reject_expected(**over) -> EvalExpectedV3:
    """REJECT 真值：按契约必须带可引用依据。"""
    base = {
        "decision": "REJECT",
        "abstain_label": "AUTO_DECIDABLE",
        "risk_type": ["POTENTIAL_IP_RISK"],
        "expected_evidence": ["RULE_HIT", "POLICY_REF"],
        "expected_tools": ["PolicySearchTool"],
        "expected_policies": ["POLICY_1.1"],
    }
    base.update(over)
    return EvalExpectedV3(**base)


class TestGradeCase:
    """逐项比对：全对 / decision 错 / risk_type 错 / 工具多调少调 / 政策缺引。"""

    def test_all_correct(self) -> None:
        expected = _reject_expected()
        record = _record(
            decision="REJECT",
            risk_level="HIGH",
            risk_type=["POTENTIAL_IP_RISK"],
            evidence=[
                {"type": "RULE_HIT", "value": "R-302 命中"},
                {"type": "POLICY_REF", "ref_id": "POLICY_1.1_v2_c1"},
            ],
            policy=["POLICY_1.1"],
            tool_calls_actual=["PolicySearchTool"],
        )
        g = grade_case(record, expected)
        assert g.decision_match is True
        assert g.risk_type_match is True
        assert g.evidence_match is True
        assert g.tools_match is True
        assert g.tools_missing == [] and g.tools_extra == []
        assert g.policies_match is True
        assert g.policies_missing == [] and g.policies_extra == []

    def test_truth_fields_carried_for_metrics_layer(self) -> None:
        """判卷结果携带真值侧 decision（票 04 指标层聚合用；abstain 可由它推导）。"""
        g = grade_case(_record(decision="HUMAN_REVIEW"), _expected(decision="HUMAN_REVIEW", abstain_label="SHOULD_ABSTAIN"))
        assert g.truth_decision == "HUMAN_REVIEW"
        assert g.predicted_decision == "HUMAN_REVIEW"
        assert g.scheme == "agent"

    def test_decision_mismatch(self) -> None:
        g = grade_case(_record(decision="PASS"), _reject_expected())
        assert g.decision_match is False
        assert g.truth_decision == "REJECT"
        assert g.predicted_decision == "PASS"

    def test_risk_type_mismatch(self) -> None:
        g = grade_case(_record(risk_type=["EVASION_PATTERN"]), _reject_expected())
        assert g.risk_type_match is False
        assert g.truth_risk_type == ["POTENTIAL_IP_RISK"]
        assert g.predicted_risk_type == ["EVASION_PATTERN"]

    def test_risk_type_set_semantics(self) -> None:
        """risk_type 按集合比对：顺序不同仍一致。"""
        expected = _reject_expected(risk_type=["POTENTIAL_IP_RISK", "EVASION_PATTERN"])
        g = grade_case(_record(risk_type=["EVASION_PATTERN", "POTENTIAL_IP_RISK"]), expected)
        assert g.risk_type_match is True

    def test_risk_type_unlabeled_is_none(self) -> None:
        """真值 risk_type 为空 → 不进比对（None），即使预测非空。"""
        g = grade_case(_record(risk_type=["POTENTIAL_IP_RISK"]), _expected())
        assert g.risk_type_match is None

    def test_evidence_match_set_semantics(self) -> None:
        """证据类型按集合比对：顺序不同仍一致；actual 取记录证据链的 type 去重保序。"""
        expected = _reject_expected()
        record = _record(
            evidence=[
                {"type": "POLICY_REF", "ref_id": "POLICY_1.1_v2_c1"},
                {"type": "RULE_HIT", "value": "R-302 命中"},
                {"type": "RULE_HIT", "value": "R-101 命中"},
            ]
        )
        g = grade_case(record, expected)
        assert g.actual_evidence == ["POLICY_REF", "RULE_HIT"]
        assert g.evidence_match is True
        assert g.evidence_missing == [] and g.evidence_extra == []

    def test_evidence_missing_and_extra(self) -> None:
        """应有证据缺失 / 出现期望外证据类型 → 差异列表按保序集合差给出。"""
        expected = _reject_expected(expected_evidence=["RULE_HIT", "POLICY_REF", "MERCHANT_HISTORY"])
        record = _record(evidence=[{"type": "RULE_HIT"}, {"type": "CASE_PRECEDENT", "ref_id": "c1"}])
        g = grade_case(record, expected)
        assert g.evidence_match is False
        assert g.evidence_missing == ["POLICY_REF", "MERCHANT_HISTORY"]
        assert g.evidence_extra == ["CASE_PRECEDENT"]

    def test_evidence_unlabeled_is_none(self) -> None:
        """期望证据为空 = 未标注 → 不进比对（None），实际证据照实记录。"""
        record = _record(evidence=[{"type": "PRODUCT_FACT", "ref_id": "P_1"}])
        g = grade_case(record, _expected())
        assert g.evidence_match is None
        assert g.evidence_missing == [] and g.evidence_extra == []
        assert g.actual_evidence == ["PRODUCT_FACT"]

    def test_tools_missing_and_extra(self) -> None:
        expected = _expected(expected_tools=["ProductTool", "MerchantTool"])
        g = grade_case(_record(tool_calls_actual=["ProductTool", "CaseSearchTool"]), expected)
        assert g.tools_match is False
        assert g.tools_missing == ["MerchantTool"]
        assert g.tools_extra == ["CaseSearchTool"]

    def test_tools_unlabeled_is_none(self) -> None:
        """expected_tools 空 = 规则层可判，不进工具分母（None），实际调用照实记录。"""
        g = grade_case(_record(tool_calls_actual=["ProductTool"]), _expected())
        assert g.tools_match is None
        assert g.tools_missing == [] and g.tools_extra == []
        assert g.actual_tools == ["ProductTool"]

    def test_policies_missing(self) -> None:
        expected = _reject_expected(expected_policies=["POLICY_1.1", "POLICY_4.5"])
        g = grade_case(_record(policy=["POLICY_1.1"]), expected)
        assert g.policies_match is False
        assert g.policies_missing == ["POLICY_4.5"]
        assert g.policies_extra == []

    def test_policies_unlabeled_is_none(self) -> None:
        g = grade_case(_record(), _expected())
        assert g.policies_match is None


class TestGradeRecords:
    """批量判卷：record 与真值按 eval_case_id 对齐，对不上显式失败。"""

    def _truth_row(self, eval_case_id: str, expected: EvalExpectedV3) -> EvalTruthV3:
        return EvalTruthV3.model_validate(
            {
                "eval_case_id": eval_case_id,
                "manifest": {
                    "text_hits": [],
                    "snapshot_brand": "潮动",
                    "snapshot_gaps": [],
                    "registry_mismatch": False,
                    "merchant": {
                        "merchant_id": "M_T1",
                        "removals": 0,
                        "title_relisting_count": 0,
                        "similar_product_count": 0,
                        "credit_score": 90,
                    },
                    "registry": {
                        "product_id": "P_T1",
                        "category": "女鞋/运动鞋",
                        "brand": "潮动",
                        "version": 1,
                        "status": "ON_SALE",
                    },
                },
                "expected": expected.model_dump(mode="json"),
            }
        )

    def test_grade_records_aligned(self) -> None:
        truth = [self._truth_row("EC_G1", _expected()), self._truth_row("EC_G2", _reject_expected())]
        records = [_record(eval_case_id="EC_G1"), _record(eval_case_id="EC_G2", decision="REJECT")]
        out = grade_records(records, truth)
        assert [g.eval_case_id for g in out] == ["EC_G1", "EC_G2"]
        assert all(isinstance(g, CaseGrading) for g in out)
        assert out[0].decision_match is True
        assert out[1].decision_match is True

    def test_record_without_truth_fails(self) -> None:
        """记录 id 不在真值中 → 显式失败，不静默跳过。"""
        with pytest.raises(ValueError, match="EC_G9"):
            grade_records([_record(eval_case_id="EC_G9")], [self._truth_row("EC_G1", _expected())])


class TestGraderHasNoLLM:
    """判卷路径零 LLM：grading 包静态扫描不得出现模型相关 import。"""

    def _imported_modules(self, path: Path) -> list[str]:
        """收集一个 .py 文件全部 import 的模块名（静态解析，不执行）。"""
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
        return names

    def test_grading_imports_no_llm_or_production_modules(self) -> None:
        py_files = sorted(GRADING_DIR.glob("*.py"))
        assert py_files, "grading 包不存在"
        for path in py_files:
            for name in self._imported_modules(path):
                top = name.split(".")[0]
                assert "llm" not in name.lower() and "litellm" not in name.lower(), (
                    f"{path.name} import 了 LLM 模块: {name}（判卷器是纯代码）"
                )
                assert top != "pra", f"{path.name} import 了生产代码: {name}"
                assert not name.startswith(("evaluation.generator", "evaluation.labeling")), (
                    f"{path.name} import 了生成器/打标模块: {name}"
                )
