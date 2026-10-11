"""数据集装配 seam：v3 cases×truth join 成执行层视图；旧格式数据显式报错而非静默混用。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from evaluation.cli_utils import resolve_v3_data_paths
from evaluation.dataset.loader import load_dataset_v3
from evaluation.dataset.schema import (
    CaseManifestV3,
    EvalCaseV3,
    EvalConstructionV3,
    EvalExpectedV3,
    EvalTruthV3,
    MerchantFactsV3,
    RegistryFactsV3,
)
from helpers import REPO_ROOT, make_case

V3_DIR = REPO_ROOT / "eval_data" / "v3"


def _case(eval_case_id: str) -> EvalCaseV3:
    """造一条 v3 案件行（只含输入与构造记录，无真值）。"""
    return EvalCaseV3(
        eval_case_id=eval_case_id,
        scene="normal",
        construction=EvalConstructionV3(pattern="clean", params={}),
        input=make_case(case_id=f"CASE_{eval_case_id}"),
    )


def _truth(eval_case_id: str, *, decision: str = "PASS") -> EvalTruthV3:
    """造一条 v3 真值行（默认干净 PASS 案）。"""
    expected = EvalExpectedV3(
        decision=decision,
        abstain_label="SHOULD_ABSTAIN" if decision == "HUMAN_REVIEW" else "AUTO_DECIDABLE",
        risk_type=[] if decision == "PASS" else ["POTENTIAL_IP_RISK"],
        expected_tools=[] if decision == "REJECT" else ["MerchantTool", "ProductTool"],
        expected_policies=["POLICY_1.1"] if decision == "REJECT" else [],
    )
    return EvalTruthV3(
        eval_case_id=eval_case_id,
        manifest=CaseManifestV3(
            text_hits=[],
            snapshot_brand="潮动",
            snapshot_gaps=[],
            registry_mismatch=False,
            merchant=MerchantFactsV3(
                merchant_id="M_T1",
                removals=0,
                title_relisting_count=0,
                similar_product_count=0,
                credit_score=90,
            ),
            registry=RegistryFactsV3(
                product_id="P_T1",
                category="女鞋/运动鞋",
                brand="潮动",
                version=1,
                status="ON_SALE",
            ),
        ),
        expected=expected,
    )


def _write_jsonl(path: Path, rows: list) -> None:
    """把模型行列表写成 JSONL 文件。"""
    path.write_text(
        "".join(json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def _write_pair(tmp_path: Path, cases: list, truth: list) -> tuple[Path, Path]:
    """在临时目录写 cases_v3.jsonl + truth_v3.jsonl 并返回路径对。"""
    cases_path = tmp_path / "cases_v3.jsonl"
    truth_path = tmp_path / "truth_v3.jsonl"
    _write_jsonl(cases_path, cases)
    _write_jsonl(truth_path, truth)
    return cases_path, truth_path


class TestRealV3Dataset:
    """真实 v3 数据集：cases×truth 全量 1:1 join。"""

    def test_v3_dataset_joins_completely(self) -> None:
        dataset = load_dataset_v3(V3_DIR / "cases_v3.jsonl", V3_DIR / "truth_v3.jsonl")
        assert len(dataset.cases) == 320
        assert len(dataset.truth) == 320
        assert [c.eval_case_id for c in dataset.cases] == [
            t.eval_case_id for t in dataset.truth
        ], "join 后 truth 必须与 cases 同序对齐"


class TestJoinValidation:
    """join 完整性：缺失 / 重复 id 显式失败。"""

    def test_join_aligns_truth_to_case_order(self, tmp_path: Path) -> None:
        """truth 文件行序与 cases 不同 → 仍按 eval_case_id 对齐回 cases 行序。"""
        cases = [_case("EC_A1"), _case("EC_A2")]
        truth = [_truth("EC_A2"), _truth("EC_A1")]
        cases_path, truth_path = _write_pair(tmp_path, cases, truth)

        dataset = load_dataset_v3(cases_path, truth_path)
        assert [t.eval_case_id for t in dataset.truth] == ["EC_A1", "EC_A2"]

    def test_missing_truth_id_fails(self, tmp_path: Path) -> None:
        cases_path, truth_path = _write_pair(tmp_path, [_case("EC_A1"), _case("EC_A2")], [_truth("EC_A1")])
        with pytest.raises(ValueError, match="EC_A2"):
            load_dataset_v3(cases_path, truth_path)

    def test_extra_truth_id_fails(self, tmp_path: Path) -> None:
        cases_path, truth_path = _write_pair(tmp_path, [_case("EC_A1")], [_truth("EC_A1"), _truth("EC_A9")])
        with pytest.raises(ValueError, match="EC_A9"):
            load_dataset_v3(cases_path, truth_path)

    def test_duplicate_case_id_fails(self, tmp_path: Path) -> None:
        cases_path, truth_path = _write_pair(
            tmp_path, [_case("EC_A1"), _case("EC_A1")], [_truth("EC_A1")]
        )
        with pytest.raises(ValueError, match="重复"):
            load_dataset_v3(cases_path, truth_path)

    def test_duplicate_truth_id_fails(self, tmp_path: Path) -> None:
        cases_path, truth_path = _write_pair(
            tmp_path, [_case("EC_A1")], [_truth("EC_A1"), _truth("EC_A1")]
        )
        with pytest.raises(ValueError, match="重复"):
            load_dataset_v3(cases_path, truth_path)


class TestOldFormatExplicitFailure:
    """旧格式数据喂入新执行层 → 给人读的显式错误，不得 pydantic 裸报错糊过去。"""

    def test_schema_version_2_file_rejected_with_reason(self, tmp_path: Path) -> None:
        """schema_version=2 的 JSONL → 显式报旧格式与迁移方向，非 ValidationError 裸输出。"""
        row = _case("EC_A1").model_dump(mode="json")
        row["schema_version"] = 2
        cases_path = tmp_path / "cases_v3.jsonl"
        cases_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        _write_jsonl(tmp_path / "truth_v3.jsonl", [_truth("EC_A1")])

        with pytest.raises(ValueError, match="schema_version=2") as exc_info:
            load_dataset_v3(cases_path, tmp_path / "truth_v3.jsonl")
        assert "ValidationError" not in str(exc_info.value)
        assert "v3" in str(exc_info.value)

    def test_resolve_v3_directory(self) -> None:
        cases_path, truth_path = resolve_v3_data_paths(str(V3_DIR))
        assert cases_path.name == "cases_v3.jsonl"
        assert truth_path.name == "truth_v3.jsonl"
        assert cases_path.is_file() and truth_path.is_file()

    def test_resolve_cases_file_auto_pairs_truth(self, tmp_path: Path) -> None:
        """``--data`` 传 cases_v3.jsonl 文件：同目录自动配对 truth_v3.jsonl。"""
        cases_path, truth_path = _write_pair(tmp_path, [_case("EC_A1")], [_truth("EC_A1")])
        resolved = resolve_v3_data_paths(str(cases_path))
        assert resolved == (cases_path, truth_path)

    def test_resolve_v2_directory_rejected(self, tmp_path: Path) -> None:
        (tmp_path / "cases_v2.jsonl").write_text("{}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="v2 旧格式"):
            resolve_v3_data_paths(str(tmp_path))

    def test_resolve_v2_file_rejected(self, tmp_path: Path) -> None:
        v2_file = tmp_path / "cases_v2.jsonl"
        v2_file.write_text("{}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="旧格式"):
            resolve_v3_data_paths(str(v2_file))

    def test_resolve_missing_path_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="不存在"):
            resolve_v3_data_paths(str(tmp_path / "nowhere"))
