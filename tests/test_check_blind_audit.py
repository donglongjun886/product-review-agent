"""盲标比对：一致率、分歧识别、Clopper-Pearson 上界。"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest
from helpers import SCRIPTS_DIR, load_script

SCRIPT = SCRIPTS_DIR / "check_blind_audit.py"
TRUTH_PATH = Path(__file__).resolve().parents[1] / "eval_data" / "v3" / "truth_v3.jsonl"
_FIELDNAMES = [
    "eval_case_id",
    "human_decision",
    "human_risk_type",
    "human_policy_ref",
    "human_reason",
]


def _truth_rows() -> list[dict]:
    with TRUTH_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_upper_bound_matches_closed_forms() -> None:
    module = load_script(SCRIPT)
    assert module.upper_bound(0, 30) == pytest.approx(1 - 0.05 ** (1 / 30), abs=1e-9)
    assert module.upper_bound(1, 2) == pytest.approx(math.sqrt(0.95), abs=1e-9)
    assert module.upper_bound(2, 2) == 1.0


def test_compare_flags_disagreement_and_counts(tmp_path: Path) -> None:
    module = load_script(SCRIPT)
    truths = _truth_rows()
    pass_case = next(row for row in truths if row["expected"]["decision"] == "PASS")
    reject_case = next(row for row in truths if row["expected"]["decision"] == "REJECT")

    path = tmp_path / "filled.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerow(
            {
                "eval_case_id": pass_case["eval_case_id"],
                "human_decision": "PASS",
                "human_risk_type": "无",
                "human_policy_ref": "无",
                "human_reason": "ok",
            }
        )
        writer.writerow(
            {
                "eval_case_id": reject_case["eval_case_id"],
                "human_decision": "PASS",
                "human_risk_type": "无",
                "human_policy_ref": "无",
                "human_reason": "wrong",
            }
        )

    assert module.main(["--input", str(path), "--out-dir", str(tmp_path)]) == 0

    report = json.loads((tmp_path / "audit_result.json").read_text(encoding="utf-8"))
    assert report["sample_size"] == 2
    assert report["agreement"]["all"] == 1
    assert report["errors"] == 1
    assert report["error_rate"] == 0.5
    assert len(report["disagreements"]) == 1
    assert report["disagreements"][0]["eval_case_id"] == reject_case["eval_case_id"]
    assert "decision" in report["disagreements"][0]["fields"]


def test_split_normalizes_empty_markers() -> None:
    module = load_script(SCRIPT)
    assert module._split("无") == []
    assert module._split(None) == []
    assert module._split("POLICY_1.2+POLICY_4.5") == [
        "POLICY_1.2",
        "POLICY_4.5",
    ]
