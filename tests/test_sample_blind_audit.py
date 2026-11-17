"""盲标审计抽样：不泄漏答案、分层/模式/真值覆盖、同种子可复现。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from helpers import SCRIPTS_DIR, load_script

SCRIPT = SCRIPTS_DIR / "sample_blind_audit.py"

_ANSWER_KEYS = {"construction", "expected", "truth", "manifest", "snapshot"}
_HUMAN_COLUMNS = (
    "human_decision",
    "human_risk_type",
    "human_policy_ref",
    "human_reason",
)
_STRATA = {
    "blacklist+evasion",
    "blacklist",
    "evasion+dirty",
    "evasion clean",
    "brand+dirty",
    "brand only",
    "gap",
    "mismatch",
    "dirty only",
    "clean",
}


def _run(out_dir: Path, *args: str) -> None:
    module = load_script(SCRIPT)
    assert module.main(["--out-dir", str(out_dir), *args]) == 0


def _rows(out_dir: Path) -> list[dict[str, str]]:
    with (out_dir / "blind_30.csv").open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture()
def sampled(tmp_path: Path) -> Path:
    _run(tmp_path)
    return tmp_path


def test_sample_has_no_answer_columns(sampled: Path) -> None:
    rows = _rows(sampled)
    assert len(rows) == 30
    assert not set(rows[0]) & _ANSWER_KEYS
    for row in rows:
        assert all(row[column] == "" for column in _HUMAN_COLUMNS)
        assert all("POLICY_" not in value for value in row.values())


def test_sampling_covers_strata_patterns_and_truth(sampled: Path) -> None:
    report = json.loads(
        (sampled / "blind_30.sampling.json").read_text(encoding="utf-8")
    )
    assert report["sample_size"] == 30
    assert set(report["strata"]) == _STRATA
    assert set(report["strata"].values()) == {3}
    assert report["pattern_coverage"] == 14
    assert report["truth_key_groups"] == 9
    assert report["truth_key_min_per_group"] >= 3


def test_same_seed_reproduces_same_sample(tmp_path: Path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    _run(first, "--seed", "7")
    _run(second, "--seed", "7")
    assert (first / "blind_30.csv").read_bytes() == (
        second / "blind_30.csv"
    ).read_bytes()


def test_exclude_file_removes_cases(sampled: Path, tmp_path: Path) -> None:
    excluded_id = _rows(sampled)[0]["eval_case_id"]
    exclude = tmp_path / "exclude.txt"
    exclude.write_text(f"# 种子案例\n{excluded_id}\n", encoding="utf-8")

    out = tmp_path / "filtered"
    _run(out, "--exclude-file", str(exclude))

    ids = {row["eval_case_id"] for row in _rows(out)}
    assert excluded_id not in ids
    assert len(ids) == 30
