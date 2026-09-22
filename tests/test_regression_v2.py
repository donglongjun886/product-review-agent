"""v2 正式集回归守护：三方案决策序列、digest 与数据字节锁。

1. 入库 v2 基线静态结构 + digest 自洽（由文件内 ids+decisions 重算）；
2. v2 三方案 320 决策序列与 digest 全部 == 入库基线；
3. 重放快照的规范化序列化与入库文件逐字节一致；
4. ``compare_snapshots`` 对 v2 基线判 PASS；
5. (320,42) 重新生成与入库 ``cases_v2.jsonl`` 逐字节一致（防手改数据行 / 生成器漂移）。

全部离线确定性（scripted 桩）；基线/数据文件只读，重放不写 eval_data 目录。
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from pra.evaluation.regression import (
    canonical_digest,
    compare_snapshots,
    compute_current_snapshot,
)
from pra.evaluation.runner import ALL_SCHEMES

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_V2 = REPO_ROOT / "eval_data" / "v2" / "cases_v2.jsonl"
BASELINE_V2 = REPO_ROOT / "eval_data" / "v2" / "regression_baseline.json"
GEN_SCRIPT = REPO_ROOT / "scripts" / "eval_dataset_gen.py"

SCHEMES: tuple[str, ...] = tuple(ALL_SCHEMES)  # rule / single_call_llm / agent


def _load_baseline(path: Path) -> dict:
    assert path.exists(), f"入库基线文件缺失: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_serialize(snapshot: dict) -> str:
    """与 regression.write_baseline 相同的落盘序列化（字节级重放口径）。"""
    return json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _load_gen_module():
    spec = importlib.util.spec_from_file_location("eval_dataset_gen_regress_mod", GEN_SCRIPT)
    assert spec and spec.loader, f"无法定位生成器脚本: {GEN_SCRIPT}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def v2_current_snapshot() -> dict:
    """三方案全量 v2 快照（确定性 ~3.4 s；模块内共享，避免每个测试重跑 agent 320 案）。

    LLM 后端由 ``AgentScheme`` 每案显式注入 ``build_agent_graph(llm=...)``（无进程级全局），
    故跨测试无需还原任何后端状态。
    """
    return asyncio.run(compute_current_snapshot(EVAL_V2, schemes=SCHEMES))


# --- 1) 入库 v2 基线：结构 + digest 自洽（不跑任何方案，纯文件静态断言）


def test_v2_baseline_file_structure_and_self_digest() -> None:
    baseline = _load_baseline(BASELINE_V2)
    assert baseline["format_version"] == 1
    assert baseline["total_cases"] == 320
    assert baseline["scheme_order"] == list(SCHEMES)
    assert len(baseline["per_case_ids"]) == 320
    assert len(set(baseline["per_case_ids"])) == 320, "per_case_ids 须唯一"
    assert baseline["per_case_ids"][0].startswith("EC_V2_")
    for s in SCHEMES:
        assert len(baseline["decisions"][s]) == 320, f"scheme={s} 决策序列长度须 320"
    # digest 自洽：由文件内 ids+decisions 重算 == 存储 digest
    assert canonical_digest(baseline["per_case_ids"], baseline["decisions"]) == baseline["digest"]
    assert baseline["data_hint"].endswith("eval_data/v2/cases_v2.jsonl")


# --- 2) v2 三方案决策序列 == 入库基线（全量重跑一次；rule/single/agent 全断言）


def test_v2_three_scheme_decisions_match_baseline(v2_current_snapshot: dict) -> None:
    baseline = _load_baseline(BASELINE_V2)
    snap = v2_current_snapshot
    assert snap["per_case_ids"] == baseline["per_case_ids"], "v2 case 序/集合漂移"
    for s in SCHEMES:
        assert snap["decisions"][s] == baseline["decisions"][s], (
            f"v2 scheme={s} 决策序列与入库基线不一致（回归漂移）"
        )
    assert snap["digest"] == baseline["digest"]
    # 决策分布不变量：agent REJECT=140=GT REJECT 数
    from collections import Counter

    assert Counter(snap["decisions"]["agent"]) == {"PASS": 124, "REJECT": 140, "HUMAN_REVIEW": 56}


def test_v2_snapshot_byte_replay(v2_current_snapshot: dict) -> None:
    baseline = _load_baseline(BASELINE_V2)
    snap = {**v2_current_snapshot, "data_hint": baseline["data_hint"]}  # 仅路径形态元数据
    assert _canonical_serialize(snap) == BASELINE_V2.read_text(encoding="utf-8"), (
        "重放快照与入库 v2 基线逐字节不一致（序列化口径漂移）"
    )


def test_v2_regression_compare_passes(v2_current_snapshot: dict) -> None:
    baseline = _load_baseline(BASELINE_V2)
    report = compare_snapshots(v2_current_snapshot, baseline)
    assert report.ok is True and report.status == "PASS"
    assert report.digest_match is True
    assert not report.mismatches
    assert report.baseline_cases == 320 and report.current_cases == 320


# --- 3) 数据字节锁：(320,42) 重新生成 == 入库 cases_v2.jsonl（防手改数据行 / 生成器漂移）


def test_v2_dataset_generator_byte_lock_320_42() -> None:
    gen = _load_gen_module()
    rows = gen.generate(count=320, seed=42)
    assert len(rows) == 320
    regenerated = ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8")
    checked_in = EVAL_V2.read_bytes()
    assert regenerated == checked_in, (
        "(320,42) 重新生成与入库 eval_data/v2/cases_v2.jsonl 逐字节不一致"
        " —— 生成器漂移或数据行被手改（两者都须人工复核后同步）"
    )
