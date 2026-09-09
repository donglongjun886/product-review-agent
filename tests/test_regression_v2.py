"""v2 正式集回归守护（tests/test_regression_v2.py）—— backlog P1-4 修法（域 B）。

背景（域 B 实测确认，eval-review-b-data.md §1 P1-1 / §5）：
- v2（320 正式集）此前零回归守护：``scripts/run_regression.py`` 默认只指 v1；
  ``docs/05-visual-similarity-gate-proposal.md`` 明示 v2 "目前无 baseline 文件"；
- 入库 v1 基线（含 rule/single_call_llm/agent 三方案 35 案）只有 agent 单方案被
  ``tests/test_rag.py:329-336`` 断言 —— v1 与 v2 的 rule/single_call_llm 决策漂移
  （如 screening 修正集合入、terms 词表变动）此前 CI 均抓不到。

本文件补齐（全部离线确定性：scripted 桩、无真 LLM/网络；基线/数据文件**只读**，
重放路径不写 eval_data 目录 —— 与 test_rag.py / test_evaluation_phase2.py 约定一致）：

1. ``test_v2_baseline_file_structure_and_self_digest``：入库 v2 基线文件静态结构 +
   digest 自洽（由文件内 ids+decisions 重算 == 存储 digest）；
2. ``test_v2_three_scheme_decisions_match_baseline``：三方案 320 决策序列 == 入库
   基线（rule/single_call_llm/agent **全部**断言，不再只 agent）+ digest 相等；
3. ``test_v2_snapshot_byte_replay``：重放快照规范化序列化 == 入库文件逐字节一致
   （digest 机制的字节级重放口径）；
4. ``test_v2_regression_compare_passes``：``compare_snapshots``（regression API）
   对 v2 基线判 PASS；
5. ``test_v1_baseline_all_three_schemes_asserted``：v1 基线三方案全断言（补
   test_rag 只断 agent 的缺口），rule/single_call_llm 漂移从此可被抓；
6. ``test_v2_dataset_generator_byte_lock``：(320,42) 重新生成 vs 入库
   ``cases_v2.jsonl`` **逐字节**一致（防手改数据行 / 生成器漂移；只读内存比对）。
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import pytest

from pra.agent.guardrails import llm_shell
from pra.evaluation.regression import (
    canonical_digest,
    compare_snapshots,
    compute_current_snapshot,
)
from pra.evaluation.runner import ALL_SCHEMES

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_V1 = REPO_ROOT / "eval_data" / "v1" / "cases_v1.jsonl"
EVAL_V2 = REPO_ROOT / "eval_data" / "v2" / "cases_v2.jsonl"
BASELINE_V1 = REPO_ROOT / "eval_data" / "v1" / "regression_baseline.json"
BASELINE_V2 = REPO_ROOT / "eval_data" / "v2" / "regression_baseline.json"
GEN_SCRIPT = REPO_ROOT / "scripts" / "eval_dataset_gen.py"

SCHEMES: tuple[str, ...] = tuple(ALL_SCHEMES)  # rule / single_call_llm / agent


def _load_baseline(path: Path) -> dict:
    assert path.exists(), f"入库基线文件缺失: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_serialize(snapshot: dict) -> str:
    """与 regression.write_baseline 完全相同的落盘序列化（字节级重放口径）。"""
    return json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _load_gen_module():
    """按路径加载生成器脚本（scripts/ 非包；importlib 载入 —— 与 dataset_v2 测试同款）。"""
    spec = importlib.util.spec_from_file_location("eval_dataset_gen_regress_mod", GEN_SCRIPT)
    assert spec and spec.loader, f"无法定位生成器脚本: {GEN_SCRIPT}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def v2_current_snapshot() -> dict:
    """三方案全量 v2 快照（确定性 ~3.4 s；模块内共享，避免每个测试重跑 agent 320 案）。

    与 conftest 的 autouse 同款卫生：前后快照/还原 llm_shell 进程级后端状态
    （AgentScheme.run 的 finally 会 set_llm_backend(None)），保证跨测试无全局污染。
    """
    saved_backend = llm_shell._backend
    saved_default = llm_shell._default_backend
    try:
        return asyncio.run(compute_current_snapshot(EVAL_V2, schemes=SCHEMES))
    finally:
        llm_shell._backend = saved_backend
        llm_shell._default_backend = saved_default


# ---------------------------------------------------------------------------
# 1) 入库 v2 基线：结构 + digest 自洽（不跑任何方案，纯文件静态断言）
# ---------------------------------------------------------------------------


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
    # digest 自洽：由文件内 ids+decisions 重算 == 存储 digest（防只改序列不改 digest）
    assert canonical_digest(baseline["per_case_ids"], baseline["decisions"]) == baseline["digest"]
    # data_hint 指向 v2 数据文件（与 v1 基线"相对路径"口径同构）
    assert baseline["data_hint"].endswith("eval_data/v2/cases_v2.jsonl")


# ---------------------------------------------------------------------------
# 2) v2 三方案决策序列 == 入库基线（全量重跑一次；rule/single/agent 全断言）
# ---------------------------------------------------------------------------


def test_v2_three_scheme_decisions_match_baseline(v2_current_snapshot: dict) -> None:
    baseline = _load_baseline(BASELINE_V2)
    snap = v2_current_snapshot
    assert snap["per_case_ids"] == baseline["per_case_ids"], "v2 case 序/集合漂移"
    for s in SCHEMES:
        assert snap["decisions"][s] == baseline["decisions"][s], (
            f"v2 scheme={s} 决策序列与入库基线不一致（回归漂移）"
        )
    assert snap["digest"] == baseline["digest"]
    # 决策分布不变量（口径见 manifest/docs §2.1）：agent REJECT=140=GT REJECT 数等
    from collections import Counter

    assert Counter(snap["decisions"]["agent"]) == {"PASS": 124, "REJECT": 140, "HUMAN_REVIEW": 56}


def test_v2_snapshot_byte_replay(v2_current_snapshot: dict) -> None:
    """重放快照（data_hint 归一为入库口径后）逐字节 == 入库基线文件。"""
    baseline = _load_baseline(BASELINE_V2)
    snap = {**v2_current_snapshot, "data_hint": baseline["data_hint"]}  # 仅路径形态元数据
    assert _canonical_serialize(snap) == BASELINE_V2.read_text(encoding="utf-8"), (
        "重放快照与入库 v2 基线逐字节不一致（序列化口径漂移）"
    )


def test_v2_regression_compare_passes(v2_current_snapshot: dict) -> None:
    """regression API 对 v2 入库基线判 PASS（digest 逐字节重放口径）。"""
    baseline = _load_baseline(BASELINE_V2)
    report = compare_snapshots(v2_current_snapshot, baseline)
    assert report.ok is True and report.status == "PASS"
    assert report.digest_match is True
    assert not report.mismatches
    assert report.baseline_cases == 320 and report.current_cases == 320


# ---------------------------------------------------------------------------
# 3) v1 基线三方案补全断言（rule/single_call_llm 此前无任何 CI 引用）
# ---------------------------------------------------------------------------


async def test_v1_baseline_all_three_schemes_asserted() -> None:
    baseline = _load_baseline(BASELINE_V1)
    assert baseline["total_cases"] == 35
    snap = await compute_current_snapshot(EVAL_V1, schemes=SCHEMES)
    assert snap["per_case_ids"] == baseline["per_case_ids"]
    for s in SCHEMES:
        assert snap["decisions"][s] == baseline["decisions"][s], (
            f"v1 scheme={s} 决策序列与入库基线不一致（screening/terms 词表漂移守卫）"
        )
    assert snap["digest"] == baseline["digest"]
    report = compare_snapshots(snap, baseline)
    assert report.ok is True and report.status == "PASS"
    # v1 基线同样逐字节可重放（data_hint 归一为入库口径）
    ser = _canonical_serialize({**snap, "data_hint": baseline["data_hint"]})
    assert ser == BASELINE_V1.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 4) 数据字节锁：(320,42) 重新生成 == 入库 cases_v2.jsonl（防手改数据行 / 生成器漂移）
# ---------------------------------------------------------------------------


def test_v2_dataset_generator_byte_lock_320_42() -> None:
    gen = _load_gen_module()
    rows = gen.generate(count=320, seed=42)
    assert len(rows) == 320
    # _write_dataset 同款序列化（见 eval_dataset_gen.py:976-980）
    regenerated = ("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n").encode("utf-8")
    checked_in = EVAL_V2.read_bytes()
    assert regenerated == checked_in, (
        "(320,42) 重新生成与入库 eval_data/v2/cases_v2.jsonl 逐字节不一致"
        " —— 生成器漂移或数据行被手改（两者都须人工复核后同步）"
    )
