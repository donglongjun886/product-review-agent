"""RAG 共享件与默认路径验收测试（CI 上恒跑的那一面）。"""

from __future__ import annotations

import json
from pathlib import Path

from helpers import tool_by_name
from inmemory_world import (
    _DEFAULT_PRECEDENTS,
    InMemoryPolicyIndex,
    build_inmemory_tools,
)

from pra.rag.corpus import load_cases, load_policies

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_V2 = REPO_ROOT / "eval_data" / "v2" / "cases_v2.jsonl"


def _eval_case_ids() -> set[str]:
    """eval_data/v2 全部 case 标识（eval_case_id / input.case_id / lineage.seed_case_id）。"""
    ids: set[str] = set()
    if EVAL_V2.exists():
        for line in EVAL_V2.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ids.add(obj["eval_case_id"])
            ids.add(obj["input"]["case_id"])
            lineage = obj.get("lineage") or {}
            if lineage.get("seed_case_id"):
                ids.add(lineage["seed_case_id"])
    for row in _DEFAULT_PRECEDENTS:
        ids.add(row["case_id"])
    return ids


def test_corpus_data_integrity() -> None:
    policies, pmeta = load_policies()
    cases, cmeta = load_cases()

    assert 20 <= len(policies) <= 40, f"Policy KB 条款数须 20~40，实际 {len(policies)}"
    assert 50 <= len(cases) <= 100, f"Case KB 条数须 50~100，实际 {len(cases)}"

    expired = [p for p in policies if p.status == "EXPIRED"]
    assert len(expired) >= 2, "Policy KB 须含 ≥2 条 EXPIRED（版本过滤测试面）"

    assert len({p.clause_id for p in policies}) == len(policies), "clause_id 须唯一"
    case_ids = [c.case_id for c in cases]
    assert len(case_ids) == len(set(case_ids)), "case_id 须唯一"
    assert all(str(cid).startswith("RAG_CASE_") for cid in case_ids)

    assert "isolation_declaration" in pmeta and "isolation_declaration" in cmeta


def test_case_kb_isolated_from_eval_gt() -> None:
    """Case KB 与 eval GT 严格隔离。"""
    cases, _ = load_cases()
    kb_ids = {c.case_id for c in cases}
    eval_ids = _eval_case_ids()

    overlap = kb_ids & eval_ids
    assert not overlap, f"Case KB 与 eval GT case_id 有交集（红线违反）: {sorted(overlap)[:10]}"
    assert kb_ids.isdisjoint({r["case_id"] for r in _DEFAULT_PRECEDENTS})


_EXPECTED_TOOLS = [
    "ProductTool",
    "MerchantTool",
    "CaseSearchTool",
    "PolicySearchTool",
]


def test_build_inmemory_tools_uses_inmemory_indexes() -> None:
    tools = build_inmemory_tools()
    assert [t.name for t in tools] == _EXPECTED_TOOLS
    assert type(tool_by_name(tools, "CaseSearchTool")._index).__name__ == "InMemoryCaseIndex"
    assert isinstance(tool_by_name(tools, "PolicySearchTool")._index, InMemoryPolicyIndex)
