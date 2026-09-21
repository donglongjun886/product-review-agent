"""RAG 共享件与默认路径验收测试（CI 上恒跑的那一面）。

保留范围（被测对象仍存在、且不依赖 ``--extra rag``）：
- corpus loader / schema 校验：Policy/Case KB 规模与唯一性、≥2 条 EXPIRED、meta 隔离声明；
- **隔离红线**：Case KB 的 case_id 与 eval_data v1+v2 全部 case 标识（含 InMemory 种子先例）无交集，
  防评测作弊；
- ``build_tools()`` 默认（memory 世界）仍是 6 个 InMemory 工具（与 ``build_tools("memory")`` 一致）；
- 默认评测路径（``tool_world="eval"``）全量 v1 agent 决策序列 == 入库基线（回归不破）。

**chroma 真实检索的离线验收用例已整体移除**：它们靠一个测试自持的假编码器（词面 sha256 特征）
离线跑，验的是管道而非检索质量；假编码器及其用例一并删除后，chroma 检索只剩
``tests/test_rag_production_wiring.py`` 的真模型端到端用例。默认路径「零额外依赖」仍由
``tests/test_rag_default_path_no_extra.py`` 守护。

确定性、无网络、无真 LLM、不写评测数据目录。
"""

from __future__ import annotations

import json
from pathlib import Path

from pra.evaluation.dataset.loader import load_dataset
from pra.evaluation.harness.agent_scheme import EVAL_PRECEDENTS, AgentScheme
from pra.evaluation.harness.base import EvalContext
from pra.evaluation.regression import compute_current_snapshot
from pra.rag.corpus import load_cases, load_policies
from pra.tools import build_tools
from pra.tools.policy_search.tool import InMemoryPolicyIndex

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_V1 = REPO_ROOT / "eval_data" / "v1" / "cases_v1.jsonl"
EVAL_V2 = REPO_ROOT / "eval_data" / "v2" / "cases_v2.jsonl"
BASELINE_FILE = REPO_ROOT / "eval_data" / "v1" / "regression_baseline.json"


def _eval_case_ids() -> set[str]:
    """eval_data v1+v2 全部 case 标识（eval_case_id / input.case_id / lineage.seed_case_id）。"""
    ids: set[str] = set()
    for path in (EVAL_V1, EVAL_V2):
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ids.add(obj["eval_case_id"])
            ids.add(obj["input"]["case_id"])
            lineage = obj.get("lineage") or {}
            if lineage.get("seed_case_id"):
                ids.add(lineage["seed_case_id"])
    # 追加 InMemory eval 世界种子先例（agent_scheme.EVAL_PRECEDENTS / CASE_1832 等）
    for row in EVAL_PRECEDENTS:
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

    # meta 携带来源/隔离声明（JSON 无注释 → meta 字段，schema 校验即生效）
    assert "isolation_declaration" in pmeta and "isolation_declaration" in cmeta


def test_case_kb_isolated_from_eval_gt() -> None:
    """Case KB 与 eval GT 严格隔离（防检索到 GT = 评测作弊）。"""
    cases, _ = load_cases()
    kb_ids = {c.case_id for c in cases}
    eval_ids = _eval_case_ids()

    overlap = kb_ids & eval_ids
    assert not overlap, f"Case KB 与 eval GT case_id 有交集（红线违反）: {sorted(overlap)[:10]}"
    # InMemory 种子先例 id 亦不得混入（CASE_1832/0911/2033/2120 等）
    assert kb_ids.isdisjoint({r["case_id"] for r in EVAL_PRECEDENTS})


_EXPECTED_TOOLS = [
    "ProductTool",
    "ImageAnalysisTool",
    "OCRTool",
    "MerchantTool",
    "CaseSearchTool",
    "PolicySearchTool",
]


def test_build_tools_memory_unchanged() -> None:
    memory = build_tools("memory")
    assert [t.name for t in memory] == _EXPECTED_TOOLS
    # 默认参数即 memory：两个检索工具仍注入 InMemory 索引
    default = build_tools()
    assert [t.name for t in default] == _EXPECTED_TOOLS
    assert type(memory[4]._index).__name__ == "InMemoryCaseIndex"
    assert isinstance(memory[5]._index, InMemoryPolicyIndex)


async def test_default_eval_path_regression_intact() -> None:
    # (a) 默认路径（tool_world="eval"）回归：全量 v1 agent 决策序列 == 入库基线
    baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    assert baseline.get("format_version") == 1
    snap = await compute_current_snapshot(EVAL_V1, schemes=("agent",))
    assert snap["decisions"]["agent"] == baseline["decisions"]["agent"], (
        "默认 InMemory 评测路径回归失败：agent 决策序列与基线不一致"
    )

    # (b) 显式 EvalContext(tool_world="eval") 与默认 ctx 等价（回归不破坏）
    scheme = AgentScheme()
    smoke = [c for c in load_dataset(EVAL_V1) if c.expected.decision == "REJECT"][:4]
    ctx_default = EvalContext()
    assert ctx_default.tool_world == "eval"
    default_recs = [await scheme.run(c, ctx_default) for c in smoke]
    eval_recs = [await scheme.run(c, EvalContext(tool_world="eval")) for c in smoke]
    assert [r.decision for r in default_recs] == [r.decision for r in eval_recs]
