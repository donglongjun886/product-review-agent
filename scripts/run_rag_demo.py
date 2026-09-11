"""RAG 检索 Demo（真实 Policy/Case KB · 三模式并排）。

用法::

    uv run python scripts/run_rag_demo.py                 # 固定 query 集 × 3 模式 × Top-3
    uv run python scripts/run_rag_demo.py --query "..."   # 追加自定义 query
    uv run python scripts/run_rag_demo.py --top-k 5

展示固定 query 集在 BM25 / Vector / Hybrid 三模式下的 Policy / Case Top-K 命中（并排如实
展示，不预设 Hybrid 最优），以及经真实 Tool（PolicySearchTool / CaseSearchTool 注入 RAG
索引）检索后的证据引用：``POLICY_REF`` weight=0.9 / ref_id=clause_id，``CASE_PRECEDENT``
weight=retrieval_score / ref_id=case_id —— 与 InMemory 世界同一引用格式。

全链路确定性：无网络、无真 LLM、固定 corpus + mock embedding；同输入可重放。
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from pra.domain.models import Budget
from pra.rag.factory import build_case_index, build_policy_index
from pra.tools.base import ToolContext
from pra.tools.case_search.tool import (
    CaseSearchArgs,
    CaseSearchFilters,
    CaseSearchTool,
)
from pra.tools.policy_search.tool import (
    PolicySearchArgs,
    PolicySearchFilters,
    PolicySearchTool,
)

MODES = ("bm25", "vector", "hybrid")

_POLICY_QUERIES = [
    "外观高度模仿知名品牌，无授权",  # 期望命中 IP 条款
    "标题含复刻高仿原单 仿冒来源词",
    "无依据功效夸大 增高磁疗 虚假宣传",
    "改标题重上架规避审核 商家多次",
]

_CASE_QUERIES = [
    "无品牌 + 高相似 + 商家多次上架",  # 期望命中对应先例
    "外观高度模仿品牌 换图规避重上架",
    "功效宣传无检测报告 虚假宣称",
    "材质标真皮实为PU 字段冲突",
]


def _clip(text: str, n: int = 46) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


async def _policy_table(query: str, top_k: int) -> None:
    print(f"\n▶ 政策检索   query = {query}")
    for mode in MODES:
        idx = build_policy_index(mode=mode)
        hits = await idx.search(
            query, PolicySearchFilters(), top_k=top_k, effective_only=True
        )
        cells = []
        for h in hits:
            tags = "/".join(t.value for t in h.risk_type) or "-"
            cells.append(f"{h.policy_id}·{h.clause_id}[{tags}]《{h.title}》{_clip(h.text)}")
        print(f"  [{mode:<7}] " + (" | ".join(cells) if cells else "(无命中)"))


async def _case_table(query: str, top_k: int) -> None:
    print(f"\n▶ 先例检索   query = {query}")
    for mode in MODES:
        idx = build_case_index(mode=mode)
        hits = await idx.search(query, CaseSearchFilters(), top_k=top_k)
        cells = []
        for h in hits:
            tags = "/".join(t.value for t in h.risk_type) or "-"
            cells.append(
                f"{h.case_id} sim={h.retrieval_score:.3f} {h.decision.value}/{h.risk_level.value}[{tags}] {_clip(h.summary, 40)}"
            )
        print(f"  [{mode:<7}] " + (" | ".join(cells) if cells else "(无命中)"))


def _tool_ctx() -> ToolContext:
    return ToolContext(run_id="rag-demo", case_id="DEMO_RAG_0001", budget=Budget())


async def _evidence_demo() -> None:
    print("\n▶ 工具引用（真实 Tool 注入 RAG 索引 → Evidence 引用格式）")
    policy_tool = PolicySearchTool(index=build_policy_index())
    p_res = await policy_tool.call(
        PolicySearchArgs(query="外观高度模仿知名品牌，无授权", top_k=3),
        _tool_ctx(),
    )
    print("  PolicySearchTool hits → Evidence:")
    for ev in policy_tool.to_evidence(p_res):
        print(f"    {ev.type:<14} weight={ev.weight} ref_id={ev.ref_id} | {_clip(ev.value, 66)}")

    case_tool = CaseSearchTool(index=build_case_index())
    c_res = await case_tool.call(
        CaseSearchArgs(query="无品牌 + 高相似 + 商家多次上架", top_k=3),
        _tool_ctx(),
    )
    print("  CaseSearchTool hits → Evidence:")
    for ev in case_tool.to_evidence(c_res):
        print(f"    {ev.type:<14} weight={ev.weight:.3f} ref_id={ev.ref_id} | {_clip(ev.value, 66)}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG 检索 Demo（三模式并排 + 工具引用）")
    parser.add_argument("--top-k", type=int, default=3, help="每路 Top-K（默认 3）")
    parser.add_argument("--query", action="append", default=[], help="追加自定义 query（政策+先例各跑）")
    return parser.parse_args(argv)


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    for q in _POLICY_QUERIES + args.query:
        await _policy_table(q, args.top_k)
    for q in _CASE_QUERIES + args.query:
        await _case_table(q, args.top_k)
    await _evidence_demo()
    print("\n[OK] RAG demo 完成（确定性 mock embedding + BM25 + 余弦；Phase 2 换本地模型 + Qdrant）")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_main(argv))
    except Exception as exc:
        print(f"[FAIL] RAG demo 运行失败: {exc!r}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
