"""RAG 检索 Demo（真实 Policy/Case KB · hybrid）。

用法::

    uv run python scripts/run_rag_demo.py                 # 固定 query 集 × Top-3
    uv run python scripts/run_rag_demo.py --query "..."   # 追加自定义 query
    uv run python scripts/run_rag_demo.py --top-k 5

展示固定 query 集在 **hybrid（生产唯一口径）** 下的 Policy / Case Top-K 命中 ——
生产/评测没有模式开关（见 ``docs/00`` 检索口径），本 demo 走公共 ``index.search()``。

以及经真实 Tool（PolicySearchTool / CaseSearchTool 注入 RAG 索引）检索后的证据引用：
``POLICY_REF`` weight=0.9 / ref_id=clause_id，``CASE_PRECEDENT`` weight=retrieval_score /
ref_id=case_id —— 与 InMemory 世界同一引用格式。

后端 = chroma（ChromaDB + LlamaIndex + BM25(jieba) + RRF；需 ``uv sync --extra rag``）。
demo **显式注入真实语义编码器** ``production_embedder()``（BAAI/bge-small-zh-v1.5，dim 512）
并配 ``ChromaConfig(ephemeral=True)``（进程内内存库）—— 故无需起服务端；但需 BGE 模型**已缓存**
（缺省 ``local_files_only=True`` 只读本地、不联网；预热须显式 ``local_files_only=False``）。

全链路确定性：无真 LLM、固定 corpus + 真语义编码器；同输入可重放（编码器确定性）。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from pra.domain.models import Budget
from pra.rag.chroma_store import ChromaConfig
from pra.rag.factory import build_case_index, build_policy_index
from pra.tools import production_embedder
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


def _build(kind: str) -> Any:
    """装配一个 chroma 索引：真实语义编码器 + 进程内 EphemeralClient（不连服务端）。

    显式传 ``embedding_model=production_embedder()`` —— 只读本地缓存的 BGE 编码器
    （``BaseEmbedding``）；只为让 demo 的编码来源一目了然。
    """
    build = build_policy_index if kind == "policy" else build_case_index
    return build(
        embedding_model=production_embedder(),
        config=ChromaConfig(ephemeral=True),
    )


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


def _policy_cell(row: Any) -> str:
    tags = "/".join(t.value for t in row.risk_type) or "-"
    return f"{row.policy_id}·{row.clause_id}[{tags}]《{row.title}》{_clip(row.text)}"


def _case_cell(row: Any, score: float) -> str:
    tags = "/".join(t.value for t in row.risk_type) or "-"
    return (
        f"{row.case_id} score={score:.3f} {row.decision.value}/{row.risk_level.value}"
        f"[{tags}] {_clip(row.summary, 40)}"
    )


async def _policy_table(index: Any, query: str, top_k: int) -> None:
    print(f"\n▶ 政策检索   query = {query}")
    hits = await index.search(query, PolicySearchFilters(), top_k, True)
    cells = [_policy_cell(h) for h in hits]
    print("  " + (" | ".join(cells) if cells else "(无命中)"))


async def _case_table(index: Any, query: str, top_k: int) -> None:
    print(f"\n▶ 先例检索   query = {query}")
    hits = await index.search(query, CaseSearchFilters(), top_k)
    cells = [_case_cell(h, h.retrieval_score) for h in hits]
    print("  " + (" | ".join(cells) if cells else "(无命中)"))


def _tool_ctx() -> ToolContext:
    return ToolContext(run_id="rag-demo", case_id="DEMO_RAG_0001", budget=Budget())


async def _evidence_demo() -> None:
    print("\n▶ 工具引用（真实 Tool 注入 RAG 索引 → Evidence 引用格式）")
    policy_tool = PolicySearchTool(index=_build("policy"))
    p_res = await policy_tool.call(
        PolicySearchArgs(query="外观高度模仿知名品牌，无授权", top_k=3),
        _tool_ctx(),
    )
    print("  PolicySearchTool hits → Evidence:")
    for ev in policy_tool.to_evidence(p_res):
        print(f"    {ev.type:<14} weight={ev.weight} ref_id={ev.ref_id} | {_clip(ev.value, 66)}")

    case_tool = CaseSearchTool(index=_build("case"))
    c_res = await case_tool.call(
        CaseSearchArgs(query="无品牌 + 高相似 + 商家多次上架", top_k=3),
        _tool_ctx(),
    )
    print("  CaseSearchTool hits → Evidence:")
    for ev in case_tool.to_evidence(c_res):
        print(f"    {ev.type:<14} weight={ev.weight:.3f} ref_id={ev.ref_id} | {_clip(ev.value, 66)}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG 检索 Demo（hybrid 单一口径）")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K（默认 3）")
    parser.add_argument("--query", action="append", default=[], help="追加自定义 query（政策+先例各跑）")
    return parser.parse_args(argv)


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    policy_index = _build("policy")
    case_index = _build("case")
    for q in _POLICY_QUERIES + args.query:
        await _policy_table(policy_index, q, args.top_k)
    for q in _CASE_QUERIES + args.query:
        await _case_table(case_index, q, args.top_k)
    await _evidence_demo()
    print("\n[OK] RAG demo 完成（chroma 后端 · hybrid 检索：真语义 BGE 编码器 + BM25 + RRF）")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_main(argv))
    except Exception as exc:
        print(f"[FAIL] RAG demo 运行失败: {exc!r}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
