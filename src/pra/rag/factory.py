"""RAG 索引装配（rag/factory.py）—— build_policy_index / build_case_index。

注入点（rag-implementation-plan.md §4.1/§4.2 的替换位约定）：
- corpus 路径：缺省取 rag/corpus/ 内静态 JSON（R-5：MVP JSON corpus + 内存索引）；
- embedder：缺省 ``MockHashEmbedder``（R-2：确定性 mock；Phase 2 换本地模型）；
- mode / weights：三模式可切换、权重可配（R-6：不预设 Hybrid 最优）。

供 ``pra.tools.build_tools(data_source="rag")`` 与评测 RAG 世界
（agent_scheme.make_rag_world_tools）注入 —— 上层只依赖本函数返回的
``RagPolicyIndex`` / ``RagCaseIndex``（实现 tools 层 Protocol）。
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pra.rag.corpus import CORPUS_DIR, load_cases, load_policies
from pra.rag.embedder import Embedder, MockHashEmbedder
from pra.rag.index import RagCaseIndex, RagPolicyIndex
from pra.rag.retrieval import DEFAULT_WEIGHTS, RetrievalMode

__all__ = [
    "CASES_DEFAULT",
    "POLICIES_DEFAULT",
    "build_case_index",
    "build_policy_index",
]

POLICIES_DEFAULT = CORPUS_DIR / "policies.json"
CASES_DEFAULT = CORPUS_DIR / "cases.json"


def _resolve_rows(
    rows: Iterable[Any] | None, corpus_path: str | Path | None, loader
) -> tuple[list[Any], dict]:
    """(显式 rows) or (corpus JSON 文件) → (记录列表, meta)。"""
    if rows is not None:
        return list(rows), {}
    return loader(corpus_path)


def build_policy_index(
    *,
    corpus_path: str | Path | None = None,
    rows: Iterable[Any] | None = None,
    embedder: Embedder | None = None,
    mode: RetrievalMode = "hybrid",
    weights: tuple[float, float] = DEFAULT_WEIGHTS,
) -> RagPolicyIndex:
    """构造真实 PolicyIndex（corpus_path 缺省 = rag/corpus/policies.json）。

    ``rows`` 显式注入时优先（跳过文件 IO，测试/程序化构造用）。
    """
    record_rows, _meta = _resolve_rows(rows, corpus_path, load_policies)
    return RagPolicyIndex(
        record_rows,
        embedder=embedder or MockHashEmbedder(),
        mode=mode,
        weights=weights,
    )


def build_case_index(
    *,
    corpus_path: str | Path | None = None,
    rows: Iterable[Any] | None = None,
    embedder: Embedder | None = None,
    mode: RetrievalMode = "hybrid",
    weights: tuple[float, float] = DEFAULT_WEIGHTS,
) -> RagCaseIndex:
    """构造真实 CaseIndex（corpus_path 缺省 = rag/corpus/cases.json）。

    ``rows`` 显式注入时优先（跳过文件 IO，测试/程序化构造用）。
    """
    record_rows, _meta = _resolve_rows(rows, corpus_path, load_cases)
    return RagCaseIndex(
        record_rows,
        embedder=embedder or MockHashEmbedder(),
        mode=mode,
        weights=weights,
    )
