"""RAG 索引装配 —— ``build_policy_index`` / ``build_case_index``。

注入点：corpus 路径（缺省取 rag/corpus/ 内静态 JSON，失败即报错不静默）、embedder（缺省
``MockHashEmbedder``）、mode / weights（三模式可切、权重可配）。

``backend`` 开关：``"local"``（**默认**）= numpy 余弦内存检索；``"qdrant"`` = Qdrant 实现
（向量存储 + 余弦打分，过滤/BM25/融合/Top-K 仍在 Python 侧同口径）；``"chroma"`` = Chroma
实现（ChromaDB cosine + LlamaIndex 检索器 + RRF）。后两者都**分支内延迟 import**，默认 local
路径不引入其中任何一个包。上层只依赖返回的索引（三者都实现 tools 层 Protocol）。
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

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
    backend: Literal["local", "qdrant", "chroma"] = "local",
    qdrant_client: Any | None = None,
    location: str | Path = ":memory:",
    collection_prefix: str | None = None,
    chroma_client: Any | None = None,
    chroma_host: str = "127.0.0.1",
    chroma_port: int = 8001,
    chroma_ephemeral: bool = False,
) -> RagPolicyIndex:
    """构造 PolicyIndex（corpus_path 缺省 = rag/corpus/policies.json）。

    ``rows`` 显式注入时优先（跳过文件 IO）。``backend`` 缺省 ``"local"`` 走 ``RagPolicyIndex``；
    ``"qdrant"`` / ``"chroma"`` 走对应实现（两者均**分支内延迟 import**），并透传专属装配参数。
    返回类型标注为 ``RagPolicyIndex``，实际可能是三者之一 —— 均实现 tools 层 ``PolicyIndex``
    Protocol。
    """
    record_rows, _meta = _resolve_rows(rows, corpus_path, load_policies)
    if backend == "qdrant":
        # 延迟 import：仅显式 qdrant 后端才拉起 qdrant_index。
        from pra.rag.qdrant_index import QdrantPolicyIndex

        return QdrantPolicyIndex(
            record_rows,
            embedder=embedder or MockHashEmbedder(),
            mode=mode,
            weights=weights,
            qdrant_client=qdrant_client,
            location=location,
            collection_prefix=collection_prefix,
        )
    if backend == "chroma":
        # 延迟 import：仅显式 chroma 后端才拉起 chroma_backend（+chromadb / llama-index /
        # bm25s / jieba）。
        from pra.rag.chroma_backend import ChromaPolicyIndex

        return ChromaPolicyIndex(
            record_rows,
            embedder=embedder or MockHashEmbedder(),
            mode=mode,
            weights=weights,
            chroma_client=chroma_client,
            host=chroma_host,
            port=chroma_port,
            ephemeral=chroma_ephemeral,
            collection_prefix=collection_prefix,
        )
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
    backend: Literal["local", "qdrant", "chroma"] = "local",
    qdrant_client: Any | None = None,
    location: str | Path = ":memory:",
    collection_prefix: str | None = None,
    chroma_client: Any | None = None,
    chroma_host: str = "127.0.0.1",
    chroma_port: int = 8001,
    chroma_ephemeral: bool = False,
) -> RagCaseIndex:
    """构造 CaseIndex（corpus_path 缺省 = rag/corpus/cases.json）。

    ``rows`` 显式注入时优先（跳过文件 IO）。``backend`` 语义与 ``build_policy_index`` 相同。
    """
    record_rows, _meta = _resolve_rows(rows, corpus_path, load_cases)
    if backend == "qdrant":
        from pra.rag.qdrant_index import QdrantCaseIndex

        return QdrantCaseIndex(
            record_rows,
            embedder=embedder or MockHashEmbedder(),
            mode=mode,
            weights=weights,
            qdrant_client=qdrant_client,
            location=location,
            collection_prefix=collection_prefix,
        )
    if backend == "chroma":
        from pra.rag.chroma_backend import ChromaCaseIndex

        return ChromaCaseIndex(
            record_rows,
            embedder=embedder or MockHashEmbedder(),
            mode=mode,
            weights=weights,
            chroma_client=chroma_client,
            host=chroma_host,
            port=chroma_port,
            ephemeral=chroma_ephemeral,
            collection_prefix=collection_prefix,
        )
    return RagCaseIndex(
        record_rows,
        embedder=embedder or MockHashEmbedder(),
        mode=mode,
        weights=weights,
    )
