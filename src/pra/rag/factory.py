"""RAG 索引装配（rag/factory.py）—— build_policy_index / build_case_index。

注入点（rag-implementation-plan.md §4.1/§4.2 的替换位约定）：
- corpus 路径：缺省取 rag/corpus/ 内静态 JSON（R-5：MVP JSON corpus + 内存索引）；
- embedder：缺省 ``MockHashEmbedder``（R-2：确定性 mock；Phase 2 换本地模型）；
- mode / weights：三模式可切换、权重可配（R-6：不预设 Hybrid 最优）。

装配开关（docs/06-rag-phase2-qdrant-bge.md §2.3，Phase 2 拍板；docs/10 §2 追加 chroma）：
- ``backend="local"``（**默认**）= 既有 ``RagPolicyIndex/RagCaseIndex``（numpy 余弦
  内存检索）——默认路径与改动前**逐字节等价**（v1/v2 回归 digest 零变化）；
- ``backend="qdrant"`` = ``QdrantPolicyIndex/QdrantCaseIndex``（qdrant-client 进程内
  模式做「向量存储 + 余弦打分」，元数据过滤/BM25/融合/Top-K 仍走 Python 侧同口径），
  **显式开启才需要 qdrant-client**：qdrant_index 模块在该分支内延迟 import，无
  qdrant-client 环境不 import 失败（P2-5）。另配 ``qdrant_client``（外部注入，测试
  用）/ ``location``（自建 client：":memory:" 默认 / path= / url=）/ ``collection_prefix``。
- ``backend="chroma"`` = ``ChromaPolicyIndex/ChromaCaseIndex``（docs/10：ChromaDB cosine
  + LlamaIndex VectorRetriever / BM25Retriever(jieba) / QueryFusionRetriever-RRF），同样
  **分支内延迟 import**（默认 local 路径零额外依赖、零额外 import）。另配
  ``chroma_client``（外部注入，如 ``EphemeralClient``）/ ``chroma_host``+``chroma_port``
  （缺省 ``127.0.0.1:8001`` 本机服务端）/ ``chroma_ephemeral``（进程内内存库，离线用）。

供 ``pra.tools.build_tools(data_source="rag", rag_backend=...)`` 与评测 RAG 世界
（agent_scheme.make_rag_world_tools）注入 —— 上层只依赖本函数返回的索引（实现
tools 层 Protocol；local/qdrant/chroma 三实现均满足同一窄接口）。
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

    ``rows`` 显式注入时优先（跳过文件 IO，测试/程序化构造用）。
    ``backend`` 开关（模块 docstring）：缺省 ``"local"`` 走既有 ``RagPolicyIndex``
    （与改动前等价）；``"qdrant"`` 走 ``QdrantPolicyIndex``、``"chroma"`` 走
    ``ChromaPolicyIndex``（两者均**分支内延迟 import**，仅显式开启才拉起对应依赖），
    并各自透传专属装配参数。返回类型标注为 ``RagPolicyIndex``（既有默认实现），实际
    可能是三者之一 —— 均实现 tools 层 ``PolicyIndex`` Protocol。
    """
    record_rows, _meta = _resolve_rows(rows, corpus_path, load_policies)
    if backend == "qdrant":
        # 延迟 import：仅显式 qdrant 后端才拉起 qdrant_index（缺省 local 路径零额外 import）。
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
        # bm25s / jieba）。缺省 local 路径不 import 其中任何一个。
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

    ``rows`` 显式注入时优先（跳过文件 IO，测试/程序化构造用）。
    ``backend`` 开关语义与 ``build_policy_index`` 相同（模块 docstring / docs/10 §2）：
    缺省 ``"local"`` 逐字节等价；``"qdrant"`` / ``"chroma"`` 均延迟 import 对应实现并
    透传其专属装配参数。
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
