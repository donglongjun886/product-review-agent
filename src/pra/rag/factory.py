"""RAG 索引装配（rag/factory.py）—— build_policy_index / build_case_index。

注入点（rag-implementation-plan.md §4.1/§4.2 的替换位约定）：
- corpus 路径：缺省取 rag/corpus/ 内静态 JSON（R-5：MVP JSON corpus + 内存索引）；
- embedder：缺省 ``MockHashEmbedder``（R-2：确定性 mock；Phase 2 换本地模型）；
- mode / weights：三模式可切换、权重可配（R-6：不预设 Hybrid 最优）。

装配开关（docs/06-rag-phase2-qdrant-bge.md §2.3，Phase 2 拍板）：
- ``backend="local"``（**默认**）= 既有 ``RagPolicyIndex/RagCaseIndex``（numpy 余弦
  内存检索）——默认路径与改动前**逐字节等价**（v1/v2 回归 digest 零变化）；
- ``backend="qdrant"`` = ``QdrantPolicyIndex/QdrantCaseIndex``（qdrant-client 进程内
  模式做「向量存储 + 余弦打分」，元数据过滤/BM25/融合/Top-K 仍走 Python 侧同口径），
  **显式开启才需要 qdrant-client**：qdrant_index 模块在该分支内延迟 import，无
  qdrant-client 环境不 import 失败（P2-5）。另配 ``qdrant_client``（外部注入，测试
  用）/ ``location``（自建 client：":memory:" 默认 / path= / url=）/ ``collection_prefix``。

供 ``pra.tools.build_tools(data_source="rag", rag_backend=...)`` 与评测 RAG 世界
（agent_scheme.make_rag_world_tools）注入 —— 上层只依赖本函数返回的索引（实现
tools 层 Protocol；local/qdrant 两实现检索语义同口径）。
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
    backend: Literal["local", "qdrant"] = "local",
    qdrant_client: Any | None = None,
    location: str | Path = ":memory:",
    collection_prefix: str | None = None,
) -> RagPolicyIndex:
    """构造 PolicyIndex（corpus_path 缺省 = rag/corpus/policies.json）。

    ``rows`` 显式注入时优先（跳过文件 IO，测试/程序化构造用）。
    ``backend`` 开关（模块 docstring）：缺省 ``"local"`` 走既有 ``RagPolicyIndex``
    （与改动前等价）；``"qdrant"`` 走 ``QdrantPolicyIndex``（延迟 import，仅显式
    开启才需要 qdrant-client），并透传 ``qdrant_client``/``location``/
    ``collection_prefix`` 三个 qdrant 专用装配参数。两实现的检索语义（过滤边界/
    打分/tie-break）同口径（docs/06 §2.1）；类型为 RagPolicyIndex 或
    QdrantPolicyIndex，均实现 tools 层 ``PolicyIndex`` Protocol。
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
    backend: Literal["local", "qdrant"] = "local",
    qdrant_client: Any | None = None,
    location: str | Path = ":memory:",
    collection_prefix: str | None = None,
) -> RagCaseIndex:
    """构造 CaseIndex（corpus_path 缺省 = rag/corpus/cases.json）。

    ``rows`` 显式注入时优先（跳过文件 IO，测试/程序化构造用）。
    ``backend`` 开关语义与 ``build_policy_index`` 相同（模块 docstring / docs/06
    §2.3）：缺省 ``"local"`` 逐字节等价；``"qdrant"`` 延迟 import
    ``QdrantCaseIndex`` 并透传 qdrant 专用参数。
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
    return RagCaseIndex(
        record_rows,
        embedder=embedder or MockHashEmbedder(),
        mode=mode,
        weights=weights,
    )
