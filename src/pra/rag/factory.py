"""RAG 索引装配 —— ``build_policy_index`` / ``build_case_index``。

注入点：corpus 路径（缺省取 rag/corpus/ 内静态 JSON，失败即报错不静默）；``embedding_model``
（LlamaIndex ``BaseEmbedding``，缺省 None → chroma 类内自建 fastembed 集成）；mode（三模式可切）；
chroma 连接参数（client / host / port / ephemeral / collection 前缀）。

检索后端只有 chroma（ChromaDB cosine + LlamaIndex 检索器 + RRF），``chroma_backend`` 在函数体内
**延迟 import**，故本模块被 tools 层 import 时顶层零额外依赖；缺 ``--extra rag`` 时抛出带指引的
``RuntimeError``，不静默降级。
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pra.rag.corpus import CORPUS_DIR, load_cases, load_policies
from pra.rag.retrieval import RetrievalMode

if TYPE_CHECKING:  # 仅注解：本模块 import 期不拉起 tools 子包
    from pra.tools.case_search.tool import CaseIndex
    from pra.tools.policy_search.tool import PolicyIndex

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
    embedding_model: Any | None = None,
    mode: RetrievalMode = "hybrid",
    collection_prefix: str | None = None,
    chroma_client: Any | None = None,
    chroma_host: str = "127.0.0.1",
    chroma_port: int = 8001,
    chroma_ephemeral: bool = False,
) -> PolicyIndex:
    """构造 PolicyIndex（corpus_path 缺省 = rag/corpus/policies.json）。

    ``rows`` 显式注入时优先（跳过文件 IO）。``embedding_model`` 缺省 None → 类内自建
    ``build_embedding_model("fastembed")``（真语义、需 rag extra 与已缓存模型）；传
    ``build_embedding_model("test")`` 可得确定性离线编码器。
    """
    record_rows, _meta = _resolve_rows(rows, corpus_path, load_policies)
    # 延迟 import：chroma / llama_index / bm25s / jieba 仅在真正装配索引时才拉起。
    from pra.rag.chroma_backend import ChromaPolicyIndex

    return ChromaPolicyIndex(
        record_rows,
        embedding_model=embedding_model,
        mode=mode,
        chroma_client=chroma_client,
        host=chroma_host,
        port=chroma_port,
        ephemeral=chroma_ephemeral,
        collection_prefix=collection_prefix,
    )


def build_case_index(
    *,
    corpus_path: str | Path | None = None,
    rows: Iterable[Any] | None = None,
    embedding_model: Any | None = None,
    mode: RetrievalMode = "hybrid",
    collection_prefix: str | None = None,
    chroma_client: Any | None = None,
    chroma_host: str = "127.0.0.1",
    chroma_port: int = 8001,
    chroma_ephemeral: bool = False,
) -> CaseIndex:
    """构造 CaseIndex（corpus_path 缺省 = rag/corpus/cases.json）；参数语义同 ``build_policy_index``。"""
    record_rows, _meta = _resolve_rows(rows, corpus_path, load_cases)
    from pra.rag.chroma_backend import ChromaCaseIndex

    return ChromaCaseIndex(
        record_rows,
        embedding_model=embedding_model,
        mode=mode,
        chroma_client=chroma_client,
        host=chroma_host,
        port=chroma_port,
        ephemeral=chroma_ephemeral,
        collection_prefix=collection_prefix,
    )
