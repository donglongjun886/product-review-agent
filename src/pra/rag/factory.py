"""RAG 索引装配 —— ``build_policy_index`` / ``build_case_index``。

注入点：corpus 路径（缺省取 rag/corpus/ 内静态 JSON，失败即报错不静默）；编码器 —— local 后端
收 ``embedder``（自家 ``Embedder``，缺省 ``MockHashEmbedder``）、chroma 后端收 ``embedding_model``
（LlamaIndex ``BaseEmbedding``，缺省 None → chroma 类内自建 fastembed 集成）；mode / weights
（三模式可切、权重可配）。

``backend`` 开关：``"local"``（**默认**）= 纯 Python 余弦内存检索；``"chroma"`` = Chroma 实现
（ChromaDB cosine + LlamaIndex 检索器 + RRF），**分支内延迟 import**，默认 local 路径不引入
chromadb / llama_index / bm25s / jieba 中的任何一个包。上层只依赖返回的索引（两者都实现
tools 层 Protocol）。
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


def _validate_encoder_args(
    backend: Literal["local", "chroma"],
    embedder: Any | None,
    embedding_model: Any | None,
) -> None:
    """编码器参数闸（**唯一把关点**）：backend 与编码器参数必须严格配对，错配当场报错。

    两个协议互不兼容 —— 自家 ``Embedder`` 只有 ``embed()``、LlamaIndex ``BaseEmbedding`` 只有
    ``get_text_embedding()``，**没有任何对象能同时服务两侧**，故不做"按 backend 自动分发"：
    那只会把错误从入口推迟到运行期（``AttributeError: 'MockHashEmbedder' object has no
    attribute 'get_text_embedding'``）。闸设入口，错误当场可见。
    """
    if backend == "chroma" and embedder is not None:
        raise ValueError(
            "chroma 后端不接收 embedder，请改用 embedding_model=（embedder 只服务 local）"
        )
    if backend == "local" and embedding_model is not None:
        raise ValueError(
            "local 后端不接收 embedding_model，请改用 embedder=（embedding_model 只服务 chroma）"
        )


def build_policy_index(
    *,
    corpus_path: str | Path | None = None,
    rows: Iterable[Any] | None = None,
    embedder: Embedder | None = None,
    embedding_model: Any | None = None,
    mode: RetrievalMode = "hybrid",
    weights: tuple[float, float] = DEFAULT_WEIGHTS,
    backend: Literal["local", "chroma"] = "local",
    collection_prefix: str | None = None,
    chroma_client: Any | None = None,
    chroma_host: str = "127.0.0.1",
    chroma_port: int = 8001,
    chroma_ephemeral: bool = False,
) -> RagPolicyIndex:
    """构造 PolicyIndex（corpus_path 缺省 = rag/corpus/policies.json）。

    ``rows`` 显式注入时优先（跳过文件 IO）。``backend`` 缺省 ``"local"`` 走 ``RagPolicyIndex``
    （收 ``embedder``）；``"chroma"`` 走 ``ChromaPolicyIndex``（**分支内延迟 import**，收
    ``embedding_model``，缺省 None → 类内自建 ``build_embedding_model("fastembed")``），并透传
    专属装配参数。编码器与 backend 必须配对（local↔``embedder``、chroma↔``embedding_model``），
    错配由 :func:`_validate_encoder_args` 当场抛 ``ValueError``。返回类型标注为 ``RagPolicyIndex``，
    实际可能是两者之一 —— 均实现 tools 层 ``PolicyIndex`` Protocol。
    """
    _validate_encoder_args(backend, embedder, embedding_model)
    record_rows, _meta = _resolve_rows(rows, corpus_path, load_policies)
    if backend == "chroma":
        # 延迟 import：仅显式 chroma 后端才拉起 chroma_backend（+chromadb / llama-index /
        # bm25s / jieba）。
        from pra.rag.chroma_backend import ChromaPolicyIndex

        return ChromaPolicyIndex(
            record_rows,
            embedding_model=embedding_model,
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
    embedding_model: Any | None = None,
    mode: RetrievalMode = "hybrid",
    weights: tuple[float, float] = DEFAULT_WEIGHTS,
    backend: Literal["local", "chroma"] = "local",
    collection_prefix: str | None = None,
    chroma_client: Any | None = None,
    chroma_host: str = "127.0.0.1",
    chroma_port: int = 8001,
    chroma_ephemeral: bool = False,
) -> RagCaseIndex:
    """构造 CaseIndex（corpus_path 缺省 = rag/corpus/cases.json）。

    ``rows`` 显式注入时优先（跳过文件 IO）。编码器按后端路由同 ``build_policy_index``
    （local 收 ``embedder``、chroma 收 ``embedding_model``），其余语义一致。
    """
    _validate_encoder_args(backend, embedder, embedding_model)
    record_rows, _meta = _resolve_rows(rows, corpus_path, load_cases)
    if backend == "chroma":
        from pra.rag.chroma_backend import ChromaCaseIndex

        return ChromaCaseIndex(
            record_rows,
            embedding_model=embedding_model,
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
