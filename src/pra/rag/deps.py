"""第三方重依赖的延迟 import 边界（全仓唯一）—— ``chroma()`` / ``llama()``。

chromadb / llama_index 只在本模块的**函数体内** import，故 ``import pra.rag`` / ``import pra.tools``
不把它们拉进 ``sys.modules``（CI 用子进程守着这条红线）；缺 rag extra 时抛出带指引的
``RuntimeError``，不静默降级成无关结果。
"""

from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import Any

__all__ = ["chroma", "llama"]


def chroma() -> Any:
    """延迟 import chromadb（**函数体内** import —— 顶层 import 会破坏默认路径零依赖红线）。"""
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover — 触发路径仅在显式开启 chroma 后端
        raise RuntimeError(
            "chroma 后端需要 chromadb 与 llama-index 集成包：请运行 `uv sync --extra rag` 安装。"
        ) from exc
    return chromadb


@functools.cache
def llama() -> SimpleNamespace:
    """延迟 import 的 LlamaIndex 装配面（进程内首个构造/检索时拉起，之后缓存复用）。

    只 import 具体集成（core + vector-stores-chroma + retrievers-bm25），不引伞包 ``llama-index``
    （伞包会拖进 llms-openai / embeddings-openai 等不用的集成）。用属性访问（``llama().TextNode``）：
    缓存对象是全局单例，故无需逐层穿透。
    """
    try:
        from llama_index.core import VectorStoreIndex
        from llama_index.core.indices.vector_store.retrievers import (
            VectorIndexRetriever,
        )
        from llama_index.core.schema import QueryBundle, TextNode
        from llama_index.retrievers.bm25 import BM25Retriever
        from llama_index.vector_stores.chroma import ChromaVectorStore
    except ImportError as exc:  # pragma: no cover — 触发路径仅在显式开启 chroma 后端
        raise RuntimeError(
            "RAG 检索需要 llama-index-core / llama-index-vector-stores-chroma / "
            "llama-index-retrievers-bm25：请运行 `uv sync --extra rag` 安装。"
        ) from exc
    return SimpleNamespace(
        BM25Retriever=BM25Retriever,
        ChromaVectorStore=ChromaVectorStore,
        QueryBundle=QueryBundle,
        TextNode=TextNode,
        VectorIndexRetriever=VectorIndexRetriever,
        VectorStoreIndex=VectorStoreIndex,
    )
