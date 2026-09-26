"""第三方重依赖的延迟 import 边界：``chroma()`` / ``llama()``。"""

from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import Any

__all__ = ["chroma", "llama"]


def chroma() -> Any:
    """返回 chromadb 模块（在函数体内 import）。"""
    try:
        import chromadb
    except ImportError as exc:  # pragma: no cover — 触发路径仅在显式开启 chroma 后端
        raise RuntimeError(
            "chroma 后端需要 chromadb 与 llama-index 集成包：请运行 `uv sync --extra rag` 安装。"
        ) from exc
    return chromadb


@functools.cache
def llama() -> SimpleNamespace:
    """返回 LlamaIndex 装配面（``SimpleNamespace``；进程内缓存复用）。"""
    try:
        from llama_index.core import VectorStoreIndex
        from llama_index.core.llms import MockLLM
        from llama_index.core.retrievers import BaseRetriever, QueryFusionRetriever
        from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
        from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle, TextNode
        from llama_index.core.vector_stores import (
            FilterCondition,
            FilterOperator,
            MetadataFilter,
            MetadataFilters,
        )
        from llama_index.core.vector_stores.utils import (
            metadata_dict_to_node,
            node_to_metadata_dict,
        )
        from llama_index.vector_stores.chroma import ChromaVectorStore
    except ImportError as exc:  # pragma: no cover — 触发路径仅在显式开启 chroma 后端
        raise RuntimeError(
            "RAG 检索需要 llama-index-core / llama-index-vector-stores-chroma："
            "请运行 `uv sync --extra rag` 安装。"
        ) from exc
    return SimpleNamespace(
        BaseRetriever=BaseRetriever,
        ChromaVectorStore=ChromaVectorStore,
        FUSION_MODES=FUSION_MODES,
        FilterCondition=FilterCondition,
        FilterOperator=FilterOperator,
        MetadataFilter=MetadataFilter,
        MetadataFilters=MetadataFilters,
        MetadataMode=MetadataMode,
        MockLLM=MockLLM,
        NodeWithScore=NodeWithScore,
        QueryBundle=QueryBundle,
        QueryFusionRetriever=QueryFusionRetriever,
        TextNode=TextNode,
        VectorStoreIndex=VectorStoreIndex,
        metadata_dict_to_node=metadata_dict_to_node,
        node_to_metadata_dict=node_to_metadata_dict,
    )
