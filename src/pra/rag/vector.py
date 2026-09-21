"""向量路取数：LlamaIndex ``ChromaVectorStore`` + ``VectorIndexRetriever``。"""

from __future__ import annotations

from typing import Any

from pra.rag.deps import llama

__all__ = ["make_vector_retriever"]


def make_vector_retriever(index: Any, *, top_k: int, filters: Any | None = None) -> Any:
    """向量检索器；``filters`` 非空时下推给 Chroma ``where``，候选在库侧收窄。

    ``top_k`` 传「所在打分域的全长」—— 截断由调用方在融合之后统一做。
    """
    kwargs: dict[str, Any] = {"index": index, "similarity_top_k": max(1, int(top_k))}
    if filters is not None:
        kwargs["filters"] = filters
    return llama().VectorIndexRetriever(**kwargs)
