"""向量路取数：LlamaIndex ``ChromaVectorStore`` + ``VectorIndexRetriever``。"""

from __future__ import annotations

from typing import Any

from pra.rag.deps import llama

__all__ = ["vector_retrieve"]


def vector_retrieve(
    index: Any,
    query_bundle: Any,
    *,
    top_k: int,
    filters: Any | None = None,
) -> list[tuple[str, float]]:
    """按向量相似度检索，返回 ``[(node id, 分数)]``。

    ``filters`` 非空时下推给 Chroma ``where``，候选在库侧收窄。分数是 ``ChromaVectorStore``
    的原值 ``exp(-distance)``（越大越近），本函数不做换算。
    """
    kwargs: dict[str, Any] = {"index": index, "similarity_top_k": max(1, int(top_k))}
    if filters is not None:
        kwargs["filters"] = filters
    retriever = llama().VectorIndexRetriever(**kwargs)
    return [(n.node.node_id, float(n.score or 0.0)) for n in retriever.retrieve(query_bundle)]
