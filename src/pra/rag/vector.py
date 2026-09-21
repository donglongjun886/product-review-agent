"""向量路取数 —— LlamaIndex ``ChromaVectorStore`` + ``VectorIndexRetriever``（向量路唯一实现）。

过滤**下推给 Chroma**：业务过滤由上层编译成 ``MetadataFilters``（``index._filters_of``），经
``VectorIndexRetriever`` → ``ChromaVectorStore.query`` → 原生 ``collection.query(where=...)``，
在库侧收窄打分域。打分域 = 库内全集 ∩ ``where``。
"""

from __future__ import annotations

import math
from typing import Any

from pra.rag.deps import llama

__all__ = ["vector_retrieve"]


def vector_retrieve(
    index: Any,
    query_bundle: Any,
    *,
    embed_model: Any,
    top_k: int,
    filters: Any | None = None,
) -> list[tuple[str, float]]:
    """向量路取数：``[(node id, 余弦相似度)]``（分数 = ``1 − distance``，见
    :func:`_cosine_from_similarity`）。

    ★ ``filters``（``MetadataFilters``；``None`` = 无过滤）由 ``VectorIndexRetriever`` 透传到
    ``ChromaVectorStore.query`` 的 ``where`` —— 打分域在**库侧**收窄。
    ⚠️ 过滤语义因此有**两处**定义：本处 ``where`` 与 ``index._*_candidates`` 的 Python 谓词
    （后者服务 BM25 路 —— ``bm25s`` 内存索引没有 ``where``）。两者必须逐条等价，由
    ``tests/test_rag_retrieval.py::test_*_where_pushdown_equals_python_candidates`` 锁住；
    改任何一方都要同步改另一方与该用例。

    ★ ``top_k`` 传「库内全集行数」：完整排名要交给上层做 tie-break 与 RRF 融合，截断统一留在
    ``index`` 层（``_retrieve_ranked``）。带过滤的 HNSW 检索在候选不足时返回条数可能**少于**
    ``n_results``，取全集上界是这一层能拿到的最大值。

    ★ 查询向量**自算**后塞进 ``QueryBundle``：``VectorIndexRetriever`` 缺省走
    ``embed_model.get_agg_embedding_from_queries``（内部再过一次 ``mean_agg`` 聚合），与向量路既有
    口径不是同一条路径；自算可保证查询向量逐位一致（``_retrieve`` 见 ``query_bundle.embedding``
    非空即跳过编码）。
    """
    llama_ = llama()
    query = llama_.QueryBundle(
        query_str=query_bundle.query_str,
        embedding=embed_model.get_query_embedding(query_bundle.query_str),
    )
    kwargs: dict[str, Any] = {"index": index, "similarity_top_k": max(1, int(top_k))}
    if filters is not None:
        kwargs["filters"] = filters
    retriever = llama_.VectorIndexRetriever(**kwargs)
    return [
        (n.node.node_id, _cosine_from_similarity(n.score)) for n in retriever.retrieve(query)
    ]


def _cosine_from_similarity(similarity: float) -> float:
    """``ChromaVectorStore`` 的分 ``exp(-distance)`` → 余弦相似度 ``1 − distance``。

    包装层给的是 ``exp(-distance)``（``vector_stores/chroma/base.py`` 的 ``_query``），**另一套
    映射**，故反解 ``distance = -ln(similarity)`` 后算 ``1 − distance``（= ``1 + ln(similarity)``）。

    实测（cosine 空间，``[1,0,0]`` 对 ``[1,0,0]`` / ``[0.9,0.1,0]``）：包装层 score =
    ``1.0`` / ``0.9939024``，反解得 ``1 + ln(score)`` = ``1.0`` / ``0.99388373``，与原生
    ``1 − distance``（``0.0`` / ``0.006116271``）**逐位一致**（取 6 位后相同）。

    夹到 ``[0,1]`` 仅作防御：浮点尾差可能给出 ``-1e-9`` / ``1+1e-9``，而
    ``CaseHit.retrieval_score`` 约束 ``ge=0, le=1``（不夹会让整个检索抛 ValidationError）；
    NaN / 非正值（``exp`` 下溢、零向量等退化输入）按 0 计。
    ⚠️ 这是**向量路的检索分**；hybrid 路是 RRF 融合分，不同量纲。
    """
    value = float(similarity)
    if math.isnan(value) or value <= 0.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 + math.log(value)))
