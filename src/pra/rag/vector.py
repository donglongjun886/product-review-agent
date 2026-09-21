"""向量检索器 —— LlamaIndex ``ChromaVectorStore`` + ``VectorIndexRetriever`` 取数（向量路唯一实现）。"""

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
    node_ids: list[str],
) -> list[tuple[str, float]]:
    """向量路取数：``[(node id, 余弦相似度)]``（分数 = ``1 − distance``，见
    :func:`_cosine_from_similarity`）。

    ★ ``node_ids`` = 候选集的 **node id**（精确候选集），经 ``vector_store_kwargs={"ids": ...}``
    透传到 Chroma 原生 ``collection.query``，让打分域恰好是候选集 —— 否则非候选行会按距离抢占
    名额、把目标文档挤出 Top-N（漏召回，**只在带过滤时暴露**）。
    ⚠️ ``VectorStoreQuery.node_ids`` 字段**不被** ``ChromaVectorStore`` 消费（`query()` 只读
    ``query_embedding`` / ``similarity_top_k`` / ``filters`` / ``mode``），故候选只能走 kwargs 旁路。

    ★ 取「候选数」而非 ``top_k``：完整候选排名交给上层做 tie-break 与 RRF 融合，截断统一留在
    ``index`` 层（``_retrieve_ranked``）。

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
    retriever = llama_.VectorIndexRetriever(
        index=index,
        similarity_top_k=max(1, len(node_ids)),
        vector_store_kwargs={"ids": list(node_ids)},
    )
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
