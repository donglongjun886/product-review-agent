"""向量路取数 —— LlamaIndex ``ChromaVectorStore`` + ``VectorIndexRetriever``（向量路唯一实现）。

过滤**下推给 Chroma**：业务过滤由上层编译成 ``MetadataFilters``（``index._filters_of``），经
``VectorIndexRetriever`` → ``ChromaVectorStore.query`` → 原生 ``collection.query(where=...)``，
在库侧收窄打分域。打分域 = 库内全集 ∩ ``where``。
"""

from __future__ import annotations

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
    """向量路取数：``[(node id, 相似度分)]`` —— 分数**直接采库口径**，不换算。

    ★ ``ChromaVectorStore`` 给的分是 ``exp(-distance)``（cosine 空间即 ``exp(-(1 − cos))``，
    ⊂ ``(0, 1]``、越大越近；浮点下可能微超 1 —— 下游不设域约束，原样容忍）。这里**原样透传**，
    不做 ``1 − distance`` 换算：

    ① **排名不需要**——任何 distance 的单调降函数都给出同一顺序，换算是纯冗余；
    ② **决策不读**——该分往下只变成 ``CaseHit.retrieval_score``（渲染给 LLM 的证据行 + 落库
       审计），``gate`` 对 ``CASE_PRECEDENT`` / ``POLICY_REF`` 只判**是否存在**、不读 value；
    ③ 换算反而有损——``1 − distance`` 在 ``d > 1`` 时被 clamp 塌成 0，低分区区分度全丢。

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
    # 原样透传，不钳位：``CaseHit.retrieval_score`` 不设取值域约束（见上方 ★），
    # cosine 距离在浮点下微负导致 ``exp(-d)`` 微超 1 属后端口径本身，不是需要修的越界。
    return [(n.node.node_id, float(n.score or 0.0)) for n in retriever.retrieve(query)]
