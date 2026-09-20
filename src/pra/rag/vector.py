"""向量检索器 —— 直接读 Chroma 原生 cosine ``distance`` 自算 ``1 − distance``（向量路唯一实现）。"""

from __future__ import annotations

import math
from typing import Any

from pra.rag.deps import llama
from pra.rag.retrieval import _RetrievalContext

__all__ = ["make_vector_retriever"]


def make_vector_retriever(
    ctx: _RetrievalContext,
    collection: Any,
) -> Any:
    """向量检索器：Chroma cosine 距离，分数 = ``1 − distance``。

    ``ChromaVectorStore.query`` 的分是 ``exp(-distance)`` —— **另一套映射、不是余弦**，故取数走
    **Chroma 原生 ``collection.query``** 自己算 ``1 − distance``；节点由 node id 直接映射，不经
    ``_node_content`` JSON 反序列化 —— 保持节点为原始权威对象（Chroma 回读重建会让 node 身份
    漂移，与按 ``node.hash`` 去重的融合冲突）。

    ★ ``ctx.node_ids`` = 候选集的 **node id**（精确候选集）：候选 id 交给 Chroma 的 ``ids=`` 参数，
    让打分域恰好是候选集 —— 否则非候选行会按距离抢占名额、把目标文档挤出 Top-N（漏召回）。
    """

    class _ChromaCosineRetriever(llama().BaseRetriever):

        def __init__(self) -> None:
            super().__init__()
            self._collection = collection
            self._emb = ctx.embed_model
            # Chroma 返回的是 node id（`_node_id()` 派生），不是 corpus 行键 —— 映射键必须用 node id。
            self._by_id = {nid: node for nid, node in zip(ctx.node_ids, ctx.nodes)}
            #: 精确候选 id（NodeWithScore 只允许这些 id 出现）。
            self._candidate_ids = list(ctx.node_ids)

        def _query_once(self, query_embedding: list[float]) -> tuple[list[str], list[float]]:
            """原生查询一次：只取候选 ``ids=``，不假定返回数量。"""
            kwargs: dict[str, Any] = {
                "query_embeddings": [list(query_embedding)],
                "include": ["distances"],
                "ids": list(self._candidate_ids),
                "n_results": max(
                    1, min(len(self._candidate_ids), int(self._collection.count()))
                ),
            }
            result = self._collection.query(**kwargs)
            return (
                list((result.get("ids") or [[]])[0]),
                list((result.get("distances") or [[]])[0]),
            )

        def _retrieve(self, query_bundle: Any) -> list[Any]:
            if self._collection is None or not self._by_id:
                return []
            query_embedding = self._emb.get_query_embedding(query_bundle.query_str)
            ids, distances = self._query_once(query_embedding)
            return [
                llama().NodeWithScore(node=self._by_id[nid], score=_cosine_from_distance(d))
                for nid, d in zip(ids, distances)
            ]

    return _ChromaCosineRetriever()


def _cosine_from_distance(distance: float) -> float:
    """Chroma cosine ``distance`` → 余弦相似度 ``1 − distance``。

    实测：``[1,0,0]`` vs ``[0.9,0.1,0]`` → ``distance = 0.006116271`` 而 ``1 − cos`` =
    0.00611627；``[1,0,0]`` vs ``[1,1,0]`` → ``distance = 0.29289323`` ↔ ``1 − cos`` =
    0.29289322。Chroma 对入库向量做过 L2 归一化、查询向量不做，其 cosine 距离即 ``1 − 余弦``。

    夹到 ``[0,1]`` 仅作防御：浮点尾差可能给出 ``-1e-9`` / ``1+1e-9``，而
    ``CaseHit.retrieval_score`` 约束 ``ge=0, le=1``（不夹会让整个检索抛 ValidationError）；
    NaN（零向量等退化输入）按 0 计。⚠️ 这是**向量路的检索分**；hybrid 路是 RRF 融合分，不同量纲。
    """
    value = float(distance)
    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, 1.0 - value))
