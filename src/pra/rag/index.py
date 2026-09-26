"""Chroma 检索索引：``ChromaPolicyIndex`` / ``ChromaCaseIndex``（向量 + BM25 + RRF hybrid）。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from pra.rag.bm25 import make_bm25_retriever
from pra.rag.chroma_store import (
    ChromaConfig,
    _build_nodes,
    _collection_name,
    _normalize_rows,
    _open_collection,
    case_node_metadata,
    make_chroma_client,
    policy_node_metadata,
    risk_type_key,
)
from pra.rag.corpus.schema import CasePrecedentRecord, PolicyClauseRecord
from pra.rag.deps import llama
from pra.rag.dto import CaseHit, CaseSearchFilters, PolicyClauseHit, PolicySearchFilters
from pra.rag.retrieval import _RetrievalContext

__all__ = ["ChromaCaseIndex", "ChromaPolicyIndex"]

_FULL_CATEGORY = "全类目"


# ---------------------------------------------------------------------------
# 候选过滤
# ---------------------------------------------------------------------------


def _policy_candidates(
    rows: list[PolicyClauseRecord], filters: PolicySearchFilters, effective_only: bool
) -> list[int]:
    """候选行索引（按 ``effective_only`` / ``category`` / ``risk_type`` 过滤）。"""
    candidates: list[int] = []
    for i, r in enumerate(rows):
        if effective_only and r.status != "EFFECTIVE":
            continue
        if filters.category and r.category not in (None, filters.category, _FULL_CATEGORY):
            continue
        if filters.risk_type:
            wanted = set(filters.risk_type)
            if not (wanted & set(r.risk_type)):
                continue
        candidates.append(i)
    return candidates


def _case_candidates(rows: list[CasePrecedentRecord], filters: CaseSearchFilters) -> list[int]:
    candidates: list[int] = []
    for i, r in enumerate(rows):
        if filters.category and r.category != filters.category:
            continue
        if filters.risk_type:
            wanted = set(filters.risk_type)
            if not (wanted & set(r.risk_type)):
                continue
        candidates.append(i)
    return candidates


def _risk_type_clause(risk_types: list[Any]) -> Any:
    """``risk_type`` 交叠非空 → ``$or`` of ``rt_<值>: 1``。"""
    llama_ = llama()
    return llama_.MetadataFilters(
        condition=llama_.FilterCondition.OR,
        filters=[llama_.MetadataFilter(key=risk_type_key(t), value=1) for t in risk_types],
    )


def _where_from(clauses: list[Any]) -> Any | None:
    """过滤子句列表 → ``MetadataFilters``（多条套 ``$and``；空 → ``None``）。"""
    if not clauses:
        return None
    llama_ = llama()
    return llama_.MetadataFilters(condition=llama_.FilterCondition.AND, filters=clauses)


# ---------------------------------------------------------------------------
# Policy / Case 两个公开索引类
# ---------------------------------------------------------------------------


class _ChromaIndexBase:
    """两个 Chroma 索引的装配 / 检索骨架；子类声明 ``_kind`` / ``_record_type`` 与逐类钩子。"""

    _kind: str
    _record_type: type

    def __init__(
        self,
        rows: Iterable[Any],
        *,
        embedding_model: Any,
        config: ChromaConfig | None = None,
    ) -> None:
        self._rows: list[Any] = _normalize_rows(rows, self._record_type)
        if not self._rows:
            raise ValueError(f"{type(self).__name__} 语料为空：至少需要 1 行才能建库")
        cfg = config or ChromaConfig()
        self._config = replace(cfg, client=make_chroma_client(cfg))
        self._embed_model: Any = embedding_model
        doc_vectors = [
            self._embed_model.get_text_embedding(self._text_of(r)) for r in self._rows
        ]
        self._dim = len(doc_vectors[0])
        self.collection_name = _collection_name(cfg.collection_prefix, self._kind, self._dim)
        self._seed(doc_vectors)

    # -- 逐类钩子（子类实现）--------------------------------------------------

    def _text_of(self, row: Any) -> str:
        """检索文本。"""
        raise NotImplementedError

    def _key_of(self, row: Any) -> str:
        """corpus 行键。"""
        raise NotImplementedError

    def _meta_of(self, row: Any) -> dict[str, Any]:
        """node metadata。"""
        raise NotImplementedError

    def _filters_of(self, filters: Any) -> Any | None:
        """业务过滤 → Chroma ``where`` 表达式；``None`` = 无过滤。"""
        raise NotImplementedError

    # -- 装配 ---------------------------------------------------------------

    def _seed(self, doc_vectors: list[list[float]]) -> None:
        """建/复用 collection + 按 node id 先删后加重建。"""
        collection = _open_collection(self._config, name=self.collection_name)
        self._nodes, self._node_ids = _build_nodes(
            self._rows,
            collection=self.collection_name,
            key_of=self._key_of,
            text_of=self._text_of,
            meta_of=self._meta_of,
        )
        for node, vec in zip(self._nodes, doc_vectors):
            node.embedding = list(vec)
        store = llama().ChromaVectorStore(chroma_collection=collection)
        stale_ids = sorted(set(collection.get(include=[])["ids"]) - set(self._node_ids))
        if stale_ids:
            store.delete_nodes(node_ids=stale_ids)
        store.delete_nodes(node_ids=list(self._node_ids))
        store.add(self._nodes)
        self._vec_index = llama().VectorStoreIndex.from_vector_store(
            store, embed_model=self._embed_model
        )

    # -- 装配子件 -----------------------------------------------------------

    def _sub_context(self, candidates: list[int]) -> _RetrievalContext:
        """候选子集上的检索上下文。"""
        return _RetrievalContext(
            node_ids=[self._node_ids[i] for i in candidates],
            nodes=[self._nodes[i] for i in candidates],
            row_index=list(candidates),
        )

    def _full_context(self) -> _RetrievalContext:
        """全量语料上的检索上下文。"""
        return _RetrievalContext(
            node_ids=list(self._node_ids),
            nodes=list(self._nodes),
            row_index=list(range(len(self._node_ids))),
        )

    def _to_ranked(self, nodes: list[Any], ctx: _RetrievalContext) -> list[tuple[int, float]]:
        """检索结果 → ``[(行索引, 检索分)]``，按分降序、corpus 原序排序；``ctx`` 之外的 node 丢弃。"""
        ranked: list[tuple[int, float]] = []
        for item in nodes:
            try:
                position = ctx.node_ids.index(item.node.node_id)
            except ValueError:
                continue
            ranked.append((ctx.row_index[position], float(item.score or 0.0)))
        ranked.sort(key=lambda t: (-t[1], t[0]))
        return ranked

    def _make_vector_retriever(self, ctx: _RetrievalContext, filters: Any | None) -> Any:
        """向量路检索器（``filters`` 在库侧收窄）。"""
        return self._vec_index.as_retriever(
            similarity_top_k=max(1, len(ctx.node_ids)), filters=filters
        )

    def _rank_vector(
        self, ctx: _RetrievalContext, query_bundle: Any, filters: Any | None
    ) -> list[tuple[int, float]]:
        """向量路排名。"""
        return self._to_ranked(
            self._make_vector_retriever(ctx, filters).retrieve(query_bundle), ctx
        )

    def _rank_bm25(
        self, sub_ctx: _RetrievalContext, query_bundle: Any, top_k: int
    ) -> list[tuple[int, float]]:
        """BM25 路排名（Python 侧只喂候选 node）。"""
        retriever = make_bm25_retriever(sub_ctx, top_k)
        return self._to_ranked(retriever.retrieve(query_bundle), sub_ctx)

    def _rank_hybrid(
        self, query: str, candidates: list[int], filters: Any | None
    ) -> list[tuple[int, float]]:
        """hybrid 排名：两路检索器交给 ``QueryFusionRetriever`` 做 RRF 融合。"""
        full_ctx = self._full_context()
        sub_ctx = self._sub_context(candidates)
        fusion = llama().QueryFusionRetriever(
            retrievers=[
                self._make_vector_retriever(full_ctx, filters),
                make_bm25_retriever(sub_ctx, len(candidates)),
            ],
            llm=llama().MockLLM(),
            mode=llama().FUSION_MODES.RECIPROCAL_RANK,
            num_queries=1,
            use_async=False,
            similarity_top_k=max(1, len(full_ctx.node_ids)),
        )
        nodes = fusion.retrieve(llama().QueryBundle(query_str=query))
        return self._to_ranked(nodes, full_ctx)

    # -- 检索核心 -----------------------------------------------------------

    def _retrieve_ranked(
        self,
        query: str,
        candidates: list[int],
        filters: Any | None,
        *,
        top_k: int,
    ) -> list[tuple[int, float]]:
        """hybrid 检索 → ``[(行索引, 检索分)]``（排序后截断 Top-K）。"""
        if not candidates:
            return []
        return self._rank_hybrid(query, candidates, filters)[:top_k]


class ChromaPolicyIndex(_ChromaIndexBase):
    """``PolicyIndex`` Protocol 的 Chroma + LlamaIndex 实现。"""

    _kind = "policy"
    _record_type = PolicyClauseRecord

    def _text_of(self, row: PolicyClauseRecord) -> str:
        return f"{row.title}。{row.text}"

    def _key_of(self, row: PolicyClauseRecord) -> str:
        return row.clause_id

    def _meta_of(self, row: PolicyClauseRecord) -> dict[str, Any]:
        return policy_node_metadata(row)

    def _filters_of(
        self, filters: PolicySearchFilters, effective_only: bool = False
    ) -> Any | None:
        """业务过滤 → Chroma ``where``。"""
        llama_ = llama()
        clauses: list[Any] = []
        if effective_only:
            clauses.append(llama_.MetadataFilter(key="status", value="EFFECTIVE"))
        if filters.category:
            clauses.append(
                llama_.MetadataFilter(
                    key="category",
                    operator=llama_.FilterOperator.IN,
                    value=[filters.category, _FULL_CATEGORY],
                )
            )
        if filters.risk_type:
            clauses.append(_risk_type_clause(filters.risk_type))
        return _where_from(clauses)

    async def search(
        self,
        query: str,
        filters: PolicySearchFilters,
        top_k: int,
        effective_only: bool,
    ) -> list[PolicyClauseHit]:
        """检索政策条款（hybrid）；无命中 → ``[]``。"""
        candidates = _policy_candidates(self._rows, filters, effective_only)
        ranked = self._retrieve_ranked(
            query,
            candidates,
            self._filters_of(filters, effective_only),
            top_k=top_k,
        )
        return [
            PolicyClauseHit.model_validate(self._rows[i].model_dump(mode="json"))
            for i, _score in ranked
        ]


class ChromaCaseIndex(_ChromaIndexBase):
    """``CaseIndex`` Protocol 的 Chroma + LlamaIndex 实现。"""

    _kind = "case"
    _record_type = CasePrecedentRecord

    def _text_of(self, row: CasePrecedentRecord) -> str:
        return row.summary

    def _key_of(self, row: CasePrecedentRecord) -> str:
        return row.case_id

    def _meta_of(self, row: CasePrecedentRecord) -> dict[str, Any]:
        return case_node_metadata(row)

    def _filters_of(self, filters: CaseSearchFilters) -> Any | None:
        """业务过滤 → Chroma ``where``。"""
        llama_ = llama()
        clauses: list[Any] = []
        if filters.category:
            clauses.append(llama_.MetadataFilter(key="category", value=filters.category))
        if filters.risk_type:
            clauses.append(_risk_type_clause(filters.risk_type))
        return _where_from(clauses)

    async def search(self, query: str, filters: CaseSearchFilters, top_k: int) -> list[CaseHit]:
        candidates = _case_candidates(self._rows, filters)
        ranked = self._retrieve_ranked(
            query,
            candidates,
            self._filters_of(filters),
            top_k=top_k,
        )
        hits: list[CaseHit] = []
        for i, score in ranked:
            row = self._rows[i]
            hits.append(
                CaseHit.model_validate(
                    {
                        **row.model_dump(mode="json"),
                        "retrieval_score": score,
                    }
                )
            )
        return hits
