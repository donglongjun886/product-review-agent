"""Chroma 检索索引：``ChromaPolicyIndex`` / ``ChromaCaseIndex``（向量 + BM25 + RRF hybrid）。

向量路过滤下推 Chroma ``where``，BM25 路在 Python 候选集上打分（``bm25s`` 没有 ``where``）；
两侧过滤语义必须逐条等价，不一致时带过滤会静默漏召回。
"""

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
from pra.rag.retrieval import _RetrievalContext
from pra.tools.case_search.tool import CaseHit, CaseSearchFilters
from pra.tools.policy_search.tool import PolicyClauseHit, PolicySearchFilters

__all__ = ["ChromaCaseIndex", "ChromaPolicyIndex"]

_FULL_CATEGORY = "全类目"


# ---------------------------------------------------------------------------
# 候选过滤（只服务 BM25 路；向量路过滤已下推给 Chroma）
# ---------------------------------------------------------------------------


def _policy_candidates(
    rows: list[PolicyClauseRecord], filters: PolicySearchFilters, effective_only: bool
) -> list[int]:
    """候选行索引：``effective_only`` / ``category``（含「全类目」）/ ``risk_type``（交叠非空）。"""
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
    """``risk_type``「交叠非空」→ ``$or`` of ``rt_<值>: 1``；键名须与写入侧 ``chroma_store.risk_type_key`` 同源，值取整数 ``1``（不收 ``bool``）。"""
    llama_ = llama()
    return llama_.MetadataFilters(
        condition=llama_.FilterCondition.OR,
        filters=[llama_.MetadataFilter(key=risk_type_key(t), value=1) for t in risk_types],
    )


def _where_from(clauses: list[Any]) -> Any | None:
    """过滤子句列表 → ``MetadataFilters``（多条套 ``$and``；空 → ``None``：chromadb 拒绝 ``where={}``）。"""
    if not clauses:
        return None
    llama_ = llama()
    return llama_.MetadataFilters(condition=llama_.FilterCondition.AND, filters=clauses)


# ---------------------------------------------------------------------------
# Policy / Case 两个公开索引类
# ---------------------------------------------------------------------------


class _ChromaIndexBase:
    """两个 Chroma 索引的装配 / 检索骨架；子类声明 ``_kind`` / ``_record_type`` 与逐类钩子。

    构造先 embed 全部文本再建 collection —— collection 名要带向量维度，维度由实际编码出的向量决定。
    """

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
        # 空语料显式失败：没有行就没有可编码文本，也解析不出向量维度。
        if not self._rows:
            raise ValueError(f"{type(self).__name__} 语料为空：至少需要 1 行才能建库")
        cfg = config or ChromaConfig()
        # 客户端在构造期解析一次（缺 rag extra / 装配错误即刻暴露）并回填进 config，后续各层复用同一实例。
        self._config = replace(cfg, client=make_chroma_client(cfg))
        # LlamaIndex ``BaseEmbedding``：查询/文本向量都走其公开方法。**必填、无兜底**（本类不构造编码器）。
        self._embed_model: Any = embedding_model
        doc_vectors = [
            self._embed_model.get_text_embedding(self._text_of(r)) for r in self._rows
        ]
        # 维度唯一来源 = 实际编码出的向量长度。
        self._dim = len(doc_vectors[0])
        self.collection_name = _collection_name(cfg.collection_prefix, self._kind, self._dim)
        self._seed(doc_vectors)

    # -- 逐类钩子（子类实现）--------------------------------------------------

    def _text_of(self, row: Any) -> str:
        """检索文本（= 入库 embed 的同一份文本）。"""
        raise NotImplementedError

    def _key_of(self, row: Any) -> str:
        """corpus 行键（node id 的哈希输入，逐行唯一）。"""
        raise NotImplementedError

    def _meta_of(self, row: Any) -> dict[str, Any]:
        """node metadata（不进检索文本，见 ``chroma_store._build_nodes``）。"""
        raise NotImplementedError

    def _filters_of(self, filters: Any) -> Any | None:
        """业务过滤 → Chroma ``where`` 表达式；policy 侧额外收 ``effective_only``，``None`` = 无过滤。

        必须与 ``_*_candidates`` 的 Python 谓词逐条等价，否则带过滤会静默漏召回；返回 ``None``
        而非空 ``MetadataFilters``（空表达式翻译出的 ``{}`` 等于全量）。
        """
        raise NotImplementedError

    # -- 装配 ---------------------------------------------------------------

    def _seed(self, doc_vectors: list[list[float]]) -> None:
        """建/复用 collection（``embedding_function=None`` + cosine 空间）+ 按 node id 先删后加重建。

        ``add`` 对**已存在的 id 静默跳过**（既不覆盖也不抛错），不先删则同 id 的旧记录留在库里、新内容写不进去。
        """
        collection = _open_collection(self._config, name=self.collection_name)
        self._nodes, self._node_ids = _build_nodes(
            self._rows,
            collection=self.collection_name,
            key_of=self._key_of,
            text_of=self._text_of,
            meta_of=self._meta_of,
        )
        # ``add`` 取 ``node.get_embedding()``，故把文本向量回填到 node。
        for node, vec in zip(self._nodes, doc_vectors):
            node.embedding = list(vec)
        # 读写共用同一 store；``embed_model`` **必须显式传**（不传会回落 ``Settings.embed_model`` → 拉本仓未装的 openai 集成）。
        store = llama().ChromaVectorStore(chroma_collection=collection)
        # 残留 id = 库内全量 id − 当前语料 id；为空时不删（``delete(ids=[])`` 被 chromadb 拒绝）。
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
        """候选子集上的检索上下文（node / node id / 行索引映射都只含候选）。"""
        return _RetrievalContext(
            node_ids=[self._node_ids[i] for i in candidates],
            nodes=[self._nodes[i] for i in candidates],
            row_index=list(candidates),
        )

    def _full_context(self) -> _RetrievalContext:
        """全量语料上的检索上下文（向量路用；``_node_ids[i]`` 对应 ``self._rows[i]``）。"""
        return _RetrievalContext(
            node_ids=list(self._node_ids),
            nodes=list(self._nodes),
            row_index=list(range(len(self._node_ids))),
        )

    def _to_ranked(self, nodes: list[Any], ctx: _RetrievalContext) -> list[tuple[int, float]]:
        """检索结果 → ``[(行索引, 检索分)]``，按 ``(分降序, corpus 原序)`` 排序；``ctx`` 之外的 node 丢弃。"""
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
        """向量路检索器（过滤在库侧按 ``filters`` 收窄，故 ``similarity_top_k`` 取 ``ctx`` 全长）。"""
        return self._vec_index.as_retriever(
            similarity_top_k=max(1, len(ctx.node_ids)), filters=filters
        )

    def _rank_vector(
        self, ctx: _RetrievalContext, query_bundle: Any, filters: Any | None
    ) -> list[tuple[int, float]]:
        """向量路排名（分数 = 库口径 ``exp(-distance)``；过滤在库侧按 ``filters`` 收窄，故传全量 ctx）。"""
        return self._to_ranked(
            self._make_vector_retriever(ctx, filters).retrieve(query_bundle), ctx
        )

    def _rank_bm25(
        self, sub_ctx: _RetrievalContext, query_bundle: Any, top_k: int
    ) -> list[tuple[int, float]]:
        """BM25 路排名（Python 侧过滤 = 只喂候选 node）；分数原样透传 ``bm25s``，不做量纲适配。"""
        retriever = make_bm25_retriever(sub_ctx, top_k)
        return self._to_ranked(retriever.retrieve(query_bundle), sub_ctx)

    def _rank_hybrid(
        self, query: str, candidates: list[int], filters: Any | None
    ) -> list[tuple[int, float]]:
        """hybrid 排名：两路检索器交给 ``QueryFusionRetriever`` 做 RRF 融合。

        向量路用全量 ctx + ``where``，BM25 路用 Python 候选子集；``num_queries=1`` + ``MockLLM``
        避开缺省的 LLM 扩展查询（本仓不做多查询扩展），``use_async=False`` 避开另起线程。
        """
        full_ctx = self._full_context()
        sub_ctx = self._sub_context(candidates)
        fusion = llama().QueryFusionRetriever(
            retrievers=[
                self._make_vector_retriever(full_ctx, filters),
                make_bm25_retriever(sub_ctx, len(candidates)),
            ],
            # ⚠️ ``llm`` 必填：不传会走 ``Settings.llm`` → 拉本仓未装的 openai 集成（``num_queries=1`` 时不会被调用）。
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
        """hybrid 检索 → ``[(行索引, 检索分)]``（排序后截断 Top-K；各路取打分域全长以喂满 RRF 排名列表）。"""
        if not candidates:
            return []
        return self._rank_hybrid(query, candidates, filters)[:top_k]


class ChromaPolicyIndex(_ChromaIndexBase):
    """``PolicyIndex`` Protocol 的 Chroma + LlamaIndex 实现。

    构造参数为 Chroma 装配参数（``config=ChromaConfig(...)``）；``search`` 签名与工具契约一致。
    """

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
        """业务过滤 → Chroma ``where``（必须与 ``_policy_candidates`` 逐条等价）。

        ``category`` 是必填非空 str，故只需 ``$in [值, 全类目]``；``risk_type`` 交叠非空 → ``$or`` of ``rt_*`` 键。
        """
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
        """检索政策条款（hybrid）；无命中 → ``[]``，工具 ok=True。"""
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
    """``CaseIndex`` Protocol 的 Chroma + LlamaIndex 实现。

    ``CaseHit.retrieval_score`` 是检索分、不是语义相似度，也不做量纲适配（hybrid 返回 RRF 融合分）。
    """

    _kind = "case"
    _record_type = CasePrecedentRecord

    def _text_of(self, row: CasePrecedentRecord) -> str:
        return row.summary

    def _key_of(self, row: CasePrecedentRecord) -> str:
        return row.case_id

    def _meta_of(self, row: CasePrecedentRecord) -> dict[str, Any]:
        return case_node_metadata(row)

    def _filters_of(self, filters: CaseSearchFilters) -> Any | None:
        """业务过滤 → Chroma ``where``（必须与 ``_case_candidates`` 逐条等价）；case 无 ``effective_only``，``category`` 精确相等。"""
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
                        "retrieval_score": score,  # 检索分（hybrid=RRF 分），**非语义相似度**
                    }
                )
            )
        return hits
