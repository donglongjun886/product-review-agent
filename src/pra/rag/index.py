"""Chroma 检索索引 —— ``ChromaPolicyIndex`` / ``ChromaCaseIndex``（向量 + BM25 + RRF）。

查询链路：业务过滤 → hybrid 检索（向量 / BM25 两路 + RRF 融合）→ 排序截断。

**两路的过滤机制不同，且必须逐条等价**：
- **向量路**：过滤**下推给 Chroma**（``_filters_of`` 把业务过滤编译成 ``MetadataFilters`` →
  ``where``），打分域在库侧收窄 = 库内全集 ∩ ``where``；取数走 LlamaIndex（``ChromaVectorStore``
  + ``VectorIndexRetriever``，分数直接采库口径 ``exp(-distance)``）。
- **BM25 路**：``bm25s`` 是内存索引、**没有 ``where``**，只能在 Python 候选集（``_*_candidates``）
  上建索引打分。

两条路径的等价性由 ``tests/test_rag_retrieval.py`` 的全组合用例锁住 —— 改任一侧都要同步另一侧。
一类 corpus 的全部逐类差异 = 子类声明的类级钩子（``_record_type`` / ``_text_of`` / ``_key_of`` /
``_meta_of`` / ``_filters_of``）+ ``_kind``，本模块不出现 ``if kind == ...`` 分派。
``effective_only`` / ``category``（含「全类目」）/ ``risk_type``（交叠非空）三个过滤语义与既有实现
逐条一致。链路零 LLM 调用、零随机；**最终排序** tie-break = (分降序, corpus 原序升序)
（融合前的名次域口径见 :meth:`_ChromaIndexBase._rank_hybrid`）。
hybrid 的融合在 :meth:`_ChromaIndexBase._rank_hybrid`（``QueryFusionRetriever`` + RRF）。
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
# 候选过滤（policy / case 各一份谓词；**只服务 BM25 路** —— 向量路的过滤已下推给 Chroma，
# 见 ``_filters_of``。两侧语义必须等价，由 tests/test_rag_retrieval.py 的组合用例锁住）
# ---------------------------------------------------------------------------


def _policy_candidates(
    rows: list[PolicyClauseRecord], filters: PolicySearchFilters, effective_only: bool
) -> list[int]:
    """候选行索引：``effective_only`` → ``status == "EFFECTIVE"``；``category`` ∈ {None, 值, 全类目}；
    ``risk_type`` 交叠非空。
    """
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
    """``risk_type`` 「交叠非空」→ ``$or`` of ``rt_<值>: 1``。

    值取整数 ``1``（不是 ``True``）：``MetadataFilter.value`` 的 pydantic 联合类型不收 ``bool``。
    键名必须与写入侧（``chroma_store.risk_type_key``）同源，否则**静默零命中**。
    """
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
# Policy / Case 两个公开索引类（构造/检索签名与既有实现对齐）
# ---------------------------------------------------------------------------


class _ChromaIndexBase:
    """两个 Chroma 索引的装配 / 检索骨架。

    子类在类体里声明 ``_kind`` / ``_record_type`` 与三个逐类钩子（``_text_of`` / ``_key_of`` /
    ``_meta_of``），故本模块无 ``if kind == ...`` 分派。构造顺序是「先 embed 全部文本、再建
    collection」：collection 名要带向量维度，而维度由**实际编码出的向量**决定（只有拿到向量后
    才知道），故不为省一轮 embed 把构造拆成两段。
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
        # 空语料显式失败：没有行就没有可编码的文本、也解析不出向量维度 → 不建库、不静默降级。
        if not self._rows:
            raise ValueError(f"{type(self).__name__} 语料为空：至少需要 1 行才能建库")
        cfg = config or ChromaConfig()
        # 客户端在构造期解析一次（缺 rag extra / 客户端装配错误即刻暴露），回填进 config →
        # 后续各层拿到的都是同一个实例，不再逐层重算 host/port/ephemeral。
        self._config = replace(cfg, client=make_chroma_client(cfg))
        # LlamaIndex ``BaseEmbedding``（官方集成承载编码）：查询/文本向量都走其公开方法。
        # **必填、无兜底** —— 构造编码器的唯一位置是 ``pra.tools.production_embedder``
        # （或调用方自己注入），不许在请求期联网下载模型。
        self._embed_model: Any = embedding_model
        doc_vectors = [
            self._embed_model.get_text_embedding(self._text_of(r)) for r in self._rows
        ]
        # 维度唯一来源 = 实际编码出的向量长度。
        self._dim = len(doc_vectors[0])
        self.collection_name = _collection_name(cfg.collection_prefix, self._kind, self._dim)
        self._seed(doc_vectors)

    # -- 逐类钩子（子类实现；基类不替任何一方兜底）------------------------------

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
        """业务过滤 → Chroma ``where`` 表达式（``MetadataFilters``；``None`` = 无过滤）。

        policy 侧额外收 ``effective_only``（case 无该语义），故其实现多一个参数。

        **必须与 ``_*_candidates`` 的 Python 谓词逐条等价** —— 前者是向量路的打分域，后者是
        BM25 路的打分域，两处不一致时「带过滤就会静默漏召回」（漏召回只在带过滤时暴露）。

        返回 ``None`` 而非空 ``MetadataFilters``：空表达式翻译出的 ``{}`` 语义上等于全量，
        但显式 ``None`` 让「无过滤」这条路径根本不经过 ``where``。
        """
        raise NotImplementedError

    # -- 装配 ---------------------------------------------------------------

    def _seed(self, doc_vectors: list[list[float]]) -> None:
        """建/复用 collection（``embedding_function=None`` **+ 显式 cosine 空间**）+ 先删后加重建。

        node 向量 = 文本向量（对同一文本编码，逐位一致）；节点按「1 行 = 1 node」写入，
        metadata 由 ``ChromaVectorStore.add`` 生成（扁平业务键 + ``_node_content``）。

        写入分两步：先按 node id 删除、再 ``add``。删除的 id = 残留 id（库内全量 id 与
        ``_node_ids`` 的差集）∪ ``_node_ids``；``add`` 对**已存在的 id 静默跳过**
        （既不覆盖也不抛错，chromadb 1.5.9 实测），不先删则同 id 的旧记录留在库里、新内容写不进去。
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
        # 读写共用同一个 store 实例。``embed_model`` **必须显式传** —— 不传会回落 ``Settings.embed_model`` →
        # ``resolve_embed_model("default")`` → 拉 ``llama_index.embeddings.openai``（本仓不装）。
        store = llama().ChromaVectorStore(chroma_collection=collection)
        # 残留 id = 库内全量 id（``get(include=[])["ids"]``）− 当前语料 id；为空时不删
        # （``delete(ids=[])`` 被 chromadb 拒绝）。集合一致时差集为空，不产生写操作。
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
        """检索结果 → ``[(行索引, 检索分)]``（按 ``(分降序, corpus 原序)`` 做**最终排序**）。

        ``ctx`` 之外的 node 直接丢弃 —— collection 里可能残留已不在语料中的旧 node id。
        融合前的名次域不由本方法决定，见 :meth:`_rank_hybrid`。
        """
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
        """向量路排名（分数 = 库口径 ``exp(-distance)``）；过滤在库侧按 ``filters`` 收窄。

        故传**全量** ctx（``_full_context``）而非候选子集。
        """
        return self._to_ranked(
            self._make_vector_retriever(ctx, filters).retrieve(query_bundle), ctx
        )

    def _rank_bm25(
        self, sub_ctx: _RetrievalContext, query_bundle: Any, top_k: int
    ) -> list[tuple[int, float]]:
        """BM25 路排名（Python 侧过滤 = 只喂候选 node）。

        分数原样透传 ``bm25s``（**无界、不做量纲适配**）—— RRF 只看名次，融合与截断都不依赖该分。
        """
        retriever = make_bm25_retriever(sub_ctx, top_k)
        return self._to_ranked(retriever.retrieve(query_bundle), sub_ctx)

    def _rank_hybrid(
        self, query: str, candidates: list[int], filters: Any | None
    ) -> list[tuple[int, float]]:
        """hybrid 排名：两路检索器交给 ``QueryFusionRetriever`` 做 RRF 融合。

        两路**打分域不同**：向量路用全量 ctx + ``where``（过滤已下推给 Chroma），BM25 路用
        Python 候选子集（``bm25s`` 内存索引没有 ``where``）。

        ⚠️ 两处偏离 ``QueryFusionRetriever`` 缺省行为：``num_queries=1`` + ``MockLLM``
        （缺省 4 会让它调 LLM 生成扩展查询，本仓不做多查询扩展）；``use_async=False``
        （缺省 True 会在调用方已有事件循环时另起线程跑检索）。``similarity_top_k`` 传库内全集
        行数 —— 截断由调用方在融合之后统一做。

        ⚠️ 名次域口径：RRF 的名次来自**各路自身的返回序**（``QueryFusionRetriever`` 对每路按分
        稳定排序），BM25 腿的并列序即 ``bm25s`` 的返回序，**不是** corpus 原序；``_to_ranked`` 的
        ``(分降序, corpus 原序)`` 只约束融合后的最终排序。
        """
        full_ctx = self._full_context()
        sub_ctx = self._sub_context(candidates)
        fusion = llama().QueryFusionRetriever(
            retrievers=[
                self._make_vector_retriever(full_ctx, filters),
                make_bm25_retriever(sub_ctx, len(candidates)),
            ],
            # ⚠️ ``llm`` 必填：不传会走 ``Settings.llm`` → 拉 ``llama-index-llms-openai``（本仓不装）
            # → ``ImportError``；``num_queries=1`` 时它一次都不会被调用。
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
        """hybrid 检索 → ``[(行索引, 检索分)]``（已按 ``(分降序, 原序)`` 排序、已截断 Top-K）。

        两路的**打分域不同**：向量路用全量 ctx + ``where``（过滤已下推给 Chroma），BM25 路用
        Python 候选子集（``candidates``）—— ``bm25s`` 内存索引没有 ``where``。

        ``top_k`` 截断在**排序之后**统一做；各路内部取「所在打分域的全长」，先拿到完整排名
        （RRF 的排名列表必须覆盖完整域，不能按 ``top_k`` 预截断）。
        """
        if not candidates:
            return []
        return self._rank_hybrid(query, candidates, filters)[:top_k]


class ChromaPolicyIndex(_ChromaIndexBase):
    """``PolicyIndex`` Protocol 的 Chroma + LlamaIndex 实现。

    构造参数为 Chroma 装配参数（``config=ChromaConfig(...)``：client 注入 / host+port 自建 /
    ephemeral 离线内存库 / collection_prefix）。检索签名与工具契约逐字一致：
    ``async def search(query, filters, top_k, effective_only) -> list[PolicyClauseHit]``。
    过滤语义（effective_only / category / risk_type）与既有实现逐条一致，差异只在
    「向量库 + 检索器 + 融合」。
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
        """业务过滤 → Chroma ``where``（**与 :func:`_policy_candidates` 逐条等价**）。

        - ``effective_only`` → ``status == "EFFECTIVE"``；
        - ``category`` 的三态 ``{None, 值, 全类目}``：``category`` 是 schema **必填非空 str**，
          故 ``None`` 分支不可达，表达式只需 ``$in [值, 全类目]``；
        - ``risk_type`` 交叠非空 → ``$or`` of ``rt_*`` 键。
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
        """检索政策条款（hybrid；无命中 → ``[]``，工具 ok=True）。

        流程：业务过滤（向量路下推 Chroma ``where``；BM25 路走 Python 候选集）→ 打分/融合 →
        ``(分降序, corpus 原序)`` 排序 → Top-K。
        """
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
    """``CaseIndex`` Protocol 的 Chroma + LlamaIndex 实现（``search`` 签名与工具契约一致）。

    ``CaseHit.retrieval_score`` 是**检索分，不是语义相似度**，且**不做量纲适配**：生产
    ``search`` 返回 hybrid 的 RRF 融合分（``Σ 1/(k+rank)``，``k=60``，落在 ~(0, ``2/60 = 1/30``]）。
    两条内部路径各自的原始分只由 ``_rank_bm25``（``bm25s`` 原始分，无界）/ ``_rank_vector``
    （库口径 ``exp(-distance)``，⊂ ``(0, 1]``）产出，仅供调试/验证，不构成对外模式开关。
    **取值域由后端定义，字段不设上下界约束**；这些分都**不参与 Gate 判定**（Gate 对
    ``CASE_PRECEDENT`` / ``POLICY_REF`` 只判存在性；见
    :func:`pra.agent.guardrails.measurements.positive_dimensions` 的类型白名单）。
    构造签名见 :meth:`_ChromaIndexBase.__init__`。
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
        """业务过滤 → Chroma ``where``（**与 :func:`_case_candidates` 逐条等价**）。

        case **没有** ``effective_only`` 语义；``category`` 是**精确相等**（不是 policy 的
        「占位 / 值 / 全类目」三态）。
        """
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
