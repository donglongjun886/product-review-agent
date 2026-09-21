"""Chroma 检索索引 —— ``ChromaPolicyIndex`` / ``ChromaCaseIndex``（向量 + BM25 + RRF）。

查询链路：业务过滤 → 三模式检索（向量 / BM25 / hybrid）→ 排序截断。

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
逐条一致。链路零 LLM、零随机；tie-break = (分降序, corpus 原序升序)。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from pra.rag.bm25 import bm25_retrieve, make_bm25_retriever
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
from pra.rag.retrieval import (
    MODES,
    RetrievalMode,
    _RetrievalContext,
    fuse_rrf,
    normalize_minmax,
)
from pra.rag.vector import vector_retrieve
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
    """过滤子句列表 → ``MetadataFilters``（空 → ``None``；单条不套 ``$and``）。"""
    if not clauses:
        return None
    llama_ = llama()
    if len(clauses) == 1:
        return llama_.MetadataFilters(filters=clauses)
    return llama_.MetadataFilters(condition=llama_.FilterCondition.AND, filters=clauses)


def _validate_mode(mode: str) -> RetrievalMode:
    if mode not in MODES:
        raise ValueError(f"未知检索模式: {mode!r}（可选: {list(MODES)}）")
    return mode  # type: ignore[return-value]


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
        rows: Iterable[dict | Any],
        *,
        embedding_model: Any,
        mode: RetrievalMode = "hybrid",
        config: ChromaConfig | None = None,
    ) -> None:
        self._rows: list[Any] = _normalize_rows(rows, self._record_type)
        self.mode: RetrievalMode = _validate_mode(mode)
        cfg = config or ChromaConfig()
        self.collection_prefix = cfg.collection_prefix
        # 空语料不建库（``collection_name`` 保持 ""）：该可见属性必须先有默认值，否则空 KB 上
        # 读它会抛 AttributeError。``_dim`` 留 0：空 KB 不解析维度（不建库、无向量可算）。
        self._vec_index: Any | None = None
        self._nodes: list[Any] = []
        self._node_ids: list[str] = []
        self._dim = 0
        self.collection_name = ""
        self._doc_vectors: list[list[float]] = []
        # 客户端在构造期解析一次（缺 rag extra / 客户端装配错误即刻暴露），回填进 config →
        # 后续各层拿到的都是同一个实例，不再逐层重算 host/port/ephemeral。
        self._config = replace(cfg, client=make_chroma_client(cfg))
        # LlamaIndex 装配面同样在构造期解析（空 KB 也不例外 —— 缺 rag extra 不推迟到检索期）。
        llama()
        # LlamaIndex ``BaseEmbedding``（官方集成承载编码）：查询/文本向量都走其公开方法。
        # **必填、无兜底** —— 构造编码器的唯一位置是 ``pra.tools.production_embedder``
        # （或调用方自己注入），不许在请求期联网下载模型。
        self._embed_model: Any = embedding_model
        if self._rows:
            # 空 KB 走不到这里（不建库，故不 embed / 不留 collection_name）。
            self._doc_vectors = [
                self._embed_model.get_text_embedding(self._text_of(r)) for r in self._rows
            ]
            # 维度唯一来源 = 实际编码出的向量长度。
            self._dim = len(self._doc_vectors[0])
            self.collection_name = _collection_name(cfg.collection_prefix, self._kind, self._dim)
            self._seed()

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

    def _filters_of(self, filters: Any, effective_only: bool = False) -> Any | None:
        """业务过滤 → Chroma ``where`` 表达式（``MetadataFilters``；``None`` = 无过滤）。

        **必须与 ``_*_candidates`` 的 Python 谓词逐条等价** —— 前者是向量路的打分域，后者是
        BM25 路的打分域，两处不一致时「带过滤就会静默漏召回」（漏召回只在带过滤时暴露）。

        返回 ``None`` 而非空 ``MetadataFilters``：空表达式翻译出的 ``{}`` 语义上等于全量，
        但显式 ``None`` 让「无过滤」这条路径根本不经过 ``where``。
        """
        raise NotImplementedError

    # -- 装配 ---------------------------------------------------------------

    def _seed(self) -> None:
        """建/复用 collection（``embedding_function=None`` **+ 显式 cosine 空间**）+ 幂等 upsert。

        node 向量 = 文本向量（对同一文本编码，逐位一致）；节点按
        「1 行 = 1 node」写入，metadata 见 ``*_node_metadata``。

        末尾顺带装配**向量路取数面**：同一个 collection 包进 ``ChromaVectorStore`` —— 建库参数
        仍归我们（``_open_collection``），包装层只用于取数。**只 upsert 不写 ``_node_content``**：
        取数时 ``metadata_dict_to_node`` 会因缺键抛错，被 ``_query`` 的 ``except`` 吸收后回落到
        legacy 分支，重建出的 node 仍带稳定 ``id_``（= Chroma id）与 ``documents`` 正文，够用。
        """
        collection = _open_collection(
            self._config,
            name=self.collection_name,
            dim=self._dim,
            kind=self._kind,
        )
        self._nodes, self._node_ids = _build_nodes(
            self._rows,
            collection=self.collection_name,
            key_of=self._key_of,
            text_of=self._text_of,
            meta_of=self._meta_of,
        )
        collection.upsert(
            ids=list(self._node_ids),
            embeddings=[list(v) for v in self._doc_vectors],
            metadatas=[node.metadata for node in self._nodes],
            documents=[node.get_content() for node in self._nodes],
        )
        # 向量路取数装配面。``embed_model`` **必须显式传** —— 不传会回落 ``Settings.embed_model`` →
        # ``resolve_embed_model("default")`` → 拉 ``llama_index.embeddings.openai``（本仓不装）。
        self._vec_index = llama().VectorStoreIndex.from_vector_store(
            llama().ChromaVectorStore(chroma_collection=collection),
            embed_model=self._embed_model,
        )

    # -- size / 统计 ---------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._rows)

    @property
    def node_ids(self) -> list[str]:
        return list(self._node_ids)

    @property
    def nodes(self) -> list[Any]:
        return list(self._nodes)

    # -- 装配子件 -----------------------------------------------------------

    def _sub_context(self, candidates: list[int]) -> _RetrievalContext:
        """候选子集上的检索上下文（node / node id / 行索引映射都只含候选）。"""
        sub_ids = [self._node_ids[i] for i in candidates]
        return _RetrievalContext(
            node_ids=sub_ids,
            nodes=[self._nodes[i] for i in candidates],
            row_index_by_key={nid: candidates[offset] for offset, nid in enumerate(sub_ids)},
        )

    def _full_context(self) -> _RetrievalContext:
        """全量语料上的检索上下文（向量路用；``_node_ids[i]`` 对应 ``self._rows[i]``）。"""
        return _RetrievalContext(
            node_ids=list(self._node_ids),
            nodes=list(self._nodes),
            row_index_by_key={nid: i for i, nid in enumerate(self._node_ids)},
        )

    def _rank_vector(
        self, ctx: _RetrievalContext, query_bundle: Any, filters: Any | None
    ) -> list[tuple[int, float]]:
        """向量路排名：``[(行索引, 分数)]``，按分数降序、corpus 原序 tie-break。

        过滤在库侧按 ``filters`` 收窄，故传**全量** ctx（``_full_context``）而非候选子集。
        """
        pairs = vector_retrieve(
            self._vec_index,
            query_bundle,
            top_k=len(ctx.node_ids),
            filters=filters,
        )
        scored: dict[str, float] = {
            nid: score for nid, score in pairs if nid in ctx.row_index_by_key
        }
        ranked = [(ctx.row_index_by_key[nid], score) for nid, score in scored.items()]
        ranked.sort(key=lambda t: (-round(t[1], 6), t[0]))
        return [(i, round(s, 6)) for i, s in ranked]

    def _rank_bm25(
        self, sub_ctx: _RetrievalContext, query_bundle: Any, top_k: int
    ) -> list[tuple[int, float]]:
        """BM25 路排名：候选集内 min-max 归一化 BM25 分（Python 侧过滤 = 只喂候选 node）。

        归一化用 ``retrieval.normalize_minmax``：候选集内最高分 = 1.0、最低 = 0.0（等值集全 1.0）。
        归一化的目的是让 BM25 的**无界**原始分落到与前两路同量级、可比较 —— **不是**为了满足
        ``CaseHit.retrieval_score`` 的约束（该字段已不设取值域，见 :class:`ChromaCaseIndex`）。
        """
        retriever = make_bm25_retriever(sub_ctx, top_k)
        nodes = bm25_retrieve(retriever, query_bundle)
        pairs = [
            (sub_ctx.row_index_by_key[n.node.node_id], float(n.score or 0.0))
            for n in nodes
            if n.node.node_id in sub_ctx.row_index_by_key
        ]
        norm = normalize_minmax([s for _i, s in pairs])
        ranked = [(i, s) for (i, _raw), s in zip(pairs, norm)]
        ranked.sort(key=lambda t: (-round(t[1], 6), t[0]))
        return [(i, round(s, 6)) for i, s in ranked]

    def _rank_fused(
        self, routed: dict[str, list[tuple[int, float]]], row_index: dict[str, int]
    ) -> list[tuple[int, float]]:
        """多路排名 → **RRF 融合**（``Σ_r 1/(60 + rank_r)``，``k=60``）。

        各路返回**各自打分域内的完整排名**（不按 ``top_k`` 预截断）→ 在完整排名列表上融合
        （:func:`~pra.rag.retrieval.fuse_rrf`，含「为什么不用 ``QueryFusionRetriever`` 现成融合」的实测理由）。
        ⚠️ 该分是**融合排名分，不是相似度**（上界 **2/60 = 1/30 ≈ 0.0333**）；排序 key
        = ``(分降序, corpus 原序 idx 升序)``。
        """
        # RRF 的输入是「各路排名 id 列表 + node_id → 行索引映射」。两路打分域不同
        # （向量 = 库内全集 ∩ where，BM25 = Python 候选集），故映射必须用**全量**行索引。
        id_by_row = {row: nid for nid, row in row_index.items()}
        return fuse_rrf(
            {name: [id_by_row[row] for row, _score in ranked] for name, ranked in routed.items()},
            row_index,
        )

    # -- 检索核心 -----------------------------------------------------------

    def _retrieve_ranked(
        self,
        query: str,
        candidates: list[int],
        filters: Any | None,
        *,
        top_k: int,
    ) -> list[tuple[int, float]]:
        """三模式检索 → ``[(行索引, 6 位检索分)]``（已按 ``(分降序, 原序)`` 排序、已截断 Top-K）。

        两路的**打分域不同**：向量路用全量 ctx + ``where``（过滤已下推给 Chroma），BM25 路用
        Python 候选子集（``candidates``）—— ``bm25s`` 内存索引没有 ``where``。

        ``top_k`` 截断在**排序之后**统一做；各路内部取「所在打分域的全长」，先拿到完整排名
        （BM25 的 min-max 归一化、RRF 的排名列表都必须覆盖完整域，不能按 ``top_k`` 预截断）。
        """
        if top_k < 1 or not candidates:
            return []
        query_bundle = llama().QueryBundle(query_str=query)
        if self.mode == "bm25":
            ranked = self._rank_bm25(self._sub_context(candidates), query_bundle, len(candidates))
        elif self.mode == "vector":
            ranked = self._rank_vector(self._full_context(), query_bundle, filters)
        else:
            full_ctx = self._full_context()
            ranked = self._rank_fused(
                {
                    "vector": self._rank_vector(full_ctx, query_bundle, filters),
                    "bm25": self._rank_bm25(
                        self._sub_context(candidates), query_bundle, len(candidates)
                    ),
                },
                full_ctx.row_index_by_key,
            )
        return ranked[:top_k]


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
        """检索政策条款（三模式；无命中 → ``[]``，工具 ok=True）。

        流程：业务过滤（向量路下推 Chroma ``where``；BM25 路走 Python 候选集）→ 打分/融合 →
        ``(分降序, corpus 原序)`` 排序 + 6 位取整 → Top-K。
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

    ``CaseHit.retrieval_score`` 是**检索分，不是语义相似度**：``bm25`` = 候选集内 min-max
    归一化 BM25 分；``vector`` = 库口径 ``exp(-distance)``；``hybrid`` = RRF 融合分（``Σ 1/(k+rank)``，
    ``k=60``，落在 ~(0, ``2/60 = 1/30``]）。**取值域由后端定义，字段不设上下界约束**。
    **三种分数量纲互不可比，且都不参与 Gate 判定**（Gate 对 ``CASE_PRECEDENT`` / ``POLICY_REF``
    只判存在性；见 :func:`pra.agent.guardrails.measurements.positive_dimensions` 的类型白名单）。
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

    def _filters_of(
        self, filters: CaseSearchFilters, effective_only: bool = False
    ) -> Any | None:
        """业务过滤 → Chroma ``where``（**与 :func:`_case_candidates` 逐条等价**）。

        case **没有** ``effective_only`` 语义（``effective_only`` 形参只为与基类钩子同签名，
        恒被忽略）；``category`` 是**精确相等**（不是 policy 的「占位 / 值 / 全类目」三态）。
        """
        del effective_only
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
