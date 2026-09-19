"""Chroma 检索索引 —— ``ChromaPolicyIndex`` / ``ChromaCaseIndex``（向量 + BM25 + RRF）。

查询链路：Python 侧候选过滤 → 向量路（Chroma cosine，``1 − distance``）+ BM25 路（bm25s + jieba）
→ RRF 融合 → Top-K。一类 corpus 的全部逐类差异 = 子类声明的类级钩子（``_record_type`` /
``_text_of`` / ``_key_of`` / ``_meta_of``）+ ``_kind``，本模块不出现 ``if kind == ...`` 分派。
向量路把 ``category`` 下推到 store（只是候选谓词的超集），最终判定与 BM25 路统一走同一份 Python
谓词 ``recheck``。``effective_only`` / ``category``（含「全类目」）/ ``risk_type``（交叠非空）三个
过滤语义与既有实现逐条一致。链路零 LLM、零随机；tie-break = (分降序, corpus 原序升序)。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
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
from pra.rag.vector import make_vector_retriever
from pra.tools.case_search.tool import CaseHit, CaseSearchFilters
from pra.tools.policy_search.tool import PolicyClauseHit, PolicySearchFilters

__all__ = ["ChromaCaseIndex", "ChromaPolicyIndex"]

_FULL_CATEGORY = "全类目"


# ---------------------------------------------------------------------------
# 候选过滤（policy / case 各一份谓词；向量路与 BM25 路共用，绝无双实现）
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


def _policy_store_filters(filters: PolicySearchFilters) -> dict[str, Any] | None:
    """Policy 下推过滤（向量路，Chroma 原生 ``where``）：仅 ``category``（含「全类目」）可下推 ——
    Chroma 的 ``$in``/``$eq`` 对列表字段恒不命中，故 ``risk_type`` 只能留在 Python 侧判。
    """
    if not filters.category:
        return None
    return {"category": {"$in": [filters.category, _FULL_CATEGORY]}}


def _case_store_filters(filters: CaseSearchFilters) -> dict[str, Any] | None:
    if not filters.category:
        return None
    return {"category": filters.category}


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
        self._collection: Any | None = None
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
        # **必填、无兜底** —— 这里曾有 `or build_embedding_model("fastembed")` 兜底，但它不传
        # `cache_dir` / `local_files_only`，等于偷偷允许请求期联网下载模型。构造编码器的唯一
        # 位置是 ``pra.tools.production_embedder``（或调用方自己注入）。
        self._embed_model: Any = embedding_model
        if self._rows:
            # 空 KB 走不到这里（不建库，故不 embed / 不留 collection_name）。
            self._doc_vectors = [
                self._embed_model.get_text_embedding(self._text_of(r)) for r in self._rows
            ]
            # 维度唯一来源 = 实际编码出的向量长度（不再有可注入的 dim 参数，也不再探测模型声明）。
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

    # -- 装配 ---------------------------------------------------------------

    def _seed(self) -> None:
        """建/复用 collection（``embedding_function=None`` **+ 显式 cosine 空间**）+ 幂等 upsert。

        node 向量 = 文本向量（对同一文本编码，逐位一致）；节点按
        「1 行 = 1 node」写入，metadata 见 ``*_node_metadata``。
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
        self._collection = collection

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
            embed_model=self._embed_model,
            row_index_by_key={nid: candidates[offset] for offset, nid in enumerate(sub_ids)},
        )

    def _rank_vector(
        self, sub_ctx: _RetrievalContext, query_bundle: Any, store_filters: dict[str, Any] | None
    ) -> list[tuple[int, float]]:
        """向量路排名：``[(行索引, 1 − distance)]``（精确候选 id 集 + store 侧 category 下推）。

        打分域 = 精确候选集（不让非候选行抢名额）：候选 id 交给 Chroma ``ids=`` 取回。
        """
        retriever = make_vector_retriever(sub_ctx, self._collection, store_filters)
        nodes = retriever.retrieve(query_bundle)
        scored: dict[str, float] = {
            n.node.node_id: n.score for n in nodes if n.node.node_id in sub_ctx.row_index_by_key
        }
        ranked = [(sub_ctx.row_index_by_key[nid], score) for nid, score in scored.items()]
        ranked.sort(key=lambda t: (-round(t[1], 6), t[0]))
        return [(i, round(s, 6)) for i, s in ranked]

    def _rank_bm25(
        self, sub_ctx: _RetrievalContext, query_bundle: Any, top_k: int
    ) -> list[tuple[int, float]]:
        """BM25 路排名：候选集内 min-max 归一化 BM25 分（Python 侧过滤 = 只喂候选 node）。

        归一化用 ``retrieval.normalize_minmax``：
        候选集内最高分 = 1.0，分数恒 ⊂ [0,1]（``CaseHit.retrieval_score`` 的 ``le=1`` 约束成立）。
        """
        retriever = make_bm25_retriever(sub_ctx, top_k)
        nodes = bm25_retrieve(retriever, query_bundle)
        pairs = [
            (sub_ctx.row_index_by_key[n.node.node_id], float(n.score or 0.0))
            for n in nodes
            if n.node.node_id in sub_ctx.row_index_by_key
        ]
        pairs.sort(key=lambda t: (-t[1], t[0]))
        norm = normalize_minmax([s for _i, s in pairs])
        ranked = [(i, s) for (i, _raw), s in zip(pairs, norm)]
        ranked.sort(key=lambda t: (-round(t[1], 6), t[0]))
        return [(i, round(s, 6)) for i, s in ranked]

    def _rank_hybrid(
        self,
        sub_ctx: _RetrievalContext,
        query_bundle: Any,
        top_k: int,
        store_filters: dict[str, Any] | None,
    ) -> list[tuple[int, float]]:
        """Hybrid 路：两路排名 → **RRF 融合**（``Σ_r 1/(60 + rank_r)``，``k=60``）。

        两路各自返回**全部候选**（``top_k`` = 候选数）→ 在完整排名列表上融合
        （:func:`~pra.rag.retrieval.fuse_rrf`，含「为什么不用 ``QueryFusionRetriever`` 现成融合」的实测理由）。
        ⚠️ 该分是**融合排名分，不是相似度**（上界 **2/60 = 1/30 ≈ 0.0333**）；排序 key
        = ``(分降序, corpus 原序 idx 升序)``。
        """
        vec_ranked = self._rank_vector(sub_ctx, query_bundle, store_filters)
        bm25_ranked = self._rank_bm25(sub_ctx, query_bundle, top_k)
        # RRF 的输入是「两路各自的排名列表」：把 (行索引, 分) 排名还原为 node id 顺序。
        id_by_row = {row: nid for nid, row in sub_ctx.row_index_by_key.items()}
        ranked_lists = {
            "vector": [id_by_row[row] for row, _score in vec_ranked],
            "bm25": [id_by_row[row] for row, _score in bm25_ranked],
        }
        return fuse_rrf(ranked_lists, sub_ctx.row_index_by_key)

    # -- 检索核心 -----------------------------------------------------------

    def _retrieve_ranked(
        self,
        query: str,
        candidates: list[int],
        *,
        top_k: int,
        store_filters: dict[str, Any] | None,
        recheck: Callable[[int], bool],
    ) -> list[tuple[int, float]]:
        """三模式检索 → ``[(行索引, 6 位检索分)]``（已按 ``(分降序, 原序)`` 排序、已截断 Top-K）。

        向量路在 store 下推之后用 ``recheck``（**与 BM25 路同一份** Python 谓词）复核一遍：
        下推只用于收窄候选，判定只有一份实现，杜绝双实现语义漂移。

        ``top_k`` 截断在**排序之后**统一做；各路内部按需取「候选数」以保证 BM25 的
        min-max 与 RRF 的排名列表覆盖完整候选集。
        """
        if top_k < 1 or not candidates:
            return []
        sub_ctx = self._sub_context(candidates)
        query_bundle = llama().QueryBundle(query_str=query)
        # 三路都取「候选数」上限：先拿到**完整候选排名**，再复核过滤、最后截断 Top-K。
        # （若这里就按 top_k 预截断，store 返回的前 top_k 里一旦有被 recheck 剔除的行，
        #   结果会不足 top_k —— 实测 policy vector + effective_only 就是这样少一条。）
        full_k = len(candidates)
        if self.mode == "bm25":
            ranked = self._rank_bm25(sub_ctx, query_bundle, full_k)
        elif self.mode == "vector":
            ranked = self._rank_vector(sub_ctx, query_bundle, store_filters)
        else:
            ranked = self._rank_hybrid(sub_ctx, query_bundle, full_k, store_filters)
        ranked = [item for item in ranked if recheck(item[0])]
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

    async def search(
        self,
        query: str,
        filters: PolicySearchFilters,
        top_k: int,
        effective_only: bool,
    ) -> list[PolicyClauseHit]:
        """检索政策条款（三模式；无命中 → ``[]``，工具 ok=True）。

        流程：Python 侧候选过滤 → 按模式装配检索器（向量路 store 侧下推 category；BM25 路只喂
        候选 node）→ 打分/融合 → ``(分降序, corpus 原序)`` 排序 + 6 位取整 → Top-K。
        """
        candidates = _policy_candidates(self._rows, filters, effective_only)
        ranked = self._retrieve_ranked(
            query,
            candidates,
            top_k=top_k,
            store_filters=_policy_store_filters(filters),
            recheck=lambda i: i in set(candidates),
        )
        return [
            PolicyClauseHit.model_validate(self._rows[i].model_dump(mode="json"))
            for i, _score in ranked
        ]


class ChromaCaseIndex(_ChromaIndexBase):
    """``CaseIndex`` Protocol 的 Chroma + LlamaIndex 实现（``search`` 签名与工具契约一致）。

    ``CaseHit.retrieval_score`` 是**检索分，不是语义相似度**：``bm25`` = 候选集内 min-max
    归一化 BM25 分；``vector`` = ``1 − distance``；``hybrid`` = RRF 融合分（``Σ 1/(k+rank)``，
    ``k=60``，落在 ~(0, ``2/60 = 1/30``]）。取值恒 ⊂ ``[0,1]``。
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

    async def search(self, query: str, filters: CaseSearchFilters, top_k: int) -> list[CaseHit]:
        candidates = _case_candidates(self._rows, filters)
        ranked = self._retrieve_ranked(
            query,
            candidates,
            top_k=top_k,
            store_filters=_case_store_filters(filters),
            recheck=lambda i: i in set(candidates),
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
