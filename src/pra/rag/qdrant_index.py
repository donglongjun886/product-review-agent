"""Qdrant 向量索引（rag/qdrant_index.py）—— QdrantPolicyIndex / QdrantCaseIndex。

Phase 2（docs/06-rag-phase2-qdrant-bge.md §2.1/§2.3 拍板）：以 qdrant-client
**进程内模式**（``":memory:"`` 默认 / ``path=<目录>`` 本地持久 / ``url=<host>``
远端 server，均 qdrant-client 原生支持）替换本地 numpy 余弦作为「向量存储 + 余弦
打分」。Qdrant 承担的角色被刻意收窄为**只算余弦分**：

- 元数据候选过滤（category / risk_type / status）、BM25、hybrid 融合、Top-K 排序
  全部仍在 Python 侧复用 ``pra.rag.retrieval`` 的既有确定性函数（P2-4）→ 与本地
  ``RagPolicyIndex/RagCaseIndex``（rag/index.py）**同口径**：同样的过滤边界、同分
  tie-break、min-max 归一化、6 位取整 → score 语义可比。
- 与 Rag*Index 的唯一实现差异 = 底层余弦实现（Qdrant COSINE 存 float32 vs 纯
  Python float64）→ 允许 1e-6 级浮点尾差（同构测试以 ``abs <= 1e-6`` 断言，
  docs/06 §2.1）。

模块顶层**不 import qdrant-client**：仅在构造（backend="qdrant" 显式开启）时延迟
import，失败抛 ``RuntimeError`` 提示 ``uv sync --extra rag``（P2-5：默认 local 路径
零依赖、零额外 import —— 无 qdrant-client 环境照常全绿）。

设计要点（对齐 docs/06 §2.1 逐条）：
- collection：每个 KB 一个（``<prefix or "pra">_<policy|case>_<dim>``），建库
  cosine + size=dim=embedder 维度；已存在校验 dim 一致后复用（不重建）。
- point id = ``sha256(clause_id/case_id)`` **前 8 字节 → u64 无符号 int**（稳定、幂等
  upsert 覆盖 → 重建幂等）。**必须落 u64**：真 server 只收 u64 或 UUID，128 位会
  400（进程内模式不校验 → 只有连 server 才暴露；见 ``_point_id`` docstring）。
  候选 id 过滤用 ``HasIdCondition``（本版 qdrant-client
  的 Filter 嵌套条件；``PointIdsList`` 在该版本是 scroll/delete 的顶层
  FilterSelector，不能放 Filter.must —— REPL 验证后以等价原生条件实现，测试以
  行为断言为准）。
- payload 存整行 record JSON（``row.model_dump(mode="json")``；检索结果重组 hit 用，
  隔离红线 R-4 由 corpus 脱敏保证不变）。
- 检索流程 a~d 与 Rag*Index 逐条对齐（见 ``_rank_candidates`` 与各 ``search``）。
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pra.rag.bm25 import BM25Index, tokenize
from pra.rag.corpus.schema import CasePrecedentRecord, PolicyClauseRecord
from pra.rag.embedder import Embedder, MockHashEmbedder
from pra.rag.retrieval import (
    DEFAULT_WEIGHTS,
    MODES,
    RetrievalMode,
    fuse_scores,
    normalize_minmax,
)
from pra.tools.case_search.tool import CaseHit, CaseSearchFilters
from pra.tools.policy_search.tool import PolicyClauseHit, PolicySearchFilters

__all__ = ["QdrantCaseIndex", "QdrantPolicyIndex"]

_FULL_CATEGORY = "全类目"
_DEFAULT_LOCATION = ":memory:"


# ---------------------------------------------------------------------------
# qdrant 装配（延迟 import / collection / point / upsert）
# ---------------------------------------------------------------------------


def _import_qdrant() -> tuple[Any, Any]:
    """延迟 import qdrant-client（仅构造路径调用；模块顶层不 import）。

    缺包时报错并提示安装 extra（docs/06 §3：``rag = ["qdrant-client", "fastembed"]``）。
    返回 ``(QdrantClient, models)`` —— models 承载 Distance/VectorParams/PointStruct/
    Filter/HasIdCondition 等类型（版本相关，一律经此引用便于升级跟随）。
    """
    try:
        from qdrant_client import QdrantClient
        from qdrant_client import models as qdrant_models
    except ImportError as exc:  # pragma: no cover — 触发路径仅在显式开启 qdrant
        raise RuntimeError(
            "qdrant 后端需要 qdrant-client：请运行 `uv sync --extra rag` 安装；"
            '默认本地检索用 backend="local" 即可（无需该依赖）'
        ) from exc
    return QdrantClient, qdrant_models


def _make_client(qdrant_client: Any | None, location: str | Path) -> Any:
    """外部注入 client 优先；否则按 location 自建 qdrant-client 进程内模式客户端。

    location 语义（docs/06 P2-3 / §2.1，均 qdrant-client 原生 kwargs）：
    ``":memory:"``（默认）= 内存库；``http(s)://`` = 远端 server（``url=``）；
    其它路径串 = 本地持久目录（``path=<dir>`` —— 注意本版 client 的 ``location=``
    只收 ":memory:" 或 url，本地目录须走 ``path=``，故按前缀分派）。
    """
    if qdrant_client is not None:
        return qdrant_client
    QdrantClient, _models = _import_qdrant()
    loc = str(location) if location is not None else _DEFAULT_LOCATION
    if loc.startswith(("http://", "https://")):
        return QdrantClient(url=loc)
    if loc == _DEFAULT_LOCATION:
        return QdrantClient(location=loc)
    return QdrantClient(path=loc)


def _ensure_collection(client: Any, qm: Any, name: str, dim: int) -> None:
    """collection 不存在则建（cosine + size=dim）；已存在校验 dim 一致后复用。"""
    if client.collection_exists(name):
        existing = _existing_dim(client.get_collection(name))
        if existing != dim:
            raise ValueError(
                f"collection {name!r} 已存在且维度 {existing} != 期望 {dim}："
                "同名不同维不可复用（换 collection_prefix 或清理旧库）"
            )
        return
    client.create_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )


def _existing_dim(info: Any) -> int:
    """从 collection info 读出向量维度（兼容单无名向量 / 命名字段两种返回形状）。"""
    vectors = info.config.params.vectors
    if isinstance(vectors, dict):
        items = list(vectors.values())
        if len(items) != 1 or not hasattr(items[0], "size"):
            raise ValueError(
                f"collection 配置含多组/未知向量（{sorted(vectors)}），本索引只支持单向量"
            )
        return int(items[0].size)
    if hasattr(vectors, "size"):
        return int(vectors.size)
    raise ValueError(f"无法解析 collection 向量配置: {type(vectors)!r}")


def _seed_collection(
    *,
    client: Any,
    qm: Any,
    kind: str,
    rows: list[Any],
    row_keys: list[str],
    vectors: list[list[float]],
    collection_prefix: str | None,
) -> tuple[str, list[int]]:
    """ensure collection + 全量幂等 upsert doc 点 → (collection_name, point_ids)。

    point id = ``sha256(行键)`` 前 16 字节 → 无符号 int（稳定；同键重复构造 upsert
    覆盖，幂等）。空 KB（无 dim 来源）→ 不建 collection，返回 (None, []) —— 检索
    恒空（候选为空提前返回 []）。
    """
    if not rows:
        return "", []
    dim = len(vectors[0])
    name = _collection_name(collection_prefix, kind, dim)
    _ensure_collection(client, qm, name, dim)
    point_ids = [_point_id(k) for k in row_keys]
    if len(set(point_ids)) != len(point_ids):
        raise ValueError("point id 冲突（sha256 前缀碰撞）——检查 corpus 行键唯一性")
    client.upsert(
        collection_name=name,
        points=[
            qm.PointStruct(id=pid, vector=vec, payload=row.model_dump(mode="json"))
            for pid, vec, row in zip(point_ids, vectors, rows)
        ],
    )
    return name, point_ids


def _point_id(key: str) -> int:
    """point id = ``sha256(key)`` **前 8 字节** → u64 无符号 int（稳定、幂等 upsert 覆盖）。

    **为什么是 8 字节（u64）而不是 16 字节**：Qdrant 服务端只接受 **u64 整数或
    UUID** 形式的 point id，超出即 ``400 Bad Request``。而 qdrant-client 的
    **进程内模式（``:memory:`` / ``path=``）不校验 id 上界** —— 取 16 字节（128 位）
    时内存/本地路径全绿，**只有连真 server（``url=``）才炸**。2026-09-10 实测暴露，
    详见 ``deploy/qdrant/README.md`` §7。

    u64 熵对 KB 规模足够，且同键碰撞由调用方
    （``len(set(point_ids)) != len(point_ids)`` → ``ValueError``）显式拦截，
    不依赖「不会撞」的假设。
    """
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def _collection_name(prefix: str | None, kind: str, dim: int) -> str:
    """collection 名：``<prefix or "pra">_<policy|case>_<dim>``（docs/06 §2.1）。"""
    return f"{prefix or 'pra'}_{kind}_{dim}"


# ---------------------------------------------------------------------------
# 检索：Python 侧过滤 + 打分/排序（与 rag/retrieval.py 同口径）
# ---------------------------------------------------------------------------


def _validate_mode(mode: str) -> RetrievalMode:
    """与 rag/index.py ``_validate_mode`` 同口径：非法模式即报错（不静默降级）。"""
    if mode not in MODES:
        raise ValueError(f"未知检索模式: {mode!r}（可选: {list(MODES)}）")
    return mode  # type: ignore[return-value]


def _normalize_rows(rows: Iterable[Any], record_type: type) -> list[Any]:
    """rows（dict 或 record 模型）→ 强校验的 record 列表（与 rag/index.py 同源码）。"""
    out: list[Any] = []
    for row in rows:
        out.append(record_type.model_validate(row) if isinstance(row, dict) else row)
    for row in out:
        if not isinstance(row, record_type):
            raise TypeError(f"corpus 行须为 {record_type.__name__} 或 dict: got {type(row)!r}")
    return out


def _query_candidate_vector_scores(
    client: Any, qm: Any, collection_name: str, query_vec: list[float], point_ids: list[int]
) -> list[float]:
    """候选 point 的余弦分（顺序与 ``point_ids`` 一致）。

    Qdrant 承担「向量存储 + 余弦打分」：query embed 后 ``query_points`` 以
    ``Filter(must=[HasIdCondition(has_id=候选 ids)])`` 把打分范围收窄到候选集
    （等价于设计稿的 "point_id ∈ 候选" 的 id-set 过滤；本版 client 的
    ``PointIdsList`` 是 scroll/delete 顶层 FilterSelector 不能放 Filter.must ——
    REPL 验证后改用原生 ``HasIdCondition``，行为以测试断言为准）。
    ``limit=len(候选)`` 取回全部候选的余弦分。

    零向量 query → 全 0（与 rag/vectors.py cosine 对零向量返 0.0 口径一致，避开
    qdrant 除零路径）。qdrant 返回数 == 候选数（本库 upsert 全量、id 稳定）；若缺
    （理论不应发生，防御性注释）按 0 补位以保与候选顺序对齐。
    """
    if not point_ids:
        return []
    if all(x == 0.0 for x in query_vec):
        return [0.0] * len(point_ids)
    res = client.query_points(
        collection_name=collection_name,
        query=list(query_vec),
        query_filter=qm.Filter(must=[qm.HasIdCondition(has_id=point_ids)]),
        limit=len(point_ids),
        with_payload=False,
    )
    score_by_id = {p.id: p.score for p in res.points}
    return [score_by_id.get(pid, 0.0) for pid in point_ids]


def _rank_candidates(
    *,
    query: str,
    embedder: Embedder,
    bm25: BM25Index,
    client: Any,
    qm: Any,
    collection_name: str,
    candidates: list[int],
    point_ids: list[int],
    mode: RetrievalMode,
    weights: tuple[float, float],
    top_k: int,
) -> list[tuple[int, float]]:
    """候选打分 + 排序 + Top-K（与 rag/retrieval.rank_documents 同口径）。

    三模式（docs/06 §2.1 打分节）：
    - ``bm25``：``BM25Index.scores`` + 候选内 min-max（``normalize_minmax``）；
    - ``vector``：qdrant 余弦分（仅对候选打分，见 ``_query_candidate_vector_scores``）；
    - ``hybrid``：``fuse_scores(bm25_raw, qdrant_vec, weights, normalize=True)``
      —— 与本地完全同函数同权重（默认 0.5/0.5）。

    排序在 **Python 侧**做：分 6 位取整 → ``(score 降序, 候选原序 idx 升序)`` ——
    不信任 qdrant 对同分点的顺序，确定性 tie-break 与本地一致（docs/06 §2.1）。
    返回 ``[(行索引, 6 位取整分)]``（已按上述 key 排序、截断 Top-K）。
    """
    if top_k < 1 or not candidates:
        return []
    bm25_raw = bm25.scores(tokenize(query), candidates)
    if mode == "bm25":
        final = normalize_minmax(bm25_raw)
    else:
        vec_scores = _query_candidate_vector_scores(
            client, qm, collection_name, embedder.embed(query), point_ids
        )
        if mode == "vector":
            final = vec_scores
        else:  # hybrid
            final = fuse_scores(bm25_raw, vec_scores, weights=weights, normalize=True)
    ranked = [(idx, round(s, 6)) for idx, s in zip(candidates, final)]
    ranked.sort(key=lambda t: (-t[1], t[0]))
    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Policy / Case 两个公开索引类（构造/检索签名与 Rag*Index 对齐）
# ---------------------------------------------------------------------------


class QdrantPolicyIndex:
    """PolicyIndex 的 Qdrant 实现：元数据过滤在 Python 侧 + Qdrant 余弦打分。

    构造参数与 ``RagPolicyIndex``（rag/index.py）对齐，另加 qdrant 装配参数
    （``qdrant_client`` 外部注入 / ``location`` 自建 client / ``collection_prefix``）。
    检索语义（过滤边界 / 打分 / tie-break）与 ``RagPolicyIndex.search`` **逐条一致**
    （docs/06 §2.1），差异仅在向量打分底层（Qdrant COSINE float32 vs 纯 Python
    float64 → 允许 1e-6 级尾差）。
    """

    _kind = "policy"

    def __init__(
        self,
        rows: Iterable[dict | PolicyClauseRecord],
        *,
        embedder: Embedder | None = None,
        mode: RetrievalMode = "hybrid",
        weights: tuple[float, float] = DEFAULT_WEIGHTS,
        qdrant_client: Any | None = None,
        location: str | Path = _DEFAULT_LOCATION,
        collection_prefix: str | None = None,
    ) -> None:
        self._rows: list[PolicyClauseRecord] = _normalize_rows(rows, PolicyClauseRecord)
        self.mode: RetrievalMode = _validate_mode(mode)
        self.weights: tuple[float, float] = tuple(weights)
        self.embedder: Embedder = embedder or MockHashEmbedder()
        self._texts: list[str] = [
            f"{r.title}。{r.text}" for r in self._rows  # title+text 均为检索文本
        ]
        self._bm25 = BM25Index(self._texts)
        # 构造期一次性建 doc 向量（与 Rag*Index 同耗时/确定性），随即送入 qdrant。
        self._doc_vectors: list[list[float]] = [
            self.embedder.embed(t) for t in self._texts
        ]
        self._client = _make_client(qdrant_client, location)
        _QdrantClient, qm = _import_qdrant()
        self._qm = qm
        self._collection_name, self._point_ids = _seed_collection(
            client=self._client,
            qm=qm,
            kind=self._kind,
            rows=self._rows,
            row_keys=[r.clause_id for r in self._rows],
            vectors=self._doc_vectors,
            collection_prefix=collection_prefix,
        )

    @property
    def size(self) -> int:
        """Policy KB 条款数（含 EXPIRED 历史版）。"""
        return len(self._rows)

    def effective_count(self) -> int:
        return sum(1 for r in self._rows if r.status == "EFFECTIVE")

    async def search(
        self,
        query: str,
        filters: PolicySearchFilters,
        top_k: int,
        effective_only: bool,
    ) -> list[PolicyClauseHit]:
        """检索政策条款 —— 语义与 ``RagPolicyIndex.search`` 逐条一致（docs/06 §2.1）。

        检索流程 a~d（KB 极小 24 条，qdrant 调用为同步小查询；async 包装仅为对齐
        tools 层 Protocol —— 取舍注释）：
        a. Python 侧元数据候选过滤（effective_only → status==EFFECTIVE；category ∈
           {None, filters.category, 全类目}；risk_type 交叠非空——空行 risk_type 在
           给了 filter 时被排除）；候选空 → 直接返回 []；
        b. BM25 构造期已建（policy 检索文本 = title。text）；
        c. bm25/vector/hybrid 三模式打分（同 retrieval 口径，vector 分取 qdrant）；
        d. Python 侧排序（6 位取整，(分降序, 候选原序 idx 升序)）→ Top-K → 重组 hit。
        """
        candidates: list[int] = []
        for i, r in enumerate(self._rows):
            if effective_only and r.status != "EFFECTIVE":
                continue
            if filters.category and r.category not in (None, filters.category, _FULL_CATEGORY):
                continue
            if filters.risk_type:
                wanted = set(filters.risk_type)
                if not (wanted & set(r.risk_type)):
                    continue
            candidates.append(i)
        if not candidates:
            return []

        ranked = _rank_candidates(
            query=query,
            embedder=self.embedder,
            bm25=self._bm25,
            client=self._client,
            qm=self._qm,
            collection_name=self._collection_name,
            candidates=candidates,
            point_ids=[self._point_ids[i] for i in candidates],
            mode=self.mode,
            weights=self.weights,
            top_k=top_k,
        )
        return [
            PolicyClauseHit.model_validate(self._rows[i].model_dump(mode="json"))
            for i, _score in ranked
        ]


class QdrantCaseIndex:
    """CaseIndex 的 Qdrant 实现：元数据过滤在 Python 侧 + Qdrant 余弦打分。

    构造/检索语义与 ``QdrantPolicyIndex`` 同架构，与 ``RagCaseIndex``（rag/index.py）
    逐条一致：category **精确匹配**（filters.category 为空不过滤）、risk_type 交叠
    非空；similarity = 检索期最终分（0~1）写入 ``CaseHit.similarity`` 作证据 weight。
    """

    _kind = "case"

    def __init__(
        self,
        rows: Iterable[dict | CasePrecedentRecord],
        *,
        embedder: Embedder | None = None,
        mode: RetrievalMode = "hybrid",
        weights: tuple[float, float] = DEFAULT_WEIGHTS,
        qdrant_client: Any | None = None,
        location: str | Path = _DEFAULT_LOCATION,
        collection_prefix: str | None = None,
    ) -> None:
        self._rows: list[CasePrecedentRecord] = _normalize_rows(rows, CasePrecedentRecord)
        self.mode: RetrievalMode = _validate_mode(mode)
        self.weights: tuple[float, float] = tuple(weights)
        self.embedder: Embedder = embedder or MockHashEmbedder()
        self._texts: list[str] = [r.summary for r in self._rows]
        self._bm25 = BM25Index(self._texts)
        self._doc_vectors: list[list[float]] = [
            self.embedder.embed(t) for t in self._texts
        ]
        self._client = _make_client(qdrant_client, location)
        _QdrantClient, qm = _import_qdrant()
        self._qm = qm
        self._collection_name, self._point_ids = _seed_collection(
            client=self._client,
            qm=qm,
            kind=self._kind,
            rows=self._rows,
            row_keys=[r.case_id for r in self._rows],
            vectors=self._doc_vectors,
            collection_prefix=collection_prefix,
        )

    @property
    def size(self) -> int:
        """Case KB 先例数。"""
        return len(self._rows)

    async def search(
        self, query: str, filters: CaseSearchFilters, top_k: int
    ) -> list[CaseHit]:
        """检索先例 —— 语义与 ``RagCaseIndex.search`` 逐条一致（docs/06 §2.1）。

        检索流程同 QdrantPolicyIndex（a~d）：category 精确匹配、risk_type 交叠非空；
        无命中返回空列表（工具 ok=True）。
        """
        candidates: list[int] = []
        for i, r in enumerate(self._rows):
            if filters.category and r.category != filters.category:
                continue
            if filters.risk_type:
                wanted = set(filters.risk_type)
                if not (wanted & set(r.risk_type)):
                    continue
            candidates.append(i)
        if not candidates:
            return []

        ranked = _rank_candidates(
            query=query,
            embedder=self.embedder,
            bm25=self._bm25,
            client=self._client,
            qm=self._qm,
            collection_name=self._collection_name,
            candidates=candidates,
            point_ids=[self._point_ids[i] for i in candidates],
            mode=self.mode,
            weights=self.weights,
            top_k=top_k,
        )
        hits: list[CaseHit] = []
        for i, score in ranked:
            row = self._rows[i]
            hits.append(
                CaseHit.model_validate(
                    {
                        **row.model_dump(mode="json"),
                        "similarity": score,  # 检索期最终分 → CaseHit.similarity
                    }
                )
            )
        return hits
