"""ChromaDB + LlamaIndex 检索后端 —— ``ChromaPolicyIndex`` / ``ChromaCaseIndex``。

查询链路：向量路（ChromaDB cosine）+ BM25 路（``bm25s`` + jieba）→ RRF 融合 → Top-K。
业务边界由 tools 层 Protocol 守护；1 条款 / 1 先例 = 1 Node，不切分，node id 由「collection
名 + corpus 行键」哈希得出，重建幂等覆盖。
三种模式：``vector`` = ``1 − distance``；``bm25`` = 候选集内 min-max 归一化；``hybrid`` =
RRF（``Σ 1/(60+rank)``）。写进 ``CaseHit.retrieval_score`` 的是检索分，不是语义相似度。
建 collection 两条缺一即静默出错：``embedding_function=None``（否则启用默认 ONNX 嵌入函数并
去下模型）、``space="cosine"``（Chroma 缺省 ``l2``，让「相似度 = 1 − distance」失效）；已存在的
l2 collection 传配置**不会**被改建，故创建与复用两条路径都校验空间。过滤位置两路不统一：向量路
把 ``category`` 下推到 store，BM25 路只喂候选 node，最终判定是同一份 Python 谓词；``risk_type``
只能在 Python 侧判（Chroma ``$in`` 对列表字段恒不命中、LlamaIndex 的 ANY/CONTAINS 无翻译）。
``bm25s.tokenize`` 无注入点，故构造与检索期间在锁内换成 jieba 实现；这**不是「天然线程安全」**，
``_TOKENIZER_LOCK`` 必须写在前面。链路零 LLM、零随机；tie-break 为「分降序、corpus 原序升序」。
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from typing import Any

# 顶层只 import 仓库内模块 + 标准库；chroma / llama_index / jieba 一律延迟 import
# （默认 local 路径零额外依赖，构造/检索时才拉起）。
from pra.rag.corpus.schema import CasePrecedentRecord, PolicyClauseRecord
from pra.rag.embedder import Embedder, MockHashEmbedder
from pra.rag.llama_embedding import LlamaIndexEmbeddingAdapter
from pra.rag.retrieval import (
    DEFAULT_WEIGHTS,
    MODES,
    RetrievalMode,
    normalize_minmax,
)
from pra.rag.vectors import cosine_similarity
from pra.tools.case_search.tool import CaseHit, CaseSearchFilters
from pra.tools.policy_search.tool import PolicyClauseHit, PolicySearchFilters

__all__ = [
    "CHROMA_DEFAULT_HOST",
    "CHROMA_DEFAULT_PORT",
    "COLLECTION_NAME_TEMPLATE",
    "SERVED_COUNTERS",
    "ChromaCaseIndex",
    "ChromaPolicyIndex",
    "check_cosine_self_check",
    "check_cosine_space",
    "delete_collection",
    "make_chroma_client",
    "reset_served_counters",
    "served_counters",
]

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 默认 Chroma 服务端地址（deploy/chroma：宿主 8001 → 容器 8000；只用 `/api/v2`）。
CHROMA_DEFAULT_HOST = "127.0.0.1"
CHROMA_DEFAULT_PORT = 8001

#: collection 名形状（与 qdrant 后端的 `_collection_name` 同形）。
COLLECTION_NAME_TEMPLATE = "<prefix or 'pra'>_<policy|case>_<dim>"

_DEFAULT_COLLECTION_PREFIX = "pra"
_FULL_CATEGORY = "全类目"
#: metadata 维度键（Chroma 不声明向量维度，故把本索引声明的 dim 存进 collection metadata 复用校验）。
_DIM_KEY = "pra_dim"
#: 风险类型列表键（空列表不写 —— Chroma 1.5.9 拒绝空列表 metadata 值）。
_RISK_TYPE_KEY = "risk_type"

#: RRF 常数 k=60（与 ``QueryFusionRetriever`` 的融合同源：Cormack 2009）。
_RRF_K = 60.0

#: 本进程内经本模块发出的计数（验收断言用）：llm 必须恒为 0（检索侧零 LLM）。
SERVED_COUNTERS: dict[str, int] = {
    "llm_calls": 0,
    "vector_searches": 0,
    "bm25_searches": 0,
    "hybrid_searches": 0,
    "served_hits": 0,
    #: 向量路在 Chroma 少返候选时，由本模块按已存向量补算 cos 的次数（正常应恒为 0）。
    "vector_bruteforce_fallbacks": 0,
}

#: 向量路覆盖率自检的重试次数（第三方少返是已知风险；重试后仍不覆盖即抛）。
_VECTOR_COVERAGE_ATTEMPTS = 3

_LATIN_RUN = re.compile(r"[0-9A-Za-z_]+")


# ---------------------------------------------------------------------------
# 延迟 import / 客户端 / collection 装配
# ---------------------------------------------------------------------------


def _import_chroma() -> tuple[Any, Any]:
    try:
        import chromadb
        from chromadb.errors import NotFoundError as ChromaNotFoundError
    except ImportError as exc:  # pragma: no cover — 触发路径仅在显式开启 chroma 后端
        raise RuntimeError(
            "chroma 后端需要 chromadb 与 llama-index 集成包：请运行 `uv sync --extra rag` 安装；"
            '默认本地检索用 backend="local" 即可（无需该依赖）'
        ) from exc
    return chromadb, ChromaNotFoundError


def _import_llama() -> dict[str, Any]:
    """延迟 import LlamaIndex 装配面（仅构造路径调用）。

    只 import core + retrievers-bm25 + vector-stores-chroma 三个具体集成，不引伞包
    ``llama-index``（伞包会拖进 llms-openai / embeddings-openai 等不用的集成）。
    """
    try:
        from llama_index.core.base.base_retriever import BaseRetriever
        from llama_index.core.llms import MockLLM
        from llama_index.core.retrievers import QueryFusionRetriever
        from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
        from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
        from llama_index.core.vector_stores.types import FilterOperator
        from llama_index.retrievers.bm25 import BM25Retriever
        from llama_index.vector_stores.chroma import ChromaVectorStore
    except ImportError as exc:  # pragma: no cover — 触发路径仅在显式开启 chroma 后端
        raise RuntimeError(
            "chroma 后端需要 llama-index-core / llama-index-vector-stores-chroma / "
            "llama-index-retrievers-bm25：请运行 `uv sync --extra rag` 安装；"
            '默认本地检索用 backend="local" 即可（无需该依赖）'
        ) from exc
    return {
        "BaseRetriever": BaseRetriever,
        "MockLLM": MockLLM,
        "QueryFusionRetriever": QueryFusionRetriever,
        "NodeWithScore": NodeWithScore,
        "QueryBundle": QueryBundle,
        "TextNode": TextNode,
        "MetadataFilter": MetadataFilter,
        "MetadataFilters": MetadataFilters,
        "FilterOperator": FilterOperator,
        "BM25Retriever": BM25Retriever,
        "ChromaVectorStore": ChromaVectorStore,
    }


def make_chroma_client(
    chroma_client: Any | None = None,
    *,
    host: str = CHROMA_DEFAULT_HOST,
    port: int = CHROMA_DEFAULT_PORT,
    ephemeral: bool = False,
) -> Any:
    """取用/自建 Chroma 客户端（外部注入优先）。

    ``ephemeral=True`` → 进程内 ``EphemeralClient``；否则 ``HttpClient(host, port)``。
    **构造不联网**（``HttpClient`` 只做参数装配，首个请求才连接）；匿名遥测显式关闭。
    """
    if chroma_client is not None:
        return chroma_client
    chromadb, _not_found = _import_chroma()
    if ephemeral:
        return chromadb.EphemeralClient()
    try:
        from chromadb.config import Settings as ChromaSettings

        return chromadb.HttpClient(
            host=host, port=port, settings=ChromaSettings(anonymized_telemetry=False)
        )
    except Exception:  # noqa: BLE001  # pragma: no cover —— 兼容无 Settings 参数的 chromadb 变体
        return chromadb.HttpClient(host=host, port=port)


def _collection_name(prefix: str | None, kind: str, dim: int) -> str:
    return f"{prefix or _DEFAULT_COLLECTION_PREFIX}_{kind}_{dim}"


def _node_id(collection: str, key: str) -> str:
    """node id = ``sha256(collection + 行键)`` 前 16 字节 hex（稳定、幂等覆盖）。

    collection 名参与哈希，避免 policy / case KB 前缀相同时 node id 撞车。
    """
    digest = hashlib.sha256(f"{collection}\x1f{key}".encode()).hexdigest()[:32]
    return f"pra-{digest}"


def _resolve_dim(embedder: Embedder, doc_vectors: list[list[float]]) -> int:
    dim = getattr(embedder, "dim", None)
    if isinstance(dim, int) and dim > 0:
        if doc_vectors and len(doc_vectors[0]) != dim:
            raise ValueError(
                f"embedder 声明维度 {dim} 与 doc 向量长度 {len(doc_vectors[0])} 不一致"
            )
        return dim
    if not doc_vectors:
        raise ValueError("无法确定向量维度：corpus 为空且 embedder 未声明 dim")
    return len(doc_vectors[0])


#: collection 的期望向量空间（**必须显式 cosine**）：Chroma 默认是 ``l2``，
#: 而本模块「相似度 = 1 − distance」的整套口径只在 cosine 空间成立。
_REQUIRED_SPACE = "cosine"

#: cosine 自检容差（实测 distance==0 时精确为 0；浮点尾差留 1e-6）。
_SELF_CHECK_TOL = 1e-6


def _collection_space(collection: Any) -> str | None:
    """读 collection 的向量空间（优先新式 ``configuration["hnsw"]["space"]``，回落旧式 metadata）。

    都读不到 → ``None``（调用方按「无法确认」处理，不假定 cosine）。
    """
    # 新式（chromadb ≥ 0.6/1.x）：实测 1.5.9 建库时传 configuration 或 metadata 都会在
    # config 里体现为 "cosine"；旧式（老版本 API）只有 metadata["hnsw:space"]。
    cfg = getattr(collection, "configuration", None) or {}
    hnsw = cfg.get("hnsw") if isinstance(cfg, dict) else None
    if isinstance(hnsw, dict) and hnsw.get("space"):
        return str(hnsw["space"]).lower()
    meta = getattr(collection, "metadata", None) or {}
    for key in ("hnsw:space", "space"):
        if meta.get(key):
            return str(meta[key]).lower()
    return None


def _verify_collection_space(collection: Any, name: str, *, context: str) -> None:
    """**断言 collection 真的是 cosine 空间**，不是 cosine 就报错（绝不静默降级）。

    ``embedding_function=None`` **只管住默认 ONNX 嵌入函数，不管向量空间** —— Chroma 默认空间是
    **l2**。实测单位向量 ``[1,0,0]`` vs ``[0.9,0.1,0]``：默认 l2 → distance ``0.020000005``
    （``1 − distance`` = 0.979999995 而 numpy 余弦 0.993883735，**静默错**）；显式 cosine →
    ``0.006116271`` → ``1 − distance`` = 0.993883729。**对已存在的 l2 collection**，再传 cosine
    配置**不会**改建库空间 —— 故新建与复用**两条路径都校验**。读不到空间（``None``）时也报错：
    宁可让调用方显式清理/换名前缀，也不在未知空间上检索。
    """
    space = _collection_space(collection)
    if space != _REQUIRED_SPACE:
        raise ValueError(
            f"collection {name!r} 的向量空间是 {space!r}（{context}），"
            f"必须是 {_REQUIRED_SPACE!r}：Chroma 默认空间是 l2，"
            "而本索引的『相似度 = 1 − distance』只在 cosine 空间成立。"
            "修法：显式建库 configuration={'hnsw': {'space': 'cosine'}}"
            f"（已存在且空间不符的库不会被自动改建 —— 请删除后重建：delete_collection({name!r})，"
            "或换 collection_prefix 用新库）"
        )


def check_cosine_self_check(
    collection: Any, sample_vector: list[float], *, name: str
) -> float:
    """cosine 自检：用库内已有向量查库，返回自身距离（cosine 空间须 ≈ 0）。

    实测自距离有 ``0.0`` 与 ``-1.1920929e-07``（float32 尾差）两类，故用
    ``abs() <= _SELF_CHECK_TOL`` 判定。自距离在 l2 空间同样为 0，单独看不区分空间；本函数是
    第二道防线：抓「库未 upsert / 被清空 / 空间被外力改掉后数值不可信」。
    """
    result = collection.query(
        query_embeddings=[list(sample_vector)], n_results=1, include=["distances"]
    )
    distances = (result.get("distances") or [[]])[0]
    if not distances:
        raise ValueError("cosine 自检失败：collection 查询未返回任何结果（库是否已 upsert？）")
    value = float(distances[0])
    if abs(value) > _SELF_CHECK_TOL:
        raise ValueError(
            f"cosine 自检失败：collection {name!r} 内向量与自身距离 {value!r}"
            f"（cosine 空间应为 0，容差 {_SELF_CHECK_TOL}）—— 库可能未正确 upsert 或空间异常"
        )
    return value


def check_cosine_space(collection: Any, name: str) -> str:
    """检索前复检空间（每次 ``search`` 都执行，返回空间名）。

    成本可忽略（本地字段读取）；收益是空间错误永远在检索前暴露，而不是让错误的
    ``1 − distance`` 静默流进 ``CaseHit.retrieval_score``。
    """
    _verify_collection_space(collection, name, context="检索前复检")
    return _collection_space(collection) or _REQUIRED_SPACE


def _open_collection(
    *, client: Any, name: str, dim: int, kind: str, prefix: str
) -> tuple[Any, bool]:
    """``get_or_create_collection``（**必须 ``embedding_function=None`` + 显式 cosine 空间**）。

    不关 ``embedding_function`` 会启用默认 ONNX 嵌入函数并去下模型；不显式 cosine 则空间是
    Chroma 缺省的 l2。返回 ``(collection, reused)``：两条路径都校验空间（复用路径尤其重要 ——
    对已存在的 l2 库再传 cosine 配置**不会**改建库空间）。
    """
    _import_chroma()  # 缺包早失败（提示装 extra）；本函数只用注入进来的 client
    try:
        existing_collection = client.get_collection(name=name)
    except Exception as exc:
        if "not found" not in str(exc).lower() and type(exc).__name__ != "NotFoundError":
            raise
    else:
        _verify_collection_space(existing_collection, name, context="已存在，复用")
        return existing_collection, True
    collection = client.get_or_create_collection(
        name=name,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},  # ★ 显式 cosine：默认是 l2
        metadata={_DIM_KEY: int(dim), "pra_kind": kind, "pra_prefix": prefix},
    )
    _verify_collection_space(collection, name, context="本次新建")
    return collection, False


def _validate_collection_dim(collection: Any, name: str, dim: int) -> None:
    existing = (collection.metadata or {}).get(_DIM_KEY)
    if existing is not None and int(existing) != int(dim):
        raise ValueError(
            f"collection {name!r} 已存在且维度 {existing} != 期望 {dim}："
            "同名不同维不可复用（换 collection_prefix 或清理旧库）"
        )


def delete_collection(
    name: str,
    *,
    chroma_client: Any | None = None,
    host: str = CHROMA_DEFAULT_HOST,
    port: int = CHROMA_DEFAULT_PORT,
) -> bool:
    client = make_chroma_client(chroma_client, host=host, port=port)
    try:
        client.delete_collection(name)
        return True
    except Exception as exc:
        if "not found" in str(exc).lower() or type(exc).__name__ == "NotFoundError":
            return False
        raise


def served_counters() -> dict[str, int]:
    return dict(SERVED_COUNTERS)


def reset_served_counters() -> None:
    for key in SERVED_COUNTERS:
        SERVED_COUNTERS[key] = 0


# ---------------------------------------------------------------------------
# jieba 分词接入（bm25s.tokenize 的受控替换）
# ---------------------------------------------------------------------------

#: 全局替换锁：bm25s.tokenize 是模块级符号，构造与检索期间独占。
_TOKENIZER_LOCK = threading.RLock()


def _jieba_tokens(text: str) -> list[str]:
    """jieba 精确模式切词（确定性；拉丁/数字串拆出并小写，中文词原样）。

    不做停用词过滤/词干化（中文语料上英文词干器无意义）；单字符词保留（文档频率高、IDF 低，
    对排序影响可忽略）。
    """
    import jieba

    tokens: list[str] = []
    for piece in jieba.cut(text, cut_all=False):
        piece = piece.strip()
        if not piece:
            continue
        for latin in _LATIN_RUN.findall(piece):
            tokens.append(latin.lower())
        if _LATIN_RUN.sub("", piece):
            tokens.append(piece)
    return tokens


def _jieba_bm25s_tokenize(texts: Any, **_kwargs: Any) -> Any:
    """``bm25s.tokenize`` 的 jieba 替身（签名兼容：忽略 token_pattern/stemmer/stopwords 等）。

    逐文档切词 → 词表「首次出现即编号」（跨文档共享，确定性）；查询与索引走同一函数 →
    缺失查询词也进词表，不会触发 bm25s 的越界 token id 报错。
    """
    from bm25s.tokenization import Tokenized

    single = isinstance(texts, str)
    items = [texts] if single else list(texts)
    vocab: dict[str, int] = {}
    ids: list[list[int]] = []
    for text in items:
        doc_ids: list[int] = []
        for tok in _jieba_tokens(str(text)):
            if tok not in vocab:
                vocab[tok] = len(vocab)
            doc_ids.append(vocab[tok])
        ids.append(doc_ids)
    return Tokenized(ids=ids, vocab=vocab)


@contextmanager
def _jieba_tokenizer():
    """在上下文内把 ``bm25s.tokenize`` 换成 jieba 实现（退出即恢复原符号）。

    ``BM25Retriever`` 的构造与 ``retrieve`` 都硬编码调用 ``bm25s.tokenize`` 且没有注入点，故索引
    与查询都必须在此上下文内完成 —— 保证两者是同一个分词器。

    🔴 **调用方必须写成 ``with _TOKENIZER_LOCK, _jieba_tokenizer():``（锁在前、补丁在后）**：
    上下文管理器按「左→右 __enter__、右→左 __exit__」执行；写成 ``with _jieba_tokenizer(),
    _TOKENIZER_LOCK:`` 则补丁在取锁**之前**、恢复在放锁**之后** —— 临界区不覆盖补丁的安装/
    撤销，并发下必然出错：线程 B 会把「A 打的补丁」当 ``original`` 存下，A 退出即恢复真身 →
    B 在自以为的 jieba 上下文里用真分词器检索 jieba 建的索引（``ValueError: The maximum token
    ID in the query ... is higher than the number of tokens in the index.``）；B 退出再把补丁
    写回 → ``bm25s.tokenize`` 进程级永久泄漏。
    """
    import bm25s

    original = bm25s.tokenize
    bm25s.tokenize = _jieba_bm25s_tokenize
    try:
        yield
    finally:
        bm25s.tokenize = original


# ---------------------------------------------------------------------------
# corpus 行 → LlamaIndex Node（1 条款 / 1 先例 = 1 Node，不切碎）
# ---------------------------------------------------------------------------


def _iso_or_empty(value: date | None) -> str:
    return value.isoformat() if value is not None else ""


def policy_node_metadata(row: PolicyClauseRecord) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "clause_id": row.clause_id,
        "policy_id": row.policy_id,
        "version": int(row.version),
        "category": row.category,
        "status": row.status,
        "effective_date": _iso_or_empty(row.effective_date),
    }
    if row.risk_type:
        meta[_RISK_TYPE_KEY] = [t.value for t in row.risk_type]
    return meta


def case_node_metadata(row: CasePrecedentRecord) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "case_id": row.case_id,
        "category": row.category,
        "decision": row.decision.value,
        "risk_level": row.risk_level.value,
    }
    if row.risk_type:
        meta[_RISK_TYPE_KEY] = [t.value for t in row.risk_type]
    return meta


def _policy_text(row: PolicyClauseRecord) -> str:
    return f"{row.title}。{row.text}"


def _case_text(row: CasePrecedentRecord) -> str:
    return row.summary


def _build_nodes(
    rows: list[Any], *, kind: str, collection: str, llama: dict[str, Any]
) -> tuple[list[Any], list[str]]:
    """corpus 行 → (TextNode 列表, node id 列表)。**1 行 = 1 Node，不切分**。

    检索文本 = 正文（policy：``title。text``；case：``summary``），与向量路 embed 的文本、
    local 后端同一份。

    **metadata 不进检索文本**：``excluded_embed_metadata_keys`` / ``excluded_llm_metadata_keys``
    设为全部 metadata 键，使 ``get_content(metadata_mode=EMBED)`` 只返回正文 —— 这对 BM25 路
    **必需**（``BM25Retriever`` 用 ``MetadataMode.EMBED``），不排除就会把 ``case_id`` /
    ``category`` / ``decision`` / ``risk_type`` 的字面值索引进去，出现「按 metadata 字面值就能
    命中」的伪检索。排除设置随 ``_node_content`` JSON 往返存活；metadata 本身仍完整保留。
    """
    nodes: list[Any] = []
    node_ids: list[str] = []
    for row in rows:
        key = row.clause_id if kind == "policy" else row.case_id
        nid = _node_id(collection, key)
        node_ids.append(nid)
        meta = policy_node_metadata(row) if kind == "policy" else case_node_metadata(row)
        text = _policy_text(row) if kind == "policy" else _case_text(row)
        nodes.append(
            llama["TextNode"](
                id_=nid,
                text=text,
                metadata=meta,
                # metadata 不进检索文本（EMBED/LLM 两个模式都排除；ALL 仍保留供审计）。
                excluded_embed_metadata_keys=list(meta.keys()),
                excluded_llm_metadata_keys=list(meta.keys()),
            )
        )
    return nodes, node_ids


# ---------------------------------------------------------------------------
# 候选过滤（policy / case 各一份谓词；向量路与 BM25 路共用，绝无双实现）
# ---------------------------------------------------------------------------


def _policy_candidates(rows: list[PolicyClauseRecord], filters: PolicySearchFilters, effective_only: bool) -> list[int]:
    """候选行索引（语义与 local / qdrant 后端逐条一致）：``effective_only`` → ``status ==
    "EFFECTIVE"``；``category`` ∈ {None, 值, 全类目}；``risk_type`` 交叠非空。
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


def _policy_store_filters(filters: PolicySearchFilters, llama: dict[str, Any]) -> Any | None:
    """Policy 下推过滤（向量路）：仅 ``category``（含「全类目」）可下推 —— Chroma 的
    ``$in``/``$eq`` 对列表字段恒不命中，LlamaIndex 的 ANY/CONTAINS 在 chroma 集成里无翻译。
    """
    if not filters.category:
        return None
    return llama["MetadataFilters"](
        filters=[
            llama["MetadataFilter"](
                key="category",
                value=[filters.category, _FULL_CATEGORY],
                operator=llama["FilterOperator"].IN,
            )
        ]
    )


def _case_store_filters(filters: CaseSearchFilters, llama: dict[str, Any]) -> Any | None:

    if not filters.category:
        return None
    return llama["MetadataFilters"](
        filters=[
            llama["MetadataFilter"](
                key="category",
                value=filters.category,
                operator=llama["FilterOperator"].EQ,
            )
        ]
    )


# ---------------------------------------------------------------------------
# 检索器装配（向量 / BM25 / RRF 融合）
# ---------------------------------------------------------------------------


@dataclass
class _RetrievalContext:

    kind: str
    rows: list[Any]
    row_keys: list[str]
    node_ids: list[str]
    nodes: list[Any]
    collection: Any | None
    embed_model: Any
    vector_store: Any
    llama: dict[str, Any]
    #: node_id → corpus 原序行索引（排序 tie-break 用原序，不依赖底层库返回顺序）。
    row_index_by_key: dict[str, int] = field(default_factory=dict)


def _filters_to_chroma_where(store_filters: Any | None) -> dict | None:
    """LlamaIndex ``MetadataFilters`` → Chroma ``where`` dict（显式下推）。

    支持本项目实际用到的 ``in`` / ``eq``（可组合成 ``$and``）；出现别的算子即报错 —— 静默忽略
    会变成「下推了但没推」的隐性错误。向量路直接读原生 distance，不经过 ``ChromaVectorStore.query``，
    故 where 也由本函数显式转换（行为等价、可审计）。
    """
    if store_filters is None:
        return None
    items = list(getattr(store_filters, "filters", []) or [])
    clauses: list[dict] = []
    for item in items:
        op = getattr(item, "operator", None)
        key = getattr(item, "key", None)
        value = getattr(item, "value", None)
        if key is None:
            raise ValueError(f"不支持的 MetadataFilters 元素（无 key）: {item!r}")
        name = getattr(op, "value", op)
        if name in (None, "==", "eq"):
            clauses.append({key: value})
        elif name in ("in",):
            clauses.append({key: {"$in": list(value)}})
        else:
            raise ValueError(f"chroma where 下推不支持算子 {name!r}（key={key!r}）")
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _make_vector_retriever(
    ctx: _RetrievalContext,
    similarity_top_k: int,
    store_filters: Any | None,
    candidate_ids: list[str] | None = None,
) -> Any:
    """向量检索器：Chroma cosine 距离，分数 = ``1 − distance``。

    ``ChromaVectorStore.query`` 的分是 ``exp(-distance)`` —— **另一套映射、不是余弦**，故取数走
    **Chroma 原生 ``collection.query``** 自己算 ``1 − distance``；节点由 node id 直接映射，不经
    ``_node_content`` JSON 反序列化（那一步在语料含相同内容行时会与 ``node.hash`` 去重冲突）。

    ★ ``candidate_ids`` = 候选集的 node id，这是漏召回 bug 的修复点：下推的 ``where`` 只是候选
    谓词的**超集**（``risk_type`` 无法下推、``effective_only`` 也不在 ``where`` 里），若只按下推
    结果取 top-N，非候选行会按距离抢占名额，被 Python 复核剔除后没有补位 → 结果变成真子集甚至
    空。修法：把精确候选 id 交给 Chroma 的 ``ids=`` 参数，随后复核 + 截断照旧，并断言覆盖率。
    """

    class _ChromaCosineRetriever(ctx.llama["BaseRetriever"]):

        def __init__(self) -> None:
            super().__init__()
            self._collection = ctx.collection
            self._emb = ctx.embed_model
            self._k = int(similarity_top_k)
            self._where = _filters_to_chroma_where(store_filters)
            # Chroma 返回的是 node id（`_node_id()` 派生），不是 corpus 行键 —— 映射键必须用 node id。
            self._by_id = {nid: node for nid, node in zip(ctx.node_ids, ctx.nodes)}
            #: 精确候选 id（NodeWithScore 只允许这些 id 出现；None = 未限定，取 ``where`` 命中的 top-N）。
            self._candidate_ids = list(candidate_ids) if candidate_ids is not None else None

        def _query_once(self, query_embedding: list[float]) -> tuple[list[str], list[float]]:
            kwargs: dict[str, Any] = {
                "query_embeddings": [list(query_embedding)],
                "include": ["distances"],
            }
            if self._candidate_ids is not None:
                kwargs["ids"] = list(self._candidate_ids)
                kwargs["n_results"] = max(1, min(len(self._candidate_ids), int(self._collection.count())))
            else:
                kwargs["n_results"] = self._k
            if self._where is not None:
                kwargs["where"] = self._where
            result = self._collection.query(**kwargs)
            return (
                list((result.get("ids") or [[]])[0]),
                list((result.get("distances") or [[]])[0]),
            )

        def _retrieve(self, query_bundle: Any) -> list[Any]:
            if self._collection is None or not self._by_id:
                return []
            query_embedding = self._emb.get_query_embedding(query_bundle.query_str)
            # 覆盖率自检（不假定「要了 N 就一定拿到 N」）：chromadb 1.5.9 实测存在少返。
            # 重试 _VECTOR_COVERAGE_ATTEMPTS 次，仍缺真候选即抛。
            wanted = set(self._candidate_ids) if self._candidate_ids is not None else None
            ids: list[str] = []
            distances: list[float] = []
            for attempt in range(1, _VECTOR_COVERAGE_ATTEMPTS + 1):
                ids, distances = self._query_once(query_embedding)
                if wanted is None or wanted.issubset(set(ids)):
                    break
                if attempt == _VECTOR_COVERAGE_ATTEMPTS:
                    missing = sorted(wanted - set(ids))
                    raise RuntimeError(
                        f"向量路覆盖率不足：{len(missing)} 个候选 node 未从 Chroma 取回 "
                        f"（如 {missing[:5]}），已重试 {_VECTOR_COVERAGE_ATTEMPTS} 次；"
                        f"collection={getattr(self._collection, 'name', '?')!r} "
                        f"count={self._collection.count()} requested={len(wanted)} got={len(ids)}。"
                        "这会让结果静默变成子集 —— 拒绝返回，请检查 collection 是否被清理/与索引不一致。"
                    )
            out: list[Any] = []
            for nid, distance in zip(ids, distances):
                node = self._by_id.get(nid)
                if node is None:  # 防御：候选集外的 node id（不应发生）
                    continue
                out.append(
                    ctx.llama["NodeWithScore"](
                        node=node, score=_cosine_from_distance(distance)
                    )
                )
            return out

    return _ChromaCosineRetriever()


def _make_bm25_retriever(ctx: _RetrievalContext, top_k: int) -> Any:
    """BM25 检索器（``bm25s`` + jieba）：**只喂候选 node**（Python 侧过滤）。

    ``similarity_top_k=top_k`` 取候选集内 Top-K；``skip_stemming=True`` / ``language=""`` 关掉
    英文词干器与停用词（中文语料无意义）；``token_pattern=""`` 只作显式标注（jieba 替身忽略它）。
    """
    # ⚠️ 顺序不可颠倒：锁**在**补丁之前（见 :func:`_jieba_tokenizer` docstring，R6 实测）。
    with _TOKENIZER_LOCK, _jieba_tokenizer():
        return ctx.llama["BM25Retriever"](
            nodes=list(ctx.nodes),
            similarity_top_k=min(top_k, len(ctx.nodes)),
            skip_stemming=True,
            language="",
            token_pattern="",
            verbose=False,
        )


def _make_fusion_retriever(
    ctx: _RetrievalContext, vector_retriever: Any, bm25_retriever: Any, top_k: int
) -> Any:
    """RRF 融合检索器 = ``QueryFusionRetriever(mode="reciprocal_rerank", num_queries=1)``。

    ⚠️ ``num_queries=1`` **必填**：默认 4 会调用 LLM 生成 query 变体，而本项目检索侧零 LLM
    调用；不传 ``llm`` 时该库回落 ``Settings.llm`` → ``ImportError:
    llama-index-llms-openai package not found``（项目刻意不装该集成），故显式传 ``MockLLM()``
    守卫（``num_queries=1`` 下永不被调用）。**本函数不参与本模块检索链路** —— 该实现会原地改写
    共享 node 的 score，理由见 :func:`_fuse_rrf`。
    """
    return ctx.llama["QueryFusionRetriever"](
        retrievers=[vector_retriever, bm25_retriever],
        llm=ctx.llama["MockLLM"](),
        mode="reciprocal_rerank",
        similarity_top_k=min(top_k, len(ctx.nodes)),
        num_queries=1,
        use_async=False,
        verbose=False,
    )


def _fuse_rrf(
    ranked: dict[str, list[str]], row_index: dict[str, int]
) -> list[tuple[int, float]]:
    """RRF（Reciprocal Rank Fusion）：``score(d) = Σ_r 1 / (k + rank_r(d))``，``k=60``。

    融合定义与 ``QueryFusionRetriever._reciprocal_rerank_fusion`` 逐条一致，但**由本模块自己
    算**：该实现在融合时会**原地改写** ``NodeWithScore.node.score``，而 node 对象跨检索共享 →
    融合分被写回共享对象，下一次检索的结果会随此前的调用序列漂移。另：该库用 ``node.hash``
    去重，而本 corpus 存在内容相同的行 → 会把不同先例合并；本模块用 **node id**（逐行唯一）
    作融合键。

    ⚠️ **上界是 ``2/60``（≈0.0333），不是 ``2/61``**：rank 从 **0** 起（``enumerate(ids)``），
    首位贡献 ``1/(60+0)``，两路都排首位即 ``2/60 = 1/30``。返回 ``[(行索引, 6 位 RRF 分)]``，
    排序 key = ``(分降序, corpus 原序 idx 升序)``。
    """
    fused: dict[str, float] = {}
    for ids in ranked.values():
        for offset, nid in enumerate(ids):
            fused[nid] = fused.get(nid, 0.0) + 1.0 / (_RRF_K + offset)
    out = [(row_index[nid], round(score, 6)) for nid, score in fused.items()]
    out.sort(key=lambda t: (-t[1], t[0]))
    return out


# ---------------------------------------------------------------------------
# Policy / Case 两个公开索引类（构造/检索签名与既有实现对齐）
# ---------------------------------------------------------------------------


class _ChromaIndexBase:

    _kind = ""

    def _setup(
        self,
        rows: Iterable[dict | Any],
        *,
        record_type: type,
        text_of,
        embedder: Embedder | None,
        mode: str,
        weights: tuple[float, float],
        chroma_client: Any | None,
        host: str,
        port: int,
        ephemeral: bool,
        collection_prefix: str | None,
    ) -> None:
        self._rows: list[Any] = _normalize_rows(rows, record_type)
        self.mode: RetrievalMode = _validate_mode(mode)
        self.weights: tuple[float, float] = tuple(weights)  # 兼容既有构造签名（RRF 不用权重）
        self.embedder: Embedder = embedder or MockHashEmbedder()
        self._texts: list[str] = [text_of(r) for r in self._rows]
        self._row_keys: list[str] = [
            (r.clause_id if self._kind == "policy" else r.case_id) for r in self._rows
        ]
        self._llama = _import_llama()
        self._client = make_chroma_client(
            chroma_client, host=host, port=port, ephemeral=ephemeral
        )
        self._embed_model = LlamaIndexEmbeddingAdapter(self.embedder)
        self._doc_vectors: list[list[float]] = [
            self._embed_model.get_text_embedding(t) for t in self._texts
        ]
        self._dim = _resolve_dim(self.embedder, self._doc_vectors)
        self.collection_prefix = collection_prefix
        self.collection_name = (
            _collection_name(collection_prefix, self._kind, self._dim) if self._rows else ""
        )
        self._collection = None
        #: ``ChromaVectorStore``（LlamaIndex 官方 store 包装）——**装配面保留**：外部/测试可
        #: 用它走标准 ``VectorStoreQuery`` 通路（vector-stores-chroma 集成确实被构造并被
        #: 引用）；本模块自己的向量取数走原生 ``collection.query``（理由见模块
        #: docstring 的 ``vector`` 段：该 store 的 ``exp(-distance)`` 不是余弦）。
        self._vector_store = None
        self._nodes: list[Any] = []
        self._node_ids: list[str] = []
        if self._rows:
            self._seed()

    # -- 装配 ---------------------------------------------------------------

    def _seed(self) -> None:
        """建/复用 collection（``embedding_function=None`` **+ 显式 cosine 空间**）+ 幂等 upsert。

        node 向量 = 文本向量（与 local 后端同一 embedder/同一文本，逐位一致）；节点按
        「1 行 = 1 node」写入，metadata 见 ``*_node_metadata``。

        空间闸：``_open_collection`` 对**新建与复用两条路径**都断言
        collection 是 cosine（默认 l2 会让「相似度 = 1 − distance」静默失效）；upsert 后
        再做一次 cosine 自检（自身距离须 ≈ 0），失败即抛 —— **绝不带着未知空间继续检索**。
        """
        collection, reused = _open_collection(
            client=self._client,
            name=self.collection_name,
            dim=self._dim,
            kind=self._kind,
            prefix=self.collection_prefix or _DEFAULT_COLLECTION_PREFIX,
        )
        if reused:
            _validate_collection_dim(collection, self.collection_name, self._dim)
        self._nodes, self._node_ids = _build_nodes(
            self._rows,
            kind=self._kind,
            collection=self.collection_name,
            llama=self._llama,
        )
        collection.upsert(
            ids=list(self._node_ids),
            embeddings=[list(v) for v in self._doc_vectors],
            metadatas=[node.metadata for node in self._nodes],
            documents=[node.get_content() for node in self._nodes],
        )
        self.collection_space = _collection_space(collection)
        self._self_check_distance = check_cosine_self_check(
            collection, self._doc_vectors[0], name=self.collection_name
        )
        self._collection = collection
        self._vector_store = self._llama["ChromaVectorStore"](chroma_collection=collection)

    # -- size / 统计 ---------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._rows)

    @property
    def space(self) -> str | None:
        """collection 实际向量空间（构造后应为 ``"cosine"``；空 KB 未建库则为 ``None``）。

        供测试/审计读取（对应 ``configuration["hnsw"]["space"]``）。
        """
        return self.collection_space

    @property
    def cosine_self_check_distance(self) -> float | None:
        return self._self_check_distance

    @property
    def collection_count(self) -> int:
        return int(self._collection.count()) if self._collection is not None else 0

    @property
    def node_ids(self) -> list[str]:
        return list(self._node_ids)

    @property
    def nodes(self) -> list[Any]:
        return list(self._nodes)

    # -- 装配子件 -----------------------------------------------------------

    def _sub_context(self, candidates: list[int]) -> _RetrievalContext:
        """候选子集上的检索上下文（node / node id / 行索引映射都只含候选）。

        ``row_index_by_key`` 把 node id 直接映射回 **corpus 原序行索引** —— 后续排序
        tie-break 用原序，因此不依赖 Chroma / bm25s 的返回顺序。
        """
        sub_ids = [self._node_ids[i] for i in candidates]
        return _RetrievalContext(
            kind=self._kind,
            rows=[self._rows[i] for i in candidates],
            row_keys=[self._row_keys[i] for i in candidates],
            node_ids=[self._node_ids[i] for i in candidates],
            nodes=[self._nodes[i] for i in candidates],
            collection=self._collection,
            embed_model=self._embed_model,
            vector_store=self._vector_store,
            llama=self._llama,
            row_index_by_key={nid: candidates[offset] for offset, nid in enumerate(sub_ids)},
        )

    def _bm25_retrieve(self, retriever: Any, query_bundle: Any) -> list[Any]:
        """BM25 检索（**在 jieba 上下文内** —— 查询与索引必须同一分词器）。

        ⚠️ 顺序不可颠倒：锁**在**补丁之前（见 :func:`_jieba_tokenizer`）。
        """
        with _TOKENIZER_LOCK, _jieba_tokenizer():
            return retriever.retrieve(query_bundle)

    def _rank_vector(
        self,
        sub_ctx: _RetrievalContext,
        query_bundle: Any,
        top_k: int,
        store_filters: Any | None,
        *,
        count_search: bool = True,
    ) -> list[tuple[int, float]]:
        """向量路排名：``[(行索引, 1 − distance)]``（精确候选 id 集 + store 侧 category 下推）。

        与 local 后端的等价性由三件事保证：打分域 = 精确候选集（不让非候选行抢名额）；覆盖率
        自检（返回 id 必须覆盖候选 id，否则重试/抛）；覆盖率失败时用 Chroma 已存的 doc 向量
        兜底现算 cos，并累加 ``SERVED_COUNTERS["vector_bruteforce_fallbacks"]`` 使其可见。
        ``count_search`` 控制本次是否记一次 ``vector_searches``（重试/兜底不重复计数）。
        """
        retriever = _make_vector_retriever(
            sub_ctx, top_k, store_filters, candidate_ids=list(sub_ctx.node_ids)
        )
        try:
            nodes = retriever.retrieve(query_bundle)
        except RuntimeError:
            # 覆盖率重试仍不足 → 已存向量兜底（结果仍完整；计数可见）。
            nodes = []
        if count_search:
            SERVED_COUNTERS["vector_searches"] += 1
        scored: dict[str, float] = {
            n.node.node_id: n.score for n in nodes if n.node.node_id in sub_ctx.row_index_by_key
        }
        missing = [nid for nid in sub_ctx.node_ids if nid not in scored]
        if missing:
            scored.update(self._score_missing_by_stored_vectors(sub_ctx, query_bundle, missing))
            SERVED_COUNTERS["vector_bruteforce_fallbacks"] += 1
        ranked = [(sub_ctx.row_index_by_key[nid], score) for nid, score in scored.items()]
        ranked.sort(key=lambda t: (-round(t[1], 6), t[0]))
        return [(i, round(s, 6)) for i, s in ranked]

    def _score_missing_by_stored_vectors(
        self, sub_ctx: _RetrievalContext, query_bundle: Any, missing: list[str]
    ) -> dict[str, float]:
        """兜底：对未取回的候选，用 Chroma 里**已存 doc 向量**与 query 向量现算余弦。

        - 向量来源 ``collection.get(ids=missing, include=["embeddings"])``（本地读取，无 ANN 近似）；
        - 余弦用仓库既有 ``rag/vectors.cosine_similarity``（纯 Python，零第三方依赖；
          与 local numpy/纯 Python 后端同源码），再按 ``_cosine_from_distance`` 同口径 clamp 到 [0,1]；
        - 任一步失败（缺向量/长度不符）即抛，**不静默少返** —— 少返正是本次要修的 bug。
        """
        raise_on_missing = (
            f"向量路兜底失败：{len(missing)} 个候选取不到向量（如 {missing[:5]}）——"
            "拒绝返回子集结果"
        )
        if self._collection is None:
            raise RuntimeError(raise_on_missing)
        got = self._collection.get(ids=list(missing), include=["embeddings"])
        # 注意：embeddings 是 numpy 数组列表 —— 不能用 ``or []``（数组真值歧义，实测
        # ``ValueError: The truth value of an array with more than one element is ambiguous``）。
        got_ids = list(got.get("ids") if got.get("ids") is not None else [])
        raw_emb = got.get("embeddings")
        got_vecs = [list(v) for v in ([] if raw_emb is None else raw_emb)]
        if len(got_ids) != len(missing) or len(got_vecs) != len(missing):
            raise RuntimeError(
                f"{raise_on_missing}（collection.get 只回 {len(got_ids)}/{len(missing)} 条）"
            )
        query_embedding = sub_ctx.embed_model.get_query_embedding(query_bundle.query_str)
        out: dict[str, float] = {}
        for nid, vec in zip(got_ids, got_vecs):
            raw = float(cosine_similarity(list(query_embedding), vec))
            if math.isnan(raw):  # 防御：退化输入（cosine_similarity 自身已 clamp 0~1）
                raw = 0.0
            out[nid] = max(0.0, min(1.0, raw))
        return out

    def _rank_bm25(
        self, sub_ctx: _RetrievalContext, query_bundle: Any, top_k: int
    ) -> list[tuple[int, float]]:
        """BM25 路排名：候选集内 min-max 归一化 BM25 分（Python 侧过滤 = 只喂候选 node）。

        归一化口径与 local 后端 ``rank_documents(mode="bm25")`` 同函数（``normalize_minmax``）：
        候选集内最高分 = 1.0，分数恒 ⊂ [0,1]（``CaseHit.retrieval_score`` 的 ``le=1`` 约束成立）。
        """
        retriever = _make_bm25_retriever(sub_ctx, top_k)
        nodes = self._bm25_retrieve(retriever, query_bundle)
        SERVED_COUNTERS["bm25_searches"] += 1
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
        store_filters: Any | None,
    ) -> list[tuple[int, float]]:
        """Hybrid 路：两路排名 → **RRF 融合**（``Σ_r 1/(60 + rank_r)``，``k=60``）。

        两路各自返回**全部候选**（``top_k`` = 候选数）→ 在完整排名列表上融合
        （:func:`_fuse_rrf`，含「为什么不用 ``QueryFusionRetriever`` 现成融合」的实测理由）。
        ⚠️ 该分是**融合排名分，不是相似度**（上界 **2/60 = 1/30 ≈ 0.0333**）；排序 key
        = ``(分降序, corpus 原序 idx 升序)``。
        """
        vec_ranked = self._rank_vector(sub_ctx, query_bundle, top_k, store_filters)
        bm25_ranked = self._rank_bm25(sub_ctx, query_bundle, top_k)
        SERVED_COUNTERS["hybrid_searches"] += 1
        # RRF 的输入是「两路各自的排名列表」：把 (行索引, 分) 排名还原为 node id 顺序。
        id_by_row = {row: nid for nid, row in sub_ctx.row_index_by_key.items()}
        ranked_lists = {
            "vector": [id_by_row[row] for row, _score in vec_ranked],
            "bm25": [id_by_row[row] for row, _score in bm25_ranked],
        }
        return _fuse_rrf(ranked_lists, sub_ctx.row_index_by_key)

    # -- 检索核心 -----------------------------------------------------------

    def _retrieve_ranked(
        self,
        query: str,
        candidates: list[int],
        *,
        top_k: int,
        store_filters: Any | None,
        recheck,
    ) -> list[tuple[int, float]]:
        """三模式检索 → ``[(行索引, 6 位检索分)]``（已按 ``(分降序, 原序)`` 排序、已截断 Top-K）。

        向量路在 store 下推之后用 ``recheck``（**与 BM25 路同一份** Python 谓词）复核一遍：
        下推只用于收窄候选，判定只有一份实现，杜绝双实现语义漂移。

        ``top_k`` 截断在**排序之后**统一做；各路内部按需取「候选数」以保证 BM25 的
        min-max 与 RRF 的排名列表覆盖完整候选集。
        """
        if top_k < 1 or not candidates:
            return []
        if self._collection is not None:
            check_cosine_space(self._collection, self.collection_name)  # 检索前空间闸
        sub_ctx = self._sub_context(candidates)
        query_bundle = self._llama["QueryBundle"](query_str=query)
        # 三路都取「候选数」上限：先拿到**完整候选排名**，再复核过滤、最后截断 Top-K。
        # （若这里就按 top_k 预截断，store 返回的前 top_k 里一旦有被 recheck 剔除的行，
        #   结果会不足 top_k —— 实测 policy vector + effective_only 就是这样少一条。）
        full_k = len(candidates)
        if self.mode == "bm25":
            ranked = self._rank_bm25(sub_ctx, query_bundle, full_k)
        elif self.mode == "vector":
            ranked = self._rank_vector(sub_ctx, query_bundle, full_k, store_filters)
        else:
            ranked = self._rank_hybrid(sub_ctx, query_bundle, full_k, store_filters)
        ranked = [item for item in ranked if recheck(item[0])]
        return ranked[:top_k]


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


def _validate_mode(mode: str) -> RetrievalMode:
    if mode not in MODES:
        raise ValueError(f"未知检索模式: {mode!r}（可选: {list(MODES)}）")
    return mode  # type: ignore[return-value]


def _normalize_rows(rows: Iterable[Any], record_type: type) -> list[Any]:
    out: list[Any] = []
    for row in rows:
        out.append(record_type.model_validate(row) if isinstance(row, dict) else row)
    for row in out:
        if not isinstance(row, record_type):
            raise TypeError(f"corpus 行须为 {record_type.__name__} 或 dict: got {type(row)!r}")
    return out


class ChromaPolicyIndex(_ChromaIndexBase):
    """``PolicyIndex`` Protocol 的 Chroma + LlamaIndex 实现。

    构造参数与 local / qdrant 后端对齐，另加 Chroma 装配参数（``chroma_client`` 注入 /
    ``host``+``port`` 自建 / ``ephemeral`` 离线内存库 / ``collection_prefix``）。检索签名与
    工具契约逐字一致：``async def search(query, filters, top_k, effective_only) -> list[PolicyClauseHit]``。
    过滤语义（effective_only / category / risk_type）与既有实现逐条一致，差异只在
    「向量库 + 检索器 + 融合」。
    """

    _kind = "policy"

    def __init__(
        self,
        rows: Iterable[dict | PolicyClauseRecord],
        *,
        embedder: Embedder | None = None,
        mode: RetrievalMode = "hybrid",
        weights: tuple[float, float] = DEFAULT_WEIGHTS,
        chroma_client: Any | None = None,
        host: str = CHROMA_DEFAULT_HOST,
        port: int = CHROMA_DEFAULT_PORT,
        ephemeral: bool = False,
        collection_prefix: str | None = None,
    ) -> None:
        self._setup(
            rows,
            record_type=PolicyClauseRecord,
            text_of=_policy_text,
            embedder=embedder,
            mode=mode,
            weights=weights,
            chroma_client=chroma_client,
            host=host,
            port=port,
            ephemeral=ephemeral,
            collection_prefix=collection_prefix,
        )

    def effective_count(self) -> int:
        return sum(1 for r in self._rows if r.status == "EFFECTIVE")

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
            store_filters=_policy_store_filters(filters, self._llama),
            recheck=lambda i: i in set(candidates),
        )
        SERVED_COUNTERS["served_hits"] += len(ranked)
        return [
            PolicyClauseHit.model_validate(self._rows[i].model_dump(mode="json"))
            for i, _score in ranked
        ]


class ChromaCaseIndex(_ChromaIndexBase):
    """``CaseIndex`` Protocol 的 Chroma + LlamaIndex 实现（``search`` 签名与工具契约一致）。

    ``CaseHit.retrieval_score`` 是**检索分，不是语义相似度**：``bm25`` = 候选集内 min-max
    归一化 BM25 分；``vector`` = ``1 − distance``；``hybrid`` = RRF 融合分（``Σ 1/(k+rank)``，
    ``k=60``，落在 ~(0, ``2/60 = 1/30``]）。取值恒 ⊂ ``[0,1]``。
    """

    _kind = "case"

    def __init__(
        self,
        rows: Iterable[dict | CasePrecedentRecord],
        *,
        embedder: Embedder | None = None,
        mode: RetrievalMode = "hybrid",
        weights: tuple[float, float] = DEFAULT_WEIGHTS,
        chroma_client: Any | None = None,
        host: str = CHROMA_DEFAULT_HOST,
        port: int = CHROMA_DEFAULT_PORT,
        ephemeral: bool = False,
        collection_prefix: str | None = None,
    ) -> None:
        self._setup(
            rows,
            record_type=CasePrecedentRecord,
            text_of=_case_text,
            embedder=embedder,
            mode=mode,
            weights=weights,
            chroma_client=chroma_client,
            host=host,
            port=port,
            ephemeral=ephemeral,
            collection_prefix=collection_prefix,
        )

    async def search(
        self, query: str, filters: CaseSearchFilters, top_k: int
    ) -> list[CaseHit]:

        candidates = _case_candidates(self._rows, filters)
        ranked = self._retrieve_ranked(
            query,
            candidates,
            top_k=top_k,
            store_filters=_case_store_filters(filters, self._llama),
            recheck=lambda i: i in set(candidates),
        )
        SERVED_COUNTERS["served_hits"] += len(ranked)
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
