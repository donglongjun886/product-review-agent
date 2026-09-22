"""Chroma 连接与 collection / node 装配：``ChromaConfig``、客户端、建库、corpus 行 → Node。

collection 名 = ``<prefix or 'pra'>_<kind>_<dim>_v<schema>``；metadata 形状一改必须递增 ``_SCHEMA_VERSION``
—— 同名旧 collection 被 ``_open_collection`` 原样复用，新 ``where`` 静默零命中。
必须显式 ``embedding_function=None``（否则启用默认 ONNX 嵌入函数并去下模型）+ ``space="cosine"``；
1 行 = 1 Node 不切分，node id = ``sha256(collection + 行键)``，写入按该 id 先删后加。
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from pra.rag.corpus.schema import CasePrecedentRecord, PolicyClauseRecord
from pra.rag.deps import chroma, llama

__all__ = [
    "CHROMA_DEFAULT_HOST",
    "CHROMA_DEFAULT_PORT",
    "ChromaConfig",
    "case_node_metadata",
    "make_chroma_client",
    "policy_node_metadata",
    "risk_type_key",
]

#: 默认 Chroma 服务端地址（deploy/chroma：宿主 8001 → 容器 8000；只用 `/api/v2`）。
CHROMA_DEFAULT_HOST = "127.0.0.1"
CHROMA_DEFAULT_PORT = 8001

_DEFAULT_COLLECTION_PREFIX = "pra"
#: schema 版本（collection 名末段）。**metadata 键形状变更时必须递增** —— 否则服务端同名旧
#: collection 会被 ``_open_collection`` 原样复用，新 ``where`` 一条都查不到、静默全空。
_SCHEMA_VERSION = 4
#: ``risk_type`` 过滤键前缀：每个枚举值一个整数键（``rt_FALSE_CLAIM: 1``），不用数组 —— Chroma
#: ``where`` 无「数组交叠」操作符，值取 ``1`` 而非 ``True``（``MetadataFilter.value`` 不收 ``bool``）。
_RISK_TYPE_PREFIX = "rt_"


def risk_type_key(value: Any) -> str:
    """``risk_type`` 枚举（或字符串）→ metadata 过滤键 ``rt_<value>``。

    键名必须与写入侧同源，否则 Chroma 对不存在的键不报错、**静默零命中**。
    """
    return f"{_RISK_TYPE_PREFIX}{getattr(value, 'value', value)}"


@dataclass(frozen=True)
class ChromaConfig:
    """Chroma 连接 + collection 装配参数（一处构造，逐层复用同一实例）。

    ``client`` 非 None 即用注入客户端；否则按 ``ephemeral`` 建进程内 ``EphemeralClient``，
    或按 ``host`` / ``port`` 建 ``HttpClient``。
    """

    client: Any | None = None
    host: str = CHROMA_DEFAULT_HOST
    port: int = CHROMA_DEFAULT_PORT
    ephemeral: bool = False
    collection_prefix: str | None = None


def make_chroma_client(config: ChromaConfig | None = None) -> Any:
    """取用/自建 Chroma 客户端（``config.client`` 注入优先）。

    ``ephemeral=True`` → 进程内 ``EphemeralClient``；否则 ``HttpClient(host, port)``（构造不联网）。
    """
    cfg = config or ChromaConfig()
    if cfg.client is not None:
        return cfg.client
    chromadb = chroma()
    if cfg.ephemeral:
        return chromadb.EphemeralClient()
    from chromadb.config import Settings as ChromaSettings

    return chromadb.HttpClient(
        host=cfg.host, port=cfg.port, settings=ChromaSettings(anonymized_telemetry=False)
    )


def _collection_name(prefix: str | None, kind: str, dim: int) -> str:
    return f"{prefix or _DEFAULT_COLLECTION_PREFIX}_{kind}_{dim}_v{_SCHEMA_VERSION}"


def _node_id(collection: str, key: str) -> str:
    """node id = ``sha256(collection + 行键)`` 前 16 字节 hex（稳定、幂等覆盖；collection 名参与哈希防撞车）。"""
    digest = hashlib.sha256(f"{collection}\x1f{key}".encode()).hexdigest()[:32]
    return f"pra-{digest}"


def _open_collection(config: ChromaConfig, *, name: str) -> Any:
    """``get_collection`` 命中即原样复用，否则 ``get_or_create_collection``（``embedding_function=None`` + cosine）。

    空间是 collection 级属性、创建时定死 —— 换空间只能换 collection 名。
    """
    client = make_chroma_client(config)
    try:
        return client.get_collection(name=name)
    except Exception as exc:
        if type(exc).__name__ != "NotFoundError":
            raise
    return client.get_or_create_collection(
        name=name,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},  # ★ 显式 cosine：默认是 l2
    )


def policy_node_metadata(row: PolicyClauseRecord) -> dict[str, Any]:
    """policy node metadata（不进检索文本）；``risk_type`` 写成 ``rt_<值>: 1`` 整数键。"""
    meta: dict[str, Any] = {
        "clause_id": row.clause_id,
        "policy_id": row.policy_id,
        "version": int(row.version),
        "category": row.category,
        "status": row.status,
        "effective_date": (
            row.effective_date.isoformat() if row.effective_date is not None else ""
        ),
    }
    for risk_type in row.risk_type:
        meta[risk_type_key(risk_type)] = 1
    return meta


def case_node_metadata(row: CasePrecedentRecord) -> dict[str, Any]:
    """case node metadata（``risk_type`` 同 :func:`policy_node_metadata`）。"""
    meta: dict[str, Any] = {
        "case_id": row.case_id,
        "category": row.category,
        "decision": row.decision.value,
        "risk_level": row.risk_level.value,
    }
    for risk_type in row.risk_type:
        meta[risk_type_key(risk_type)] = 1
    return meta


def _build_nodes(
    rows: list[Any],
    *,
    collection: str,
    key_of: Callable[[Any], str],
    text_of: Callable[[Any], str],
    meta_of: Callable[[Any], dict[str, Any]],
) -> tuple[list[Any], list[str]]:
    """corpus 行 → (TextNode 列表, node id 列表)。**1 行 = 1 Node，不切分**；检索文本 = 正文。

    ``excluded_embed_metadata_keys`` 设为全部 metadata 键：不排除就会把 ``case_id`` / ``rt_*`` 等
    字面值索引进 BM25 文本，出现「按 metadata 字面值就能命中」的伪检索。
    """
    llama_ = llama()
    nodes: list[Any] = []
    node_ids: list[str] = []
    for row in rows:
        nid = _node_id(collection, key_of(row))
        node_ids.append(nid)
        meta = meta_of(row)
        text = text_of(row)
        nodes.append(
            llama_.TextNode(
                id_=nid,
                text=text,
                metadata=meta,
                # metadata 不进 EMBED 检索文本（ALL 模式仍可见）。
                excluded_embed_metadata_keys=list(meta.keys()),
            )
        )
    return nodes, node_ids


def _normalize_rows(rows: Iterable[Any], record_type: type) -> list[Any]:
    out: list[Any] = []
    for row in rows:
        if not isinstance(row, record_type):
            raise TypeError(f"corpus 行须为 {record_type.__name__}: got {type(row)!r}")
        out.append(row)
    return out
