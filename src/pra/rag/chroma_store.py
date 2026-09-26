"""Chroma 连接与 collection / node 装配：``ChromaConfig``、客户端、建库、corpus 行 → Node。"""

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

#: 默认 Chroma 服务端地址。
CHROMA_DEFAULT_HOST = "127.0.0.1"
CHROMA_DEFAULT_PORT = 8001

_DEFAULT_COLLECTION_PREFIX = "pra"
#: schema 版本（collection 名末段）。
_SCHEMA_VERSION = 4
#: ``risk_type`` 过滤键前缀。
_RISK_TYPE_PREFIX = "rt_"


def risk_type_key(value: Any) -> str:
    """``risk_type`` 枚举（或字符串）→ metadata 过滤键 ``rt_<value>``。"""
    return f"{_RISK_TYPE_PREFIX}{getattr(value, 'value', value)}"


@dataclass(frozen=True)
class ChromaConfig:
    """Chroma 连接 + collection 装配参数。

    ``client`` 非 None 时优先用注入的客户端。
    """

    client: Any | None = None
    host: str = CHROMA_DEFAULT_HOST
    port: int = CHROMA_DEFAULT_PORT
    ephemeral: bool = False
    collection_prefix: str | None = None


def make_chroma_client(config: ChromaConfig | None = None) -> Any:
    """取用/自建 Chroma 客户端。"""
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
    """node id = ``sha256(collection + 行键)`` 的 hex 前缀。"""
    digest = hashlib.sha256(f"{collection}\x1f{key}".encode()).hexdigest()[:32]
    return f"pra-{digest}"


def _open_collection(config: ChromaConfig, *, name: str) -> Any:
    """取已有 collection，不存在则创建（``embedding_function=None`` + cosine）。"""
    client = make_chroma_client(config)
    try:
        return client.get_collection(name=name)
    except Exception as exc:
        if type(exc).__name__ != "NotFoundError":
            raise
    return client.get_or_create_collection(
        name=name,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
    )


def policy_node_metadata(row: PolicyClauseRecord) -> dict[str, Any]:
    """policy node metadata。"""
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
    """case node metadata。"""
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
    """corpus 行 → (TextNode 列表, node id 列表)；1 行 = 1 Node，检索文本 = 正文。"""
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
