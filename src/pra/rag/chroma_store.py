"""Chroma 连接与 collection / node 装配 —— ``ChromaConfig``、客户端、建库、corpus 行 → Node。

collection 名形状 ``<prefix or 'pra'>_<policy|case>_<dim>_v<schema>``（维度由实际编码出的向量决定；
末段是 metadata 形状版本 —— 形状一改必须换名字，``_open_collection`` 是「先 get 后 create」，
同名旧库会被原样复用，新过滤表达式将对旧 metadata 静默零命中）。
建库两条缺一即静默出错：``embedding_function=None``（否则启用默认 ONNX 嵌入函数并去下模型）、
``space="cosine"``（Chroma 缺省 ``l2``，让「相似度 = 1 − distance」失效）。
1 行 = 1 Node 不切分；node id = ``sha256(collection + 行键)`` → 重建幂等覆盖。
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
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
#: metadata 形状版本（collection 名末段）。**metadata 键形状变更时必须递增** —— 否则服务端上
#: 已存在的同名 collection 会被 ``_open_collection`` 的 get 原样复用（新 ``where`` 一条都查不到、
#: 静默全空）。
_SCHEMA_VERSION = 2
#: metadata 维度键（Chroma 不声明向量维度，故把本索引声明的 dim 存进 collection metadata）。
_DIM_KEY = "pra_dim"
#: ``risk_type`` 过滤键前缀：**每个枚举值一个整数键**（``rt_FALSE_CLAIM: 1``），不用数组。
#: Chroma ``where`` 只支持标量比较、没有「数组交叠」操作符，而过滤语义恰恰是**交叠非空**。
#: 值取 ``1`` 而非 ``True``：``MetadataFilter.value`` 的 pydantic 联合类型不收 ``bool``。
_RISK_TYPE_PREFIX = "rt_"


def risk_type_key(value: Any) -> str:
    """``risk_type`` 枚举（或字符串）→ metadata 过滤键 ``rt_<value>``。

    写入（:func:`policy_node_metadata` / :func:`case_node_metadata`）与查询（``where`` 构造）
    必须同源，否则键名不一致会**静默零命中**（Chroma 对不存在的键不报错）。
    """
    return f"{_RISK_TYPE_PREFIX}{getattr(value, 'value', value)}"


@dataclass(frozen=True)
class ChromaConfig:
    """Chroma 连接 + collection 装配参数（一处构造，逐层复用同一实例）。

    ``client`` 非 None 即用注入的客户端；否则按 ``ephemeral`` 建进程内 ``EphemeralClient``
    或按 ``host`` / ``port`` 建 ``HttpClient``。``collection_prefix`` 参与 collection 名
    （``<prefix or 'pra'>_<policy|case>_<dim>``）与 collection metadata。
    """

    client: Any | None = None
    host: str = CHROMA_DEFAULT_HOST
    port: int = CHROMA_DEFAULT_PORT
    ephemeral: bool = False
    collection_prefix: str | None = None


def make_chroma_client(config: ChromaConfig | None = None) -> Any:
    """取用/自建 Chroma 客户端（``config.client`` 注入优先）。

    ``ephemeral=True`` → 进程内 ``EphemeralClient``；否则 ``HttpClient(host, port)``。
    **构造不联网**（``HttpClient`` 只做参数装配，首个请求才连接）；匿名遥测显式关闭。
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
    """node id = ``sha256(collection + 行键)`` 前 16 字节 hex（稳定、幂等覆盖）。

    collection 名参与哈希，避免 policy / case KB 前缀相同时 node id 撞车。
    """
    digest = hashlib.sha256(f"{collection}\x1f{key}".encode()).hexdigest()[:32]
    return f"pra-{digest}"


def _open_collection(config: ChromaConfig, *, name: str, dim: int, kind: str) -> Any:
    """``get_or_create_collection``（**必须 ``embedding_function=None`` + 显式 cosine 空间**）。

    不关 ``embedding_function`` 会启用默认 ONNX 嵌入函数并去下模型；不显式 cosine 则空间是
    Chroma 缺省的 l2 —— 而「相似度 = 1 − distance」这套口径只在 cosine 空间成立。
    """
    chroma()  # 缺包早失败（提示装 extra）；客户端由 config 解析（注入优先）
    client = make_chroma_client(config)
    try:
        return client.get_collection(name=name)
    except Exception as exc:
        if "not found" not in str(exc).lower() and type(exc).__name__ != "NotFoundError":
            raise
    return client.get_or_create_collection(
        name=name,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},  # ★ 显式 cosine：默认是 l2
        metadata={
            _DIM_KEY: int(dim),
            "pra_kind": kind,
            "pra_prefix": config.collection_prefix or _DEFAULT_COLLECTION_PREFIX,
        },
    )


def _iso_or_empty(value: date | None) -> str:
    return value.isoformat() if value is not None else ""


def policy_node_metadata(row: PolicyClauseRecord) -> dict[str, Any]:
    """policy node metadata（**不进检索文本**，见 :func:`_build_nodes`）。

    ``risk_type`` 写成 ``rt_<值>: 1`` 整数键（供 Chroma ``where`` 表达「交叠非空」），
    见 :func:`risk_type_key`。
    """
    meta: dict[str, Any] = {
        "clause_id": row.clause_id,
        "policy_id": row.policy_id,
        "version": int(row.version),
        "category": row.category,
        "status": row.status,
        "effective_date": _iso_or_empty(row.effective_date),
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
    """corpus 行 → (TextNode 列表, node id 列表)。**1 行 = 1 Node，不切分**。

    检索文本 = 正文（``text_of``：policy 为 ``title。text``、case 为 ``summary``），
    与向量路 embed 的文本同一份。

    **metadata 不进检索文本**：``excluded_embed_metadata_keys`` / ``excluded_llm_metadata_keys``
    设为全部 metadata 键，使 ``get_content(metadata_mode=EMBED)`` 只返回正文 —— 这对 BM25 路
    **必需**（``BM25Retriever`` 用 ``MetadataMode.EMBED``），不排除就会把 ``case_id`` /
    ``category`` / ``decision`` / ``rt_*`` 的字面值索引进去，出现「按 metadata 字面值就能
    命中」的伪检索。排除设置随 ``_node_content`` JSON 往返存活；metadata 本身仍完整保留。
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
                # metadata 不进检索文本（EMBED/LLM 两个模式都排除；ALL 仍保留供审计）。
                excluded_embed_metadata_keys=list(meta.keys()),
                excluded_llm_metadata_keys=list(meta.keys()),
            )
        )
    return nodes, node_ids


def _normalize_rows(rows: Iterable[Any], record_type: type) -> list[Any]:
    out: list[Any] = []
    for row in rows:
        out.append(record_type.model_validate(row) if isinstance(row, dict) else row)
    for row in out:
        if not isinstance(row, record_type):
            raise TypeError(f"corpus 行须为 {record_type.__name__} 或 dict: got {type(row)!r}")
    return out
