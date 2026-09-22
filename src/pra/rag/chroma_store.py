"""Chroma 连接与 collection / node 装配 —— ``ChromaConfig``、客户端、建库、corpus 行 → Node。

collection 名形状 ``<prefix or 'pra'>_<policy|case>_<dim>_v<schema>``（维度由实际编码出的向量决定；
末段是 metadata 形状版本 —— 形状一改必须换名字，``_open_collection`` 是「先 get 后 create」，
同名旧库会被原样复用，新过滤表达式将对旧 metadata 静默零命中）。
建库两条必须显式声明：``embedding_function=None``（否则启用默认 ONNX 嵌入函数并去下模型）、
``space="cosine"``（Chroma 缺省是 ``l2``；本仓按 cosine 声明 —— 当前 BGE 向量已归一化，
两空间对排名与分数的影响**未经实测对比**，此处不预设谁对）。
1 行 = 1 Node 不切分；node id = ``sha256(collection + 行键)`` → 写入按该 id 先删后加、重建覆盖。
写入的 metadata 由 ``ChromaVectorStore.add`` 生成 = 扁平业务键（供 ``where`` 过滤）
+ ``_node_content``（整份 node JSON，供取数时无损还原）。
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
#: schema 版本（collection 名末段；全名 ``<prefix or 'pra'>_<kind>_<dim>_v<schema>``）。
#: **metadata 键形状变更时必须递增** —— 否则服务端上已存在的同名 collection 会被
#: ``_open_collection`` 的 get 原样复用（新 ``where`` 一条都查不到、静默全空）。
#: v2 = ``risk_type`` 数组键改为 ``rt_<值>`` 整数键；v3 = 写入 ``_node_content``；
#: v4 = metadata 改由 ``ChromaVectorStore.add`` 生成（``_node_content`` 内的 text 被清空）。
_SCHEMA_VERSION = 4
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
    （``<prefix or 'pra'>_<kind>_<dim>_v<schema>``）。
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


def _open_collection(config: ChromaConfig, *, name: str) -> Any:
    """``get_or_create_collection``（``embedding_function=None`` + 显式 cosine 空间）。

    不关 ``embedding_function`` 会启用默认 ONNX 嵌入函数并去下模型；空间取 cosine（Chroma 缺省是 l2），
    两者对排名 / 分数的影响**未经实测对比**，此处只是显式声明本仓口径。
    ⚠️ 空间是 collection 级属性、**创建时定死**，且 ``get_collection`` 命中同名旧库会原样复用
    —— 换空间只能换 collection 名。
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
    """corpus 行 → (TextNode 列表, node id 列表)。**1 行 = 1 Node，不切分**。

    检索文本 = 正文（``text_of``：policy 为 ``title。text``、case 为 ``summary``），
    与向量路 embed 的文本同一份。

    **metadata 不进检索文本**：``excluded_embed_metadata_keys`` 设为全部 metadata 键，使
    ``get_content(metadata_mode=EMBED)`` 只返回正文 —— 这对 BM25 路
    **必需**（`bm25.py` 的检索器用 ``MetadataMode.EMBED`` 取索引文本），不排除就会把 ``case_id`` /
    ``category`` / ``decision`` / ``rt_*`` 的字面值索引进去，出现「按 metadata 字面值就能
    命中」的伪检索。排除设置随 ``_node_content`` JSON 往返存活；metadata 本身仍完整保留
    （``MetadataMode.ALL`` 仍可见）。
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
                # metadata 不进 EMBED 检索文本（ALL 仍保留供审计）。
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
