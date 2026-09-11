"""真 Qdrant 服务端（``url=`` 远端模式）集成测试；服务端不可达时整文件跳过。

必须有它：``test_rag_qdrant.py`` 只覆盖 qdrant-client 的进程内模式（``:memory:`` /
``path=``），而进程内模式**不校验 point id 上界** —— 于是「``_point_id`` 产出 128 位
int」在 pytest 全绿下无人发现，直到真 server 以
``400 Bad Request: ... is not a valid point ID`` 拒绝 upsert。本文件用真 server
建库 + 全量 upsert + 检索，并与 ``local`` 后端断言同口径。

id 取值域另有 ``tests/test_rag_qdrant_point_id.py`` 兜底（刻意不依赖 qdrant-client，
任何环境恒跑）；本文件与 ``test_rag_qdrant.py`` 都 ``importorskip("qdrant_client")``，
而 CI 只跑 ``uv sync --frozen``（不装 extra）→ 两者在 CI 上都不执行。

起服务：``cd deploy/qdrant && docker compose up -d``（仅绑 ``127.0.0.1:6333``，
URL 可用 ``PRA_QDRANT_URL`` 覆盖）。collection 带 ``uuid4`` 后缀且 ``finally`` 自删，
可重复运行；embedder 一律 MockHashEmbedder（确定性、零模型下载）。
"""

from __future__ import annotations

import os
import socket
from urllib.parse import urlparse
from uuid import uuid4

import pytest

qdrant_client = pytest.importorskip("qdrant_client")  # 无 qdrant-client 整文件跳过

from pra.rag.corpus import load_cases, load_policies
from pra.rag.factory import build_case_index, build_policy_index
from pra.tools.case_search.tool import CaseSearchFilters
from pra.tools.policy_search.tool import PolicySearchFilters

_QDRANT_URL = os.environ.get("PRA_QDRANT_URL", "http://127.0.0.1:6333")
POLICY_ROWS = load_policies()[0]
CASE_ROWS = load_cases()[0]

_QUERY = "外观高度模仿知名品牌"


def _server_reachable() -> bool:
    parsed = urlparse(_QDRANT_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 6333
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _server_reachable(),
    reason=(
        f"Qdrant 服务端不可达（{_QDRANT_URL}）→ 跳过真服务端集成测试；"
        "起服务：cd deploy/qdrant && docker compose up -d"
    ),
)


def _prefix() -> str:
    return f"pytest_srv_{uuid4().hex[:8]}"


def _collections_of(client: object, prefix: str) -> list[str]:
    return [c.name for c in client.get_collections().collections if c.name.startswith(prefix)]


def _delete_all(client: object, prefix: str) -> None:
    for name in _collections_of(client, prefix):
        client.delete_collection(name)


async def test_qdrant_policy_index_against_real_server() -> None:
    prefix = _prefix()
    client = qdrant_client.QdrantClient(url=_QDRANT_URL)
    try:
        remote = build_policy_index(
            backend="qdrant", location=_QDRANT_URL, collection_prefix=prefix
        )
        local = build_policy_index()

        cols = _collections_of(client, prefix)
        assert len(cols) == 1, f"应恰好建 1 个 collection，实际 {cols}"
        assert client.get_collection(cols[0]).points_count == len(POLICY_ROWS)

        rh = await remote.search(_QUERY, PolicySearchFilters(), top_k=5, effective_only=True)
        lh = await local.search(_QUERY, PolicySearchFilters(), top_k=5, effective_only=True)
        assert rh, "真 server 检索不应为空"
        assert [h.clause_id for h in rh] == [h.clause_id for h in lh]
    finally:
        _delete_all(client, prefix)
    assert _collections_of(client, prefix) == []


async def test_qdrant_case_index_against_real_server() -> None:
    prefix = _prefix()
    client = qdrant_client.QdrantClient(url=_QDRANT_URL)
    try:
        remote = build_case_index(
            backend="qdrant", location=_QDRANT_URL, collection_prefix=prefix
        )
        local = build_case_index()

        cols = _collections_of(client, prefix)
        assert len(cols) == 1, f"应恰好建 1 个 collection，实际 {cols}"
        assert client.get_collection(cols[0]).points_count == len(CASE_ROWS)

        rh = await remote.search(_QUERY, CaseSearchFilters(), top_k=5)
        lh = await local.search(_QUERY, CaseSearchFilters(), top_k=5)
        assert rh, "真 server 检索不应为空"
        assert [h.case_id for h in rh] == [h.case_id for h in lh]
        assert all(h.case_id.startswith("RAG_CASE_") for h in rh), "R-4：Case KB 隔离"
    finally:
        _delete_all(client, prefix)
    assert _collections_of(client, prefix) == []


def test_point_id_is_accepted_by_real_server() -> None:
    from pra.rag.qdrant_index import _point_id

    prefix = f"{_prefix()}_idprobe"
    client = qdrant_client.QdrantClient(url=_QDRANT_URL)
    name = prefix
    try:
        client.create_collection(
            collection_name=name,
            vectors_config=qdrant_client.models.VectorParams(
                size=4, distance=qdrant_client.models.Distance.COSINE
            ),
        )
        ids = [_point_id(r.clause_id) for r in POLICY_ROWS] + [
            _point_id(r.case_id) for r in CASE_ROWS
        ]
        client.upsert(
            collection_name=name,
            points=[
                qdrant_client.models.PointStruct(id=pid, vector=[0.1, 0.2, 0.3, 0.4], payload={})
                for pid in ids
            ],
        )
        assert client.get_collection(name).points_count == len(ids)
    finally:
        _delete_all(client, prefix)
