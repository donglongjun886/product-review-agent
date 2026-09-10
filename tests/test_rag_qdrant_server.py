"""真 Qdrant 服务端（``url=`` 远端模式）集成测试 —— 服务端不可达时跳过。

**为什么必须有它**（2026-09-10 缺陷复盘）：docs/06 §5 曾把远端 server 标注为
「代码路径同一，**仅连接串差异**」，实测**证伪**。原因：``tests/test_rag_qdrant.py``
只覆盖 qdrant-client 的**进程内模式**（``:memory:`` / ``path=``），而**进程内模式不校验
point id 上界** —— 于是「``_point_id`` 产出 128 位 int」这个缺陷，在
``pytest 全绿`` 的假象下无人发现，直到真 server 以
``400 Bad Request: ... is not a valid point ID`` 拒绝 upsert。本文件把那条路径钉住：
**真 server 建库 + 全量 upsert + 检索**，并与 ``local`` 后端断言同口径。

跳过代价（诚实标注）：服务端未起时本测试**静默跳过而非失败** —— 但 id 取值域另有
``tests/test_rag_qdrant.py::test_point_id_fits_in_u64`` **离线兜底（恒跑）**，
不依赖本文件即可拦住该类回归（与 ``test_infra_persist_smoke.py`` × ``test_infra_settings.py``
的「真库冒烟 + 纯单测兜底」同一分工）。

服务端起法：``cd deploy/qdrant && docker compose up -d``（仅绑 ``127.0.0.1:6333``，
说明见同目录 README）。URL 可用环境变量 ``PRA_QDRANT_URL`` 覆盖（默认
``http://127.0.0.1:6333``）。

清理：``collection_prefix`` 带 ``uuid4`` 后缀 → 测试**自建自删**（``finally``），
可重复运行、与其它测试/人工残留互不冲突。embedder 一律 MockHashEmbedder
（确定性、零模型下载）。

语法/import 约定：顶部 ``from __future__ import annotations``；import 一律 pra.*。
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
    """按 ``PRA_QDRANT_URL`` 做 1s socket 探测（不建 client、不发请求）。"""
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
    """本次测试专用 collection 前缀（uuid 后缀 → 并发/残留互不干扰）。"""
    return f"pytest_srv_{uuid4().hex[:8]}"


def _collections_of(client: object, prefix: str) -> list[str]:
    return [c.name for c in client.get_collections().collections if c.name.startswith(prefix)]


def _delete_all(client: object, prefix: str) -> None:
    """删除本测试建的全部 collection（幂等；真 server 上不留残留）。"""
    for name in _collections_of(client, prefix):
        client.delete_collection(name)


async def test_qdrant_policy_index_against_real_server() -> None:
    """policy KB 经 ``location=<url>``（真 server）建库 upsert + 检索与 local 同口径。

    覆盖点：① ``_make_client`` 的 ``http(s)://`` 分支真的连上远端；② 全量点**落在服务端**
    （数量 == 语料行数，防「本地假象」）；③ 检索 id 序与 local 后端逐条一致
    （docs/06 §2.1 同构契约在**真 server** 上成立 —— 此前只在进程内模式验证过）。
    """
    prefix = _prefix()
    client = qdrant_client.QdrantClient(url=_QDRANT_URL)
    try:
        remote = build_policy_index(
            backend="qdrant", location=_QDRANT_URL, collection_prefix=prefix
        )
        local = build_policy_index()

        # ② 服务端确有 1 个本前缀 collection，且点数 == 语料行数
        cols = _collections_of(client, prefix)
        assert len(cols) == 1, f"应恰好建 1 个 collection，实际 {cols}"
        assert client.get_collection(cols[0]).points_count == len(POLICY_ROWS)

        # ③ 与 local 同口径（先断言非空，防「两边都空」式假通过）
        rh = await remote.search(_QUERY, PolicySearchFilters(), top_k=5, effective_only=True)
        lh = await local.search(_QUERY, PolicySearchFilters(), top_k=5, effective_only=True)
        assert rh, "真 server 检索不应为空"
        assert [h.clause_id for h in rh] == [h.clause_id for h in lh]
    finally:
        _delete_all(client, prefix)
    # 确认清理干净（不把残留留给下一次运行）
    assert _collections_of(client, prefix) == []


async def test_qdrant_case_index_against_real_server() -> None:
    """case KB 同上一路：真 server 全量落库 + 与 local 同口径。

    case 侧额外断言 R-4 隔离红线：命中的 ``case_id`` 均为 ``RAG_CASE_`` 前缀
    （Case KB 与评测 GT 数据集物理隔离，不因换了检索后端而变）。
    """
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
    """id 兼容性直测：把 corpus 全部 point id upsert 到真 server，不得 400。

    这是**最贴近原缺陷的**断言 —— 原实现（128 位 int）在此必然
    ``400 Bad Request: ... is not a valid point ID``。用最小维度 collection，
    只验 id 接受性、不验检索语义（语义由上面两个用例覆盖）。
    """
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
