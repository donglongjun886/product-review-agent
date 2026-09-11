"""真 Chroma 服务端集成测试 —— 服务端不可达时整文件 skip。

内存库（``EphemeralClient``）与真服务端的差异不只是连接串：Qdrant ``point id`` 上界缺陷
就曾在进程内模式全绿、只在真 server 炸。本文件钉住真服务端路径：建库 → 全量 upsert →
服务端点数校验 → 检索与 local 同口径。覆盖点：

1. ``build_*_index(backend="chroma", chroma_host=…, chroma_port=…)`` 的 ``HttpClient`` 分支真连上；
2. 点数落在服务端（24 policy / 67 case）—— 用另开的 client 读 ``collection.count()``，防本地假象；
3. vector 模式与 local 同序同 id（组合含过滤），case 侧另断言 R-4 隔离；
4. 建库用 ``configuration={"hnsw": {"space": "cosine"}}``：服务端缺省 ``l2``，会让
   「相似度 = 1 − distance」静默失效 → 回读配置断言；
5. ``collection_prefix`` 带 uuid4 后缀 → 自建自删；``finally`` 中断言本前缀库已清空，
   且测试前就存在的别人的库一个都没少（服务端是共享单实例，绝不动非本次创建的 collection）。

服务端未起时整文件 skip；CI 只跑 ``uv sync --frozen``（不装 extra）→ 必然不执行，
CI 上真正跑得动的守护是 ``tests/test_rag_default_path_no_extra.py``。

起服务：``cd deploy/chroma && docker compose up -d``（仅绑 ``127.0.0.1:8001``，容器内 8000；
``/api/v1`` 已废弃返回 410，只用 ``/api/v2``）。URL 可用 ``PRA_CHROMA_URL`` 覆盖；
embedder 一律 ``MockHashEmbedder``（确定性、零模型下载）。
"""

from __future__ import annotations

import os
import socket
from urllib.parse import urlparse
from uuid import uuid4

import pytest

# 无 chromadb 整文件跳过（CI 只跑 uv sync --frozen，不装 rag extra → 正是这种情况）
chromadb = pytest.importorskip(
    "chromadb",
    reason="未安装 chromadb（CI 只跑 uv sync --frozen，不装 rag extra）→ 整文件跳过；装上后：uv sync --extra rag",
)

from pra.domain.models import RiskType
from pra.rag.chroma_backend import (
    ChromaCaseIndex,
    ChromaPolicyIndex,
    make_chroma_client,
    reset_served_counters,
    served_counters,
)
from pra.rag.corpus import load_cases, load_policies
from pra.rag.embedder import MockHashEmbedder
from pra.rag.factory import build_case_index, build_policy_index
from pra.rag.index import RagCaseIndex, RagPolicyIndex
from pra.tools.case_search.tool import CaseSearchFilters
from pra.tools.policy_search.tool import PolicySearchFilters

_CHROMA_URL = os.environ.get("PRA_CHROMA_URL", "http://127.0.0.1:8001")
_PARSED = urlparse(_CHROMA_URL)
_HOST = _PARSED.hostname or "127.0.0.1"
_PORT = _PARSED.port or 8001

POLICY_ROWS = load_policies()[0]
CASE_ROWS = load_cases()[0]

_QUERY = "外观高度模仿知名品牌"
_BAG_CATEGORY = "箱包/女包"
_SCORE_TOL = 1e-6


def _server_reachable() -> bool:
    """按 ``PRA_CHROMA_URL`` 做 1s socket 探测（不建 client、不发 HTTP 请求）。"""
    try:
        with socket.create_connection((_HOST, _PORT), timeout=1):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not _server_reachable(),
    reason=(
        f"Chroma 服务端不可达（{_CHROMA_URL}）→ 跳过真服务端集成测试；"
        "起服务：cd deploy/chroma && docker compose up -d（仅 127.0.0.1:8001）"
    ),
)


def _prefix(tag: str) -> str:
    # uuid 后缀 → 与遗留库/并发运行互不干扰
    return f"pytest_srv_{tag}_{uuid4().hex[:8]}"


def _names_with_prefix(client: object, prefix: str) -> list[str]:
    return sorted(
        c.name for c in client.list_collections() if c.name.startswith(prefix)  # type: ignore[attr-defined]
    )


def _all_names(client: object) -> set[str]:
    return {c.name for c in client.list_collections()}  # type: ignore[attr-defined]


def _delete_mine(client: object, prefix: str) -> None:
    """只删本测试创建的（带 uuid 前缀的）collection —— 共享服务端上**绝不**动别人的库。"""
    for name in _names_with_prefix(client, prefix):
        client.delete_collection(name)  # type: ignore[attr-defined]


async def test_chroma_policy_index_against_real_server() -> None:
    """① ``chroma_host``/``chroma_port`` 分支真连上；② 24 个 node 落在服务端；③ 向量空间回读
    cosine；④ vector 命中序/id 与 local 一致（组合含过滤，``risk_type`` 是修复前漏召回的那类）。"""
    prefix = _prefix("policy")
    # 独立连接（走被测的 make_chroma_client 装配路径）：读的是服务端真实状态
    reader = make_chroma_client(host=_HOST, port=_PORT)
    before = _all_names(reader)
    try:
        remote = build_policy_index(
            backend="chroma",
            embedder=MockHashEmbedder(),
            mode="vector",
            chroma_host=_HOST,
            chroma_port=_PORT,
            collection_prefix=prefix,
        )
        local = RagPolicyIndex(POLICY_ROWS, embedder=MockHashEmbedder(), mode="vector")
        assert isinstance(remote, ChromaPolicyIndex), "factory backend='chroma' 应给出 ChromaPolicyIndex"

        cols = _names_with_prefix(reader, prefix)
        assert cols == [f"{prefix}_policy_256"], f"应恰好建 1 个 collection，实际 {cols}"
        remote_col = reader.get_collection(cols[0])
        assert remote_col.count() == len(POLICY_ROWS) == 24, "服务端点数必须 == corpus 行数"
        assert remote_col.configuration_json["hnsw"]["space"] == "cosine", (
            "服务端落库空间必须显式 cosine（默认 l2 会让 1−distance 静默失效）"
        )
        assert remote_col.metadata["pra_dim"] == 256

        for query, filters, effective_only, top_k in (
            (_QUERY, PolicySearchFilters(), False, 5),
            (_QUERY, PolicySearchFilters(), True, 5),
            # 修复前的漏召回组合（real server 版对照；当时 local 3 / chroma 0 类情形）
            ("外观模仿", PolicySearchFilters(risk_type=[RiskType.FALSE_CLAIM]), True, 6),
            ("外观模仿", PolicySearchFilters(risk_type=[RiskType.POTENTIAL_IP_RISK]), True, 6),
            ("外观模仿", PolicySearchFilters(category=_BAG_CATEGORY), True, 20),
        ):
            rh = await remote.search(query, filters, top_k, effective_only)
            lh = await local.search(query, filters, top_k, effective_only)
            assert [h.clause_id for h in rh] == [h.clause_id for h in lh], (
                f"真服务端 vector 模式应与 local 同序同 id："
                f"q={query!r} filters={filters} eff={effective_only} k={top_k}"
            )
        assert served_counters()["vector_bruteforce_fallbacks"] == 0, (
            "真服务端上向量路动用了兜底补算 —— 精确 id 取数 / 覆盖率自检失效"
        )
    finally:
        _delete_mine(reader, prefix)
    # 清理干净 + 别人的库一个都没少（共享单实例，绝不误删遗留 collection）
    assert _names_with_prefix(reader, prefix) == []
    assert before - _all_names(reader) == set(), "测试不得删除任何非本次创建的 collection"


async def test_chroma_case_index_against_real_server() -> None:
    prefix = _prefix("case")
    reader = make_chroma_client(host=_HOST, port=_PORT)
    before = _all_names(reader)
    try:
        remote = build_case_index(
            backend="chroma",
            embedder=MockHashEmbedder(),
            mode="vector",
            chroma_host=_HOST,
            chroma_port=_PORT,
            collection_prefix=prefix,
        )
        local = RagCaseIndex(CASE_ROWS, embedder=MockHashEmbedder(), mode="vector")
        assert isinstance(remote, ChromaCaseIndex), "factory backend='chroma' 应给出 ChromaCaseIndex"

        cols = _names_with_prefix(reader, prefix)
        assert cols == [f"{prefix}_case_256"], f"应恰好建 1 个 collection，实际 {cols}"
        remote_col = reader.get_collection(cols[0])
        assert remote_col.count() == len(CASE_ROWS) == 67, "服务端点数必须 == corpus 行数"
        assert remote_col.configuration_json["hnsw"]["space"] == "cosine"

        reset_served_counters()
        for filters, top_k in (
            (CaseSearchFilters(category="女鞋/运动鞋"), 5),
            # 修复前的漏召回组合（candidate 25 条而库里匹配 67 条）
            (CaseSearchFilters(risk_type=[RiskType.POTENTIAL_IP_RISK]), 30),
            (CaseSearchFilters(category="女鞋/运动鞋",
                               risk_type=[RiskType.POTENTIAL_IP_RISK]), 10),
        ):
            rh = await remote.search(_QUERY if top_k == 5 else "外观模仿", filters, top_k)
            lh = await local.search(_QUERY if top_k == 5 else "外观模仿", filters, top_k)
            assert rh, "真服务端检索不应为空"
            assert [h.case_id for h in rh] == [h.case_id for h in lh], (
                f"case 真服务端应与 local 同序同 id：filters={filters} k={top_k}"
            )
            for a, b in zip(lh, rh):
                assert round(abs(a.retrieval_score - b.retrieval_score), 9) <= _SCORE_TOL
            assert all(str(h.case_id).startswith("RAG_CASE_") for h in rh), "R-4：Case KB 隔离"
        assert served_counters()["vector_bruteforce_fallbacks"] == 0, (
            "真服务端上向量路动用了兜底补算 —— 精确 id 取数 / 覆盖率自检失效"
        )
    finally:
        _delete_mine(reader, prefix)
    assert _names_with_prefix(reader, prefix) == []
    assert before - _all_names(reader) == set(), "测试不得删除任何非本次创建的 collection"


async def test_chroma_server_reuses_same_prefix_collection_idempotently() -> None:
    """同前缀再次构造 → 复用服务端 collection（幂等 upsert，点不翻倍）。"""
    prefix = _prefix("reuse")
    reader = make_chroma_client(host=_HOST, port=_PORT)
    before = _all_names(reader)
    try:
        for _ in range(2):
            build_case_index(
                backend="chroma",
                embedder=MockHashEmbedder(),
                mode="bm25",
                chroma_host=_HOST,
                chroma_port=_PORT,
                collection_prefix=prefix,
            )
        cols = _names_with_prefix(reader, prefix)
        assert cols == [f"{prefix}_case_256"], f"复用不得新建第二个库，实际 {cols}"
        assert reader.get_collection(cols[0]).count() == len(CASE_ROWS)
    finally:
        _delete_mine(reader, prefix)
    assert _names_with_prefix(reader, prefix) == []
    assert before - _all_names(reader) == set(), "测试不得删除任何非本次创建的 collection"
