"""真 Chroma 服务端集成测试（tests/test_rag_chroma_server.py）—— 服务端不可达时**整文件 skip**。

**为什么必须有它**：``tests/test_rag_chroma.py`` 只覆盖 ``EphemeralClient``（进程内内存库）。
内存库与真服务端（Rust 内核 + HttpClient）的差异**不是「仅连接串差异」**—— 本仓已吃过这类
亏（docs/06 §7：Qdrant ``point id`` 上界缺陷在进程内模式全绿、只在真 server 炸）。故这里把
真服务端路径钉住：**建库 → 全量 upsert → 服务端点数校验 → 检索与 local 同口径**。

覆盖点（docs/10 §5-8「真服务端」验收）：

1. ``build_*_index(backend="chroma", chroma_host=…, chroma_port=…)`` 的 ``HttpClient`` 分支真连上；
2. **点数落在服务端**（24 policy / 67 case）—— 用**另开的** client 读 ``collection.count()``，
   防「本地假象」；
3. **vector 模式与 local 同序同 id**（§5-1 第一行；组合选「过滤可完整下推」的那批，理由见
   docs/10 §3 与 ``tests/test_rag_chroma.py`` 第 6 节的实测偏差），case 侧另断言 R-4 隔离；
4. 用的是 ``configuration={"hnsw": {"space": "cosine"}}`` 建库（服务端回读配置断言）；
5. 清理：``collection_prefix`` 带 uuid4 后缀 → 测试**自建自删**，且在 ``finally`` 中断言
   「本前缀库已清空」+「测试前就存在的**别人的**库一个都没少」（服务端是共享单实例，
   遗留库不少 —— 绝不动非本次创建的 collection）。

跳过代价（诚实标注）：服务端未起时本文件**整文件 skip 而非失败**；CI 只跑
``uv sync --frozen``（不装任何 extra）→ 本文件在 CI 上必然不执行。CI 上真正跑得动的守护是
``tests/test_rag_default_path_no_extra.py``（docs/10 §5-5）。

服务端起法：``cd deploy/chroma && docker compose up -d``（仅绑 ``127.0.0.1:8001``，容器内 8000；
``/api/v1`` 已废弃返回 410，只用 ``/api/v2``）。URL 可用环境变量 ``PRA_CHROMA_URL`` 覆盖
（默认 ``http://127.0.0.1:8001``）。embedder 一律 ``MockHashEmbedder``（确定性、零模型下载）。
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
    """本次测试专用 collection 前缀（uuid 后缀 → 与遗留库/并发运行互不干扰）。"""
    return f"pytest_srv_{tag}_{uuid4().hex[:8]}"


def _names_with_prefix(client: object, prefix: str) -> list[str]:
    """服务端当前存在的、属于本测试前缀的 collection 名（用于清理与清理断言）。"""
    return sorted(
        c.name for c in client.list_collections() if c.name.startswith(prefix)  # type: ignore[attr-defined]
    )


def _all_names(client: object) -> set[str]:
    """服务端当前全部 collection 名（用于「不动别人的库」断言）。"""
    return {c.name for c in client.list_collections()}  # type: ignore[attr-defined]


def _delete_mine(client: object, prefix: str) -> None:
    """只删本测试创建的（带 uuid 前缀的）collection —— 共享服务端上**绝不**动别人的库。"""
    for name in _names_with_prefix(client, prefix):
        client.delete_collection(name)  # type: ignore[attr-defined]


async def test_chroma_policy_index_against_real_server() -> None:
    """policy KB 经真 Chroma 服务端建库 upsert + 检索与 local 同口径（vector 模式）。

    覆盖点：① factory 的 ``chroma_host``/``chroma_port`` 分支真连上服务端；② 24 个 node
    **落在服务端**（换一个 client 连接读 ``collection.count()``）；③ 服务端 collection 的
    向量空间回读为 cosine（docs/10 §3 硬要求）；④ vector 模式命中序/id 与 local 逐条一致
    —— 组合**含过滤**：无过滤、``effective_only=True``、``risk_type``（后者正是修复前会
    漏召回的那类，修复后由「精确候选 id 集 + 覆盖率自检」在真服务端同样保证完整）。
    """
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
                f"真服务端 vector 模式应与 local 同序同 id（§5-1 第一行）："
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
    """case KB 同上一路：真服务端 67 node 落库 + vector 同口径 + R-4 隔离。

    case 侧 category 走**精确匹配**下推（与 Python 谓词等价），故 category 组合上
    vector 的「同序同 id」在真服务端也应成立；``retrieval_score`` 允许 float32 尾差 ≤1e-6。
    """
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
                f"case 真服务端应与 local 同序同 id（§5-1 第一行）：filters={filters} k={top_k}"
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
    """同前缀再次构造 → **复用**服务端 collection（幂等 upsert，不产生重复点）。

    这条覆盖真服务端的 ``get_collection`` 复用路径（内存库上覆盖不到同一条 RPC 语义），
    并断言复用后点数仍 == corpus 行数（重复 upsert 不得翻倍）。
    """
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
