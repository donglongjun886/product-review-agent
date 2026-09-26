"""MerchantTool 的 MySQL 数据源（``mysql_repo``）：真库集成（可跳过）+ 恒跑纯单测。

真库部分照抄 ``tests/test_infra_persist_smoke.py`` 的默认配置路径 + 1s socket 探测 ``skipif``，
**不 mock ``db._settings``、不注 ``_env_file``** —— 专测「.env / DATABASE_URL 配错就连不上」
这条链；用例自建自删（``PYTEST_MERCHANT_`` 前缀 + ``finally`` 清理），重复跑不污染开发库。

纯单测部分注入假 sessionmaker，覆盖 DB 行 → ``MerchantProfile`` 的全部边界，
并对照生产装配（MySQL）与 InMemory 测试世界（``inmemory_world``）的数据源。
"""

from __future__ import annotations

import socket
from typing import Self
from urllib.parse import urlparse
from uuid import uuid4

import pytest
from helpers import WalkthroughBackend, make_case, tool_by_name
from inmemory_world import (
    _DEFAULT_MERCHANTS,
    InMemoryMerchantRepository,
    build_inmemory_tools,
)
from sqlalchemy import text

from pra import wiring
from pra.domain.measurement import (
    DIM_MERCHANT_PROFILE,
    MEASUREMENT_TYPE,
    VERDICT_NEGATIVE,
    VERDICT_POSITIVE,
)
from pra.domain.models import Budget, ProductReviewCase
from pra.infra import persist_service as ps
from pra.infra.db import Settings, get_sessionmaker
from pra.tools import build_production_tools
from pra.tools.base import ToolContext
from pra.tools.merchant.mysql_repo import (
    MerchantORM,
    MySQLMerchantRepository,
    to_profile,
)
from pra.tools.merchant.tool import (
    MERCHANT_HISTORY_TYPE,
    MERCHANT_HISTORY_WEIGHT,
    MerchantArgs,
    MerchantTool,
)

# 模块导入期绑定真身：``tests/conftest.py`` 的 autouse fixture 会把 ``pra.tools`` 上的这个名字
# 换成 InMemory 装配（测试不连库），此处留住原函数供真库集成用例显式换回去。
_REAL_BUILD_PRODUCTION_TOOLS = build_production_tools


def _inmemory_case_index():
    from inmemory_world import InMemoryCaseIndex

    return InMemoryCaseIndex()


def _inmemory_policy_index():
    from inmemory_world import InMemoryPolicyIndex

    return InMemoryPolicyIndex()


# ---------------------------------------------------------------------------
# 替身：假 session / 假 sessionmaker（形状与 async_sessionmaker 一致，可 async with）
# ---------------------------------------------------------------------------


class _FakeResult:
    """够用即止：``scalar_one_or_none()`` 给首行。"""

    def __init__(self, *, first: object = None) -> None:
        self._first = first

    def scalar_one_or_none(self) -> object:
        return self._first


class _FakeSession:
    def __init__(self, results: list[_FakeResult]) -> None:
        self._results = list(results)
        self.queries: list = []

    async def execute(self, stmt: object) -> _FakeResult:
        self.queries.append(stmt)
        return self._results.pop(0)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _FakeSessionmaker:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    def __call__(self) -> _FakeSession:
        return self._session


def _repo_with_fake_sessions(results: list[_FakeResult], *, error: Exception | None = None):
    session = _FakeSession(results)

    if error is not None:
        async def _raise(stmt: object) -> _FakeResult:
            session.queries.append(stmt)
            raise error

        session.execute = _raise  # type: ignore[method-assign]
    return MySQLMerchantRepository(sessionmaker_factory=lambda: _FakeSessionmaker(session)), session


def _merchant_row(**overrides: object) -> MerchantORM:
    row = MerchantORM(
        merchant_id="M_TEST",
        similar_product_count=23,
        removals=5,
        title_relisting_count=3,
        credit_score=62,
    )
    for key, value in overrides.items():
        setattr(row, key, value)
    return row


def _ctx() -> ToolContext:
    return ToolContext(run_id="RUN_T", case_id="CASE_T", budget=Budget())


# ---------------------------------------------------------------------------
# 纯单测：行 → 画像（不连库）
# ---------------------------------------------------------------------------


def test_to_profile_maps_every_field():
    profile = to_profile(_merchant_row())
    assert profile.merchant_id == "M_TEST"
    assert profile.similar_product_count == 23
    assert profile.removals == 5
    assert profile.title_relisting_count == 3
    assert profile.credit_score == 62


def test_construction_does_not_touch_the_sessionmaker_provider():
    """构造期不得建 engine / 连库（连库延迟到首次调用）。"""
    calls: list[int] = []

    def _provider():
        calls.append(1)
        raise AssertionError("构造期不应调用 sessionmaker provider")

    MySQLMerchantRepository(sessionmaker_factory=_provider)
    assert calls == []


async def test_get_profile_loads_profile():
    """行为不变量：命中商家时画像非空且字段来自库行。"""
    repo, _ = _repo_with_fake_sessions([_FakeResult(first=_merchant_row())])

    profile = await repo.get_profile("M_TEST", window_days=90)
    assert profile is not None and profile.credit_score == 62
    assert profile.removals == 5


async def test_missing_merchant_returns_none():
    repo, _ = _repo_with_fake_sessions([_FakeResult(first=None)])

    assert await repo.get_profile("M_NOPE", window_days=90) is None


async def test_infrastructure_error_propagates_and_is_never_swallowed_as_none():
    """连不上库/查询报错必须上抛 —— 吞成 None 会把「查不到」伪装成「证明无」。"""
    repo, _ = _repo_with_fake_sessions([], error=RuntimeError("connection refused"))

    with pytest.raises(RuntimeError, match="connection refused"):
        await repo.get_profile("M_TEST", window_days=90)


async def test_missing_merchant_flows_to_ok_false_and_no_evidence():
    repo, _ = _repo_with_fake_sessions([_FakeResult(first=None)])
    tool = MerchantTool(repo=repo)
    res = await tool.call(MerchantArgs(merchant_id="M_NOPE"), _ctx())
    assert res.ok is False
    assert res.profile is None
    assert tool.to_evidence(res) == []


def test_production_assembly_and_inmemory_world_have_distinct_data_sources():
    """数据源对照：生产装配的 MerchantTool 读 MySQL，InMemory 世界的读进程内种子。"""
    prod_repo = tool_by_name(_REAL_BUILD_PRODUCTION_TOOLS(), "MerchantTool")._repo
    memory_repo = tool_by_name(build_inmemory_tools(), "MerchantTool")._repo
    assert isinstance(prod_repo, MySQLMerchantRepository)
    assert isinstance(memory_repo, InMemoryMerchantRepository)


# ---------------------------------------------------------------------------
# 真库集成（MySQL 不可达时跳过）：默认配置路径，自建自删
# ---------------------------------------------------------------------------


def _mysql_reachable() -> bool:
    """按 **默认配置路径** 解析 DSN 并做 1s socket 探测（不连库、不建 engine）。

    ``Settings()`` 抛错时**不吞异常**：配置层坏掉就该让本文件变红（收集期报错），
    而不是伪装成「环境没 DB」跳过。
    """
    parsed = urlparse(Settings().database_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 3306
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


async def _insert_merchant(merchant_id: str, seed: dict) -> None:
    """用**裸 SQL** 写入 M_5512 同构种子（独立 merchant_id）。

    绕开 ORM 写：这样列名口径由手写 DDL 独立钉住，读路径才走 ORM ——
    DDL 与 ORM 谁漂移了都直接报错。
    """
    sm = get_sessionmaker()
    async with sm() as s:
        await s.execute(
            text(
                "insert into merchant (merchant_id, similar_product_count, removals, "
                "title_relisting_count, credit_score) values (:mid, :sp, :rm, :tr, :cs)"
            ),
            {
                "mid": merchant_id,
                "sp": seed["similar_product_count"],
                "rm": seed["removals"],
                "tr": seed["title_relisting_count"],
                "cs": seed["credit_score"],
            },
        )
        await s.commit()


async def _delete_merchant(merchant_id: str) -> None:
    sm = get_sessionmaker()
    async with sm() as s:
        await s.execute(
            text("delete from merchant where merchant_id = :mid"), {"mid": merchant_id}
        )
        await s.commit()


@pytest.mark.skipif(
    not _mysql_reachable(), reason="MySQL 不可达（未起 mysql-dev 容器）→ 跳过真库集成"
)
async def test_mysql_merchant_repository_roundtrip_against_real_db():
    """MySQL → MerchantProfile → Evidence 全字段往返（种子取自 M_5512 的结构）。"""
    merchant_id = f"PYTEST_MERCHANT_{uuid4().hex[:8]}"
    seed = dict(_DEFAULT_MERCHANTS["M_5512"])
    try:
        await _insert_merchant(merchant_id, seed)

        profile = await MySQLMerchantRepository().get_profile(merchant_id, window_days=90)
        assert profile is not None
        assert profile.merchant_id == merchant_id
        assert profile.similar_product_count == seed["similar_product_count"]
        assert profile.removals == seed["removals"]
        assert profile.title_relisting_count == seed["title_relisting_count"]
        assert profile.credit_score == seed["credit_score"]

        tool = MerchantTool(repo=MySQLMerchantRepository())
        res = await tool.call(MerchantArgs(merchant_id=merchant_id), _ctx())
        assert res.ok is True
        evs = tool.to_evidence(res)
        # 1 条聚合画像事实 + 1 条测量结论（含阳性/阴性判定）
        history = [e for e in evs if e.type == MERCHANT_HISTORY_TYPE]
        assert len(history) == 1
        assert history[0].ref_id == merchant_id
        assert history[0].weight == MERCHANT_HISTORY_WEIGHT
        assert history[0].source == "MerchantTool"
        measurements = [e for e in evs if e.type == MEASUREMENT_TYPE]
        assert len(measurements) == 1
        assert measurements[0].extra["dimension"] == DIM_MERCHANT_PROFILE
        assert measurements[0].extra["verdict"] in (VERDICT_NEGATIVE, VERDICT_POSITIVE)

        missing = await MySQLMerchantRepository().get_profile(
            f"{merchant_id}_NOPE", window_days=90
        )
        assert missing is None
        res_missing = await tool.call(MerchantArgs(merchant_id=f"{merchant_id}_NOPE"), _ctx())
        assert res_missing.ok is False and res_missing.profile is None
    finally:
        await _delete_merchant(merchant_id)


# ---------------------------------------------------------------------------
# 生产路径端到端（真库，可跳过）：HTTP/落库入口 → MySQL → Evidence
# ---------------------------------------------------------------------------

# 只在真库种子里的商家（``inmemory_world.py`` 的 _DEFAULT_MERCHANTS 只有 M_5512）—— 用它才能在
# 证据层面区分「读了真库」与「读了 InMemory 测试世界」。
_MYSQL_ONLY_MERCHANT = "M_8801"


def _case_for_merchant(case_id: str) -> ProductReviewCase:
    """COMPLEX 案件（brand 空缺 → R-301），商家只有真库有（与 InMemory 测试世界可区分）。"""
    return make_case(
        case_id=case_id, brand=None, product_id="P_77310", merchant_id=_MYSQL_ONLY_MERCHANT
    )


async def _merchant_history_rows(run_id: str) -> list:
    sm = get_sessionmaker()
    async with sm() as s:
        return (
            await s.execute(
                text(
                    "select source_tool, ref_id, value from review_evidence "
                    "where run_id = :r and type = 'MERCHANT_HISTORY'"
                ),
                {"r": run_id},
            )
        ).fetchall()


async def _drop_cases(case_ids: tuple[str, ...]) -> None:
    sm = get_sessionmaker()
    async with sm() as s:
        for cid in case_ids:
            await s.execute(
                text(
                    "delete from review_trace where run_id in "
                    "(select run_id from review_run where case_id = :c)"
                ),
                {"c": cid},
            )
            await s.execute(
                text(
                    "delete from review_evidence where run_id in "
                    "(select run_id from review_run where case_id = :c)"
                ),
                {"c": cid},
            )
            await s.execute(text("delete from review_result where case_id = :c"), {"c": cid})
            await s.execute(text("delete from review_run where case_id = :c"), {"c": cid})
            await s.execute(text("delete from review_case where case_id = :c"), {"c": cid})
        await s.commit()


@pytest.mark.skipif(
    not _mysql_reachable(), reason="MySQL 不可达（未起 mysql-dev 容器）→ 跳过真库集成"
)
async def test_production_path_reads_merchant_history_from_mysql(monkeypatch):
    """生产装配 → 落库入口 → 图 → MerchantTool → MySQL → MERCHANT_HISTORY 落 review_evidence。

    同一案件跑两遍做**回退反证**：生产装配（真库）应取到只存在于库中的商家并落证据；换回
    InMemory 测试世界后该商家不存在，同一位置不应有 MERCHANT_HISTORY —— 若有人把生产装配改回
    InMemory，前一半会立刻变红。
    """
    monkeypatch.setattr(
        wiring, "build_llm_backend", lambda *, tools=None: WalkthroughBackend()
    )
    monkeypatch.setattr("pra.tools.build_production_tools", _REAL_BUILD_PRODUCTION_TOOLS)
    # 本用例只验证 MySQL 链路，不需要真实 RAG 检索：把生产装配的案例/政策索引构建器钉回
    # InMemory 种子，免去真实 RAG 在缺 rag extra / 缺模型缓存时联网下载模型。
    monkeypatch.setattr("pra.tools._build_production_case_index", _inmemory_case_index)
    monkeypatch.setattr("pra.tools._build_production_policy_index", _inmemory_policy_index)
    tag = uuid4().hex[:8]
    mysql_case_id, memory_case_id = f"PYTEST_MH_MYSQL_{tag}", f"PYTEST_MH_MEM_{tag}"
    try:
        monkeypatch.setattr(wiring, "_graph", None)
        out_mysql = await ps.process_review(_case_for_merchant(mysql_case_id))
        rows = await _merchant_history_rows(out_mysql["run_id"])
        assert len(rows) == 1, "生产路径应恰好落 1 条 MERCHANT_HISTORY"
        assert rows[0][0] == "MerchantTool"
        assert rows[0][1] == _MYSQL_ONLY_MERCHANT, "ref_id 必须是案件商家（真库读到的那个）"
        assert "credit=38" in rows[0][2], "画像必须来自真库行（M_8801 的 credit=38）"

        monkeypatch.setattr("pra.tools.build_production_tools", build_inmemory_tools)
        monkeypatch.setattr(wiring, "_graph", None)
        out_mem = await ps.process_review(_case_for_merchant(memory_case_id))
        assert await _merchant_history_rows(out_mem["run_id"]) == [], (
            "InMemory 测试世界没有该商家 → 不应有 MERCHANT_HISTORY（本断言是上面那条的回退反证）"
        )
    finally:
        await _drop_cases((mysql_case_id, memory_case_id))
