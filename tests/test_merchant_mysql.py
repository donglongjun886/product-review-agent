"""MerchantTool 的 MySQL 数据源（``mysql_repo``）：真库集成（可跳过）+ 恒跑纯单测。

真库部分照抄 ``tests/test_infra_persist_smoke.py`` 的默认配置路径 + 1s socket 探测 ``skipif``，
**不 mock ``db._settings``、不注 ``_env_file``** —— 专测「.env / DATABASE_URL 配错就连不上」
这条链；用例自建自删（``PYTEST_MERCHANT_`` 前缀 + ``finally`` 清理），重复跑不污染开发库。

纯单测部分注入假 sessionmaker，覆盖 DB 行 → ``MerchantProfile`` 的边界（``violations_by_type``
为 SQL NULL/空对象均归一 ``{}``、无事件 → 空列表、``DATETIME(3)`` → ISO8601 ``Z`` 展示串），
并守护「默认装配路径仍是 InMemory、不连库」。
"""

from __future__ import annotations

import json
import socket
from datetime import datetime
from typing import Self
from urllib.parse import urlparse
from uuid import uuid4

import pytest
from helpers import make_case
from sqlalchemy import text

from pra.domain.models import Budget, ProductImage, ProductReviewCase
from pra.infra import persist_service as ps
from pra.infra.db import Settings, get_sessionmaker
from pra.tools import build_production_tools, build_tools
from pra.tools.base import ToolContext
from pra.tools.merchant.mysql_repo import (
    MerchantEventORM,
    MerchantORM,
    MySQLMerchantRepository,
    to_profile,
)
from pra.tools.merchant.tool import (
    _DEFAULT_MERCHANTS,
    MERCHANT_HISTORY_TYPE,
    MERCHANT_HISTORY_WEIGHT,
    InMemoryMerchantRepository,
    MerchantArgs,
    MerchantTool,
)

# 模块导入期绑定真身：``tests/conftest.py`` 的 autouse fixture 会把 ``pra.tools`` 上的这个名字
# 换成 InMemory 装配（测试不连库），此处留住原函数供真库集成用例显式换回去。
_REAL_BUILD_PRODUCTION_TOOLS = build_production_tools


def _inmemory_case_index():
    from pra.tools.case_search.tool import InMemoryCaseIndex

    return InMemoryCaseIndex()


def _inmemory_policy_index():
    from pra.tools.policy_search.tool import InMemoryPolicyIndex

    return InMemoryPolicyIndex()


# ---------------------------------------------------------------------------
# 替身：假 session / 假 sessionmaker（形状与 async_sessionmaker 一致，可 async with）
# ---------------------------------------------------------------------------


class _FakeResult:
    """够用即止：``scalar_one_or_none()`` 给首行，``scalars().all()`` 给子行列表。"""

    def __init__(self, *, first: object = None, rows: list | None = None) -> None:
        self._first = first
        self._rows = list(rows or [])

    def scalar_one_or_none(self) -> object:
        return self._first

    def scalars(self) -> Self:
        return self

    def all(self) -> list:
        return self._rows


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
        product_total=120,
        similar_product_count=23,
        removals=5,
        title_relisting_count=3,
        violations_total=2,
        violations_by_type={"IP_MIMIC": 1, "FALSE_CLAIM": 1},
        credit_score=62,
    )
    for key, value in overrides.items():
        setattr(row, key, value)
    return row


def _event_row(**overrides: object) -> MerchantEventORM:
    row = MerchantEventORM(
        event_id=1,
        merchant_id="M_TEST",
        event_type="改标题重上架",
        ts=datetime.fromisoformat("2024-09-01T10:00:00"),  # naive（DB DATETIME 口径）
        sort_order=1,
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
    profile = to_profile(_merchant_row(), [_event_row()])
    assert profile.merchant_id == "M_TEST"
    assert profile.product_total == 120
    assert profile.similar_product_count == 23
    assert profile.removals == 5
    assert profile.title_relisting_count == 3
    assert profile.violations.total == 2
    assert profile.violations.by_type == {"IP_MIMIC": 1, "FALSE_CLAIM": 1}
    assert profile.credit_score == 62
    assert [(e.event_type, e.ts) for e in profile.recent_events] == [
        ("改标题重上架", "2024-09-01T10:00:00Z")
    ]


def test_violations_by_type_null_or_empty_both_become_empty_dict():
    for raw in (None, {}):
        profile = to_profile(_merchant_row(violations_total=0, violations_by_type=raw), [])
        assert profile.violations.by_type == {}, f"raw={raw!r} 应归一为空 dict"
        assert profile.violations.total == 0


def test_no_events_yield_empty_list():
    assert to_profile(_merchant_row(), []).recent_events == []


def test_event_ts_is_formatted_as_iso8601_z():
    """``DATETIME(3)`` → 带 ``Z`` 的展示串；毫秒截断（与 InMemory 种子逐字一致）。"""
    row = _event_row(ts=datetime.fromisoformat("2024-09-01T10:00:00.123000"))
    assert to_profile(_merchant_row(), [row]).recent_events[0].ts == "2024-09-01T10:00:00Z"


def test_construction_does_not_touch_the_sessionmaker_provider():
    """构造期不得建 engine / 连库（连库延迟到首次调用）。"""
    calls: list[int] = []

    def _provider():
        calls.append(1)
        raise AssertionError("构造期不应调用 sessionmaker provider")

    MySQLMerchantRepository(sessionmaker_factory=_provider)
    assert calls == []


async def test_get_profile_assembles_profile_and_queries_events():
    repo, session = _repo_with_fake_sessions(
        [_FakeResult(first=_merchant_row()), _FakeResult(rows=[_event_row()])]
    )

    profile = await repo.get_profile("M_TEST", window_days=90)
    assert profile is not None and profile.credit_score == 62
    assert len(session.queries) == 2, "应查 1 次主表 + 1 次事件表"
    assert "merchant_event" in str(session.queries[1])


async def test_missing_merchant_returns_none_without_event_query():
    repo, session = _repo_with_fake_sessions([_FakeResult(first=None)])

    assert await repo.get_profile("M_NOPE", window_days=90) is None
    assert len(session.queries) == 1, "商家不存在时应短路，不再查事件表"


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


def test_default_merchant_tool_is_still_inmemory():
    """守护：默认装配路径不许连库（谁把默认改成真库，这里变红）。"""
    assert type(MerchantTool()._repo).__name__ == "InMemoryMerchantRepository"
    assert isinstance(MerchantTool()._repo, InMemoryMerchantRepository)
    assert type(build_tools()[3]._repo).__name__ == "InMemoryMerchantRepository"


def test_build_tools_uses_mysql_merchant_repo_only_when_explicitly_injected():
    repo, _ = _repo_with_fake_sessions([])
    assert build_tools()[3]._repo is not repo
    assert build_tools(merchant_repo=repo)[3]._repo is repo


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


async def _insert_merchant(merchant_id: str, seed: dict, events: list[dict]) -> None:
    """用**裸 SQL** 写入 M_5512 同构种子（独立 merchant_id）。

    绕开 ORM 写：这样列名/JSON/时间口径由手写 DDL 独立钉住，读路径才走 ORM ——
    DDL 与 ORM 谁漂移了都直接报错。
    """
    sm = get_sessionmaker()
    async with sm() as s:
        await s.execute(
            text(
                "insert into merchant (merchant_id, product_total, similar_product_count, "
                "removals, title_relisting_count, violations_total, violations_by_type, "
                "credit_score) values (:mid, :pt, :sp, :rm, :tr, :vt, :vbt, :cs)"
            ),
            {
                "mid": merchant_id,
                "pt": seed["product_total"],
                "sp": seed["similar_product_count"],
                "rm": seed["removals"],
                "tr": seed["title_relisting_count"],
                "vt": seed["violations"]["total"],
                "vbt": json.dumps(seed["violations"]["by_type"], ensure_ascii=False),
                "cs": seed["credit_score"],
            },
        )
        for order, event in enumerate(events, start=1):
            await s.execute(
                text(
                    "insert into merchant_event (merchant_id, event_type, ts, sort_order) "
                    "values (:mid, :et, :ts, :so)"
                ),
                {
                    "mid": merchant_id,
                    "et": event["event_type"],
                    "ts": event["ts"].replace("T", " ").replace("Z", ""),
                    "so": order,
                },
            )
        await s.commit()


async def _delete_merchant(merchant_id: str) -> None:
    sm = get_sessionmaker()
    async with sm() as s:
        for table in ("merchant_event", "merchant"):
            await s.execute(
                text(f"delete from {table} where merchant_id = :mid"), {"mid": merchant_id}
            )
        await s.commit()


@pytest.mark.skipif(
    not _mysql_reachable(), reason="MySQL 不可达（未起 mysql-dev 容器）→ 跳过真库集成"
)
async def test_mysql_merchant_repository_roundtrip_against_real_db():
    """MySQL → MerchantProfile → Evidence 全字段往返（种子取自 M_5512 的结构）。"""
    merchant_id = f"PYTEST_MERCHANT_{uuid4().hex[:8]}"
    seed = dict(_DEFAULT_MERCHANTS["M_5512"])
    events = list(seed["recent_events"])
    try:
        await _insert_merchant(merchant_id, seed, events)

        profile = await MySQLMerchantRepository().get_profile(merchant_id, window_days=90)
        assert profile is not None
        assert profile.merchant_id == merchant_id
        assert profile.product_total == seed["product_total"]
        assert profile.similar_product_count == seed["similar_product_count"]
        assert profile.removals == seed["removals"]
        assert profile.title_relisting_count == seed["title_relisting_count"]
        assert profile.violations.total == seed["violations"]["total"]
        assert profile.violations.by_type == seed["violations"]["by_type"]
        assert profile.credit_score == seed["credit_score"]
        assert [(e.event_type, e.ts) for e in profile.recent_events] == [
            (e["event_type"], e["ts"]) for e in events
        ], "事件必须按 sort_order 稳定读出，ts 格式与 InMemory 种子逐字一致"

        tool = MerchantTool(repo=MySQLMerchantRepository())
        res = await tool.call(MerchantArgs(merchant_id=merchant_id), _ctx())
        assert res.ok is True
        evs = tool.to_evidence(res)
        assert len(evs) == 1
        assert evs[0].type == MERCHANT_HISTORY_TYPE
        assert evs[0].ref_id == merchant_id
        assert evs[0].weight == MERCHANT_HISTORY_WEIGHT
        assert evs[0].source == "MerchantTool"

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

# 只在真库种子里的商家（``tool.py`` 的 _DEFAULT_MERCHANTS 只有 M_5512）—— 用它才能在证据层面
# 区分「读了真库」与「读了 InMemory 默认世界」。
_MYSQL_ONLY_MERCHANT = "M_8801"
# 默认 Mock 图像源认得的图（认不得的图产不出 IMAGE_SIMILARITY，脚本化 plan 就不会去调 MerchantTool）。
_KNOWN_IMAGE_URL = "https://cdn.example.com/products/P_88231/img1.jpg"


def _case_for_merchant(case_id: str) -> ProductReviewCase:
    """COMPLEX 案件（brand 空缺 → R-301），商家只有真库有，图片是 Mock 认得的那张。"""
    case = make_case(
        case_id=case_id, brand=None, product_id="P_77310", merchant_id=_MYSQL_ONLY_MERCHANT
    )
    return case.model_copy(
        update={
            "product": case.product.model_copy(
                update={"images": [ProductImage(url=_KNOWN_IMAGE_URL, source="主图")]}
            )
        }
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
    InMemory 默认世界后该商家不存在，同一位置不应有 MERCHANT_HISTORY —— 若有人把生产装配改回
    InMemory，前一半会立刻变红。
    """
    monkeypatch.setattr("pra.tools.build_production_tools", _REAL_BUILD_PRODUCTION_TOOLS)
    # 本用例只验证 MySQL 链路：把生产装配的 RAG 侧钉回 InMemory 种子（scripted plan 分支 3 会调
    # CaseSearch/PolicySearch，真实 RAG 在缺 rag extra / 模型缓存时会尝试联网下载模型而阻塞，
    # 与「读真库商家」这一被测目标无关）。
    monkeypatch.setattr("pra.tools._build_production_case_index", _inmemory_case_index)
    monkeypatch.setattr("pra.tools._build_production_policy_index", _inmemory_policy_index)
    tag = uuid4().hex[:8]
    mysql_case_id, memory_case_id = f"PYTEST_MH_MYSQL_{tag}", f"PYTEST_MH_MEM_{tag}"
    try:
        monkeypatch.setattr(ps, "_compiled_graph", None)
        out_mysql = await ps.process_review(_case_for_merchant(mysql_case_id))
        rows = await _merchant_history_rows(out_mysql["run_id"])
        assert len(rows) == 1, "生产路径应恰好落 1 条 MERCHANT_HISTORY"
        assert rows[0][0] == "MerchantTool"
        assert rows[0][1] == _MYSQL_ONLY_MERCHANT, "ref_id 必须是案件商家（真库读到的那个）"
        assert "credit=38" in rows[0][2], "画像必须来自真库行（M_8801 的 credit=38）"

        monkeypatch.setattr("pra.tools.build_production_tools", build_tools)
        monkeypatch.setattr(ps, "_compiled_graph", None)
        out_mem = await ps.process_review(_case_for_merchant(memory_case_id))
        assert await _merchant_history_rows(out_mem["run_id"]) == [], (
            "InMemory 默认世界没有该商家 → 不应有 MERCHANT_HISTORY（本断言是上面那条的回退反证）"
        )
    finally:
        await _drop_cases((mysql_case_id, memory_case_id))
