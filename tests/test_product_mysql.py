"""ProductTool 的 MySQL 数据源（``mysql_repo``）：真库集成（可跳过）+ 恒跑纯单测。"""

from __future__ import annotations

from typing import Self
from uuid import uuid4

import pytest
from helpers import (
    AlwaysRaiseBackend,
    WalkthroughBackend,
    make_case,
    mysql_reachable,
    tool_by_name,
)
from inmemory_world import (
    _DEFAULT_PRODUCTS,
    InMemoryMerchantRepository,
    InMemoryProductRepository,
    build_inmemory_tools,
)
from sqlalchemy import text

from pra import wiring
from pra.domain.measurement import (
    DEFAULT_EVIDENCE_WEIGHT,
    DIM_LISTING_REGISTRY,
    MEASUREMENT_TYPE,
    VERDICT_NEGATIVE,
)
from pra.domain.models import Budget, ProductReviewCase
from pra.infra import persist_service as ps
from pra.infra.db import get_sessionmaker
from pra.tools import build_production_tools
from pra.tools.base import ToolContext
from pra.tools.product.mysql_repo import (
    MySQLProductRepository,
    ProductORM,
    to_snapshot,
)
from pra.tools.product.tool import (
    PRODUCT_FACT_TYPE,
    ProductArgs,
    ProductResult,
    ProductTool,
)

_REAL_BUILD_PRODUCTION_TOOLS = build_production_tools


def _inmemory_case_index():
    from inmemory_world import InMemoryCaseIndex

    return InMemoryCaseIndex()


def _inmemory_policy_index():
    from inmemory_world import InMemoryPolicyIndex

    return InMemoryPolicyIndex()


class _FakeResult:
    """``session.execute()`` 的替身：``scalar_one_or_none()`` 走 ``scalar``。"""

    def __init__(self, *, scalar: object = None) -> None:
        self._scalar = scalar

    def scalar_one_or_none(self) -> object:
        return self._scalar


class _FakeSession:
    """``AsyncSession`` 替身：按调用顺序弹出预置结果；``error`` 非 None 时 execute 恒抛。"""

    def __init__(self, results: list, *, error: Exception | None = None) -> None:
        self._results = list(results)
        self._error = error
        self.queries: list[object] = []

    async def execute(self, statement: object) -> _FakeResult:
        self.queries.append(statement)
        if self._error is not None:
            raise self._error
        return self._results.pop(0)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False


class _FakeSessionmaker:
    """``async_sessionmaker`` 替身：``() -> AsyncSession``；``sessions`` 记录开出的会话。"""

    def __init__(self, results: list, *, error: Exception | None = None) -> None:
        self._results = results
        self._error = error
        self.sessions: list[_FakeSession] = []

    def __call__(self) -> _FakeSession:
        session = _FakeSession(self._results, error=self._error)
        self.sessions.append(session)
        return session


def _repo_with_fake_sessions(results: list, *, error: Exception | None = None):
    """造一个注入了假 sessionmaker 的 repo（提供者形状与 ``get_sessionmaker`` 一致）。"""
    sessionmaker = _FakeSessionmaker(results, error=error)
    repo = MySQLProductRepository(sessionmaker_factory=lambda: sessionmaker)
    return repo, sessionmaker


def _product_row(**overrides: object) -> ProductORM:
    values: dict[str, object] = {
        "product_id": "P_TEST",
        "category": "女鞋/运动鞋",
        "brand": None,
        "version": 3,
        "status": "ON_SALE",
    }
    values.update(overrides)
    return ProductORM(**values)


def _ctx() -> ToolContext:
    return ToolContext(run_id="R_TEST", case_id="C_TEST", budget=Budget())


def test_to_snapshot_maps_every_field():
    snap = to_snapshot(_product_row(brand="潮动"))
    assert snap.product_id == "P_TEST"
    assert snap.category == "女鞋/运动鞋"
    assert snap.brand == "潮动"
    assert snap.version == 3
    assert snap.status == "ON_SALE"


def test_brand_sql_null_is_not_normalised_to_empty_string():
    """brand 真空缺必须原样是 ``None``（不是 ''、不是 'null'）。"""
    snap = to_snapshot(_product_row(brand=None))
    assert snap.brand is None
    assert snap.brand != ""
    assert snap.brand != "null"

    evs = ProductTool(repo=MySQLProductRepository()).to_evidence(ProductResult(product=snap))
    facts = [e for e in evs if e.type == PRODUCT_FACT_TYPE]
    assert len(facts) == 1
    assert "brand=null" in facts[0].value


def test_construction_does_not_touch_the_sessionmaker_provider():
    """构造期不建 engine / 不取 sessionmaker。"""
    calls: list[int] = []

    def provider():
        calls.append(1)
        return _FakeSessionmaker([])

    MySQLProductRepository(sessionmaker_factory=provider)
    assert calls == []


async def test_get_latest_assembles_snapshot():
    """命中商品时 snapshot 非空且字段来自库行。"""
    repo, _ = _repo_with_fake_sessions([_FakeResult(scalar=_product_row(brand="云步"))])
    snap = await repo.get_latest("P_TEST")
    assert snap is not None
    assert snap.product_id == "P_TEST"
    assert snap.brand == "云步"
    assert snap.status == "ON_SALE"


async def test_get_latest_missing_product_returns_none():
    repo, _ = _repo_with_fake_sessions([_FakeResult(scalar=None)])
    assert await repo.get_latest("P_NOPE") is None


async def test_infrastructure_error_propagates_and_is_never_swallowed_as_none():
    """连不上库 / SQL 报错必须抛出，不能伪装成「商品不存在」。"""
    repo, _ = _repo_with_fake_sessions([_FakeResult(scalar=None)], error=RuntimeError("db down"))
    with pytest.raises(RuntimeError, match="db down"):
        await repo.get_latest("P_TEST")


async def test_missing_product_flows_to_ok_false_and_no_evidence():
    repo, _ = _repo_with_fake_sessions([_FakeResult(scalar=None)])
    tool = ProductTool(repo=repo)
    res = await tool.call(ProductArgs(product_id="P_NOPE"), _ctx())
    assert res.ok is False
    assert res.product is None
    assert res.error is not None and "P_NOPE" in res.error
    assert tool.to_evidence(res) == []


def test_production_and_inmemory_world_product_sources_are_distinct():
    """生产装配的 ProductTool 读真库，``build_inmemory_tools()`` 的读进程内种子。"""
    prod = _REAL_BUILD_PRODUCTION_TOOLS()
    memory = build_inmemory_tools()
    assert isinstance(tool_by_name(prod, "ProductTool")._repo, MySQLProductRepository)
    assert isinstance(tool_by_name(memory, "ProductTool")._repo, InMemoryProductRepository)


async def _insert_seed(product_id: str, seed: dict) -> None:
    """用**裸 SQL** 写入 P_88231 同构种子（独立 product_id）。"""
    sm = get_sessionmaker()
    async with sm() as s:
        await s.execute(
            text(
                "insert into product (product_id, category, brand, version, status) values "
                "(:pid, :cat, :brand, :ver, :status)"
            ),
            {
                "pid": product_id,
                "cat": seed["category"],
                "brand": seed["brand"],
                "ver": seed["version"],
                "status": seed["status"],
            },
        )
        await s.commit()


async def _delete_seed(product_id: str) -> None:
    """自建自删（可重复运行）。"""
    sm = get_sessionmaker()
    async with sm() as s:
        await s.execute(
            text("delete from product where product_id = :pid"), {"pid": product_id}
        )
        await s.commit()


@pytest.mark.skipif(
    not mysql_reachable(), reason="MySQL 不可达（未起 mysql-dev 容器）→ 跳过真库集成"
)
async def test_mysql_product_repository_roundtrip_against_real_db():
    """MySQL → ProductSnapshot → Evidence 全字段往返（种子取自 InMemory 世界种子的 P_88231）。"""
    tag = uuid4().hex[:8]
    product_id = f"PYTEST_PRODUCT_{tag}"
    missing_id = f"PYTEST_PRODUCT_MISSING_{tag}"
    seed = dict(_DEFAULT_PRODUCTS["P_88231"])
    try:
        await _insert_seed(product_id, seed)

        snap = await MySQLProductRepository().get_latest(product_id)
        assert snap is not None
        assert snap.product_id == product_id
        assert snap.category == seed["category"]
        assert snap.brand is None, "真空缺 brand 必须读成 None，不得是空串/'null'"
        assert snap.version == seed["version"]
        assert snap.status == seed["status"]

        tool = ProductTool(repo=MySQLProductRepository())

        assert await MySQLProductRepository().get_latest(missing_id) is None
        res_missing = await tool.call(ProductArgs(product_id=missing_id), _ctx())
        assert res_missing.ok is False
        assert res_missing.product is None

        res = await tool.call(ProductArgs(product_id=product_id), _ctx())
        assert res.ok is True
        evs = tool.to_evidence(res)
        facts = [e for e in evs if e.type == PRODUCT_FACT_TYPE]
        assert len(facts) == 1
        assert facts[0].ref_id == product_id
        assert facts[0].weight == DEFAULT_EVIDENCE_WEIGHT
        assert facts[0].source == "ProductTool"
        measurements = [e for e in evs if e.type == MEASUREMENT_TYPE]
        assert len(measurements) == 1
        assert measurements[0].extra["dimension"] == DIM_LISTING_REGISTRY
        assert measurements[0].extra["verdict"] == VERDICT_NEGATIVE
    finally:
        await _delete_seed(product_id)


def test_build_production_tools_swaps_only_the_mysql_backed_sources():
    """生产装配 = 4 件工具，只把 ProductTool / MerchantTool 的数据源换成真库（装配期不连库）。"""
    prod = _REAL_BUILD_PRODUCTION_TOOLS()
    memory = build_inmemory_tools()
    assert type(tool_by_name(prod, "ProductTool")._repo).__name__ == "MySQLProductRepository"
    assert type(tool_by_name(prod, "MerchantTool")._repo).__name__ == "MySQLMerchantRepository"
    assert [t.name for t in prod] == [t.name for t in memory]
    assert [type(t).__name__ for t in prod[1:]] == [type(t).__name__ for t in memory[1:]]


def test_tests_are_pinned_to_the_inmemory_world():
    """测试期生产装配必须被钉回 InMemory。"""
    import pra.tools as tools_pkg

    tools = tools_pkg.build_production_tools()
    assert isinstance(tool_by_name(tools, "ProductTool")._repo, InMemoryProductRepository), (
        "conftest 的 InMemory 钉回 fixture 失效了 —— 测试会去连真库"
    )
    assert isinstance(tool_by_name(tools, "MerchantTool")._repo, InMemoryMerchantRepository), (
        "conftest 的 InMemory 钉回 fixture 失效了 —— 测试会去连真库"
    )


def test_single_source_assembly_feeds_the_production_world(monkeypatch):
    """生产图只在组合根 ``pra.wiring.get_production_graph`` 组装一次。"""
    captured: dict = {}

    def fake_build_agent_graph(*, tools=None, checkpointer=None, llm=None):
        captured["tools"] = tools
        return object()

    sentinel = object()
    monkeypatch.setattr("pra.tools.build_production_tools", lambda: sentinel)
    monkeypatch.setattr(wiring, "build_llm_backend", lambda *, tools=None: AlwaysRaiseBackend())
    monkeypatch.setattr(wiring, "build_agent_graph", fake_build_agent_graph)
    monkeypatch.setattr(wiring, "_graph", None)

    wiring.get_production_graph()
    assert captured.pop("tools") is sentinel, "组合根未把生产工具世界注入图装配"

    assert not hasattr(ps, "_get_graph"), "落库编排又自建了装配入口"
    assert not hasattr(ps, "_compiled_graph"), "落库编排又自带了图单例"


_MYSQL_ONLY_PRODUCT = "P_77310"


def _case_for_product(case_id: str) -> ProductReviewCase:
    """COMPLEX 案件（brand 空缺 → R-301），商品只有真库有。"""
    return make_case(
        case_id=case_id, brand=None, product_id=_MYSQL_ONLY_PRODUCT, merchant_id="M_5512"
    )


async def _product_fact_rows(run_id: str) -> list:
    sm = get_sessionmaker()
    async with sm() as s:
        return (
            await s.execute(
                text(
                    "select type, source_tool, ref_id, value from review_evidence "
                    "where run_id = :r and type = 'PRODUCT_FACT'"
                ),
                {"r": run_id},
            )
        ).fetchall()


async def _drop_cases(case_ids: tuple[str, ...]) -> None:
    """按 case_id 清理本测试写入的五表行。"""
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
    not mysql_reachable(), reason="MySQL 不可达（未起 mysql-dev 容器）→ 跳过真库集成"
)
async def test_production_path_reads_product_fact_from_mysql(monkeypatch):
    """生产装配 → 落库入口 → 图 → ProductTool → MySQL → PRODUCT_FACT 落 review_evidence。"""
    monkeypatch.setattr("pra.tools.build_production_tools", _REAL_BUILD_PRODUCTION_TOOLS)
    monkeypatch.setattr(wiring, "build_llm_backend", lambda *, tools=None: WalkthroughBackend())
    monkeypatch.setattr("pra.tools._build_production_case_index", _inmemory_case_index)
    monkeypatch.setattr("pra.tools._build_production_policy_index", _inmemory_policy_index)
    tag = uuid4().hex[:8]
    mysql_case_id, memory_case_id = f"PYTEST_HTTP_MYSQL_{tag}", f"PYTEST_HTTP_MEM_{tag}"
    try:
        monkeypatch.setattr(wiring, "_graph", None)
        out_mysql = await ps.process_review(_case_for_product(mysql_case_id))
        facts = await _product_fact_rows(out_mysql["run_id"])
        assert len(facts) == 1, "生产路径应恰好落 1 条 PRODUCT_FACT"
        assert facts[0][1] == "ProductTool"
        assert facts[0][2] == _MYSQL_ONLY_PRODUCT, "ref_id 必须是案件商品（真库读到的那个）"
        assert "version=1" in facts[0][3], "版本必须来自真库行（P_77310 的 version=1）"

        monkeypatch.setattr("pra.tools.build_production_tools", build_inmemory_tools)
        monkeypatch.setattr(wiring, "_graph", None)
        out_mem = await ps.process_review(_case_for_product(memory_case_id))
        assert await _product_fact_rows(out_mem["run_id"]) == [], (
            "InMemory 世界没有该商品 → 不应有 PRODUCT_FACT（本断言是上面那条的回退反证）"
        )
    finally:
        await _drop_cases((mysql_case_id, memory_case_id))
