"""ProductTool 的 MySQL 数据源（``mysql_repo``）：真库集成（可跳过）+ 恒跑纯单测。

真库部分照抄 ``tests/test_infra_persist_smoke.py`` 的默认配置路径 + 1s socket 探测 ``skipif``，
**不 mock ``db._settings``、不注 ``_env_file``** —— 专测「.env / DATABASE_URL 配错就连不上」
这条链；用例自建自删（``PYTEST_PRODUCT_`` 前缀 + ``finally`` 清理），重复跑不污染开发库。

纯单测部分注入假 sessionmaker，覆盖 DB 行 → ``ProductSnapshot`` 的全部边界（``brand`` 为
SQL NULL 不得归一成空串/``'null'``），并守护「默认装配路径仍是 InMemory、不连库」。
"""

from __future__ import annotations

import socket
from typing import Self
from urllib.parse import urlparse
from uuid import uuid4

import pytest
from helpers import make_case, tool_by_name
from sqlalchemy import text

from pra import wiring
from pra.domain.measurement import (
    DEFAULT_EVIDENCE_WEIGHT,
    DIM_LISTING_REGISTRY,
    MEASUREMENT_TYPE,
    VERDICT_NEGATIVE,
)
from pra.domain.models import Budget, ProductImage, ProductReviewCase
from pra.infra import persist_service as ps
from pra.infra.db import Settings, get_sessionmaker
from pra.tools import build_production_tools, build_tools
from pra.tools.base import ToolContext
from pra.tools.merchant.tool import InMemoryMerchantRepository
from pra.tools.product.mysql_repo import (
    MySQLProductRepository,
    ProductORM,
    to_snapshot,
)
from pra.tools.product.tool import (
    _DEFAULT_PRODUCTS,
    PRODUCT_FACT_TYPE,
    InMemoryProductRepository,
    ProductArgs,
    ProductResult,
    ProductTool,
)

# 模块导入期绑定真身：``tests/conftest.py`` 的 autouse fixture 会把 ``pra.tools`` 上的这个名字
# 换成 InMemory 装配（测试不连库），此处留住原函数供「生产装配」与真库集成用例显式换回去。
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


# ---------------------------------------------------------------------------
# 行构造（用真实 ORM 类：列名/类型漂移会被这里连带测到）
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# 纯单测：DB 行 → ProductSnapshot 映射边界
# ---------------------------------------------------------------------------


def test_to_snapshot_maps_every_field():
    snap = to_snapshot(_product_row(brand="潮动"))
    assert snap.product_id == "P_TEST"
    assert snap.category == "女鞋/运动鞋"
    assert snap.brand == "潮动"
    assert snap.version == 3
    assert snap.status == "ON_SALE"


def test_brand_sql_null_is_not_normalised_to_empty_string():
    """业务红线：brand 真空缺必须原样是 ``None``（不是 ''、不是 'null'）。"""
    snap = to_snapshot(_product_row(brand=None))
    assert snap.brand is None
    assert snap.brand != ""
    assert snap.brand != "null"

    evs = ProductTool().to_evidence(ProductResult(product=snap))
    facts = [e for e in evs if e.type == PRODUCT_FACT_TYPE]
    assert len(facts) == 1
    assert "brand=null" in facts[0].value


# ---------------------------------------------------------------------------
# 纯单测：查询编排与边界（假 sessionmaker，不连库）
# ---------------------------------------------------------------------------


def test_construction_does_not_touch_the_sessionmaker_provider():
    """构造期不建 engine / 不取 sessionmaker（engine 延迟到首次 get_latest）。"""
    calls: list[int] = []

    def provider():
        calls.append(1)
        return _FakeSessionmaker([])

    MySQLProductRepository(sessionmaker_factory=provider)
    assert calls == []


async def test_get_latest_assembles_snapshot():
    """行为不变量：命中商品时 snapshot 非空且字段来自库行。"""
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
    """业务红线：连不上库 / SQL 报错必须抛出，不能伪装成「商品不存在」。"""
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


# ---------------------------------------------------------------------------
# 守护：默认装配路径仍是 InMemory（谁把默认改成连库，这里变红）
# ---------------------------------------------------------------------------


def test_default_product_tool_is_still_inmemory():
    assert isinstance(tool_by_name(build_tools(), "ProductTool")._repo, InMemoryProductRepository)


def test_build_tools_uses_mysql_repo_only_when_explicitly_injected():
    repo, _ = _repo_with_fake_sessions([])
    assert tool_by_name(build_tools(), "ProductTool")._repo is not repo
    assert tool_by_name(build_tools(product_repo=repo), "ProductTool")._repo is repo


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


async def _insert_seed(product_id: str, seed: dict) -> None:
    """用**裸 SQL** 写入 P_88231 同构种子（独立 product_id）。

    绕开 ORM 写：这样列名/NULL 口径由手写 DDL 独立钉住，读路径才走 ORM ——
    DDL 与 ORM 谁漂移了都直接报错。
    """
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
                "brand": seed["brand"],  # None → SQL NULL
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
    not _mysql_reachable(), reason="MySQL 不可达（未起 mysql-dev 容器）→ 跳过真库集成"
)
async def test_mysql_product_repository_roundtrip_against_real_db():
    """MySQL → ProductSnapshot → Evidence 全字段往返（种子取自工具默认商品的 P_88231）。

    独立 product_id + ``finally`` 清理：可重复跑、不污染开发库。
    """
    tag = uuid4().hex[:8]
    product_id = f"PYTEST_PRODUCT_{tag}"
    missing_id = f"PYTEST_PRODUCT_MISSING_{tag}"
    seed = dict(_DEFAULT_PRODUCTS["P_88231"])
    try:
        await _insert_seed(product_id, seed)

        # ① 全字段读回
        snap = await MySQLProductRepository().get_latest(product_id)
        assert snap is not None
        assert snap.product_id == product_id
        assert snap.category == seed["category"]
        assert snap.brand is None, "真空缺 brand 必须读成 None，不得是空串/'null'"
        assert snap.version == seed["version"]
        assert snap.status == seed["status"]

        tool = ProductTool(repo=MySQLProductRepository())

        # ② 不存在的商品 → Repository 返回 None → ProductResult.ok is False
        assert await MySQLProductRepository().get_latest(missing_id) is None
        res_missing = await tool.call(ProductArgs(product_id=missing_id), _ctx())
        assert res_missing.ok is False
        assert res_missing.product is None

        # ③ 命中 → to_evidence() = 1 条 PRODUCT_FACT + 1 条 MEASUREMENT(listing_registry)
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


# ---------------------------------------------------------------------------
# 生产装配：HTTP/落库入口读真库，默认路径与测试仍走 InMemory
# ---------------------------------------------------------------------------


def test_build_production_tools_swaps_only_the_mysql_backed_sources():
    """生产装配 = 默认 6 工具，只把 ProductTool / MerchantTool 的数据源换成真库（装配期不连库）。"""
    prod = _REAL_BUILD_PRODUCTION_TOOLS()
    default = build_tools()
    assert type(tool_by_name(prod, "ProductTool")._repo).__name__ == "MySQLProductRepository"
    assert type(tool_by_name(prod, "MerchantTool")._repo).__name__ == "MySQLMerchantRepository"
    assert [t.name for t in prod] == [t.name for t in default]
    assert [type(t).__name__ for t in prod[1:]] == [type(t).__name__ for t in default[1:]]


def test_tests_are_pinned_to_the_inmemory_world():
    """守护：测试期生产装配必须被 ``tests/conftest.py`` 钉回 InMemory（CI 上没有 MySQL）。

    本用例是那条 autouse fixture 的承重件 —— 删掉 fixture，这里立刻变红，提醒后来者：
    走到生产入口的用例会真的去连库。真库集成用例自行把真身换回去，不受本守护约束。
    """
    import pra.tools as tools_pkg

    tools = tools_pkg.build_production_tools()
    assert isinstance(tool_by_name(tools, "ProductTool")._repo, InMemoryProductRepository), (
        "conftest 的 InMemory 钉回 fixture 失效了 —— 测试会去连真库"
    )
    assert isinstance(tool_by_name(tools, "MerchantTool")._repo, InMemoryMerchantRepository), (
        "conftest 的 InMemory 钉回 fixture 失效了 —— 测试会去连真库"
    )


def test_single_source_assembly_feeds_the_production_world(monkeypatch):
    """装配唯一处守护：生产图只在组合根 ``pra.wiring.get_production_graph`` 组装一次。

    单点 patch 接缝（``pra.wiring.build_agent_graph``）+ 单份单例重置（``pra.wiring._graph``）即可
    覆盖全部生产入口；两个入口模块**不得**再自带装配/单例 —— 重复装配一旦复现，这里立刻变红。
    """
    captured: dict = {}

    def fake_build_agent_graph(*, tools=None, checkpointer=None, llm=None):
        captured["tools"] = tools
        return object()

    sentinel = object()
    monkeypatch.setattr("pra.tools.build_production_tools", lambda: sentinel)
    monkeypatch.setattr(wiring, "build_agent_graph", fake_build_agent_graph)
    monkeypatch.setattr(wiring, "_graph", None)

    from pra.api import service as api_service

    wiring.get_production_graph()
    assert captured.pop("tools") is sentinel, "组合根未把生产工具世界注入图装配"

    # 回退反证：任一入口重新自建装配/单例，以下断言即变红。
    assert not hasattr(api_service, "get_graph"), "HTTP 执行器又自建了装配入口"
    assert not hasattr(ps, "_get_graph"), "落库编排又自建了装配入口"
    assert not hasattr(ps, "_compiled_graph"), "落库编排又自带了图单例"


# ---------------------------------------------------------------------------
# 生产路径端到端（真库，可跳过）：HTTP/落库入口 → MySQL → Evidence
# ---------------------------------------------------------------------------

# 只在真库种子里的商品（``tool.py`` 的 _DEFAULT_PRODUCTS 只有 P_88231）—— 用它才能在证据层面
# 区分「读了真库」与「读了 InMemory 默认世界」。
_MYSQL_ONLY_PRODUCT = "P_77310"
# 默认 Mock 图像源认得的图（认不得的图产不出 IMAGE_SIMILARITY，脚本化 plan 就不会去调 ProductTool）。
_KNOWN_IMAGE_URL = "https://cdn.example.com/products/P_88231/img1.jpg"


def _case_for_product(case_id: str) -> ProductReviewCase:
    """COMPLEX 案件（brand 空缺 → R-301），商品只有真库有，图片是 Mock 认得的那张。"""
    case = make_case(
        case_id=case_id, brand=None, product_id=_MYSQL_ONLY_PRODUCT, merchant_id="M_5512"
    )
    return case.model_copy(
        update={
            "product": case.product.model_copy(
                update={"images": [ProductImage(url=_KNOWN_IMAGE_URL, source="主图")]}
            )
        }
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
    """按 case_id 清理本测试写入的五表行（先子表后主表；可重复运行）。"""
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
async def test_production_path_reads_product_fact_from_mysql(monkeypatch):
    """生产装配 → 落库入口 → 图 → ProductTool → MySQL → PRODUCT_FACT 落 review_evidence。

    同一案件跑两遍做**回退反证**：生产装配（真库）应取到只存在于库中的商品并落证据；换回
    InMemory 默认世界后该商品不存在，同一位置不应有 PRODUCT_FACT —— 若有人把生产装配改回
    InMemory，前一半会立刻变红。
    """
    monkeypatch.setattr("pra.tools.build_production_tools", _REAL_BUILD_PRODUCTION_TOOLS)
    # 本用例只验证 MySQL 链路：把生产装配的 RAG 侧钉回 InMemory 种子（scripted plan 分支 3 会调
    # CaseSearch/PolicySearch，真实 RAG 在缺 rag extra / 模型缓存时会尝试联网下载模型而阻塞，
    # 与「读真库商品」这一被测目标无关）。
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

        monkeypatch.setattr("pra.tools.build_production_tools", build_tools)
        monkeypatch.setattr(wiring, "_graph", None)
        out_mem = await ps.process_review(_case_for_product(memory_case_id))
        assert await _product_fact_rows(out_mem["run_id"]) == [], (
            "InMemory 默认世界没有该商品 → 不应有 PRODUCT_FACT（本断言是上面那条的回退反证）"
        )
    finally:
        await _drop_cases((mysql_case_id, memory_case_id))
