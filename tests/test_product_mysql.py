"""ProductTool 的 MySQL 数据源（``mysql_repo``）：真库集成（可跳过）+ 恒跑纯单测。

真库部分照抄 ``tests/test_infra_persist_smoke.py`` 的默认配置路径 + 1s socket 探测 ``skipif``，
**不 mock ``db._settings``、不注 ``_env_file``** —— 专测「.env / DATABASE_URL 配错就连不上」
这条链；用例自建自删（``PYTEST_PRODUCT_`` 前缀 + ``finally`` 清理），重复跑不污染开发库。

纯单测部分注入假 sessionmaker，覆盖 DB 行 → ``ProductSnapshot`` 的全部边界（``brand`` 为
SQL NULL 不得归一成空串/``'null'``、``attributes`` 为 NULL/空、无 SKU、无图片、``DATETIME(3)``
→ 展示串、``DECIMAL`` → float），并守护「默认装配路径仍是 InMemory、不连库」。
"""

from __future__ import annotations

import json
import socket
from datetime import datetime
from decimal import Decimal
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
from pra.tools.product.mysql_repo import (
    MySQLProductRepository,
    ProductImageORM,
    ProductORM,
    ProductSkuORM,
    to_snapshot,
)
from pra.tools.product.tool import (
    _DEFAULT_PRODUCTS,
    PRODUCT_FACT_TYPE,
    PRODUCT_FACT_WEIGHT,
    InMemoryProductRepository,
    ProductArgs,
    ProductResult,
    ProductTool,
)

# 模块导入期绑定真身：``tests/conftest.py`` 的 autouse fixture 会把 ``pra.tools`` 上的这个名字
# 换成 InMemory 装配（测试不连库），此处留住原函数供「生产装配」与真库集成用例显式换回去。
_REAL_BUILD_PRODUCTION_TOOLS = build_production_tools

# ---------------------------------------------------------------------------
# 替身：假 session / 假 sessionmaker（形状与 async_sessionmaker 一致，可 async with）
# ---------------------------------------------------------------------------


class _FakeResult:
    """``session.execute()`` 的替身：``scalar_one_or_none()`` 走 ``scalar``，``scalars().all()`` 走 ``many``。"""

    def __init__(self, *, scalar: object = None, many: tuple = ()) -> None:
        self._scalar = scalar
        self._many = list(many)

    def scalar_one_or_none(self) -> object:
        return self._scalar

    def scalars(self) -> _FakeResult:
        return self

    def all(self) -> list:
        return list(self._many)


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
        "merchant_id": "M_TEST",
        "title": "新款厚底复古跑鞋 女士百搭运动鞋",
        "description": "经典复古跑鞋设计，轻量缓震。",
        "category": "女鞋/运动鞋",
        "brand": None,
        "attributes": {"材质": "PU"},
        "version": 3,
        "listing_time": _NAIVE_LISTING_TIME,
        "status": "ON_SALE",
    }
    values.update(overrides)
    return ProductORM(**values)


def _sku_row(**overrides: object) -> ProductSkuORM:
    values: dict[str, object] = {
        "product_id": "P_TEST",
        "sku_id": "S_1",
        "color": "米白",
        "size": "36-40",
        "price": Decimal("129.00"),
        "sort_order": 1,
    }
    values.update(overrides)
    return ProductSkuORM(**values)


def _image_row(**overrides: object) -> ProductImageORM:
    values: dict[str, object] = {
        "image_id": 1,
        "product_id": "P_TEST",
        "url": "https://cdn.example.com/products/P_TEST/img1.jpg",
        "source": "主图",
        "sort_order": 1,
    }
    values.update(overrides)
    return ProductImageORM(**values)


def _ctx() -> ToolContext:
    return ToolContext(run_id="R_TEST", case_id="C_TEST", budget=Budget())


# 库中 listing_time 是 naive DATETIME(3)（见 pra.infra.db 时间口径）；用 fromisoformat 造，
# 避免 datetime(...) 触发 DTZ001 的同时保持 naive 语义。
_NAIVE_LISTING_TIME = datetime.fromisoformat("2024-09-06T14:00:00")
_NAIVE_LISTING_TIME_MS = datetime.fromisoformat("2024-09-06T14:00:00.123000")


# ---------------------------------------------------------------------------
# 纯单测：DB 行 → ProductSnapshot 映射边界
# ---------------------------------------------------------------------------


def test_to_snapshot_maps_every_field():
    snap = to_snapshot(_product_row(brand="潮动"), [_sku_row()], [_image_row()])
    assert snap.product_id == "P_TEST"
    assert snap.merchant_id == "M_TEST"
    assert snap.title == "新款厚底复古跑鞋 女士百搭运动鞋"
    assert snap.description == "经典复古跑鞋设计，轻量缓震。"
    assert snap.category == "女鞋/运动鞋"
    assert snap.brand == "潮动"
    assert snap.attributes == {"材质": "PU"}
    assert snap.version == 3
    assert snap.listing_time == "2024-09-06 14:00:00"
    assert snap.status == "ON_SALE"
    assert [(s.sku_id, s.color, s.size, s.price) for s in snap.sku_list] == [
        ("S_1", "米白", "36-40", 129.0)
    ]
    assert [(i.url, i.source) for i in snap.images] == [
        ("https://cdn.example.com/products/P_TEST/img1.jpg", "主图")
    ]


def test_brand_sql_null_is_not_normalised_to_empty_string():
    """业务红线：brand 真空缺必须原样是 ``None``（不是 ''、不是 'null'）。"""
    snap = to_snapshot(_product_row(brand=None), [], [])
    assert snap.brand is None
    assert snap.brand != ""
    assert snap.brand != "null"

    evs = ProductTool().to_evidence(ProductResult(product=snap))
    assert len(evs) == 1
    assert "brand=null" in evs[0].value


@pytest.mark.parametrize("attributes", [None, {}])
def test_attributes_null_or_empty_both_become_empty_dict(attributes):
    snap = to_snapshot(_product_row(attributes=attributes), [], [])
    assert snap.attributes == {}


def test_no_sku_and_no_image_yield_empty_lists():
    snap = to_snapshot(_product_row(), [], [])
    assert snap.sku_list == []
    assert snap.images == []


@pytest.mark.parametrize(
    "raw",
    [
        _NAIVE_LISTING_TIME,
        _NAIVE_LISTING_TIME_MS,  # DATETIME(3) 毫秒截断（展示串只到秒）
        "2024-09-06 14:00:00",  # 驱动/替身直接给字符串时原样兜底
    ],
)
def test_listing_time_is_formatted_to_display_string(raw):
    snap = to_snapshot(_product_row(listing_time=raw), [], [])
    assert snap.listing_time == "2024-09-06 14:00:00"


def test_decimal_price_becomes_float():
    snap = to_snapshot(_product_row(), [_sku_row(price=Decimal("159.50"))], [])
    assert snap.sku_list[0].price == 159.5
    assert isinstance(snap.sku_list[0].price, float)


def test_sku_and_image_order_follows_provided_rows():
    """顺序由 repo 的 ORDER BY 决定，映射层原样保序（不重排）。"""
    snap = to_snapshot(
        _product_row(),
        [_sku_row(sku_id="S_2", sort_order=2), _sku_row(sku_id="S_1", sort_order=1)],
        [_image_row(image_id=2, source="附图1"), _image_row(image_id=1, source="主图")],
    )
    assert [s.sku_id for s in snap.sku_list] == ["S_2", "S_1"]
    assert [i.source for i in snap.images] == ["附图1", "主图"]


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


async def test_get_latest_assembles_snapshot_and_queries_children():
    repo, sm = _repo_with_fake_sessions(
        [
            _FakeResult(scalar=_product_row()),
            _FakeResult(many=(_sku_row(),)),
            _FakeResult(many=(_image_row(),)),
        ]
    )
    snap = await repo.get_latest("P_TEST")
    assert snap is not None
    assert snap.product_id == "P_TEST"
    assert len(snap.sku_list) == 1
    assert len(snap.images) == 1
    assert len(sm.sessions) == 1
    assert len(sm.sessions[0].queries) == 3, "主表 + SKU + 图片各一条查询"


async def test_get_latest_missing_product_returns_none_without_child_queries():
    repo, sm = _repo_with_fake_sessions([_FakeResult(scalar=None)])
    assert await repo.get_latest("P_NOPE") is None
    assert len(sm.sessions[0].queries) == 1, "主表查无该商品即返回，不查子表"


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
    assert type(ProductTool()._repo).__name__ == "InMemoryProductRepository"
    assert type(build_tools()[0]._repo).__name__ == "InMemoryProductRepository"
    assert isinstance(ProductTool()._repo, InMemoryProductRepository)


def test_build_tools_uses_mysql_repo_only_when_explicitly_injected():
    repo, _ = _repo_with_fake_sessions([])
    assert build_tools()[0]._repo is not repo
    assert build_tools(product_repo=repo)[0]._repo is repo


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

    绕开 ORM 写：这样列名/JSON/NULL/DECIMAL 口径由手写 DDL 独立钉住，读路径才走 ORM ——
    DDL 与 ORM 谁漂移了都直接报错。
    """
    sm = get_sessionmaker()
    async with sm() as s:
        await s.execute(
            text(
                "insert into product (product_id, merchant_id, title, description, category, "
                "brand, attributes, version, listing_time, status) values "
                "(:pid, :mid, :title, :desc, :cat, :brand, :attrs, :ver, :lt, :status)"
            ),
            {
                "pid": product_id,
                "mid": seed["merchant_id"],
                "title": seed["title"],
                "desc": seed["description"],
                "cat": seed["category"],
                "brand": seed["brand"],  # None → SQL NULL
                "attrs": json.dumps(seed["attributes"], ensure_ascii=False),
                "ver": seed["version"],
                "lt": seed["listing_time"],
                "status": seed["status"],
            },
        )
        for order, sku in enumerate(seed["sku_list"], start=1):
            await s.execute(
                text(
                    "insert into product_sku (product_id, sku_id, color, size, price, sort_order) "
                    "values (:pid, :sid, :color, :size, :price, :order)"
                ),
                {
                    "pid": product_id,
                    "sid": sku["sku_id"],
                    "color": sku["color"],
                    "size": sku["size"],
                    "price": sku["price"],
                    "order": order,
                },
            )
        for order, image in enumerate(seed["images"], start=1):
            await s.execute(
                text(
                    "insert into product_image (product_id, url, source, sort_order) "
                    "values (:pid, :url, :source, :order)"
                ),
                {
                    "pid": product_id,
                    "url": image["url"],
                    "source": image["source"],
                    "order": order,
                },
            )
        await s.commit()


async def _delete_seed(product_id: str) -> None:
    """自建自删：先子表后主表（级联已开，仍显式删，语义不依赖 DDL 开关）。"""
    sm = get_sessionmaker()
    async with sm() as s:
        for table in ("product_image", "product_sku", "product"):
            await s.execute(
                text(f"delete from {table} where product_id = :pid"), {"pid": product_id}
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
        assert snap.merchant_id == seed["merchant_id"]
        assert snap.title == seed["title"]
        assert snap.description == seed["description"]
        assert snap.category == seed["category"]
        assert snap.brand is None, "真空缺 brand 必须读成 None，不得是空串/'null'"
        assert snap.attributes == seed["attributes"]
        assert snap.version == seed["version"]
        assert snap.listing_time == seed["listing_time"] == "2024-09-06 14:00:00"
        assert snap.status == seed["status"]
        assert [(s.sku_id, s.color, s.size, s.price) for s in snap.sku_list] == [
            ("S_1", "米白", "36-40", 129.0)
        ]
        assert [(i.url, i.source) for i in snap.images] == [
            ("https://cdn.example.com/products/P_88231/img1.jpg", "主图")
        ]

        tool = ProductTool(repo=MySQLProductRepository())

        # ② 不存在的商品 → Repository 返回 None → ProductResult.ok is False
        assert await MySQLProductRepository().get_latest(missing_id) is None
        res_missing = await tool.call(ProductArgs(product_id=missing_id), _ctx())
        assert res_missing.ok is False
        assert res_missing.product is None

        # ③ 命中 → to_evidence() 恰好 1 条 PRODUCT_FACT
        res = await tool.call(ProductArgs(product_id=product_id), _ctx())
        assert res.ok is True
        evs = tool.to_evidence(res)
        assert len(evs) == 1
        assert evs[0].type == PRODUCT_FACT_TYPE
        assert evs[0].ref_id == product_id
        assert evs[0].weight == PRODUCT_FACT_WEIGHT
        assert evs[0].source == "ProductTool"
    finally:
        await _delete_seed(product_id)


# ---------------------------------------------------------------------------
# 生产装配：HTTP/落库入口读真库，默认路径与测试仍走 InMemory
# ---------------------------------------------------------------------------


def test_build_production_tools_swaps_only_the_mysql_backed_sources():
    """生产装配 = 默认 6 工具，只把 ProductTool / MerchantTool 的数据源换成真库（装配期不连库）。"""
    prod = _REAL_BUILD_PRODUCTION_TOOLS()
    default = build_tools()
    assert type(prod[0]._repo).__name__ == "MySQLProductRepository"
    assert type(prod[3]._repo).__name__ == "MySQLMerchantRepository"
    assert [t.name for t in prod] == [t.name for t in default]
    assert [type(t).__name__ for t in prod[1:]] == [type(t).__name__ for t in default[1:]]


def test_tests_are_pinned_to_the_inmemory_world():
    """守护：测试期生产装配必须被 ``tests/conftest.py`` 钉回 InMemory（CI 上没有 MySQL）。

    本用例是那条 autouse fixture 的承重件 —— 删掉 fixture，这里立刻变红，提醒后来者：
    走到生产入口的用例会真的去连库。真库集成用例自行把真身换回去，不受本守护约束。
    """
    import pra.tools as tools_pkg

    tools = tools_pkg.build_production_tools()
    assert type(tools[0]._repo).__name__ == "InMemoryProductRepository", (
        "conftest 的 InMemory 钉回 fixture 失效了 —— 测试会去连真库"
    )
    assert type(tools[3]._repo).__name__ == "InMemoryMerchantRepository", (
        "conftest 的 InMemory 钉回 fixture 失效了 —— 测试会去连真库"
    )


def test_http_graph_entries_pass_the_production_world(monkeypatch):
    """两个生产入口（HTTP 执行器 / 落库编排）都必须把生产工具世界显式注入图装配。"""
    captured: dict = {}

    def fake_build_agent_graph(*, tools=None, checkpointer=None, llm=None):
        captured["tools"] = tools
        return object()

    sentinel = object()
    monkeypatch.setattr("pra.agent.graph.build_agent_graph", fake_build_agent_graph)
    monkeypatch.setattr("pra.tools.build_production_tools", lambda: sentinel)

    from pra.api import service as api_service

    monkeypatch.setattr(api_service, "_graph", None)
    monkeypatch.setattr(ps, "_compiled_graph", None)
    api_service.get_graph()
    assert captured.pop("tools") is sentinel, "HTTP 执行器未注入生产工具世界"
    ps._get_graph()
    assert captured.pop("tools") is sentinel, "落库编排未注入生产工具世界"


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
    tag = uuid4().hex[:8]
    mysql_case_id, memory_case_id = f"PYTEST_HTTP_MYSQL_{tag}", f"PYTEST_HTTP_MEM_{tag}"
    try:
        monkeypatch.setattr(ps, "_compiled_graph", None)
        out_mysql = await ps.process_review(_case_for_product(mysql_case_id))
        facts = await _product_fact_rows(out_mysql["run_id"])
        assert len(facts) == 1, "生产路径应恰好落 1 条 PRODUCT_FACT"
        assert facts[0][1] == "ProductTool"
        assert facts[0][2] == _MYSQL_ONLY_PRODUCT, "ref_id 必须是案件商品（真库读到的那个）"
        assert "version=1" in facts[0][3], "版本必须来自真库行（P_77310 的 version=1）"

        monkeypatch.setattr("pra.tools.build_production_tools", build_tools)
        monkeypatch.setattr(ps, "_compiled_graph", None)
        out_mem = await ps.process_review(_case_for_product(memory_case_id))
        assert await _product_fact_rows(out_mem["run_id"]) == [], (
            "InMemory 默认世界没有该商品 → 不应有 PRODUCT_FACT（本断言是上面那条的回退反证）"
        )
    finally:
        await _drop_cases((mysql_case_id, memory_case_id))
