"""ProductTool —— 事实锚点工具。

回答的业务问题：判断「规避品牌」前，先确认商品在库最新事实 —— brand 是否真空缺、快照版本。

``ProductRepository`` 是窄接口（按 product_id 取在库事实快照）；``InMemoryProductRepository``
是 **Mock 默认实现**（默认装配路径恒用它，CI 不连库、评测可重放），真实实现是
``pra.tools.product.mysql_repo.MySQLProductRepository``（显式 opt-in：``build_tools(product_repo=…)``）。
本模块**不含业务判定**：只取事实并结构化为 Evidence 原料。``ref_id`` 填 ``product_id``。
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from pydantic import BaseModel, Field

from ...domain.measurement import (
    DEFAULT_EVIDENCE_WEIGHT,
    DIM_LISTING_REGISTRY,
    VERDICT_NEGATIVE,
    make_measurement,
)
from ...domain.models import Evidence
from ..base import ToolArgs, ToolContext, ToolResult

# ---- 受控证据类型 ----
PRODUCT_FACT_TYPE = "PRODUCT_FACT"  # EvidenceType 词表


# ---------------------------------------------------------------------------
# 数据访问契约（依赖倒置）
# ---------------------------------------------------------------------------


class ProductSnapshot(BaseModel):
    """商品在库事实快照（只保留决策链真正消费的列，与迁移 003 的 ``product`` 表同形）。

    ``status`` 为上下架状态（如 ON_SALE / REMOVED）。
    """

    product_id: str
    category: str
    brand: str | None = Field(default=None, description="真空缺为 null —— '规避品牌'调查的起点信号")
    version: int = Field(description="乐观锁版本号（库中最新 version）")
    status: str = Field(default="ON_SALE", description="商品状态，如 ON_SALE / REMOVED")


class ProductRepository(Protocol):
    """商品数据源窄接口。

    语义：取该商品**当前行** —— ``product`` 表以 product_id 为主键只存当前行，该行 ``version``
    即最新乐观锁版本（历史版本不入库），故「取当前行」与「取最新 version 行」是同一件事。
    实现：``InMemoryProductRepository``（Mock 默认）/ ``MySQLProductRepository``（真库，显式注入）。
    **商品不存在返回 None**（确定性「无结果」，由工具转 ``ok=False``，不抛异常）；基础设施异常
    由实现直接抛出，不得吞成 None。
    """

    async def get_latest(self, product_id: str) -> ProductSnapshot | None: ...


_DEFAULT_PRODUCTS: Mapping[str, dict[str, Any]] = {
    "P_88231": {
        "product_id": "P_88231",
        "category": "女鞋/运动鞋",
        "brand": None,
        "version": 3,
        "status": "ON_SALE",
    },
}


class InMemoryProductRepository:
    """ProductRepository 的 Mock 默认实现（仅供开发/测试/演示）。

    ``data`` 构造入参可注入自定义种子；不传则用模块级演示数据。
    """

    def __init__(self, data: Mapping[str, dict[str, Any]] | None = None) -> None:
        self._store: dict[str, ProductSnapshot] = {
            pid: ProductSnapshot.model_validate(row) for pid, row in (data or _DEFAULT_PRODUCTS).items()
        }

    async def get_latest(self, product_id: str) -> ProductSnapshot | None:
        return self._store.get(product_id)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class ProductArgs(ToolArgs):
    """ProductTool 入参。

    ``version`` 可选：库中只存当前行（product_id 主键，乐观锁 version 即最新），传入值仅作
    审计/比对提示，读取结果恒为库中当前行的 version。
    """

    product_id: str = Field(description="商品 ID，如 P_88231")
    version: int | None = Field(default=None, description="期望版本（可选，默认取库中最新）")


class ProductResult(ToolResult):
    """ProductTool 出参信封 + 负载。

    ``ok=False``（商品不存在/已删除）时 ``product`` 为 None。
    """

    product: ProductSnapshot | None = Field(default=None, description="库中最新商品事实快照")


class ProductTool:

    name = "ProductTool"
    description = "读取商品在库最新事实快照（品牌/类目/版本/状态），用于确认 brand 真空缺与版本漂移"
    args_model = ProductArgs
    # 本工具覆盖的风险维度 + 本部署是否真的能测（gate 的 required/coverage 判定读它）
    measured_dimensions: frozenset[str] = frozenset({DIM_LISTING_REGISTRY})
    measurement_available: bool = True

    def __init__(self, repo: ProductRepository | None = None) -> None:
        self._repo: ProductRepository = repo or InMemoryProductRepository()

    async def call(self, args: ProductArgs, ctx: ToolContext) -> ProductResult:
        product = await self._repo.get_latest(args.product_id)
        if product is None:
            return ProductResult(ok=False, error=f"商品不存在或已删除: {args.product_id}")
        return ProductResult(product=product)

    def to_evidence(self, result: ProductResult) -> list[Evidence]:
        """结果 → Evidence：1 条 ``PRODUCT_FACT`` + 1 条 ``MEASUREMENT``（listing_registry）。

        ``source`` = 工具名；``ref_id=product_id``（稳定业务标识，供去重/回溯）。value 只陈述
        库中事实（brand/version/status）；「标题/描述是否含品牌词」的措辞需规则引擎词表支撑，
        本工具不臆断。

        **查无商品（``ok=False``）刻意不产任何证据** —— 包括不产 ``MEASUREMENT``：
        "没测到"不是"测过且阴性"，gate 侧按 ``NOT_MEASURED`` 处理（`domain/measurement.py`）。

        ``listing_registry`` 是**可核验性维度**，本次只判"事实取到了没有"，
        不存在阳性路径（案件声明与在库事实不一致的确定性比对是既有已知边界，另行设计）。
        """
        if not result.ok or result.product is None:
            return []
        p = result.product
        brand_repr = "null" if p.brand is None else p.brand
        value = (
            f"brand={brand_repr}, version={p.version}（库中最新）, "
            f"status={p.status}, category={p.category}"
        )
        return [
            Evidence(
                type=PRODUCT_FACT_TYPE,
                source=self.name,
                value=value,
                weight=DEFAULT_EVIDENCE_WEIGHT,
                ref_id=p.product_id,
            ),
            make_measurement(
                dimension=DIM_LISTING_REGISTRY,
                source=self.name,
                source_ref=p.product_id,
                verdict=VERDICT_NEGATIVE,
                weight=DEFAULT_EVIDENCE_WEIGHT,
                value=f"商品事实已在库核验：brand={brand_repr}, category={p.category}",
            ),
        ]
