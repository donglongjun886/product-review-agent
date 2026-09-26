"""ProductTool —— 事实锚点工具。"""

from __future__ import annotations

from typing import Protocol

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
# 数据访问契约
# ---------------------------------------------------------------------------


class ProductSnapshot(BaseModel):
    """商品在库事实快照（``status`` 如 ON_SALE / REMOVED）。"""

    product_id: str
    category: str
    brand: str | None = Field(default=None, description="真空缺为 null —— '规避品牌'调查的起点信号")
    version: int = Field(description="乐观锁版本号（库中最新 version）")
    status: str = Field(default="ON_SALE", description="商品状态，如 ON_SALE / REMOVED")


class ProductRepository(Protocol):
    """商品数据源窄接口（按 product_id 取当前行）。

    商品不存在返回 ``None``；基础设施异常由实现直接抛出。
    """

    async def get_latest(self, product_id: str) -> ProductSnapshot | None: ...


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class ProductArgs(ToolArgs):
    """ProductTool 入参。

    ``version`` 仅作审计/比对提示，读取结果恒为库中当前行。
    """

    product_id: str = Field(description="商品 ID，如 P_88231")
    version: int | None = Field(default=None, description="期望版本（可选，默认取库中最新）")


class ProductResult(ToolResult):
    """ProductTool 出参信封 + 负载。

    ``ok=False`` 时 ``product`` 为 None。
    """

    product: ProductSnapshot | None = Field(default=None, description="库中最新商品事实快照")


class ProductTool:

    name = "ProductTool"
    description = "读取商品在库最新事实快照（品牌/类目/版本/状态），用于确认 brand 真空缺与版本漂移"
    args_model = ProductArgs
    measured_dimensions: frozenset[str] = frozenset({DIM_LISTING_REGISTRY})
    measurement_available: bool = True

    def __init__(self, repo: ProductRepository) -> None:
        self._repo: ProductRepository = repo

    async def call(self, args: ProductArgs, ctx: ToolContext) -> ProductResult:
        product = await self._repo.get_latest(args.product_id)
        if product is None:
            return ProductResult(ok=False, error=f"商品不存在或已删除: {args.product_id}")
        return ProductResult(product=product)

    def to_evidence(self, result: ProductResult) -> list[Evidence]:
        """结果 → Evidence：1 条 ``PRODUCT_FACT`` + 1 条 ``MEASUREMENT``（listing_registry）。

        ``ok=False`` 时不产任何证据。
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
