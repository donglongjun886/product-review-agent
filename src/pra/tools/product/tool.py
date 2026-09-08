"""ProductTool —— 事实锚点工具（docs/01-agent-loop.md §5.1 /《00》§5）。

回答的业务问题：判断"规避品牌"前，先确认商品在库最新事实 —— brand 是否真空缺、
字段是否冲突、快照版本（《00》§5.1）。

分层（依赖倒置，§15.1）：

- ``ProductRepository``（Protocol）：窄接口 —— 只暴露"按 product_id 取库中最新事实
  快照"。``call()`` 只依赖该接口，不依赖任何数据源实现。
- ``InMemoryProductRepository``：**Mock 默认实现**（构造入参可注入；自带与
  《00》§4.4 / 01 §8 走查一致的演示数据）。真实 MySQL 实现（SQLAlchemy async 读
  ``product/product_sku/product_image``）待 infra 阶段接入 —— 届时实现同一 Protocol
  并注入构造器即可，本工具零改动。

本模块**不含业务判定**：只取事实并结构化为 Evidence 原料（§5.1 → Evidence 列），
"是否违规 / 阈值 / 版本漂移是否构成风险"归 reevaluate/decide/guardrails 层。
注意：§5.1 的 ``version_drift``（库中 version vs ``case.product.version``）需比对
案件快照，而 ``ToolContext`` 只带 ``case_id`` 不带 ``case`` 对象 —— O-4 已拍板：
该比对放 tools_node 的 evidence processing 层（其持有 case 快照），不扩 ToolContext；
本工具在 value 里只陈述库中事实版本。ref_id 填 ``product_id``（O-1：稳定业务标识，
非 RAG 工具也可填以支持去重/回溯）。
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from pydantic import BaseModel, Field

from ...domain.models import Evidence
from ..base import ToolArgs, ToolContext, ToolResult

# ---- §5.7 受控证据类型 & §5.1 默认证据强度 ----
PRODUCT_FACT_TYPE = "PRODUCT_FACT"  # §5.7 EvidenceType 词表
PRODUCT_FACT_WEIGHT = 0.6  # §5.1 默认权重；暂定默认，待 T-5 拍板后可调


# ---------------------------------------------------------------------------
# 数据访问契约（依赖倒置）
# ---------------------------------------------------------------------------


class SkuSnapshot(BaseModel):
    """商品 SKU 行（对应 §5.1 result.data.sku_list[]）。"""

    sku_id: str
    color: str
    size: str
    price: float


class ProductImageSnapshot(BaseModel):
    """商品图片行（§5.1 result.data.images[]）。

    注意：库中图片**不含 ocr_text** —— OCR 归 OCRTool，避免重复劳动（§5.1）。
    """

    url: str
    source: str


class ProductSnapshot(BaseModel):
    """商品在库事实快照（§5.1 result.data 的完整形状）。

    ``listing_time`` 按 §5.1 定为 str（DB DATETIME 展示串）；``status`` 为商品
    上下架状态（如 ON_SALE / REMOVED），供调查判断商品当前是否在架。
    """

    product_id: str
    merchant_id: str
    title: str
    description: str
    category: str
    brand: str | None = Field(default=None, description="真空缺为 null —— '规避品牌'调查的起点信号")
    attributes: dict[str, str] = Field(default_factory=dict)
    sku_list: list[SkuSnapshot] = Field(default_factory=list)
    images: list[ProductImageSnapshot] = Field(default_factory=list)
    version: int = Field(description="乐观锁版本号（库中最新 version）")
    listing_time: str = Field(description="上架时间展示串，如 2024-09-06 14:00:00")
    status: str = Field(default="ON_SALE", description="商品状态，如 ON_SALE / REMOVED")


class ProductRepository(Protocol):
    """商品数据源窄接口。

    真实实现：MySQL ``product/product_sku/product_image``（async 读，返回最新
    version 行）；Mock 实现见 ``InMemoryProductRepository``。找不到商品返回 None
    （确定性"无结果"，由工具转 ok=False，不抛异常）。
    """

    async def get_latest(self, product_id: str) -> ProductSnapshot | None: ...


_DEFAULT_PRODUCTS: Mapping[str, dict[str, Any]] = {
    "P_88231": {
        "product_id": "P_88231",
        "merchant_id": "M_5512",
        "title": "新款厚底复古跑鞋 女士百搭运动鞋",
        "description": "经典复古跑鞋设计，轻量缓震，适合日常通勤和运动。",
        "category": "女鞋/运动鞋",
        "brand": None,
        "attributes": {"材质": "PU", "鞋底": "橡胶", "适用人群": "女士"},
        "sku_list": [{"sku_id": "S_1", "color": "米白", "size": "36-40", "price": 129.0}],
        "images": [{"url": "https://cdn.example.com/products/P_88231/img1.jpg", "source": "主图"}],
        "version": 3,
        "listing_time": "2024-09-06 14:00:00",
        "status": "ON_SALE",
    },
}


class InMemoryProductRepository:
    """ProductRepository 的 Mock 默认实现（显式标注，仅供开发/测试/演示）。

    ``data`` 构造入参可注入自定义种子；不传则用模块级演示数据（对齐走查的商品）。
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
    """ProductTool 入参（§5.1 args Schema）。

    ``version`` 可选：v1 商品主表只存当前行（乐观锁 version），传入值仅作审计/比对
    提示，读取结果恒为库中最新 version（与 §5.1 默认语义一致）。
    """

    product_id: str = Field(description="商品 ID，如 P_88231")
    version: int | None = Field(default=None, description="期望版本（可选，默认取库中最新）")


class ProductResult(ToolResult):
    """ProductTool 出参信封 + 负载（§5.1 result.data）。

    ``ok=False``（商品不存在/已删除）时 ``product`` 为 None。
    """

    product: ProductSnapshot | None = Field(default=None, description="库中最新商品事实快照")


class ProductTool:
    """读取商品在库最新事实快照，用于确认 brand 真空缺、字段冲突、版本漂移。"""

    name = "ProductTool"
    description = "读取商品在库最新事实快照（标题/描述/属性/品牌/SKU/图片/版本），用于确认 brand 真空缺、字段冲突、版本漂移"
    args_model = ProductArgs

    def __init__(self, repo: ProductRepository | None = None) -> None:
        self._repo: ProductRepository = repo or InMemoryProductRepository()

    async def call(self, args: ProductArgs, ctx: ToolContext) -> ProductResult:
        product = await self._repo.get_latest(args.product_id)
        if product is None:
            return ProductResult(ok=False, error=f"商品不存在或已删除: {args.product_id}")
        return ProductResult(product=product)

    def to_evidence(self, result: ProductResult) -> list[Evidence]:
        """结果 → Evidence（§5.1 → Evidence 列）：每商品 1 条 PRODUCT_FACT。

        ``source`` = 工具名；``ref_id=product_id``（O-1 拍板：稳定业务标识，供去重/回溯，
        不再一律留 None）。value 只陈述库中事实（brand/version/status），
        "标题/描述是否含品牌词"的措辞需规则引擎词表支撑，真实实现再补 —— 本工具不臆断。
        ``version_drift`` 判定归 tools_node（O-4，需 case 快照比对），本工具不产。
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
                weight=PRODUCT_FACT_WEIGHT,
                ref_id=p.product_id,
            )
        ]
