"""ProductTool 的 MySQL 数据源实现（``ProductRepository`` Protocol 的真实实现）。

为什么：工具默认读进程内种子（CI 不连库、评测可重放）；生产/HTTP 路径需要真实商品表。
不变量（与 InMemory 版逐字一致）：商品不存在 → 返回``None``（确定性「无结果」，由工具转
``ok=False``，不抛）；基础设施异常（连不上库 / SQL 报错）**一律向上抛**，绝不吞成 ``None``
—— 把「查不到」伪装成「证明无」是本项目的业务红线。

坑：构造期与 import 期都不建 engine / 不连库（engine 经 ``get_sessionmaker`` 懒加载，首次
``get_latest`` 才建立）；async engine 绑定创建它的 event loop，跨 loop 复用会报
``attached to a different loop``（见 ``pra.infra.db``）。装配入口：``build_tools(product_repo=...)``。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from sqlalchemy import String, Text, select
from sqlalchemy.dialects.mysql import BIGINT, DATETIME, DECIMAL, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from ...infra.db import get_sessionmaker
from .tool import ProductImageSnapshot, ProductSnapshot, SkuSnapshot

__all__ = [
    "LISTING_TIME_FORMAT",
    "MySQLProductRepository",
    "ProductImageORM",
    "ProductORM",
    "ProductSkuORM",
    "to_snapshot",
]


# ---------------------------------------------------------------------------
# 商品 3 表 ORM（独立 DeclarativeBase：rdb_models.Base 只映射核心审核 5 表，勿混）
# ---------------------------------------------------------------------------


class _ProductBase(DeclarativeBase):
    """商品 3 表专属 metadata 归属（与 ``pra.infra.rdb_models.Base`` 相互独立）。"""


class ProductORM(_ProductBase):
    """``product`` 行 —— 每商品只存当前行（``version`` 即当前乐观锁版本，历史版本不入库）。

    ``brand`` 可空且**不归一**：真空缺读出来就是 ``None``，不得变空串或 ``'null'`` 字符串。
    ``attributes`` 无属性为 SQL NULL，映射层归一为 ``{}``。

    刻意不声明 ``relationship``：取 SKU / 图片走显式查询 —— 避免隐式懒加载在
    ``async`` 会话里抛 ``MissingGreenlet``，也不引入双向导航。
    """

    __tablename__ = "product"

    product_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    merchant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(128), nullable=False)
    brand: Mapped[str | None] = mapped_column(String(64), nullable=True)
    attributes: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    version: Mapped[int] = mapped_column(nullable=False)
    listing_time: Mapped[object] = mapped_column(DATETIME(fsp=3), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)


class ProductSkuORM(_ProductBase):
    """``product_sku`` 行（1 商品 N SKU；``sort_order`` 是读出顺序键，SQL 结果本身无序）。"""

    __tablename__ = "product_sku"

    product_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    sku_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    color: Mapped[str] = mapped_column(String(64), nullable=False)
    size: Mapped[str] = mapped_column(String(64), nullable=False)
    price: Mapped[Any] = mapped_column(DECIMAL(10, 2), nullable=False)  # → Decimal（映射层转 float）
    sort_order: Mapped[int] = mapped_column(nullable=False)


class ProductImageORM(_ProductBase):
    """``product_image`` 行 —— **不含 ocr_text**（OCR 归 OCRTool，避免重复劳动）。"""

    __tablename__ = "product_image"

    image_id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    product_id: Mapped[str] = mapped_column(String(64), nullable=False)
    url: Mapped[str] = mapped_column(String(512), nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    sort_order: Mapped[int] = mapped_column(nullable=False)


# ---------------------------------------------------------------------------
# 行 → 快照（纯函数：不连库即可测全部字段边界）
# ---------------------------------------------------------------------------

LISTING_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def _format_listing_time(value: object) -> str:
    """``DATETIME(3)`` 读出的 datetime → ``ProductSnapshot.listing_time`` 展示串口径。

    只保留秒（毫秒截断）—— 展示串格式与 InMemory 种子逐字一致；非 datetime（驱动/替身给
    字符串）原样 ``str()`` 兜底。
    """
    if isinstance(value, datetime):
        return value.strftime(LISTING_TIME_FORMAT)
    return str(value)


def to_snapshot(
    product: ProductORM,
    skus: list[ProductSkuORM],
    images: list[ProductImageORM],
) -> ProductSnapshot:
    """DB 行 → ``ProductSnapshot``：只做类型/形态归一，不做业务判定。"""
    return ProductSnapshot(
        product_id=product.product_id,
        merchant_id=product.merchant_id,
        title=product.title,
        description=product.description,
        category=product.category,
        # NULL 原样保留 None：这是「规避品牌」调查的起点信号，不归一为空串/'null'
        brand=product.brand,
        attributes=dict(product.attributes or {}),  # NULL / {} 均归一为 {}
        sku_list=[
            SkuSnapshot(
                sku_id=s.sku_id, color=s.color, size=s.size, price=float(s.price)
            )
            for s in skus
        ],
        images=[ProductImageSnapshot(url=i.url, source=i.source) for i in images],
        version=int(product.version),
        listing_time=_format_listing_time(product.listing_time),
        status=product.status,
    )


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

# () -> async_sessionmaker 的提供者；默认 get_sessionmaker，测试可注入返回假 sessionmaker 的替身。
SessionFactory = Callable[[], Any]


class MySQLProductRepository:
    """``ProductRepository`` 的 MySQL 实现（**显式 opt-in**，默认装配路径不用它）。

    ``sessionmaker_factory`` 是 ``() -> async_sessionmaker`` 的**提供者**，默认
    ``pra.infra.db.get_sessionmaker``（进程级懒加载单例）。构造期不调用它 —— engine 到首次
    ``get_latest`` 才建立，故 import / 构造无副作用。注入替身（返回假 sessionmaker）即可在
    无库环境单测查询编排与边界。
    """

    def __init__(self, sessionmaker_factory: SessionFactory = get_sessionmaker) -> None:
        self._sessionmaker_factory = sessionmaker_factory

    async def get_latest(self, product_id: str) -> ProductSnapshot | None:
        """取该商品的当前行（= 最新 version；库中只存当前行）。

        不存在 → ``None``；基础设施异常不捕获，直接抛出（见模块 docstring 红线）。
        """
        sessionmaker = self._sessionmaker_factory()
        async with sessionmaker() as session:
            product = (
                await session.execute(
                    select(ProductORM).where(ProductORM.product_id == product_id)
                )
            ).scalar_one_or_none()
            if product is None:
                return None
            skus = (
                (
                    await session.execute(
                        select(ProductSkuORM)
                        .where(ProductSkuORM.product_id == product_id)
                        .order_by(ProductSkuORM.sort_order, ProductSkuORM.sku_id)
                    )
                )
                .scalars()
                .all()
            )
            images = (
                (
                    await session.execute(
                        select(ProductImageORM)
                        .where(ProductImageORM.product_id == product_id)
                        .order_by(ProductImageORM.sort_order, ProductImageORM.image_id)
                    )
                )
                .scalars()
                .all()
            )
            # 在 session 关闭前物料化：避免依赖 detached 实例的属性访问语义
            return to_snapshot(product, list(skus), list(images))
