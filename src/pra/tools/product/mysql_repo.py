"""ProductTool 的 MySQL 数据源实现（``ProductRepository`` Protocol 的真实实现）。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy import String, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from ...infra.db import get_sessionmaker
from .tool import ProductSnapshot

__all__ = [
    "MySQLProductRepository",
    "ProductORM",
    "to_snapshot",
]


# ---------------------------------------------------------------------------
# 商品表 ORM
# ---------------------------------------------------------------------------


class _ProductBase(DeclarativeBase):
    """``product`` 表专属 metadata 归属。"""


class ProductORM(_ProductBase):
    """``product`` 行（每商品只存当前行）。"""

    __tablename__ = "product"

    product_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    category: Mapped[str] = mapped_column(String(128), nullable=False)
    brand: Mapped[str | None] = mapped_column(String(64), nullable=True)
    version: Mapped[int] = mapped_column(nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)


# ---------------------------------------------------------------------------
# 行 → 快照（纯函数）
# ---------------------------------------------------------------------------


def to_snapshot(product: ProductORM) -> ProductSnapshot:
    """DB 行 → ``ProductSnapshot``：只做类型/形态归一，不做业务判定。"""
    return ProductSnapshot(
        product_id=product.product_id,
        category=product.category,
        brand=product.brand,
        version=int(product.version),
        status=product.status,
    )


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

# () -> async_sessionmaker 的提供者。
SessionFactory = Callable[[], Any]


class MySQLProductRepository:
    """``ProductRepository`` 的 MySQL 实现。

    ``sessionmaker_factory`` 为 ``() -> async_sessionmaker`` 提供者，默认 ``pra.infra.db.get_sessionmaker``。
    """

    def __init__(self, sessionmaker_factory: SessionFactory = get_sessionmaker) -> None:
        self._sessionmaker_factory = sessionmaker_factory

    async def get_latest(self, product_id: str) -> ProductSnapshot | None:
        """取该商品的当前行。

        不存在 → ``None``；基础设施异常直接抛出。
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
            return to_snapshot(product)
