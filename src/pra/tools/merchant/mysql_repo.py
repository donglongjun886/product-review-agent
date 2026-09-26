"""MerchantTool 的 MySQL 数据源实现（``MerchantRepository`` Protocol 的真实实现）。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sqlalchemy import String, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from ...infra.db import get_sessionmaker
from .tool import MerchantProfile

__all__ = [
    "MerchantORM",
    "MySQLMerchantRepository",
    "to_profile",
]


# ---------------------------------------------------------------------------
# 商家表 ORM
# ---------------------------------------------------------------------------


class _MerchantBase(DeclarativeBase):
    """``merchant`` 表专属 metadata 归属。"""


class MerchantORM(_MerchantBase):
    """``merchant`` 行 —— 每商家一行画像。"""

    __tablename__ = "merchant"

    merchant_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    similar_product_count: Mapped[int] = mapped_column(nullable=False)
    removals: Mapped[int] = mapped_column(nullable=False)
    title_relisting_count: Mapped[int] = mapped_column(nullable=False)
    credit_score: Mapped[int] = mapped_column(nullable=False)


# ---------------------------------------------------------------------------
# 行 → 画像（纯函数）
# ---------------------------------------------------------------------------


def to_profile(merchant: MerchantORM) -> MerchantProfile:
    """DB 行 → ``MerchantProfile``：只做类型/形态归一，不做业务判定。"""
    return MerchantProfile(
        merchant_id=merchant.merchant_id,
        similar_product_count=int(merchant.similar_product_count),
        removals=int(merchant.removals),
        title_relisting_count=int(merchant.title_relisting_count),
        credit_score=int(merchant.credit_score),
    )


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

# () -> async_sessionmaker 的提供者。
SessionFactory = Callable[[], Any]


class MySQLMerchantRepository:
    """``MerchantRepository`` 的 MySQL 实现。

    ``sessionmaker_factory`` 为 ``() -> async_sessionmaker`` 提供者，默认 ``pra.infra.db.get_sessionmaker``。
    """

    def __init__(self, sessionmaker_factory: SessionFactory = get_sessionmaker) -> None:
        self._sessionmaker_factory = sessionmaker_factory

    async def get_profile(self, merchant_id: str, window_days: int) -> MerchantProfile | None:
        """取该商家的行为画像。

        不存在 → ``None``；基础设施异常直接抛出。
        """
        sessionmaker = self._sessionmaker_factory()
        async with sessionmaker() as session:
            merchant = (
                await session.execute(
                    select(MerchantORM).where(MerchantORM.merchant_id == merchant_id)
                )
            ).scalar_one_or_none()
            if merchant is None:
                return None
            return to_profile(merchant)
