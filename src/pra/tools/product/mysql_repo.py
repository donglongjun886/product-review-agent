"""ProductTool 的 MySQL 数据源实现（``ProductRepository`` Protocol 的真实实现）。

为什么：工具默认读进程内种子（CI 不连库、评测可重放）；生产/HTTP 路径需要真实商品表。
不变量（与 InMemory 版逐字一致）：商品不存在 → 返回``None``（确定性「无结果」，由工具转
``ok=False``，不抛）；基础设施异常（连不上库 / SQL 报错）**一律向上抛**，绝不吞成 ``None``
—— 把「查不到」伪装成「证明无」是本项目的业务红线。

坑：构造期与 import 期都不建 engine / 不连库（engine 经 ``get_sessionmaker`` 懒加载，首次
``get_latest`` 才建立）；async engine 绑定创建它的 event loop，跨 loop 复用会报
``attached to a different loop``（见 ``pra.infra.db``）。装配入口：``build_production_tools()``。
"""

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
# 商品表 ORM（独立 DeclarativeBase：rdb_models.Base 只映射核心审核 5 表，勿混）
# ---------------------------------------------------------------------------


class _ProductBase(DeclarativeBase):
    """``product`` 表专属 metadata 归属（与 ``pra.infra.rdb_models.Base`` 相互独立）。"""


class ProductORM(_ProductBase):
    """``product`` 行 —— 每商品只存当前行（``version`` 即当前乐观锁版本，历史版本不入库）。

    ``brand`` 可空且**不归一**：真空缺读出来就是 ``None``，不得变空串或 ``'null'`` 字符串。
    只声明决策链真正消费的列（与 ``ProductSnapshot`` 同形）。
    """

    __tablename__ = "product"

    product_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    category: Mapped[str] = mapped_column(String(128), nullable=False)
    brand: Mapped[str | None] = mapped_column(String(64), nullable=True)
    version: Mapped[int] = mapped_column(nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)


# ---------------------------------------------------------------------------
# 行 → 快照（纯函数：不连库即可测全部字段边界）
# ---------------------------------------------------------------------------


def to_snapshot(product: ProductORM) -> ProductSnapshot:
    """DB 行 → ``ProductSnapshot``：只做类型/形态归一，不做业务判定。"""
    return ProductSnapshot(
        product_id=product.product_id,
        category=product.category,
        # NULL 原样保留 None：这是「规避品牌」调查的起点信号，不归一为空串/'null'
        brand=product.brand,
        version=int(product.version),
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
            return to_snapshot(product)
