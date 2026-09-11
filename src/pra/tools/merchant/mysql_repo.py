"""MerchantTool 的 MySQL 数据源实现（``MerchantRepository`` Protocol 的真实实现）。

为什么：工具默认读进程内种子（CI 不连库、评测可重放）；生产/HTTP 路径需要真实商家行为数据。
不变量（与 InMemory 版逐字一致）：商家不存在 → 返回 ``None``（确定性「无结果」，由工具转
``ok=False``，不抛）；基础设施异常（连不上库 / SQL 报错）**一律向上抛**，绝不吞成 ``None``
—— 把「查不到」伪装成「证明无」是本项目的业务红线。

``window_days`` 不参与查询：库中存的是数据源侧预计算的固定窗口快照（见 ``MerchantRepository``
docstring 的口径说明）。按墙钟重算会让同一案件随运行时间改变结果，破坏可重放。

坑：构造期与 import 期都不建 engine / 不连库（engine 经 ``get_sessionmaker`` 懒加载，首次
``get_profile`` 才建立）；async engine 绑定创建它的 event loop，跨 loop 复用会报
``attached to a different loop``（见 ``pra.infra.db``）。装配入口：``build_tools(merchant_repo=…)``。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from sqlalchemy import String, select
from sqlalchemy.dialects.mysql import BIGINT, DATETIME, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from ...infra.db import get_sessionmaker
from .tool import MerchantEvent, MerchantProfile, MerchantViolations

__all__ = [
    "EVENT_TS_FORMAT",
    "MerchantEventORM",
    "MerchantORM",
    "MySQLMerchantRepository",
    "to_profile",
]


# ---------------------------------------------------------------------------
# 商家 2 表 ORM（独立 DeclarativeBase：每个工具子包自持映射，勿跨包混用）
# ---------------------------------------------------------------------------


class _MerchantBase(DeclarativeBase):
    """商家 2 表专属 metadata 归属（与审核 5 表、商品 3 表的 Base 相互独立）。"""


class MerchantORM(_MerchantBase):
    """``merchant`` 行 —— 每商家一行画像（预计算固定窗口快照，不按 ``window_days`` 重算）。

    ``violations_by_type`` 无违规为 SQL NULL 或空对象，映射层统一归一为 ``{}``。

    刻意不声明 ``relationship``：取事件走显式查询 —— 避免隐式懒加载在 ``async`` 会话里抛
    ``MissingGreenlet``，也不引入双向导航。
    """

    __tablename__ = "merchant"

    merchant_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    product_total: Mapped[int] = mapped_column(nullable=False)
    similar_product_count: Mapped[int] = mapped_column(nullable=False)
    removals: Mapped[int] = mapped_column(nullable=False)
    title_relisting_count: Mapped[int] = mapped_column(nullable=False)
    violations_total: Mapped[int] = mapped_column(nullable=False)
    violations_by_type: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    credit_score: Mapped[int] = mapped_column(nullable=False)


class MerchantEventORM(_MerchantBase):
    """``merchant_event`` 行（1 商家 N 事件；``sort_order`` 是读出顺序键，SQL 结果本身无序）。"""

    __tablename__ = "merchant_event"

    event_id: Mapped[int] = mapped_column(BIGINT, primary_key=True, autoincrement=True)
    merchant_id: Mapped[str] = mapped_column(String(64), nullable=False)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    ts: Mapped[object] = mapped_column(DATETIME(fsp=3), nullable=False)
    sort_order: Mapped[int] = mapped_column(nullable=False)


# ---------------------------------------------------------------------------
# 行 → 画像（纯函数：不连库即可测全部字段边界）
# ---------------------------------------------------------------------------

EVENT_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def _format_event_ts(value: object) -> str:
    """``DATETIME(3)`` 读出的 datetime → ``MerchantEvent.ts`` 的 ISO8601 展示串口径。

    输出恒带 ``Z``（库中存 naive UTC）—— 与 InMemory 种子逐字一致；非 datetime（驱动/替身给
    字符串）原样 ``str()`` 兜底。
    """
    if isinstance(value, datetime):
        return value.strftime(EVENT_TS_FORMAT)
    return str(value)


def to_profile(merchant: MerchantORM, events: list[MerchantEventORM]) -> MerchantProfile:
    """DB 行 → ``MerchantProfile``：只做类型/形态归一，不做业务判定。"""
    return MerchantProfile(
        merchant_id=merchant.merchant_id,
        product_total=int(merchant.product_total),
        similar_product_count=int(merchant.similar_product_count),
        removals=int(merchant.removals),
        title_relisting_count=int(merchant.title_relisting_count),
        violations=MerchantViolations(
            total=int(merchant.violations_total),
            by_type=dict(merchant.violations_by_type or {}),  # NULL / {} 均归一为 {}
        ),
        credit_score=int(merchant.credit_score),
        recent_events=[
            MerchantEvent(event_type=e.event_type, ts=_format_event_ts(e.ts)) for e in events
        ],
    )


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------

# () -> async_sessionmaker 的提供者；默认 get_sessionmaker，测试可注入返回假 sessionmaker 的替身。
SessionFactory = Callable[[], Any]


class MySQLMerchantRepository:
    """``MerchantRepository`` 的 MySQL 实现（**显式 opt-in**，默认装配路径不用它）。

    ``sessionmaker_factory`` 是 ``() -> async_sessionmaker`` 的**提供者**，默认
    ``pra.infra.db.get_sessionmaker``（进程级懒加载单例）。构造期不调用它 —— engine 到首次
    ``get_profile`` 才建立，故 import / 构造无副作用。注入替身（返回假 sessionmaker）即可在
    无库环境单测查询编排与边界。

    ``window_days`` 只作调用方语义声明，不参与查询（库中存预计算的固定窗口快照）。
    """

    def __init__(self, sessionmaker_factory: SessionFactory = get_sessionmaker) -> None:
        self._sessionmaker_factory = sessionmaker_factory

    async def get_profile(self, merchant_id: str, window_days: int) -> MerchantProfile | None:
        """取该商家的行为画像。

        不存在 → ``None``；基础设施异常不捕获，直接抛出（见模块 docstring 红线）。
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
            events = (
                (
                    await session.execute(
                        select(MerchantEventORM)
                        .where(MerchantEventORM.merchant_id == merchant_id)
                        .order_by(MerchantEventORM.sort_order, MerchantEventORM.event_id)
                    )
                )
                .scalars()
                .all()
            )
            # 在 session 关闭前物料化：避免依赖 detached 实例的属性访问语义
            return to_profile(merchant, list(events))
