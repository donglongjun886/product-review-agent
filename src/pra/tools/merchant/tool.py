"""MerchantTool —— 行为模式工具。

回答的业务问题：单商品看不出问题，商家的**历史行为**才是「规避」的关键信号 —— 相似商品数、
下架/改标题重上架次数、信用分。

``MerchantRepository`` 是窄接口（按 merchant_id 取行为画像；返回 None = 商家不存在 → ``ok=False``）；
``InMemoryMerchantRepository`` 是 **Mock 默认实现**（默认装配路径恒用它，CI 不连库、评测可重放），
真实实现见 ``pra.tools.merchant.mysql_repo.MySQLMerchantRepository``。本工具不含业务判定：只
交付「取到的事实」，「违规 + 下架 + 改标题重上架是否构成规避」归 guardrails/reevaluate。
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from pydantic import BaseModel, Field

from ...domain.measurement import (
    DIM_MERCHANT_PROFILE,
    MERCHANT_DIRTY_MIN,
    VERDICT_NEGATIVE,
    VERDICT_POSITIVE,
    make_measurement,
)
from ...domain.models import Evidence
from ..base import ToolArgs, ToolContext, ToolResult

# ---- 受控证据类型 & 默认证据强度 ----
MERCHANT_HISTORY_TYPE = "MERCHANT_HISTORY"
MERCHANT_HISTORY_WEIGHT = 0.85  # 多信号聚合型证据默认高权重（暂定默认，可调）
# 「系统性规避行为」阈值单一来源：pra.domain.measurement（上面按名引入，本工具用它做测量裁决）。


# ---------------------------------------------------------------------------
# 数据访问契约（依赖倒置）
# ---------------------------------------------------------------------------


class MerchantProfile(BaseModel):
    """商家行为画像（只保留决策链真正消费的列，与迁移 004 的 ``merchant`` 表同形）。"""

    merchant_id: str
    similar_product_count: int = Field(default=0, ge=0, description="与本案相似的商品数")
    removals: int = Field(default=0, ge=0, description="窗口内下架次数")
    title_relisting_count: int = Field(default=0, ge=0, description="窗口内改标题重上架次数")
    credit_score: int = Field(default=0, ge=0, description="商家信用分")


class MerchantRepository(Protocol):
    """商家行为数据源窄接口（按 merchant_id 取行为画像）。

    不变量：商家不存在返回 ``None``（确定性「无结果」，由工具转 ``ok=False``）；基础设施异常由
    实现直接抛出，不得吞成 ``None``。

    ``window_days`` 只作调用方语义声明 —— **两个实现都返回数据源侧预计算的固定窗口快照，不按
    window_days 重算**。按墙钟重算会让同一案件随运行时间改变结果（破坏可重放），也会让真库世界
    与 InMemory 世界不等价；窗口切分属数据源侧职责（如离线物化不同窗口的聚合）。
    """
    async def get_profile(self, merchant_id: str, window_days: int) -> MerchantProfile | None: ...


_DEFAULT_MERCHANTS: Mapping[str, dict[str, Any]] = {
    "M_5512": {
        "merchant_id": "M_5512",
        "similar_product_count": 23,
        "removals": 5,
        "title_relisting_count": 3,
        "credit_score": 62,
    },
}


class InMemoryMerchantRepository:
    """MerchantRepository 的 Mock 默认实现（仅供开发/测试/演示）。

    种子画像按 merchant_id 匹配；``window_days`` 在 mock 中不改变聚合结果（真实实现按其
    截取事件窗口）。
    """

    def __init__(self, data: Mapping[str, dict[str, Any]] | None = None) -> None:
        self._store: dict[str, MerchantProfile] = {
            mid: MerchantProfile.model_validate(row) for mid, row in (data or _DEFAULT_MERCHANTS).items()
        }

    async def get_profile(self, merchant_id: str, window_days: int) -> MerchantProfile | None:
        return self._store.get(merchant_id)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class MerchantArgs(ToolArgs):

    merchant_id: str = Field(description="商家 ID，如 M_5512")
    window_days: int = Field(default=90, ge=1, le=365, description="行为统计观察窗口（天，默认 90）")


class MerchantResult(ToolResult):
    """MerchantTool 出参信封 + 负载。

    ``ok=False``（商家不存在）时 ``profile`` 为 None。
    """

    profile: MerchantProfile | None = Field(default=None, description="商家行为画像")


class MerchantTool:

    name = "MerchantTool"
    description = "查询商家的系统性行为画像：相似商品数、历史下架/改标题重上架次数、信用分"
    args_model = MerchantArgs
    measured_dimensions: frozenset[str] = frozenset({DIM_MERCHANT_PROFILE})
    measurement_available: bool = True

    def __init__(self, repo: MerchantRepository | None = None) -> None:
        self._repo: MerchantRepository = repo or InMemoryMerchantRepository()

    async def call(self, args: MerchantArgs, ctx: ToolContext) -> MerchantResult:
        profile = await self._repo.get_profile(args.merchant_id, window_days=args.window_days)
        if profile is None:
            return MerchantResult(ok=False, error=f"商家不存在: {args.merchant_id}")
        return MerchantResult(profile=profile)

    def to_evidence(self, result: MerchantResult) -> list[Evidence]:
        """结果 → Evidence：1 条聚合 ``MERCHANT_HISTORY`` + 1 条 ``MEASUREMENT``。

        value 拼成 ``"23 similar / 5 removals / 3 title-relisting, credit=62"``；
        ``ref_id=merchant_id``（稳定业务标识）；结构化数值从画像直接写入 ``extra``
        （``similar`` / ``removals`` / ``title`` / ``credit``），供 guardrails 确定性层
        读，不由下游反向解析 value。

        测量结论按 ``MERCHANT_DIRTY_MIN`` 硬阈值给出：达到阈值 → ``POSITIVE``（行为模式成立），
        否则 → ``NEGATIVE``（**画像全 0 是有效的阴性测量，不是"没查到"**）。
        **商家查无（``ok=False``）刻意不产任何证据**：那是 ``NOT_MEASURED``，与"全 0"不可混。
        """
        if not result.ok or result.profile is None:
            return []
        p = result.profile
        value = (
            f"{p.similar_product_count} similar / {p.removals} removals / "
            f"{p.title_relisting_count} title-relisting, credit={p.credit_score}"
        )
        dirty = p.removals >= MERCHANT_DIRTY_MIN or p.title_relisting_count >= MERCHANT_DIRTY_MIN
        return [
            Evidence(
                type=MERCHANT_HISTORY_TYPE,
                source=self.name,
                value=value,
                weight=MERCHANT_HISTORY_WEIGHT,
                ref_id=p.merchant_id,
                extra={
                    "similar": p.similar_product_count,
                    "removals": p.removals,
                    "title": p.title_relisting_count,
                    "credit": p.credit_score,
                },
            ),
            make_measurement(
                dimension=DIM_MERCHANT_PROFILE,
                source=self.name,
                source_ref=p.merchant_id,
                verdict=VERDICT_POSITIVE if dirty else VERDICT_NEGATIVE,
                weight=MERCHANT_HISTORY_WEIGHT,
                value=(
                    f"商家行为画像已测：removals={p.removals}, "
                    f"title-relisting={p.title_relisting_count}（阈值 {MERCHANT_DIRTY_MIN}）"
                ),
            ),
        ]
