"""MerchantTool —— 行为模式工具。"""

from __future__ import annotations

from typing import Protocol

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
MERCHANT_HISTORY_WEIGHT = 0.85


# ---------------------------------------------------------------------------
# 数据访问契约
# ---------------------------------------------------------------------------


class MerchantProfile(BaseModel):
    """商家行为画像。"""

    merchant_id: str
    similar_product_count: int = Field(default=0, ge=0, description="与本案相似的商品数")
    removals: int = Field(default=0, ge=0, description="窗口内下架次数")
    title_relisting_count: int = Field(default=0, ge=0, description="窗口内改标题重上架次数")
    credit_score: int = Field(default=0, ge=0, description="商家信用分")


class MerchantRepository(Protocol):
    """商家行为数据源窄接口（按 merchant_id 取行为画像）。

    商家不存在返回 ``None``；基础设施异常由实现直接抛出。
    """
    async def get_profile(self, merchant_id: str, window_days: int) -> MerchantProfile | None: ...


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

    def __init__(self, repo: MerchantRepository) -> None:
        self._repo: MerchantRepository = repo

    async def call(self, args: MerchantArgs, ctx: ToolContext) -> MerchantResult:
        profile = await self._repo.get_profile(args.merchant_id, window_days=args.window_days)
        if profile is None:
            return MerchantResult(ok=False, error=f"商家不存在: {args.merchant_id}")
        return MerchantResult(profile=profile)

    def to_evidence(self, result: MerchantResult) -> list[Evidence]:
        """结果 → Evidence：1 条聚合 ``MERCHANT_HISTORY`` + 1 条 ``MEASUREMENT``。

        ``ok=False`` 时不产任何证据。
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
