"""MerchantTool —— 行为模式工具（docs/01-agent-loop.md §5.4 /《00》§5）。

回答的业务问题：单商品看不出问题，商家的**历史行为**才是"规避"的关键信号 ——
相似商品数、违规/下架/改标题重上架次数、信用分（《00》§5.1）。

分层（依赖倒置）：

- ``MerchantRepository``（Protocol）：窄接口 —— 按 merchant_id + 观察窗口取商家
  行为画像。返回 None = 商家不存在（确定性无结果 → ok=False）。
- ``InMemoryMerchantRepository``：**Mock 默认实现**（显式标注，仅供开发/测试/
  演示），种子对齐《00》§4.4 走查商家 M_5512（23 similar / 5 removals /
  3 title-relisting / credit 62）。真实实现 = MySQL 聚合 + 向量扫描（§5.4），
  待 infra 阶段接入。

本工具不含业务判定：只聚合"取到事实"（结果即画像字段），"违规+下架+改标题
重上架组合是否构成规避行为"（§5.4 extra.signals）的判定归 guardrails/reevaluate。
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from pydantic import BaseModel, Field

from ...domain.models import Evidence
from ..base import ToolArgs, ToolContext, ToolResult

# ---- §5.7 受控证据类型 & §5.4 默认证据强度 ----
MERCHANT_HISTORY_TYPE = "MERCHANT_HISTORY"
MERCHANT_HISTORY_WEIGHT = 0.85  # §5.4：多信号聚合型证据默认高权重；暂定默认，待 T-5 拍板后可调


# ---------------------------------------------------------------------------
# 数据访问契约（依赖倒置）
# ---------------------------------------------------------------------------


class MerchantEvent(BaseModel):
    """商家行为事件（§5.4 recent_events[] 元素；DB ``merchant_event``）。"""

    event_type: str = Field(description="事件类型：违规 / 下架 / 改标题重上架 等")
    ts: str = Field(description="事件时间 ISO8601")


class MerchantViolations(BaseModel):
    """违规统计（§5.4 violations）。"""

    total: int = Field(default=0, ge=0)
    by_type: dict[str, int] = Field(default_factory=dict, description="按违规类型计数")


class MerchantProfile(BaseModel):
    """商家行为画像（§5.4 result.data 完整形状）。"""

    merchant_id: str
    product_total: int = Field(default=0, ge=0, description="在架商品总数")
    similar_product_count: int = Field(default=0, ge=0, description="与本案相似的商品数")
    removals: int = Field(default=0, ge=0, description="窗口内下架次数")
    title_relisting_count: int = Field(default=0, ge=0, description="窗口内改标题重上架次数")
    violations: MerchantViolations = Field(default_factory=MerchantViolations)
    credit_score: int = Field(default=0, ge=0, description="商家信用分")
    recent_events: list[MerchantEvent] = Field(default_factory=list, max_length=20, description="最近事件（≤20 条）")


class MerchantRepository(Protocol):
    """商家行为数据源窄接口。

    ``window_days`` 为聚合观察窗口；实现须返回窗口内统计。商家不存在返回 None。
    """

    async def get_profile(self, merchant_id: str, window_days: int) -> MerchantProfile | None: ...


_DEFAULT_MERCHANTS: Mapping[str, dict[str, Any]] = {
    "M_5512": {
        "merchant_id": "M_5512",
        "product_total": 120,
        "similar_product_count": 23,
        "removals": 5,
        "title_relisting_count": 3,
        "violations": {"total": 2, "by_type": {"IP_MIMIC": 1, "FALSE_CLAIM": 1}},
        "credit_score": 62,
        "recent_events": [
            {"event_type": "改标题重上架", "ts": "2024-09-01T10:00:00Z"},
            {"event_type": "下架", "ts": "2024-08-20T09:00:00Z"},
        ],
    },
}


class InMemoryMerchantRepository:
    """MerchantRepository 的 Mock 默认实现（显式标注，仅供开发/测试/演示）。

    种子画像按 merchant_id 匹配；``window_days`` 在 mock 中不改变聚合结果
    （真实实现按其截取事件窗口）。
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
    """MerchantTool 入参（§5.4 args Schema）。"""

    merchant_id: str = Field(description="商家 ID，如 M_5512")
    window_days: int = Field(default=90, ge=1, le=365, description="行为统计观察窗口（天，默认 90）")


class MerchantResult(ToolResult):
    """MerchantTool 出参信封 + 负载（§5.4 result.data）。

    ``ok=False``（商家不存在）时 ``profile`` 为 None。
    """

    profile: MerchantProfile | None = Field(default=None, description="商家行为画像")


class MerchantTool:
    """查询商家的系统性行为画像：在架商品数、相似商品数、历史违规/下架/改标题重上架次数、信用分。"""

    name = "MerchantTool"
    description = "查询商家的系统性行为画像：在架商品数、相似商品数、历史违规/下架/改标题重上架次数、信用分"

    def __init__(self, repo: MerchantRepository | None = None) -> None:
        self._repo: MerchantRepository = repo or InMemoryMerchantRepository()

    async def call(self, args: MerchantArgs, ctx: ToolContext) -> MerchantResult:
        profile = await self._repo.get_profile(args.merchant_id, window_days=args.window_days)
        if profile is None:
            return MerchantResult(ok=False, error=f"商家不存在: {args.merchant_id}")
        return MerchantResult(profile=profile)

    def to_evidence(self, result: MerchantResult) -> list[Evidence]:
        """结果 → Evidence（§5.4 → Evidence 列）：1 条聚合 MERCHANT_HISTORY。

        value 按 §5.4 示例形态拼装（"23 similar / 5 removals / 3 title-relisting,
        credit=62"）；ref_id 留空（非 RAG）；"规避行为模式"的判定（§5.4
        extra.signals）归确定性 guardrails，本工具只交付画像事实。
        """
        if not result.ok or result.profile is None:
            return []
        p = result.profile
        value = (
            f"{p.similar_product_count} similar / {p.removals} removals / "
            f"{p.title_relisting_count} title-relisting, credit={p.credit_score}"
        )
        return [
            Evidence(
                type=MERCHANT_HISTORY_TYPE,
                source=self.name,
                value=value,
                weight=MERCHANT_HISTORY_WEIGHT,
                ref_id=None,
            )
        ]
