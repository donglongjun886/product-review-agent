"""PolicySearchTool：政策依据检索工具（RAG · Policy KB）。

生产/HTTP 入口**构造时必填**注入真实 RAG 索引（测试世界见 ``tests/inmemory_world.py``）；
本工具不含业务判定，每个 hit → 1 条 ``POLICY_REF`` 证据。
"""

from __future__ import annotations

from datetime import date
from typing import Protocol

from pydantic import BaseModel, Field

from ...domain.measurement import DEFAULT_EVIDENCE_WEIGHT
from ...domain.models import Evidence, RiskType
from ..base import ToolArgs, ToolContext, ToolResult

# ---- 受控证据类型 ----
POLICY_REF_TYPE = "POLICY_REF"
POLICY_TEXT_MAX_CHARS = 120  # value 内嵌条款原文的截断上限（全文在 Result）


# ---------------------------------------------------------------------------
# 数据访问契约（依赖倒置）
# ---------------------------------------------------------------------------


class PolicySearchFilters(BaseModel):

    category: str | None = Field(default=None, description="类目过滤，如 女鞋/运动鞋")
    risk_type: list[RiskType] | None = Field(default=None, description="风险类型过滤（受控词表）")


class PolicyClauseHit(BaseModel):
    """单个政策条款命中（DB ``policy / policy_clause``）；``status`` 取 "EFFECTIVE" / "EXPIRED"。"""

    policy_id: str
    version: int
    clause_id: str = Field(description="条款 ID —— Policy KB 检索/分块的最小单元")
    title: str = Field(default="", description="条款标题")
    text: str = Field(description="条款原文")
    category: str | None = Field(default=None)
    risk_type: list[RiskType] = Field(default_factory=list)
    status: str = Field(default="EFFECTIVE", description="EFFECTIVE / EXPIRED")
    effective_date: date | None = Field(default=None)


class PolicyIndex(Protocol):

    async def search(
        self,
        query: str,
        filters: PolicySearchFilters,
        top_k: int,
        effective_only: bool,
    ) -> list[PolicyClauseHit]: ...


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class PolicySearchArgs(ToolArgs):

    query: str = Field(description="政策检索描述，如 '标题使用仿冒规避用语'")
    filters: PolicySearchFilters = Field(default_factory=PolicySearchFilters)
    top_k: int = Field(default=5, ge=1, le=10)
    effective_only: bool = Field(default=True, description="只查当前生效版本（默认 true）")


class PolicySearchResult(ToolResult):

    hits: list[PolicyClauseHit] = Field(default_factory=list, description="Top-K 政策条款命中")


class PolicySearchTool:

    name = "PolicySearchTool"
    description = "检索当前有效平台政策条款（按类目/风险类型过滤），返回条款原文与版本引用"
    args_model = PolicySearchArgs

    def __init__(self, index: PolicyIndex) -> None:
        self._index: PolicyIndex = index

    async def call(self, args: PolicySearchArgs, ctx: ToolContext) -> PolicySearchResult:
        hits = await self._index.search(
            args.query, args.filters, top_k=args.top_k, effective_only=args.effective_only
        )
        return PolicySearchResult(hits=hits)

    def to_evidence(self, result: PolicySearchResult) -> list[Evidence]:
        """结果 → Evidence：每个 hit 1 条 ``POLICY_REF``。

        ``ref_id=clause_id`` 必填（可追溯）；``policy_id`` / ``policy_version`` 写入 ``extra`` 供 Gate 读引用。
        """
        evidences: list[Evidence] = []
        for h in result.hits:
            text = h.text[:POLICY_TEXT_MAX_CHARS]
            evidences.append(
                Evidence(
                    type=POLICY_REF_TYPE,
                    source=self.name,
                    value=f"{h.policy_id} v{h.version} 条款：{text}",
                    weight=DEFAULT_EVIDENCE_WEIGHT,
                    ref_id=h.clause_id,
                    extra={"policy_id": h.policy_id, "policy_version": h.version},
                )
            )
        return evidences
