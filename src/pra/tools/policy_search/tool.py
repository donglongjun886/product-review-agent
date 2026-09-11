"""PolicySearchTool —— 政策依据检索工具（RAG · Policy KB）。

回答的业务问题：当前有效政策对这类情况怎么说 —— 决定「能不能判、判到什么程度」，是
REJECT/HUMAN_REVIEW 的**可引用依据**来源。

``PolicyIndex`` 是窄接口（政策库检索的查询面）；``InMemoryPolicyIndex`` 是 **Mock 默认实现**，
只做版本有效性 + 元数据过滤与种子排序截断，**不做真实语义检索**。生产/HTTP 入口
（``pra.tools.build_production_tools()``）注入真实 RAG 索引（chroma + BGE + hybrid，装配期
惰性、首次检索才建库）；默认与评测世界仍是 ``InMemoryPolicyIndex``。本工具不含业务判定：每个 hit
→ 1 条 POLICY_REF 证据（``weight=0.9``、``ref_id=clause_id`` 必填），政策是否适用归 reevaluate/decide。
"""

from __future__ import annotations

from datetime import date
from typing import Any, Protocol

from pydantic import BaseModel, Field

from ...domain.models import Evidence, RiskType
from ..base import ToolArgs, ToolContext, ToolResult

# ---- 受控证据类型 & 默认证据强度 ----
POLICY_REF_TYPE = "POLICY_REF"
POLICY_REF_WEIGHT = 0.9  # 默认权重（暂定默认，可调）
POLICY_TEXT_MAX_CHARS = 120  # value 内嵌条款原文的截断上限（全文在 Result/审计）


# ---------------------------------------------------------------------------
# 数据访问契约（依赖倒置）
# ---------------------------------------------------------------------------


class PolicySearchFilters(BaseModel):

    category: str | None = Field(default=None, description="类目过滤，如 女鞋/运动鞋")
    risk_type: list[RiskType] | None = Field(default=None, description="风险类型过滤（受控词表）")


class PolicyClauseHit(BaseModel):
    """单个政策条款命中（DB ``policy / policy_clause``）。

    ``status`` 取 "EFFECTIVE"（生效）/ "EXPIRED"（失效）；``effective_date`` 为生效日期
    （ISO date）。``version`` 用于 policy_id + version 的唯一引用。
    """

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


_DEFAULT_CLAUSES: list[dict[str, Any]] = [
    {
        "policy_id": "POLICY_3.2",
        "version": 2,
        "clause_id": "POLICY_3.2_v2_c1",
        "title": "外观高度模仿知名品牌",
        "text": "商品外观高度模仿知名品牌设计且无品牌授权，判定为高风险，转人工审核处理",
        "category": "女鞋/运动鞋",
        "risk_type": ["POTENTIAL_IP_RISK"],
        "status": "EFFECTIVE",
        "effective_date": "2024-03-01",
    },
    {
        "policy_id": "POLICY_3.1",
        "version": 1,
        "clause_id": "POLICY_3.1_v1_c2",
        "title": "品牌词滥用（旧版）",
        "text": "标题/描述不得出现未授权品牌词（旧版，已失效）",
        "category": "全类目",
        "risk_type": ["POTENTIAL_IP_RISK"],
        "status": "EXPIRED",
        "effective_date": "2023-01-01",
    },
]


class InMemoryPolicyIndex:
    """PolicyIndex 的 Mock 默认实现（仅供开发/测试/演示）。

    检索 = 版本有效性过滤（effective_only 时只留 status=EFFECTIVE）+ 元数据过滤
    （category / risk_type）+ top_k 截断；**query 不参与匹配**。
    """

    def __init__(self, clauses: list[dict[str, Any]] | None = None) -> None:
        self._rows: list[dict[str, Any]] = list(clauses or _DEFAULT_CLAUSES)

    async def search(
        self,
        query: str,
        filters: PolicySearchFilters,
        top_k: int,
        effective_only: bool,
    ) -> list[PolicyClauseHit]:
        rows = self._rows
        if effective_only:
            rows = [r for r in rows if r.get("status") == "EFFECTIVE"]
        if filters.category:
            rows = [r for r in rows if r.get("category") in (None, filters.category, "全类目")]
        if filters.risk_type:
            wanted = set(filters.risk_type)
            rows = [r for r in rows if wanted & set(r.get("risk_type", []))]
        return [PolicyClauseHit.model_validate(r) for r in rows[:top_k]]


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class PolicySearchArgs(ToolArgs):

    query: str = Field(description="政策检索描述，如 '外观高度模仿品牌设计'")
    filters: PolicySearchFilters = Field(default_factory=PolicySearchFilters)
    top_k: int = Field(default=5, ge=1, le=10)
    effective_only: bool = Field(default=True, description="只查当前生效版本（默认 true）")


class PolicySearchResult(ToolResult):

    hits: list[PolicyClauseHit] = Field(default_factory=list, description="Top-K 政策条款命中")


class PolicySearchTool:

    name = "PolicySearchTool"
    description = "检索当前有效平台政策条款（按类目/风险类型过滤），返回条款原文与版本引用"
    args_model = PolicySearchArgs

    def __init__(self, index: PolicyIndex | None = None) -> None:
        self._index: PolicyIndex = index or InMemoryPolicyIndex()

    async def call(self, args: PolicySearchArgs, ctx: ToolContext) -> PolicySearchResult:
        hits = await self._index.search(
            args.query, args.filters, top_k=args.top_k, effective_only=args.effective_only
        )
        return PolicySearchResult(hits=hits)

    def to_evidence(self, result: PolicySearchResult) -> list[Evidence]:
        """结果 → Evidence：每个 hit 1 条 POLICY_REF。

        ``weight=0.9``（政策条款为强依据）；``ref_id=clause_id`` **必填**（可追溯）；
        value 形如 ``"POLICY_3.2 v2 条款：…"``。
        """
        evidences: list[Evidence] = []
        for h in result.hits:
            text = h.text[:POLICY_TEXT_MAX_CHARS]
            evidences.append(
                Evidence(
                    type=POLICY_REF_TYPE,
                    source=self.name,
                    value=f"{h.policy_id} v{h.version} 条款：{text}",
                    weight=POLICY_REF_WEIGHT,
                    ref_id=h.clause_id,
                )
            )
        return evidences
