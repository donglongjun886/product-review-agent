"""PolicySearchTool —— 政策依据检索工具（RAG · Policy KB）（docs/01-agent-loop.md §5.6 /《00》§5）。

回答的业务问题：当前有效政策对这类情况怎么说 —— 决定"能不能判、判到什么程度"，
是 REJECT/HUMAN_REVIEW 的**可引用依据**来源（§5.6 /《00》§7.2.2）。

分层（依赖倒置）：

- ``PolicyIndex``（Protocol）：窄接口 —— 政策库检索（元数据过滤 + 版本有效性
  过滤 + 语义检索，《00》§6.3）的查询面。
- ``InMemoryPolicyIndex``：**Mock 默认实现**（显式标注，仅供开发/测试/演示）。
  做元数据/版本有效性过滤 + 按种子排序截断，**不做真实语义检索**（真实
  BM25/向量/rerank 在 rag 阶段接入，替换同一接口）。

本工具不含业务判定：每个 hit → 1 条 POLICY_REF 证据（weight=0.9、
ref_id=clause_id 必填，可追溯），政策是否适用归 reevaluate/decide。
"""

from __future__ import annotations

from datetime import date
from typing import Any, Protocol

from pydantic import BaseModel, Field

from ...domain.models import Evidence, RiskType
from ..base import ToolArgs, ToolContext, ToolResult

# ---- §5.7 受控证据类型 & §5.6 默认证据强度 ----
POLICY_REF_TYPE = "POLICY_REF"
POLICY_REF_WEIGHT = 0.9  # §5.6 默认权重；暂定默认，待 T-5 拍板后可调
POLICY_TEXT_MAX_CHARS = 120  # value 内嵌条款原文的截断上限（人读摘要用，全文在 Result/审计）


# ---------------------------------------------------------------------------
# 数据访问契约（依赖倒置）
# ---------------------------------------------------------------------------


class PolicySearchFilters(BaseModel):
    """Policy KB 元数据过滤（§5.6 args.filters）。"""

    category: str | None = Field(default=None, description="类目过滤，如 女鞋/运动鞋")
    risk_type: list[RiskType] | None = Field(default=None, description="风险类型过滤（受控词表）")


class PolicyClauseHit(BaseModel):
    """单个政策条款命中（§5.6 result.data.hits[] 元素；DB ``policy / policy_clause``）。

    ``status`` 取 "EFFECTIVE"（生效）/ "EXPIRED"（失效）；``effective_date`` 为
    生效日期（ISO date）。版本号 ``version`` 用于 policy_id + version 的唯一引用。
    """

    policy_id: str
    version: int
    clause_id: str = Field(description="条款 ID —— Policy KB 检索/分块的最小单元（§6.2）")
    title: str = Field(default="", description="条款标题")
    text: str = Field(description="条款原文")
    category: str | None = Field(default=None)
    risk_type: list[RiskType] = Field(default_factory=list)
    status: str = Field(default="EFFECTIVE", description="EFFECTIVE / EXPIRED")
    effective_date: date | None = Field(default=None)


class PolicyIndex(Protocol):
    """政策库检索窄接口。无命中返回空列表（合法结果，工具 ok=True）。"""

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
    """PolicyIndex 的 Mock 默认实现（显式标注，仅供开发/测试/演示）。

    检索 = 版本有效性过滤（effective_only 时只留 status=EFFECTIVE）+ 元数据过滤
    （category / risk_type）+ top_k 截断；**query 不参与匹配**，真实语义检索待
    rag 阶段实现同一接口。
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
    """PolicySearchTool 入参（§5.6 args Schema）。"""

    query: str = Field(description="政策检索描述，如 '外观高度模仿品牌设计'")
    filters: PolicySearchFilters = Field(default_factory=PolicySearchFilters)
    top_k: int = Field(default=5, ge=1, le=10)
    effective_only: bool = Field(default=True, description="只查当前生效版本（默认 true）")


class PolicySearchResult(ToolResult):
    """PolicySearchTool 出参信封 + 负载（§5.6 result.data.hits）。"""

    hits: list[PolicyClauseHit] = Field(default_factory=list, description="Top-K 政策条款命中")


class PolicySearchTool:
    """检索当前有效平台政策条款（按类目/风险类型过滤），返回条款原文与版本引用。"""

    name = "PolicySearchTool"
    description = "检索当前有效平台政策条款（按类目/风险类型过滤），返回条款原文与版本引用"

    def __init__(self, index: PolicyIndex | None = None) -> None:
        self._index: PolicyIndex = index or InMemoryPolicyIndex()

    async def call(self, args: PolicySearchArgs, ctx: ToolContext) -> PolicySearchResult:
        hits = await self._index.search(
            args.query, args.filters, top_k=args.top_k, effective_only=args.effective_only
        )
        return PolicySearchResult(hits=hits)

    def to_evidence(self, result: PolicySearchResult) -> list[Evidence]:
        """结果 → Evidence（§5.6 → Evidence 列）：每个 hit 1 条 POLICY_REF。

        weight=0.9（政策条款为强依据）；``ref_id=clause_id`` **必填**（可追溯，
        §5.6 /《00》§6.3 引用格式）；value 形如 "POLICY_3.2 v2 条款：…"。
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
