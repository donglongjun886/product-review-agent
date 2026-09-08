"""CaseSearchTool —— 先例检索工具（RAG · Case KB）（docs/01-agent-loop.md §5.5 /《00》§5）。

回答的业务问题：有没有类似且已有人工裁决的先例？结论是什么 —— 提供决策参照，
是 REJECT 的**可引用依据**来源之一（§5.5 /《00》§7.2.2）。

分层（依赖倒置）：

- ``CaseIndex``（Protocol）：窄接口 —— 混合检索（BM25 + 向量融合、元数据过滤、
  粗召回 Top-50 → rerank Top-K，《00》§6.3）的查询面。返回按相似度降序的命中。
- ``InMemoryCaseIndex``：**Mock 默认实现**（显式标注，仅供开发/测试/演示）。
  只做元数据过滤（category / risk_type）+ 按种子 similarity 排序截断，**不做真实
  语义检索**（真实 BM25/向量/rerank 在 rag 阶段接入，替换同一接口）。

本工具不含业务判定：每个 hit → 1 条 CASE_PRECEDENT 证据（weight=similarity、
ref_id=case_id 必填，可追溯），是否采信先例归 reevaluate/decide。
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from ...domain.models import Decision, Evidence, RiskLevel, RiskType
from ..base import ToolArgs, ToolContext, ToolResult

# ---- §5.7 受控证据类型 ----
CASE_PRECEDENT_TYPE = "CASE_PRECEDENT"


# ---------------------------------------------------------------------------
# 数据访问契约（依赖倒置）
# ---------------------------------------------------------------------------


class CaseSearchFilters(BaseModel):
    """Case KB 元数据过滤（§5.5 args.filters）。"""

    category: str | None = Field(default=None, description="类目过滤，如 女鞋/运动鞋")
    risk_type: list[RiskType] | None = Field(default=None, description="风险类型过滤（受控词表）")


class CaseHit(BaseModel):
    """单个先例命中（§5.5 result.data.hits[] 元素；DB ``case_precedent``）。"""

    case_id: str = Field(description="回案库引用主键（脱敏文本只含摘要，§5.5）")
    similarity: float = Field(ge=0.0, le=1.0, description="与查询的相似度")
    decision: Decision = Field(description="人工裁决：PASS / REJECT / HUMAN_REVIEW")
    risk_level: RiskLevel = Field(default=RiskLevel.NONE)
    risk_type: list[RiskType] = Field(default_factory=list)
    summary: str = Field(default="", description="案件摘要（检索文本，脱敏）")
    key_evidence: list[str] = Field(default_factory=list, description="关键证据摘要，如 image_similarity>=0.85")
    policy_refs: list[str] = Field(default_factory=list, description="适用政策，如 POLICY_3.2")


class CaseIndex(Protocol):
    """案例库检索窄接口。无命中返回空列表（合法结果，工具 ok=True）。"""

    async def search(self, query: str, filters: CaseSearchFilters, top_k: int) -> list[CaseHit]: ...


_DEFAULT_PRECEDENTS: list[dict[str, Any]] = [
    {
        "case_id": "CASE_1832",
        "similarity": 0.86,
        "decision": "REJECT",
        "risk_level": "HIGH",
        "risk_type": ["POTENTIAL_IP_RISK"],
        "summary": "无品牌标识 + 外观高度模仿知名品牌复古跑鞋 + 商家多次改标题重上架",
        "key_evidence": ["image_similarity>=0.85", "merchant_history>=5_removals"],
        "policy_refs": ["POLICY_3.2"],
        "category": "女鞋/运动鞋",
    },
    {
        "case_id": "CASE_0911",
        "similarity": 0.31,
        "decision": "PASS",
        "risk_level": "NONE",
        "risk_type": [],
        "summary": "普通休闲鞋，无品牌标识且无相似外观",
        "key_evidence": [],
        "policy_refs": [],
        "category": "女鞋/运动鞋",
    },
]


class InMemoryCaseIndex:
    """CaseIndex 的 Mock 默认实现（显式标注，仅供开发/测试/演示）。

    检索 = 元数据过滤（category 精确 / risk_type 交叠）+ 按种子 similarity 降序 +
    top_k 截断；**query 不参与匹配**（种子相似度即最终排序值），真实语义检索待
    rag 阶段实现同一接口。
    """

    def __init__(self, precedents: list[dict[str, Any]] | None = None) -> None:
        # 保留原始行（含元数据过滤字段 category），检索过滤后再校验为 CaseHit。
        self._rows: list[dict[str, Any]] = list(precedents or _DEFAULT_PRECEDENTS)

    async def search(self, query: str, filters: CaseSearchFilters, top_k: int) -> list[CaseHit]:
        rows = self._rows
        if filters.category:
            rows = [r for r in rows if r.get("category") == filters.category]
        if filters.risk_type:
            wanted = set(filters.risk_type)
            rows = [r for r in rows if wanted & set(r.get("risk_type", []))]
        ranked = sorted(rows, key=lambda r: r.get("similarity", 0.0), reverse=True)
        return [CaseHit.model_validate(r) for r in ranked[:top_k]]


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class CaseSearchArgs(ToolArgs):
    """CaseSearchTool 入参（§5.5 args Schema）。"""

    query: str = Field(description="自然语言或结构化检索描述，如 '无品牌标识+外观高度模仿+商家多次重上架'")
    filters: CaseSearchFilters = Field(default_factory=CaseSearchFilters)
    top_k: int = Field(default=5, ge=1, le=10)


class CaseSearchResult(ToolResult):
    """CaseSearchTool 出参信封 + 负载（§5.5 result.data.hits）。"""

    hits: list[CaseHit] = Field(default_factory=list, description="按相似度降序的 Top-K 先例")


class CaseSearchTool:
    """检索历史人工裁决的相似案件（先例），返回 Top-K 相似案例及其决策/风险类型/关键证据/适用政策。"""

    name = "CaseSearchTool"
    description = "检索历史人工裁决的相似案件（先例），返回 Top-K 相似案例及其决策/风险类型/关键证据/适用政策"

    def __init__(self, index: CaseIndex | None = None) -> None:
        self._index: CaseIndex = index or InMemoryCaseIndex()

    async def call(self, args: CaseSearchArgs, ctx: ToolContext) -> CaseSearchResult:
        hits = await self._index.search(args.query, args.filters, top_k=args.top_k)
        return CaseSearchResult(hits=hits)

    def to_evidence(self, result: CaseSearchResult) -> list[Evidence]:
        """结果 → Evidence（§5.5 → Evidence 列）：每个 hit 1 条 CASE_PRECEDENT。

        weight=similarity（相似度即证据强度）；``ref_id=case_id`` **必填**
        （RAG 可追溯引用，§5.5 /《00》§6.3）；``policy_refs`` 等信息留待后续
        Evidence.extra（契约扩展）承载，value 只给决策参照摘要。
        """
        evidences: list[Evidence] = []
        for h in result.hits:
            risk_types = "/".join(t.value for t in h.risk_type) or "风险类型未标注"
            evidences.append(
                Evidence(
                    type=CASE_PRECEDENT_TYPE,
                    source=self.name,
                    value=f"{h.case_id} 高度相似 → {h.decision.value}（{risk_types}）",
                    weight=h.similarity,
                    ref_id=h.case_id,
                )
            )
        return evidences
