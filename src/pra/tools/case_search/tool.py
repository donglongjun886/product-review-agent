"""CaseSearchTool：先例检索工具（RAG · Case KB）。

默认/评测世界用 ``InMemoryCaseIndex``（Mock：只做元数据过滤 + 种子分排序，``query`` 不参与匹配），
生产/HTTP 入口注入真实 RAG 索引；``retrieval_score`` 是检索分、不是语义相似度，也不做量纲适配。
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field

from ...domain.models import Decision, Evidence, RiskLevel, RiskType
from ..base import ToolArgs, ToolContext, ToolResult

# ---- 受控证据类型 ----
CASE_PRECEDENT_TYPE = "CASE_PRECEDENT"


# ---------------------------------------------------------------------------
# 数据访问契约（依赖倒置）
# ---------------------------------------------------------------------------


class CaseSearchFilters(BaseModel):

    category: str | None = Field(default=None, description="类目过滤，如 女鞋/运动鞋")
    risk_type: list[RiskType] | None = Field(default=None, description="风险类型过滤（受控词表）")


class CaseHit(BaseModel):

    case_id: str = Field(description="回案库引用主键（脱敏文本只含摘要）")
    retrieval_score: float = Field(
        description=(
            "检索分（**不是语义相似度**，不做量纲适配）：生产口径恒 hybrid = RRF 融合分 "
            "Σ1/(k+rank)；bm25s 原始分（无界）/ exp(-distance) 只是 hybrid 内部两路的原始分。"
            "取值域由检索后端定义"
        ),
    )
    decision: Decision = Field(description="人工裁决：PASS / REJECT / HUMAN_REVIEW")
    risk_level: RiskLevel = Field(default=RiskLevel.NONE)
    risk_type: list[RiskType] = Field(default_factory=list)
    summary: str = Field(default="", description="案件摘要（检索文本，脱敏）")
    key_evidence: list[str] = Field(default_factory=list, description="关键证据摘要，如 image_similarity>=0.85")
    policy_refs: list[str] = Field(default_factory=list, description="适用政策，如 POLICY_3.2")


class CaseIndex(Protocol):

    async def search(self, query: str, filters: CaseSearchFilters, top_k: int) -> list[CaseHit]: ...


_DEFAULT_PRECEDENTS: list[dict[str, Any]] = [
    {
        "case_id": "CASE_1832",
        "retrieval_score": 0.86,
        "decision": "REJECT",
        "risk_level": "HIGH",
        "risk_type": ["POTENTIAL_IP_RISK"],
        "summary": "无品牌标识 + 标题含高仿/复刻规避用语 + 商家多次改标题重上架",
        "key_evidence": ["text_evasion_word", "merchant_history>=5_removals"],
        "policy_refs": ["POLICY_3.2"],
        "category": "女鞋/运动鞋",
    },
    {
        "case_id": "CASE_0911",
        "retrieval_score": 0.31,
        "decision": "PASS",
        "risk_level": "NONE",
        "risk_type": [],
        "summary": "普通休闲鞋，无品牌标识且无规避用语",
        "key_evidence": [],
        "policy_refs": [],
        "category": "女鞋/运动鞋",
    },
]


class InMemoryCaseIndex:
    """``CaseIndex`` 的 Mock 默认实现（仅供开发/测试/演示）。

    检索 = 元数据过滤（category 精确 / risk_type 交叠）+ 种子分降序 + top_k；**query 不参与匹配**。
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
        ranked = sorted(rows, key=lambda r: r.get("retrieval_score", 0.0), reverse=True)
        return [CaseHit.model_validate(r) for r in ranked[:top_k]]


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class CaseSearchArgs(ToolArgs):

    query: str = Field(description="自然语言或结构化检索描述，如 '无品牌标识+标题含高仿用语+商家多次重上架'")
    filters: CaseSearchFilters = Field(default_factory=CaseSearchFilters)
    top_k: int = Field(default=5, ge=1, le=10)


class CaseSearchResult(ToolResult):

    hits: list[CaseHit] = Field(default_factory=list, description="按检索分降序的 Top-K 先例")


class CaseSearchTool:

    name = "CaseSearchTool"
    description = "检索历史人工裁决的相似案件（先例），返回 Top-K 相似案例及其决策/风险类型/关键证据/适用政策"
    args_model = CaseSearchArgs

    def __init__(self, index: CaseIndex | None = None) -> None:
        self._index: CaseIndex = index or InMemoryCaseIndex()

    async def call(self, args: CaseSearchArgs, ctx: ToolContext) -> CaseSearchResult:
        hits = await self._index.search(args.query, args.filters, top_k=args.top_k)
        return CaseSearchResult(hits=hits)

    def to_evidence(self, result: CaseSearchResult) -> list[Evidence]:
        """结果 → Evidence：每个 hit 1 条 ``CASE_PRECEDENT``。

        ``weight=retrieval_score``（检索分即证据强度，**不是**语义相似度）；``ref_id=case_id`` 必填。
        """
        evidences: list[Evidence] = []
        for h in result.hits:
            risk_types = "/".join(t.value for t in h.risk_type) or "风险类型未标注"
            evidences.append(
                Evidence(
                    type=CASE_PRECEDENT_TYPE,
                    source=self.name,
                    value=f"{h.case_id} 高度相似 → {h.decision.value}（{risk_types}）",
                    weight=h.retrieval_score,
                    ref_id=h.case_id,
                )
            )
        return evidences
