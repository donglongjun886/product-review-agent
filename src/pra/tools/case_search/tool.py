"""CaseSearchTool：先例检索工具（RAG · Case KB）。"""

from __future__ import annotations

from pydantic import Field

from pra.rag.dto import CaseHit, CaseIndex, CaseSearchFilters

from ...domain.models import Evidence
from ..base import ToolArgs, ToolContext, ToolResult

# ---- 受控证据类型 ----
CASE_PRECEDENT_TYPE = "CASE_PRECEDENT"


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

    def __init__(self, index: CaseIndex) -> None:
        self._index: CaseIndex = index

    async def call(self, args: CaseSearchArgs, ctx: ToolContext) -> CaseSearchResult:
        hits = await self._index.search(args.query, args.filters, top_k=args.top_k)
        return CaseSearchResult(hits=hits)

    def to_evidence(self, result: CaseSearchResult) -> list[Evidence]:
        """结果 → Evidence：每个 hit 1 条 ``CASE_PRECEDENT``（``weight=retrieval_score``、``ref_id=case_id``）。"""
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
