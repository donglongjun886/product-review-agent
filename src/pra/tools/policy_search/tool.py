"""PolicySearchTool：政策依据检索工具（RAG · Policy KB）。"""

from __future__ import annotations

from pydantic import Field

from pra.rag.dto import PolicyClauseHit, PolicyIndex, PolicySearchFilters

from ...domain.measurement import DEFAULT_EVIDENCE_WEIGHT
from ...domain.models import Evidence
from ..base import ToolArgs, ToolContext, ToolResult

# ---- 受控证据类型 ----
POLICY_REF_TYPE = "POLICY_REF"
POLICY_TEXT_MAX_CHARS = 120  # value 内嵌条款原文的截断上限


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
        """结果 → Evidence：每个 hit 1 条 ``POLICY_REF``（``ref_id=clause_id``；``policy_id`` / ``policy_version`` 进 ``extra``）。"""
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
