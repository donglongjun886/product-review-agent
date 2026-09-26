"""检索契约：Policy / Case 两路的过滤条件、命中结果与索引接口。"""

from __future__ import annotations

from datetime import date
from typing import Protocol

from pydantic import BaseModel, Field

from pra.domain.models import Decision, RiskLevel, RiskType

__all__ = [
    "CaseHit",
    "CaseIndex",
    "CaseSearchFilters",
    "PolicyClauseHit",
    "PolicyIndex",
    "PolicySearchFilters",
]


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


class PolicySearchFilters(BaseModel):

    category: str | None = Field(default=None, description="类目过滤，如 女鞋/运动鞋")
    risk_type: list[RiskType] | None = Field(default=None, description="风险类型过滤（受控词表）")


class PolicyClauseHit(BaseModel):
    """单个政策条款命中（``status`` 取 EFFECTIVE / EXPIRED）。"""

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
