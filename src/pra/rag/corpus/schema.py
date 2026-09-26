"""RAG corpus 数据契约（Pydantic 强校验）：``PolicyClauseRecord`` / ``CasePrecedentRecord`` 与顶层信封。"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pra.domain.models import Decision, RiskLevel, RiskType

ClauseStatus = Literal["EFFECTIVE", "EXPIRED"]


class _RowModel(BaseModel):

    model_config = ConfigDict(extra="forbid")


class PolicyClauseRecord(_RowModel):

    policy_id: str = Field(description="政策编号，如 POLICY_1.2")
    version: int = Field(description="版本号（policy_id + version 唯一标识一次修订）")
    clause_id: str = Field(description="条款 ID（= policy_id + version + 序号的扁平引用键）")
    title: str = Field(description="条款标题（人读/检索标签）")
    text: str = Field(description="条款原文（检索/分块最小单元）")
    category: str = Field(description="适用类目：具体类目或 全类目")
    risk_type: list[RiskType] = Field(default_factory=list, description="风险类型（受控词表）")
    status: ClauseStatus = Field(default="EFFECTIVE", description="EFFECTIVE / EXPIRED")
    effective_date: date | None = Field(default=None, description="生效日期（ISO date）")


class PolicyCorpus(_RowModel):

    meta: dict = Field(default_factory=dict, description="来源/隔离声明等元数据（人读）")
    policies: list[PolicyClauseRecord] = Field(default_factory=list)


class CasePrecedentRecord(_RowModel):
    case_id: str = Field(description="先例主键（脱敏；RAG_CASE_ 前缀，与 eval GT 隔离）")
    category: str = Field(description="商品类目（元数据过滤用）")
    decision: Decision = Field(description="人工裁决：PASS / REJECT / HUMAN_REVIEW")
    risk_level: RiskLevel = Field(default=RiskLevel.NONE, description="风险等级")
    risk_type: list[RiskType] = Field(default_factory=list, description="风险类型（受控词表）")
    summary: str = Field(description="案件摘要（检索文本，脱敏）")
    key_evidence: list[str] = Field(default_factory=list, description="关键证据摘要，如 image_similarity>=0.85")
    policy_refs: list[str] = Field(default_factory=list, description="适用政策条款/政策编号")


class CaseCorpus(_RowModel):

    meta: dict = Field(default_factory=dict, description="来源/隔离声明等元数据（人读）")
    cases: list[CasePrecedentRecord] = Field(default_factory=list)


__all__ = [
    "CaseCorpus",
    "CasePrecedentRecord",
    "ClauseStatus",
    "PolicyClauseRecord",
    "PolicyCorpus",
]
