"""RAG 知识库 Corpus Schema —— 数据契约 + 强校验。

``PolicyClauseRecord`` 是 Policy KB 一条**条款**（检索最小单元，字段对齐 ``PolicyClauseHit``）；
同 policy_id 的不同 version 行共存（旧版 EXPIRED / 新版 EFFECTIVE）→ 版本有效性过滤的测试面。
``CasePrecedentRecord`` 是 Case KB 一条**先例**（对齐 ``CaseHit`` 除 ``retrieval_score`` —— 那是
检索期计算值，不入库）。顶层信封（``PolicyCorpus`` / ``CaseCorpus``）的 ``meta`` 承载来源/隔离
声明等人读元数据（JSON 无注释，故用 meta 记录）。

**严禁包含 eval 的 ground-truth 案**：case_id 前缀统一 ``RAG_CASE_``，与 eval 各案号无交集，
剧情不与任何 eval 违规案一一对应（防「检索到 GT = 评测作弊」）。
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pra.domain.models import Decision, RiskLevel, RiskType

# status 受控取值（与 PolicyClauseHit 同口径）
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
