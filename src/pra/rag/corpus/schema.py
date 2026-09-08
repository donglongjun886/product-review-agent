"""RAG 知识库 Corpus Schema（rag/corpus/schema.py）—— 数据契约 + 强校验。

对应设计（docs/00-system-design.md §6 + rag-implementation-plan.md §3 / §5）：
- ``PolicyClauseRecord``：Policy KB 一条**条款**（检索/分块最小单元，对齐工具契约
  ``PolicyClauseHit``：policy_id + version + clause_id + title + text + category +
  risk_type + status + effective_date）。同 policy_id 的不同 version 行共存
  （旧版 status=EXPIRED / 新版 status=EFFECTIVE）→ 版本有效性过滤的测试面。
- ``CasePrecedentRecord``：Case KB 一条**先例**（对齐 ``CaseHit`` 除 similarity
  —— similarity 是检索期计算值，不入库，见 rag/retrieval.py）。
- 顶层信封（``PolicyCorpus`` / ``CaseCorpus``）：``meta`` 承载来源/隔离声明等
  人读元数据（JSON 无注释，故用 meta 字段记录）；``records`` 为数据本体。

数据来源与隔离声明（红线 R-4，勿破坏）：
- Policy KB：以电商平台商品内容治理常见政策为蓝本**手写**（品牌/IP、虚假宣传、
  类目准入/材质标识、规避行为四族），非复制任何 eval 世界种子；
- Case KB：demo 演示案剧情**改写**（P_88231/M_5512 同源剧情改写而非照抄）+
  合成先例 + 确定性 seed 程序化变体（见 scripts/build_rag_corpus.py）；
  **严禁包含 eval_data/v1、v2 的 ground-truth 案**：case_id 前缀统一
  ``RAG_CASE_``（与 eval 的 EC_* / EC_V2_* / CASE_EC_* / CASE_EC_V2_* 及 InMemory
  先例 CASE_1832 等全部无交集），剧情不与任何 eval 违规案一一对应
  （防"检索到 GT = 评测作弊"）。隔离断言见 tests/test_rag.py 与
  scripts/build_rag_corpus.py 自检。
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pra.domain.models import Decision, RiskLevel, RiskType

# status 受控取值（与 PolicyClauseHit.status 注释同口径）
ClauseStatus = Literal["EFFECTIVE", "EXPIRED"]


class _RowModel(BaseModel):
    """数据行基类：拒绝未声明字段（防 corpus 悄悄塞键导致口径漂移）。"""

    model_config = ConfigDict(extra="forbid")


class PolicyClauseRecord(_RowModel):
    """Policy KB 单条条款记录（与 PolicyClauseHit 对齐；effective_date 为生效日期）。"""

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
    """policies.json 顶层信封。"""

    meta: dict = Field(default_factory=dict, description="来源/隔离声明等元数据（人读）")
    policies: list[PolicyClauseRecord] = Field(default_factory=list)


class CasePrecedentRecord(_RowModel):
    """Case KB 单条先例记录（对齐 CaseHit 除 similarity —— 检索期计算，不入库）。"""

    case_id: str = Field(description="先例主键（脱敏；RAG_CASE_ 前缀，与 eval GT 隔离）")
    category: str = Field(description="商品类目（元数据过滤用）")
    decision: Decision = Field(description="人工裁决：PASS / REJECT / HUMAN_REVIEW")
    risk_level: RiskLevel = Field(default=RiskLevel.NONE, description="风险等级")
    risk_type: list[RiskType] = Field(default_factory=list, description="风险类型（受控词表）")
    summary: str = Field(description="案件摘要（检索文本，脱敏）")
    key_evidence: list[str] = Field(default_factory=list, description="关键证据摘要，如 image_similarity>=0.85")
    policy_refs: list[str] = Field(default_factory=list, description="适用政策条款/政策编号")


class CaseCorpus(_RowModel):
    """cases.json 顶层信封。"""

    meta: dict = Field(default_factory=dict, description="来源/隔离声明等元数据（人读）")
    cases: list[CasePrecedentRecord] = Field(default_factory=list)


__all__ = [
    "CaseCorpus",
    "CasePrecedentRecord",
    "ClauseStatus",
    "PolicyClauseRecord",
    "PolicyCorpus",
]
