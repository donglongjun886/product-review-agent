"""LLM 节点结构化输出模型：四个 LLM 节点的 I/O 契约。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from pra.domain.models import Decision, HypothesisStatus, RiskLevel, RiskType

# 子结构（多个 OutputModel 复用）


class HypothesisProposal(BaseModel):
    """单条风险假设提案（hypothesize 初始 / reevaluate 新增共用）。"""

    statement: str = Field(description="假设内容，必须一句话可验证")
    prior: float = Field(ge=0.0, le=1.0, description="先验（0..1，未经调查的怀疑度）")
    evidence_hint: list[str] = Field(default_factory=list, description="想用什么证据验证（供 plan 参考，可选）")


# hypothesize


class HypothesizeOutput(BaseModel):
    """hypothesize 节点输出：初始风险假设集。"""

    hypotheses: list[HypothesisProposal] = Field(min_length=1, max_length=5)
    rationale: str = Field(default="", description="一句话说明假设来源（进审计）")


# plan


class PlannedToolCall(BaseModel):
    """单条待执行工具调用（``pending_tool_calls`` 元素）。"""

    tool: str = Field(description="工具名（4 个受控名之一，须与实际装配的工具一致）")
    args: dict = Field(default_factory=dict, description="工具入参 dict（args_model 校验/解析）")
    reason: str = Field(default="", description="为什么调（验证哪个假设→要哪条证据）")
    priority: int = Field(default=5, ge=1, le=5, description="1 最优先（同轮按 priority 升序执行）")


class PlanOutput(BaseModel):
    """plan 节点输出：调查动作建议；``next_action`` 为 ``call_tools`` / ``conclude``。"""

    next_action: Literal["call_tools", "conclude"]
    tools: list[PlannedToolCall] = Field(default_factory=list, max_length=3, description="≤3 条/轮")
    rationale: str = Field(default="", description="一句解释'验证哪个假设→要哪条证据'")


# reevaluate


class HypothesisUpdate(BaseModel):
    """单条假设状态更新，``id`` 必须命中已有假设 id。"""

    id: str = Field(description="命中已有假设 id（H1..Hn）")
    posterior: float = Field(ge=0.0, le=1.0, description="证据综合后的后验")
    status: Literal["SUPPORTED", "REFUTED", "UNRESOLVED"] = Field(description="reevaluate 不产出 PENDING（新增假设才 PENDING）")
    evidence_for: list[str] = Field(default_factory=list, description="支持本假设的证据引用/摘要（DTO 无 E_nn，用 type/value 摘要）")
    evidence_against: list[str] = Field(default_factory=list, description="反驳本假设的证据引用/摘要")


class ConflictNote(BaseModel):
    """矛盾证据对说明。"""

    between: list[str] = Field(default_factory=list, description="两个冲突证据的引用/摘要")
    description: str = Field(default="")


class ReevaluateOutput(BaseModel):
    """reevaluate 节点输出：把本批新证据综合进假设；``new_hypotheses`` 为运行中新发现的风险维度。"""

    hypothesis_updates: list[HypothesisUpdate] = Field(default_factory=list)
    new_hypotheses: list[HypothesisProposal] = Field(default_factory=list)
    evidence_sufficiency: Literal["SUFFICIENT", "INSUFFICIENT"] = "INSUFFICIENT"
    conflicts: list[ConflictNote] = Field(default_factory=list)
    rationale: str = Field(default="")


# decide


class DecisionProposal(BaseModel):
    """decide 节点 LLM 提案（Gate 校验/改写后才是终值）。"""

    decision: Literal["PASS", "REJECT", "HUMAN_REVIEW"]
    risk_level: Literal["NONE", "LOW", "MEDIUM", "HIGH"]
    risk_type: list[RiskType] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, description="提案自评把握（参考值；Gate 不读，仅观测）")
    evidence_ids: list[str] = Field(default_factory=list, description="支撑证据引用/摘要（提示存在性由 prompt 约束）")
    policy: list[str] = Field(default_factory=list, description="policy_id / clause 引用")
    rationale: str = Field(default="")


__all__ = [
    "ConflictNote",
    "Decision",
    "DecisionProposal",
    "HypothesisProposal",
    "HypothesisStatus",
    "HypothesisUpdate",
    "HypothesizeOutput",
    "PlanOutput",
    "PlannedToolCall",
    "ReevaluateOutput",
    "RiskLevel",
    "RiskType",
]
