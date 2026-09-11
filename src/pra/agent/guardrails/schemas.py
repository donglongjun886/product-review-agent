"""LLM 节点结构化输出模型：四个 LLM 节点的 I/O 契约。

字段复用 domain 的受控词表（``HypothesisStatus`` / ``RiskType`` / ``Decision`` /
``RiskLevel`` 枚举），pydantic 校验保证 LLM 输出只能在词表内取值。**LLM 不写 id**：
hypotheses 的 H1..Hn 序号由 hypothesize 节点 apply 生成，证据序号由 tools_node 分配。

llm_shell 用 ``model_validate_json`` 强校验；gate 的 overlay 只消费
``DecisionProposal``。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from pra.domain.models import Decision, HypothesisStatus, RiskLevel, RiskType

# 子结构（被多个 OutputModel 复用）


class HypothesisProposal(BaseModel):
    """单条风险假设提案（hypothesize 初始 / reevaluate 新增共用）。

    ``prior`` 是未经调查的先验怀疑度（0..1）。
    """

    statement: str = Field(description="假设内容，必须一句话可验证")
    prior: float = Field(ge=0.0, le=1.0, description="先验（0..1，未经调查的怀疑度）")
    evidence_hint: list[str] = Field(default_factory=list, description="想用什么证据验证（供 plan 参考，可选）")


class QueueProposal(BaseModel):
    """初始调查问题项（``investigation_queue`` 元素）。"""

    q: str = Field(description="待验证问题")
    priority: int = Field(ge=1, le=5, description="1 最优先")


# hypothesize


class HypothesizeOutput(BaseModel):
    """hypothesize 节点输出：初始风险假设集 + 调查问题队列。

    schema 只保证 ``hypotheses`` 非空；超上限（5 / 8）由节点 apply 按 prior 截断。
    """

    hypotheses: list[HypothesisProposal] = Field(min_length=1, max_length=5)
    investigation_queue: list[QueueProposal] = Field(default_factory=list, max_length=8)
    rationale: str = Field(default="", description="一句话说明假设来源（进审计）")


# plan


class PlannedToolCall(BaseModel):
    """单条待执行工具调用（``pending_tool_calls`` 元素）。

    ``args`` 由 tools_node 经 ``ToolRegistry.parse_args`` 用该工具的 ``args_model``
    确定性校验。
    """

    tool: str = Field(description="工具名（6 个受控名之一，须与 ToolRegistry 一致）")
    args: dict = Field(default_factory=dict, description="工具入参 dict（args_model 校验/解析）")
    reason: str = Field(default="", description="为什么调（验证哪个假设→要哪条证据）")
    priority: int = Field(default=5, ge=1, le=5, description="1 最优先（同轮按 priority 升序执行）")


class PlanOutput(BaseModel):
    """plan 节点输出：本轮调查的动作建议。

    ``next_action``：``call_tools``（tools 非空）｜``conclude``（tools 为空）。
    """

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


class QueueUpdate(BaseModel):
    """调查队列项状态变更（命中已有问题原文）。"""

    q: str
    status: Literal["OPEN", "DONE"]


class ConflictNote(BaseModel):
    """矛盾证据对说明（供 HUMAN_REVIEW 参考，权威判定在 ``contradiction_detect``）。"""

    between: list[str] = Field(default_factory=list, description="两个冲突证据的引用/摘要")
    description: str = Field(default="")


class ReevaluateOutput(BaseModel):
    """reevaluate 节点输出：把本批新证据综合进假设。

    ``new_hypotheses`` 是运行中新发现的风险维度（追加为 PENDING）。
    ``evidence_sufficiency`` 只给路由参考，**非路由权威**。
    """

    hypothesis_updates: list[HypothesisUpdate] = Field(default_factory=list)
    queue_updates: list[QueueUpdate] = Field(default_factory=list)
    new_hypotheses: list[HypothesisProposal] = Field(default_factory=list)
    evidence_sufficiency: Literal["SUFFICIENT", "INSUFFICIENT"] = "INSUFFICIENT"
    conflicts: list[ConflictNote] = Field(default_factory=list)
    rationale: str = Field(default="")


# decide


class DecisionProposal(BaseModel):
    """decide 节点 LLM 提案 —— 只产**提案**，Gate 校验/改写后才是终值。

    ``confidence`` 是参考值，落库终值由 overlay 重算；``policy`` 只能填证据中真实
    出现的 policy_id/条款。
    """

    decision: Literal["PASS", "REJECT", "HUMAN_REVIEW"]
    risk_level: Literal["NONE", "LOW", "MEDIUM", "HIGH"]
    risk_type: list[RiskType] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, description="decision_confidence 提案值（overlay 重算为准）")
    evidence_ids: list[str] = Field(default_factory=list, description="支撑证据引用/摘要（提示存在性由 prompt 约束）")
    policy: list[str] = Field(default_factory=list, description="policy_id / clause 引用")
    rationale: str = Field(default="")


# 便于 gate/decide 使用词表（避免每处 import domain 枚举）。
__all__ = [
    "ConflictNote",
    "DecisionProposal",
    "HypothesisProposal",
    "HypothesisUpdate",
    "HypothesizeOutput",
    "PlanOutput",
    "PlannedToolCall",
    "QueueProposal",
    "QueueUpdate",
    "ReevaluateOutput",
    # 复导出（供节点 apply / gate 使用）
    "Decision",
    "HypothesisStatus",
    "RiskLevel",
    "RiskType",
]
