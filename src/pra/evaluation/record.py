"""评测记录契约：EvalRecord / DecisionLabel / SchemeName。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SchemeName = Literal["rule", "single", "agent"]
DecisionLabel = Literal["PASS", "REJECT", "HUMAN_REVIEW"]


class EvalRecord(BaseModel):
    """统一评测记录。"""

    model_config = ConfigDict(extra="forbid")

    eval_case_id: str = Field(description="评测案 ID（与 EvalCase.eval_case_id 对齐）")
    scheme: SchemeName = Field(description="产出本记录的方案名")
    decision: DecisionLabel = Field(description="三分类裁决（归一化后）")
    risk_level: str | None = Field(default=None, description="风险等级（PASS 通常 NONE）")
    risk_type: list[str] = Field(default_factory=list, description="命中的风险类型（受控词表字符串）")
    decision_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="decision_confidence（agent 取确定性 dc；rule 无 → None）",
    )
    evidence: list[dict] = Field(default_factory=list, description="[{type, value, extra?}] 支撑证据")
    policy: list[str] = Field(default_factory=list, description="引用政策条款 ID（REJECT 案可引用依据）")
    tool_calls_actual: list[str] = Field(
        default_factory=list, description="实际调用工具名（去重保序；rule / single 恒 []）"
    )
    cost: dict = Field(
        default_factory=dict,
        description="成本摘要 {llm_calls, tool_calls, tokens}（不含墙钟 latency）",
    )
    detail: dict = Field(default_factory=dict, description="scheme 特有细节（rule: hits；agent: trace 摘要）")


__all__ = ["DecisionLabel", "EvalRecord", "SchemeName"]
