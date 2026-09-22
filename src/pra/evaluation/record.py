"""评测记录契约：EvalRecord / DecisionLabel / SchemeName。

``EvalRecord`` 是各臂归一化后的**统一输出**，也是 metrics 层唯一输入（metrics 不直接吃
DB / AgentState）。``decision`` ∈ PASS/REJECT/HUMAN_REVIEW（rule 的 COMPLEX、single 调用
失败、agent 的 Gate / overlay 收口都映射 HUMAN_REVIEW）；``evidence`` 为
``[{type, value, extra?}]`` 摘要；``tool_calls_actual`` 为实际调用工具名（去重保序，
rule / single 恒 []）；``cost`` = ``{llm_calls, tool_calls, tokens}``，**刻意不含墙钟
latency_ms**（进程相关量会让跨 run 比对漂移）；``detail`` 只进审计，不参与指标。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# 三分类输出空间（rule 的 COMPLEX、single 调用失败、agent 的 Gate / overlay 收口都映射
# HUMAN_REVIEW）；三条臂名与 metrics 层的 scheme 同源。
SchemeName = Literal["rule", "single", "agent"]
DecisionLabel = Literal["PASS", "REJECT", "HUMAN_REVIEW"]


class EvalRecord(BaseModel):
    """统一评测记录 —— metrics 层唯一输入。"""

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
