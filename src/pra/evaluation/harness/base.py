"""评测 Harness 基座：SchemeRunner / EvalRecord。

- ``SchemeRunner``（ABC）：统一接口 ``run(case) -> EvalRecord``；指标层只吃 EvalRecord，
  不直接吃 DB / AgentState；
- ``EvalRecord``：各方案归一化后的**统一输出**。``decision`` ∈ PASS/REJECT/HUMAN_REVIEW
  （Rule 的 COMPLEX 与 Agent 的 Gate / overlay 收口都映射 HUMAN_REVIEW）；``evidence`` 为
  ``[{type, value, extra?}]`` 摘要；``tool_calls_actual`` 为实际调用工具名（去重保序，
  rule 恒 []）；``cost`` = ``{llm_calls, tool_calls, tokens}``，**刻意不含墙钟 latency_ms**
  （进程相关量会让跨 run 比对漂移）；``detail`` 只进审计，不参与指标。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pra.evaluation.dataset.schema import EvalCase

# 三分类输出空间（Rule 的 COMPLEX 与 Agent 的 Gate / overlay 收口都映射 HUMAN_REVIEW）
SchemeName = Literal["rule", "agent"]
DecisionLabel = Literal["PASS", "REJECT", "HUMAN_REVIEW"]


class SchemeRunner(ABC):
    """方案统一接口。

    实现约定：
    - ``name`` ∈ {"rule", "agent"}（EvalRecord.scheme 同源）；
    - runner 层负责遍历数据集并收集 EvalRecord，本抽象不感知数据集。
    """

    name: SchemeName

    @abstractmethod
    async def run(self, case: EvalCase) -> EvalRecord:
        """对单条 eval_case 执行一次方案评测，返回统一 EvalRecord。"""
        ...


class EvalRecord(BaseModel):
    """统一评测记录 —— metrics 层唯一输入。"""

    model_config = ConfigDict(extra="forbid")

    eval_case_id: str = Field(description="评测案 ID（与 EvalCase.eval_case_id 对齐）")
    scheme: SchemeName = Field(description="产出本记录的方案名")
    decision: DecisionLabel = Field(description="三分类裁决（归一化后）")
    risk_level: str | None = Field(default=None, description="风险等级（PASS 通常 NONE）")
    risk_type: list[str] = Field(default_factory=list, description="命中的风险类型（受控词表字符串）")
    decision_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="decision_confidence（agent 取确定性 dc；rule 无 → None）"
    )
    evidence: list[dict] = Field(default_factory=list, description="[{type, value, extra?}] 支撑证据")
    policy: list[str] = Field(default_factory=list, description="引用政策条款 ID（REJECT 案可引用依据）")
    tool_calls_actual: list[str] = Field(default_factory=list, description="实际调用工具名（去重保序；rule 恒 []）")
    cost: dict = Field(
        default_factory=dict,
        description="成本摘要 {llm_calls, tool_calls, tokens}（不含墙钟 latency）",
    )
    detail: dict = Field(default_factory=dict, description="scheme 特有细节（rule: hits；agent: trace 摘要）")


__all__ = ["DecisionLabel", "EvalRecord", "SchemeName", "SchemeRunner"]
