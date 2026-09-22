"""评测 Harness 基座：EvalContext / SchemeRunner / EvalRecord。

- ``EvalContext``：一次评测运行的配置注入点；
- ``SchemeRunner``（ABC）：统一接口 ``run(case, ctx) -> EvalRecord``；指标层只吃
  EvalRecord，不直接吃 DB / AgentState —— 这是三方案可比的前提；
- ``EvalRecord``：三方案归一化后的**统一输出**。``decision`` ∈
  PASS/REJECT/HUMAN_REVIEW（Rule 的 COMPLEX 与 Single-call 的置信不足后处理都映射
  HUMAN_REVIEW）；``evidence`` 为 ``[{type, value, extra?}]`` 摘要；``tool_calls_actual``
  为实际调用工具名（去重保序，rule / single_call 恒 []）；``cost`` =
  ``{llm_calls, tool_calls, tokens}``，**刻意不含墙钟 latency_ms**（进程相关量会让
  scripted 路径的重跑比对漂移）；``detail`` 只进审计，不参与指标。

**scripted 路径全链路确定性**：无真 LLM / 无网络 / 无随机；EvalRecord 不含进程相关字段 →
同数据重跑可逐字节比对（Regression 与单测依赖）。**real 路径（``AgentScheme(llm=...)``）
不在此列** —— 真实模型不要求逐字节可重放。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pra.evaluation.dataset.schema import EvalCase

# 三分类输出空间（Rule 的 COMPLEX 与 Single-call 的置信不足后处理都映射 HUMAN_REVIEW）
SchemeName = Literal["rule", "single_call_llm", "agent"]
DecisionLabel = Literal["PASS", "REJECT", "HUMAN_REVIEW"]


class EvalContext(BaseModel):
    """一次评测运行的配置注入（不动判定逻辑）。

    - ``abstain_confidence_threshold``：Single-call 的 REJECT 候选置信门槛 ——
      confidence < 门槛的 REJECT 候选确定性改记 HUMAN_REVIEW（与
      ``gate.CONFIDENCE_ABSTAIN_THRESHOLD=0.7`` 同口径）；
    - ``tool_world``：Agent 的工具数据源 —— "eval" = 与 eval_data/v2 同一份种子世界
      （公平性：Rule/Single-call 只用基础输入，Agent 经工具取"基础输入之外"的证据）；
      "default" = 仓库默认演示种子；"rag" = RAG 世界（CaseSearch / PolicySearch 注入
      真实 RAG 索引，模式固定生产口径 hybrid；事实工具沿用 eval 世界）。评测默认 "eval"。
    """

    model_config = ConfigDict(extra="forbid")

    abstain_confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Single-call REJECT 候选转人工的置信门槛"
    )
    tool_world: Literal["eval", "default", "rag"] = Field(
        default="eval",
        description="Agent 工具数据源：eval=评测种子世界 / default=仓库默认演示种子 / rag=RAG 世界（真实 Policy/Case KB）",
    )


class SchemeRunner(ABC):
    """三方案统一接口。

    实现约定：
    - ``name`` ∈ {"rule", "single_call_llm", "agent"}（EvalRecord.scheme 同源）；
    - ``run`` 须为**确定性纯执行**（同 case 同 ctx → 同 EvalRecord，可重放）；
    - runner 层负责遍历数据集并收集 EvalRecord，本抽象不感知数据集。
    """

    name: SchemeName

    @abstractmethod
    async def run(self, case: EvalCase, ctx: EvalContext) -> EvalRecord:
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
    tool_calls_actual: list[str] = Field(default_factory=list, description="实际调用工具名（去重保序；rule/llm 为 []）")
    cost: dict = Field(
        default_factory=dict,
        description="成本摘要 {llm_calls, tool_calls, tokens}（确定性；不含墙钟 latency）",
    )
    detail: dict = Field(default_factory=dict, description="scheme 特有细节（rule: hits；llm: raw；agent: trace 摘要）")


__all__ = ["DecisionLabel", "EvalContext", "EvalRecord", "SchemeName", "SchemeRunner"]
