"""评测 Harness 基座：EvalContext / SchemeRunner / EvalRecord。

- ``EvalContext``：一次评测运行的配置注入点（sweep 只改这里的常量）；
- ``SchemeRunner``（ABC）：统一接口 ``run(case, ctx) -> EvalRecord``；指标层只吃
  EvalRecord，不直接吃 DB / AgentState —— 这是三方案可比的前提；
- ``EvalRecord``：三方案归一化后的**统一输出**。``decision`` ∈
  PASS/REJECT/HUMAN_REVIEW（Rule 的 COMPLEX 与 Single-call 的置信不足后处理都映射
  HUMAN_REVIEW）；``evidence`` 为 ``[{type, value, extra?}]`` 摘要；``tool_calls_actual``
  为实际调用工具名（去重保序，rule / single_call 恒 []）；``cost`` =
  ``{llm_calls, tool_calls, tokens}``，**刻意不含墙钟 latency_ms**（评测要可逐字节
  重放，墙钟是进程相关量）；``detail`` 只进审计，不参与指标。

全链路确定性：无真 LLM / 无网络 / 无随机；EvalRecord 不含进程相关字段 → 同数据
重跑可逐字节比对。
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
    """一次评测运行的配置注入（sweep 只改这里，不动判定逻辑）。

    - ``abstain_confidence_threshold``：Single-call 的 REJECT 候选置信门槛 ——
      confidence < 门槛的 REJECT 候选确定性改记 HUMAN_REVIEW（与
      ``gate.CONFIDENCE_ABSTAIN_THRESHOLD=0.7`` 同口径）；
    - ``tool_world``：Agent 的工具数据源 —— "eval" = 与 eval_data/v1 同一份种子世界
      （公平性：Rule/Single-call 只用基础输入，Agent 经工具取"基础输入之外"的证据）；
      "default" = 仓库默认演示种子；"rag" = RAG 世界（CaseSearch / PolicySearch 注入
      真实 RAG 索引，事实工具沿用 eval 世界）。评测默认 "eval"。
    - ``evidence_thresholds``：Evidence 阈值覆盖 ``{"min_sim", "strong"}``，None = 默认
      （0.70 / 0.85，镜像 ``pra.tools.image_analysis.tool`` 常量）。生效范围 =
      评测侧相似度分档读取路径（agent_scheme 的确定性审查员模型）；真实图 tools_node 的
      quality_filter / gate overlay 常量属 pra.agent 业务层，不经本字段改动。
    - RAG 世界参数（仅 tool_world="rag" 生效）：``rag_mode`` = "bm25" / "vector" /
      "hybrid"；``rag_backend`` = "local"（默认）/ "qdrant" / "chroma"，经
      ``make_rag_world_tools`` 透传给 ``pra.rag.factory``；``rag_backend_options`` =
      后端装配参数透传（键名与 ``pra.rag.factory`` 构造参数逐字对应，None = 不传）。
      只影响索引装配，零判定逻辑改动。
    """

    model_config = ConfigDict(extra="forbid")

    abstain_confidence_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Single-call REJECT 候选转人工的置信门槛"
    )
    tool_world: Literal["eval", "default", "rag"] = Field(
        default="eval",
        description="Agent 工具数据源：eval=评测种子世界 / default=仓库默认演示种子 / rag=RAG 世界（真实 Policy/Case KB）",
    )
    rag_mode: Literal["bm25", "vector", "hybrid"] | None = Field(
        default=None,
        description="RAG 世界检索模式（tool_world='rag' 时生效；None → hybrid 0.5/0.5）",
    )
    rag_backend: Literal["local", "qdrant", "chroma"] = Field(
        default="local",
        description="RAG 世界索引后端（tool_world='rag' 时生效；默认 local = 既有实现）",
    )
    rag_backend_options: dict | None = Field(
        default=None,
        description="RAG 后端专属装配参数透传（None = 不传；键名同 pra.rag.factory 参数）",
    )
    evidence_thresholds: dict | None = Field(
        default=None,
        description="Evidence 阈值覆盖 {\"min_sim\": float, \"strong\": float}；None = 默认 0.70/0.85",
    )

    def resolve_evidence_thresholds(self) -> dict:
        """把 ``evidence_thresholds`` 解析为确定性 (min_sim, strong) dict。

        None（或只给一档）→ 另一档取当前默认 ``EVIDENCE_MIN_SIM=0.70`` /
        ``EVIDENCE_STRONG=0.85``（镜像 tools 常量，避免评测侧手抄漂移）。
        惰性 import tools 常量，避免本模块导入期拉起工具包。
        """
        from pra.tools.image_analysis.tool import EVIDENCE_MIN_SIM, EVIDENCE_STRONG

        overrides = dict(self.evidence_thresholds or {})
        return {
            "min_sim": float(overrides.get("min_sim", EVIDENCE_MIN_SIM)),
            "strong": float(overrides.get("strong", EVIDENCE_STRONG)),
        }


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
