"""评测 Harness 基座（harness/base.py）—— EvalContext / SchemeRunner / EvalRecord。

对齐 docs/02-evaluation.md §3.5/§3.6 的 SchemeRunner 契约（本 prompt 口径收敛）：

- ``EvalContext``：一次评测运行的**配置注入点**（sweep 只改这里的常量，Phase 2）；
  Phase 1 最小字段：置信门槛（abstain 后处理用）+ Agent 工具数据源开关。
- ``SchemeRunner``（ABC）：三方案的统一接口 —— ``run(case, ctx) -> EvalRecord``；
  指标层（metrics）只吃 EvalRecord，不直接吃 DB / AgentState（§3.6 可比性前提）。
- ``EvalRecord``：**统一输出** —— 三方案各自执行结果归一化后的可比形态。
  字段语义（口径注释，勿漂移）：
  - ``decision`` ∈ PASS/REJECT/HUMAN_REVIEW（Rule 的 COMPLEX 映射 HUMAN_REVIEW；
    Single-call 的置信后处理映射 HUMAN_REVIEW —— §4.5 三分类输出空间对齐）；
  - ``evidence``：list[dict]，元素 {type, value, extra?}（Evidence 对象的人读/结构化
    摘要，Rule 为 RULE_HIT；agent 为最终 decision.evidence 转录）；
  - ``tool_calls_actual``：agent 为该 case 实际调用的工具名集合（去重、保序）；
    rule / single_call 恒为 []（无权调用工具）；
  - ``cost``：确定性成本摘要 {llm_calls, tool_calls, tokens} —— **刻意不含墙钟
    latency_ms**：评测要可逐字节重放，墙钟延迟是进程相关量，Phase 1 不落 EvalRecord
    （工程指标分位数在 Phase 2 单列口径）；
  - ``detail``：scheme 特有细节（rule 的 hits / single_call 的 raw 决策与理由 /
    agent 的 hypothesis_trace 摘要），只进审计不参与指标。
  Phase 1 全链路确定性：无真 LLM / 无网络 / 无随机；EvalRecord 不含任何
  进程相关字段 → 同数据重跑可逐字节比对（重放断言见 tests/test_evaluation_phase1.py）。
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
    """一次评测运行的配置注入（docs/02-evaluation.md §3.5；Phase 2 sweep 只改这里）。

    Phase 1 字段：
    - ``abstain_confidence_threshold``：Single-call 的 REJECT 候选置信门槛 ——
      confidence < 门槛的 REJECT 候选按确定性后处理记 HUMAN_REVIEW（对齐
      docs/02-evaluation.md §4.5 与 gate.CONFIDENCE_ABSTAIN_THRESHOLD=0.7 口径）；
    - ``tool_world``：Agent 使用的 InMemory 工具数据源 —— "eval" = 与
      eval_data/v1 同一份种子世界（三方案公平性：Rule/Single-call 只用基础输入，
      Agent 经工具取"基础输入之外"的证据，见《00》§12.0）；"rag" = **RAG 世界**
      （rag-implementation-plan.md R-4：CaseSearch/PolicySearch 注入真实 RAG 索引，
      事实工具沿用 eval 世界 —— 评测默认仍 "eval"，RAG 单独模式跑，见
      agent_scheme.make_rag_world_tools）。
    Phase 2 新增字段（sweep / ablation 用，docs/02-evaluation.md §5.1"只动配置"）：
    - ``evidence_thresholds``：Evidence 阈值覆盖 —— dict {"min_sim": float,
      "strong": float}，None = 用当前默认（min_sim=0.70 / strong=0.85，镜像
      pra.tools.image_analysis.tool 的 EVIDENCE_MIN_SIM / EVIDENCE_STRONG）。
      **只动配置不改判定逻辑**：sweep 每档阈值换一个 EvalContext 重跑即可；
      CONFIDENCE_ABSTAIN_THRESHOLD 本轮不参与扫描（固定 0.7，见 sweep.py docstring）。
      注入生效范围 = 评测侧相似度分档读取路径（agent_scheme 的确定性审查员模型），
      真实图 tools_node 的 quality_filter / gate overlay 常量属 pra.agent 业务层，
      不经本字段改动（报告须注明，见 sweep.py）。
    RAG 世界参数（仅 tool_world="rag" 生效；默认 None → hybrid）：
    - ``rag_mode``：检索模式 "bm25" / "vector" / "hybrid" —— 三路对比实验用
      （R-6：不预设 Hybrid 优于单路，由 Evaluation 实验回答）。
    - ``rag_backend``：RAG 索引后端（docs/10 §5）—— "local"（默认，行为不变）/
      "qdrant" / "chroma"；``tool_world="rag"`` 时经 ``make_rag_world_tools``
      透传给 ``pra.rag.factory``（缺省 "local" → 装配与改动前逐字节等价）。
    - ``rag_backend_options``：后端专属装配参数的透传字典（默认 None = 不传任何选项
      → 装配不变）—— 键名与 ``pra.rag.factory`` 构造参数逐字对应（如 chroma 的
      ``collection_prefix``）；只影响索引装配，零判定逻辑改动。
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
        """把本 ctx 的 ``evidence_thresholds`` 解析为确定性 (min_sim, strong) 字段 dict。

        None（或只给一档）→ 另一档取当前默认：``EVIDENCE_MIN_SIM=0.70`` /
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
    """三方案统一接口（docs/02-evaluation.md §3.5 SchemeRunner 契约）。

    实现约定：
    - ``name`` 为受控值 ∈ {"rule", "single_call_llm", "agent"}（EvalRecord.scheme 同源）；
    - ``run`` 须为**确定性纯执行**（同 case 同 ctx → 同 EvalRecord，可重放）；
    - runner 层负责遍历数据集并收集 EvalRecord，本抽象不感知数据集。
    """

    name: SchemeName

    @abstractmethod
    async def run(self, case: EvalCase, ctx: EvalContext) -> EvalRecord:
        """对单条 eval_case 执行一次方案评测，返回统一 EvalRecord。"""
        ...


class EvalRecord(BaseModel):
    """统一评测记录（docs/02-evaluation.md §3.6）—— metrics 层唯一输入。"""

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
