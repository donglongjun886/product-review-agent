"""领域模型 —— 复杂风险调查 Agent 的数据契约层。

三类业务对象：输入事实快照（``ProductReviewCase`` → ``ProductInfo`` / ``SkuInfo`` /
``ProductImage`` / ``ScreeningSignal``）、运行中状态（``Hypothesis`` / ``Evidence`` /
``Budget`` / ``BudgetLimits``）、输出裁决（``ReviewDecision``：三分类 + 证据链 + 假设轨迹 +
预算快照）。

边界：只声明结构与取值约束，不含业务/决策逻辑（预算超限判断、假设后验更新属
agent/guardrails 层）。``extra="forbid"``：拒收未声明字段，上游悄悄加字段应立即报错。
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class _StrictModel(BaseModel):
    """包内基类：默认拒绝未在模型上声明的字段（契约 DTO，防 schema 静默漂移）。"""

    model_config = ConfigDict(extra="forbid")


# ---- 枚举：受控词表 ----


class Decision(str, Enum):
    """最终裁决三分类：PASS 放行 / REJECT 拒绝上架 / HUMAN_REVIEW 转人工（证据不足、置信不足、预算耗尽、政策模糊时主动克制地转人工）。"""

    PASS = "PASS"
    REJECT = "REJECT"
    HUMAN_REVIEW = "HUMAN_REVIEW"


class RiskLevel(str, Enum):
    """风险等级：供人工队列排序与统计；PASS 案件对应 NONE。"""

    NONE = "NONE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RiskType(str, Enum):
    """风险类型受控词表 —— 评测集标签 / 政策库元数据 / 决策输出三方共用，保证召回与精确率可按类型统计。"""

    POTENTIAL_IP_RISK = "POTENTIAL_IP_RISK"  # 疑似 IP / 品牌模仿
    EVASION_PATTERN = "EVASION_PATTERN"      # 规避审核行为模式
    FALSE_CLAIM = "FALSE_CLAIM"              # 虚假 / 无依据宣传
    FIELD_CONFLICT = "FIELD_CONFLICT"        # 商品字段信息冲突


class HypothesisStatus(str, Enum):
    """假设生命周期状态：PENDING=生成后尚未调查验证；SUPPORTED=被证据支持；REFUTED=被证伪；UNRESOLVED=已核查但证据不足、无定论。

    PENDING 与 UNRESOLVED 的区分是业务不变量：高优先假设处于 UNRESOLVED → 不可 PASS →
    导向 HUMAN_REVIEW（不能把「没查到」当「证明无」）。
    """

    PENDING = "PENDING"
    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    UNRESOLVED = "UNRESOLVED"


# ---- 输入事实快照 ----


class ProductImage(_StrictModel):
    """商品图片（DB ``product_image``）。

    ``ocr_text`` 预填机审阶段已产出的 OCR 结果，避免 Agent 重复调用 OCR Tool。
    """

    url: str = Field(description="图片地址")
    ocr_text: str | None = Field(default=None, description="机审阶段已识别的 OCR 文本；为空表示未知，需 OCR Tool 补查")
    source: str = Field(description="图片位次/来源，如 主图 / 附图1")


class SkuInfo(_StrictModel):
    """SKU 维度信息（DB ``product_sku``）：颜色/尺码/价格，供字段冲突类调查比对。"""

    sku_id: str = Field(description="SKU ID，如 S_1")
    color: str = Field(description="颜色/款式规格")
    size: str = Field(description="尺码/规格")
    price: float = Field(description="展示价（非计算精度敏感场景，仅比对/展示用）")


class ProductInfo(_StrictModel):
    """商品事实快照（DB ``product`` 主表）。

    回答「商品事实到底是什么」—— 尤其 brand 是否真空缺、字段间是否冲突，是 ProductTool /
    OCRTool 交叉验证的事实锚点。``version`` 为乐观锁版本号，同一商品不同版本只审一次。
    """

    product_id: str = Field(description="商品 ID，如 P_88231")
    title: str = Field(description="商品标题")
    description: str = Field(description="商品描述")
    category: str = Field(description="类目路径，如 女鞋/运动鞋")
    brand: str | None = Field(default=None, description="品牌；真空缺为 null —— '规避品牌'调查的起点信号")
    attributes: dict[str, str] = Field(default_factory=dict, description="关键属性键值对，如 {材质: PU}")
    sku_list: list[SkuInfo] = Field(default_factory=list, description="SKU 列表，可为空")
    images: list[ProductImage] = Field(default_factory=list, description="图片列表，可为空")
    listing_time: datetime = Field(description="上架时间（DB DATETIME，naive；JSON 示例为 'YYYY-MM-DD HH:MM:SS'）")
    version: int = Field(description="商品乐观锁版本号（同一 product_id+version 幂等只审一次）")


class ScreeningSignal(_StrictModel):
    """传统机审信号（DB ``review_signal``）—— Agent 的起点信息，避免重复劳动；``name`` 如 KEYWORD / LOGO_DETECT。"""

    name: str = Field(description="信号器名称，如 KEYWORD / LOGO_DETECT / CATEGORY_RULE")
    result: str = Field(description="信号器结果（文本语义由信号器定义，示例为 PASS）")
    score: float = Field(description="信号分；量纲由具体机审模型定义（示例为 0~1 概率类）")


class ProductReviewCase(_StrictModel):
    """审核案件 —— Agent 的输入 DTO（DB ``review_case``）：一次上架/改标题/改属性/改图片事件 = 一个案件。"""

    case_id: str = Field(description="案件 ID，如 CASE_20240907_001")
    product: ProductInfo = Field(description="商品事实快照（可含空 brand 等缺口，恰是调查对象）")
    merchant_id: str = Field(description="商家 ID，如 M_5512")
    event_type: str = Field(description="触发审核的事件类型，如 NEW_LISTING / UPDATE_TITLE / UPDATE_IMAGE")
    screening_signals: list[ScreeningSignal] = Field(default_factory=list, description="机审信号起点，可为空（直接命中复杂队列）")


# ---- 运行中状态对象 ----


class Evidence(_StrictModel):
    """单条证据（DB ``evidence``）—— 结论依据的最小可引用单元，区别于过程审计 ``tool_call_history``。

    ``source`` 记工具名，``ref_id`` 指向源对象（图片 URL / 商家 ID / 先例 case_id / 政策条款
    ID 等）保证可回溯。``extra`` 承载结构化附加数值（similarity / removals /
    violations_total / conflict 信号）供确定性函数机器读取 —— 人读摘要只进 ``value``。
    """

    type: str = Field(description="证据类型（开放性文本），如 IMAGE_SIMILARITY / MERCHANT_HISTORY / CASE_PRECEDENT")
    source: str = Field(description="证据来源工具，如 ImageAnalysisTool / MerchantTool")
    value: str = Field(description="证据内容（人读摘要），如 similarity=0.91, match=某品牌经典鞋款")
    weight: float = Field(ge=0.0, le=1.0, description="证据强度（0~1，供证据综合加权）")
    ref_id: str | None = Field(default=None, description="引用的源对象 ID（用于去重/回溯，可空）")
    extra: dict = Field(default_factory=dict, description="结构化附加数值（similarity/removals 等），供确定性函数读取，不进人读 value")


class Hypothesis(_StrictModel):
    """风险假设（``hypotheses[]`` / ``hypothesis_trace[]`` 元素）—— Agent 是假设验证器而非分类器。

    每条假设携带 ``prior → posterior`` 的可解释演变与生命周期状态。``evidence_for /
    evidence_against`` 为证据引用/摘要字符串（如 ``"image_similarity=0.91"``、
    ``"brand=null"``），结构化证据本体统一存 ``AgentState.evidence[]``，避免同一事实双份存。
    ``prior`` / ``posterior`` 为 None 表示尚未赋值/未评估，确定性公式按 0 处理。
    """

    id: str = Field(description="假设 ID，如 H1 / H2")
    statement: str = Field(description="假设陈述，如 '刻意规避品牌识别'")
    prior: float | None = Field(default=None, ge=0.0, le=1.0, description="先验概率（hypothesize 显式给出；未赋值前为 None，公式按 0 处理）")
    posterior: float | None = Field(default=None, ge=0.0, le=1.0, description="后验概率；生成后尚未经 reevaluate 更新前为 None")
    status: HypothesisStatus = Field(default=HypothesisStatus.PENDING, description="验证阶段（PENDING→SUPPORTED/REFUTED/UNRESOLVED）")
    evidence_for: list[str] = Field(default_factory=list, description="支持本假设的证据引用/摘要字符串")
    evidence_against: list[str] = Field(default_factory=list, description="反对本假设的证据引用/摘要字符串")


class BudgetLimits(_StrictModel):
    """预算限额（Guardrail 上限）。

    默认 10 次 LLM / 15 次 Tool / 40000 token / 30s。语义是上界而非目标：正常案件实际调用
    明显低于上限（主链路常态 8 次 LLM / 5 次 Tool），余量用于 schema 校验失败重试、工具失败
    恢复与防无限循环；运行时可由配置覆盖。超限语义为「带部分证据转人工止损」而非失败。
    """

    max_llm_calls: int = Field(default=10, gt=0, description="最大 LLM 调用次数（Guardrail 上界，可配置覆盖）")
    max_tool_calls: int = Field(default=15, gt=0, description="最大 Tool 调用次数（Guardrail 上界，可配置覆盖）")
    max_tokens: int = Field(default=40000, gt=0, description="最大 Token 用量")
    max_latency_ms: int = Field(default=30000, gt=0, description="最大执行时长（毫秒）")


class Budget(_StrictModel):
    """已用预算 + 限额 —— AgentState 的硬字段，由条件边路由函数每轮进入节点前确定性检查，超限即转人工。

    本模型是「运行态字段 + 决策输出快照字段（含 ``latency_ms``）」的并集，额外字段均有默认值，
    故运行态与 ``ReviewDecision.budget_used`` 共用同一类型（一份真相，无派生形态）。
    """

    llm_calls: int = Field(default=0, ge=0, description="已用 LLM 调用次数")
    tool_calls: int = Field(default=0, ge=0, description="已用 Tool 调用次数")
    tokens: int = Field(default=0, ge=0, description="已用 Token 数")
    latency_ms: int = Field(default=0, ge=0, description="已耗时（毫秒，可由 start_time 推算，亦支持逐步累加）")
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="本轮调查启动时间（UTC）")
    limits: BudgetLimits = Field(default_factory=BudgetLimits, description="本次调查的限额配置")


# ---- 输出裁决 ----


class ReviewDecision(_StrictModel):
    """最终结构化裁决 —— Agent 的输出 DTO（DB ``decision``）。

    三分类 + 风险等级/类型 + 置信度 + 证据链 + 政策引用 + 假设轨迹 + 预算快照，全部可回溯到
    「哪个工具提供的哪条证据导致该结论」。``risk_type`` 语义上必填（PASS 时为 ``[]``）。
    ``overrides`` 记录确定性 overlay 的改判/归因原因码（R1_HARD_RULE / R2_* / R3_* / R4_* /
    R5_*）—— 空 = overlay 未改判（LLM 提案即终值），是「谁把 PASS 改成了 HUMAN_REVIEW」的
    可审计落点。图内运行唯一终态为 DECIDED（含三种决策结果），ESCALATED / BUDGET_EXCEEDED
    不再作为主终态，超限归因只记在 overrides 与预算快照。
    """

    decision: Decision = Field(description="三分类裁决：PASS / REJECT / HUMAN_REVIEW")
    risk_level: RiskLevel = Field(description="风险等级（PASS 为 NONE）")
    risk_type: list[RiskType] = Field(default_factory=list, description="命中的风险类型受控词表（PASS 为 []）")
    decision_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="decision_confidence —— 自动决策安全门槛（确定性重算值，非模型真实概率/非违规概率；"
        "只回答'如果自动判，判错风险够不够低'，见 docs/00 §7.4；已由 confidence 改名）",
    )
    evidence: list[Evidence] = Field(default_factory=list, description="支撑本裁决的证据链（与运行期 evidence 同型）")
    policy: list[str] = Field(default_factory=list, description="引用的政策条款 ID，如 POLICY_3.2（REJECT 必须有可引用依据，见 docs/00 §7.2）")
    hypothesis_trace: list[Hypothesis] = Field(default_factory=list, description="关键假设的演变轨迹（prior→posterior→status），供解释与 eval 重放")
    budget_used: Budget = Field(default_factory=Budget, description="裁决时的预算快照（引用运行期 Budget 实例，含限额与启动时间）")
    overrides: list[str] = Field(default_factory=list, description="overlay 改判/归因原因码（R1_HARD_RULE / R2_* / R3_* / R4_* / R5_*）；空=overlay 未改判")
