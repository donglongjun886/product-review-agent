"""评测集 Schema（dataset/schema.py）—— EvalCase 结构化标签的 Pydantic 契约。

对齐 docs/00-system-design.md §11.2 / docs/02-evaluation.md §2.2 的标签结构，并按
**Phase 1/Phase 2 两阶段口径**裁剪（docs/02 §2.1/§4.4，P-3/P-4 拍板）：

- ``expected.decision`` Phase 2 起为**三值真值** PASS / REJECT / HUMAN_REVIEW
  （Phase 1 只有二值案；HUMAN_REVIEW 仅出现在 Phase 2 的 ``SHOULD_ABSTAIN`` 案）；
- ``expected.abstain_label`` Phase 2 新增的 abstention 语义标签
  （docs/02 §4.4 AUTO_DECIDABLE / SHOULD_ABSTAIN）：
  - ``AUTO_DECIDABLE``：本可自动判（decision ∈ {PASS, REJECT}）—— Phase 2 默认；
  - ``SHOULD_ABSTAIN``：应转人工（decision == HUMAN_REVIEW）—— Phase 2 新增；
  - ``None``：Phase 1 老数据（无该字段），语义等价 AUTO_DECIDABLE
    （loader 读 v1 JSONL 照常工作，缺失即 None）；
  - 约束（``model_validator`` 校验，冲突即报错）：
    ``SHOULD_ABSTAIN`` ⇔ ``decision == HUMAN_REVIEW``；
    ``AUTO_DECIDABLE`` ⇒ decision ∈ {PASS, REJECT}；
    decision == HUMAN_REVIEW ⇒ abstain_label == SHOULD_ABSTAIN（不允许裸 HUMAN
    真值 —— Phase 2 起真值要么可自动判、要么应转人工，无第三种歧义态）。
- ``input`` 直接复用 ``pra.domain.models.ProductReviewCase``（形状即《00》§2.1）——
  该模型 ``extra="forbid"``，故 eval_case 的 ``input`` 不允许塞任何 domain 未声明的
  额外键（评测数据与线上 DTO 完全对齐，也保证 loader 校验 = 线上 DTO 校验）；
- 五类 scene / source_type / hard_case / expected 的 risk_level / risk_type /
  evidence / expected_tools / applicable_policy 全部保留，供 Phase 2
  （EvidenceEvaluator / Agent 指标 / AbstentionEvaluator）与人工评审使用；
  ``schema_version`` 由数据文件自声明：v1 数据 = 1，v2 = 2（读入后
  abstain_label 缺失一律 None，见上）。

受控词表：scene 五类、source_type 三来源（SYNTHETIC / REAL_DESENSITIZED / VARIANT）。
风险类型 / 证据标签等为**开放性字符串标签**（标注口径），不做枚举约束
（与运行时 RiskType 枚举的映射留给 Phase 2 EvidenceEvaluator）。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pra.domain.models import ProductReviewCase

# 五类 scene（docs/00-system-design.md §11.1 / docs/02-evaluation.md §2.1）
SceneName = Literal["normal", "violation", "boundary", "multi-signal", "evasion"]

# 数据来源三分类（docs/00-system-design.md §13.2 / docs/02-evaluation.md §2.3）
SourceType = Literal["SYNTHETIC", "REAL_DESENSITIZED", "VARIANT"]

# 真值三分类（docs/02 §2.2/§4.4）：Phase 1 只用前两者；HUMAN_REVIEW 仅 Phase 2
# SHOULD_ABSTAIN 案（decision 与 abstain_label 的一致性由 EvalExpected 校验器保证）。
TruthDecision = Literal["PASS", "REJECT", "HUMAN_REVIEW"]

# abstention 语义标签（docs/02 §4.4 两阶段口径）：
# AUTO_DECIDABLE=本可自动判 / SHOULD_ABSTAIN=应转人工（克制 abstention）。
AbstainLabel = Literal["AUTO_DECIDABLE", "SHOULD_ABSTAIN"]


class EvalLineage(BaseModel):
    """程序化变异的溯源（docs/02 §2.2 lineage 对象；Phase 2 变异生成数据必备）。

    ``seed_case_id`` 指向模板/种子案（v1 老案 EC_xxxx 或本文件手工种子）；
    ``mutation`` 为人类可读的变异摘要（如 ``similarity:0.72→0.91; title:加规避词``），
    供评审追溯"这条数据从哪来、改了哪些维度"。
    """

    model_config = ConfigDict(extra="forbid")

    seed_case_id: str = Field(description="种子/模板案的 eval_case_id（溯源锚点）")
    mutation: str = Field(description="变异摘要（确定性描述本次字段变异与真值调整口径）")


class EvalExpected(BaseModel):
    """单条 eval_case 的标注期望（docs/02-evaluation.md §2.2 expected 对象）。

    Phase 1 只有二值真值（PASS/REJECT，无 abstain_label 字段 → None 等价
    AUTO_DECIDABLE）；Phase 2 引入三值真值与 abstention 语义标签（见模块
    docstring 的约束）。risk_level / risk_type / evidence / expected_tools /
    applicable_policy 为标签完整性字段 —— 供人工评审与 Phase 2 的
    EvidenceEvaluator / Agent 指标 / AbstentionEvaluator 使用。
    """

    model_config = ConfigDict(extra="forbid")

    decision: TruthDecision = Field(description="业务真值（PASS/REJECT/ HUMAN_REVIEW=SHOULD_ABSTAIN 案）")
    abstain_label: AbstainLabel | None = Field(
        default=None,
        description=(
            "abstention 语义标签：AUTO_DECIDABLE=本可自动判（decision∈{PASS,REJECT}，"
            "Phase 2 默认可省）；SHOULD_ABSTAIN=应转人工（decision==HUMAN_REVIEW）；"
            "None=Phase 1 老数据（等价 AUTO_DECIDABLE，见模块 docstring 约束）"
        ),
    )
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "NONE"] | None = Field(
        default=None, description="标注风险等级（可选；PASS 通常 NONE）"
    )
    risk_type: list[str] = Field(default_factory=list, description="风险类型标签（如 POTENTIAL_IP_RISK）")
    evidence: list[str] = Field(
        default_factory=list, description="证据类型标签（如 image_similarity>=0.85 / merchant_history>=5_removals）"
    )
    expected_tools: list[str] = Field(default_factory=list, description="Agent 应调用的工具集合（Phase 2 Tool Selection 真值）")
    applicable_policy: list[str] = Field(default_factory=list, description="REJECT 案的政策依据（条款 ID 列表）")

    @model_validator(mode="after")
    def _abstain_consistency(self) -> EvalExpected:
        """abstain_label ⇔ decision 的一致性校验（docs/02 §4.4；冲突即报错）。

        - SHOULD_ABSTAIN ⇒ decision == HUMAN_REVIEW（应转人工案不许标自动真值）；
        - AUTO_DECIDABLE ⇒ decision ∈ {PASS, REJECT}（本可自动判案不许标人工真值）；
        - decision == HUMAN_REVIEW ⇒ abstain_label == SHOULD_ABSTAIN
          （Phase 2 起真值要么可自动判、要么应转人工 —— 裸 HUMAN 真值是歧义标注，
          直接报错而非静默兜底；Phase 1 老数据无 HUMAN 真值故不受影响）。
        """
        if self.abstain_label == "SHOULD_ABSTAIN" and self.decision != "HUMAN_REVIEW":
            raise ValueError(
                "abstain_label=SHOULD_ABSTAIN 要求 decision=HUMAN_REVIEW "
                f"（实际 decision={self.decision!r}）—— 应转人工案不许标自动真值"
            )
        if self.abstain_label == "AUTO_DECIDABLE" and self.decision == "HUMAN_REVIEW":
            raise ValueError(
                "abstain_label=AUTO_DECIDABLE 要求 decision ∈ {PASS, REJECT} "
                f"（实际 decision={self.decision!r}）—— 本可自动判案不许标人工真值"
            )
        if self.decision == "HUMAN_REVIEW" and self.abstain_label != "SHOULD_ABSTAIN":
            raise ValueError(
                "decision=HUMAN_REVIEW 要求 abstain_label=SHOULD_ABSTAIN "
                f"（实际 abstain_label={self.abstain_label!r}）—— 裸 HUMAN 真值是歧义标注"
            )
        return self


class EvalCase(BaseModel):
    """一条完整评测案（docs/02-evaluation.md §2.2 JSONL 行的对象形态）。

    ``input`` 复用 ``ProductReviewCase``（extra=forbid → 与 domain 完全对齐）；
    ``eval_case_id`` 与业务 ``case_id`` 解耦（评审/报告引用稳定标识）。
    ``schema_version`` 声明数据版本（v1 老数据 = 1；Phase 2 v2 数据 = 2）——
    v2 数据读入后 abstain_label 缺失一律 None（等价 AUTO_DECIDABLE，兼容 v1）。
    """

    model_config = ConfigDict(extra="forbid")

    eval_case_id: str = Field(description="评测案唯一 ID，如 EC_0001（与业务 case_id 解耦）")
    schema_version: int = Field(default=1, description="评测 schema 版本（v1 数据 = 1，Phase 2 v2 = 2）")
    scene: SceneName = Field(description="五类场景标签之一")
    source_type: SourceType = Field(default="SYNTHETIC", description="数据来源（SYNTHETIC/REAL_DESENSITIZED/VARIANT）")
    lineage: EvalLineage | None = Field(
        default=None,
        description="程序化变异溯源（docs/02 §2.2；v1 手工数据无此字段 → None）",
    )
    hard_case: bool = Field(default=False, description="是否入选 Hard Case（docs/02-evaluation.md §2.4）")
    input: ProductReviewCase = Field(description="商品事实快照（《00》§2.1 形状，线上 DTO 同型）")
    expected: EvalExpected = Field(description="标注期望（真值 + abstention 标签 + 标签字段）")
    annotation: dict | None = Field(
        default=None,
        description="人工标注记录（{labelers, agreed, notes}；notes 记录本案设计意图/三方案预期）",
    )


__all__ = [
    "AbstainLabel",
    "EvalCase",
    "EvalExpected",
    "EvalLineage",
    "SceneName",
    "SourceType",
    "TruthDecision",
]
