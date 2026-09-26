"""eval_case 的结构化标签契约：EvalCase / EvalExpected。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pra.domain.models import ProductReviewCase

SceneName = Literal["normal", "violation", "boundary", "multi-signal", "evasion"]
TruthDecision = Literal["PASS", "REJECT", "HUMAN_REVIEW"]

# abstention 语义标签：AUTO_DECIDABLE=本可自动判 / SHOULD_ABSTAIN=应转人工。
AbstainLabel = Literal["AUTO_DECIDABLE", "SHOULD_ABSTAIN"]


class EvalLineage(BaseModel):
    """程序化变异的溯源锚点（``seed_case_id`` 指向模板/种子案）。"""

    model_config = ConfigDict(extra="forbid")

    seed_case_id: str = Field(description="种子/模板案的 eval_case_id（溯源锚点）")


class EvalExpected(BaseModel):
    """单条 eval_case 的标注期望。"""

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
    expected_tools: list[str] = Field(
        default_factory=list,
        description="Agent 应调用的工具集合（Phase 2 Tool Selection 真值）；空列表 = 未标注工具期望的干净案，不计入 Tool Selection 分母 —— 不得解读为「应调用 0 个工具」",
    )

    @model_validator(mode="after")
    def _abstain_consistency(self) -> EvalExpected:
        """abstain_label ⇔ decision 的一致性校验；冲突即报错。"""
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
    """一条完整评测案（JSONL 行的对象形态）。"""

    model_config = ConfigDict(extra="forbid")

    eval_case_id: str = Field(description="评测案唯一 ID，如 EC_0001（与业务 case_id 解耦）")
    schema_version: int = Field(default=1, description="评测 schema 版本（v1 数据 = 1，Phase 2 v2 = 2）")
    scene: SceneName = Field(description="五类场景标签之一")
    lineage: EvalLineage | None = Field(
        default=None,
        description="程序化变异溯源（v1 手工数据无此字段 → None）",
    )
    input: ProductReviewCase = Field(description="商品事实快照（线上 DTO 同型）")
    expected: EvalExpected = Field(description="标注期望（真值 + abstention 标签 + 标签字段）")


__all__ = [
    "AbstainLabel",
    "EvalCase",
    "EvalExpected",
    "EvalLineage",
    "SceneName",
    "TruthDecision",
]
