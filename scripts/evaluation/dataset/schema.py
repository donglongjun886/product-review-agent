"""eval 数据集的结构化契约：v3 构造即标签（cases / truth 双文件）。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pra.domain.models import ProductReviewCase

SceneName = Literal["normal", "violation", "boundary", "multi-signal", "evasion"]
TruthDecision = Literal["PASS", "REJECT", "HUMAN_REVIEW"]

# abstention 语义标签：AUTO_DECIDABLE=本可自动判 / SHOULD_ABSTAIN=应转人工。
AbstainLabel = Literal["AUTO_DECIDABLE", "SHOULD_ABSTAIN"]


RiskTypeV3 = Literal["POTENTIAL_IP_RISK", "EVASION_PATTERN"]
ToolNameV3 = Literal["ProductTool", "MerchantTool", "CaseSearchTool", "PolicySearchTool"]
# 证据类型受控词表：与生产证据 type 常量对齐（RULE_HIT=初筛规则命中 / MERCHANT_HISTORY /
# PRODUCT_FACT / POLICY_REF / CASE_PRECEDENT=四工具 / MEASUREMENT=快照测量），
# 漂移由 tests/test_eval_rulebook_drift.py 守护。
EvidenceTypeV3 = Literal[
    "RULE_HIT", "MERCHANT_HISTORY", "PRODUCT_FACT", "POLICY_REF", "CASE_PRECEDENT", "MEASUREMENT",
]


class TextHitV3(BaseModel):
    """一处文本埋点：哪个字段、植入了哪个词表词。"""

    model_config = ConfigDict(extra="forbid")

    field: Literal["title", "description"] = Field(description="埋点所在快照字段")
    term: str = Field(description="植入的词表原词（归类由打标函数按规则书词表推导）")


class MerchantFactsV3(BaseModel):
    """商家画像事实（快照之外，须 MerchantTool 取证；与种子 SQL 行一致）。"""

    model_config = ConfigDict(extra="forbid")

    merchant_id: str
    removals: int = Field(ge=0)
    title_relisting_count: int = Field(ge=0)
    similar_product_count: int = Field(ge=0)
    credit_score: int = Field(ge=0)


class RegistryFactsV3(BaseModel):
    """商品在库事实（快照之外，须 ProductTool 取证；与种子 SQL 行一致）。"""

    model_config = ConfigDict(extra="forbid")

    product_id: str
    category: str
    brand: str | None = Field(description="在库品牌；真空缺为 null")
    version: int = Field(ge=1)
    status: Literal["ON_SALE", "REMOVED"]


class CaseManifestV3(BaseModel):
    """埋点清单（只对真值侧可见）：本案植入了哪些语义事实、埋在哪。"""

    model_config = ConfigDict(extra="forbid")

    text_hits: list[TextHitV3] = Field(default_factory=list, description="标题/描述植入的词表词")
    snapshot_brand: str | None = Field(description="快照品牌字段原值（黑名单命中由打标函数按规则书推导）")
    snapshot_gaps: list[Literal["brand", "category"]] = Field(
        default_factory=list, description="快照测量缺口（品牌/类目空缺）"
    )
    registry_mismatch: bool = Field(
        default=False, description="在库事实与快照是否不一致（ProductTool 取证发现的冲突）"
    )
    merchant: MerchantFactsV3 = Field(description="商家画像事实")
    registry: RegistryFactsV3 = Field(description="商品在库事实")


class EvalConstructionV3(BaseModel):
    """构造参数记录（生成器侧溯源：用了哪个构造模式、什么参数）。"""

    model_config = ConfigDict(extra="forbid")

    pattern: str = Field(description="构造模式名，如 clean / evasion_word_dirty_merchant")
    params: dict[str, str | int | bool] = Field(
        default_factory=dict, description="构造参数快照（难例维度在此可调）"
    )


class EvalExpectedV3(BaseModel):
    """v3 真值：由独立打标函数按冻结规则书从埋点清单推导。"""

    model_config = ConfigDict(extra="forbid")

    decision: TruthDecision
    abstain_label: AbstainLabel = Field(description="v3 数据必填：HUMAN_REVIEW ⇔ SHOULD_ABSTAIN")
    risk_type: list[RiskTypeV3] = Field(default_factory=list)
    expected_evidence: list[EvidenceTypeV3] = Field(
        default_factory=list,
        description="应产出的证据类型集合；空列表 = 未标注，不进证据类型比对分母",
    )
    expected_tools: list[ToolNameV3] = Field(
        default_factory=list,
        description="Agent 应调用的工具集合；空列表 = 规则层可判、无需工具，不进工具指标分母",
    )
    expected_policies: list[str] = Field(
        default_factory=list, description="应引政策条款 ID（REJECT 必须非空：自动拒绝须有可引用依据）"
    )

    @model_validator(mode="after")
    def _truth_consistency(self) -> EvalExpectedV3:
        """真值不变量：abstain ⇔ decision；PASS 无风险；REJECT 必有可引用依据。"""
        if self.decision == "HUMAN_REVIEW" and self.abstain_label != "SHOULD_ABSTAIN":
            raise ValueError("decision=HUMAN_REVIEW 要求 abstain_label=SHOULD_ABSTAIN")
        if self.decision != "HUMAN_REVIEW" and self.abstain_label != "AUTO_DECIDABLE":
            raise ValueError("decision ∈ {PASS, REJECT} 要求 abstain_label=AUTO_DECIDABLE")
        if self.decision == "PASS" and (self.risk_type or self.expected_policies):
            raise ValueError("decision=PASS 要求 risk_type=[] / expected_policies=[]")
        if self.decision == "REJECT" and not self.expected_policies:
            raise ValueError("decision=REJECT 要求 expected_policies 非空（自动拒绝须有可引用依据）")
        return self


class EvalCaseV3(BaseModel):
    """v3 案件行（cases_v3.jsonl）：只含输入与构造记录，不含埋点与真值。"""

    model_config = ConfigDict(extra="forbid")

    eval_case_id: str
    schema_version: Literal[3] = 3
    scene: SceneName
    construction: EvalConstructionV3
    input: ProductReviewCase


class EvalTruthV3(BaseModel):
    """v3 真值行（truth_v3.jsonl）：埋点清单 + 打标函数推导的真值。"""

    model_config = ConfigDict(extra="forbid")

    eval_case_id: str
    manifest: CaseManifestV3
    expected: EvalExpectedV3


__all__ = [
    "AbstainLabel",
    "CaseManifestV3",
    "EvalCaseV3",
    "EvalConstructionV3",
    "EvalExpectedV3",
    "EvalTruthV3",
    "EvidenceTypeV3",
    "MerchantFactsV3",
    "RegistryFactsV3",
    "RiskTypeV3",
    "SceneName",
    "TextHitV3",
    "ToolNameV3",
    "TruthDecision",
]
