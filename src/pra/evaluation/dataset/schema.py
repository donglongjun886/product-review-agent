"""评测集 Schema（dataset/schema.py）—— EvalCase 结构化标签的 Pydantic 契约。

对齐 docs/00-system-design.md §11.2 / docs/02-evaluation.md §2.2 的标签结构，但按
**Phase 1 最小闭环**裁剪（本 prompt 口径，比 02-evaluation v0.1 骨架更收敛）：

- ``expected.decision`` 只允许 PASS / REJECT（Phase 1 无 HUMAN 真值案）；
- ``input`` 直接复用 ``pra.domain.models.ProductReviewCase``（形状即《00》§2.1）——
  该模型 ``extra="forbid"``，故 eval_case 的 ``input`` 不允许塞任何 domain 未声明的
  额外键（评测数据与线上 DTO 完全对齐，也保证 loader 校验 = 线上 DTO 校验）；
- 五类 scene / source_type / hard_case / expected 的 risk_level / risk_type /
  evidence / expected_tools / applicable_policy 全部保留，供 Phase 2
  （EvidenceEvaluator / Agent 指标）与人工评审使用，Phase 1 只消费 expected.decision。

受控词表：scene 五类、source_type 三来源（SYNTHETIC / REAL_DESENSITIZED / VARIANT）。
风险类型 / 证据标签等为**开放性字符串标签**（标注口径），不做枚举约束
（与运行时 RiskType 枚举的映射留给 Phase 2 EvidenceEvaluator）。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pra.domain.models import ProductReviewCase

# 五类 scene（docs/00-system-design.md §11.1 / docs/02-evaluation.md §2.1）
SceneName = Literal["normal", "violation", "boundary", "multi-signal", "evasion"]

# 数据来源三分类（docs/00-system-design.md §13.2 / docs/02-evaluation.md §2.3）
SourceType = Literal["SYNTHETIC", "REAL_DESENSITIZED", "VARIANT"]


class EvalExpected(BaseModel):
    """单条 eval_case 的标注期望（docs/02-evaluation.md §2.2 expected 对象）。

    Phase 1 只有二值真值（PASS/REJECT）；risk_level / risk_type / evidence /
    expected_tools / applicable_policy 为标签完整性字段 —— 供人工评审与 Phase 2
    的 EvidenceEvaluator / Agent 指标使用，本阶段不参与 DecisionEvaluator 判定
    （唯一消费字段是 ``decision``）。
    """

    model_config = ConfigDict(extra="forbid")

    decision: Literal["PASS", "REJECT"] = Field(description="业务真值（Phase 1 仅 PASS/REJECT）")
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "NONE"] | None = Field(
        default=None, description="标注风险等级（可选）"
    )
    risk_type: list[str] = Field(default_factory=list, description="风险类型标签（如 POTENTIAL_IP_RISK）")
    evidence: list[str] = Field(
        default_factory=list, description="证据类型标签（如 image_similarity>=0.85 / merchant_history>=5_removals）"
    )
    expected_tools: list[str] = Field(default_factory=list, description="Agent 应调用的工具集合（Phase 2 Tool Selection 真值）")
    applicable_policy: list[str] = Field(default_factory=list, description="REJECT 案的政策依据（条款 ID 列表）")


class EvalCase(BaseModel):
    """一条完整评测案（docs/02-evaluation.md §2.2 JSONL 行的对象形态）。

    ``input`` 复用 ``ProductReviewCase``（extra=forbid → 与 domain 完全对齐）；
    ``eval_case_id`` 与业务 ``case_id`` 解耦（评审/报告引用稳定标识）。
    """

    model_config = ConfigDict(extra="forbid")

    eval_case_id: str = Field(description="评测案唯一 ID，如 EC_0001（与业务 case_id 解耦）")
    schema_version: int = Field(default=1, description="评测 schema 版本（v1 = 1）")
    scene: SceneName = Field(description="五类场景标签之一")
    source_type: SourceType = Field(default="SYNTHETIC", description="数据来源（SYNTHETIC/REAL_DESENSITIZED/VARIANT）")
    hard_case: bool = Field(default=False, description="是否入选 Hard Case（docs/02-evaluation.md §2.4）")
    input: ProductReviewCase = Field(description="商品事实快照（《00》§2.1 形状，线上 DTO 同型）")
    expected: EvalExpected = Field(description="标注期望（真值 + 标签）")
    annotation: dict | None = Field(
        default=None,
        description="人工标注记录（{labelers, agreed, notes}；notes 记录本案设计意图/三方案预期）",
    )


__all__ = ["EvalCase", "EvalExpected", "SceneName", "SourceType"]
