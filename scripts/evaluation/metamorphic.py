"""蜕变测试与 pass^k（docs/02 §6）：确定性扰动 + 裁决严格度比较 + 重复运行一致率。

蜕变扰动是纯函数（同案同扰动结果一致，可重放）：
- 保语义扰动：只改不动语义的字段（价格 / SKU ID / 上架时间 / 图片地址），或换表述
  （生成器模板的同义改写；匹配不到模板回退追加中性从句），初筛规则命中不变；
- 风险增强扰动：标题前缀植入规避词（词表取自冻结规则书，真值口径单一来源），
  裁决不得变松（严格度序 REJECT > HUMAN_REVIEW > PASS）。

pass^k：同一批案重复运行 k 次，逐案比对裁决是否全部一致。
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import timedelta
from typing import Literal

from pydantic import BaseModel, Field

from evaluation.dataset.schema import EvalCaseV3
from evaluation.generator.builder import DESCRIPTION_PARAPHRASES, TITLE_PARAPHRASES
from evaluation.labeling.labeler import load_rulebook
from evaluation.record import EvalRecord

__all__ = [
    "MetamorphicFailure",
    "PassKMetrics",
    "check_metamorphic",
    "pass_k_consistency",
    "perturb_paraphrase",
    "perturb_risk_amplifying",
    "perturb_semantics_preserving",
    "strictness",
]

_STRICTNESS = {"REJECT": 2, "HUMAN_REVIEW": 1, "PASS": 0}

_NEUTRAL_SKU_SUFFIX = "_P"
_NEUTRAL_PRICE_DELTA = 7.5
_NEUTRAL_TIME_SHIFT = timedelta(days=1)
_NEUTRAL_CLAUSE = "页面信息持续更新中。"
_RISK_TERM = "高仿"


def strictness(decision: str) -> int:
    """裁决严格度：REJECT(2) > HUMAN_REVIEW(1) > PASS(0)。"""
    return _STRICTNESS[decision]


def perturb_semantics_preserving(case: EvalCaseV3) -> EvalCaseV3:
    """保语义扰动：改价格 / SKU ID / 上架时间 / 图片地址，标题/描述/品牌/类目一律不动。"""
    product = case.input.product
    skus = [
        sku.model_copy(update={"sku_id": f"{sku.sku_id}{_NEUTRAL_SKU_SUFFIX}", "price": sku.price + _NEUTRAL_PRICE_DELTA})
        for sku in product.sku_list
    ]
    images = [
        img.model_copy(update={"url": f"{img.url}?v=perturbed"}) for img in product.images
    ]
    return case.model_copy(
        update={
            "input": case.input.model_copy(
                update={
                    "product": product.model_copy(
                        update={
                            "sku_list": skus,
                            "images": images,
                            "listing_time": product.listing_time + _NEUTRAL_TIME_SHIFT,
                        }
                    )
                }
            )
        }
    )


def perturb_risk_amplifying(case: EvalCaseV3) -> EvalCaseV3:
    """风险增强扰动：标题前缀植入一个词表规避词（其余字段不动）。"""
    product = case.input.product
    evasion_terms = [str(t) for t in load_rulebook()["evasion_terms"]]
    term = _RISK_TERM if _RISK_TERM in evasion_terms else min(evasion_terms)
    return case.model_copy(
        update={
            "input": case.input.model_copy(
                update={
                    "product": product.model_copy(update={"title": f"{term} {product.title}"})
                }
            )
        }
    )


def perturb_paraphrase(case: EvalCaseV3) -> EvalCaseV3:
    """保语义"换表述"扰动：命中生成器模板同义改写映射则替换；
    匹配不到模板回退为描述末尾追加中性从句（不含任何词表词）。"""
    product = case.input.product
    title, description = product.title, product.description
    for src, dst in TITLE_PARAPHRASES.items():
        if src in title:
            title = title.replace(src, dst)
            break
    for src, dst in DESCRIPTION_PARAPHRASES.items():
        if src in description:
            description = description.replace(src, dst)
            break
    if title == product.title and description == product.description:
        description = f"{description}{_NEUTRAL_CLAUSE}"
    return case.model_copy(
        update={
            "input": case.input.model_copy(
                update={"product": product.model_copy(update={"title": title, "description": description})}
            )
        }
    )


class MetamorphicFailure(BaseModel):
    """一条蜕变违例：哪一案、哪种关系被破坏、原裁决与扰动后裁决。"""

    eval_case_id: str
    kind: Literal["NEUTRAL_FLIP", "RISK_AMPLIFIED_LOOSENED"] = Field(
        description="NEUTRAL_FLIP=保语义扰动裁决翻转；RISK_AMPLIFIED_LOOSENED=风险增强后裁决变松"
    )
    original: str
    perturbed: str


def check_metamorphic(
    original: EvalRecord, neutrals: list[EvalRecord], amplified: EvalRecord
) -> list[MetamorphicFailure]:
    """比对一案的原裁决与各扰动后的裁决，返回违例清单（可为空）。

    :param original: 原案的评测记录；
    :param neutrals: 各保语义扰动（改无关字段 / 换表述）后的记录（裁决不得翻转）；
    :param amplified: 风险增强扰动后的记录（裁决不得变松）。
    """
    failures: list[MetamorphicFailure] = []
    cid = original.eval_case_id
    for neutral in neutrals:
        if neutral.decision != original.decision:
            failures.append(
                MetamorphicFailure(
                    eval_case_id=cid,
                    kind="NEUTRAL_FLIP",
                    original=original.decision,
                    perturbed=neutral.decision,
                )
            )
    if strictness(amplified.decision) < strictness(original.decision):
        failures.append(
            MetamorphicFailure(
                eval_case_id=cid,
                kind="RISK_AMPLIFIED_LOOSENED",
                original=original.decision,
                perturbed=amplified.decision,
            )
        )
    return failures


class PassKMetrics(BaseModel):
    """pass^k 一致率：重复运行 k 次后裁决逐案一致的比例（None = 无可比对案）。"""

    compared_cases: int = Field(description="有 ≥2 次运行记录的案数（一致率分母）")
    consistent_cases: int = Field(description="k 次裁决全部一致的案数（一致率分子）")
    consistency_rate: float | None = None


def pass_k_consistency(decisions: Mapping[str, list[str]]) -> PassKMetrics:
    """统计裁决一致率。

    :param decisions: ``{eval_case_id: [各次运行的裁决]}``；只有 0/1 次记录的案不参与比对。
    """
    compared = consistent = 0
    for run_decisions in decisions.values():
        if len(run_decisions) < 2:
            continue
        compared += 1
        if len(set(run_decisions)) == 1:
            consistent += 1
    return PassKMetrics(
        compared_cases=compared,
        consistent_cases=consistent,
        consistency_rate=(consistent / compared) if compared else None,
    )
