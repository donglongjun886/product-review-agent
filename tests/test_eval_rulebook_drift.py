"""跨源一致性守护：冻结规则书 / 真值证据词表与生产代码的漂移即显式测试失败。

规则书（``scripts/evaluation/labeling/labeling`` 的 rulebook.json）按设计自带冻结副本、
不随生产代码漂移（独立性 seam 守住）；此处只校验**会导致静默错误的部分**：
词表 / 阈值若与生产当前值不一致，评测口径就与线上行为脱节 —— 显式失败，不静默混用。
"""

from __future__ import annotations

from typing import get_args

from evaluation.dataset.schema import EvidenceTypeV3, ToolNameV3
from evaluation.labeling.labeler import load_rulebook

from pra.domain.measurement import MEASUREMENT_TYPE, MERCHANT_DIRTY_MIN
from pra.screening.engine import RULE_EVIDENCE_TYPE
from pra.screening.rule_engine import terms
from pra.tools.case_search.tool import CASE_PRECEDENT_TYPE
from pra.tools.merchant.tool import MERCHANT_HISTORY_TYPE
from pra.tools.policy_search.tool import POLICY_REF_TYPE
from pra.tools.product.tool import PRODUCT_FACT_TYPE


def test_rulebook_terms_and_threshold_mirror_production() -> None:
    """规则书黑名单 / 品牌词 / 规避词 / 商家脏阈值与生产当前值一致（漂移即改规则书并评审）。"""
    rulebook = load_rulebook()
    assert set(rulebook["blacklisted_brands"]) == set(terms.BLACKLISTED_BRANDS)
    assert set(rulebook["brand_terms"]) == set(terms.BRAND_TERMS)
    assert set(rulebook["evasion_terms"]) == set(terms.EVASION_TERMS)
    assert rulebook["merchant_dirty_min"] == MERCHANT_DIRTY_MIN


def test_evidence_type_vocab_mirrors_production() -> None:
    """真值期望证据类型词表 = 生产证据 type 常量全集（初筛规则命中 + 四工具 + 测量）。"""
    production_vocab = {
        RULE_EVIDENCE_TYPE,
        MERCHANT_HISTORY_TYPE,
        PRODUCT_FACT_TYPE,
        POLICY_REF_TYPE,
        CASE_PRECEDENT_TYPE,
        MEASUREMENT_TYPE,
    }
    assert set(get_args(EvidenceTypeV3)) == production_vocab


def test_rulebook_evidence_rules_stay_in_vocab() -> None:
    """规则书证据推导规则的取值不超出受控词表（防规则书写出词表外的类型）。"""
    rulebook = load_rulebook()
    vocab = set(get_args(EvidenceTypeV3))
    rules = rulebook["evidence_rules"]
    assert rules["rule_text_hit"] in vocab
    assert rules["snapshot_gap"] in vocab
    assert rules["reject_citable"] in vocab
    assert set(rules["tool_to_evidence"].values()) <= vocab
    assert set(rules["tool_to_evidence"]) == set(get_args(ToolNameV3))
