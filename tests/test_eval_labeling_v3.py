"""打标函数 seam：规则书典型情形逐格 + 真值不变量 + 可复现性 + 规则书冻结。"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from evaluation.dataset.schema import (
    CaseManifestV3,
    MerchantFactsV3,
    RegistryFactsV3,
    TextHitV3,
)
from evaluation.labeling.labeler import label_case, load_rulebook

REPO_ROOT = Path(__file__).resolve().parents[1]
RULEBOOK_PATH = REPO_ROOT / "scripts" / "evaluation" / "labeling" / "rulebook.json"

# 规则书冻结：任何改动都会打破该哈希，必须连同标签口径评审一起改。
RULEBOOK_SHA256 = "6649a2fcc2b4a7cc6ddf4113bd92af371965b28c04b9d5b8a443eea23615ebda"


def _manifest(**over) -> CaseManifestV3:
    """默认干净案埋点清单；``over`` 覆盖差异字段。"""
    base = {
        "text_hits": [],
        "snapshot_brand": "潮动",
        "snapshot_gaps": [],
        "registry_mismatch": False,
        "merchant": MerchantFactsV3(
            merchant_id="M_T1", removals=0, title_relisting_count=0,
            similar_product_count=0, credit_score=90,
        ),
        "registry": RegistryFactsV3(
            product_id="P_T1", category="女鞋/运动鞋", brand="潮动", version=1, status="ON_SALE",
        ),
    }
    base.update(over)
    return CaseManifestV3(**base)


def _dirty(m: CaseManifestV3, removals: int = 5, title: int = 0) -> CaseManifestV3:
    """把埋点清单的商家数值改为指定下架/改标题计数。"""
    return m.model_copy(
        update={"merchant": m.merchant.model_copy(
            update={"removals": removals, "title_relisting_count": title})}
    )


def _hit(term: str, field: str = "title") -> TextHitV3:
    return TextHitV3(field=field, term=term)


@pytest.fixture(scope="module")
def rulebook() -> dict:
    return load_rulebook()


class TestRulebookTypicalCases:
    """规则书判定表：每种典型情形一格。"""

    def test_blacklist_brand_reject(self, rulebook) -> None:
        """黑名单品牌 → REJECT / [POTENTIAL_IP_RISK] / POLICY_1.1 / 无需工具 / 规则命中+可引用证据。"""
        exp = label_case(_manifest(snapshot_brand="某违禁品牌"), rulebook)
        assert exp.decision == "REJECT"
        assert exp.risk_type == ["POTENTIAL_IP_RISK"]
        assert exp.expected_policies == ["POLICY_1.1"]
        assert exp.expected_tools == []
        assert exp.expected_evidence == ["RULE_HIT", "POLICY_REF"]

    def test_evasion_word_dirty_merchant_aggravated(self, rulebook) -> None:
        """规避词 + 商家脏 → 从重 REJECT / [IP, EVASION] / POLICY_1.2 + POLICY_4.5。"""
        m = _dirty(_manifest(text_hits=[_hit("高仿")]), removals=5, title=3)
        exp = label_case(m, rulebook)
        assert exp.decision == "REJECT"
        assert exp.risk_type == ["POTENTIAL_IP_RISK", "EVASION_PATTERN"]
        assert exp.expected_policies == ["POLICY_1.2", "POLICY_4.5"]
        assert "MerchantTool" in exp.expected_tools
        assert "PolicySearchTool" in exp.expected_tools
        assert exp.expected_evidence == ["RULE_HIT", "POLICY_REF", "MERCHANT_HISTORY"]

    def test_evasion_word_clean_merchant_reject(self, rulebook) -> None:
        """规避词 + 商家干净 → REJECT（文本自证 + POLICY_1.2 可引）/ [IP]。"""
        exp = label_case(_manifest(text_hits=[_hit("同款", "description")]), rulebook)
        assert exp.decision == "REJECT"
        assert exp.risk_type == ["POTENTIAL_IP_RISK"]
        assert exp.expected_policies == ["POLICY_1.2"]
        assert exp.expected_tools == ["PolicySearchTool"]
        assert exp.expected_evidence == ["RULE_HIT", "POLICY_REF"]

    def test_brand_word_only_human_review(self, rulebook) -> None:
        """只命中品牌词（无规避词、非黑名单）→ HUMAN_REVIEW / [IP] / SHOULD_ABSTAIN。"""
        exp = label_case(_manifest(text_hits=[_hit("NIKE")]), rulebook)
        assert exp.decision == "HUMAN_REVIEW"
        assert exp.abstain_label == "SHOULD_ABSTAIN"
        assert exp.risk_type == ["POTENTIAL_IP_RISK"]
        assert exp.expected_policies == []
        assert exp.expected_evidence == ["RULE_HIT", "MERCHANT_HISTORY", "PRODUCT_FACT"]

    def test_dirty_merchant_only_human_review(self, rulebook) -> None:
        """只有商家脏（无文本命中）→ HUMAN_REVIEW / [EVASION_PATTERN]（画像不得单独拒）。"""
        exp = label_case(_dirty(_manifest(), removals=4), rulebook)
        assert exp.decision == "HUMAN_REVIEW"
        assert exp.risk_type == ["EVASION_PATTERN"]
        assert exp.expected_policies == []
        assert exp.expected_evidence == ["MERCHANT_HISTORY", "PRODUCT_FACT"]

    def test_brand_word_plus_dirty_merchant(self, rulebook) -> None:
        """品牌词 + 商家脏（多信号）→ HUMAN_REVIEW / [IP, EVASION]。"""
        exp = label_case(_dirty(_manifest(text_hits=[_hit("LV")])), rulebook)
        assert exp.decision == "HUMAN_REVIEW"
        assert exp.risk_type == ["POTENTIAL_IP_RISK", "EVASION_PATTERN"]

    def test_brand_gap_human_review(self, rulebook) -> None:
        """品牌空缺 → HUMAN_REVIEW（测量缺口），无风险类型。"""
        m = _manifest(snapshot_brand=None, snapshot_gaps=["brand"])
        m = m.model_copy(update={"registry": m.registry.model_copy(update={"brand": None})})
        exp = label_case(m, rulebook)
        assert exp.decision == "HUMAN_REVIEW"
        assert exp.abstain_label == "SHOULD_ABSTAIN"
        assert exp.risk_type == []
        assert exp.expected_evidence == ["MEASUREMENT", "MERCHANT_HISTORY", "PRODUCT_FACT"]

    def test_category_gap_human_review(self, rulebook) -> None:
        """类目空缺 → HUMAN_REVIEW（测量缺口）。"""
        exp = label_case(_manifest(snapshot_gaps=["category"]), rulebook)
        assert exp.decision == "HUMAN_REVIEW"
        assert exp.risk_type == []

    def test_registry_mismatch_human_review(self, rulebook) -> None:
        """在库事实与快照不一致 → HUMAN_REVIEW（ProductTool 取证发现的冲突）。"""
        m = _manifest(registry_mismatch=True)
        m = m.model_copy(update={"registry": m.registry.model_copy(update={"brand": None})})
        exp = label_case(m, rulebook)
        assert exp.decision == "HUMAN_REVIEW"
        assert "ProductTool" in exp.expected_tools
        assert exp.expected_evidence == ["PRODUCT_FACT"]

    def test_merchant_count_two_passes(self, rulebook) -> None:
        """临界：商家计数正好 2（阈值下）且无文本命中 → PASS。"""
        exp = label_case(_dirty(_manifest(), removals=2, title=2), rulebook)
        assert exp.decision == "PASS"
        assert exp.risk_type == []
        assert exp.expected_tools == ["MerchantTool", "ProductTool"]
        assert exp.expected_evidence == ["MERCHANT_HISTORY", "PRODUCT_FACT"]

    def test_merchant_count_three_is_dirty(self, rulebook) -> None:
        """临界：商家计数正好 3 → 商家脏 → 无文本时 HUMAN_REVIEW。"""
        exp = label_case(_dirty(_manifest(), removals=3, title=0), rulebook)
        assert exp.decision == "HUMAN_REVIEW"
        assert exp.risk_type == ["EVASION_PATTERN"]

    def test_title_relisting_alone_makes_dirty(self, rulebook) -> None:
        """title_relisting_count 单项达阈值同样构成商家脏。"""
        exp = label_case(_dirty(_manifest(), removals=0, title=3), rulebook)
        assert exp.decision == "HUMAN_REVIEW"

    def test_clean_case_pass(self, rulebook) -> None:
        """干净案 → PASS / [] / 必需维度工具 [MerchantTool, ProductTool]。"""
        exp = label_case(_manifest(), rulebook)
        assert exp.decision == "PASS"
        assert exp.abstain_label == "AUTO_DECIDABLE"
        assert exp.risk_type == []
        assert exp.expected_policies == []
        assert exp.expected_tools == ["MerchantTool", "ProductTool"]
        assert exp.expected_evidence == ["MERCHANT_HISTORY", "PRODUCT_FACT"]

    def test_blacklist_dominates_evasion_word(self, rulebook) -> None:
        """黑名单品牌 + 规避词叠加 → 硬规则支配（REJECT / [IP] / POLICY_1.1 / 无工具）。"""
        exp = label_case(
            _manifest(snapshot_brand="山寨", text_hits=[_hit("高仿")]), rulebook
        )
        assert exp.decision == "REJECT"
        assert exp.risk_type == ["POTENTIAL_IP_RISK"]
        assert exp.expected_policies == ["POLICY_1.1"]
        assert exp.expected_tools == []
        assert exp.expected_evidence == ["RULE_HIT", "POLICY_REF"]


class TestTruthInvariants:
    """真值不变量：对判定表全 battery 成立。"""

    def _battery(self) -> list[CaseManifestV3]:
        return [
            _manifest(),
            _manifest(snapshot_brand="某违禁品牌"),
            _manifest(text_hits=[_hit("复刻")]),
            _dirty(_manifest(text_hits=[_hit("1:1")]), removals=9, title=4),
            _manifest(text_hits=[_hit("GUCCI")]),
            _dirty(_manifest(text_hits=[_hit("ADIDAS")]), removals=3),
            _dirty(_manifest(), removals=6),
            _dirty(_manifest(), removals=2, title=1),
            _manifest(snapshot_brand=None, snapshot_gaps=["brand"]),
            _manifest(snapshot_gaps=["category"]),
            _manifest(registry_mismatch=True),
            _manifest(snapshot_brand="山寨", text_hits=[_hit("原单")]),
        ]

    def test_abstain_iff_human_review(self, rulebook) -> None:
        """HUMAN_REVIEW ⇔ SHOULD_ABSTAIN；自动判 ⇔ AUTO_DECIDABLE。"""
        for m in self._battery():
            exp = label_case(m, rulebook)
            if exp.decision == "HUMAN_REVIEW":
                assert exp.abstain_label == "SHOULD_ABSTAIN"
            else:
                assert exp.abstain_label == "AUTO_DECIDABLE"

    def test_pass_has_no_risk(self, rulebook) -> None:
        """PASS ⇒ risk_type=[] / 无应引政策。"""
        for m in self._battery():
            exp = label_case(m, rulebook)
            if exp.decision == "PASS":
                assert exp.risk_type == []
                assert exp.expected_policies == []

    def test_reject_has_citable_policy(self, rulebook) -> None:
        """REJECT ⇒ 必有可引用政策依据（正确性契约），且期望证据含可引用依据类。"""
        for m in self._battery():
            exp = label_case(m, rulebook)
            if exp.decision == "REJECT":
                assert exp.expected_policies
                assert "POLICY_REF" in exp.expected_evidence

    def test_risk_type_vocab_is_two_values(self, rulebook) -> None:
        """风险类型词表只有 POTENTIAL_IP_RISK / EVASION_PATTERN 两值。"""
        for m in self._battery():
            exp = label_case(m, rulebook)
            assert set(exp.risk_type) <= {"POTENTIAL_IP_RISK", "EVASION_PATTERN"}

    def test_expected_tools_vocab(self, rulebook) -> None:
        """期望工具只从生产工具集四值中取。"""
        for m in self._battery():
            exp = label_case(m, rulebook)
            assert set(exp.expected_tools) <= {
                "ProductTool", "MerchantTool", "CaseSearchTool", "PolicySearchTool",
            }

    def test_expected_evidence_vocab_and_nonempty(self, rulebook) -> None:
        """期望证据只从受控词表六值中取，且每个真值案都有标注（全量进比对）。"""
        for m in self._battery():
            exp = label_case(m, rulebook)
            assert exp.expected_evidence
            assert set(exp.expected_evidence) <= {
                "RULE_HIT", "MERCHANT_HISTORY", "PRODUCT_FACT",
                "POLICY_REF", "CASE_PRECEDENT", "MEASUREMENT",
            }


class TestReproducibility:
    """可复现性：同输入必得同真值。"""

    def test_same_manifest_same_truth(self) -> None:
        rb1 = load_rulebook()
        rb2 = load_rulebook()
        assert rb1 == rb2
        m = _dirty(_manifest(text_hits=[_hit("高仿"), _hit("同款", "description")]), removals=5)
        assert label_case(m, rb1).model_dump() == label_case(m, rb2).model_dump()


class TestRulebookFrozen:
    """规则书冻结为数据文件：内容与词表副本钉死。"""

    def test_rulebook_hash_pinned(self) -> None:
        digest = hashlib.sha256(RULEBOOK_PATH.read_bytes()).hexdigest()
        assert digest == RULEBOOK_SHA256, "规则书已冻结；变更须连同标签口径评审并更新本钉值"

    def test_rulebook_vocab_mirrors_frozen_terms(self, rulebook) -> None:
        assert set(rulebook["blacklisted_brands"]) == {"某违禁品牌", "山寨"}
        assert set(rulebook["brand_terms"]) == {"NIKE", "ADIDAS", "GUCCI", "LOUIS VUITTON", "LV"}
        assert set(rulebook["evasion_terms"]) == {"同款", "复刻", "高仿", "1:1", "原单"}
        assert rulebook["merchant_dirty_min"] == 3
