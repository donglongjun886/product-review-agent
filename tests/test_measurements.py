"""测量维度 × 三态建模（``guardrails/measurements.py`` + ``domain/measurement.py``）单测。

锁住本次语义重构的事实侧契约：

- **三态**：``COVERED``（含阴/阳） / ``NOT_MEASURED``（本环境可测却没测） /
  ``UNMEASURABLE``（本环境测不了）—— 三者**不可互相冒充**；
- 阴性测量是有效事实（商家画像全 0 ≠ 商家查无）；干净图产出 ``NEGATIVE`` 而不是零证据；
- 弱相似 0.70~0.85 **不是**阳性；强相似 / Logo / 达阈值商家行为 / 显式 POSITIVE 才是；
- ``required`` 只由案件可观测事实导出，**不读真值**（AST 守卫）；
- ``capabilities_from_tools`` 是环境能力的唯一权威来源（生产视觉桩 → UNMEASURABLE）。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from helpers import all_measureable_caps, covered_evidence, ev, make_case, measurement

from pra.agent.guardrails.measurements import (
    ALWAYS_COVERED_DIMENSIONS,
    capabilities_from_tools,
    coverage_report,
    dimension_strength,
    listing_signal_present,
    positive_dimensions,
    reject_positive_dims,
    required_dimensions,
    required_dimensions_for_reject,
    text_brand_word_hit,
    text_compliance_positive,
    text_evasion_hit,
)
from pra.domain.measurement import (
    ALL_DIMENSIONS,
    DIM_IMAGE_APPEARANCE,
    DIM_LISTING_REGISTRY,
    DIM_MERCHANT_PROFILE,
    DIM_POLICY_CITATION,
    DIM_TEXT_COMPLIANCE,
    MEASUREMENT_TYPE,
    VERDICT_NEGATIVE,
    VERDICT_POSITIVE,
    is_measurement,
    measurement_dimension,
    measurement_ref_id,
    measurement_verdict,
)
from pra.tools.image_analysis.tool import (
    EVIDENCE_STRONG,
    ImageAnalysisTool,
    MockImageAnalysisProvider,
)
from pra.tools.merchant.tool import MerchantTool
from pra.tools.product.tool import ProductTool

MEASUREMENTS_MODULE = Path("src/pra/agent/guardrails/measurements.py")
# required set 绝不允许依赖的真值侧标识符
_FORBIDDEN_IDENTIFIERS = {
    "expected",
    "annotation",
    "family",
    "scene",
    "abstain_label",
    "ground_truth",
    "label",
    "gt",
    "truth",
}


# ---- 领域契约 ----


def test_measurement_roundtrip_and_ref_id_embeds_dimension():
    m = measurement(DIM_IMAGE_APPEARANCE, source_ref="https://x/img1.jpg")
    assert is_measurement(m) and m.type == MEASUREMENT_TYPE
    assert measurement_dimension(m) == DIM_IMAGE_APPEARANCE
    assert measurement_verdict(m) == VERDICT_NEGATIVE
    # ref_id 必须含维度：否则同一源对象在不同维度上的测量会互相吞并（去重指纹口径）
    assert m.ref_id == measurement_ref_id(DIM_IMAGE_APPEARANCE, "https://x/img1.jpg")
    assert m.ref_id.startswith(f"{DIM_IMAGE_APPEARANCE}:")


def test_measurement_readers_are_safe_on_non_measurement_and_bad_verdict():
    plain = ev("PRODUCT_FACT", ref_id="P_1")
    assert measurement_dimension(plain) is None and measurement_verdict(plain) is None
    bad = measurement(DIM_LISTING_REGISTRY)
    bad.extra["verdict"] = "MAYBE"
    assert measurement_verdict(bad) is None  # 非法结论 → 不冒充任何一态


def test_measurement_dedup_key_keeps_positive_apart_from_negative():
    """阳性走各自类型、阴性走 MEASUREMENT ⇒ 两者 key 不同，阳性不会被阴性覆盖。"""
    negative = measurement(DIM_IMAGE_APPEARANCE, source_ref="https://x/img1.jpg")
    positive = ev("IMAGE_SIMILARITY", weight=0.93, ref_id="https://x/img1.jpg")
    assert (negative.type, negative.ref_id) != (positive.type, positive.ref_id)


# ---- 阳性判定 ----


@pytest.mark.parametrize(
    "weight,expected",
    [(0.93, True), (EVIDENCE_STRONG, True), (0.84, False), (0.72, False), (0.69, False)],
)
def test_strong_similarity_is_positive_weak_is_not(weight, expected):
    evs = [ev("IMAGE_SIMILARITY", weight=weight, ref_id="img")]
    assert bool(positive_dimensions(evs).get(DIM_IMAGE_APPEARANCE)) is expected


def test_logo_and_dirty_merchant_are_positive():
    assert positive_dimensions([ev("IMAGE_LOGO", weight=0.7, ref_id="img")])[DIM_IMAGE_APPEARANCE]
    assert positive_dimensions(
        [ev("MERCHANT_HISTORY", value="1 similar / 3 removals / 0 title-relisting, credit=60",
            ref_id="M_1", extra={"removals": 3, "title": 0})]
    )[DIM_MERCHANT_PROFILE]


def test_positive_measurement_declares_dimension_positive():
    m = measurement(DIM_MERCHANT_PROFILE, verdict=VERDICT_POSITIVE)
    assert DIM_MERCHANT_PROFILE in positive_dimensions([m])


# ---- 授权自动 REJECT 的阳性（比"阻塞 PASS 的阳性"更严）----


def test_merchant_dirty_alone_blocks_pass_but_does_not_authorize_reject():
    """仅"商家历史脏"（本 listing 图/文本无确证）：阻塞 PASS，但**不足以**自动拒绝。

    reviewer 语义：疑似规避但无确证 → 克制转人工（GT 家族 ``dirty_brand_missing_cleanimg``）。
    """
    evs = covered_evidence(merchant_removals=5)  # 干净图（无 IMAGE_SIMILARITY）
    assert DIM_MERCHANT_PROFILE in positive_dimensions(evs)  # 阻塞 PASS
    assert reject_positive_dims(evs) == frozenset()  # 不授权 REJECT
    assert listing_signal_present(evs) is False


def test_merchant_dirty_with_listing_signal_authorizes_reject():
    """商家脏 + 本 listing 外观信号 → 成立（GT 家族 ``wsim_dirty``：弱相似亦足够）。"""
    weak = covered_evidence(merchant_removals=5, similarity=0.72)
    assert reject_positive_dims(weak) == frozenset({DIM_MERCHANT_PROFILE})
    assert listing_signal_present(weak) is True

    strong = covered_evidence(merchant_removals=5, similarity=0.93)
    assert reject_positive_dims(strong) == frozenset(
        {DIM_MERCHANT_PROFILE, DIM_IMAGE_APPEARANCE}
    )


def test_listing_level_strong_positive_authorizes_reject_without_merchant():
    evs = covered_evidence(merchant_removals=0, similarity=0.93)
    assert reject_positive_dims(evs) == frozenset({DIM_IMAGE_APPEARANCE})


def test_logo_alone_is_a_listing_signal():
    evs = [ev("IMAGE_LOGO", weight=0.7, ref_id="img")]
    assert listing_signal_present(evs) is True
    assert DIM_IMAGE_APPEARANCE in reject_positive_dims(evs)


def test_text_rules_split_by_platform_severity():
    """R-302 规避词 = 本 listing 文本自证 → 授权 REJECT；R-102 品牌词 = 需调查 → 只阻塞 PASS。"""
    evasion = make_case()
    evasion.product.title = "高仿 1:1 复古跑鞋"
    assert text_evasion_hit(evasion) is True and text_brand_word_hit(evasion) is False

    brand = make_case()
    brand.product.title = "NIKE 联名风格宽松卫衣"
    assert text_evasion_hit(brand) is False and text_brand_word_hit(brand) is True
    assert text_compliance_positive(brand) is True


def test_brand_missing_is_neither_evasion_nor_brand_word():
    case = make_case(brand=None)  # R-301 命中
    assert text_evasion_hit(case) is False and text_brand_word_hit(case) is False


# ---- 三态 ----


def test_all_required_covered_negative_is_complete_and_has_no_missing():
    case = make_case()
    cov = coverage_report(case, covered_evidence(), all_measureable_caps())
    assert set(cov.required) == {
        DIM_LISTING_REGISTRY, DIM_MERCHANT_PROFILE, DIM_TEXT_COMPLIANCE, DIM_IMAGE_APPEARANCE,
    }
    assert cov.complete and cov.negative_only
    assert cov.missing == () and cov.unmeasurable == ()
    assert cov.negative_dims >= {DIM_LISTING_REGISTRY, DIM_IMAGE_APPEARANCE}


def test_merchant_all_zero_is_measured_negative_not_missing():
    """N2a：商家画像全 0 是**有效阴性**，不得当成"没测"。"""
    cov = coverage_report(make_case(), covered_evidence(merchant_removals=0), all_measureable_caps())
    assert DIM_MERCHANT_PROFILE in cov.covered
    assert DIM_MERCHANT_PROFILE not in cov.positive
    assert DIM_MERCHANT_PROFILE not in cov.missing


def test_merchant_not_found_is_not_measured_not_negative():
    """N2b：商家查无（工具 ok=False、零证据）→ NOT_MEASURED，与"全 0"严格区分。"""
    evs = [e for e in covered_evidence() if e.type != "MERCHANT_HISTORY"
           and not (is_measurement(e) and measurement_dimension(e) == DIM_MERCHANT_PROFILE)]
    cov = coverage_report(make_case(), evs, all_measureable_caps())
    assert DIM_MERCHANT_PROFILE in cov.missing
    assert DIM_MERCHANT_PROFILE in cov.missing and DIM_MERCHANT_PROFILE not in cov.covered


def test_image_dimension_missing_when_tool_never_ran():
    """N1：外观维度零测量（工具没跑）→ NOT_MEASURED，不得被当成"测过且阴性"。"""
    evs = [e for e in covered_evidence()
           if not (is_measurement(e) and measurement_dimension(e) == DIM_IMAGE_APPEARANCE)]
    cov = coverage_report(make_case(), evs, all_measureable_caps())
    assert DIM_IMAGE_APPEARANCE in cov.missing


def test_unmeasurable_when_capability_declares_not_available():
    """环境测不了 ⇒ UNMEASURABLE（与"没测"分开归因：重跑无用）。"""
    caps = all_measureable_caps()
    caps[DIM_IMAGE_APPEARANCE] = False
    evs = [e for e in covered_evidence()
           if not (is_measurement(e) and measurement_dimension(e) == DIM_IMAGE_APPEARANCE)]
    cov = coverage_report(make_case(), evs, caps)
    assert cov.unmeasurable == (DIM_IMAGE_APPEARANCE,)
    assert cov.missing == ()


def test_capability_defaults_to_measurable_so_missing_is_not_hidden():
    cov = coverage_report(make_case(), [], None)
    assert cov.missing and not cov.unmeasurable


def test_required_dimensions_follow_observable_facts():
    assert DIM_IMAGE_APPEARANCE in required_dimensions(make_case())  # 带图
    no_image = make_case()
    no_image.product.images = []
    assert DIM_IMAGE_APPEARANCE not in required_dimensions(no_image)
    assert DIM_LISTING_REGISTRY in required_dimensions(no_image)  # 恒必需


def test_policy_citation_is_reject_only_not_pass_required():
    case = make_case()
    assert DIM_POLICY_CITATION not in required_dimensions(case)
    assert DIM_POLICY_CITATION in required_dimensions_for_reject(case)


def test_text_compliance_is_always_covered():
    cov = coverage_report(make_case(), [], all_measureable_caps())
    assert ALWAYS_COVERED_DIMENSIONS == frozenset({DIM_TEXT_COMPLIANCE})
    assert DIM_TEXT_COMPLIANCE in cov.covered


def test_missing_case_requires_nothing():
    assert required_dimensions(None) == ()
    cov = coverage_report(None, [], None)
    assert cov.required == () and cov.complete


# ---- 文本规则维度 ----


def test_text_rule_brand_word_and_evasion_word_are_positive():
    assert text_compliance_positive(make_case()) is False
    brand = make_case()
    brand.product.title = "NIKE 联名风格宽松卫衣"
    assert text_compliance_positive(brand) is True
    evasion = make_case()
    evasion.product.title = "高仿 1:1 复古跑鞋"
    assert text_compliance_positive(evasion) is True


def test_brand_missing_alone_is_not_text_positive():
    """R-301（brand/category 空缺）是**可核验性**信号，由 listing_registry 承担，不是文本阳性。

    否则"品牌空缺但在库可验证"的正常案会被一并挡死。
    """
    case = make_case(brand=None)
    assert text_compliance_positive(case) is False


# ---- 环境能力 ----


def test_capabilities_from_tools_reads_declared_dimensions():
    caps = capabilities_from_tools([ProductTool(), MerchantTool(), ImageAnalysisTool()])
    assert caps[DIM_LISTING_REGISTRY] and caps[DIM_MERCHANT_PROFILE]
    assert caps[DIM_IMAGE_APPEARANCE] is True
    assert caps[DIM_TEXT_COMPLIANCE] is True  # 纯函数维度恒可测
    assert caps[DIM_POLICY_CITATION] is False  # 没有检索工具


def test_production_vision_stub_declares_image_unmeasurable():
    """生产装配：视觉是 Mock 桩 ⇒ 外观维度必须声明为不可测。"""
    stub = ImageAnalysisTool(provider=MockImageAnalysisProvider(), measurement_available=False)
    caps = capabilities_from_tools([stub, MerchantTool()])
    assert caps[DIM_IMAGE_APPEARANCE] is False


# ---- 强度（dc 的事实侧输入） ----


def test_dimension_strength_prefers_positive_then_negative_measurement():
    evs = covered_evidence(similarity=0.93)
    strength = dimension_strength(evs)
    assert strength[DIM_IMAGE_APPEARANCE] == pytest.approx(0.93)
    assert strength[DIM_LISTING_REGISTRY] == pytest.approx(0.6)


# ---- 来源守卫：required set 不得读真值 ----


def test_required_dimension_sources_do_not_reference_ground_truth():
    """AST 守卫：``measurements`` 模块不得出现任何真值侧标识符（防 evaluation gaming）。"""
    tree = ast.parse(MEASUREMENTS_MODULE.read_text(encoding="utf-8"))
    offenders: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_IDENTIFIERS:
            offenders.append((node.id, node.lineno))
        elif isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_IDENTIFIERS:
            offenders.append((node.attr, node.lineno))
        elif isinstance(node, ast.Constant) and node.value in _FORBIDDEN_IDENTIFIERS:
            offenders.append((str(node.value), node.lineno))
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and node.slice.value in _FORBIDDEN_IDENTIFIERS
        ):
            offenders.append((str(node.slice.value), node.lineno))
    assert offenders == [], f"required set 引用了真值侧标识符: {offenders}"


def test_all_declared_dimensions_have_a_requiredness_source():
    """新增维度必须能被 required 派生识别，防"声明了但没人用"的悬空词表。"""
    case = make_case()
    used = set(required_dimensions(case)) | {DIM_POLICY_CITATION}  # PASS 必需 ∪ REJECT 候选
    assert used == set(ALL_DIMENSIONS)
    assert DIM_POLICY_CITATION not in required_dimensions(case)
    assert DIM_POLICY_CITATION in required_dimensions_for_reject(case)
