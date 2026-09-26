"""测量维度 × 三态建模（``guardrails/measurements.py`` + ``domain/measurement.py``）单测。

锁住事实侧契约：

- **三态**：``COVERED``（含阴/阳） / ``NOT_MEASURED``（本环境可测却没测） /
  ``UNMEASURABLE``（本环境测不了）—— 三者**不可互相冒充**；
- 阴性测量是有效事实（商家画像全 0 ≠ 商家查无）；
- 达阈值商家行为才是阳性；
- ``required`` 只由案件可观测事实导出，**不读真值**（AST 守卫）；
- ``capabilities_from_tools`` 是环境能力的唯一权威来源。
"""

from __future__ import annotations

import ast
from pathlib import Path

from helpers import (
    WalkthroughBackend,
    all_measureable_caps,
    covered_evidence,
    ev,
    make_case,
    measurement,
)
from inmemory_world import InMemoryMerchantRepository, InMemoryProductRepository

from pra.agent.guardrails.measurements import (
    ALWAYS_COVERED_DIMENSIONS,
    RULE_BRAND_WORD,
    RULE_EVASION_WORD,
    capabilities_from_tools,
    coverage_report,
    positive_dimensions,
    required_dimensions,
    rule_hit_ids,
)
from pra.domain.measurement import (
    ALL_DIMENSIONS,
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
    m = measurement(DIM_MERCHANT_PROFILE, source_ref="M_1")
    assert is_measurement(m) and m.type == MEASUREMENT_TYPE
    assert measurement_dimension(m) == DIM_MERCHANT_PROFILE
    assert measurement_verdict(m) == VERDICT_NEGATIVE
    # ref_id 必须含维度：否则同一源对象在不同维度上的测量会互相吞并（去重指纹口径）
    assert m.ref_id == measurement_ref_id(DIM_MERCHANT_PROFILE, "M_1")
    assert m.ref_id.startswith(f"{DIM_MERCHANT_PROFILE}:")


def test_measurement_readers_are_safe_on_non_measurement_and_bad_verdict():
    plain = ev("PRODUCT_FACT", ref_id="P_1")
    assert measurement_dimension(plain) is None and measurement_verdict(plain) is None
    bad = measurement(DIM_LISTING_REGISTRY)
    bad.extra["verdict"] = "MAYBE"
    assert measurement_verdict(bad) is None  # 非法结论 → 不冒充任何一态


def test_measurement_dedup_key_keeps_positive_apart_from_negative():
    """阳性走各自类型、阴性走 MEASUREMENT ⇒ 两者 key 不同，阳性不会被阴性覆盖。"""
    negative = measurement(DIM_MERCHANT_PROFILE, source_ref="M_1")
    positive = ev("MERCHANT_HISTORY", weight=0.85, ref_id="M_1",
                  extra={"removals": 5, "title": 0})
    assert (negative.type, negative.ref_id) != (positive.type, positive.ref_id)


# ---- 阳性判定 ----


def test_dirty_merchant_is_positive():
    assert positive_dimensions(
        [ev("MERCHANT_HISTORY", value="1 similar / 3 removals / 0 title-relisting, credit=60",
            ref_id="M_1", extra={"removals": 3, "title": 0})]
    )[DIM_MERCHANT_PROFILE]


def test_positive_measurement_declares_dimension_positive():
    m = measurement(DIM_MERCHANT_PROFILE, verdict=VERDICT_POSITIVE)
    assert DIM_MERCHANT_PROFILE in positive_dimensions([m])


def test_text_rules_split_by_platform_severity():
    """R-302 规避词 = 本 listing 文本自证 → 授权 REJECT；R-102 品牌词 = 需调查 → 只阻塞 PASS。"""
    evasion = make_case()
    evasion.product.title = "高仿 1:1 复古跑鞋"
    assert RULE_EVASION_WORD in rule_hit_ids(evasion)
    assert RULE_BRAND_WORD not in rule_hit_ids(evasion)

    brand = make_case()
    brand.product.title = "NIKE 联名风格宽松卫衣"
    assert RULE_BRAND_WORD in rule_hit_ids(brand)
    assert RULE_EVASION_WORD not in rule_hit_ids(brand)


def test_brand_missing_is_neither_evasion_nor_brand_word():
    case = make_case(brand=None)  # R-301 命中
    hits = rule_hit_ids(case)
    assert RULE_EVASION_WORD not in hits and RULE_BRAND_WORD not in hits


# ---- 三态 ----


def test_all_required_covered_negative_is_complete_and_has_no_missing():
    case = make_case()
    cov = coverage_report(case, covered_evidence(), all_measureable_caps())
    assert set(cov.required) == {
        DIM_LISTING_REGISTRY, DIM_MERCHANT_PROFILE, DIM_TEXT_COMPLIANCE,
    }
    assert cov.missing == () and cov.unmeasurable == ()
    assert not cov.positive


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


def test_unmeasurable_when_capability_declares_not_available():
    """环境测不了 ⇒ UNMEASURABLE（与"没测"分开归因：重跑无用）。"""
    caps = all_measureable_caps()
    caps[DIM_MERCHANT_PROFILE] = False
    evs = [e for e in covered_evidence()
           if not (is_measurement(e) and measurement_dimension(e) == DIM_MERCHANT_PROFILE)]
    cov = coverage_report(make_case(), evs, caps)
    assert cov.unmeasurable == (DIM_MERCHANT_PROFILE,)
    assert cov.missing == ()


def test_capability_defaults_to_measurable_so_missing_is_not_hidden():
    cov = coverage_report(make_case(), [], None)
    assert cov.missing and not cov.unmeasurable


def test_policy_citation_is_not_pass_required():
    """``policy_citation`` 不进 PASS 必需集（它是「可引用依据可得」，与证明无风险无关）。"""
    assert DIM_POLICY_CITATION not in required_dimensions(make_case())


def test_text_compliance_is_always_covered():
    cov = coverage_report(make_case(), [], all_measureable_caps())
    assert ALWAYS_COVERED_DIMENSIONS == frozenset({DIM_TEXT_COMPLIANCE})
    assert DIM_TEXT_COMPLIANCE in cov.covered


def test_missing_case_requires_nothing():
    assert required_dimensions(None) == ()
    cov = coverage_report(None, [], None)
    assert cov.required == () and cov.missing == () and cov.unmeasurable == ()


# ---- 文本规则维度 ----


def test_text_rule_brand_word_and_evasion_word_are_positive():
    assert rule_hit_ids(make_case()) == frozenset({"R-301"})
    brand = make_case()
    brand.product.title = "NIKE 联名风格宽松卫衣"
    assert rule_hit_ids(brand) & {RULE_BRAND_WORD, RULE_EVASION_WORD}
    evasion = make_case()
    evasion.product.title = "高仿 1:1 复古跑鞋"
    assert rule_hit_ids(evasion) & {RULE_BRAND_WORD, RULE_EVASION_WORD}


def test_brand_missing_alone_is_not_text_positive():
    """R-301（brand/category 空缺）是**可核验性**信号，由 listing_registry 承担，不是文本阳性。

    否则"品牌空缺但在库可验证"的正常案会被一并挡死。
    """
    case = make_case(brand=None)
    assert not (rule_hit_ids(case) & {RULE_BRAND_WORD, RULE_EVASION_WORD})


# ---- 环境能力 ----


def test_capabilities_from_tools_reads_declared_dimensions():
    caps = capabilities_from_tools(
        [
            ProductTool(repo=InMemoryProductRepository()),
            MerchantTool(repo=InMemoryMerchantRepository()),
        ]
    )
    assert caps[DIM_LISTING_REGISTRY] and caps[DIM_MERCHANT_PROFILE]
    assert caps[DIM_TEXT_COMPLIANCE] is True  # 纯函数维度恒可测
    assert caps[DIM_POLICY_CITATION] is False  # 没有检索工具


async def test_graph_writes_tool_derived_capabilities_into_state():
    """图装配把工具集导出的能力表写进 state（唯一写入点）—— Gate/收敛判定据此分三态。

    反证：若注入丢失（state 里没有能力表或表与工具集不一致），缺测维度会回落"可测"
    ⇒ 生产环境的"测不出"被误判成"可测却没测"（NOT_MEASURED，可补救），而不是
    UNMEASURABLE（环境缺失，重跑无用）。
    """
    from inmemory_world import build_inmemory_tools

    from pra.agent.graph import build_agent_graph
    from pra.agent.state import build_initial_state

    tools = build_inmemory_tools()
    graph = build_agent_graph(tools=tools, llm=WalkthroughBackend(), checkpointer=None)
    state = await graph.ainvoke(
        build_initial_state(make_case()),
        {"configurable": {"thread_id": "caps-tool-derived"}},
    )
    assert state["measurement_capabilities"] == capabilities_from_tools(tools)
    assert state["measurement_capabilities"][DIM_LISTING_REGISTRY] is True
    assert state["measurement_capabilities"][DIM_MERCHANT_PROFILE] is True


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
    """新增维度必须能被 required 派生或能力表识别，防"声明了但没人用"的悬空词表。"""
    case = make_case()
    required = set(required_dimensions(case))
    assert required | {DIM_POLICY_CITATION} == set(ALL_DIMENSIONS)
    assert DIM_POLICY_CITATION not in required
