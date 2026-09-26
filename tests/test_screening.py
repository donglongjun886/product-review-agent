"""Screening 三分流（``pra/screening/**`` + ``persist_service.process_review``）单测 —— 不连真库。"""

from __future__ import annotations

from datetime import datetime

import pytest

import pra.infra.persist_service as ps
from pra.domain.models import (
    Decision,
    ProductImage,
    ProductInfo,
    ProductReviewCase,
    ReviewDecision,
    RiskLevel,
    RiskType,
)
from pra.screening.engine import RULE_EVIDENCE_WEIGHT, RuleHit, rule_evidence, triage
from pra.screening.rule_engine import terms
from pra.screening.rule_engine.rules import DEFAULT_RULES


def _case(
    *,
    case_id: str,
    brand: str | None,
    title: str,
    description: str,
    category: str,
) -> ProductReviewCase:
    product = ProductInfo(
        product_id=f"P_{case_id}",
        title=title,
        description=description,
        category=category,
        brand=brand,
        images=[ProductImage(url="https://cdn.example.com/img1.jpg", source="主图")],
        listing_time=datetime(2024, 9, 6, 14, 0, 0),
        version=1,
    )
    return ProductReviewCase(
        case_id=case_id,
        product=product,
        merchant_id="M_TEST",
        event_type="NEW_LISTING",
    )


_CLEAN = _case(
    case_id="CASE_PASS",
    brand="自营品牌A",
    title="北欧风陶瓷花瓶 简约桌面装饰",
    description="釉面细腻，清新雅致。",
    category="家居日用",
)
_COMPLEX_HIGH_RISK = _case(
    case_id="CASE_CMPLX_HIGHRISK",
    brand=None,
    title="新款厚底复古跑鞋",
    description="复古厚底设计，舒适百搭。",
    category="女鞋/运动鞋",
)
_COMPLEX_EVASION = _case(
    case_id="CASE_CMPLX_EVASION",
    brand="自营品牌A",
    title="设计师简约布艺沙发",
    description="同款复刻工艺，1:1 细节还原。",
    category="家居日用",
)
_COMPLEX_BRANDTERM = _case(
    case_id="CASE_CMPLX_BRANDTERM",
    brand=None,
    title="NIKE 新款复古跑鞋 轻便透气",
    description="舒适运动。",
    category="女鞋/运动鞋",
)
_LOWERCASE_BRANDTERM = _case(
    case_id="CASE_CMPLX_LOWER",
    brand=None,
    title="nike air 复古跑鞋",
    description="舒适运动。",
    category="女鞋/运动鞋",
)
_REJECT_BLACKLIST = _case(
    case_id="CASE_REJECT_BLACKLIST",
    brand="某违禁品牌",
    title="复古运动鞋 轻便透气",
    description="舒适运动。",
    category="女鞋/运动鞋",
)
_CLEAN_NO_BRAND_EMPTY = _case(
    case_id="CASE_CMPLX_BRAND_EMPTY",
    brand="",
    title="北欧风陶瓷花瓶 简约桌面装饰",
    description="釉面细腻，清新雅致。",
    category="家居日用",
)
_CLEAN_NO_BRAND_NONE = _case(
    case_id="CASE_CMPLX_BRAND_NONE_NONRISK",
    brand=None,
    title="北欧风陶瓷花瓶 简约桌面装饰",
    description="釉面细腻，清新雅致。",
    category="家居日用",
)
_CLEAN_CATEGORY_MISSING = _case(
    case_id="CASE_CMPLX_CATEGORY_MISSING",
    brand="自营品牌A",
    title="北欧风陶瓷花瓶 简约桌面装饰",
    description="釉面细腻，清新雅致。",
    category="",
)
_SILVER_SUBSTRING_TITLE = _case(
    case_id="CASE_SILVER_925",
    brand="自营品牌A",
    title="925 SILVER 银饰项链 简约锁骨链",
    description="S925 纯银，工艺细腻。",
    category="家居日用",
)
_LV_REAL_BRAND_TERM = _case(
    case_id="CASE_LV_REAL",
    brand="自营品牌A",
    title="LV 经典老花手袋 通勤单肩包",
    description="经典老花纹理，配皮饰边。",
    category="家居日用",
)
_COMPLEX_MULTI_HIT_NONRISK = _case(
    case_id="CASE_CMPLX_MULTI_NONRISK",
    brand=None,
    title="北欧风布艺沙发 简约设计",
    description="同款复刻工艺，1:1 细节还原。",
    category="家居日用",
)


def test_triage_pass_clean_case():
    """brand 明确 + 标题/描述干净 + 类目非高危 → PASS（零命中）。"""
    t = triage(_CLEAN)
    assert t.verdict == "PASS"
    assert t.hits == []


def test_triage_complex_brand_none_high_risk_category():
    """brand=None + 类目命中高危前缀（女鞋/运动鞋）→ COMPLEX，命中 R-301。"""
    t = triage(_COMPLEX_HIGH_RISK)
    assert t.verdict == "COMPLEX"
    assert [h.rule_id for h in t.hits] == ["R-301"]
    assert "高危类目前缀" in t.hits[0].detail


def test_triage_complex_evasion_terms():
    """描述含规避词（同款/复刻/1:1）→ COMPLEX，命中 R-302。"""
    t = triage(_COMPLEX_EVASION)
    assert t.verdict == "COMPLEX"
    assert [h.rule_id for h in t.hits] == ["R-302"]
    assert "同款" in t.hits[0].detail and "复刻" in t.hits[0].detail


def test_triage_reject_blacklisted_brand():
    """内置 demo 黑名单：brand ∈ terms.BLACKLISTED_BRANDS → REJECT（R-101）。"""
    assert "某违禁品牌" in terms.BLACKLISTED_BRANDS
    t = triage(_REJECT_BLACKLIST)
    assert t.verdict == "REJECT"
    assert [h.rule_id for h in t.hits] == ["R-101"]
    assert "黑名单" in t.hits[0].detail


def test_triage_brand_term_in_title_is_complex():
    """标题含 BRAND_TERMS（NIKE）→ COMPLEX 交 Agent。"""
    t = triage(_COMPLEX_BRANDTERM)
    assert t.verdict == "COMPLEX"
    assert [h.rule_id for h in t.hits] == ["R-102", "R-301"]
    assert "NIKE" in t.hits[0].detail


def test_triage_brand_term_matching_brand_is_complex():
    """brand=NIKE（未上黑名单）+ 标题含 NIKE → COMPLEX。"""
    case = _case(
        case_id="CASE_BRAND_TITLE_SAME",
        brand="NIKE",
        title="NIKE 官方旗舰店 运动跑鞋 轻便透气",
        description="舒适运动。",
        category="女鞋/运动鞋",
    )
    t = triage(case)
    assert t.verdict == "COMPLEX"
    assert [h.rule_id for h in t.hits] == ["R-102"]


def test_triage_brand_term_case_insensitive_complex():
    """BRAND_TERMS 大小写不敏感：小写 nike 同样命中 R-102（→ COMPLEX）。"""
    t = triage(_LOWERCASE_BRANDTERM)
    assert t.verdict == "COMPLEX"
    assert [h.rule_id for h in t.hits] == ["R-102", "R-301"]


def test_triage_r102_short_term_word_boundary_no_misfire():
    """R-102 词边界：英文词内部 "lv" 子串（SILVER/valve 等）不触发 —— 925 SILVER 银饰项链零命中 PASS。"""
    t = triage(_SILVER_SUBSTRING_TITLE)
    assert t.verdict == "PASS"
    assert t.hits == []


def test_triage_r102_real_brand_term_still_hits_complex():
    """真正的 "LV" 品牌词（独立成词）仍命中 R-102 → COMPLEX。"""
    t = triage(_LV_REAL_BRAND_TERM)
    assert t.verdict == "COMPLEX"
    assert [h.rule_id for h in t.hits] == ["R-102"]
    assert "LV" in t.hits[0].detail


def test_triage_r102_lv_adjacent_to_chinese_still_hits_complex():
    """中文邻接的真实 "lv"（新款lv手袋）仍命中 R-102 → COMPLEX。"""
    case = _case(
        case_id="CASE_LV_CJK",
        brand="自营品牌A",
        title="新款lv经典手袋 气质通勤",
        description="皮质细腻，走线工整。",
        category="家居日用",
    )
    t = triage(case)
    assert t.verdict == "COMPLEX"
    assert [h.rule_id for h in t.hits] == ["R-102"]
    assert "LV" in t.hits[0].detail


def test_triage_reject_priority_over_complex():
    """REJECT 优先于 COMPLEX：R-101（黑名单 REJECT）+ R-102（品牌词 COMPLEX）同轮 → 终裁 REJECT。"""
    case = _case(
        case_id="CASE_REJECT_PRIORITY",
        brand="某违禁品牌",
        title="NIKE 新款复古跑鞋",
        description="舒适运动。",
        category="女鞋/运动鞋",
    )
    t = triage(case)
    assert t.verdict == "REJECT"
    assert [h.rule_id for h in t.hits] == ["R-101", "R-102"]


def test_triage_hits_collected_in_rule_order():
    """同轮多命中全收集，hits 按规则声明序（R-102→R-301→R-302，均 COMPLEX）。"""
    case = _case(
        case_id="CASE_MULTI_HIT",
        brand=None,
        title="NIKE 新款跑鞋",
        description="同款复刻工艺 1:1 细节。",
        category="女鞋/运动鞋",
    )
    t = triage(case)
    assert t.verdict == "COMPLEX"
    assert [h.rule_id for h in t.hits] == ["R-102", "R-301", "R-302"]
    assert "同款" in t.hits[2].detail and "复刻" in t.hits[2].detail


def test_triage_brand_empty_string_is_complex_not_pass():
    """brand=''（空串空缺）→ 干净文本也 COMPLEX，不 PASS 直放。"""
    t = triage(_CLEAN_NO_BRAND_EMPTY)
    assert t.verdict == "COMPLEX"
    assert [h.rule_id for h in t.hits] == ["R-301"]
    assert "brand 空缺('')" in t.hits[0].detail
    assert "未命中高危前缀" in t.hits[0].detail


def test_triage_brand_none_non_high_risk_category_is_complex():
    """brand=None + 类目不在高危表 → COMPLEX，不 PASS 直放。"""
    t = triage(_CLEAN_NO_BRAND_NONE)
    assert t.verdict == "COMPLEX"
    assert [h.rule_id for h in t.hits] == ["R-301"]


def test_triage_category_missing_is_complex():
    """brand 明确但 category 空缺 → COMPLEX。"""
    t = triage(_CLEAN_CATEGORY_MISSING)
    assert t.verdict == "COMPLEX"
    assert [h.rule_id for h in t.hits] == ["R-301"]
    assert "类目空缺" in t.hits[0].detail


def test_blacklist_disjoint_from_brand_terms():
    """黑名单与 BRAND_TERMS 不得相交。"""
    assert terms.BLACKLISTED_BRANDS.isdisjoint(terms.BRAND_TERMS)


def test_default_rules_declaration_order():
    """DEFAULT_RULES 声明序与动作（kind）。"""
    assert [r.rule_id for r in DEFAULT_RULES] == ["R-101", "R-102", "R-301", "R-302"]
    assert [r.kind for r in DEFAULT_RULES] == ["REJECT", "COMPLEX", "COMPLEX", "COMPLEX"]


def test_rule_evidence_shape():
    """rule_evidence 构造确定性直判证据（只构造不落库）。"""
    hit = RuleHit(rule_id="R-102", name="品牌词命中", detail="标题/描述命中品牌词: NIKE")
    ev = rule_evidence(hit)
    assert ev.type == "RULE_HIT"
    assert ev.source == "ScreeningRuleEngine"
    assert ev.weight == RULE_EVIDENCE_WEIGHT
    assert ev.ref_id is None
    assert ev.extra.get("rule_id") == "R-102"
    assert ev.value.startswith("R-102")
    assert "品牌词命中" in ev.value
    assert "标题/描述命中品牌词: NIKE" in ev.value


def _fake_graph_decision() -> ReviewDecision:
    return ReviewDecision(
        decision=Decision.HUMAN_REVIEW,
        risk_level=RiskLevel.HIGH,
        risk_type=[RiskType.POTENTIAL_IP_RISK, RiskType.EVASION_PATTERN],
        decision_confidence=0.87,
        policy=["POLICY_3.2"],
        overrides=[],
    )


async def test_process_review_complex_goes_to_agent(monkeypatch):
    """verdict=COMPLEX → 调 run_and_persist(triage_result='COMPLEX'，携带 RULE_HIT evidence)。"""
    calls: dict = {}

    async def fake_run_and_persist(case, *, run_id=None, trigger_type="INITIAL",
                                   triage_result=None, extra_evidence=None):
        calls["agent"] = {"run_id": run_id, "trigger_type": trigger_type,
                          "triage_result": triage_result, "case_id": case.case_id,
                          "evidence": list(extra_evidence or [])}
        return {"case_id": case.case_id, "run_id": "RUN_CMPLX",
                "decision": _fake_graph_decision(), "counts": {"trace": 11, "evidence": 5}}

    async def fake_direct(case, triage_result, *, run_id=None):  # pragma: no cover
        raise AssertionError("COMPLEX 不应走 run_screening_direct")

    monkeypatch.setattr(ps, "run_and_persist", fake_run_and_persist)
    monkeypatch.setattr(ps, "run_screening_direct", fake_direct)

    out = await ps.process_review(_COMPLEX_HIGH_RISK, run_id="RUN_CMPLX")
    assert calls["agent"]["run_id"] == "RUN_CMPLX"
    assert calls["agent"]["trigger_type"] == "INITIAL"
    assert calls["agent"]["triage_result"] == "COMPLEX"
    assert calls["agent"]["case_id"] == "CASE_CMPLX_HIGHRISK"
    assert [e.extra["rule_id"] for e in calls["agent"]["evidence"]] == ["R-301"]
    assert out["verdict"] == "COMPLEX"
    assert out["run_id"] == "RUN_CMPLX"
    assert out["decision"].decision == Decision.HUMAN_REVIEW
    assert out["counts"] == {"trace": 11, "evidence": 5}


async def test_process_review_complex_forwards_rule_hit_evidence(monkeypatch):
    """COMPLEX 分支把触发进 Agent 的 RULE_HIT evidence（t.hits）传给 run_and_persist 落 review_evidence。"""
    seen: dict = {}

    async def fake_run_and_persist(case, *, run_id=None, trigger_type="INITIAL",
                                   triage_result=None, extra_evidence=None):
        seen["run_id"] = run_id
        seen["extra_evidence"] = list(extra_evidence or [])
        return {"case_id": case.case_id, "run_id": run_id,
                "decision": _fake_graph_decision(), "counts": {"trace": 11, "evidence": 5}}

    monkeypatch.setattr(ps, "run_and_persist", fake_run_and_persist)

    await ps.process_review(_COMPLEX_HIGH_RISK, run_id="RUN_CMPLX")
    assert seen["run_id"] == "RUN_CMPLX"
    evs = seen["extra_evidence"]
    assert len(evs) == 1
    ev = evs[0]
    assert ev.type == "RULE_HIT"
    assert ev.source == "ScreeningRuleEngine"
    assert ev.weight == RULE_EVIDENCE_WEIGHT
    assert ev.ref_id is None
    assert ev.extra == {"rule_id": "R-301"}
    assert ev.value.startswith("R-301")

    await ps.process_review(_COMPLEX_MULTI_HIT_NONRISK, run_id="RUN_CMPLX_MULTI")
    assert seen["run_id"] == "RUN_CMPLX_MULTI"
    evs2 = seen["extra_evidence"]
    assert [e.extra["rule_id"] for e in evs2] == ["R-301", "R-302"]
    assert all(e.type == "RULE_HIT" and e.source == "ScreeningRuleEngine"
               for e in evs2)

    await ps.process_review(_LV_REAL_BRAND_TERM, run_id="RUN_CMPLX_LV")
    assert seen["run_id"] == "RUN_CMPLX_LV"
    evs3 = seen["extra_evidence"]
    assert [e.extra["rule_id"] for e in evs3] == ["R-102"]
    assert evs3[0].value.startswith("R-102")


@pytest.mark.parametrize(
    ("case", "expect_verdict"),
    [(_CLEAN, "PASS"), (_REJECT_BLACKLIST, "REJECT")],
)
async def test_process_review_direct_branches(monkeypatch, case, expect_verdict):
    """verdict=PASS/REJECT → 调 run_screening_direct（直判）。"""
    calls: dict = {}

    async def fake_direct(c, triage_result, *, run_id=None):
        calls["direct"] = {"run_id": run_id, "verdict": triage_result.verdict}
        return {"case_id": c.case_id, "run_id": "RUN_DIRECT",
                "verdict": triage_result.verdict,
                "decision": ps._direct_decision(
                    triage_result.verdict,
                    [ps.rule_evidence(h) for h in triage_result.hits],
                ),
                "counts": {"evidence": len(triage_result.hits)}}

    async def fake_agent(case, **kwargs):  # pragma: no cover
        raise AssertionError("直判不应走 run_and_persist")

    monkeypatch.setattr(ps, "run_and_persist", fake_agent)
    monkeypatch.setattr(ps, "run_screening_direct", fake_direct)

    out = await ps.process_review(case, run_id="RUN_DIRECT")
    assert calls["direct"] == {"run_id": "RUN_DIRECT", "verdict": expect_verdict}
    assert out["verdict"] == expect_verdict
    assert out["run_id"] == "RUN_DIRECT"
    assert isinstance(out["decision"], ReviewDecision)
    assert out["decision"].decision_confidence == 1.0
    assert out["decision"].decision == (Decision.PASS if expect_verdict == "PASS"
                                        else Decision.REJECT)
    assert out["decision"].risk_level == (RiskLevel.NONE if expect_verdict == "PASS"
                                          else RiskLevel.HIGH)


async def test_process_review_direct_decision_evidence(monkeypatch):
    """REJECT 直判的返回 decision.evidence 携带 RULE_HIT（R-101），PASS 无命中为空。"""
    async def fake_direct(c, triage_result, *, run_id=None):
        return {"case_id": c.case_id, "run_id": "RUN_DIRECT",
                "verdict": triage_result.verdict,
                "decision": ps._direct_decision(
                    triage_result.verdict,
                    [ps.rule_evidence(h) for h in triage_result.hits],
                ),
                "counts": {"evidence": len(triage_result.hits)}}

    monkeypatch.setattr(ps, "run_screening_direct", fake_direct)

    out = await ps.process_review(_REJECT_BLACKLIST)
    rules_in_evidence = {ev.extra.get("rule_id") for ev in out["decision"].evidence}
    assert rules_in_evidence == {"R-101"}

    out_pass = await ps.process_review(_CLEAN)
    assert out_pass["decision"].evidence == []
