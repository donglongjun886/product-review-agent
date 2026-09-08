"""Screening 三分流（pra/screening/** + persist_service.process_review）单测 —— 不连真库。

覆盖（任务书新增）：
- ``triage`` 纯函数：PASS（brand 明确 + 干净）/ COMPLEX（brand=None+高危类目；
  EVASION_TERMS 命中）/ REJECT（注入黑名单 / BRAND_TERMS 命中，含大小写不敏感）/
  REJECT 优先于 COMPLEX / hits 收集顺序 / 空 rules / 注入 rules；
- ``rule_evidence`` 证据形状（RULE_HIT / ScreeningRuleEngine / weight=1.0 / extra）；
- ``process_review`` 分支（COMPLEX → run_and_persist；PASS/REJECT →
  run_screening_direct）—— monkeypatch persist 层假函数，不触真库。
"""

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
from pra.screening.engine import RuleHit, rule_evidence, triage
from pra.screening.rule_engine import terms
from pra.screening.rule_engine.rules import DEFAULT_RULES, Rule

# ---------------------------------------------------------------------------
# case 工厂
# ---------------------------------------------------------------------------


def _case(
    *,
    case_id: str,
    brand: str | None,
    title: str,
    description: str,
    category: str,
) -> ProductReviewCase:
    """构造最小合法 case（可定制 brand/title/description/category）。"""
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
        screening_signals=[],
    )


_CLEAN = _case(
    case_id="CASE_PASS",
    brand="自营品牌A",  # brand 明确（不在黑名单）
    title="北欧风陶瓷花瓶 简约桌面装饰",
    description="釉面细腻，清新雅致。",
    category="家居日用",  # 非高危前缀类目
)
_COMPLEX_HIGH_RISK = _case(  # brand=None + 高危类目女鞋/运动鞋
    case_id="CASE_CMPLX_HIGHRISK",
    brand=None,
    title="新款厚底复古跑鞋",
    description="复古厚底设计，舒适百搭。",
    category="女鞋/运动鞋",
)
_COMPLEX_EVASION = _case(  # brand 明确但描述含规避词
    case_id="CASE_CMPLX_EVASION",
    brand="自营品牌A",
    title="设计师简约布艺沙发",
    description="同款复刻工艺，1:1 细节还原。",
    category="家居日用",
)
_REJECT_BRANDTERM = _case(  # brand=None + 标题 NIKE（R-102）+ 高危类目（R-301）
    case_id="CASE_REJECT_BRANDTERM",
    brand=None,
    title="NIKE 新款复古跑鞋 轻便透气",
    description="舒适运动。",
    category="女鞋/运动鞋",
)
_LOWERCASE_BRANDTERM = _case(  # 大小写不敏感：小写 nike 也命中 R-102
    case_id="CASE_REJECT_LOWER",
    brand=None,
    title="nike air 复古跑鞋",
    description="舒适运动。",
    category="女鞋/运动鞋",
)


# ---------------------------------------------------------------------------
# triage：PASS / COMPLEX / REJECT
# ---------------------------------------------------------------------------


def test_triage_pass_clean_case():
    """brand 明确 + 标题/描述干净 + 类目非高危 → PASS（确定性放行，零命中）。"""
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
    """描述含规避词（同款/复刻/1:1）→ COMPLEX，命中 R-302（不直判）。"""
    t = triage(_COMPLEX_EVASION)
    assert t.verdict == "COMPLEX"
    assert [h.rule_id for h in t.hits] == ["R-302"]
    assert "同款" in t.hits[0].detail and "复刻" in t.hits[0].detail


def test_triage_reject_blacklisted_brand(monkeypatch):
    """注入黑名单：brand ∈ BLACKLISTED_BRANDS → REJECT（R-101）。"""
    monkeypatch.setattr(terms, "BLACKLISTED_BRANDS", frozenset({"NIKE"}))
    case = _case(
        case_id="CASE_REJECT_BLACKLIST",
        brand="NIKE",  # 标题无品牌词，只命中 R-101
        title="复古运动鞋 轻便透气",
        description="舒适运动。",
        category="女鞋/运动鞋",
    )
    t = triage(case)
    assert t.verdict == "REJECT"
    assert [h.rule_id for h in t.hits] == ["R-101"]
    assert "黑名单" in t.hits[0].detail


def test_triage_reject_brand_term_in_title():
    """标题含 BRAND_TERMS（NIKE）→ REJECT（R-102）。"""
    t = triage(_REJECT_BRANDTERM)
    assert t.verdict == "REJECT"
    assert "R-102" in [h.rule_id for h in t.hits]


def test_triage_reject_brand_term_case_insensitive():
    """BRAND_TERMS 大小写不敏感：小写 nike 同样命中 R-102。"""
    t = triage(_LOWERCASE_BRANDTERM)
    assert t.verdict == "REJECT"
    assert [h.rule_id for h in t.hits] == ["R-102", "R-301"]  # R-102 REJECT + R-301 COMPLEX


def test_triage_reject_priority_over_complex():
    """REJECT 优先于 COMPLEX：同轮 R-102(REJECT)+R-301(COMPLEX) → 终裁 REJECT。"""
    t = triage(_REJECT_BRANDTERM)
    assert t.verdict == "REJECT"  # 而非 COMPLEX
    reject_hits = [h for h in t.hits if h.rule_id.startswith("R-1")]
    assert reject_hits and reject_hits[0].rule_id == "R-102"


def test_triage_hits_collected_in_rule_order():
    """同轮多命中全收集，hits 按规则声明序（R-101→R-102→R-301→R-302）。"""
    case = _case(
        case_id="CASE_MULTI_HIT",
        brand=None,
        title="NIKE 新款跑鞋",  # R-102
        description="同款复刻工艺 1:1 细节。",  # R-302
        category="女鞋/运动鞋",  # R-301
    )
    t = triage(case)
    assert t.verdict == "REJECT"  # R-102 命中 → REJECT 优先
    assert [h.rule_id for h in t.hits] == ["R-102", "R-301", "R-302"]
    assert "同款" in t.hits[2].detail and "复刻" in t.hits[2].detail


# ---------------------------------------------------------------------------
# triage：空 / 注入 rules
# ---------------------------------------------------------------------------


def test_triage_empty_rules_pass():
    """rules=[] → 零规则零命中 → PASS。"""
    t = triage(_REJECT_BRANDTERM, rules=[])
    assert t.verdict == "PASS"
    assert t.hits == []


def test_triage_injected_rules_custom():
    """注入自定义规则集：COMPLEX 规则命中 → COMPLEX；再注入 REJECT 规则 → REJECT 优先。"""
    complex_rule = Rule("R-900", "强制COMPLEX", "COMPLEX", lambda case: "always complex")
    t = triage(_CLEAN, rules=[complex_rule])
    assert t.verdict == "COMPLEX"
    assert [h.rule_id for h in t.hits] == ["R-900"]

    reject_rule = Rule("R-901", "强制REJECT", "REJECT", lambda case: "always reject")
    t2 = triage(_CLEAN, rules=[complex_rule, reject_rule])
    assert t2.verdict == "REJECT"  # REJECT 规则优先
    assert [h.rule_id for h in t2.hits] == ["R-900", "R-901"]  # 按规则序全收集


def test_default_rules_declaration_order():
    """DEFAULT_RULES 声明序与 rule_id 前缀规范（R-1xx=REJECT / R-3xx=COMPLEX）。"""
    assert [r.rule_id for r in DEFAULT_RULES] == ["R-101", "R-102", "R-301", "R-302"]
    assert [r.kind for r in DEFAULT_RULES] == ["REJECT", "REJECT", "COMPLEX", "COMPLEX"]


# ---------------------------------------------------------------------------
# rule_evidence
# ---------------------------------------------------------------------------


def test_rule_evidence_shape():
    """rule_evidence 构造确定性直判证据（只构造不落库）。"""
    hit = RuleHit(rule_id="R-102", name="品牌词命中", detail="标题/描述命中品牌词: NIKE")
    ev = rule_evidence(_REJECT_BRANDTERM, hit)
    assert ev.type == "RULE_HIT"
    assert ev.source == "ScreeningRuleEngine"
    assert ev.value == "R-102 品牌词命中: 标题/描述命中品牌词: NIKE"
    assert ev.weight == 1.0
    assert ev.ref_id is None
    assert ev.extra == {"rule_id": "R-102"}


# ---------------------------------------------------------------------------
# process_review：COMPLEX → run_and_persist；PASS/REJECT → run_screening_direct
# ---------------------------------------------------------------------------


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
    """verdict=COMPLEX → 调 run_and_persist(triage_result='COMPLEX')，不走直判。"""
    calls: dict = {}

    async def fake_run_and_persist(case, *, run_id=None, trigger_type="INITIAL",
                                   triage_result=None):
        calls["agent"] = {"run_id": run_id, "trigger_type": trigger_type,
                          "triage_result": triage_result, "case_id": case.case_id}
        return {"case_id": case.case_id, "run_id": "RUN_CMPLX",
                "decision": _fake_graph_decision(), "counts": {"trace": 11, "evidence": 5}}

    async def fake_direct(case, triage_result, *, run_id=None):  # pragma: no cover
        raise AssertionError("COMPLEX 不应走 run_screening_direct")

    monkeypatch.setattr(ps, "run_and_persist", fake_run_and_persist)
    monkeypatch.setattr(ps, "run_screening_direct", fake_direct)

    out = await ps.process_review(_COMPLEX_HIGH_RISK, run_id="RUN_CMPLX")
    assert calls["agent"] == {"run_id": "RUN_CMPLX", "trigger_type": "INITIAL",
                              "triage_result": "COMPLEX", "case_id": "CASE_CMPLX_HIGHRISK"}
    assert out["verdict"] == "COMPLEX"
    assert out["run_id"] == "RUN_CMPLX"
    assert out["decision"].decision == Decision.HUMAN_REVIEW
    assert out["counts"] == {"trace": 11, "evidence": 5}


@pytest.mark.parametrize(
    ("case", "expect_verdict"),
    [(_CLEAN, "PASS"), (_REJECT_BRANDTERM, "REJECT")],
)
async def test_process_review_direct_branches(monkeypatch, case, expect_verdict):
    """verdict=PASS/REJECT → 调 run_screening_direct（直判），不走 Agent。"""
    calls: dict = {}

    async def fake_direct(c, triage_result, *, run_id=None):
        calls["direct"] = {"run_id": run_id, "verdict": triage_result.verdict}
        return {"case_id": c.case_id, "run_id": "RUN_DIRECT",
                "verdict": triage_result.verdict,
                "counts": {"evidence": len(triage_result.hits)}}

    async def fake_agent(case, **kwargs):  # pragma: no cover
        raise AssertionError("直判不应走 run_and_persist")

    monkeypatch.setattr(ps, "run_and_persist", fake_agent)
    monkeypatch.setattr(ps, "run_screening_direct", fake_direct)

    out = await ps.process_review(case, run_id="RUN_DIRECT")
    assert calls["direct"] == {"run_id": "RUN_DIRECT", "verdict": expect_verdict}
    assert out["verdict"] == expect_verdict
    assert out["run_id"] == "RUN_DIRECT"
    # 直判 decision 与落库 decision_json 同构：确定性 confidence=1.0
    assert isinstance(out["decision"], ReviewDecision)
    assert out["decision"].decision_confidence == 1.0
    assert out["decision"].decision == (Decision.PASS if expect_verdict == "PASS"
                                        else Decision.REJECT)
    assert out["decision"].risk_level == (RiskLevel.NONE if expect_verdict == "PASS"
                                          else RiskLevel.HIGH)


async def test_process_review_direct_decision_evidence(monkeypatch):
    """REJECT 直判的返回 decision.evidence 携带 RULE_HIT（R-102），PASS 无命中为空。"""
    async def fake_direct(c, triage_result, *, run_id=None):
        return {"case_id": c.case_id, "run_id": "RUN_DIRECT",
                "verdict": triage_result.verdict,
                "counts": {"evidence": len(triage_result.hits)}}

    monkeypatch.setattr(ps, "run_screening_direct", fake_direct)

    out = await ps.process_review(_REJECT_BRANDTERM)
    rules_in_evidence = {ev.extra.get("rule_id") for ev in out["decision"].evidence}
    assert rules_in_evidence == {"R-102", "R-301"}  # 同轮全命中都进证据链

    out_pass = await ps.process_review(_CLEAN)
    assert out_pass["decision"].evidence == []  # PASS 零命中 → 空证据链
