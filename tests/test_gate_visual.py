"""R3_VISUAL_CLAIM_UNSUPPORTED 单测：外观/视觉声称但无视觉证据时 Gate 判弃权。

命中条件：SUPPORTED 且高优先的外观假设；证据里没有任何视觉证据。
判定 = REJECT 提案被 abstention 拦成 HUMAN_REVIEW。

存在性判定（与命中相反）：
- ``IMAGE_SIMILARITY`` weight ≥ 0.70 即算有视觉证据（gate 只读 weight，不依赖 ``extra.strong``：
  weight=0.65 即使 extra 声称 strong 也不放行）；边界 0.70 存在 / 0.69 不存在；
- 任一 ``IMAGE_LOGO`` 即满足存在性，与置信无关；
- ``OCR_TEXT`` / ``CASE_PRECEDENT`` / ``POLICY_REF`` / ``PRODUCT_FACT`` / ``MERCHANT_HISTORY``
  一律不算视觉证据；
- 非外观维度假设、非 SUPPORTED 或低 prior、空态 → 不命中（安全 False）。

其他纪律：谓词只读，执行前后 hypotheses/evidence 零变化；PASS 侧不受影响；命中时与
R3_POLICY_UNCERTAIN / R3_HYPOTHESES_INDISTINGUISHABLE 多码全量收集、顺序固定；
关键词表与 llm_prompts 例句同源；负向句 SUPPORTED 照样命中（误判代价在 REJECT 侧）。
全部确定性、无网络、无 API key。
"""

from __future__ import annotations

import pytest
from helpers import ev, hp

from pra.agent.guardrails import gate
from pra.agent.guardrails.gate import (
    R3_HYPOTHESES_INDISTINGUISHABLE,
    R3_POLICY_UNCERTAIN,
    R3_VISUAL_CLAIM_UNSUPPORTED,
    run_decision_overlay,
    visual_claim_unsupported,
    visual_evidence_present,
)
from pra.agent.guardrails.schemas import DecisionProposal
from pra.domain.models import Budget, Decision, Evidence, HypothesisStatus

# EC_0007 形态假设句与对照句
_VISUAL = "外观与经典小白鞋高度相似"
_NEG_VISUAL = "外观与品牌款明显不同"  # 负向句异常态（命中属安全侧）
_BEHAVIOR = "商家系统性类似上架行为"
_BRAND_EVASION = "刻意规避品牌识别（品牌字段空缺）"
_TEXT_CLAIM = "商品标题含某品牌字样（OCR 检出）"  # 文本声称不属本拦截面


def _proposal(decision: str = "REJECT", risk_level: str = "HIGH") -> DecisionProposal:
    return DecisionProposal(
        decision=decision,  # type: ignore[arg-type]
        risk_level=risk_level,  # type: ignore[arg-type]
        risk_type=[],
        confidence=0.9,
        evidence_ids=[],
        policy=[],
        rationale="test",
    )


def _policy_ref() -> Evidence:
    return ev("POLICY_REF", value="POLICY_3.2 v2 条款：外观高度模仿知名品牌设计", weight=0.9,
              ref_id="POLICY_3.2_v2_c1", extra={"policy_id": "POLICY_3.2", "policy_version": 2})


def _precedent() -> Evidence:
    return ev("CASE_PRECEDENT", value="case_1001 无品牌标识+外观高度模仿", weight=0.8,
              ref_id="case_1001")


def _ocr() -> Evidence:
    return ev("OCR_TEXT", value="图内含复刻宣传字样", weight=0.7, ref_id="img1")


def _supported_state(
    *,
    statement: str = _VISUAL,
    prior: float = 0.5,
    status: HypothesisStatus = HypothesisStatus.SUPPORTED,
    evidence: list[Evidence] | None = None,
    posterior: float = 0.9,
) -> dict:
    # 高优先（默认）SUPPORTED 视觉假设 + 证据的完整 state 骨架（含 overlay 必需键）
    return {
        "hypotheses": [
            hp("H1", statement=statement, prior=prior, posterior=posterior, status=status,
               evidence_for=["POLICY_REF POLICY_3.2 v2 条款：外观高度模仿知名品牌设计"]),
        ],
        "evidence": list(evidence or []),
        "budget": Budget(),
        "failures": [],
        "tool_call_history": [],
        "degraded": False,
    }


# 谓词层：命中/不命中 + 边界


def test_a_visual_supported_without_visual_evidence_hits():
    st = _supported_state(evidence=[_policy_ref(), _precedent()])
    assert visual_claim_unsupported(st) is True
    assert R3_VISUAL_CLAIM_UNSUPPORTED in gate._abstention_codes(st)


def test_a_overlay_reject_intercepted_to_human():
    st = _supported_state(evidence=[_policy_ref(), _precedent()])
    dc = gate.finalize_decision_confidence(st)
    # 该形态改判前能过 REJECT Gate（dc=0.77），现被 abstention 拦成 HUMAN
    assert gate.reject_gate(st, dc) is True  # 证明拦截点确在 abstention（改动前会自动 REJECT）
    final = run_decision_overlay(st, _proposal("REJECT", "HIGH"))
    assert final.decision == Decision.HUMAN_REVIEW
    assert final.overrides == [R3_VISUAL_CLAIM_UNSUPPORTED]


def test_b_similarity_072_present_not_hit():
    sim = ev("IMAGE_SIMILARITY", value="similarity=0.72, match=某品牌经典鞋款", weight=0.72,
             ref_id="img1", extra={"similarity": 0.72, "strong": False})
    st = _supported_state(evidence=[_policy_ref(), _precedent(), sim])
    assert visual_claim_unsupported(st) is False
    final = run_decision_overlay(st, _proposal("REJECT", "HIGH"))
    assert final.decision == Decision.REJECT
    assert final.overrides == []


def test_c_similarity_065_below_threshold_hits():
    weak = ev("IMAGE_SIMILARITY", value="similarity=0.65, match=疑似", weight=0.65,
              ref_id="img1", extra={"similarity": 0.65, "strong": False})
    st = _supported_state(evidence=[_policy_ref(), _precedent(), weak])
    assert visual_claim_unsupported(st) is True
    assert R3_VISUAL_CLAIM_UNSUPPORTED in gate._abstention_codes(st)


def test_d_image_logo_satisfies_presence():
    logo = ev("IMAGE_LOGO", value="logo=某品牌, conf=0.60", weight=0.6, ref_id="img2")
    st = _supported_state(evidence=[_policy_ref(), _precedent(), logo])
    assert visual_evidence_present(st["evidence"]) is True
    assert visual_claim_unsupported(st) is False


def test_e_ocr_precedent_policy_are_not_visual():
    base = [_policy_ref(), _precedent(), _ocr()]
    assert visual_claim_unsupported(_supported_state(evidence=base)) is True
    richer = base + [
        ev("PRODUCT_FACT", value="brand=null, version=3（库中最新）", weight=0.6, ref_id="P_1"),
        ev("MERCHANT_HISTORY", value="0 similar / 0 removals / 0 title-relisting, credit=90",
           weight=0.85, ref_id="M_1", extra={"removals": 0, "title": 0}),
    ]
    assert visual_claim_unsupported(_supported_state(evidence=richer)) is True


def test_f_non_visual_dimensions_untouched():
    for statement in (_BEHAVIOR, _BRAND_EVASION, _TEXT_CLAIM):
        st = _supported_state(statement=statement, evidence=[_policy_ref(), _precedent(), _ocr()])
        assert gate._looks_visual_claim(statement) is False
        assert visual_claim_unsupported(st) is False


def test_g_non_supported_or_low_prior_not_hit():
    for status in (HypothesisStatus.REFUTED, HypothesisStatus.UNRESOLVED,
                   HypothesisStatus.PENDING):
        st = _supported_state(status=status, evidence=[_policy_ref(), _ocr()])
        assert visual_claim_unsupported(st) is False
    low = _supported_state(prior=0.2, evidence=[_policy_ref(), _ocr()])
    assert visual_claim_unsupported(low) is False


# 只读纪律 / PASS 侧 / 共存 / 边界


def test_h_predicate_read_only():
    st = _supported_state(evidence=[_policy_ref(), _precedent(), _ocr()])
    before = ([h.model_dump() for h in st["hypotheses"]],
              [e.model_dump() for e in st["evidence"]])
    assert visual_claim_unsupported(st) is True
    gate._abstention_codes(st)
    after = ([h.model_dump() for h in st["hypotheses"]],
             [e.model_dump() for e in st["evidence"]])
    assert after == before


def test_i_pass_side_unaffected():
    st = {
        "hypotheses": [
            hp("H1", statement=_VISUAL, prior=0.5, status=HypothesisStatus.REFUTED,
               evidence_against=["OCR_TEXT 图内无品牌字样"]),
            hp("H2", statement="普通设计非品牌款", prior=0.4, status=HypothesisStatus.REFUTED,
               evidence_against=["IMAGE_SIMILARITY similarity=0.42, match=无"]),
        ],
        "evidence": [_ocr()],
        "budget": Budget(),
        "failures": [],
        "tool_call_history": [],
        "degraded": False,
    }
    assert visual_claim_unsupported(st) is False
    assert gate.pass_gate(st) is True
    final = run_decision_overlay(st, _proposal("PASS", "NONE"))
    assert final.decision == Decision.PASS
    assert final.overrides == []


def test_j_coexists_with_policy_uncertain():
    st = _supported_state(evidence=[_ocr()])  # 无 POLICY_REF → policy_indeterminate 亦命中
    assert gate.policy_indeterminate(st) is True
    assert gate._abstention_codes(st) == [R3_POLICY_UNCERTAIN, R3_VISUAL_CLAIM_UNSUPPORTED]
    final = run_decision_overlay(st, _proposal("REJECT", "HIGH"))
    assert final.decision == Decision.HUMAN_REVIEW
    assert final.overrides == [R3_POLICY_UNCERTAIN, R3_VISUAL_CLAIM_UNSUPPORTED]


def test_j_coexists_with_indistinguishable():
    st = {
        "hypotheses": [
            hp("H1", statement=_VISUAL, prior=0.5, status=HypothesisStatus.SUPPORTED,
               posterior=0.9, evidence_for=["OCR_TEXT 图内有字样"]),
            hp("H2", statement=_VISUAL, prior=0.4, status=HypothesisStatus.SUPPORTED,
               posterior=0.85, evidence_for=["OCR_TEXT 图内有字样"]),  # 集合相等
        ],
        "evidence": [_policy_ref(), _ocr()],  # 有政策 → policy_indeterminate 不参与
        "budget": Budget(),
        "failures": [],
        "tool_call_history": [],
        "degraded": False,
    }
    assert gate.indistinguishable_hypotheses(st) is True
    assert visual_claim_unsupported(st) is True
    assert gate._abstention_codes(st) == [
        R3_HYPOTHESES_INDISTINGUISHABLE, R3_VISUAL_CLAIM_UNSUPPORTED,
    ]


def test_evidence_presence_boundaries_and_empty():
    assert visual_evidence_present(
        [ev("IMAGE_SIMILARITY", value="s", weight=0.70, ref_id="i")]) is True
    assert visual_evidence_present(
        [ev("IMAGE_SIMILARITY", value="s", weight=0.69, ref_id="i")]) is False
    assert visual_evidence_present(
        [ev("IMAGE_LOGO", value="logo=x, conf=0.10", weight=0.1, ref_id="i")]) is True
    assert visual_evidence_present([]) is False
    assert visual_evidence_present(None) is False


def test_empty_state_safe():
    assert visual_claim_unsupported({}) is False
    assert visual_claim_unsupported({"hypotheses": [], "evidence": []}) is False
    st = {"budget": Budget(), "hypotheses": None, "evidence": None,
          "failures": None, "degraded": False}
    assert visual_claim_unsupported(st) is False
    assert R3_VISUAL_CLAIM_UNSUPPORTED not in gate._abstention_codes(st)


@pytest.mark.parametrize(
    ("statement", "expected"),
    [
        # llm_prompts reevaluate 第 8 条例句逐字镜像（同步契约）
        ("外观与某品牌款外观相似", True),
        ("外观与经典小白鞋高度相似", True),
        ("与品牌款同款外观", True),
        ("属于视觉仿冒知名品牌设计", True),
        ("复刻外观设计", True),
        ("与经典款版型一致", True),
        ("长得像某品牌经典款", True),
        # 维度词
        ("配色与图案照搬某品牌", True),
        ("刻意规避品牌识别", False),
        ("商家系统性类似上架行为", False),
        ("商品标题含某品牌字样（OCR 检出）", False),  # 文本声称不拦
    ],
)
def test_keyword_dimension_classification(statement: str, expected: bool):
    assert gate._looks_visual_claim(statement) is expected


def test_negative_sentence_supported_hits_by_design():
    st = _supported_state(statement=_NEG_VISUAL, evidence=[_policy_ref(), _precedent()])
    assert visual_claim_unsupported(st) is True
    final = run_decision_overlay(st, _proposal("REJECT", "HIGH"))
    assert final.decision == Decision.HUMAN_REVIEW
    assert R3_VISUAL_CLAIM_UNSUPPORTED in final.overrides
