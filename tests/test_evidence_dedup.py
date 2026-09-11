"""证据质量过滤 + extra 回填（guardrails/evidence.py）与 plan 去重（guardrails/dedup.py）。

``quality_filter``：只按 weight 丢弱 ``IMAGE_SIMILARITY``（0.70 保留 / 0.699 丢弃），其它类型全保留。
``backfill_extra``：按类型从 value 解析派生键（IMAGE_SIMILARITY / MERCHANT_HISTORY / POLICY_REF /
PRODUCT_FACT / IMAGE_LOGO）；解析失败保留原 extra；不改入参。``dedup_pending``：ok 命中 →
skipped、error 允许重试、同轮自去重、canonical 键序无关。
"""

from __future__ import annotations

from pra.agent.guardrails.dedup import canonical_args, dedup_pending
from pra.agent.guardrails.evidence import backfill_extra, quality_filter
from helpers import ev, make_case


# quality_filter（EVIDENCE_MIN_SIM=0.70）


def test_quality_filter_boundary_keeps_at_threshold():
    e = ev("IMAGE_SIMILARITY", value="similarity=0.70", weight=0.70, ref_id="img1")
    out = quality_filter([e])
    assert out == [e]


def test_quality_filter_drops_below_threshold():
    e = ev("IMAGE_SIMILARITY", value="similarity=0.699", weight=0.699, ref_id="img1")
    assert quality_filter([e]) == []


def test_quality_filter_keeps_strong_and_drops_weak_in_mix():
    """混合输入：弱相似丢弃、强相似保留、顺序不变。"""

    weak = ev("IMAGE_SIMILARITY", value="similarity=0.42", weight=0.42, ref_id="img2")
    strong = ev("IMAGE_SIMILARITY", value="similarity=0.91", weight=0.91, ref_id="img1")
    out = quality_filter([weak, strong])
    assert out == [strong]


def test_quality_filter_keeps_non_similarity_types():
    logo = ev("IMAGE_LOGO", value="logo=某品牌, conf=0.93", weight=0.93, ref_id="img2")
    low_logo = ev("IMAGE_LOGO", value="logo=某品牌, conf=0.30", weight=0.30, ref_id="img3")
    merch = ev("MERCHANT_HISTORY", value="23 similar / 5 removals", weight=0.85, ref_id="M1")
    out = quality_filter([logo, low_logo, merch])
    assert out == [logo, low_logo, merch]


def test_quality_filter_none_and_empty_safe_and_pure():
    assert quality_filter(None) == []
    assert quality_filter([]) == []
    e = ev("IMAGE_SIMILARITY", value="similarity=0.99", weight=0.99, ref_id="img1")
    out = quality_filter([e])
    assert out[0] is e  # 保留元素复用原引用，不复制


# backfill_extra：各类型回填


def test_backfill_image_similarity():
    """IMAGE_SIMILARITY → similarity=round(weight,3)、strong=weight>=0.85（确定性派生）。"""
    strong = ev("IMAGE_SIMILARITY", value="similarity=0.91, match=x", weight=0.91, ref_id="i1")
    weak = ev("IMAGE_SIMILARITY", value="similarity=0.70, match=y", weight=0.70, ref_id="i2")
    s, w = backfill_extra([strong, weak])
    assert s.extra == {"similarity": 0.91, "strong": True}
    assert w.extra == {"similarity": 0.70, "strong": False}


def test_backfill_merchant_history_parses_value():
    e = ev("MERCHANT_HISTORY", value="23 similar / 5 removals / 3 title-relisting, credit=62",
           weight=0.85, ref_id="M_5512")
    out = backfill_extra([e])[0]
    assert out.extra == {"similar": 23, "removals": 5, "title": 3, "credit": 62}


def test_backfill_policy_ref():
    e = ev("POLICY_REF", value="POLICY_3.2 v2 条款：外观高度模仿知名品牌设计",
           weight=0.9, ref_id="POLICY_3.2_v2_c1")
    out = backfill_extra([e])[0]
    assert out.extra == {"policy_id": "POLICY_3.2", "policy_version": 2}


def test_backfill_product_fact_no_drift_when_versions_equal():
    """PRODUCT_FACT：库中 version == case.product.version → 不写 version_drift 键。"""
    e = ev("PRODUCT_FACT", value="brand=null, version=3（库中最新）, status=ON_SALE",
           weight=0.6, ref_id="P_88231")
    out = backfill_extra([e], case=make_case(version=3))[0]
    assert "version_drift" not in out.extra
    assert out.extra == {}


def test_backfill_product_fact_drift_when_versions_differ():
    e = ev("PRODUCT_FACT", value="brand=null, version=3（库中最新）, status=ON_SALE",
           weight=0.6, ref_id="P_88231")
    out = backfill_extra([e], case=make_case(version=2))[0]
    assert out.extra == {"version_drift": True}


def test_backfill_product_fact_without_case_no_key():
    """无 case / 无法解析版本 → 不写键（缺失 == 无漂移）。"""
    e = ev("PRODUCT_FACT", value="brand=null, version=3（库中最新）, status=ON_SALE",
           weight=0.6, ref_id="P_88231")
    assert backfill_extra([e])[0].extra == {}
    bad = ev("PRODUCT_FACT", value="brand=null（无版本标注）", weight=0.6, ref_id="P_88231")
    assert backfill_extra([bad], case=make_case(version=2))[0].extra == {}


def test_backfill_image_logo():
    e = ev("IMAGE_LOGO", value="logo=某品牌, conf=0.93", weight=0.93, ref_id="img2")
    out = backfill_extra([e])[0]
    assert out.extra == {"logo_brand": "某品牌", "confidence": 0.93}


def test_backfill_parse_failure_keeps_original_extra():
    """解析失败（尽力而为）→ 保留原 extra、不报错。"""

    e = ev("POLICY_REF", value="这段文本不符合 POLICY_x.y vN 前缀", weight=0.9,
           ref_id="c1", extra={"custom": 1})
    out = backfill_extra([e])[0]
    assert out.extra == {"custom": 1}


def test_backfill_merges_with_existing_extra():
    e = ev("MERCHANT_HISTORY", value="1 similar / 2 removals / 0 title-relisting, credit=80",
           weight=0.85, ref_id="M1", extra={"source_note": "seed"})
    out = backfill_extra([e])[0]
    assert out.extra == {"source_note": "seed", "similar": 1, "removals": 2, "title": 0,
                         "credit": 80}


def test_backfill_does_not_mutate_inputs():
    """不改入参：输入列表与元素（extra 内容/引用）均不变；输出为 model_copy。"""
    e = ev("IMAGE_SIMILARITY", value="similarity=0.91", weight=0.91, ref_id="i1")
    original = list([e])
    snapshot_extra = dict(e.extra)
    out = backfill_extra([e])[0]
    assert out is not e
    assert out.extra is not e.extra  # merged 恒为新 dict，不与原对象共享引用
    assert e.extra == snapshot_extra
    assert e.weight == 0.91 and e.value == "similarity=0.91"
    # 未改动原列表（pydantic 模型不可变语义下仍返回新列表）
    assert original[0] is e
    # backfill 对无回填类型也返回 model_copy（无别名）
    plain = ev("CASE_PRECEDENT", value="case_1001", weight=0.8, ref_id="c1")
    p_out = backfill_extra([plain])[0]
    assert p_out is not plain and p_out.extra == {}


def test_backfill_none_empty_safe():
    assert backfill_extra(None) == []
    assert backfill_extra([]) == []


# dedup_pending：plan 输出确定性去重


def _ok_record(tool: str, args: dict, seq: int) -> dict:
    return {"tool": tool, "args": args, "status": "ok", "seq": seq, "tokens": 0}


def test_canonical_args_key_order_independent():
    """canonical 序列化：key 排序 + ``default=str``。"""
    assert canonical_args({"a": 1, "b": 2}) == canonical_args({"b": 2, "a": 1})
    assert canonical_args({"when": "x"}) != canonical_args({"when": "x", "extra": "y"})


def test_dedup_ok_hit_goes_to_skipped_with_seq_continuing():
    """命中已执行成功 → 进 skipped（seq 从 ``len(history)+1`` 顺延），cleaned 为空。"""
    state = {"tool_call_history": [_ok_record("ImageAnalysisTool", {"u": "u1"}, 1)]}
    planned = [
        {"tool": "ImageAnalysisTool", "args": {"u": "u1"}, "reason": "r", "priority": 1},
    ]
    cleaned, skipped = dedup_pending(state, planned)
    assert cleaned == []
    assert skipped == [
        {"seq": 2, "tool": "ImageAnalysisTool", "args": {"u": "u1"}, "status": "skipped",
         "reason": "duplicate", "latency_ms": 0, "tokens": 0}
    ]


def test_dedup_error_record_allows_retry():
    """曾 ``status=error`` 的 (tool, args) 不进已执行集合 → 允许重试。"""
    state = {
        "tool_call_history": [
            {"tool": "MerchantTool", "args": {"merchant_id": "M1"}, "status": "error",
             "seq": 1, "error": "boom"},
        ]
    }
    planned = [
        {"tool": "MerchantTool", "args": {"merchant_id": "M1"}, "reason": "retry", "priority": 1},
    ]
    cleaned, skipped = dedup_pending(state, planned)
    assert skipped == []
    assert cleaned == planned  # 保留（浅拷贝、内容一致）


def test_dedup_same_round_self_dedup_silent():
    state = {"tool_call_history": []}
    planned = [
        {"tool": "ProductTool", "args": {"product_id": "P1"}, "priority": 1},
        {"tool": "ProductTool", "args": {"product_id": "P1"}, "priority": 1},  # 重复
        {"tool": "ProductTool", "args": {"product_id": "P2"}, "priority": 2},
    ]
    cleaned, skipped = dedup_pending(state, planned)
    assert [c["args"] for c in cleaned] == [{"product_id": "P1"}, {"product_id": "P2"}]
    assert skipped == []


def test_dedup_same_round_duplicate_of_executed_only_one_skipped():
    state = {"tool_call_history": [_ok_record("ProductTool", {"product_id": "P1"}, 1)]}
    planned = [
        {"tool": "ProductTool", "args": {"product_id": "P1"}, "priority": 1},
        {"tool": "ProductTool", "args": {"product_id": "P1"}, "priority": 1},
    ]
    cleaned, skipped = dedup_pending(state, planned)
    assert cleaned == []
    assert len(skipped) == 1 and skipped[0]["seq"] == 2


def test_dedup_cleaned_are_shallow_copies():
    """cleaned 保留原 dict 内容但为浅拷贝（防别名污染后续执行）。"""

    planned = [{"tool": "ProductTool", "args": {"product_id": "P1"}, "reason": "r", "priority": 1}]
    cleaned, _ = dedup_pending({"tool_call_history": []}, planned)
    assert cleaned[0] == planned[0]
    assert cleaned[0] is not planned[0]


def test_dedup_skipped_seq_monotonic_after_history():
    state = {"tool_call_history": [_ok_record("ProductTool", {"product_id": "P1"}, 1)]}
    planned = [
        {"tool": "ProductTool", "args": {"product_id": "P1"}, "priority": 1},
        {"tool": "MerchantTool", "args": {"merchant_id": "M1"}, "priority": 2},
    ]
    cleaned, skipped = dedup_pending(state, planned)
    assert skipped[0]["seq"] == 2
    # MerchantTool 未执行过 → 进 cleaned（不是 skipped）
    assert [c["tool"] for c in cleaned] == ["MerchantTool"]


def test_dedup_none_and_defensive_inputs():
    assert dedup_pending({"tool_call_history": []}, None) == ([], [])
    assert dedup_pending({}, None) == ([], [])
    assert dedup_pending(None, None) == ([], [])
    assert dedup_pending(None, [{"tool": "T", "args": {}}]) == ([{"tool": "T", "args": {}}], [])
    cleaned, skipped = dedup_pending({}, ["not-a-dict", {"tool": "T", "args": {}}])
    assert cleaned == [{"tool": "T", "args": {}}]
    assert skipped == []
