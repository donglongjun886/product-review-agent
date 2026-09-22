"""证据质量过滤（guardrails/evidence.py）与 plan 去重（guardrails/dedup.py）。

``quality_filter``：只按 weight 丢弱 ``IMAGE_SIMILARITY``（0.70 保留 / 0.699 丢弃），其它类型全保留。
``dedup_pending``：ok 命中 → skipped、error 允许重试、同轮自去重、canonical 键序无关。
"""

from __future__ import annotations

from pra.agent.guardrails.dedup import canonical_args, dedup_pending
from pra.agent.guardrails.evidence import quality_filter
from helpers import ev


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
    assert out == [e]  # 保留元素内容不变（是否复用同一对象属实现细节，不断言）


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
    assert len(skipped) == 1
    record = skipped[0]
    # 关键字段语义：命中去重、seq 顺延、原因可归因；不锁整条记录的全部键位
    assert record["tool"] == "ImageAnalysisTool" and record["args"] == {"u": "u1"}
    assert record["status"] == "skipped"
    assert record["seq"] == 2
    assert record["reason"] == "duplicate"


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


def test_dedup_cleaned_keep_content():
    """cleaned 保留原 dict 的全部内容（后续执行不会因去重丢字段）。"""

    planned = [{"tool": "ProductTool", "args": {"product_id": "P1"}, "reason": "r", "priority": 1}]
    cleaned, _ = dedup_pending({"tool_call_history": []}, planned)
    assert cleaned[0] == planned[0]


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
