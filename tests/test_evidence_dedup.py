"""plan 去重（guardrails/dedup.py）单测。"""

from __future__ import annotations

from pra.agent.guardrails.dedup import canonical_args, dedup_pending


def _ok_record(tool: str, args: dict, seq: int) -> dict:
    return {"tool": tool, "args": args, "status": "ok", "seq": seq, "tokens": 0}


def test_canonical_args_key_order_independent():
    """canonical 序列化：key 排序 + ``default=str``。"""
    assert canonical_args({"a": 1, "b": 2}) == canonical_args({"b": 2, "a": 1})
    assert canonical_args({"when": "x"}) != canonical_args({"when": "x", "extra": "y"})


def test_dedup_ok_hit_goes_to_skipped_with_seq_continuing():
    """命中已执行成功 → 进 skipped（seq 从 ``len(history)+1`` 顺延），cleaned 为空。"""
    state = {"tool_call_history": [_ok_record("ProductTool", {"u": "u1"}, 1)]}
    planned = [
        {"tool": "ProductTool", "args": {"u": "u1"}, "reason": "r", "priority": 1},
    ]
    cleaned, skipped = dedup_pending(state, planned)
    assert cleaned == []
    assert len(skipped) == 1
    record = skipped[0]
    assert record["tool"] == "ProductTool" and record["args"] == {"u": "u1"}
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
    assert cleaned == planned


def test_dedup_same_round_self_dedup_silent():
    state = {"tool_call_history": []}
    planned = [
        {"tool": "ProductTool", "args": {"product_id": "P1"}, "priority": 1},
        {"tool": "ProductTool", "args": {"product_id": "P1"}, "priority": 1},
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
    """cleaned 保留原 dict 的全部内容。"""

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
    assert [c["tool"] for c in cleaned] == ["MerchantTool"]


def test_dedup_none_and_defensive_inputs():
    assert dedup_pending({"tool_call_history": []}, None) == ([], [])
    assert dedup_pending({}, None) == ([], [])
    assert dedup_pending(None, None) == ([], [])
    assert dedup_pending(None, [{"tool": "T", "args": {}}]) == ([{"tool": "T", "args": {}}], [])
    cleaned, skipped = dedup_pending({}, ["not-a-dict", {"tool": "T", "args": {}}])
    assert cleaned == [{"tool": "T", "args": {}}]
    assert skipped == []
