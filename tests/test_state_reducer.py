"""merge_evidence / _evidence_key（docs/04-graph-design.md §2.3，O-1 拍板）单测。

覆盖：merge 幂等（left None / right None / right 自去重 / 同 key 丢弃新增 /
ref_id None 回退 value 防互相吞并）；_evidence_key 指纹规则；结果保序确定性。
"""

from __future__ import annotations

from pra.agent.state import _evidence_key, merge_evidence
from helpers import ev


def test_evidence_key_ref_id_priority():
    """O-1：ref_id 非 None 时指纹 = (type, source, ref_id)，与 value 无关。"""
    e1 = ev("IMAGE_SIMILARITY", source="ImageAnalysisTool", value="similarity=0.91",
            weight=0.91, ref_id="img1")
    e2 = ev("IMAGE_SIMILARITY", source="ImageAnalysisTool", value="similarity=0.91, match=x",
            weight=0.91, ref_id="img1")
    assert _evidence_key(e1) == _evidence_key(e2)
    assert _evidence_key(e1) == ("IMAGE_SIMILARITY", "ImageAnalysisTool", "img1")


def test_evidence_key_none_ref_falls_back_to_value():
    """O-1：ref_id None 回退 value —— 同 (type, source) 无 ref 证据不互相吞并。"""
    a = ev("MERCHANT_HISTORY", source="MerchantTool", value="v1", weight=0.85, ref_id=None)
    b = ev("MERCHANT_HISTORY", source="MerchantTool", value="v2", weight=0.85, ref_id=None)
    assert _evidence_key(a) == ("MERCHANT_HISTORY", "MerchantTool", "v1")
    assert _evidence_key(b) == ("MERCHANT_HISTORY", "MerchantTool", "v2")
    assert _evidence_key(a) != _evidence_key(b)


def test_evidence_key_ignores_weight_and_extra():
    """指纹不含 weight/extra —— 同源同 ref 的新证据只留首条（证据不可篡改）。"""
    a = ev("POLICY_REF", source="PolicySearchTool", value="POLICY_3.2 v2 条款：x",
           weight=0.9, ref_id="POLICY_3.2_v2_c1")
    b = ev("POLICY_REF", source="PolicySearchTool", value="完全不同的内容",
           weight=0.5, ref_id="POLICY_3.2_v2_c1", extra={"policy_id": "POLICY_3.2"})
    assert _evidence_key(a) == _evidence_key(b)


def test_merge_left_none():
    """left 为 None（首写/空态）按空列表处理 → 全量收右。"""
    e1, e2 = ev("A", value="1"), ev("B", value="2")
    out = merge_evidence(None, [e1, e2])
    assert out == [e1, e2]


def test_merge_right_none():
    """right 为 None/空 → 原样返回 left（不崩、不丢）。"""
    e1, e2 = ev("A", value="1"), ev("B", value="2")
    assert merge_evidence([e1, e2], None) == [e1, e2]
    assert merge_evidence([e1, e2], []) == [e1, e2]


def test_merge_both_empty():
    assert merge_evidence([], []) == []
    assert merge_evidence(None, None) == []


def test_merge_same_key_in_right_dropped_idempotent():
    """同 key 已存在 → 丢弃新增（幂等：重放/重试不重复累积）。"""
    left_e = ev("A", source="S", value="left", weight=0.9, ref_id="r1")
    right_e = ev("A", source="S", value="right-new", weight=0.5, ref_id="r1")
    out = merge_evidence([left_e], [right_e])
    assert out == [left_e]  # 新增被丢，保留左（首条不可篡改）


def test_merge_new_key_appended_in_order():
    """新 key append，结果保序：left 原序在前、新增按 right 出现序在后。"""
    l1 = ev("A", value="a")
    r1, r2 = ev("B", value="b"), ev("C", value="c")
    out = merge_evidence([l1], [r1, r2])
    assert out == [l1, r1, r2]


def test_merge_right_self_dedup():
    """right 内部自去重：同 key 在 right 出现多次只收首次。"""
    r1 = ev("A", source="S", value="v1", ref_id="x")
    dup = ev("A", source="S", value="v2", ref_id="x")
    other = ev("B", value="b")
    out = merge_evidence([], [r1, dup, other])
    assert out == [r1, other]


def test_merge_ref_none_values_distinct_both_kept():
    """ref_id 全 None 但 value 不同的两条 → key 不同 → 两条都留（不互相吞并）。"""
    a = ev("MERCHANT_HISTORY", value="23 similar / 5 removals", weight=0.85)
    b = ev("MERCHANT_HISTORY", value="0 similar / 0 removals", weight=0.85)
    out = merge_evidence([a], [b])
    assert out == [a, b]


def test_merge_left_unchanged_mutation_safety():
    """reducer 不改写入参列表与元素（纯函数）。"""
    l = [ev("A", value="a")]
    r = [ev("A", value="b", ref_id="same-ref")]
    snapshot = (list(l), list(r))
    merge_evidence(l, r)
    assert len(l) == snapshot[0].__len__() and l[0].value == "a"
    assert r[0].value == "b"
