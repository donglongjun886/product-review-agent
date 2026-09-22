"""``merge_evidence`` 单测：指纹规则、幂等与保序（行为级，经公开 merge API 锁定）。

覆盖 merge 的 None 入参、right 自去重、同 key 丢弃新增、``ref_id`` 为 None 时回退
``value``（防同源证据互相吞并）、同 ``ref_id`` 下 weight/extra/value 差异不阻止判重，
以及结果保序。指纹规则（``_evidence_key``）的语义一律通过 merge 行为间接验证，
不断言私有函数返回的内部元组形状。
"""

from __future__ import annotations

from pra.agent.state import merge_evidence
from helpers import ev


def test_merge_same_ref_id_ignores_weight_extra_and_value_diffs():
    """同 ``ref_id`` 下 weight/extra/value 差异不阻止判重（指纹取 ref_id，保留首条）。

    “``ref_id`` 优先于 value、``ref_id`` 为 None 回退 value”两个语义已分别由
    ``test_merge_same_key_in_right_dropped_idempotent`` 与
    ``test_merge_ref_none_values_distinct_both_kept`` 以 merge 行为锁定。
    """
    a = ev("POLICY_REF", source="PolicySearchTool", value="POLICY_3.2 v2 条款：x",
           weight=0.9, ref_id="POLICY_3.2_v2_c1")
    b = ev("POLICY_REF", source="PolicySearchTool", value="完全不同的内容",
           weight=0.5, ref_id="POLICY_3.2_v2_c1", extra={"policy_id": "POLICY_3.2"})
    assert merge_evidence([a], [b]) == [a]


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
    left_e = ev("A", source="S", value="left", weight=0.9, ref_id="r1")
    right_e = ev("A", source="S", value="right-new", weight=0.5, ref_id="r1")
    out = merge_evidence([left_e], [right_e])
    assert out == [left_e]  # 新增被丢，保留左（首条不可篡改）


def test_merge_new_key_appended_in_order():
    l1 = ev("A", value="a")
    r1, r2 = ev("B", value="b"), ev("C", value="c")
    out = merge_evidence([l1], [r1, r2])
    assert out == [l1, r1, r2]


def test_merge_right_self_dedup():
    r1 = ev("A", source="S", value="v1", ref_id="x")
    dup = ev("A", source="S", value="v2", ref_id="x")
    other = ev("B", value="b")
    out = merge_evidence([], [r1, dup, other])
    assert out == [r1, other]


def test_merge_ref_none_values_distinct_both_kept():
    a = ev("MERCHANT_HISTORY", value="23 similar / 5 removals", weight=0.85)
    b = ev("MERCHANT_HISTORY", value="0 similar / 0 removals", weight=0.85)
    out = merge_evidence([a], [b])
    assert out == [a, b]


def test_merge_left_unchanged_mutation_safety():
    l = [ev("A", value="a")]
    r = [ev("A", value="b", ref_id="same-ref")]
    snapshot = (list(l), list(r))
    merge_evidence(l, r)
    assert len(l) == snapshot[0].__len__() and l[0].value == "a"
    assert r[0].value == "b"
