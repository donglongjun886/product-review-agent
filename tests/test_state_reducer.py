"""``merge_evidence`` 单测：指纹规则、幂等与保序（行为级，经公开 merge API 锁定）。"""

from __future__ import annotations

from helpers import ev

from pra.agent.state import merge_evidence


def test_merge_same_ref_id_ignores_weight_extra_and_value_diffs():
    """同 ``ref_id`` 下 weight/extra/value 差异不阻止判重（指纹取 ref_id，保留首条）。"""
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
    assert out == [left_e]


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


def test_initial_state_carries_llm_history_channels():
    """``build_initial_state`` 带 append-only 对话历史 channel 初值（空历史 + 游标 0）。"""
    from helpers import make_case

    from pra.agent.state import build_initial_state

    state = build_initial_state(make_case())
    assert state["llm_messages"] == []
    assert state["llm_evidence_cursor"] == 0


def test_llm_history_channels_survive_checkpoint_serde():
    """llm_messages / llm_evidence_cursor 经 checkpointer serde 往返不丢（纯 dict/int 原生支持）。"""
    from pra.agent.checkpointer import make_serde

    serde = make_serde()
    data = {
        "llm_messages": [
            {"role": "user", "content": "案件事实…"},
            {"role": "assistant", "content": '{"hypotheses": []}'},
        ],
        "llm_evidence_cursor": 3,
    }
    _, blob = serde.dumps_typed(data)
    restored = serde.loads_typed(serde.dumps_typed(data))
    assert restored == data
    assert isinstance(blob, bytes)
