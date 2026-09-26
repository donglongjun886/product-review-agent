"""persist 服务的**不连真库**单测：只钉纯映射/摘要函数，无 DB、无 I/O。"""

from __future__ import annotations

from pra.domain.models import (
    Budget,
    Decision,
    Hypothesis,
    HypothesisStatus,
    RiskLevel,
    RiskType,
    ReviewDecision,
)
from pra.infra.persist_service import (
    _enum_value,
    _hypothesis_summary,
    _json_cap,
    _node_input_summary,
    _node_output_summary,
    _token_count,
)
from helpers import ev, hp


def test_enum_value_maps_enum_to_value():
    assert _enum_value(Decision.PASS) == "PASS"
    assert _enum_value(RiskLevel.HIGH) == "HIGH"
    assert _enum_value(3) == 3
    assert _enum_value("str") == "str"


def test_token_count_accepts_budget_and_none():
    assert _token_count(Budget(tokens=123)) == 123
    assert _token_count(None) == 0


def test_json_cap_pass_through_primitives_and_small_objects():
    assert _json_cap(None) is None
    assert _json_cap(5) == 5
    assert _json_cap("text") == "text"
    obj = {"a": 1, "b": ["x", "y"]}
    assert _json_cap(obj) == obj
    assert _json_cap(obj) is obj


def test_json_cap_truncates_oversized_objects():
    big = {"payload": "x" * 1000}
    capped = _json_cap(big, cap=64)
    assert capped["_truncated"] is True
    assert capped["_type"] == "dict"
    assert capped["_length"] > 64
    assert capped["_preview"]


def test_hypothesis_summary_shape():
    h = Hypothesis(
        id="H1", statement="刻意规避品牌识别", prior=0.4,
        posterior=0.91, status=HypothesisStatus.SUPPORTED,
    )
    summary = _hypothesis_summary(h)
    assert summary == {
        "id": "H1",
        "statement": "刻意规避品牌识别",
        "status": "SUPPORTED",
        "prior": 0.4,
        "posterior": 0.91,
    }


def test_node_output_summary_hypothesize():
    hypos = [hp("H1", prior=0.5, status=HypothesisStatus.REFUTED, posterior=0.05)]
    update = {
        "hypotheses": hypos,
        "degraded": False,
        "budget": Budget(llm_calls=2, tool_calls=0, tokens=30),
    }
    s = _node_output_summary("hypothesize", update)
    assert s["node"] == "hypothesize"
    assert s["degraded"] is False
    assert s["hypotheses_count"] == 1
    assert s["hypotheses"][0]["status"] == "REFUTED"
    assert s["budget"] == {"llm_calls": 2, "tool_calls": 0, "tokens": 30}


def test_trace_summaries_carry_no_investigation_queue_fields():
    """摘要不落 investigation_queue 队列键（即使 update 里混入）。"""
    update = {
        "hypotheses": [hp("H1", prior=0.5)],
        "investigation_queue": [{"q": "文本合规问题", "priority": 1, "status": "DONE"}],
        "budget": Budget(),
    }
    for node in ("hypothesize", "reevaluate"):
        s = _node_output_summary(node, update)
        assert "investigation_queue" not in s, node
        assert "investigation_queue_count" not in s, node
        assert "queue_done_count" not in s, node
    inp = _node_input_summary("reevaluate", update, "CASE_1")
    assert "investigation_queue_count" not in inp


def test_node_output_summary_plan_dedup_skipped():
    update = {
        "pending_tool_calls": [
            {"tool": "ProductTool", "args": {}, "reason": "r", "priority": 1}
        ],
        "tool_call_history": [
            {"tool": "ProductTool", "status": "skipped", "seq": 5, "reason": "duplicate"}
        ],
        "degraded": False,
        "budget": Budget(llm_calls=1),
    }
    s = _node_output_summary("plan", update)
    assert s["node"] == "plan"
    assert s["pending_tool_calls_count"] == 1
    assert s["pending_tool_calls"][0]["tool"] == "ProductTool"
    assert s["dedup_skipped_count"] == 1
    assert s["budget"]["llm_calls"] == 1


def test_node_output_summary_reevaluate_status_counts():
    update = {
        "hypotheses": [
            hp("H1", prior=0.5, status=HypothesisStatus.SUPPORTED, posterior=0.9),
            hp("H2", prior=0.4, status=HypothesisStatus.REFUTED, posterior=0.05),
            hp("H3", prior=0.2),
        ],
        "degraded": False,
        "budget": Budget(),
    }
    s = _node_output_summary("reevaluate", update)
    assert s["node"] == "reevaluate"
    assert s["hypotheses_count"] == 3
    assert s["status_counts"] == {"SUPPORTED": 1, "REFUTED": 1, "PENDING": 1}
    assert s["degraded"] is False


def test_node_output_summary_decide():
    decision = ReviewDecision(
        decision=Decision.HUMAN_REVIEW,
        risk_level=RiskLevel.HIGH,
        risk_type=[RiskType.POTENTIAL_IP_RISK, RiskType.EVASION_PATTERN],
        decision_confidence=0.91,
        policy=["POLICY_3.2"],
        evidence=[ev("POLICY_REF", value="POLICY_3.2 v2", weight=0.9,
                     ref_id="POLICY_3.2_v2_c1", extra={"policy_id": "POLICY_3.2"})],
        hypothesis_trace=[hp("H1", prior=0.5, status=HypothesisStatus.SUPPORTED,
                             posterior=0.91)],
        overrides=["R5_DEGRADED_OR_FAILED_STEP"],
    )
    s = _node_output_summary(
        "decide", {"decision": decision, "budget": Budget(llm_calls=8)}
    )
    assert s["node"] == "decide"
    assert s["decision"] == "HUMAN_REVIEW"
    assert s["risk_level"] == "HIGH"
    assert s["risk_type"] == ["POTENTIAL_IP_RISK", "EVASION_PATTERN"]
    assert s["decision_confidence"] == 0.91
    assert s["policy"] == ["POLICY_3.2"]
    assert s["overrides"] == ["R5_DEGRADED_OR_FAILED_STEP"]
    assert s["evidence_count"] == 1
    assert s["hypothesis_trace_count"] == 1
    assert s["budget"]["llm_calls"] == 8


def test_node_output_summary_decide_none_decision():
    """decide update 无 decision（理论不出现）→ decision=None，不抛。"""
    s = _node_output_summary("decide", {"degraded": True})
    assert s["node"] == "decide"
    assert s["degraded"] is True
    assert s["decision"] is None
