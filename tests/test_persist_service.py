"""persist 服务（pra/infra/persist_service.py）—— **不连真库**的单测。

run_and_persist 的真库冒烟路径由 scripts/demo_api.py（协调者/带 DB 环境执行）覆盖，
单测只钉本模块的**纯映射/摘要函数**（无 DB/无 I/O）：_enum_value / _token_count /
_json_cap / _hypothesis_summary / _node_output_summary（各节点 trace output_json 形状，
供落库 review_trace 列的确定性契约）。
"""

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
    _node_output_summary,
    _token_count,
)
from helpers import ev, hp


def test_enum_value_maps_enum_to_value():
    assert _enum_value(Decision.PASS) == "PASS"
    assert _enum_value(RiskLevel.HIGH) == "HIGH"
    assert _enum_value(3) == 3  # 普通值原样返回
    assert _enum_value("str") == "str"


def test_token_count_accepts_budget_and_dict():
    assert _token_count(Budget(tokens=123)) == 123
    assert _token_count({"tokens": 45}) == 45
    assert _token_count({"tokens": None}) == 0
    assert _token_count(None) == 0


def test_json_cap_pass_through_primitives_and_small_objects():
    assert _json_cap(None) is None
    assert _json_cap(5) == 5
    assert _json_cap("text") == "text"
    obj = {"a": 1, "b": ["x", "y"]}
    assert _json_cap(obj) == obj
    assert _json_cap(obj) is obj  # 正常路径原对象返回（可序列化）


def test_json_cap_truncates_oversized_objects():
    big = {"payload": "x" * 1000}
    capped = _json_cap(big, cap=64)
    assert capped["_truncated"] is True
    assert capped["_type"] == "dict"
    assert capped["_length"] > 64
    assert capped["_preview"]  # 截断预览非空


def test_hypothesis_summary_shape():
    h = Hypothesis(
        id="H1", statement="刻意规避品牌识别", prior=0.4,
        posterior=0.91, status=HypothesisStatus.SUPPORTED,
    )
    summary = _hypothesis_summary(h)
    assert summary == {
        "id": "H1",
        "statement": "刻意规避品牌识别",
        "status": "SUPPORTED",  # Enum → .value（JSON 标量）
        "prior": 0.4,
        "posterior": 0.91,
    }


def test_node_output_summary_hypothesize():
    hypos = [hp("H1", prior=0.5, status=HypothesisStatus.REFUTED, posterior=0.05)]
    update = {
        "hypotheses": hypos,
        "investigation_queue": [{"q": "外观问题", "priority": 1, "status": "OPEN"}],
        "degraded": False,
        "budget": Budget(llm_calls=2, tool_calls=0, tokens=30),
    }
    s = _node_output_summary("hypothesize", update)
    assert s["node"] == "hypothesize"
    assert s["degraded"] is False
    assert s["hypotheses_count"] == 1
    assert s["hypotheses"][0]["status"] == "REFUTED"
    assert s["investigation_queue_count"] == 1
    assert s["investigation_queue"][0] == {"q": "外观问题", "priority": 1, "status": "OPEN"}
    assert s["budget"] == {"llm_calls": 2, "tool_calls": 0, "tokens": 30}


def test_node_output_summary_plan_dedup_skipped():
    update = {
        "pending_tool_calls": [
            {"tool": "ProductTool", "args": {}, "reason": "r", "priority": 1}
        ],
        "tool_call_history": [  # dedup 跳过审计并入本行
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
        "investigation_queue": [
            {"q": "q1", "priority": 1, "status": "DONE"},
            {"q": "q2", "priority": 2, "status": "OPEN"},
        ],
        "degraded": False,
        "budget": Budget(),
    }
    s = _node_output_summary("reevaluate", update)
    assert s["node"] == "reevaluate"
    assert s["hypotheses_count"] == 3
    assert s["status_counts"] == {"SUPPORTED": 1, "REFUTED": 1, "PENDING": 1}
    assert s["queue_done_count"] == 1
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
    """decide update 无 decision（理论不出现）→ decision=None 摘要，不抛。"""
    s = _node_output_summary("decide", {"degraded": True})
    assert s["node"] == "decide"
    assert s["degraded"] is True
    assert s["decision"] is None
