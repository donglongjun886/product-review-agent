"""步骤失败分级（guardrails/errors.py）单测：failures 形状与关键失败口径。

``make_failure``：``step_type``/``severity``/``reason``/``ts`` 必需，``tool``/``seq``
可选（有才带）。``key_tool_failure``：critical TOOL_CALL 失败且其后（``seq`` 更大）
无同 tool 的 status=ok 调用 → True（关键证据缺失）；同 tool 后置 ok 即解决；
warn 不触发；非 TOOL_CALL 不算。
"""

from __future__ import annotations

from pra.agent.guardrails.errors import (
    SEV_CRITICAL,
    SEV_WARN,
    STEP_DECIDE,
    STEP_TOOL_CALL,
    key_tool_failure,
    make_failure,
)


def test_make_failure_default_shape():
    f = make_failure(step_type=STEP_DECIDE, reason="schema 校验重试仍失败")
    assert set(f.keys()) == {"step_type", "severity", "reason", "ts"}
    assert f["step_type"] == STEP_DECIDE
    assert f["severity"] == SEV_WARN
    assert f["reason"] == "schema 校验重试仍失败"
    assert isinstance(f["ts"], str) and f["ts"]  # UTC ISO8601，非空


def test_make_failure_with_tool_and_seq():
    f = make_failure(step_type=STEP_TOOL_CALL, severity=SEV_CRITICAL,
                     reason="boom", tool="ImageAnalysisTool", seq=3)
    assert f["tool"] == "ImageAnalysisTool"
    assert f["seq"] == 3
    assert f["severity"] == SEV_CRITICAL


def test_key_tool_failure_critical_failure_with_later_ok_resolved():
    state = {"tool_call_history": [{"tool": "ImageAnalysisTool", "status": "ok", "seq": 2}]}
    failures = [make_failure(step_type=STEP_TOOL_CALL, severity=SEV_CRITICAL,
                             reason="boom", tool="ImageAnalysisTool", seq=1)]
    assert key_tool_failure(state, failures) is False


def test_key_tool_failure_no_ok_after_true():
    state = {"tool_call_history": [{"tool": "ProductTool", "status": "ok", "seq": 5}]}
    failures = [make_failure(step_type=STEP_TOOL_CALL, severity=SEV_CRITICAL,
                             reason="boom", tool="ImageAnalysisTool", seq=1)]
    assert key_tool_failure(state, failures) is True


def test_key_tool_failure_ok_before_failure_still_unresolved():
    state = {"tool_call_history": [{"tool": "ImageAnalysisTool", "status": "ok", "seq": 1}]}
    failures = [make_failure(step_type=STEP_TOOL_CALL, severity=SEV_CRITICAL,
                             reason="boom", tool="ImageAnalysisTool", seq=2)]
    assert key_tool_failure(state, failures) is True


def test_key_tool_failure_no_history_at_all_true():
    failures = [make_failure(step_type=STEP_TOOL_CALL, severity=SEV_CRITICAL,
                             reason="boom", tool="ImageAnalysisTool", seq=1)]
    assert key_tool_failure({"tool_call_history": []}, failures) is True


def test_key_tool_failure_warn_does_not_trigger():
    failures = [make_failure(step_type=STEP_TOOL_CALL, severity=SEV_WARN,
                             reason="非关键抖动", tool="ImageAnalysisTool", seq=1)]
    assert key_tool_failure({"tool_call_history": []}, failures) is False


def test_key_tool_failure_non_tool_call_step_ignored():
    failures = [make_failure(step_type=STEP_DECIDE, severity=SEV_CRITICAL,
                             reason="schema 校验重试仍失败")]
    assert key_tool_failure({"tool_call_history": []}, failures) is False


def test_key_tool_failure_failure_without_tool_skipped():
    failures = [make_failure(step_type=STEP_TOOL_CALL, severity=SEV_CRITICAL,
                             reason="boom")]
    assert key_tool_failure({"tool_call_history": []}, failures) is False


def test_key_tool_failure_empty_and_missing_safe():
    assert key_tool_failure({}, []) is False
    assert key_tool_failure({"tool_call_history": None}, []) is False


def test_key_tool_failure_multiple_failures_any_triggers():
    state = {
        "tool_call_history": [
            {"tool": "ImageAnalysisTool", "status": "ok", "seq": 4},  # 解决 H1
        ]
    }
    failures = [
        make_failure(step_type=STEP_TOOL_CALL, severity=SEV_CRITICAL, reason="boom",
                     tool="ImageAnalysisTool", seq=1),  # 已解决
        make_failure(step_type=STEP_TOOL_CALL, severity=SEV_CRITICAL, reason="boom",
                     tool="MerchantTool", seq=2),  # 未解决
    ]
    assert key_tool_failure(state, failures) is True
