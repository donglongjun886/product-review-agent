"""步骤失败分级（guardrails/errors.py）单测：failures 形状。"""

from __future__ import annotations

from pra.agent.guardrails.errors import (
    SEV_CRITICAL,
    SEV_WARN,
    STEP_DECIDE,
    STEP_TOOL_CALL,
    make_failure,
)


def test_make_failure_default_shape():
    f = make_failure(step_type=STEP_DECIDE, reason="schema 校验重试仍失败")
    assert set(f.keys()) == {"step_type", "severity", "reason", "ts"}
    assert f["step_type"] == STEP_DECIDE
    assert f["severity"] == SEV_WARN
    assert f["reason"] == "schema 校验重试仍失败"
    assert isinstance(f["ts"], str) and f["ts"]


def test_make_failure_with_tool_and_seq():
    f = make_failure(step_type=STEP_TOOL_CALL, severity=SEV_CRITICAL,
                     reason="boom", tool="ProductTool", seq=3)
    assert f["tool"] == "ProductTool"
    assert f["seq"] == 3
    assert f["severity"] == SEV_CRITICAL
