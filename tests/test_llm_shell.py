"""LLM 壳（guardrails/llm_shell.py）单测：强校验 + 失败分类重试 + 失败不抛异常。"""

from __future__ import annotations

import pytest

from pra.agent.guardrails.llm_shell import (
    LLMBackendError,
    LLMCallOutcome,
    LLMResponse,
    call_structured_llm,
)
from pra.agent.guardrails.schemas import PlanOutput
from helpers import AlwaysRaiseBackend, plan_conclude_json

_STATE = {"case": {"case_id": "CASE_SHELL_01"}}
_INVALID = "this is not valid json"
_SCHEMA_BAD = '{"next_action": "SOMETHING_ELSE", "tools": []}'


class _FeedbackSequenceBackend:
    """按序返回 content 的替身（元素 None = 抛后端异常）；**记录每次 state + feedback**。"""

    name = "test-sequence"

    def __init__(self, contents: list, tokens: int = 7) -> None:
        self._contents = list(contents)
        self._tokens = tokens
        self.calls: list[dict] = []

    async def complete(
        self, *, node: str, state: dict, json_schema: dict, feedback: list[str] | None = None
    ) -> LLMResponse:
        self.calls.append({"state": state, "feedback": feedback})
        if not self._contents:
            raise LLMBackendError("stub 内容耗尽")
        content = self._contents.pop(0)
        if content is None:
            raise LLMBackendError("injected backend failure")
        return LLMResponse(content=content, tokens=self._tokens)


async def _call(backend, *, output_model=PlanOutput, node="plan", state=None):
    """用显式注入的 backend 跑一次 call_structured_llm。"""
    return await call_structured_llm(
        OutputModel=output_model,
        node=node,
        llm=backend,
        state=dict(state) if state is not None else dict(_STATE),
    )


async def test_schema_fail_then_success_retries_once():
    backend = _FeedbackSequenceBackend(contents=[_SCHEMA_BAD, plan_conclude_json()], tokens=5)
    outcome = await _call(backend)
    assert isinstance(outcome, LLMCallOutcome)
    assert outcome.model is not None
    assert isinstance(outcome.model, PlanOutput)
    assert outcome.model.next_action == "conclude"
    assert outcome.attempts == 2
    assert outcome.error is None
    assert outcome.tokens == 10
    assert len(backend.calls) == 2
    assert backend.calls[0]["feedback"] in (None, [])
    assert backend.calls[0]["state"] == _STATE
    repair = backend.calls[1]["feedback"]
    assert isinstance(repair, list) and len(repair) == 1
    assert isinstance(repair[0], str)
    assert "重新输出" in repair[0]
    assert '"next_action": "SOMETHING_ELSE"' in repair[0]
    assert "validation" in repair[0].lower()
    assert backend.calls[1]["state"] == _STATE


async def test_valid_first_attempt_succeeds():
    backend = _FeedbackSequenceBackend(contents=[plan_conclude_json()], tokens=3)
    outcome = await _call(backend)
    assert outcome.model is not None and outcome.attempts == 1
    assert outcome.tokens == 3
    assert len(backend.calls) == 1
    assert backend.calls[0]["feedback"] in (None, [])


async def test_always_invalid_two_failures_no_raise():
    backend = _FeedbackSequenceBackend(contents=[_INVALID, _SCHEMA_BAD], tokens=0)
    outcome = await _call(backend)
    assert outcome.model is None
    assert outcome.attempts == 2
    assert outcome.error
    assert "validation" in outcome.error.lower() or "ValidationError" in outcome.error


async def test_backend_raises_both_attempts_no_raise():
    backend = AlwaysRaiseBackend()
    outcome = await _call(backend)
    assert backend.calls == 2
    assert outcome.model is None
    assert outcome.attempts == 2
    assert outcome.error == "injected backend failure"
    assert outcome.tokens == 0


async def test_backend_raise_then_success_recovers():
    backend = _FeedbackSequenceBackend(contents=[None, plan_conclude_json()], tokens=4)
    outcome = await _call(backend)
    assert outcome.model is not None and outcome.model.next_action == "conclude"
    assert outcome.attempts == 2
    assert outcome.tokens == 4
    assert len(backend.calls) == 2
    assert backend.calls[1]["feedback"] in (None, [])


async def test_caller_state_not_polluted():
    state = dict(_STATE)
    snapshot = {"case": dict(state["case"])}
    backend = _FeedbackSequenceBackend(contents=[_SCHEMA_BAD, plan_conclude_json()], tokens=1)
    await _call(backend, state=state)
    assert state == snapshot


async def test_missing_backend_raises_type_error():
    """``llm=None``（装配缺陷）→ ``TypeError``。"""
    with pytest.raises(TypeError, match="llm=None"):
        await call_structured_llm(
            OutputModel=PlanOutput, node="plan", state=dict(_STATE), llm=None
        )


async def test_invalid_output_model_type_returns_error():
    outcome = await call_structured_llm(
        OutputModel=dict, node="plan", state={}, llm=AlwaysRaiseBackend()
    )
    assert outcome.model is None
    assert outcome.attempts == 1
    assert outcome.error
    assert "dict" in (outcome.error or "")


async def test_backend_protocol_violation_is_caught_as_failure():
    class BoomBackend:
        name = "boom"

        async def complete(self, *, node, state, json_schema, feedback=None):
            raise KeyError("anything unexpected")

    outcome = await _call(BoomBackend())
    assert outcome.model is None
    assert outcome.attempts == 2
    assert "anything unexpected" in (outcome.error or "")


class _TruncatedOnceBackend:
    """返回一次 truncated 且内容非法的替身；再调用即失败哨兵。"""

    name = "test-truncated-once"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self, *, node: str, state: dict, json_schema: dict, feedback: list[str] | None = None
    ) -> LLMResponse:
        self.calls += 1
        if self.calls > 1:
            raise AssertionError("截断失败不应触发第 2 次调用")
        return LLMResponse(content=_INVALID, tokens=3, truncated=True)


async def test_truncated_failure_is_not_retried_attempts_one():
    backend = _TruncatedOnceBackend()
    outcome = await _call(backend)
    assert backend.calls == 1
    assert outcome.model is None
    assert outcome.attempts == 1
    assert outcome.tokens == 3
    assert outcome.error and "截断" in outcome.error


class _TruncatedValidBackend:
    """返回一次 truncated 但内容恰好合法的替身 —— 应照常成功。"""

    name = "test-truncated-valid"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(
        self, *, node: str, state: dict, json_schema: dict, feedback: list[str] | None = None
    ) -> LLMResponse:
        self.calls += 1
        if self.calls > 1:
            raise AssertionError("合法截断输出不应触发第 2 次调用")
        return LLMResponse(content=plan_conclude_json(), tokens=5, truncated=True)


async def test_truncated_but_valid_content_succeeds():
    backend = _TruncatedValidBackend()
    outcome = await _call(backend)
    assert backend.calls == 1
    assert outcome.model is not None and outcome.model.next_action == "conclude"
    assert outcome.attempts == 1
    assert outcome.error is None
    assert outcome.tokens == 5
