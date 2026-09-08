"""LLM 壳（guardrails/llm_shell.py）单测 —— 强校验 + 重试 1 次 + 失败不抛异常。

注入替身（实现 LLMBackend Protocol，见 tests/helpers.py）验证：
- schema 校验失败 → 追加修正提示重试 1 次 → 成功 attempts=2；
- 恒失败 / 后端异常 → model=None、attempts=2、不抛异常；
- set_llm_backend(None) 恢复默认 scripted 桩；
- 调用方 messages 不被污染（修正提示只追加在工作副本）。
"""

from __future__ import annotations

import pytest

from pra.agent.guardrails.llm_shell import (
    LLMCallOutcome,
    LLMBackendError,
    call_structured_llm,
    get_llm_backend,
    set_llm_backend,
)
from pra.agent.guardrails.schemas import PlanOutput
from helpers import AlwaysRaiseBackend, SequenceBackend, plan_conclude_json

_MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "__STATE__ {}"}]
_INVALID = "this is not valid json"
# 合法 JSON 但 schema 不满足（next_action 超词表）→ pydantic ValidationError
_SCHEMA_BAD = '{"next_action": "SOMETHING_ELSE", "tools": []}'


async def _call(backend, *, output_model=PlanOutput, node="plan",
                messages=None):
    set_llm_backend(backend)
    return await call_structured_llm(
        OutputModel=output_model, node=node, messages=messages if messages is not None
        else [dict(m) for m in _MESSAGES],
    )


async def test_schema_fail_then_success_retries_once():
    """校验失败 → 修正提示重试 1 次 → 成功：attempts=2、model 校验通过。"""
    backend = SequenceBackend(contents=[_SCHEMA_BAD, plan_conclude_json()], tokens=5)
    outcome = await _call(backend)
    assert isinstance(outcome, LLMCallOutcome)
    assert outcome.model is not None
    assert isinstance(outcome.model, PlanOutput)
    assert outcome.model.next_action == "conclude"
    assert outcome.attempts == 2
    assert outcome.error is None
    assert outcome.tokens == 10  # 两次响应 tokens 累计
    # 后端第 2 次收到的 messages 含修正提示（role=user、含"重新输出"文案）
    assert len(backend.calls) == 2
    assert len(backend.calls[1]) == len(_MESSAGES) + 1
    correction = backend.calls[1][-1]
    assert correction["role"] == "user"
    assert "请严格按 Schema 重新输出" in correction["content"]


async def test_valid_first_attempt_succeeds():
    """首次即校验通过 → attempts=1、无修正提示追加。"""
    backend = SequenceBackend(contents=[plan_conclude_json()], tokens=3)
    outcome = await _call(backend)
    assert outcome.model is not None and outcome.attempts == 1
    assert outcome.tokens == 3
    assert len(backend.calls) == 1
    assert len(backend.calls[0]) == len(_MESSAGES)  # 未追加修正


async def test_always_invalid_two_failures_no_raise():
    """两次均 schema 校验失败 → model=None、attempts=2、不抛异常，error 含校验错误。"""
    backend = SequenceBackend(contents=[_INVALID, _SCHEMA_BAD], tokens=0)
    outcome = await _call(backend)
    assert outcome.model is None
    assert outcome.attempts == 2
    assert outcome.error  # 最后一次错误文本（供 failure.reason）
    assert "validation" in outcome.error.lower() or "ValidationError" in outcome.error


async def test_backend_raises_both_attempts_no_raise():
    """后端异常（LLMBackendError）也走修正重试；两次失败 → model=None、attempts=2。"""
    backend = AlwaysRaiseBackend()
    outcome = await _call(backend)
    assert backend.calls == 2
    assert outcome.model is None
    assert outcome.attempts == 2
    assert outcome.error == "injected backend failure"
    assert outcome.tokens == 0


async def test_backend_raise_then_success_recovers():
    """第 1 次后端异常、第 2 次成功 → 恢复：attempts=2、model 校验通过。"""
    backend = SequenceBackend(contents=[None, plan_conclude_json()], tokens=4)
    outcome = await _call(backend)
    assert outcome.model is not None and outcome.model.next_action == "conclude"
    assert outcome.attempts == 2
    assert outcome.tokens == 4
    assert len(backend.calls) == 2
    correction = backend.calls[1][-1]
    assert correction["role"] == "user" and "重新输出" in correction["content"]


async def test_caller_messages_not_polluted():
    """修正提示只追加在工作副本 —— 调用方传入的 messages 列表不被改动。"""
    messages = [dict(m) for m in _MESSAGES]
    snapshot = [dict(m) for m in messages]
    backend = SequenceBackend(contents=[_SCHEMA_BAD, plan_conclude_json()], tokens=1)
    await _call(backend, messages=messages)
    assert messages == snapshot
    assert len(messages) == len(_MESSAGES)


async def test_set_llm_backend_none_restores_default():
    """set_llm_backend(None) 恢复默认 scripted 桩（name="scripted-walkthrough"）。"""
    set_llm_backend(SequenceBackend(contents=[plan_conclude_json()]))
    assert get_llm_backend().name == "test-sequence"
    set_llm_backend(None)
    backend = get_llm_backend()
    assert backend.name == "scripted-walkthrough"
    # 默认桩可正常服务（node=plan 需 __STATE__ 兜底 → conclude）
    resp = await backend.complete(node="plan", messages=[], json_schema={})
    assert "conclude" in resp.content


async def test_invalid_output_model_type_returns_error():
    """OutputModel 非 pydantic 模型 → 不可恢复失败（attempts=1、不碰后端）。"""
    outcome = await call_structured_llm(OutputModel=dict, node="plan", messages=[])
    assert outcome.model is None
    assert outcome.attempts == 1
    assert "不是 pydantic 模型" in (outcome.error or "")


async def test_backend_protocol_violation_is_caught_as_failure():
    """后端抛非 LLMBackendError 的任意异常（如 KeyError）同样被捕获为失败。"""

    class BoomBackend:
        name = "boom"

        async def complete(self, *, node, messages, json_schema):
            raise KeyError("anything unexpected")

    outcome = await _call(BoomBackend())
    assert outcome.model is None
    assert outcome.attempts == 2
    assert "anything unexpected" in (outcome.error or "")


async def test_backend_error_type_exposed():
    assert issubclass(LLMBackendError, RuntimeError)
    with pytest.raises(LLMBackendError):
        raise LLMBackendError("boom")
