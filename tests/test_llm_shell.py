"""LLM 壳（guardrails/llm_shell.py）单测 —— 强校验 + 失败分类重试 + 失败不抛异常。

注入替身（实现 LLMBackend Protocol，见 tests/helpers.py）验证：
- schema 校验失败（后端成功返回但内容不合法）→ 追加含**非法输出原文 + 校验错误**
  的修正提示重试 1 次 → 成功 attempts=2（P2-15 回喂原文）；
- 后端异常（transport 类：超时/网络等）→ 无输出可修正，**不追加 schema 修正文案**、
  退避后原样重试 1 次；
- 截断（LLMResponse.truncated，finish_reason=length）且校验失败 → 不重试
  （attempts=1 直接降级，省一次大概率无效调用）；
- 恒失败 / 后端异常 → model=None、attempts=2、不抛异常；
- set_llm_backend(None) 恢复默认 scripted 桩；
- 调用方 messages 不被污染（修正提示只追加在工作副本）。
"""

from __future__ import annotations

import pytest

from pra.agent.guardrails.llm_shell import (
    LLMCallOutcome,
    LLMBackendError,
    LLMResponse,
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
    """校验失败 → 修正提示重试 1 次 → 成功：attempts=2、model 校验通过。

    P2-15：修正提示回喂**第 1 次非法输出原文**（模型第 2 次能看到自己上一版输出）。
    """
    backend = SequenceBackend(contents=[_SCHEMA_BAD, plan_conclude_json()], tokens=5)
    outcome = await _call(backend)
    assert isinstance(outcome, LLMCallOutcome)
    assert outcome.model is not None
    assert isinstance(outcome.model, PlanOutput)
    assert outcome.model.next_action == "conclude"
    assert outcome.attempts == 2
    assert outcome.error is None
    assert outcome.tokens == 10  # 两次响应 tokens 累计
    # 后端第 2 次收到的 messages 含修正提示（role=user、含"重新输出"文案 + 原文）
    assert len(backend.calls) == 2
    assert len(backend.calls[1]) == len(_MESSAGES) + 1
    correction = backend.calls[1][-1]
    assert correction["role"] == "user"
    assert "请严格按 Schema 重新输出" in correction["content"]
    assert '"next_action": "SOMETHING_ELSE"' in correction["content"]  # 原文回喂
    assert "validation error" in correction["content"].lower()  # 校验错误一并回喂


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
    """后端异常（transport 类，LLMBackendError）也重试 1 次；两次失败 → model=None。"""
    backend = AlwaysRaiseBackend()
    outcome = await _call(backend)
    assert backend.calls == 2
    assert outcome.model is None
    assert outcome.attempts == 2
    assert outcome.error == "injected backend failure"
    assert outcome.tokens == 0


async def test_backend_raise_then_success_recovers():
    """第 1 次后端异常（transport 类）、第 2 次成功 → 恢复：attempts=2、model 通过。

    P2-15：transport 失败没有可"修正"的输出 —— 重试**不追加 schema 修正文案**
    （旧版对超时/HTTP 也按"schema 修正"重试是误导）；messages 保持原长度纯重试。
    """
    backend = SequenceBackend(contents=[None, plan_conclude_json()], tokens=4)
    outcome = await _call(backend)
    assert outcome.model is not None and outcome.model.next_action == "conclude"
    assert outcome.attempts == 2
    assert outcome.tokens == 4
    assert len(backend.calls) == 2
    assert len(backend.calls[1]) == len(_MESSAGES)  # 未追加修正提示
    assert "重新输出" not in backend.calls[1][-1]["content"]  # 无 schema 修正文案


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


class _TruncatedOnceBackend:
    """返回一次 truncated（finish_reason=length）且内容非法的替身；再调用即失败哨兵。"""

    name = "test-truncated-once"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, *, node: str, messages: list, json_schema: dict) -> LLMResponse:
        self.calls += 1
        if self.calls > 1:
            raise AssertionError("截断失败不应触发第 2 次调用")
        return LLMResponse(content=_INVALID, tokens=3, truncated=True)


async def test_truncated_failure_is_not_retried_attempts_one():
    """P2-15（壳级）：截断（truncated=True）且校验失败 → 不重试、attempts=1、error 标注截断。

    截断按 transport 类处理：同 max_tokens 下重试大概率再截断，不白烧第 2 次调用
    （替身若被第 2 次调用会直接 AssertionError）。
    """
    backend = _TruncatedOnceBackend()
    outcome = await _call(backend)
    assert backend.calls == 1  # 恰好一次
    assert outcome.model is None
    assert outcome.attempts == 1
    assert outcome.tokens == 3  # 截断响应本身已计 token
    assert outcome.error and "截断" in outcome.error


class _TruncatedValidBackend:
    """返回一次 truncated 但内容恰好合法的替身 —— 应照常成功（不浪费）。"""

    name = "test-truncated-valid"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, *, node: str, messages: list, json_schema: dict) -> LLMResponse:
        self.calls += 1
        if self.calls > 1:
            raise AssertionError("合法截断输出不应触发第 2 次调用")
        return LLMResponse(content=plan_conclude_json(), tokens=5, truncated=True)


async def test_truncated_but_valid_content_succeeds():
    """P2-15：截断但内容恰好通过校验 → 照常成功 attempts=1（不误伤）。"""
    backend = _TruncatedValidBackend()
    outcome = await _call(backend)
    assert backend.calls == 1
    assert outcome.model is not None and outcome.model.next_action == "conclude"
    assert outcome.attempts == 1
    assert outcome.error is None
    assert outcome.tokens == 5
