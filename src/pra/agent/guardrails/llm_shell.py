"""LLM 壳：节点调 LLM 的入口，做 JSON 强校验、失败重试 1 次并按降级返回。

两次尝试均失败时返回 ``LLMCallOutcome(model=None, ...)``，不抛 LLM 异常。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable


class LLMBackendError(RuntimeError):
    """LLM 后端调用失败（超时/网络/未知 node 等）。"""


@dataclass
class LLMResponse:
    """``LLMBackend.complete`` 的返回：LLM 输出的 JSON 文本 + 本次 token 数。"""

    content: str
    tokens: int
    truncated: bool = False  # finish_reason=="length"（校验失败不重试）
    usage: dict | None = None  # 可选 token 拆分，形状 {"input","output","total"}；取不到为 None


@runtime_checkable
class LLMBackend(Protocol):
    """可注入 LLM 后端接口；失败须抛 ``LLMBackendError``。"""

    name: str  # 后端标识（如 "litellm-gpt-4o-mini"）

    async def complete(
        self,
        *,
        node: str,
        state: dict,
        json_schema: dict,
        feedback: list[str] | None = None,
    ) -> LLMResponse:
        """按 node 分发生成结构化输出 JSON 文本；``feedback`` 为上一轮校验失败的修正提示（首次尝试为 None）。"""
        ...


@dataclass
class LLMCallOutcome:
    """一次 ``call_structured_llm`` 的结果（成功 / 失败统一携带）。"""

    model: Optional[Any]  # 校验通过的 OutputModel 实例；失败为 None
    attempts: int  # 实际尝试次数（1 或 2；截断等不可恢复失败为 1）
    tokens: int  # 已累计 token 数；全失败可能为 0
    error: Optional[str]  # 最后一次失败原因；成功为 None


# transport 类失败重试的指数退避底数（秒）
_TRANSPORT_BACKOFF_BASE_S = 0.05


async def _transport_backoff(failure_index: int = 1) -> None:
    """transport 类失败后、重试前的指数退避等待：第 ``failure_index`` 次失败后等待 ``base * 2**(failure_index-1)``。"""
    await asyncio.sleep(_TRANSPORT_BACKOFF_BASE_S * (2 ** (failure_index - 1)))


def _correction_message(content: str, error: Exception) -> str:
    """schema 修正提示文本：回喂上次非法输出原文 + 校验错误。"""
    return (
        "输出不满足 JSON Schema，请严格按 Schema 重新输出。以下是你上一次的"
        f"输出（供对照修正，不要复述它）：\n```\n{content}\n```\n"
        f"校验错误如下：\n{error}"
    )


async def call_structured_llm(
    *,
    OutputModel: Any,
    node: str,
    state: dict,
    llm: LLMBackend,
) -> LLMCallOutcome:
    """强校验 LLM 调用壳：校验失败重试 1 次并按降级返回，不抛 LLM 异常。

    成功返回校验通过的 ``model``；两次均失败返回 ``model=None``、``error`` 为最后一次
    失败文本、``tokens`` 为已累计。

    :param llm: 必须显式注入；``None`` 立刻抛 ``TypeError``。
    """
    if llm is None:
        raise TypeError(
            f"call_structured_llm（node={node!r}）需要显式注入 LLM 后端（llm=None）—— "
            "请在装配处把 LLMBackend 注入 build_agent_graph(llm=...)"
        )

    try:
        json_schema: dict = OutputModel.model_json_schema()
    except Exception as exc:  # pragma: no cover - OutputModel 恒为 pydantic 模型
        return LLMCallOutcome(
            model=None,
            attempts=1,
            tokens=0,
            error=f"OutputModel 不是 pydantic 模型（无 model_json_schema）: {exc}",
        )

    backend = llm
    feedback: list[str] = []  # schema 修正提示，重试时以 feedback= 回喂
    total_tokens = 0
    last_error: Optional[str] = None

    for attempt in (1, 2):
        resp: LLMResponse | None = None
        try:
            resp = await backend.complete(
                node=node,
                state=state,
                json_schema=json_schema,
                feedback=feedback or None,
            )
        except Exception as exc:  # transport / 后端失败
            last_error = str(exc)
            if attempt == 1:
                await _transport_backoff(failure_index=attempt)
            continue
        total_tokens += resp.tokens
        try:
            model = OutputModel.model_validate_json(resp.content)
        except Exception as exc:  # 校验失败
            last_error = str(exc)
            if attempt == 1:
                if resp.truncated:
                    # 截断（finish_reason=length）：直接降级
                    return LLMCallOutcome(
                        model=None,
                        attempts=1,
                        tokens=total_tokens,
                        error=f"模型输出被截断（finish_reason=length）且校验失败: {exc}",
                    )
                # schema 修正类：收集非法输出原文 + 校验错误，重试时回喂
                feedback.append(_correction_message(resp.content, exc))
            continue
        return LLMCallOutcome(
            model=model, attempts=attempt, tokens=total_tokens, error=None
        )

    return LLMCallOutcome(
        model=None,
        attempts=2,
        tokens=total_tokens,
        error=last_error or "LLM 调用失败（无错误信息）",
    )


__all__ = [
    "LLMBackend",
    "LLMBackendError",
    "LLMCallOutcome",
    "LLMResponse",
    "call_structured_llm",
]
