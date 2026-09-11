"""LLM 壳：节点调 LLM 的唯一入口 —— 强校验、重试 1 次、失败不抛异常。

``call_structured_llm`` 把 messages 交给后端 ``complete``，对返回 JSON 做 OutputModel
的 ``model_validate_json`` 强校验。失败分类（重试 1 次，两次均失败返回
``LLMCallOutcome(model=None, ...)``，永不抛异常）：

- schema 修正类（后端成功但校验失败）：追加修正提示（含第 1 次非法输出原文 + 校验
  错误），让第 2 次请求能看到自己上一版输出并修正；
- transport 类（``complete`` 抛异常：超时/网络/HTTP/未知 node 等）：没有可修正的
  输出，**不追加修正文案**，指数退避后按原 messages 重试；
- 截断（``truncated``，finish_reason=length）：内容可能不完整，校验失败时**不重试**
  （同 max_tokens 下大概率再截断），直接降级；截断但内容合法则照常成功。

后端可注入，默认 ``ScriptedLLMBackend``。本壳不做预算/降级检查（在节点入口），LLM
记账由节点按 ``outcome.attempts`` bump。token 口径 = ``usage.total_tokens``
（input+output 合计、含 provider 缓存命中）；**schema 校验失败的尝试也全额累计**，
transport 失败无响应不计。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

from pra.observability.tracing import Observation, get_tracer


class LLMBackendError(RuntimeError):
    """LLM 后端调用失败（超时/网络/未知 node 等）。

    供 ``pra.agent.scripted_llm`` import —— scripted 桩对未知 node 抛本异常。
    """


@dataclass
class LLMResponse:
    """``LLMBackend.complete`` 的返回：LLM 输出的 JSON 文本 + 本次 token 数。"""

    content: str  # LLM 返回的 JSON 文本（须能被对应 OutputModel 校验通过）
    tokens: int  # = 后端 usage.total_tokens（input+output 合计、含缓存命中；桩可给 0）
    truncated: bool = False  # finish_reason=="length"；按 transport 类处理（见模块 docstring）
    usage: dict | None = None  # 可选 token 拆分，形状 {"input","output","total"}；
                               # 键名对齐 Langfuse ``usage_details``。取不到的键不放
                               # （不填 0 冒充），整个拿不到 → None（**绝不伪造**）。


@runtime_checkable
class LLMBackend(Protocol):
    """可注入 LLM 后端接口 —— 真实 litellm 后端后续实现同一 Protocol。

    失败须抛 ``LLMBackendError``；消息/状态进出**只经方法参数**。
    """

    name: str  # 后端标识（如 "scripted" / "litellm-gpt-4o-mini"），仅审计/展示用

    async def complete(
        self, *, node: str, messages: list[dict], json_schema: dict
    ) -> LLMResponse:
        """按 node 分发生成结构化输出 JSON 文本。

        ``json_schema`` 为 OutputModel 的 JSON Schema（供真实后端约束输出；
        scripted 桩可忽略 —— messages 仅审计占位）。
        """
        ...


@dataclass
class LLMCallOutcome:
    """一次 ``call_structured_llm`` 的结果（成功/失败统一携带，供节点记账与降级）。"""

    model: Optional[Any]  # 校验通过的 OutputModel 实例；失败为 None
    attempts: int  # 实际尝试次数（1 或 2；截断等不可恢复失败为 1）
    tokens: int  # 已累计 token 数（口径同 usage.total_tokens；**schema 校验失败的
                 # 尝试也全额计入**、transport 失败无响应不计；全失败可能为 0）
    error: Optional[str]  # 最后一次失败原因（供节点写 failure.reason）；成功为 None
    usage: dict | None = None  # 可选 token 拆分：多次尝试**按键累加**（与 tokens 同
                               # 口径）；全部为 None → None。


# 模块级注入位点：_backend 非 None = 显式注入的后端；为 None = 用默认桩
# （惰性实例化 ScriptedLLMBackend，惰性 import 防循环依赖）。
_backend: Optional[LLMBackend] = None
_default_backend: Optional[LLMBackend] = None  # 默认桩实例缓存（只建一次）


def set_llm_backend(backend: Optional[LLMBackend]) -> None:
    """注入/替换 LLM 后端；``backend=None`` 恢复默认桩（ScriptedLLMBackend）。"""
    global _backend
    _backend = backend


def get_llm_backend() -> LLMBackend:
    """取当前生效后端：显式注入者优先；否则惰性实例化默认 scripted 桩并缓存。"""
    global _default_backend
    if _backend is not None:
        return _backend
    if _default_backend is None:
        # 惰性 import 防循环：scripted_llm 会反向 import 本模块的 LLMBackendError
        from pra.agent.scripted_llm import ScriptedLLMBackend

        _default_backend = ScriptedLLMBackend()
    return _default_backend


# transport 类失败重试的指数退避底数（秒）；保持小底数避免拖慢测试路径。
_TRANSPORT_BACKOFF_BASE_S = 0.05


async def _transport_backoff(failure_index: int = 1) -> None:
    """transport 类失败后、重试前的指数退避等待。

    第 ``failure_index`` 次失败后等待 ``base * 2**(failure_index-1)``。本壳每节点最多
    2 次尝试，实际只触发第 1 档。
    """
    await asyncio.sleep(_TRANSPORT_BACKOFF_BASE_S * (2 ** (failure_index - 1)))


def _correction_message(content: str, error: Exception) -> dict:
    """schema 修正提示：**回喂第 1 次非法输出原文** + 校验错误。

    只用于「后端成功返回但校验失败」；transport 失败不追加本文案。
    """
    return {
        "role": "user",
        "content": (
            "输出不满足 JSON Schema，请严格按 Schema 重新输出。以下是你上一次的"
            f"输出（供对照修正，不要复述它）：\n```\n{content}\n```\n"
            f"校验错误如下：\n{error}"
        ),
    }


# 观测（旁路）：只读、只写观测，绝不改变控制流。


def _elapsed_ms(t0: float) -> int:
    """距 ``t0`` 的墙钟耗时（毫秒）。"""
    return int((time.perf_counter() - t0) * 1000)


def _observe_update(obs: Observation, **kwargs: Any) -> None:
    """观测旁路：``update`` 绝不抛（适配层已吞异常，此处双保险）。"""
    try:
        obs.update(**kwargs)
    except Exception:  # noqa: BLE001 - 观测失败不得影响业务
        return


def _observe_record_error(obs: Observation, exc: BaseException) -> None:
    """观测旁路：``record_error`` 绝不抛。"""
    try:
        obs.record_error(exc)
    except Exception:  # noqa: BLE001 - 观测失败不得影响业务
        return


def _merge_usage(total: dict | None, usage: dict | None) -> dict | None:
    """按键累加 token usage（``input``/``output``/``total``）；皆空 → None。

    形状对齐 Langfuse ``usage_details``：取不到的键不放（不填 0 冒充），非 int 值
    跳过（防御畸形后端）。
    """
    if not isinstance(usage, dict):
        return total
    merged: dict = dict(total or {})
    for key, value in usage.items():
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        merged[key] = merged.get(key, 0) + value
    return merged or None


def _generation_update(resp: LLMResponse, *, attempt: int, latency_ms: int) -> dict:
    """一次成功响应的 generation 字段（``usage`` 为 None 时不传 usage_details）。"""
    kwargs: dict = {
        "output": resp.content,
        "metadata": {
            "latency_ms": latency_ms,
            "attempt": attempt,
            "truncated": resp.truncated,
        },
    }
    if resp.usage is not None:
        kwargs["usage_details"] = resp.usage
    return kwargs


async def call_structured_llm(
    *,
    OutputModel: Any,
    node: str,
    messages: list[dict],
) -> LLMCallOutcome:
    """强校验 LLM 调用壳：按失败类分类重试 + 降级返回（永不抛异常）。

    成功时首轮通过 ``attempts=1``，失败后按类处理重试 → ``attempts=2``；两次均失败
    → ``model=None``、``error`` 为最后一次失败文本、``tokens`` 为已累计。不做预算
    检查。失败分类/回喂/退避只出现在失败重试路径，scripted 桩恒返回可校验内容 →
    默认路径决策序列零变化。
    """
    # JSON Schema 只算一次；非 pydantic 模型按不可恢复失败返回
    try:
        json_schema: dict = OutputModel.model_json_schema()
    except Exception as exc:  # pragma: no cover - OutputModel 恒为 pydantic 模型
        return LLMCallOutcome(
            model=None,
            attempts=1,
            tokens=0,
            error=f"OutputModel 不是 pydantic 模型（无 model_json_schema）: {exc}",
        )

    backend = get_llm_backend()
    tracer = get_tracer()  # 进程级单例；无凭据 = NullTracer（全 no-op）
    work_messages: list[dict] = list(messages)  # 修正提示追加在工作副本，不污染调用方
    total_tokens = 0
    total_usage: dict | None = None  # 按键累加的 usage（全 None → None，不伪造 0）
    last_error: Optional[str] = None

    for attempt in (1, 2):
        # 每次真实 backend.complete() 记一条 generation：一次调用最多 2 次真实调用
        # （schema 修正重试 / transport 重试），必须逐次记录。模型名取后端自报。
        t0 = time.perf_counter()
        resp: LLMResponse | None = None
        try:
            with tracer.llm_generation(
                name=f"llm.{node}",
                model=getattr(backend, "name", "unknown"),
                input=work_messages,
                metadata={"node": node, "attempt": attempt},
            ) as obs:
                try:
                    resp = await backend.complete(
                        node=node, messages=work_messages, json_schema=json_schema
                    )
                except Exception as exc:  # 后端失败（LLMBackendError/网络/任何异常）
                    _observe_record_error(obs, exc)
                    raise
                _observe_update(
                    obs,
                    **_generation_update(resp, attempt=attempt, latency_ms=_elapsed_ms(t0)),
                )
        except Exception as exc:  # transport/后端失败
            if resp is None:
                last_error = str(exc)
                if attempt == 1:
                    # transport 类没有可「修正」的输出 → 不追加修正文案，退避后重试
                    await _transport_backoff(failure_index=attempt)
                continue
            # 响应已拿到 → 异常只来自观测层退出；忽略它，业务按「成功响应」继续
        total_tokens += resp.tokens
        total_usage = _merge_usage(total_usage, resp.usage)
        try:
            model = OutputModel.model_validate_json(resp.content)
        except Exception as exc:  # 校验失败（pydantic.ValidationError 等）
            last_error = str(exc)
            if attempt == 1:
                if resp.truncated:
                    # 截断（finish_reason=length）：重试大概率再截断，直接降级
                    return LLMCallOutcome(
                        model=None,
                        attempts=1,
                        tokens=total_tokens,
                        error=f"模型输出被截断（finish_reason=length）且校验失败: {exc}",
                        usage=total_usage,
                    )
                # schema 修正类：回喂第 1 次非法输出原文 + 校验错误
                work_messages.append(_correction_message(resp.content, exc))
            continue
        # 成功：attempts=1（首次通过）或 2（重试后通过）
        return LLMCallOutcome(
            model=model, attempts=attempt, tokens=total_tokens, error=None, usage=total_usage
        )

    # 两次尝试均失败 → 降级返回；节点据此置 degraded / 写 failure
    return LLMCallOutcome(
        model=None,
        attempts=2,
        tokens=total_tokens,
        error=last_error or "LLM 调用失败（无错误信息）",
        usage=total_usage,
    )


__all__ = [
    "LLMBackend",
    "LLMBackendError",
    "LLMCallOutcome",
    "LLMResponse",
    "call_structured_llm",
    "get_llm_backend",
    "set_llm_backend",
]
