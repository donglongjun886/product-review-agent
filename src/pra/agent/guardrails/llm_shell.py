"""LLM 壳：节点调 LLM 的唯一入口 —— 强校验、重试 1 次、失败不抛异常。

``call_structured_llm`` 把节点构造的结构化 ``state`` 直传给后端 ``complete``，对返回 JSON
做 OutputModel 的 ``model_validate_json`` 强校验。失败分类（重试 1 次，两次均失败返回
``LLMCallOutcome(model=None, ...)``，永不抛异常）：

- schema 修正类（后端成功但校验失败）：把修正提示（含第 1 次非法输出原文 + 校验错误）
  收集进 ``feedback``，第 2 次请求以 ``feedback=`` 回喂，让模型能看到自己上一版输出并修正；
- transport 类（``complete`` 抛异常：超时/网络/HTTP/未知 node 等）：没有可修正的
  输出，**不产生 feedback**，指数退避后按原 state 重试；
- 截断（``truncated``，finish_reason=length）：内容可能不完整，校验失败时**不重试**
  （同 max_tokens 下大概率再截断），直接降级；截断但内容合法则照常成功。

后端**由调用方显式注入**（每个节点从图装配处拿到 ``llm=`` 后透传给本壳）—— 本壳不持
任何进程级默认后端，也不会在缺后端时回落到 scripted 桩。本壳不做预算/降级检查（在节点
入口），LLM 记账由节点按 ``outcome.attempts`` bump。token 口径 = ``usage.total_tokens``
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

    供测试侧 ``tests/stub_llm.ScriptedLLMBackend`` import —— 该桩对未知 node 抛本异常。
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
    """可注入 LLM 后端接口 —— ``ScriptedLLMBackend`` / ``LiteLLMBackend`` 实现同一 Protocol。

    失败须抛 ``LLMBackendError``；状态进出**只经方法参数**。
    """

    name: str  # 后端标识（如 "scripted" / "litellm-gpt-4o-mini"），仅审计/展示用

    async def complete(
        self,
        *,
        node: str,
        state: dict,
        json_schema: dict,
        feedback: list[str] | None = None,
    ) -> LLMResponse:
        """按 node 分发生成结构化输出 JSON 文本。

        :param state: 节点构造并直传的、JSON 可序列化的结构化 state 子集；
        :param json_schema: OutputModel 的 JSON Schema（供真实后端约束输出；scripted 桩忽略）；
        :param feedback: 上一轮 schema 校验失败的修正提示文本列表（首次尝试为 None）。
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


# transport 类失败重试的指数退避底数（秒）；保持小底数避免拖慢测试路径。
_TRANSPORT_BACKOFF_BASE_S = 0.05


async def _transport_backoff(failure_index: int = 1) -> None:
    """transport 类失败后、重试前的指数退避等待。

    第 ``failure_index`` 次失败后等待 ``base * 2**(failure_index-1)``。本壳每节点最多
    2 次尝试，实际只触发第 1 档。
    """
    await asyncio.sleep(_TRANSPORT_BACKOFF_BASE_S * (2 ** (failure_index - 1)))


def _correction_message(content: str, error: Exception) -> str:
    """schema 修正提示文本：**回喂第 1 次非法输出原文** + 校验错误。

    只用于「后端成功返回但校验失败」；transport 失败不产生本文案。
    """
    return (
        "输出不满足 JSON Schema，请严格按 Schema 重新输出。以下是你上一次的"
        f"输出（供对照修正，不要复述它）：\n```\n{content}\n```\n"
        f"校验错误如下：\n{error}"
    )


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
    state: dict,
    llm: LLMBackend,
) -> LLMCallOutcome:
    """强校验 LLM 调用壳：按失败类分类重试 + 降级返回（永不抛 LLM 异常）。

    成功时首轮通过 ``attempts=1``，失败后按类处理重试 → ``attempts=2``；两次均失败
    → ``model=None``、``error`` 为最后一次失败文本、``tokens`` 为已累计。不做预算
    检查。失败分类/回喂/退避只出现在失败重试路径，scripted 桩恒返回可校验内容 →
    默认路径决策序列零变化。

    :param state: 节点构造的 JSON 可序列化 state 子集 —— 直传后端，不再经 ``__STATE__``
        文本协议；同一 state + 空 feedback 渲染出的 prompt 与改造前等价。
    :param llm: 本次调用使用的 ``LLMBackend`` —— **必须显式注入**；``None`` 是装配缺陷，
        立刻抛 ``TypeError``（不回落任何默认桩、不降级成 HUMAN_REVIEW 掩盖问题）。
    """
    if llm is None:
        raise TypeError(
            f"call_structured_llm（node={node!r}）需要显式注入 LLM 后端（llm=None）—— "
            "不再回落 scripted 桩；请在装配处把 LLMBackend 注入 build_agent_graph(llm=...)"
        )

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

    backend = llm
    tracer = get_tracer()  # 进程级单例；无凭据 = NullTracer（全 no-op）
    feedback: list[str] = []  # schema 修正提示（transport 失败不产生）；重试时以 feedback= 回喂
    total_tokens = 0
    last_error: Optional[str] = None

    for attempt in (1, 2):
        # 每次真实 backend.complete() 记一条 generation：一次调用最多 2 次真实调用
        # （schema 修正重试 / transport 重试），必须逐次记录。模型名取后端自报；
        # input 记结构化 state（反馈文本不进 input，见 metadata.attempt 区分尝试）。
        t0 = time.perf_counter()
        resp: LLMResponse | None = None
        try:
            with tracer.llm_generation(
                name=f"llm.{node}",
                model=getattr(backend, "name", "unknown"),
                input=state,
                metadata={"node": node, "attempt": attempt},
            ) as obs:
                try:
                    resp = await backend.complete(
                        node=node,
                        state=state,
                        json_schema=json_schema,
                        feedback=feedback or None,
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
                    # transport 类没有可「修正」的输出 → 不产生 feedback，退避后重试
                    await _transport_backoff(failure_index=attempt)
                continue
            # 响应已拿到 → 异常只来自观测层退出；忽略它，业务按「成功响应」继续
        total_tokens += resp.tokens
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
                    )
                # schema 修正类：收集第 1 次非法输出原文 + 校验错误，重试时回喂
                feedback.append(_correction_message(resp.content, exc))
            continue
        # 成功：attempts=1（首次通过）或 2（重试后通过）
        return LLMCallOutcome(
            model=model, attempts=attempt, tokens=total_tokens, error=None
        )

    # 两次尝试均失败 → 降级返回；节点据此置 degraded / 写 failure
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
