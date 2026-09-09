"""LLM 壳 —— 节点调 LLM 的唯一入口（强校验 + 重试 1 次 + 失败不抛异常）。

职责（对齐 docs/04-graph-design.md §4 LLM 壳契约 / graph-mvp-contracts §4）：

- ``call_structured_llm``：把 ``messages`` 交给后端 ``LLMBackend.complete``，对返回
  JSON 做 **OutputModel pydantic 强校验**（``model_validate_json``）。失败按类分类
  （P2-15 —— 避免"schema 修正"文案/重试被 transport 失败错用）：
  - **schema 修正类**（后端成功返回但校验失败）→ 向 messages 追加修正提示
    （role=user；**内容含第 1 次非法输出原文 + 校验错误**，让第 2 次请求能"看到"
    自己上一版输出并修正，而非只给错误文本）再试第 2 次；
  - **transport 类**（后端 ``complete`` 抛异常：超时 / 网络 / HTTP / 连接 / 未知
    node 等）→ 没有可"修正"的输出：**不追加 schema 修正文案**，指数退避
    （``_transport_backoff``，轻量实现）后按原 messages 重试第 2 次；
  - **截断（finish_reason=length，``LLMResponse.truncated``）**→ 内容可能不完整，
    按 transport 类处理：校验失败时**不重试**（同 max_tokens 下大概率再截断，
    attempts=1 直接降级 —— 不白烧一次调用）；截断但内容恰好合法则照常成功。
  仍失败返回 ``LLMCallOutcome(model=None, attempts=实际尝试次数, tokens=已累计,
  error=最后一次错误文本)``。**永不抛异常** —— 校验失败如何降级由节点按 §2.1 决定
  （节点返回自己的降级结果并置 ``degraded=True``）。
- 后端可注入（**MVP 注入位点**）：默认后端 = ``pra.agent.scripted_llm.
  ScriptedLLMBackend`` 实例，模块级 ``_backend`` 变量承载注入 —— ``set_llm_backend``
  供 ``build_agent_graph(llm=...)`` 与测试注入；``set_llm_backend(None)`` 恢复默认。
  真实 litellm 后端后续接入：实现同一 ``LLMBackend`` Protocol 后在此接线即可，
  节点/测试侧无感。
- 代理语义：**本模块不做预算/降级检查**（预算与 degraded 短路在节点入口，§2.1/§4.2）；
  LLM 记账由节点按 ``outcome.attempts`` bump（``bump_llm_usage(budget,
  llm_calls=outcome.attempts, tokens=outcome.tokens)``）。

``node`` 取值 ∈ {"hypothesize","plan","reevaluate","decide"}；本壳不校验该取值
（交给后端：默认 scripted 后端对未知 node 抛 ``LLMBackendError``，供降级路径测试）。

注：修正提示追加在工作副本上（不污染调用方传入的 messages 列表 —— 节点每次
``_build_messages(state)`` 新造，两个语义等价）。

token 口径（P2-16 注记）：``LLMResponse.tokens`` / ``LLMCallOutcome.tokens`` =
后端 ``usage.total_tokens``（input+output 合计、含 provider 缓存命中 token）；
**schema 校验失败的尝试也全额累计**；transport 失败无响应不计。节点按此 bump
``budget.tokens`` —— 真实跑分的 token 维度是**混合口径**，非纯输出生成量。
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

from pra.observability.tracing import Observation, get_tracer


class LLMBackendError(RuntimeError):
    """LLM 后端调用失败（超时/网络/未知 node 等）。

    定义在本文件（G2），供 ``pra.agent.scripted_llm`` import —— scripted 桩对未知
    node 抛本异常，测试据此走降级路径。
    """


@dataclass
class LLMResponse:
    """``LLMBackend.complete`` 的返回：LLM 输出的 JSON 文本 + 本次 token 数。"""

    content: str  # LLM 返回的 JSON 文本（须能被对应 OutputModel 校验通过）
    tokens: int  # 本次 token 数 = 后端 usage.total_tokens 口径（input+output 合计、
                 # 含 provider 缓存命中 token —— P2-16 注记；MVP 桩可给 0）
    truncated: bool = False  # finish_reason=="length"（输出被截断、内容可能不完整，
                             # P2-15：llm_shell 按 transport 类处理，见模块 docstring）
    usage: dict | None = None  # 可选 token 拆分（S3 观测）：形状
                               # ``{"input": int, "output": int, "total": int}``
                               # —— 键名对齐 Langfuse ``usage_details``；取不到的键
                               # **不放**（不填 0 冒充）、整个 usage 拿不到 → None；
                               # 确定性桩无真实 token → None（**绝不伪造**）。


@runtime_checkable
class LLMBackend(Protocol):
    """可注入 LLM 后端接口 —— 真实 litellm 后端后续实现同一 Protocol。

    - ``node`` ∈ {"hypothesize","plan","reevaluate","decide"}；
    - 失败（超时/网络/内容异常等）须抛 ``LLMBackendError``（本模块定义）；
    - 契约要求消息/状态进出 **只经方法参数**（后端不直接读 AgentState）。
    """

    name: str  # 后端标识（如 "scripted" / "litellm-gpt-4o-mini"），仅审计/展示用

    async def complete(
        self, *, node: str, messages: list[dict], json_schema: dict
    ) -> LLMResponse:
        """按 node 分发生成结构化输出 JSON 文本。

        ``json_schema`` 为 OutputModel 的 JSON Schema（``model_json_schema()``，
        供真实后端约束输出；MVP scripted 桩可忽略 —— messages 仅审计占位）。
        """
        ...


@dataclass
class LLMCallOutcome:
    """一次 ``call_structured_llm`` 的结果（成功/失败统一携带，供节点记账与降级）。"""

    model: Optional[Any]  # 校验通过的 OutputModel 实例；失败为 None
    attempts: int  # 本次实际尝试次数（1 或 2；截断/不可恢复失败可为 1；
                   # 入口短路 attempts=0 由节点自理）
    tokens: int  # 已累计 token 数（口径 = usage.total_tokens：input+output 合计、
                 # 含缓存命中；**schema 校验失败的尝试也全额计入**、transport 失败
                 # 无响应不计 —— P2-16 注记；成功响应之和，全失败可能为 0）
    error: Optional[str]  # 最后一次失败原因（供节点写 failure.reason）；成功为 None
    usage: dict | None = None  # 可选 token 拆分（S3 观测）：多次尝试**按键累加**
                               # （与 tokens 同口径：schema 校验失败的尝试也计入、
                               # transport 失败无响应不计）；全部为 None → None。


# 模块级注入位点（MVP）：
# - _backend 非 None = 显式注入的后端（测试 / build_agent_graph(llm=...)）；
# - _backend 为 None = 用默认桩（get_llm_backend 惰性实例化 ScriptedLLMBackend，
#   惰性 import 防与 scripted_llm / graph 装配产生循环依赖）。
_backend: Optional[LLMBackend] = None
_default_backend: Optional[LLMBackend] = None  # 默认桩实例缓存（只建一次）


def set_llm_backend(backend: Optional[LLMBackend]) -> None:
    """注入/替换 LLM 后端；``backend=None`` 恢复默认桩（ScriptedLLMBackend）。

    MVP 注入位点：``build_agent_graph(llm=...)`` 与测试经此接线；真实 litellm
    后端后续实现 ``LLMBackend`` Protocol 后同样经此注入，节点/测试侧无感。
    """
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


# transport 类失败重试的指数退避底数（秒）—— P2-15 轻量实现：真实生产可调大/换抖动
# 退避表；保持小底数避免确定性/测试路径被拖慢。
_TRANSPORT_BACKOFF_BASE_S = 0.05


async def _transport_backoff(failure_index: int = 1) -> None:
    """transport 类失败后、重试前的指数退避等待（P2-15 轻量实现）。

    第 ``failure_index`` 次失败后等待 ``base * 2**(failure_index-1)``（第 1 次失败后
    base、第 2 次 base*2 …）。本壳每节点最多 2 次尝试（attempts∈{1,2}），实际只触发
    第 1 档；按指数结构留档，扩档无需改调用点。TODO（实现位）：真实生产可换抖动 +
    更长退避表，并把 schedule 做成可注入以便测试快进。
    """
    await asyncio.sleep(_TRANSPORT_BACKOFF_BASE_S * (2 ** (failure_index - 1)))


def _correction_message(content: str, error: Exception) -> dict:
    """schema 修正提示（P2-15）：**回喂第 1 次非法输出原文** + 校验错误。

    只用于"后端成功返回但校验失败"（schema 修正类）；transport 失败没有可修正的
    输出、不追加本文案（分类见 ``call_structured_llm``）。含原文后，真实模型第 2 次
    请求能"看到"自己上一版输出并针对性修正 —— 旧版只带 ValidationError 文本时，
    它常常无法修正"它没看到"的输出。
    """
    return {
        "role": "user",
        "content": (
            "输出不满足 JSON Schema，请严格按 Schema 重新输出。以下是你上一次的"
            f"输出（供对照修正，不要复述它）：\n```\n{content}\n```\n"
            f"校验错误如下：\n{error}"
        ),
    }


# --------------------------------------------------------------------------------------
# 观测（S3，旁路）—— 只读、只写观测，绝不改变控制流（docs/09 §3）
# --------------------------------------------------------------------------------------


def _elapsed_ms(t0: float) -> int:
    """距 ``t0`` 的墙钟耗时（毫秒，int）—— 本壳原先**没有任何 latency 记录**。"""
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

    形状约定对齐 Langfuse ``usage_details``（S3）：取不到的键不放（不填 0 冒充），
    非 int 值跳过（防御畸形后端），两者皆无 → None。
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
    """一次成功响应的 generation 字段（``usage`` 为 None 时**不传** usage_details 键）。"""
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

    - 成功：第 1 次校验通过 → ``attempts=1``；第 1 次失败 → 按类处理后重试第 2 次
      → ``attempts=2``（model 为校验通过的 OutputModel 实例）。
    - **schema 修正类**（后端返回成功但校验失败）：追加含**第 1 次非法输出原文 +
      校验错误**的修正提示后重试（P2-15：第 2 次请求能看到自己上一版输出）；
    - **transport 类**（后端 ``complete`` 抛异常：超时/网络/HTTP/未知 node 等）：
      无输出可修正 → 不追加 schema 修正文案，指数退避后按原 messages 重试 1 次；
    - **截断**（``LLMResponse.truncated``，finish_reason=length）：内容可能不完整，
      校验失败时按 transport 类处理 —— **不重试**，attempts=1 直接降级（省一次
      大概率无效的调用）；截断但内容恰好合法则照常成功。
    - 失败（第 1 次与第 2 次均失败）：``model=None``，``error`` = 最后一次失败文本
      （供节点写 ``make_failure(reason=...)``），``tokens`` = 已累计 —— 不抛异常。
    - 不做预算检查（预算/降级短路在节点入口，§2.1 / §4.2）。

    ``OutputModel`` 为 pydantic 模型类（schemas.py 的 HypothesizeOutput / PlanOutput /
    ReevaluateOutput / DecisionProposal）；其 ``model_json_schema()`` 生成传给后端的
    ``json_schema``，``model_validate_json(content)`` 做强校验。

    确定性约束（P2-15 守护）：本壳的失败分类/回喂/退避只出现在**失败重试路径**；
    确定性 scripted 桩（EvalScriptedLLMBackend / ScriptedLLMBackend）恒返回可校验
    通过内容、不抛异常 → 默认（无 llm 注入）路径决策序列零变化。
    """
    # OutputModel 的 JSON Schema（只算一次；非 pydantic 模型按不可恢复失败返回）
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
    tracer = get_tracer()  # 进程级单例；无凭据 = NullTracer（全 no-op，零开销）
    work_messages: list[dict] = list(messages)  # 修正提示追加在工作副本，不污染调用方
    total_tokens = 0
    total_usage: dict | None = None  # 按键累加的 usage（全 None → None，不伪造 0）
    last_error: Optional[str] = None

    for attempt in (1, 2):
        # 观测（S3）：**每次真实 backend.complete() 一条 generation** —— 埋点在内层
        # 循环而非外壳：一次 call_structured_llm 最多 2 次真实调用（schema 修正重试 /
        # transport 重试），Langfuse 必须逐次记录。模型名取后端自报（不伪造）。
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
                    # 观测：本次真实调用失败（随后照旧走既有 transport 重试逻辑）
                    _observe_record_error(obs, exc)
                    raise
                # 观测：成功调用 —— output + usage + latency/attempt/truncated
                _observe_update(
                    obs,
                    **_generation_update(resp, attempt=attempt, latency_ms=_elapsed_ms(t0)),
                )
        except Exception as exc:  # transport/后端失败（行为与埋点前完全一致）
            if resp is None:
                last_error = str(exc)
                if attempt == 1:
                    # transport 类：没有可"修正"的输出 —— 不追加 schema 修正文案
                    # （旧版对超时/HTTP 也按"schema 修正"重试是误导）；退避后原样重试。
                    await _transport_backoff(failure_index=attempt)
                continue
            # 响应已拿到 → 异常只可能来自观测层退出（适配层契约：绝不抛）。
            # 忽略它、业务继续按"成功响应"走 —— 埋点不得改变控制流。
        total_tokens += resp.tokens
        total_usage = _merge_usage(total_usage, resp.usage)
        try:
            model = OutputModel.model_validate_json(resp.content)
        except Exception as exc:  # 校验失败（pydantic.ValidationError 等）
            last_error = str(exc)
            if attempt == 1:
                if resp.truncated:
                    # 截断（finish_reason=length）：输出未完成，按 transport 类处理。
                    # 同 max_tokens 下重试大概率再截断 → 不烧第 2 次调用，直接降级。
                    return LLMCallOutcome(
                        model=None,
                        attempts=1,
                        tokens=total_tokens,
                        error=f"模型输出被截断（finish_reason=length）且校验失败: {exc}",
                        usage=total_usage,
                    )
                # schema 修正类：回喂第 1 次非法输出原文 + 校验错误（P2-15）
                work_messages.append(_correction_message(resp.content, exc))
            continue
        # 成功：第 1 次校验通过 attempts=1；重试后通过 attempts=2
        return LLMCallOutcome(
            model=model, attempts=attempt, tokens=total_tokens, error=None, usage=total_usage
        )

    # 两次尝试均失败 → 降级返回（不抛异常）；节点据此置 degraded / 写 failure
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
