"""LLM 壳 —— 节点调 LLM 的唯一入口（强校验 + 重试 1 次 + 失败不抛异常）。

职责（对齐 docs/04-graph-design.md §4 LLM 壳契约 / graph-mvp-contracts §4）：

- ``call_structured_llm``：把 ``messages`` 交给后端 ``LLMBackend.complete``，对返回
  JSON 做 **OutputModel pydantic 强校验**（``model_validate_json``）。第 1 次失败
  （ValidationError 或任何后端异常 —— 异常一律按 backend 失败处理）→ 向 messages
  追加修正提示（role=user，内容含错误信息与"请严格按 Schema 重新输出"）再试第 2 次；
  仍失败返回 ``LLMCallOutcome(model=None, attempts=2, tokens=已累计,
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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable


class LLMBackendError(RuntimeError):
    """LLM 后端调用失败（超时/网络/未知 node 等）。

    定义在本文件（G2），供 ``pra.agent.scripted_llm`` import —— scripted 桩对未知
    node 抛本异常，测试据此走降级路径。
    """


@dataclass
class LLMResponse:
    """``LLMBackend.complete`` 的返回：LLM 输出的 JSON 文本 + 本次 token 数。"""

    content: str  # LLM 返回的 JSON 文本（须能被对应 OutputModel 校验通过）
    tokens: int  # 本次调用 token 数（MVP 桩可给 0）


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
    attempts: int  # 本次实际尝试次数（1 或 2；入口短路 attempts=0 由节点自理）
    tokens: int  # 已累计 token 数（成功响应 token 之和；失败可能为 0）
    error: Optional[str]  # 最后一次失败原因（供节点写 failure.reason）；成功为 None


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


def _correction_message(error: Exception) -> dict:
    """修正提示（§4.1 定稿文案）：追加到 messages 末尾驱动后端第 2 次重试。"""
    return {
        "role": "user",
        "content": "输出不满足 JSON Schema，错误如下，请严格按 Schema 重新输出：\n"
        + str(error),
    }


async def call_structured_llm(
    *,
    OutputModel: Any,
    node: str,
    messages: list[dict],
) -> LLMCallOutcome:
    """强校验 LLM 调用壳：重试 1 次的输出校验 + 降级返回（永不抛异常）。

    - 成功：第 1 次校验通过 → ``attempts=1``；第 1 次失败 → 追加修正提示重试
      第 2 次 → ``attempts=2``（model 为校验通过的 OutputModel 实例）。
    - 失败（第 1 次与第 2 次均失败）：``model=None``，``error`` = 最后一次错误文本
      （供节点写 ``make_failure(reason=...)``），``tokens`` = 已累计 —— 不抛异常。
    - 不做预算检查（预算/降级短路在节点入口，§2.1 / §4.2）。

    ``OutputModel`` 为 pydantic 模型类（schemas.py 的 HypothesizeOutput / PlanOutput /
    ReevaluateOutput / DecisionProposal）；其 ``model_json_schema()`` 生成传给后端的
    ``json_schema``，``model_validate_json(content)`` 做强校验。
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
    work_messages: list[dict] = list(messages)  # 修正提示追加在工作副本，不污染调用方
    total_tokens = 0
    last_error: Optional[str] = None

    for attempt in (1, 2):
        try:
            resp = await backend.complete(
                node=node, messages=work_messages, json_schema=json_schema
            )
        except Exception as exc:  # backend 层失败（LLMBackendError/网络/任何异常）
            last_error = str(exc)
            if attempt == 1:
                work_messages.append(_correction_message(exc))  # 契约：异常也走修正重试
            continue
        total_tokens += resp.tokens
        try:
            model = OutputModel.model_validate_json(resp.content)
        except Exception as exc:  # 校验失败（pydantic.ValidationError 等）
            last_error = str(exc)
            if attempt == 1:
                work_messages.append(_correction_message(exc))
            continue
        # 成功：第 1 次校验通过 attempts=1；重试后通过 attempts=2
        return LLMCallOutcome(model=model, attempts=attempt, tokens=total_tokens, error=None)

    # 两次尝试均失败 → 降级返回（不抛异常）；节点据此置 degraded / 写 failure
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
    "get_llm_backend",
    "set_llm_backend",
]
