"""真实 litellm 后端 ``LiteLLMBackend``（实现 ``LLMBackend`` Protocol，真实 LLM 评测用）。"""

from __future__ import annotations

import os
from typing import Any, Literal

from pra.agent.guardrails.llm_shell import (
    LLMBackend,
    LLMBackendError,
    LLMResponse,
)
from pra.agent.llm_prompts import (
    SYSTEM_PROMPTS,
    build_system_prompt,
    build_turn_message,
    build_user_prompt,
)

__all__ = ["LiteLLMBackend"]

_API_KEY_ENV = "DEEPSEEK_API_KEY"


def _int_or_none(value: Any) -> int | None:
    """``value`` 转 int；缺失 / None / 非法 → ``None``。"""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _cache_token_split(usage: Any) -> tuple[int, int] | None:
    """从 usage 提取 (cache_hit, cache_miss)；提供方未给缓存字段 → None（显式缺失）。

    优先 DeepSeek 原生 ``prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens``；
    回退 OpenAI 风格 ``prompt_tokens_details.cached_tokens``（miss 由 input - hit 推算）。
    """
    hit = _int_or_none(getattr(usage, "prompt_cache_hit_tokens", None))
    miss = _int_or_none(getattr(usage, "prompt_cache_miss_tokens", None))
    if hit is not None or miss is not None:
        return (hit or 0, miss or 0)
    details = getattr(usage, "prompt_tokens_details", None)
    cached = (
        _int_or_none(getattr(details, "cached_tokens", None))
        if details is not None
        else None
    )
    if cached is None:
        return None
    prompt_tokens = _int_or_none(getattr(usage, "prompt_tokens", None))
    return (cached, (prompt_tokens - cached) if prompt_tokens is not None else 0)


class LiteLLMBackend(LLMBackend):
    """真实 litellm 后端：渲染结构化 state 为 prompt 后调用 API。"""

    name: str = "litellm"  # Protocol 属性（``__init__`` 覆写）

    def __init__(
        self,
        *,
        model: str = "deepseek/deepseek-flash",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout_s: float = 60.0,
        max_tokens: int | None = None,
        tools: list | None = None,
        thinking: Literal["enabled", "disabled"] | None = None,
        reasoning_effort: Literal["none", "low", "high", "max"] | None = None,
    ) -> None:
        """构造真实 LLM 后端（只固化参数，不发起网络调用）。

        ``api_key`` None → 读 ``DEEPSEEK_API_KEY``；``base_url`` 非 None → 以 ``api_base`` 传给 litellm。
        ``tools`` → 工具目录 ``{name, description, args_schema}``；``thinking`` / ``reasoning_effort``
        None → 不传，用网关默认。
        """
        self.model = model
        self.name = f"litellm-{model}"
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self._api_key: str | None = api_key or os.environ.get(_API_KEY_ENV) or None
        self._tool_catalog: list[dict] = self._extract_tool_catalog(tools)
        # 实例级缓存命中累计（纯观测，不进 state/预算语义）；None = 提供方从未给过缓存字段
        self._cache_hit_tokens: int | None = None
        self._cache_miss_tokens: int | None = None

    @property
    def total_cache_hit_tokens(self) -> int | None:
        """累计缓存命中 token；提供方从未上报缓存字段 → None（缺失即缺失）。"""
        return self._cache_hit_tokens

    @property
    def total_cache_miss_tokens(self) -> int | None:
        """累计缓存未命中 token；提供方从未上报缓存字段 → None。"""
        return self._cache_miss_tokens

    @staticmethod
    def _extract_tool_catalog(tools: list | None) -> list[dict]:
        """``Tool`` 列表 → 工具目录：元素 ``{"name", "description", "args_schema"}``；
        单个工具提取失败则跳过，空输入 → 空目录。
        """
        catalog: list[dict] = []
        for tool in tools or []:
            name = getattr(tool, "name", None)
            if not name:
                continue
            args_model = getattr(tool, "args_model", None)
            try:
                args_schema = (
                    args_model.model_json_schema() if args_model is not None else {}
                )
            except Exception:  # noqa: BLE001 - 单个工具提取失败则跳过
                args_schema = {}
            catalog.append(
                {
                    "name": str(name),
                    "description": str(getattr(tool, "description", "") or ""),
                    "args_schema": args_schema,
                }
            )
        return catalog

    @staticmethod
    def _validated_history(state: dict) -> list[dict]:
        """``state["llm_messages"]`` 原样返回；形状非法 → ``LLMBackendError``（显式失败）。

        历史由本后端自行产出，形状恒合法；发现坏条目（非 list / 条目非 dict /
        content 缺失或空）即 state 损坏或代码 bug，显式失败走节点降级，不静默截断。
        """
        history = state.get("llm_messages")
        if history is None:
            return []
        if not isinstance(history, list):
            raise LLMBackendError(
                f"state[\"llm_messages\"] 应为 list，实为 {type(history).__name__}"
            )
        for m in history:
            if not (
                isinstance(m, dict)
                and isinstance(m.get("content"), str)
                and m["content"]
            ):
                raise LLMBackendError(f"llm_messages 含非法条目（非 dict 或 content 为空）：{m!r}")
        return history

    def _render(
        self,
        *,
        node: str,
        state: dict,
        json_schema: dict,
        feedback: list[str] | None = None,
    ) -> tuple[list[dict], list[dict]]:
        """组装本轮消息：返回 ``(发送消息, 本轮新增待持久化消息)``。

        发送消息 = [system, *对话历史, *本轮 user 消息]；agent 节点走增量轮次消息
        （``build_turn_message``），修正反馈以独立 user 消息追加（重试时轮次消息已在
        历史里，只把反馈计入新增）；single_call 一次性全量渲染（``build_user_prompt``），
        忽略对话历史、修正反馈内嵌同一条 user。
        """
        system = {"role": "system", "content": build_system_prompt(node)}
        if node == "single_call":
            user = {
                "role": "user",
                "content": build_user_prompt(
                    node=node, state=state, json_schema=json_schema, feedbacks=feedback
                ),
            }
            return [system, user], [user]
        turn_user = {
            "role": "user",
            "content": build_turn_message(
                node=node,
                state=state,
                json_schema=json_schema,
                tool_catalog=self._tool_catalog,
            ),
        }
        sent: list[dict] = [system, *self._validated_history(state), turn_user]
        new_messages: list[dict] = [turn_user]
        if feedback:
            correction = {"role": "user", "content": "\n\n".join(feedback)}
            sent.append(correction)
            # 重试：轮次消息已在历史（首试已上报），本轮新增只计修正反馈
            new_messages = [correction]
        return sent, new_messages

    async def complete(
        self,
        *,
        node: str,
        state: dict,
        json_schema: dict,
        feedback: list[str] | None = None,
    ) -> LLMResponse:
        """按 ``node`` 渲染 prompt 并调用 litellm。

        返回 ``LLMResponse``：``content`` 为模型原样文本，``truncated`` 标记
        ``finish_reason == "length"``；``node`` 词表外或无 key / 调用失败 → ``LLMBackendError``。
        """
        if node not in SYSTEM_PROMPTS:
            raise LLMBackendError(f"未知 node: {node!r}，应为 {sorted(SYSTEM_PROMPTS)}")
        if not self._api_key:
            raise LLMBackendError(
                f"未配置 DeepSeek API key：请在构造 LiteLLMBackend(api_key=...) 时传入，"
                f"或设置环境变量 {_API_KEY_ENV} 后重试（当前 model={self.model}）。"
            )
        sent_messages, new_messages = self._render(
            node=node, state=state, json_schema=json_schema, feedback=feedback
        )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": sent_messages,
            "temperature": self.temperature,
            "timeout": self.timeout_s,
            "response_format": {"type": "json_object"},
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self.base_url:
            kwargs["api_base"] = self.base_url
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.thinking is not None:
            kwargs["extra_body"] = {"thinking": {"type": self.thinking}}
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        try:
            os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
            import litellm

            resp = await litellm.acompletion(**kwargs)
        except LLMBackendError:
            raise
        except Exception as exc:
            raise LLMBackendError(
                f"litellm 调用失败（node={node}, model={self.model}）：{exc}"
            ) from exc
        content = ""
        truncated = False
        try:
            choice = resp.choices[0]
            message = choice.message
            content = message.content or ""
            truncated = str(getattr(choice, "finish_reason", "") or "") == "length"
        except Exception:  # noqa: BLE001 - 响应结构防御性解析
            content = ""
        tokens = 0
        usage_details: dict[str, int] | None = None
        usage = getattr(resp, "usage", None)
        if usage is not None:
            total_tokens = _int_or_none(getattr(usage, "total_tokens", None))
            tokens = total_tokens or 0
            parts: dict[str, int] = {}
            prompt_tokens = _int_or_none(getattr(usage, "prompt_tokens", None))
            completion_tokens = _int_or_none(getattr(usage, "completion_tokens", None))
            if prompt_tokens is not None:
                parts["input"] = prompt_tokens
            if completion_tokens is not None:
                parts["output"] = completion_tokens
            if total_tokens is not None:
                parts["total"] = total_tokens
            cache_split = _cache_token_split(usage)
            if cache_split is not None:
                hit, miss = cache_split
                parts["cache_hit"] = hit
                parts["cache_miss"] = miss
                self._cache_hit_tokens = (self._cache_hit_tokens or 0) + hit
                self._cache_miss_tokens = (self._cache_miss_tokens or 0) + miss
            usage_details = parts or None
        clean = self._clean_json_text(content)
        # 空 content 的 assistant 消息不持久化：形状非法（发回 API 会被拒），
        # 且该次输出必然校验失败、由修正反馈取代
        assistant = (
            [{"role": "assistant", "content": clean}] if clean.strip() else []
        )
        return LLMResponse(
            content=clean,
            tokens=tokens,
            truncated=truncated,
            usage=usage_details,
            messages=[*new_messages, *assistant],
        )

    @staticmethod
    def _clean_json_text(content: str) -> str:
        """从模型返回文本里取 JSON 子串：截取首个 ``{`` 到末个 ``}``；找不到则原样返回。"""
        if not isinstance(content, str) or not content.strip():
            return content if isinstance(content, str) else ""
        start = content.find("{")
        end = content.rfind("}")
        if 0 <= start < end:
            return content[start : end + 1]
        return content
