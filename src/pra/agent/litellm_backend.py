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
        self._api_key: str | None = (api_key or os.environ.get(_API_KEY_ENV) or None)
        self._tool_catalog: list[dict] = self._extract_tool_catalog(tools)

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
            except Exception:
                args_schema = {}
            catalog.append(
                {
                    "name": str(name),
                    "description": str(getattr(tool, "description", "") or ""),
                    "args_schema": args_schema,
                }
            )
        return catalog

    def _render(
        self,
        *,
        node: str,
        state: dict,
        json_schema: dict,
        feedback: list[str] | None = None,
    ) -> tuple[str, str]:
        """组装 (system, user) 消息：system = 角色约束指令，user = 分节上下文 + Schema 要点
        + 可选修正提示。
        """
        system = build_system_prompt(node)
        user = build_user_prompt(
            node=node,
            state=state,
            json_schema=json_schema,
            tool_catalog=self._tool_catalog,
            feedbacks=feedback,
        )
        return system, user

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
            raise LLMBackendError(
                f"未知 node: {node!r}，应为 {sorted(SYSTEM_PROMPTS)}"
            )
        if not self._api_key:
            raise LLMBackendError(
                f"未配置 DeepSeek API key：请在构造 LiteLLMBackend(api_key=...) 时传入，"
                f"或设置环境变量 {_API_KEY_ENV} 后重试（当前 model={self.model}）。"
            )
        system, user = self._render(
            node=node, state=state, json_schema=json_schema, feedback=feedback
        )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
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
        except Exception:
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
            usage_details = parts or None
        return LLMResponse(
            content=self._clean_json_text(content),
            tokens=tokens,
            truncated=truncated,
            usage=usage_details,
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
