"""真实 litellm 后端 ``LiteLLMBackend``（real LLM 评测用）。

默认后端是 :class:`pra.agent.scripted_llm.ScriptedLLMBackend`（CI 回归依赖它）；本后端
实现同一 ``LLMBackend`` Protocol，经 ``set_llm_backend`` 注入即可替换。语义差异：

- 桩把 ``__STATE__`` 解析后做确定性分支；本后端把同样状态子集渲染成人读中文上下文
  （见 :mod:`pra.agent.llm_prompts`）发给真实模型 —— **输出非确定性**：同
  (node, __STATE__) 不保证同结果、不可重放，与桩的"逐字节一致"口径不可混用。schema
  强校验 / 重试 1 次仍在 llm_shell（``model_validate_json``），后端不做内容校验。

配置：``model`` 默认 ``"deepseek/deepseek-chat"``；``api_key`` 构造传入或读
``DEEPSEEK_API_KEY``（两处都缺不报错，首次 ``complete`` 前才抛 ``LLMBackendError``）；
``base_url`` 非 None 时以 ``api_base`` 传 litellm；``tools`` 提取
``{name, description, args_schema}`` 存为 plan 渲染用工具目录。

失败形态：任一错误（网络/超时/HTTP/无 key/上游异常）→ ``LLMBackendError``（llm_shell
按 transport 类处理：退避重试 1 次）；内容提取失败不抛，返回原样文本 + ``truncated``
标记。token 口径：``tokens`` = ``usage.total_tokens``（input+output 合计、含缓存命中；
无 usage → 0），schema 校验失败的尝试也全额累计（transport 失败不计）。
"""

from __future__ import annotations

import os
from typing import Any

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
from pra.agent.scripted_llm import (  # __STATE__ 解析（本项目惯例）
    _extract_state,
)

__all__ = ["LiteLLMBackend"]

# 环境变量：api_key 缺省来源
_API_KEY_ENV = "DEEPSEEK_API_KEY"


def _int_or_none(value: Any) -> int | None:
    """把 usage 字段转 int；缺失/None/非法 → ``None``（不填 0 冒充）。"""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class LiteLLMBackend(LLMBackend):
    """真实 litellm 后端：解析 __STATE__ → 渲染完整 prompt → 调 API（见模块 docstring）。"""

    name: str = "litellm"  # Protocol 属性；__init__ 覆写为 f"litellm-{model}"

    def __init__(
        self,
        *,
        model: str = "deepseek/deepseek-chat",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout_s: float = 60.0,
        max_tokens: int | None = None,
        tools: list | None = None,
    ) -> None:
        """构造真实 LLM 后端（只固化参数，不触发任何网络/API 调用）。

        :param model: litellm 模型名（provider 前缀写法）；
        :param api_key: None → 读 ``DEEPSEEK_API_KEY``（仍缺不报错，首次 ``complete``
            时抛 ``LLMBackendError``）；
        :param base_url: 可选 OpenAI 兼容端点，非 None 时以 ``api_base`` 传 litellm；
        :param temperature: 采样温度（0.0 —— 尽量稳定，但真实模型仍非确定性）；
        :param timeout_s: 单次请求超时秒数；max_tokens：输出上限（None = 模型默认）；
        :param tools: ``Tool`` 协议对象列表 —— 提取 {name, description, args_schema}
            存为 plan 渲染用工具目录；None/空 → plan 上下文说明"无可用工具"。
        """
        self.model = model
        self.name = f"litellm-{model}"
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.base_url = base_url
        self._api_key: str | None = (api_key or os.environ.get(_API_KEY_ENV) or None)
        self._tool_catalog: list[dict] = self._extract_tool_catalog(tools)

    # -- 工具目录提取（构造时调用；plan 渲染依赖） --------------------------------

    @staticmethod
    def _extract_tool_catalog(tools: list | None) -> list[dict]:
        """把 ``Tool`` 协议对象列表 → 工具目录。

        元素 = ``{"name", "description", "args_schema"}``（args_schema 由
        ``args_model.model_json_schema()`` 生成）；单个工具提取失败 → 跳过不崩；
        None/空/全失败 → 空目录（plan 走"无可用工具"分支）。
        """
        catalog: list[dict] = []
        for tool in tools or []:
            # 逐个提取并跳过异常工具：协议缺字段/args_model 异常只影响该工具。
            name = getattr(tool, "name", None)
            if not name:
                continue
            args_model = getattr(tool, "args_model", None)
            try:
                args_schema = (
                    args_model.model_json_schema() if args_model is not None else {}
                )
            except Exception:  # 防御：args_model 非 pydantic 模型时无 model_json_schema
                args_schema = {}
            catalog.append(
                {
                    "name": str(name),
                    "description": str(getattr(tool, "description", "") or ""),
                    "args_schema": args_schema,
                }
            )
        return catalog

    # -- state 解析 --------------------------------------------------------------

    @staticmethod
    def _parse_state(messages: list) -> dict:
        """解析 ``__STATE__ {json}`` 状态行；失败/找不到 → 空 dict（再包一层 try 双保险）。"""
        try:
            state = _extract_state(messages)
        except Exception:
            state = {}
        return state if isinstance(state, dict) else {}

    # -- 渲染（system=角色+约束；user=结构化人读上下文 + Schema 要点） -----------

    def _render(
        self,
        *,
        node: str,
        state: dict,
        json_schema: dict,
        messages: list | None = None,
    ) -> tuple[str, str]:
        """组装发给真实 LLM 的 (system, user) 完整消息（本类唯一渲染入口）。

        system 来自 :mod:`pra.agent.llm_prompts` 的约束中文指令；user 是分节上下文 +
        Schema 要点，并追加第 2 次尝试带回的修正提示（避免重试退化成同 prompt 空转）。
        """
        system = build_system_prompt(node)
        feedbacks = self._collect_feedbacks(messages)
        user = build_user_prompt(
            node=node,
            state=state,
            json_schema=json_schema,
            tool_catalog=self._tool_catalog,
            feedbacks=feedbacks,
        )
        return system, user

    @staticmethod
    def _collect_feedbacks(messages: list | None) -> list[str]:
        """收集修正提示（role=user 且 content 非 ``__STATE__`` 行），即上一轮校验错误。"""
        feedbacks: list[str] = []
        for msg in messages or []:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "user":
                continue  # 只回喂 user 角色；system 不参与
            content = msg.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            if "__STATE__" in content:
                continue  # 状态行本身不是修正提示
            feedbacks.append(content.strip())
        return feedbacks

    # -- LLMBackend.complete ------------------------------------------------------

    async def complete(
        self, *, node: str, messages: list, json_schema: dict
    ) -> LLMResponse:
        """按 node 渲染完整 prompt 并调用 litellm，返回 content 原样文本 + token 数。

        node 词表外或调用失败（无 key / 网络 / 超时 / HTTP / 上游异常）→
        ``LLMBackendError``（与 scripted 桩一致）；content 提取/清理失败**不抛**，返回
        原样文本交 llm_shell 强校验；截断（``finish_reason == "length"``）→
        ``LLMResponse.truncated=True``。
        """
        # 1) node 词表校验（与 SYSTEM_PROMPTS 词表一致）
        if node not in SYSTEM_PROMPTS:
            raise LLMBackendError(
                f"未知 node: {node!r}，应为 {sorted(SYSTEM_PROMPTS)}"
            )
        # 2) 解析 __STATE__（失败 → 空 state 兜底，渲染层自防御）
        state = self._parse_state(messages)
        # 3) 无 key 检查（构造后到真正调用前才抛）
        if not self._api_key:
            raise LLMBackendError(
                f"未配置 DeepSeek API key：请在构造 LiteLLMBackend(api_key=...) 时传入，"
                f"或设置环境变量 {_API_KEY_ENV} 后重试（当前 model={self.model}）。"
            )
        # 4) 渲染完整消息
        system, user = self._render(
            node=node, state=state, json_schema=json_schema, messages=messages
        )
        # 5) litellm 调用（延迟 import：模块 import 期零 litellm 依赖）
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature,
            "timeout": self.timeout_s,
            # 强制 JSON 对象输出（deepseek 等 OpenAI 兼容网关支持；不支持时由上层降级）
            "response_format": {"type": "json_object"},
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self.base_url:
            kwargs["api_base"] = self.base_url
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        try:
            # 离线化：litellm import 时默认联网拉远程 model cost map，无外网会白等一次
            # 超时 —— 置 LITELLM_LOCAL_MODEL_COST_MAP=True 用本地备份，import 零网络。
            os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
            import litellm  # 延迟 import：避免本模块被 import 就拉起 litellm

            resp = await litellm.acompletion(**kwargs)
        except LLMBackendError:
            raise  # 不二次包装
        except Exception as exc:
            # 网络/超时/HTTP/上游 4xx-5xx/参数错误等一律收敛为 LLMBackendError
            raise LLMBackendError(
                f"litellm 调用失败（node={node}, model={self.model}）：{exc}"
            ) from exc
        # 6) 取 content + 截断标记（choices 异常 → 空串，交上层校验）
        content = ""
        truncated = False
        try:
            choice = resp.choices[0]
            message = choice.message
            content = message.content or ""
            # finish_reason == "length" ⇒ 内容可能不完整；标记给 llm_shell 按
            # transport 类处理（校验失败不重试），不再当普通校验失败烧一次全量调用。
            truncated = str(getattr(choice, "finish_reason", "") or "") == "length"
        except Exception:
            content = ""
        # 7) token 记账：tokens = usage.total_tokens（input+output 合计、含缓存命中；
        #    无 usage → 0）；另透出拆分 usage（input/output/total，对齐 Langfuse
        #    usage_details），取不到的键不放（不填 0 冒充），整体拿不到 → None。
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

    # -- content 清理（best-effort，任何失败不抛，原样返回） ----------------------

    @staticmethod
    def _clean_json_text(content: str) -> str:
        """从模型返回文本里取 JSON 子串（best-effort，绝不抛）。

        已是干净 JSON / 找不到 '{' → 原样返回；带 ```json 围栏 / 前后杂文本 → 截取首个
        '{' 到末个 '}'，提高首次校验通过率。
        """
        if not isinstance(content, str) or not content.strip():
            return content if isinstance(content, str) else ""
        start = content.find("{")
        end = content.rfind("}")
        if 0 <= start < end:
            return content[start : end + 1]
        return content
