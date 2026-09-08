"""Phase 3（real LLM Evaluation）第一块：真实 litellm 后端 ``LiteLLMBackend``。

与确定性桩的关系：
- 默认后端 = :class:`pra.agent.scripted_llm.ScriptedLLMBackend`（确定性走查，无 API
  key / 无网络 / 同 (node, __STATE__) 恒同输出，Phase 1 CI 回归依赖它）。
- 本后端实现同一 :class:`pra.agent.guardrails.llm_shell.LLMBackend` Protocol
  （``name`` + ``async complete(*, node, messages, json_schema)``），经
  ``pra.agent.guardrails.llm_shell.set_llm_backend`` 注入即可替换默认桩 —— 节点 /
  llm_shell / 测试侧无感（04 §3.1 / llm_shell 模块注记）。

与桩的差异（**结论边界，写报告/评测时必须声明**）：
- 桩把 ``__STATE__`` JSON 解析后做**确定性分支**；本后端把同样的 JSON 状态子集
  **渲染成结构化人读中文上下文**（见 :mod:`pra.agent.llm_prompts`，不贴裸 JSON），
  发给 litellm 网关背后的真实模型 —— **输出非确定性**：同 (node, __STATE__) 不保证
  同结果（temperature=0.0 只是尽量稳定），**Phase 3 评测不可重放**，与 Phase 1 的
  "同 case 重跑逐字节一致"验收口径不可混用。
- schema 强校验/重试 1 次仍在 llm_shell（``OutputModel.model_validate_json``）完成：
  后端只负责"把状态渲染成完整 prompt → 调 litellm → 返回 content 原样文本 +
  usage.total_tokens"，与 scripted 桩一致不在后端做内容校验。

配置来源：
- ``model``（默认 ``"deepseek/deepseek-chat"``，litellm provider 前缀写法）；
- ``api_key``：构造传入，或为 None 时读环境变量 ``DEEPSEEK_API_KEY``；两处都缺 →
  构造不报错（便于在无 key 环境 import/渲染调试），**真正 ``complete`` 调用前仍无
  key 才抛 ``LLMBackendError``**（清晰中文提示）；
- ``base_url``：可选，非 None 时以 ``api_base=`` 传给 litellm（OpenAI 兼容端点）；
- ``tools``：可选，实现 :class:`pra.tools.base.Tool` 协议的对象列表 —— 构造时提取
  ``{name, description, args_schema(args_model.model_json_schema())}`` 存为 plan
  渲染用的工具目录；None/空 → plan 上下文注明"无可用工具"并提示输出 conclude。

本模块只做"渲染 + 转发"，含任一失败形态（网络/超时/HTTP/无 key/上游异常）→ 抛
``LLMBackendError``（llm_shell 捕获后按失败处理：第 1 次追加修正提示重试 1 次，仍
失败走节点降级）。**内容提取失败不抛**：返回 content 原样文本，交由上层
``model_validate_json`` 校验（与 scripted 桩一致，见模块底部 ``_clean_json_text``
说明）。

语法约定：顶部 ``from __future__ import annotations``；import 一律 ``pra.*`` 风格。
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
from pra.agent.scripted_llm import (  # __STATE__ 解析（G5/G6 隐式契约，本项目惯例）
    _extract_state,
)

__all__ = ["LiteLLMBackend"]

# 环境变量：api_key 缺省来源（与任务规格/评测接线约定一致）。
_API_KEY_ENV = "DEEPSEEK_API_KEY"


class LiteLLMBackend(LLMBackend):
    """真实 litellm 后端（Phase 3 real LLM）：解析 __STATE__ → 渲染完整 prompt → 调 API。

    实现 :class:`pra.agent.guardrails.llm_shell.LLMBackend` Protocol：``name`` 标识 +
    ``async complete``。细节见模块 docstring。
    """

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
        """构造真实 LLM 后端（只做参数固化，不触发任何网络/API 调用）。

        :param model: litellm 模型名（provider 前缀写法，如 "deepseek/deepseek-chat"）；
        :param api_key: API key；None → 读环境变量 ``DEEPSEEK_API_KEY``（仍缺不报错，
            首次 ``complete`` 时抛 ``LLMBackendError``）；
        :param base_url: 可选 OpenAI 兼容端点，非 None 时以 ``api_base`` 传给 litellm；
        :param temperature: 采样温度（默认 0.0 —— 尽量稳定，但真实模型仍非确定性）；
        :param timeout_s: 单次请求超时秒数（litellm timeout）；
        :param max_tokens: 可选输出上限 token（None = 不传，用模型默认）；
        :param tools: 可选、实现 ``pra.tools.base.Tool`` 协议的对象列表 —— 构造时
            提取 {name, description, args_schema} 存为 plan 渲染用工具目录；
            None/空 → plan 上下文说明"无可用工具"并提示输出 conclude。
        """
        self.model = model
        self.name = f"litellm-{model}"
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self.base_url = base_url
        # api_key 固化：构造入参优先，其次环境变量；两者皆缺 → 留空，complete 前兜底。
        self._api_key: str | None = (api_key or os.environ.get(_API_KEY_ENV) or None)
        self._tool_catalog: list[dict] = self._extract_tool_catalog(tools)

    # -- 工具目录提取（构造时调用；plan 渲染依赖） --------------------------------

    @staticmethod
    def _extract_tool_catalog(tools: list | None) -> list[dict]:
        """把实现 ``pra.tools.base.Tool`` 协议的对象列表 → 工具目录。

        目录元素 = ``{"name", "description", "args_schema"}``，其中 args_schema 由该
        工具的 ``args_model.model_json_schema()`` 生成（供 plan 上下文列出合法入参）。
        单个工具提取失败（协议缺字段/args_model 异常）→ 跳过该工具不崩；
        None/空/全失败 → 空目录（plan 渲染走"无可用工具"分支）。
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
        """解析 ``__STATE__ {json}`` 状态行（复用 scripted_llm 私有函数，本项目惯例）。

        解析失败/找不到 → 空 dict（各节点渲染函数自带空事实兜底，不许崩）——
        额外 try/except 双保险：_extract_state 本身已容错，这里兜底任何异常。
        """
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

        - system：来自 :mod:`pra.agent.llm_prompts` 的完整约束中文指令（按 node）；
        - user：state 渲染成分节人读中文上下文 + 输出 JSON Schema 要点；若 ``messages``
          里带 llm_shell 第 2 次尝试追加的修正提示（role=user 且非 ``__STATE__`` 行），
          追加到 user 尾部回喂模型修正输出 —— 保证真实后端"校验失败重试 1 次"不是
          同 prompt 空转（§3.0：把校验错误原文回喂给 LLM 修正输出）。
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
        """收集 llm_shell 追加的修正提示文本（role=user 且 content 非 ``__STATE__`` 行）。

        节点 ``_build_messages`` 产出 [system, user("__STATE__ {json}")]；llm_shell
        第 1 次失败后向工作副本追加一条 role=user 的修正提示 —— 该条不带 ``__STATE__``
        前缀，正是这里要回喂模型的"上一轮校验错误"。返回列表（通常 0~1 条）。
        """
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

        - node ∈ {"hypothesize","plan","reevaluate","decide"}（词表外 → LLMBackendError，
          与 scripted 桩对未知 node 的处理一致，供降级路径测试）；
        - 失败形态（无 key / 网络 / 超时 / HTTP / 上游异常）→ ``LLMBackendError``；
        - 响应 content 提取/清理失败**不抛**：返回原样文本，交给 llm_shell 的
          ``model_validate_json`` 强校验（与 scripted 桩一致，校验在 llm_shell）。
        """
        # 1) node 词表校验（与 system prompt 词表一致）
        if node not in SYSTEM_PROMPTS:
            raise LLMBackendError(
                f"未知 node: {node!r}，应为 {sorted(SYSTEM_PROMPTS)}"
            )
        # 2) 解析 __STATE__（失败 → 空 state 兜底，渲染层自防御）
        state = self._parse_state(messages)
        # 3) 无 key 检查（构造后到真正调用前才抛；含清晰中文提示）
        if not self._api_key:
            raise LLMBackendError(
                f"未配置 DeepSeek API key：请在构造 LiteLLMBackend(api_key=...) 时传入，"
                f"或设置环境变量 {_API_KEY_ENV} 后重试（当前 model={self.model}）。"
            )
        # 4) 渲染完整消息
        system, user = self._render(
            node=node, state=state, json_schema=json_schema, messages=messages
        )
        # 5) litellm 调用（延迟 import：本模块 import 期零网络/零 litellm 依赖）
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
            # 延迟 import + 离线化：litellm 1.100.0 import 时默认联网拉远程 model
            # cost map（raw.githubusercontent.com），无外网会白等一次网络超时 ——
            # 置 LITELLM_LOCAL_MODEL_COST_MAP=True 用本地备份，import 零网络。
            os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
            import litellm  # 延迟 import：避免本模块被 import 就拉起 litellm

            resp = await litellm.acompletion(**kwargs)
        except LLMBackendError:
            raise  # 不二次包装
        except Exception as exc:
            # 网络/超时/HTTP/缺 key/上游 4xx-5xx/参数错误等一律收敛为 LLMBackendError
            raise LLMBackendError(
                f"litellm 调用失败（node={node}, model={self.model}）：{exc}"
            ) from exc
        # 6) 取 content（choices 异常结构 → 空串交由上层校验，不抛）
        content = ""
        try:
            message = resp.choices[0].message
            content = message.content or ""
        except Exception:
            content = ""
        # 7) token 记账（无 usage → 0）
        tokens = 0
        usage = getattr(resp, "usage", None)
        if usage is not None:
            try:
                tokens = int(getattr(usage, "total_tokens", 0) or 0)
            except (TypeError, ValueError):
                tokens = 0
        return LLMResponse(content=self._clean_json_text(content), tokens=tokens)

    # -- content 清理（best-effort，任何失败不抛，原样返回） ----------------------

    @staticmethod
    def _clean_json_text(content: str) -> str:
        """从模型返回文本里取 JSON 子串（best-effort，绝不抛）。

        - 已是干净 JSON / 找不到 '{'：原样返回（交给 llm_shell ``model_validate_json``
          强校验并回喂重试 —— 后端不做内容校验，与 scripted 桩一致）；
        - 带 ```json 围栏 / 前后杂文本：截取首个 '{' 到末个 '}' 的子串返回（提高第 1
          次校验通过率，减少一次修正重试成本）。
        """
        if not isinstance(content, str) or not content.strip():
            return content if isinstance(content, str) else ""
        start = content.find("{")
        end = content.rfind("}")
        if 0 <= start < end:
            return content[start : end + 1]
        return content
