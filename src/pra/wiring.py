"""组合根 —— 生产图装配；``get_production_graph`` 返回其编译单例。"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pra import tools as tools_pkg
from pra.agent.checkpointer import make_memory_checkpointer
from pra.agent.graph import build_agent_graph
from pra.agent.litellm_backend import LiteLLMBackend
from pra.infra.db import Settings

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from pra.agent.guardrails.llm_shell import LLMBackend

__all__ = ["build_llm_backend", "get_production_graph", "trace_callbacks"]

logger = logging.getLogger(__name__)


_graph: CompiledStateGraph | None = None

# 模块级 memo：Langfuse 回调（进程内单例）。None = 尚未装配。
_trace_callbacks: list | None = None


def trace_callbacks() -> list:
    """返回 Langfuse 回调列表；无凭据或 SDK 不可用时返回空列表。

    :return: 含一个 ``langfuse.langchain.CallbackHandler`` 的列表，或 ``[]``。
    """
    global _trace_callbacks
    if _trace_callbacks is None:
        _trace_callbacks = _build_trace_callbacks()
    return _trace_callbacks


def _build_trace_callbacks() -> list:
    """按 ``Settings`` 凭据装配 Langfuse 回调；失败记 warning 并返回空列表。"""
    settings = Settings()
    public_key = settings.langfuse_public_key
    secret_key = settings.langfuse_secret_key
    if not public_key or not secret_key:
        return []
    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler
    except Exception as exc:  # noqa: BLE001
        logger.warning("Langfuse SDK 不可用，观测关闭：%s", exc)
        return []
    try:
        Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=settings.langfuse_host or "http://localhost:3000",
        )
        return [CallbackHandler()]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Langfuse 装配失败，观测关闭：%s", exc)
        return []


def build_llm_backend(*, tools: list | None = None) -> LLMBackend:
    """构造生产 LLM 后端，配置来自 ``Settings``。

    :param tools: 取证工具列表 —— 经 ``LiteLLMBackend`` 渲染进 plan 节点的工具目录；
        None → 真实模型看到"无可用工具"。
    :raises RuntimeError: ``DEEPSEEK_API_KEY`` 缺失或全空白。
    """
    settings = Settings()
    api_key = settings.deepseek_api_key
    if api_key is None or not api_key.strip():
        raise RuntimeError(
            "未配置 DEEPSEEK_API_KEY：生产装配需要真实 LLM 网关凭据，"
            "请在仓库根 .env 中设置 DEEPSEEK_API_KEY（参见 .env.example）。"
        )
    return LiteLLMBackend(
        model=settings.deepseek_model,
        api_key=api_key,
        base_url=settings.deepseek_base_url,
        tools=tools,
    )


def get_production_graph() -> CompiledStateGraph:
    """返回编译图单例，首次调用时装配。"""
    global _graph
    if _graph is None:
        tools = tools_pkg.build_production_tools()
        llm = build_llm_backend(tools=tools)
        _graph = build_agent_graph(
            tools=tools,
            checkpointer=make_memory_checkpointer(),
            llm=llm,
        )
    return _graph
