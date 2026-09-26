"""组合根 —— 生产图装配的唯一处；``get_production_graph`` 返回其编译单例。

首次调用才装配 ``build_agent_graph(tools=build_production_tools(),
checkpointer=make_memory_checkpointer(), llm=build_llm_backend(tools=...))``（商品与商家读
MySQL、案例与政策读真实 RAG、LLM 走真实 litellm 网关；凭据经 ``Settings`` 读仓库根
``.env``，缺 ``DEEPSEEK_API_KEY`` 显式抛 ``RuntimeError`` 而非回落桩），之后复用同一实例。

并发：build_agent_graph / InMemorySaver 均为同步、无 I/O（不 await），同一事件循环内检查与赋值
之间无协程切换点，故「先查缓存再构建」天然原子，无需加锁。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pra import tools as tools_pkg
from pra.agent.checkpointer import make_memory_checkpointer
from pra.agent.graph import build_agent_graph
from pra.agent.litellm_backend import LiteLLMBackend
from pra.infra.db import Settings

if TYPE_CHECKING:  # 仅类型：运行时不需要（future annotations 惰性求值）
    from langgraph.graph.state import CompiledStateGraph

    from pra.agent.guardrails.llm_shell import LLMBackend

__all__ = ["build_llm_backend", "get_production_graph"]


# 模块级单例缓存：首次调用 get_production_graph 时装配编译，之后复用（见模块 docstring 并发说明）。
_graph: CompiledStateGraph | None = None


def build_llm_backend(*, tools: list | None = None) -> LLMBackend:
    """构造生产 LLM 后端（真实 litellm 网关），配置来自 ``Settings``（仓库根 ``.env``）。

    :param tools: 取证工具列表 —— 经 ``LiteLLMBackend`` 渲染进 plan 节点的工具目录；
        None → 真实模型看到"无可用工具"。
    :raises RuntimeError: ``DEEPSEEK_API_KEY`` 缺失或全空白 —— 显式失败。
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
    """返回编译图单例（真库商品/商家 + 真实 RAG + 真实 LLM 网关），首次调用时装配。

    工具世界取 ``pra.tools.build_production_tools()``（商品/商家读 MySQL，案例/政策读真实
    RAG），同一份列表分别交给 LLM 后端（渲染工具目录）与图；LLM 凭据取 ``Settings``（仓库根
    ``.env``），缺失即抛 ``RuntimeError``。测试/CI 由 ``tests/conftest.py`` 的 autouse fixture
    把工具装配钉回 ``tests/inmemory_world.py`` 的 InMemory 世界，LLM 后端由各用例自行注入。
    """
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
