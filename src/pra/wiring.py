"""组合根 —— 生产图装配的唯一处；``get_production_graph`` 返回其编译单例。

首次调用才装配 ``build_agent_graph(tools=build_production_tools(),
checkpointer=make_memory_checkpointer())``（商品与商家读 MySQL、案例与政策读真实 RAG；
scripted LLM 桩，无需 API key），之后复用同一实例。

并发：build_agent_graph / InMemorySaver 均为同步、无 I/O（不 await），同一事件循环内检查与赋值
之间无协程切换点，故「先查缓存再构建」天然原子，无需加锁。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pra import tools as tools_pkg
from pra.agent.checkpointer import make_memory_checkpointer
from pra.agent.graph import build_agent_graph

if TYPE_CHECKING:  # 仅类型：运行时不需要 CompiledStateGraph（future annotations 惰性求值）
    from langgraph.graph.state import CompiledStateGraph

__all__ = ["get_production_graph"]


# 模块级单例缓存：首次调用 get_production_graph 时装配编译，之后复用（见模块 docstring 并发说明）。
_graph: CompiledStateGraph | None = None


def get_production_graph() -> CompiledStateGraph:
    """返回编译图单例（scripted LLM 桩 + 生产工具世界），首次调用时装配。

    工具世界取 ``pra.tools.build_production_tools()``（商品/商家读 MySQL，案例/政策读真实
    RAG）—— 测试/CI 由 ``tests/conftest.py`` 的 autouse fixture 把该装配钉回 InMemory。
    """
    global _graph
    if _graph is None:
        _graph = build_agent_graph(
            tools=tools_pkg.build_production_tools(),
            checkpointer=make_memory_checkpointer(),
        )
    return _graph
