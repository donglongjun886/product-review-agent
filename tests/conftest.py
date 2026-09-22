"""pytest 共享配置：只放跨测试的 autouse fixture。

asyncio 由 pyproject 的 asyncio_mode="auto" 驱动。
``_disable_langfuse_tracing`` 保证测试永不联网、观测为 no-op（见下）。
``_production_tools_use_inmemory_world`` / ``_production_entry_uses_scripted_llm`` 把生产入口
的工具与 LLM 装配钉回确定性桩世界（见下）—— LLM 后端**不再有进程级全局**，注入一律经
``build_agent_graph(llm=...)``，故测试之间无需还原任何全局状态。
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_langfuse_tracing():
    """强制观测为 no-op：本机 `.env` 配了真实凭据时也不发 trace（永不联网）。

    ``get_tracer()`` 是进程级单例且配置来源含仓库根 ``.env``；实现是**预置单例为
    NullTracer**，不设环境变量（设 ``PRA_LANGFUSE_ENABLED`` 会污染 ``Settings``
    用例，真实环境变量优先级高于 .env）。需要观测行为的用例自行用
    ``pra.observability.tracing.set_tracer(...)`` 注入假实现。
    """
    from pra.observability import tracing as _tracing

    saved = _tracing._tracer
    _tracing._tracer = _tracing.NullTracer("test-isolation")
    try:
        yield
    finally:
        _tracing._tracer = saved


@pytest.fixture(autouse=True)
def _production_tools_use_inmemory_world(monkeypatch):
    """单测/CI 不连库：把生产入口的工具装配钉回 InMemory 世界。

    生产入口唯一装配处 ``pra.wiring.get_production_graph`` 按设计调
    ``pra.tools.build_production_tools()`` 读真库商品表；而测试必须无库可跑（CI 上没有
    MySQL），故把该装配函数替换为 ``build_tools()``。真库集成测试不依赖本 fixture —— 它们自行
    把真实装配函数换回去并显式连库。
    """
    import pra.tools as tools_pkg
    from pra.tools import build_tools

    monkeypatch.setattr(tools_pkg, "build_production_tools", build_tools)


@pytest.fixture(autouse=True)
def _production_entry_uses_scripted_llm(monkeypatch):
    """单测/CI 不调真实模型：把生产入口的 LLM 装配钉回 ``ScriptedLLMBackend``。

    生产入口唯一装配处 ``pra.wiring.get_production_graph`` 按设计调
    ``pra.wiring.build_llm_backend`` 读 ``Settings``（本机 `.env` 配了真实 key）；不钉回则走生产
    入口的用例会真的联网调模型：结果不确定、按量计费，CI 上没有凭据还会直接报错。故把该装配
    函数替换为构造确定性桩的替身（签名吃 ``tools=`` 关键字）。
    """
    from pra.agent.scripted_llm import ScriptedLLMBackend

    monkeypatch.setattr(
        "pra.wiring.build_llm_backend",
        lambda *, tools=None: ScriptedLLMBackend(),
    )
