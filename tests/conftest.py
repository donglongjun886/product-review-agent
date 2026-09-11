"""pytest 共享配置：只放跨测试的 autouse fixture。

asyncio 由 pyproject 的 asyncio_mode="auto" 驱动。
``_reset_llm_backend`` 负责测试隔离：llm_shell 的后端注入位 ``set_llm_backend`` 是
进程级全局，注入假后端后必须还原，否则后续依赖默认 scripted 桩的测试会被污染；
惰性默认桩缓存 ``_default_backend`` 一并还原。
``_disable_langfuse_tracing`` 保证测试永不联网、观测为 no-op（见下）。
"""

from __future__ import annotations

import pytest

from pra.agent.guardrails import llm_shell


@pytest.fixture(autouse=True)
def _reset_llm_backend():
    """测试级隔离：注入位点与默认桩缓存前后还原（autouse，无需各文件声明依赖）。"""
    saved_backend = llm_shell._backend
    saved_default = llm_shell._default_backend
    try:
        yield
    finally:
        llm_shell._backend = saved_backend
        llm_shell._default_backend = saved_default


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

    生产/HTTP 入口（``pra.api.service.get_graph`` / ``pra.infra.persist_service._get_graph``）
    按设计调 ``pra.tools.build_production_tools()`` 读真库商品表；而测试必须无库可跑（CI 上没有
    MySQL），故把该装配函数替换为 ``build_tools()``。真库集成测试不依赖本 fixture —— 它们自行
    把真实装配函数换回去并显式连库。
    """
    import pra.tools as tools_pkg
    from pra.tools import build_tools

    monkeypatch.setattr(tools_pkg, "build_production_tools", build_tools)
