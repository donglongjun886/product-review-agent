"""pytest 共享配置 —— pytest-asyncio 由 pyproject [tool.pytest.ini_options] 的
asyncio_mode="auto" 驱动；本文件只放跨测试的 autouse fixture。

关键 autouse fixture：``_reset_llm_backend`` —— llm_shell 的后端注入是**进程级全局**
（``set_llm_backend``，见 pra/agent/guardrails/llm_shell.py 契约 §4.1），任一测试注入
假后端后必须还原，否则后续依赖默认 scripted 桩的测试（节点成功路径/脚本确定性）会被
污染。本 fixture 在每个测试前后快照并恢复模块级 ``_backend``，并把惰性默认桩缓存
``_default_backend`` 一并清掉，保证测试之间完全隔离、顺序无关。

语法/import 约定：顶部 ``from __future__ import annotations``；import 一律 pra.*。
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
    """测试级隔离：强制观测为 no-op（即便本机 `.env` / 环境变量配了真实 Langfuse）。

    理由：``get_tracer()`` 是进程级单例，且配置来源包含仓库根 ``.env`` —— 开发机
    配了真实凭据时，业务路径会向真实 Langfuse 发 trace（联网 + 污染观测数据）。

    实现：**预置单例为 NullTracer**（而不是设环境变量）—— 设 ``PRA_LANGFUSE_ENABLED``
    会污染 ``Settings`` 相关用例（真实环境变量优先级高于 .env）。需要观测行为的用例
    自行 ``pra.observability.tracing.set_tracer(...)`` 注入假实现。
    """
    from pra.observability import tracing as _tracing

    saved = _tracing._tracer
    _tracing._tracer = _tracing.NullTracer("test-isolation")
    try:
        yield
    finally:
        _tracing._tracer = saved
