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
