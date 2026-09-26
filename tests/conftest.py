"""pytest 共享配置：只放跨测试的 autouse fixture。"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _production_tools_use_inmemory_world(monkeypatch):
    """单测/CI 不连库：把生产入口的工具装配钉回 InMemory 世界。"""
    from inmemory_world import build_inmemory_tools

    import pra.tools as tools_pkg

    monkeypatch.setattr(tools_pkg, "build_production_tools", build_inmemory_tools)
