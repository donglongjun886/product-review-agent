"""工具装配清单契约 —— CI 上恒跑。"""

from __future__ import annotations

from helpers import tool_by_name
from inmemory_world import InMemoryCaseIndex, InMemoryPolicyIndex, build_inmemory_tools

import pra.tools as tools_pkg
from pra.tools.case_search.tool import CaseSearchTool
from pra.tools.policy_search.tool import PolicySearchTool

_REAL_BUILD_PRODUCTION_TOOLS = tools_pkg.build_production_tools

_EXPECTED_FOUR = (
    "ProductTool",
    "MerchantTool",
    "CaseSearchTool",
    "PolicySearchTool",
)


def test_build_inmemory_tools_has_four_tools():
    tools = build_inmemory_tools()
    assert len(tools) == 4
    assert {t.name for t in tools} == set(_EXPECTED_FOUR)


def test_build_production_tools_has_four_tools():
    prod = _REAL_BUILD_PRODUCTION_TOOLS()
    assert len(prod) == 4
    assert {t.name for t in prod} == set(_EXPECTED_FOUR)


def test_search_tools_use_index_injected_at_construction():
    """``CaseSearchTool(index=)`` / ``PolicySearchTool(index=)`` 直接持有注入实例；
    ``build_inmemory_tools()`` 注入自己的 InMemory 索引（与调用方实例不同一）。"""
    case_index = InMemoryCaseIndex([])
    policy_index = InMemoryPolicyIndex([])

    assert CaseSearchTool(index=case_index)._index is case_index
    assert PolicySearchTool(index=policy_index)._index is policy_index

    world_tools = build_inmemory_tools()
    assert isinstance(tool_by_name(world_tools, "CaseSearchTool")._index, InMemoryCaseIndex)
    assert isinstance(tool_by_name(world_tools, "PolicySearchTool")._index, InMemoryPolicyIndex)
    assert tool_by_name(world_tools, "CaseSearchTool")._index is not case_index
    assert tool_by_name(world_tools, "PolicySearchTool")._index is not policy_index
