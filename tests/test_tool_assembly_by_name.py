"""工具装配清单契约 —— CI 上恒跑。

契约：工具的数据源一律**构造时注入**（如 ``CaseSearchTool(index=...)``），装配后不再就地替换
列表元素；工具列表的顺序对下游无语义（tools_node 全程按 name 调度），故断言一律按名取工具。

本模块钉住**业务清单**（工具数量与 name 集合是生产事实口径）：

- (a) ``build_inmemory_tools()`` / ``build_production_tools()`` 各 4 件、name 集合相同；
- (b) 数据源构造时注入：注入哪个实例就用哪个；``build_inmemory_tools()`` 注入的是它自己的
  InMemory 索引，不是调用方的实例。
"""

from __future__ import annotations

from helpers import tool_by_name
from inmemory_world import InMemoryCaseIndex, InMemoryPolicyIndex, build_inmemory_tools

import pra.tools as tools_pkg
from pra.tools.case_search.tool import CaseSearchTool
from pra.tools.policy_search.tool import PolicySearchTool

# 模块导入期抓真实装配函数：conftest 的 autouse fixture 会把 ``pra.tools.build_production_tools``
# 钉回 InMemory 世界（测试不连库），此处留住原函数做「生产装配清单」断言。
_REAL_BUILD_PRODUCTION_TOOLS = tools_pkg.build_production_tools

_EXPECTED_FOUR = (
    "ProductTool",
    "MerchantTool",
    "CaseSearchTool",
    "PolicySearchTool",
)


# ---------------------------------------------------------------------------
# (a) tests 世界 / 生产装配：各 4 件
# ---------------------------------------------------------------------------


def test_build_inmemory_tools_has_four_tools():
    tools = build_inmemory_tools()
    assert len(tools) == 4
    assert {t.name for t in tools} == set(_EXPECTED_FOUR)


def test_build_production_tools_has_four_tools():
    prod = _REAL_BUILD_PRODUCTION_TOOLS()
    assert len(prod) == 4
    assert {t.name for t in prod} == set(_EXPECTED_FOUR)


# ---------------------------------------------------------------------------
# (b) 数据源构造时注入：注入哪个实例就用哪个（回退反证）
# ---------------------------------------------------------------------------


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
