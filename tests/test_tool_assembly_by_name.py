"""工具装配清单契约 —— CI 上恒跑。

契约：工具的数据源一律**构造时注入**（``build_tools(..., case_index=...)``），装配后不再就地
替换列表元素；工具列表的顺序对下游无语义（tools_node 全程按 name 调度），故断言一律按名取工具。

本模块钉住**业务清单**（工具数量与 name 集合是生产事实口径）：

- (a) ``build_tools()`` / ``build_production_tools()`` 各 4 件；
- (b) 数据源构造时注入：给了就用、不给即 InMemory 种子。
"""

from __future__ import annotations

from helpers import tool_by_name

import pra.tools as tools_pkg
from pra.tools import build_tools
from pra.tools.case_search.tool import InMemoryCaseIndex
from pra.tools.policy_search.tool import InMemoryPolicyIndex

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
# (a) 生产 / 默认世界：4 件
# ---------------------------------------------------------------------------


def test_build_tools_has_four_tools():
    tools = build_tools()
    assert len(tools) == 4
    assert {t.name for t in tools} == set(_EXPECTED_FOUR)


def test_build_production_tools_has_four_tools():
    prod = _REAL_BUILD_PRODUCTION_TOOLS()
    assert len(prod) == 4
    assert {t.name for t in prod} == set(_EXPECTED_FOUR)


# ---------------------------------------------------------------------------
# (b) 数据源构造时注入：不给就不换（回退反证）
# ---------------------------------------------------------------------------


def test_build_tools_injects_search_indexes_at_construction():
    """``build_tools(case_index=/policy_index=)`` 直接作用于对应工具；不给即 InMemory 种子。"""
    case_index = InMemoryCaseIndex([])
    policy_index = InMemoryPolicyIndex([])

    tools = build_tools(case_index=case_index, policy_index=policy_index)
    assert tool_by_name(tools, "CaseSearchTool")._index is case_index
    assert tool_by_name(tools, "PolicySearchTool")._index is policy_index

    default_tools = build_tools()
    assert isinstance(tool_by_name(default_tools, "CaseSearchTool")._index, InMemoryCaseIndex)
    assert isinstance(tool_by_name(default_tools, "PolicySearchTool")._index, InMemoryPolicyIndex)
    assert tool_by_name(default_tools, "CaseSearchTool")._index is not case_index
