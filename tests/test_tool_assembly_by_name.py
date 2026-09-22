"""工具装配清单契约 —— CI 上恒跑。

契约：工具的数据源一律**构造时注入**（``build_tools(..., case_index=...)`` /
``make_eval_world_tools()`` 固定 Eval World 种子），装配后不再就地替换列表元素；工具列表的
顺序对下游无语义（tools_node 全程按 name 调度），故断言一律按名取工具。

本模块钉住两组**业务清单**（工具数量与 name 集合是评测/生产事实口径）：

- (a) ``build_tools()`` / ``build_production_tools()`` 各 6 件（含 ``OCRTool``）；
- (b) ``make_eval_world_tools()`` 5 件且 name 集合无 ``OCRTool``；

(c) 数据源构造时注入：给了就用、不给即 InMemory 种子。
"""

from __future__ import annotations

from helpers import tool_by_name

import pra.tools as tools_pkg
from pra.evaluation.harness.agent_scheme import make_eval_world_tools
from pra.tools import build_tools
from pra.tools.case_search.tool import InMemoryCaseIndex
from pra.tools.policy_search.tool import InMemoryPolicyIndex

# 模块导入期抓真实装配函数：conftest 的 autouse fixture 会把 ``pra.tools.build_production_tools``
# 钉回 InMemory 世界（测试不连库），此处留住原函数做「生产装配清单」断言。
_REAL_BUILD_PRODUCTION_TOOLS = tools_pkg.build_production_tools

_EXPECTED_SIX = (
    "ProductTool",
    "ImageAnalysisTool",
    "OCRTool",
    "MerchantTool",
    "CaseSearchTool",
    "PolicySearchTool",
)
# 评测世界刻意只有 5 件（无 OCRTool）—— 见 agent_scheme.make_eval_world_tools docstring。
_EXPECTED_EVAL_FIVE = (
    "ProductTool",
    "ImageAnalysisTool",
    "MerchantTool",
    "CaseSearchTool",
    "PolicySearchTool",
)


# ---------------------------------------------------------------------------
# (a) 生产 / 默认世界：6 件
# ---------------------------------------------------------------------------


def test_build_tools_has_six_tools():
    tools = build_tools()
    assert len(tools) == 6
    assert {t.name for t in tools} == set(_EXPECTED_SIX)


def test_build_production_tools_has_six_tools():
    prod = _REAL_BUILD_PRODUCTION_TOOLS()
    assert len(prod) == 6
    assert {t.name for t in prod} == set(_EXPECTED_SIX)


# ---------------------------------------------------------------------------
# (b) 评测世界：5 件（无 OCR）
# ---------------------------------------------------------------------------


def test_eval_world_has_exactly_five_tools_without_ocr():
    """评测世界刻意只有 5 件且**无 OCRTool** —— 红线：不得顺手补齐成 6 件。"""
    tools = make_eval_world_tools()
    assert len(tools) == 5
    names = {t.name for t in tools}
    assert "OCRTool" not in names, "评测世界刻意不带 OCRTool（案件自带 OCR 文本，不走 OCRTool）"
    assert names == set(_EXPECTED_EVAL_FIVE)


# ---------------------------------------------------------------------------
# (c) 数据源构造时注入：不给就不换（回退反证）
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
