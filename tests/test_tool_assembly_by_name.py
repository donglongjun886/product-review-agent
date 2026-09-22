"""工具装配清单契约 —— CI 上恒跑。

契约：工具的数据源一律**构造时注入**（``build_tools(..., case_index=...)`` /
``make_eval_world_tools(case_index=...)``），装配后不再就地替换列表元素；工具列表的顺序对
下游无语义（tools_node 全程按 name 调度），故断言一律按名取工具。

本模块钉住三组**业务清单**（工具数量与 name 集合是评测/生产事实口径）：

- (a) ``build_tools()`` / ``build_production_tools()`` 各 6 件（含 ``OCRTool``）；
- (b) ``make_eval_world_tools()`` / ``make_rag_world_tools()`` 各 5 件且 name 集合无
  ``OCRTool``；
- (c) RAG 世界的 Case/Policy 工具注入了 RAG 侧数据源，且与 eval 世界不是同一实例。
"""

from __future__ import annotations

from helpers import tool_by_name

import pra.tools as tools_pkg
from pra.evaluation.harness.agent_scheme import (
    make_eval_world_tools,
    make_rag_world_tools,
)
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
# 评测 / RAG 世界刻意只有 5 件（无 OCRTool）—— 见 agent_scheme.make_eval_world_tools docstring。
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
# (b) 评测 / RAG 世界：5 件（无 OCR）
# ---------------------------------------------------------------------------


def test_eval_world_has_exactly_five_tools_without_ocr():
    """评测世界刻意只有 5 件且**无 OCRTool** —— 红线：不得顺手补齐成 6 件。"""
    tools = make_eval_world_tools()
    assert len(tools) == 5
    names = {t.name for t in tools}
    assert "OCRTool" not in names, "评测世界刻意不带 OCRTool（案件自带 OCR 文本，不走 OCRTool）"
    assert names == set(_EXPECTED_EVAL_FIVE)


def test_rag_world_has_five_tools_with_swapped_knowledge_sources(monkeypatch):
    """RAG 世界：5 件、无 OCRTool；Case/Policy 构造时就注入了 RAG 侧数据源。"""
    sentinel_case = InMemoryCaseIndex([])
    sentinel_policy = InMemoryPolicyIndex([])
    from pra.rag import factory

    # 注入假 builder：避免真 chroma / fastembed（CI 无 extra 也能跑）；只验「换了数据源」这一层。
    monkeypatch.setattr(factory, "build_case_index", lambda *a, **kw: sentinel_case)
    monkeypatch.setattr(factory, "build_policy_index", lambda *a, **kw: sentinel_policy)

    rag = make_rag_world_tools(options={"embedding_model": object()})
    assert len(rag) == 5
    names = {t.name for t in rag}
    assert "OCRTool" not in names
    assert names == set(_EXPECTED_EVAL_FIVE)

    rag_case = tool_by_name(rag, "CaseSearchTool")
    rag_policy = tool_by_name(rag, "PolicySearchTool")
    assert rag_case._index is sentinel_case, "CaseSearchTool 应注入了 RAG 侧数据源"
    assert rag_policy._index is sentinel_policy, "PolicySearchTool 应注入了 RAG 侧数据源"

    eval_tools = make_eval_world_tools()
    assert rag_case is not tool_by_name(eval_tools, "CaseSearchTool"), (
        "RAG 世界的 CaseSearchTool 必须与 eval 世界不是同一实例（换了数据源）"
    )
    assert rag_policy is not tool_by_name(eval_tools, "PolicySearchTool"), (
        "RAG 世界的 PolicySearchTool 必须与 eval 世界不是同一实例（换了数据源）"
    )


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
