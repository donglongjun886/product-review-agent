"""工具装配「按名替换」契约守护 —— CI 上恒跑。

契约：替换工具世界里某个工具，只允许**按 name 就地替换**（``replace_tool_by_name``）。
工具列表的顺序对下游无语义（``ToolRegistry`` / ``tools_node`` 全程按名调度），所以任何
``<expr>[<整数常量>] = …`` 的位置写死都是脆弱耦合 —— 一旦装配顺序变化就静默错位。本模块把
这条约束钉成可执行契约：

- (a) AST 静态反证：两个装配模块（``pra/tools/__init__.py``、
  ``pra/evaluation/harness/agent_scheme.py``）内不得出现任何整数常量下标赋值
  （``<expr>[<int>] = …``，与列表命名无关）—— 回退即红。
- (b) 行为单测：``replace_tool_by_name`` 在乱序列表上按名命中（放在非 [4]/[5] 下标也能替换，
  其余元素与顺序不变）；未命中抛 ``KeyError`` 且消息含该 name。
- (c) 清单断言：``build_tools()`` / ``build_production_tools()`` 各 6 件；
  ``make_eval_world_tools()`` / ``make_rag_world_tools()`` 各 5 件且 name 集合无 ``OCRTool``；
  RAG 世界的 Case/Policy 工具与 eval 世界不是同一实例（换了数据源）。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from helpers import tool_by_name

import pra.tools as tools_pkg
from pra.evaluation.harness.agent_scheme import (
    make_eval_world_tools,
    make_rag_world_tools,
)
from pra.tools import build_tools
from pra.tools.case_search.tool import CaseSearchTool, InMemoryCaseIndex
from pra.tools.policy_search.tool import InMemoryPolicyIndex

# 模块导入期抓真实装配函数：conftest 的 autouse fixture 会把 ``pra.tools.build_production_tools``
# 钉回 InMemory 世界（测试不连库），此处留住原函数做「生产装配清单」断言。
_REAL_BUILD_PRODUCTION_TOOLS = tools_pkg.build_production_tools

_REPO_ROOT = Path(__file__).resolve().parents[1]
# 两处会替换工具数据源的装配模块（工具世界组装在这里发生）。
_ASSEMBLY_MODULES = (
    _REPO_ROOT / "src" / "pra" / "tools" / "__init__.py",
    _REPO_ROOT / "src" / "pra" / "evaluation" / "harness" / "agent_scheme.py",
)
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
# (a) AST 静态反证：装配模块不得出现整数常量下标赋值
# ---------------------------------------------------------------------------


def _parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _name_of(node: ast.expr) -> str | None:
    """取 ``Name`` / ``Attribute`` 的短名（识别被调用的函数名与下标表达式的名字）。"""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _int_index_subscript_assignments(tree: ast.AST) -> list[tuple[int, str | None, int]]:
    """返回所有「整数常量下标赋值」的 (行号, 下标表达式名, 下标) —— **与列表命名无关**。

    只认赋值目标形如 ``<expr>[<int>] = …``（``Assign`` / ``AugAssign`` / ``AnnAssign``）。
    变量下标（``tools[index] = tool``，按名查找的实现细节）与字符串键（``opts["k"] = v``，
    字典读写）都不算；``bool`` 是 ``int`` 子类，显式排除。Python 3.9+ 的 ``Subscript.slice``
    直接是表达式节点（无 ``ast.Index`` 包装）。
    """
    findings: list[tuple[int, str | None, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        else:
            continue
        for tgt in targets:
            if not isinstance(tgt, ast.Subscript):
                continue
            idx = tgt.slice
            if (
                isinstance(idx, ast.Constant)
                and isinstance(idx.value, int)
                and not isinstance(idx.value, bool)
            ):
                findings.append((node.lineno, _name_of(tgt.value), idx.value))
    return findings


def test_assembly_modules_have_no_int_index_assignment():
    """反证：两个装配模块内不得出现 ``<expr>[<整数常量>] = …``（回退即红，与命名无关）。"""
    findings: list[tuple[str, int, str | None, int]] = []
    for path in _ASSEMBLY_MODULES:
        for line, base, idx in _int_index_subscript_assignments(_parse_module(path)):
            findings.append((path.relative_to(_REPO_ROOT).as_posix(), line, base, idx))
    assert findings == [], (
        f"装配模块出现整数常量下标赋值 {findings} —— 工具装配必须按名替换"
        "（replace_tool_by_name）；位置索引已废弃，回退即红"
    )


# ---------------------------------------------------------------------------
# (b) 行为单测：按名命中（乱序）/ 未命中 KeyError
# ---------------------------------------------------------------------------


def test_replace_tool_by_name_hits_out_of_canonical_index():
    """乱序列表上按名命中：目标落在非 [4]/[5] 下标也能替换，其余元素与顺序不变。"""
    tools = list(reversed(build_tools()))  # CaseSearchTool → index 1、PolicySearchTool → index 0
    case_pos = next(i for i, t in enumerate(tools) if t.name == "CaseSearchTool")
    assert case_pos not in (4, 5), "取反后应落在非规范下标，才能证明是『按名』而非『按下标』命中"

    replacement = CaseSearchTool(index=InMemoryCaseIndex([]))
    before = list(tools)

    tools_pkg.replace_tool_by_name(tools, replacement)

    assert tools[case_pos] is replacement, "同名工具应被就地替换"
    for i, original in enumerate(before):
        if i == case_pos:
            continue
        assert tools[i] is original, "未命中的元素不得被改动"
    assert [t.name for t in tools] == [t.name for t in before], "列表顺序不得变"


def test_replace_tool_by_name_miss_raises_keyerror_naming_the_tool():
    """未命中抛 ``KeyError``，且消息含该 name。"""
    tools = [t for t in build_tools() if t.name != "CaseSearchTool"]
    replacement = CaseSearchTool(index=InMemoryCaseIndex([]))

    with pytest.raises(KeyError, match="CaseSearchTool"):
        tools_pkg.replace_tool_by_name(tools, replacement)


# ---------------------------------------------------------------------------
# (c) 清单断言：6 / 6 / 5（无 OCR）/ 5（Case·Policy 换了数据源）
# ---------------------------------------------------------------------------


def test_build_tools_has_six_tools():
    tools = build_tools()
    assert len(tools) == 6
    assert {t.name for t in tools} == set(_EXPECTED_SIX)


def test_build_production_tools_has_six_tools():
    prod = _REAL_BUILD_PRODUCTION_TOOLS()
    assert len(prod) == 6
    assert {t.name for t in prod} == set(_EXPECTED_SIX)


def test_eval_world_has_exactly_five_tools_without_ocr():
    """评测世界刻意只有 5 件且**无 OCRTool** —— 红线：不得顺手补齐成 6 件。"""
    tools = make_eval_world_tools()
    assert len(tools) == 5
    names = {t.name for t in tools}
    assert "OCRTool" not in names, "评测世界刻意不带 OCRTool（案件自带 OCR 文本，不走 OCRTool）"
    assert names == set(_EXPECTED_EVAL_FIVE)


def test_rag_world_has_five_tools_with_swapped_knowledge_sources(monkeypatch):
    """RAG 世界：5 件、无 OCRTool；Case/Policy 换成新数据源（与 eval 世界非同一实例）。"""
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
