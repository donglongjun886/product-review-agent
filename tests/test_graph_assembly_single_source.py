"""装配唯一处契约：生产图只在组合根 ``pra.wiring`` 组装一次。

生产图必须只在组合根 ``pra.wiring.get_production_graph`` 组装一次：入口模块
（``pra.infra.persist_service``）不得自写一份装配、自持一个模块级单例。
本模块把「装配只有一处」变成**可执行契约**：

- (a) 单例同一性：进程内 ``get_production_graph()`` 恒返回同一实例。
- (b) 命名空间：入口模块不再自带 ``get_graph`` / ``_get_graph`` / ``_graph`` /
  ``_compiled_graph`` / ``build_agent_graph``。
- (c) 结构：AST 扫 ``src/pra/**/*.py`` 的 ``build_agent_graph(...)`` 调用点，生产装配调用点
  集合必须恰为 ``{src/pra/wiring.py}``（评测跑分脚本在 ``scripts/`` 下，不属于 ``src/pra``）。

任一回退（删 ``pra/wiring.py`` 或入口重新自建单例/装配）→ 对应断言立刻变红。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from helpers import AlwaysRaiseBackend

from pra import wiring
from pra.infra import persist_service as ps

# 生产装配调用点 —— 有且仅有组合根一处。
_PRODUCTION_ASSEMBLY_SITES = {"src/pra/wiring.py"}

# 入口模块不得再自带的装配 / 单例件（出现任一即「重复装配」回退）。
_FORBIDDEN_ENTRY_ATTRS = (
    "get_graph",
    "_get_graph",
    "_graph",
    "_compiled_graph",
    "build_agent_graph",
)


def test_get_production_graph_is_a_process_singleton(monkeypatch):
    """(a) 单例同一性：连续两次取图是同一实例（模块级缓存生效，不再双份单例）。

    ``monkeypatch`` 注入必抛后端并清空图缓存 —— 本用例只校验「取两次是同一实例」，
    不构造真 LLM 后端（不读 Settings / 凭据）。
    """
    monkeypatch.setattr(wiring, "build_llm_backend", lambda *, tools=None: AlwaysRaiseBackend())
    monkeypatch.setattr(wiring, "_graph", None)
    assert wiring.get_production_graph() is wiring.get_production_graph()


@pytest.mark.parametrize("module", [ps], ids=["infra.persist_service"])
def test_entry_modules_do_not_self_assemble(module):
    """(b) 命名空间：入口模块不再自带装配入口 / 图单例 / 顶层 ``build_agent_graph``。"""
    leftover = [name for name in _FORBIDDEN_ENTRY_ATTRS if hasattr(module, name)]
    assert leftover == [], (
        f"{module.__name__} 又自带了装配件 {leftover} —— 装配应唯一位于 pra.wiring"
    )


def _build_agent_graph_call_sites() -> set[str]:
    """扫 ``src/pra/**/*.py``，返回所有 ``build_agent_graph(...)`` 调用点（repo 相对 posix 路径）。

    只认 ``Call`` 节点 —— ``src/pra/agent/graph.py`` 的 ``def build_agent_graph`` 是定义不是调用，
    自然被排除；``__all__`` 与 docstring 里的裸名同理不计。
    """
    src_root = Path(__file__).resolve().parents[1] / "src"
    sites: set[str] = set()
    for path in sorted((src_root / "pra").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            else:
                name = None
            if name == "build_agent_graph":
                sites.add(path.relative_to(src_root.parent).as_posix())
    return sites


def test_production_assembly_has_exactly_one_call_site():
    """(c) 结构契约：``src/pra`` 内 ``build_agent_graph`` 调用点恰为 ``{pra/wiring.py}``。"""
    sites = _build_agent_graph_call_sites()

    assert sites == _PRODUCTION_ASSEMBLY_SITES, (
        f"生产装配调用点应恰为 {sorted(_PRODUCTION_ASSEMBLY_SITES)}，实为 {sorted(sites)}"
        " —— 装配被复制到别处（回退到「双图单例」）"
    )
