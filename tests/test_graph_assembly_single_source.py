"""装配唯一处契约：生产图只在组合根 ``pra.wiring`` 组装一次。"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from helpers import AlwaysRaiseBackend

from pra import wiring
from pra.infra import persist_service as ps

_PRODUCTION_ASSEMBLY_SITES = {"src/pra/wiring.py"}

_FORBIDDEN_ENTRY_ATTRS = (
    "get_graph",
    "_get_graph",
    "_graph",
    "_compiled_graph",
    "build_agent_graph",
)


def test_get_production_graph_is_a_process_singleton(monkeypatch):
    """连续两次取图是同一实例。"""
    monkeypatch.setattr(wiring, "build_llm_backend", lambda *, tools=None: AlwaysRaiseBackend())
    monkeypatch.setattr(wiring, "_graph", None)
    assert wiring.get_production_graph() is wiring.get_production_graph()


@pytest.mark.parametrize("module", [ps], ids=["infra.persist_service"])
def test_entry_modules_do_not_self_assemble(module):
    """入口模块不自带装配入口 / 图单例 / 顶层 ``build_agent_graph``。"""
    leftover = [name for name in _FORBIDDEN_ENTRY_ATTRS if hasattr(module, name)]
    assert leftover == [], (
        f"{module.__name__} 又自带了装配件 {leftover} —— 装配应唯一位于 pra.wiring"
    )


def _build_agent_graph_call_sites() -> set[str]:
    """扫 ``src/pra/**/*.py``，返回所有 ``build_agent_graph(...)`` 调用点（repo 相对 posix 路径）。"""
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
    """``src/pra`` 内 ``build_agent_graph`` 调用点恰为 ``{pra/wiring.py}``。"""
    sites = _build_agent_graph_call_sites()

    assert sites == _PRODUCTION_ASSEMBLY_SITES, (
        f"生产装配调用点应恰为 {sorted(_PRODUCTION_ASSEMBLY_SITES)}，实为 {sorted(sites)}"
        " —— 装配被复制到别处（回退到「双图单例」）"
    )
