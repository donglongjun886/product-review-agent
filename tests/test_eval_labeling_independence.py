"""独立性 seam：labeling 包不得 import 生成器模块，也不得 import 生产代码（规则书自带冻结词表副本）。"""

from __future__ import annotations

import ast
from pathlib import Path

LABELING_DIR = Path(__file__).resolve().parents[1] / "scripts" / "evaluation" / "labeling"


def _imported_modules(path: Path) -> list[str]:
    """收集一个 .py 文件全部 import 的模块名（静态解析，不执行）。"""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return names


def test_labeling_does_not_import_generator_or_production() -> None:
    """打标函数与生成器/生产代码解耦：只许依赖 dataset 契约与标准库。"""
    py_files = sorted(LABELING_DIR.glob("*.py"))
    assert py_files, "labeling 包不存在"
    for path in py_files:
        for name in _imported_modules(path):
            top = name.split(".")[0]
            assert not name.startswith("evaluation.generator"), f"{path.name} import 了生成器: {name}"
            assert top != "pra", f"{path.name} import 了生产代码: {name}（规则书须自带冻结词表副本）"
