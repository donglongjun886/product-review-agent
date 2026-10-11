"""评测集加载器：JSONL 强校验读取。"""

from __future__ import annotations

from pathlib import Path

from evaluation.dataset.schema import EvalCase, EvalCaseV3, EvalTruthV3
from pydantic import BaseModel

__all__ = ["load_cases_v3", "load_dataset", "load_truth_v3"]


def _load_jsonl[T: BaseModel](path: str | Path, model: type[T]) -> list[T]:
    """读取 JSONL 并按 ``model`` 强校验；返回按文件行序的列表（稳定序）。

    :raises ValueError: 文件缺失 / 空文件 / 某行 JSON 损坏或 schema 校验失败（带行号）。
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"评测集文件不存在: {p}")
    rows: list[T] = []
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(model.model_validate_json(stripped))
            except Exception as exc:  # ValueError(JSON) / ValidationError(schema)
                raise ValueError(f"评测集第 {lineno} 行解析失败（含行号定位）: {exc}") from exc
    if not rows:
        raise ValueError(f"评测集为空（无有效行）: {p}")
    return rows


def load_dataset(path: str | Path) -> list[EvalCase]:
    """读取 v1/v2 JSONL 评测集（EvalCase 契约）。"""
    return _load_jsonl(path, EvalCase)


def load_cases_v3(path: str | Path) -> list[EvalCaseV3]:
    """读取 v3 案件文件（cases_v3.jsonl，EvalCaseV3 契约）。"""
    return _load_jsonl(path, EvalCaseV3)


def load_truth_v3(path: str | Path) -> list[EvalTruthV3]:
    """读取 v3 真值文件（truth_v3.jsonl，EvalTruthV3 契约）。"""
    return _load_jsonl(path, EvalTruthV3)
