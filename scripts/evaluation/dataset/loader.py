"""评测集加载器：JSONL 强校验读取。"""

from __future__ import annotations

from pathlib import Path

from evaluation.dataset.schema import EvalCase

__all__ = ["load_dataset"]


def load_dataset(path: str | Path) -> list[EvalCase]:
    """读取 JSONL 评测集并强校验；返回按文件行序的 EvalCase 列表（稳定序）。

    :raises ValueError: 文件缺失 / 空文件 / 某行 JSON 损坏或 schema 校验失败（带行号）。
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"评测集文件不存在: {p}")
    cases: list[EvalCase] = []
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                case = EvalCase.model_validate_json(stripped)
            except Exception as exc:  # ValueError(JSON) / ValidationError(schema)
                raise ValueError(f"评测集第 {lineno} 行解析失败（含行号定位）: {exc}") from exc
            cases.append(case)
    if not cases:
        raise ValueError(f"评测集为空（无有效 case 行）: {p}")
    return cases
