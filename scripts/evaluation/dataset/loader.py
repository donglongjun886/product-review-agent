"""评测集加载器：JSONL 强校验读取 + v3 双文件装配。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from evaluation.dataset.schema import EvalCase, EvalCaseV3, EvalTruthV3
from pydantic import BaseModel

__all__ = ["EvalDatasetV3", "load_cases_v3", "load_dataset", "load_dataset_v3", "load_truth_v3"]


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


# ---- v3 数据集装配：cases × truth 按 eval_case_id 1:1 join ----


@dataclass(frozen=True)
class EvalDatasetV3:
    """v3 执行层数据集视图：cases 与 truth 按 eval_case_id join 且同序对齐。"""

    cases: list[EvalCaseV3]
    truth: list[EvalTruthV3]


def _reject_old_schema_version(path: Path) -> None:
    """cases 文件首行 ``schema_version`` 预检：非 3 → 显式报旧格式（不让 pydantic 裸报错糊过去）。

    JSON 损坏 / 空文件不预检，交给 :func:`_load_jsonl` 报带行号的错误。
    """
    with path.open("r", encoding="utf-8") as fh:
        head = next((line.strip() for line in fh if line.strip()), "")
    if not head:
        return
    try:
        row = json.loads(head)
    except json.JSONDecodeError:
        return
    version = row.get("schema_version") if isinstance(row, dict) else None
    if version != 3:
        raise ValueError(
            f"评测数据 {path} 是旧格式（schema_version={version!r}）：新执行层只接受 v3 数据集"
            "（schema_version=3，cases_v3.jsonl + truth_v3.jsonl 双文件，真值独立成行），"
            "新旧数据契约不兼容、不做静默混用"
        )


def _check_unique_ids(kind: str, ids: list[str]) -> None:
    """id 唯一性校验：重复即显式失败（带重复 id 列表）。"""
    seen: set[str] = set()
    dups: list[str] = []
    for i in ids:
        if i in seen and i not in dups:
            dups.append(i)
        seen.add(i)
    if dups:
        raise ValueError(f"{kind} eval_case_id 重复: {dups}（join 要求两侧 id 均唯一）")


def load_dataset_v3(cases_path: str | Path, truth_path: str | Path) -> EvalDatasetV3:
    """读取 v3 cases+truth 双文件并按 eval_case_id join（truth 对齐回 cases 行序）。

    :raises ValueError: 旧格式（schema_version != 3）/ 一侧 id 重复 / 两侧 id 不能 1:1 对上。
    """
    cases_path = Path(cases_path)
    _reject_old_schema_version(cases_path)
    cases = load_cases_v3(cases_path)
    truth_rows = load_truth_v3(truth_path)

    _check_unique_ids("cases", [c.eval_case_id for c in cases])
    _check_unique_ids("truth", [t.eval_case_id for t in truth_rows])
    truth_index = {t.eval_case_id: t for t in truth_rows}
    case_ids = {c.eval_case_id for c in cases}
    missing = [c.eval_case_id for c in cases if c.eval_case_id not in truth_index]
    extra = [tid for tid in truth_index if tid not in case_ids]
    if missing or extra:
        raise ValueError(
            f"cases×truth join 失败：cases 有 {len(missing)} 个 id 无真值 {missing[:5]}"
            f"；truth 有 {len(extra)} 个 id 无案件 {extra[:5]}（双文件必须 1:1 对齐）"
        )
    return EvalDatasetV3(cases=cases, truth=[truth_index[c.eval_case_id] for c in cases])
