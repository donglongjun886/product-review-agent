"""评测入口脚本共享的 CLI 小件：dotenv 注入 / v3 数据路径解析 / 比率渲染 / JSON 落盘。

run_evaluation 与 cross_check_labels 两个入口共用同一实现（同一逻辑只保留一处）。
"""

from __future__ import annotations

import json
import os
from collections.abc import Collection
from pathlib import Path

__all__ = ["load_dotenv_keys", "ratio_cell", "resolve_v3_data_paths", "write_json_file"]


def load_dotenv_keys(key_names: Collection[str]) -> None:
    """把仓库根 ``.env`` 的指定键注入进程环境（setdefault 语义；找不到或键缺失 → 跳过）。"""
    here = Path(__file__).resolve()
    env_file: Path | None = None
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            env_file = parent / ".env"
            break
    if env_file is None or not env_file.is_file():
        return
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key in key_names and value:
            os.environ.setdefault(key, value)


def resolve_v3_data_paths(raw: str) -> tuple[Path, Path]:
    """把 ``--data`` 解析为 v3 双文件 ``(cases, truth)``：v3 目录，或 ``cases_v3.jsonl``
    文件（同目录自动配对 ``truth_v3.jsonl``）。v2 目录 / 旧格式文件 → 显式报错。"""
    p = Path(raw)
    if p.is_dir():
        cases = p / "cases_v3.jsonl"
        truth = p / "truth_v3.jsonl"
        if cases.is_file() and truth.is_file():
            return cases, truth
        if (p / "cases_v2.jsonl").is_file():
            raise ValueError(
                f"评测数据目录 {p} 是 v2 旧格式：新执行层只接受 v3 双文件"
                "（cases_v3.jsonl + truth_v3.jsonl），新旧数据契约不兼容、不做静默混用；"
                "请改用 eval_data/v3"
            )
        raise ValueError(
            f"评测数据目录 {p} 缺少 v3 双文件 cases_v3.jsonl / truth_v3.jsonl"
        )
    if p.is_file():
        if p.name == "cases_v3.jsonl":
            truth = p.with_name("truth_v3.jsonl")
            if not truth.is_file():
                raise ValueError(f"v3 cases 文件 {p} 缺少同目录 truth_v3.jsonl（v3 为双文件契约）")
            return p, truth
        raise ValueError(
            f"评测数据文件 {p} 是旧格式（非 cases_v3.jsonl）：新执行层只接受 v3 数据集，"
            "请传 eval_data/v3 目录或 cases_v3.jsonl 文件；v1/v2 单文件契约不兼容、不做静默混用"
        )
    raise ValueError(f"评测数据路径不存在: {p}（支持 eval_data/v3 目录，或 cases_v3.jsonl 文件）")


def ratio_cell(value: float | None, numer: int, denom: int) -> str:
    """比率文本 ``0.950(285/300)``；分母为 0（value 为 None）→ ``-``。"""
    if value is None:
        return "-"
    return f"{value:.3f}({numer}/{denom})"


def write_json_file(path: Path, payload: object) -> None:
    """把 JSON 写入 ``path``（父目录须已存在，缺失即报错）。"""
    if not path.parent.exists():
        raise ValueError(f"输出父目录不存在: {path.parent}（请先创建目录）")
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
