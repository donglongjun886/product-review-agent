"""Policy KB / Case KB 静态 corpus 加载器：读 JSON → 信封校验 → 返回 ``(records, meta)``。

数据文件是可评审的 git 入库 JSON，schema 见 ``schema.py``。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pra.rag.corpus.schema import (
    CaseCorpus,
    CasePrecedentRecord,
    PolicyClauseRecord,
    PolicyCorpus,
)

__all__ = [
    "CORPUS_DIR",
    "load_cases",
    "load_policies",
]

CORPUS_DIR = Path(__file__).resolve().parent
_POLICIES_FILE = CORPUS_DIR / "policies.json"
_CASES_FILE = CORPUS_DIR / "cases.json"


def _load_envelope(path: Path, model_type: type) -> tuple[list[Any], dict]:
    if not path.exists():
        raise ValueError(f"corpus 数据文件缺失: {path}（请先跑 scripts/build_rag_corpus.py 重建）")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ValueError(f"corpus 数据文件解析失败: {path}: {exc}") from exc
    envelope = model_type.model_validate(raw)
    return list(envelope.policies if model_type is PolicyCorpus else envelope.cases), dict(
        envelope.meta
    )


def load_policies() -> tuple[list[PolicyClauseRecord], dict]:
    return _load_envelope(_POLICIES_FILE, PolicyCorpus)


def load_cases() -> tuple[list[CasePrecedentRecord], dict]:
    return _load_envelope(_CASES_FILE, CaseCorpus)
