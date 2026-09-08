"""RAG 知识库数据包（rag/corpus/）—— Policy KB / Case KB 静态 corpus。

结构：
- ``schema.py``：数据契约 + Pydantic 强校验（见其 docstring：来源与隔离声明）；
- ``policies.json`` / ``cases.json``：静态数据（git 入库、可评审；
  由 scripts/build_rag_corpus.py 确定性生成/重建，幂等）；
- 本模块：加载器 —— 读 JSON → 信封校验 → 返回（records, meta）。

设计要点：数据是**可评审的静态文件**（R-5：MVP 不落 DB）；运行时索引
（BM25 / embedding）由 rag/factory.py 在加载之上构建，数据文件本身不含派生索引。
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
    """读取 corpus JSON 信封并强校验；返回 (records, meta)。损坏即报错（不静默）。"""
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


def load_policies(path: str | Path | None = None) -> tuple[list[PolicyClauseRecord], dict]:
    """读取 Policy KB → (clause records, meta)。``path`` 缺省取包内 policies.json。"""
    return _load_envelope(Path(path) if path is not None else _POLICIES_FILE, PolicyCorpus)


def load_cases(path: str | Path | None = None) -> tuple[list[CasePrecedentRecord], dict]:
    """读取 Case KB → (precedent records, meta)。``path`` 缺省取包内 cases.json。"""
    return _load_envelope(Path(path) if path is not None else _CASES_FILE, CaseCorpus)
