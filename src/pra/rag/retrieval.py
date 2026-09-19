"""检索公共口径 —— 模式枚举与 BM25 分归一化。

``MODES`` / ``RetrievalMode`` 是检索侧对外的模式契约（``bm25`` | ``vector`` | ``hybrid``，不预设谁优）；
``normalize_minmax`` 把 BM25 原始分在**候选集内** min-max 到 [0,1]（空/等值集 → 全 1.0 防除零），
供 chroma 后端的 ``bm25`` 模式使用。打分编排与 Top-K 由后端自持。
"""

from __future__ import annotations

from typing import Literal

__all__ = [
    "MODES",
    "normalize_minmax",
]

MODES: tuple[Literal["bm25", "vector", "hybrid"], ...] = ("bm25", "vector", "hybrid")
RetrievalMode = Literal["bm25", "vector", "hybrid"]


def normalize_minmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]
