"""检索编排 —— 三模式打分 + 融合 + Top-K（确定性）。

1. **元数据过滤先行**（category / risk_type / status）：由调用方在入参前完成 —— 本模块只对
   **候选文档位**打分与排序；
2. **三模式打分**（``bm25`` | ``vector`` | ``hybrid``，不预设谁优）：bm25 = 自写 Okapi BM25
   原始分；vector = query 与 doc embedding 的余弦；hybrid = ``w_bm25 · norm(bm25) +
   w_vec · cosine``（默认 0.5 : 0.5）；
3. **Top-K**：融合分降序截断（同分按 corpus 原序，无随机）。

归一化口径：bm25 原始分在**候选集内 min-max 到 [0,1]**（空/等值集 → 全 1.0 防除零），保证三
模式同量纲；vector 余弦天然 [0,1]。返回元素带 ``score``（最终分）与 ``details``（各路原始值）；
case 索引据此写 ``CaseHit.retrieval_score``（检索分，非语义相似度）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pra.rag.bm25 import BM25Index, tokenize
from pra.rag.embedder import Embedder
from pra.rag.vectors import cosine_similarity

__all__ = [
    "DEFAULT_WEIGHTS",
    "MODES",
    "RankedHit",
    "fuse_scores",
    "normalize_minmax",
    "rank_documents",
]

MODES: tuple[Literal["bm25", "vector", "hybrid"], ...] = ("bm25", "vector", "hybrid")
RetrievalMode = Literal["bm25", "vector", "hybrid"]

DEFAULT_WEIGHTS: tuple[float, float] = (0.5, 0.5)  # (w_bm25, w_vector)


def normalize_minmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def fuse_scores(
    bm25_scores: list[float],
    vector_scores: list[float],
    weights: tuple[float, float] = DEFAULT_WEIGHTS,
    *,
    normalize: bool = True,
) -> list[float]:
    """BM25 分与向量分加权融合（hybrid 路）。

    输入长度须一致；``normalize=True`` 先把 bm25 分 min-max 归一化（向量分要求已在 [0,1]，
    如余弦）再融合。
    """
    if len(bm25_scores) != len(vector_scores):
        raise ValueError(
            f"融合要求两路分数等长: {len(bm25_scores)} != {len(vector_scores)}"
        )
    w_bm25, w_vec = weights
    bm25_norm = normalize_minmax(bm25_scores) if normalize else list(bm25_scores)
    return [w_bm25 * b + w_vec * v for b, v in zip(bm25_norm, vector_scores)]


@dataclass
class RankedHit:

    index: int
    score: float
    details: dict


def rank_documents(
    *,
    query: str,
    texts: list[str],
    embedder: Embedder,
    bm25: BM25Index,
    mode: RetrievalMode = "hybrid",
    weights: tuple[float, float] = DEFAULT_WEIGHTS,
    candidates: list[int] | None = None,
    top_k: int = 5,
    doc_vectors: list[list[float]] | None = None,
) -> list[RankedHit]:
    """对候选文档打分并返回 Top-K（三模式可切换；确定性无随机）。

    :param texts: 与 ``bm25`` 对齐的全库检索文本。:param candidates: 候选文档**行索引**，
        None = 全库。:param top_k: 截断数（>=1）。:param doc_vectors: 可选预计算 doc 向量，
        None → 本函数按需 embed。
    """

    if mode not in MODES:
        raise ValueError(f"未知检索模式: {mode!r}（可选: {list(MODES)}）")
    idxs = list(range(bm25.doc_count)) if candidates is None else list(candidates)
    if top_k < 1 or not idxs:
        return []

    query_vec = embedder.embed(query)
    bm25_raw = bm25.scores(tokenize(query), idxs)
    bm25_norm = normalize_minmax(bm25_raw)
    if doc_vectors is not None:
        vec_scores = [cosine_similarity(query_vec, doc_vectors[i]) for i in idxs]
    else:
        vec_scores = [cosine_similarity(query_vec, embedder.embed(texts[i])) for i in idxs]

    if mode == "bm25":
        final = bm25_norm
    elif mode == "vector":
        final = vec_scores
    else:  # hybrid
        final = fuse_scores(bm25_raw, vec_scores, weights=weights, normalize=True)

    hits = [
        RankedHit(
            index=idx,
            score=round(s, 6),
            details={
                "bm25_raw": round(bm25_raw[k], 6),
                "bm25_norm": round(bm25_norm[k], 6),
                "vector": round(vec_scores[k], 6),
                "fused": round(final[k], 6),
            },
        )
        for k, (idx, s) in enumerate(zip(idxs, final))
    ]
    # 确定性排序：分降序；同分按 corpus 原序（idx 升序）—— 无随机、可重放。
    hits.sort(key=lambda h: (-h.score, h.index))
    return hits[:top_k]
