"""检索编排（rag/retrieval.py）—— 三模式打分 + 融合 + Top-K（确定性）。

对应《00》§6.3 / rag-implementation-plan.md §4.3 的检索流程（MVP 无模型 reranker，
R-6 拍板）：

1. **元数据过滤先行**（category / risk_type / status 等）：由调用方（rag/index.py）
   在入参前完成 —— 本模块只对**候选文档位**做打分与排序；
2. **三模式打分**（mode="bm25" | "vector" | "hybrid"，可切换、不预设谁优）：
   - bm25：自写 Okapi BM25（rag/bm25.py）原始分；
   - vector：query embedding 与 doc embedding 的余弦（rag/vectors.py）；
   - hybrid：归一化加权融合 —— 默认权重 BM25:Vector = 0.5 : 0.5（可配），
     ``fused = w_bm25 · norm(bm25) + w_vec · cosine``；
3. **Top-K**：融合分降序截断（确定性 tie-break：同分按 corpus 原序，无随机）。

归一化口径：bm25 原始分在**候选集内 min-max 到 [0,1]**（候选全集 = 1.0，空/等值集
= 1.0 防除零）—— 保证三模式分数同量纲、可并排比较；vector 余弦天然 [0,1]。
返回元素带 ``score``（该模式最终分，0~1）与 ``details``（bm25_raw / bm25_norm /
vector / fused 原始值，供报告与调试）—— case 索引据此写 ``CaseHit.similarity``。
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
    """min-max 归一化到 [0,1]；空/全等值集 → 全 1.0（无区分度时给满分的确定性约定）。"""
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

    输入长度须一致；``normalize=True`` 先把 bm25 分 min-max 归一化（向量分要求已
    在 [0,1]，如余弦）再融合。权重默认 0.5/0.5；改权重的单测在 tests/test_rag.py
    （构造两极值验证权重生效，R-6：不预设 Hybrid 最优，权重是实验变量）。
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
    """单条排序结果：corpus 行索引 + 最终分（0~1）+ 三路分解值（报告/调试用）。"""

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

    :param texts: 与 ``bm25`` 对齐的全库检索文本（取子集喂给 candidate 索引）。
    :param candidates: 候选文档**行索引**（元数据过滤后的子集）；None = 全库。
    :param top_k: 截断数（>=1）。
    :param doc_vectors: 可选预计算 doc 向量（与 texts 对齐）；None → 本函数按需 embed。
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
