"""检索公共件 —— 模式枚举、BM25 分归一化、RRF 融合、候选集检索上下文。

``MODES`` / ``RetrievalMode`` 是检索侧对外的模式契约（``bm25`` | ``vector`` | ``hybrid``，不预设谁优）；
``normalize_minmax`` 把 BM25 原始分在**候选集内** min-max 到 [0,1]（空/等值集 → 全 1.0 防除零）；
``fuse_rrf`` 把多路排名按 RRF 融成 ``(行索引, 分)``；``_RetrievalContext`` 是候选子集上的检索上下文。
打分编排与 Top-K 由后端自持。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = ["MODES", "RetrievalMode", "fuse_rrf", "normalize_minmax"]

MODES: tuple[Literal["bm25", "vector", "hybrid"], ...] = ("bm25", "vector", "hybrid")
RetrievalMode = Literal["bm25", "vector", "hybrid"]

#: RRF 常数 k=60（与 ``QueryFusionRetriever`` 的融合同源：Cormack 2009）。
_RRF_K = 60.0


def normalize_minmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def fuse_rrf(ranked: dict[str, list[str]], row_index: dict[str, int]) -> list[tuple[int, float]]:
    """RRF（Reciprocal Rank Fusion）：``score(d) = Σ_r 1 / (k + rank_r(d))``，``k=60``。

    融合定义与 ``QueryFusionRetriever._reciprocal_rerank_fusion`` 逐条一致，但**由本模块自己
    算**：该实现在融合时会**原地改写** ``NodeWithScore.node.score``。另：该库默认
    ``num_queries=4``，会调用 ``self._llm.complete`` 生成扩展查询 —— 与本仓「检索链路零 LLM」
    的红线冲突。该库还按 ``node.hash`` 去重，而 Chroma 回读 metadata 的键序每次调用都不同
    （1.5.9），同一行在两路上算出的 ``node.hash`` 不同 → 同一文档被拆成两个、融合退化成单路
    并列。本模块用 **node id**（逐行唯一）作融合键。

    ⚠️ **上界是 ``2/60``（≈0.0333），不是 ``2/61``**：rank 从 **0** 起（``enumerate(ids)``），
    首位贡献 ``1/(60+0)``，两路都排首位即 ``2/60 = 1/30``。返回 ``[(行索引, 6 位 RRF 分)]``，
    排序 key = ``(分降序, corpus 原序 idx 升序)``。
    """
    fused: dict[str, float] = {}
    for ids in ranked.values():
        for offset, nid in enumerate(ids):
            fused[nid] = fused.get(nid, 0.0) + 1.0 / (_RRF_K + offset)
    out = [(row_index[nid], round(score, 6)) for nid, score in fused.items()]
    out.sort(key=lambda t: (-t[1], t[0]))
    return out


@dataclass
class _RetrievalContext:
    """一次检索的上下文：候选子集上的 node / node id / 行索引映射（只有检索链路会读的字段）。"""

    node_ids: list[str]
    nodes: list[Any]
    #: node_id → corpus 原序行索引（排序 tie-break 用原序，不依赖底层库返回顺序）。
    row_index_by_key: dict[str, int] = field(default_factory=dict)
