"""检索公共件 —— 候选集检索上下文。

生产检索口径**恒为 hybrid**（BM25 + Vector + RRF 融合）；BM25 / Vector 是 hybrid 的两条内部
路径（``index._rank_bm25`` / ``index._rank_vector``），不存在可切换的对外模式开关。
hybrid 的融合由 ``index`` 层交给 ``QueryFusionRetriever(mode=RECIPROCAL_RANK)``，本模块不含融合实现。
两路分数**不做量纲适配**（量纲互不可比），由后端自持。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class _RetrievalContext:
    """一次检索的上下文：候选子集上的 node / node id / 行索引（三个列表同序并行）。"""

    node_ids: list[str]
    nodes: list[Any]
    #: 与 ``node_ids`` / ``nodes`` 同序并行的 corpus 原序行索引（排序 tie-break 用原序）。
    row_index: list[int] = field(default_factory=list)
