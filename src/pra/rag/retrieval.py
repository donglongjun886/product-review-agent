"""检索公共件 —— 模式枚举与候选集检索上下文。

``MODES`` / ``RetrievalMode`` 是检索侧对外的模式契约（``bm25`` | ``vector`` | ``hybrid``，不预设谁优）；
``_RetrievalContext`` 是候选子集上的检索上下文。hybrid 的融合由 ``index`` 层交给
``QueryFusionRetriever(mode=RECIPROCAL_RANK)``，本模块不含融合实现。
各路分数**不做量纲适配**（三模式量纲互不可比），由后端自持。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

__all__ = ["MODES", "RetrievalMode"]

MODES: tuple[Literal["bm25", "vector", "hybrid"], ...] = ("bm25", "vector", "hybrid")
RetrievalMode = Literal["bm25", "vector", "hybrid"]


@dataclass
class _RetrievalContext:
    """一次检索的上下文：候选子集上的 node / node id / 行索引映射（只有检索链路会读的字段）。"""

    node_ids: list[str]
    nodes: list[Any]
    #: node_id → corpus 原序行索引（排序 tie-break 用原序，不依赖底层库返回顺序）。
    row_index_by_key: dict[str, int] = field(default_factory=dict)
