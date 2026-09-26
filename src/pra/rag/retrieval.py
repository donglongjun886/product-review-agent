"""检索公共件：候选集检索上下文 ``_RetrievalContext``。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class _RetrievalContext:
    """一次检索的上下文：候选子集上的 node / node id / 行索引（三个列表同序并行）。"""

    node_ids: list[str]
    nodes: list[Any]
    #: 与 ``node_ids`` / ``nodes`` 同序并行的 corpus 原序行索引。
    row_index: list[int] = field(default_factory=list)
