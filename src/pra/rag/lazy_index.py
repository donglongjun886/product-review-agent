"""惰性索引代理：构造期只记 ``builder``，首次 ``search`` 才建索引并复用；构建失败不缓存、异常原样上抛。"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pra.tools.case_search.tool import CaseHit, CaseSearchFilters
    from pra.tools.policy_search.tool import PolicyClauseHit, PolicySearchFilters

__all__ = ["LazyCaseIndex", "LazyPolicyIndex"]


class _LazyIndex:
    """惰性索引代理基类（``LazyPolicyIndex`` / ``LazyCaseIndex`` 共享）。"""

    def __init__(self, builder: Callable[[], Any]) -> None:
        self._builder = builder
        self._index: Any | None = None

    def _resolve(self) -> Any:
        if self._index is None:
            self._index = self._builder()
        return self._index


class LazyPolicyIndex(_LazyIndex):
    """``PolicyIndex`` 的惰性代理；``builder`` 为无参可调用。"""

    async def search(
        self,
        query: str,
        filters: PolicySearchFilters,
        top_k: int,
        effective_only: bool,
    ) -> list[PolicyClauseHit]:
        return await self._resolve().search(query, filters, top_k, effective_only)


class LazyCaseIndex(_LazyIndex):
    """``CaseIndex`` 的惰性代理；``builder`` 为无参可调用。"""

    async def search(
        self, query: str, filters: CaseSearchFilters, top_k: int
    ) -> list[CaseHit]:
        return await self._resolve().search(query, filters, top_k)
