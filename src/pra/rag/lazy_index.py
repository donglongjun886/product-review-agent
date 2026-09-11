"""惰性索引代理 —— 把「构建 RAG 索引」推迟到首次检索。

为什么需要：``build_production_tools()`` 是图装配期调用的一次性动作，而真实后端（chroma +
BGE）的索引构造是重活 —— import chromadb / llama_index、加载语义模型、对全语料编码、连服务端
建库 upsert。装配期就做会把「装配」变成带副作用且可能直接抛错的动作（服务未起 / 模型未缓存 /
未装 ``--extra rag``），也会让生产工具世界的装配在 CI 上不可调用。

语义：
- 构造只记 builder（**零 import、零 IO**）；首次 ``search`` 才调用它，之后复用同一实例；
- 构建失败**不缓存** —— 下一次 ``search`` 重试（服务端短暂不可达可自愈）；异常原样上抛，
  由 ``tools_node`` 记 warn failure，**绝不静默回退 InMemory 种子**；
- 构建是同步的、到 builder 返回前没有 ``await``，故同一事件循环内的并发 ``search`` 不会交错
  构建（与 ``pra.api.service.get_graph`` 的懒加载同一论证）。跨线程 / 跨事件循环并发不在本
  代理的保证范围内。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # 仅注解：本模块 import 期不拉起 tools 子包
    from pra.tools.case_search.tool import CaseHit, CaseIndex, CaseSearchFilters
    from pra.tools.policy_search.tool import (
        PolicyClauseHit,
        PolicyIndex,
        PolicySearchFilters,
    )

__all__ = ["LazyCaseIndex", "LazyPolicyIndex"]


class _LazyIndex:
    """共享的「构造期不建、首次用到才建」骨架（``LazyPolicyIndex`` / ``LazyCaseIndex`` 的基类）。"""

    def __init__(self, builder: Callable[[], Any], *, label: str) -> None:
        self._builder = builder
        self._label = label
        self._index: Any | None = None

    @property
    def is_built(self) -> bool:
        """是否已构建（测试/观测用；构建成功后恒 True）。"""
        return self._index is not None

    @property
    def index(self) -> Any | None:
        """已构建的底层索引；未构建时为 None（**不触发**构建）。"""
        return self._index

    def _resolve(self) -> Any:
        if self._index is None:
            self._index = self._builder()
        return self._index


class LazyPolicyIndex(_LazyIndex):
    """``PolicyIndex`` 的惰性代理：``builder`` 为无参可调用，返回任意实现该 Protocol 的索引。"""

    def __init__(self, builder: Callable[[], PolicyIndex], *, label: str = "PolicyIndex") -> None:
        super().__init__(builder, label=label)

    async def search(
        self,
        query: str,
        filters: PolicySearchFilters,
        top_k: int,
        effective_only: bool,
    ) -> list[PolicyClauseHit]:
        return await self._resolve().search(query, filters, top_k, effective_only)


class LazyCaseIndex(_LazyIndex):
    """``CaseIndex`` 的惰性代理：``builder`` 为无参可调用，返回任意实现该 Protocol 的索引。"""

    def __init__(self, builder: Callable[[], CaseIndex], *, label: str = "CaseIndex") -> None:
        super().__init__(builder, label=label)

    async def search(
        self, query: str, filters: CaseSearchFilters, top_k: int
    ) -> list[CaseHit]:
        return await self._resolve().search(query, filters, top_k)
