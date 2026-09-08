"""真实 PolicyIndex / CaseIndex 实现（rag/index.py）—— 替换 InMemory 的注入点。

对齐工具契约（tools/{policy_search,case_search}/tool.py 的 Protocol），**Tool 层零改动**：
- ``RagPolicyIndex`` 实现 ``PolicyIndex.search(query, filters, top_k, effective_only)``，
  返回 ``list[PolicyClauseHit]``（与 InMemoryPolicyIndex 同型，可 model_validate）；
- ``RagCaseIndex`` 实现 ``CaseIndex.search(query, filters, top_k)``，
  返回 ``list[CaseHit]``（similarity 为检索期融合分 0~1，同 InMemory 的种子相似度
  语义 —— 排序 + 证据 weight 用）。

与 InMemory 的差异（这正是 RAG 的意义）：query **参与匹配** —— 元数据过滤
（category/risk_type/status，语义与 InMemory 对齐）→ 三模式打分（bm25 / vector /
hybrid，可切换）→ Top-K。embedding 经 ``Embedder`` provider 注入（MVP =
MockHashEmbedder，确定性 mock；Phase 2 换本地模型 + Qdrant 不动本类接口）。

构造约定：``rows`` 为 corpus 记录（dict 或 schema 模型均可，构造期强校验）；
检索文本（policy：title+text；case：summary）与 embedding/BM25 在构造期一次性
建好（确定性、离线）。搜索是确定性纯计算（async 包装以对齐工具 Protocol）。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pra.rag.bm25 import BM25Index
from pra.rag.corpus.schema import CasePrecedentRecord, PolicyClauseRecord
from pra.rag.embedder import Embedder, MockHashEmbedder
from pra.rag.retrieval import (
    DEFAULT_WEIGHTS,
    MODES,
    RetrievalMode,
    rank_documents,
)
from pra.tools.case_search.tool import CaseHit, CaseSearchFilters
from pra.tools.policy_search.tool import (
    PolicyClauseHit,
    PolicySearchFilters,
)

__all__ = ["RagCaseIndex", "RagPolicyIndex"]

_FULL_CATEGORY = "全类目"


def _validate_mode(mode: str) -> RetrievalMode:
    if mode not in MODES:
        raise ValueError(f"未知检索模式: {mode!r}（可选: {list(MODES)}）")
    return mode  # type: ignore[return-value]


def _normalize_rows(rows: Iterable[Any], record_type: type) -> list[Any]:
    """rows（dict 或 record 模型）→ 强校验的 record 列表（构造期防脏数据）。"""
    out: list[Any] = []
    for row in rows:
        out.append(record_type.model_validate(row) if isinstance(row, dict) else row)
    for row in out:
        if not isinstance(row, record_type):
            raise TypeError(f"corpus 行须为 {record_type.__name__} 或 dict: got {type(row)!r}")
    return out


class RagPolicyIndex:
    """PolicyIndex 的真实实现：版本/元数据过滤 + 三模式检索（Policy KB）。"""

    def __init__(
        self,
        rows: Iterable[dict | PolicyClauseRecord],
        *,
        embedder: Embedder | None = None,
        mode: RetrievalMode = "hybrid",
        weights: tuple[float, float] = DEFAULT_WEIGHTS,
    ) -> None:
        self._rows: list[PolicyClauseRecord] = _normalize_rows(rows, PolicyClauseRecord)
        self.mode: RetrievalMode = _validate_mode(mode)
        self.weights: tuple[float, float] = tuple(weights)
        self.embedder: Embedder = embedder or MockHashEmbedder()
        self._texts: list[str] = [
            f"{r.title}。{r.text}" for r in self._rows  # title+text 均为检索文本
        ]
        self._bm25 = BM25Index(self._texts)
        self._doc_vectors: list[list[float]] = [
            self.embedder.embed(t) for t in self._texts
        ]

    @property
    def size(self) -> int:
        """Policy KB 条款数（含 EXPIRED 历史版）。"""
        return len(self._rows)

    def effective_count(self) -> int:
        return sum(1 for r in self._rows if r.status == "EFFECTIVE")

    async def search(
        self,
        query: str,
        filters: PolicySearchFilters,
        top_k: int,
        effective_only: bool,
    ) -> list[PolicyClauseHit]:
        """检索当前政策条款（query 参与匹配；语义与 InMemory 对齐，见模块 docstring）。"""
        candidates: list[int] = []
        for i, r in enumerate(self._rows):
            if effective_only and r.status != "EFFECTIVE":
                continue
            if filters.category and r.category not in (None, filters.category, _FULL_CATEGORY):
                continue
            if filters.risk_type:
                wanted = set(filters.risk_type)
                if not (wanted & set(r.risk_type)):
                    continue
            candidates.append(i)

        ranked = rank_documents(
            query=query,
            texts=self._texts,
            embedder=self.embedder,
            bm25=self._bm25,
            mode=self.mode,
            weights=self.weights,
            candidates=candidates,
            top_k=top_k,
            doc_vectors=self._doc_vectors,
        )
        return [
            PolicyClauseHit.model_validate(self._rows[h.index].model_dump(mode="json"))
            for h in ranked
        ]


class RagCaseIndex:
    """CaseIndex 的真实实现：元数据过滤 + 三模式检索（Case KB，先例）。"""

    def __init__(
        self,
        rows: Iterable[dict | CasePrecedentRecord],
        *,
        embedder: Embedder | None = None,
        mode: RetrievalMode = "hybrid",
        weights: tuple[float, float] = DEFAULT_WEIGHTS,
    ) -> None:
        self._rows: list[CasePrecedentRecord] = _normalize_rows(rows, CasePrecedentRecord)
        self.mode: RetrievalMode = _validate_mode(mode)
        self.weights: tuple[float, float] = tuple(weights)
        self.embedder: Embedder = embedder or MockHashEmbedder()
        self._texts: list[str] = [r.summary for r in self._rows]
        self._bm25 = BM25Index(self._texts)
        self._doc_vectors: list[list[float]] = [
            self.embedder.embed(t) for t in self._texts
        ]

    @property
    def size(self) -> int:
        """Case KB 先例数。"""
        return len(self._rows)

    async def search(
        self, query: str, filters: CaseSearchFilters, top_k: int
    ) -> list[CaseHit]:
        """检索先例（query 参与匹配；元数据过滤语义与 InMemory 对齐：category 精确 /
        risk_type 交叠；无命中返回空列表 —— 工具 ok=True）。"""
        candidates: list[int] = []
        for i, r in enumerate(self._rows):
            if filters.category and r.category != filters.category:
                continue
            if filters.risk_type:
                wanted = set(filters.risk_type)
                if not (wanted & set(r.risk_type)):
                    continue
            candidates.append(i)

        ranked = rank_documents(
            query=query,
            texts=self._texts,
            embedder=self.embedder,
            bm25=self._bm25,
            mode=self.mode,
            weights=self.weights,
            candidates=candidates,
            top_k=top_k,
            doc_vectors=self._doc_vectors,
        )
        hits: list[CaseHit] = []
        for h in ranked:
            row = self._rows[h.index]
            hits.append(
                CaseHit.model_validate(
                    {
                        **row.model_dump(mode="json"),
                        "similarity": h.score,  # 检索期融合分 → CaseHit.similarity
                    }
                )
            )
        return hits
