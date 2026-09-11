"""真实 PolicyIndex / CaseIndex 实现 —— 替换 InMemory 的注入点。

``RagPolicyIndex`` / ``RagCaseIndex`` 分别实现 tools 层 Policy / Case 检索 Protocol；返回的检索
分供排序与证据 weight 用，不是语义相似度。与 InMemory 的差异：query **参与匹配** —— 元数据过滤
（category / risk_type / status，语义与 InMemory 对齐）→ 三模式打分 → Top-K；embedding 经
``Embedder`` provider 注入（缺省确定性 mock）。

构造期强校验 ``rows``（dict 或 schema 模型均可），检索文本与 embedding/BM25 一次性建好；搜索是
确定性纯计算。
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

    out: list[Any] = []
    for row in rows:
        out.append(record_type.model_validate(row) if isinstance(row, dict) else row)
    for row in out:
        if not isinstance(row, record_type):
            raise TypeError(f"corpus 行须为 {record_type.__name__} 或 dict: got {type(row)!r}")
    return out


class RagPolicyIndex:

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
                        "retrieval_score": h.score,  # 检索期融合分 → CaseHit.retrieval_score
                    }
                )
            )
        return hits
