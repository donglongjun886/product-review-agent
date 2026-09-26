"""RAG 索引装配入口：``build_policy_index`` / ``build_case_index``（chroma + hybrid 检索）。"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from pra.rag.corpus import load_cases, load_policies

if TYPE_CHECKING:
    from pra.rag.chroma_store import ChromaConfig
    from pra.tools.case_search.tool import CaseIndex
    from pra.tools.policy_search.tool import PolicyIndex

__all__ = [
    "build_case_index",
    "build_policy_index",
]


def _build_index(
    index_cls: type,
    *,
    loader: Callable[[], tuple[list[Any], dict]],
    embedding_model: Any,
    rows: Iterable[Any] | None = None,
    config: ChromaConfig | None = None,
) -> Any:
    """两个 builder 的唯一实现；``rows`` 注入优先，未注入时由 ``loader()`` 读缺省语料。"""
    if rows is None:
        record_rows, _meta = loader()
    else:
        record_rows = list(rows)
    return index_cls(
        record_rows, embedding_model=embedding_model, config=config
    )


def build_policy_index(
    rows: Iterable[Any] | None = None,
    *,
    embedding_model: Any,
    config: ChromaConfig | None = None,
) -> PolicyIndex:
    """构造 PolicyIndex（语料 = rag/corpus/policies.json；``rows`` 注入时优先）。"""
    from pra.rag.index import ChromaPolicyIndex

    return _build_index(
        ChromaPolicyIndex,
        loader=load_policies,
        rows=rows,
        embedding_model=embedding_model,
        config=config,
    )


def build_case_index(
    rows: Iterable[Any] | None = None,
    *,
    embedding_model: Any,
    config: ChromaConfig | None = None,
) -> CaseIndex:
    """构造 CaseIndex（语料 = rag/corpus/cases.json）；参数语义同 ``build_policy_index``。"""
    from pra.rag.index import ChromaCaseIndex

    return _build_index(
        ChromaCaseIndex,
        loader=load_cases,
        rows=rows,
        embedding_model=embedding_model,
        config=config,
    )
