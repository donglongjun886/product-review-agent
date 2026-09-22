"""RAG 索引装配入口：``build_policy_index`` / ``build_case_index``（chroma + hybrid 检索）。

语料固定取 ``rag/corpus/`` 静态 JSON；``embedding_model`` **必填**、本层不代建；
``pra.rag.index`` / ``chroma_store`` 在函数体内延迟 import，缺 ``--extra rag`` 时抛带指引的 ``RuntimeError``。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from pra.rag.corpus import load_cases, load_policies

if TYPE_CHECKING:  # 仅注解：本模块 import 期不拉起 tools / chroma_store
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
    """两个 builder 的唯一实现；``rows`` 显式注入时优先（跳过文件 IO），否则 ``loader()`` 读缺省语料。

    ``embedding_model`` 必填：本层不构造编码器，漏传即 ``TypeError``。
    """
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
    """构造 PolicyIndex（语料 = rag/corpus/policies.json）。

    ``rows`` 注入时优先；``embedding_model`` 必填（``BaseEmbedding``）；``config`` = Chroma 参数（None → 默认值）。
    """
    # 延迟 import：真正装配索引时才拉起 chroma / llama_index / bm25s / jieba。
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
