"""RAG 索引装配 —— ``build_policy_index`` / ``build_case_index``。

注入点：corpus 路径（缺省取 rag/corpus/ 内静态 JSON，失败即报错不静默）；``embedding_model``
（LlamaIndex ``BaseEmbedding``，缺省 None → chroma 类内自建 fastembed 集成）；mode（三模式可切）；
``config``（``ChromaConfig``：client / host / port / ephemeral / collection 前缀，缺省全取默认值）。

两个 builder 只在「行类型 + corpus 加载器 + 缺省语料文件」上有别，实现只有 ``_build_index`` 一份
（corpus_path 缺省由加载器自己兜底）。

检索后端只有 chroma（ChromaDB cosine + LlamaIndex 检索器 + RRF），``chroma_backend`` 在函数体内
**延迟 import**，故本模块被 tools 层 import 时顶层零额外依赖；缺 ``--extra rag`` 时抛出带指引的
``RuntimeError``，不静默降级。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pra.rag.corpus import CORPUS_DIR, load_cases, load_policies
from pra.rag.retrieval import RetrievalMode

if TYPE_CHECKING:  # 仅注解：本模块 import 期不拉起 tools / chroma_backend
    from pra.rag.chroma_backend import ChromaConfig
    from pra.tools.case_search.tool import CaseIndex
    from pra.tools.policy_search.tool import PolicyIndex

__all__ = [
    "CASES_DEFAULT",
    "POLICIES_DEFAULT",
    "build_case_index",
    "build_policy_index",
]

POLICIES_DEFAULT = CORPUS_DIR / "policies.json"
CASES_DEFAULT = CORPUS_DIR / "cases.json"


def _build_index(
    index_cls: type,
    *,
    loader: Callable[[str | Path | None], tuple[list[Any], dict]],
    embedding_model: Any,
    corpus_path: str | Path | None = None,
    rows: Iterable[Any] | None = None,
    mode: RetrievalMode = "hybrid",
    config: ChromaConfig | None = None,
) -> Any:
    """两个公开 builder 的唯一实现；差异只在调用方传进来的 ``index_cls`` / ``loader``。

    ``rows`` 显式注入时优先（跳过文件 IO，``corpus_path`` 随之失效）；否则 ``loader(corpus_path)``
    （``corpus_path=None`` → 加载器自带的缺省语料文件）。

    ``embedding_model`` **必填**：本层不构造任何编码器 —— 谁要 RAG，谁给编码器（`build_bge_embedder`
    或自备 ``BaseEmbedding``）。漏传即为 ``TypeError``，不会悄悄替你造一个。
    """
    if rows is None:
        record_rows, _meta = loader(corpus_path)
    else:
        record_rows = list(rows)
    return index_cls(
        record_rows, embedding_model=embedding_model, mode=mode, config=config
    )


def build_policy_index(
    rows: Iterable[Any] | None = None,
    *,
    embedding_model: Any,
    corpus_path: str | Path | None = None,
    mode: RetrievalMode = "hybrid",
    config: ChromaConfig | None = None,
) -> PolicyIndex:
    """构造 PolicyIndex（corpus_path 缺省 = rag/corpus/policies.json）。

    ``rows`` 显式注入时优先（跳过文件 IO）。``embedding_model`` **必填**（``BaseEmbedding``；
    常用 ``pra.rag.embedder.build_bge_embedder()``）—— 本函数不替你构造。
    ``config`` = Chroma 连接 / collection 参数（见 :class:`~pra.rag.chroma_backend.ChromaConfig`；
    缺省 None → 全取默认值）。
    """
    # 延迟 import：chroma / llama_index / bm25s / jieba 仅在真正装配索引时才拉起。
    from pra.rag.chroma_backend import ChromaPolicyIndex

    return _build_index(
        ChromaPolicyIndex,
        loader=load_policies,
        corpus_path=corpus_path,
        rows=rows,
        embedding_model=embedding_model,
        mode=mode,
        config=config,
    )


def build_case_index(
    rows: Iterable[Any] | None = None,
    *,
    embedding_model: Any,
    corpus_path: str | Path | None = None,
    mode: RetrievalMode = "hybrid",
    config: ChromaConfig | None = None,
) -> CaseIndex:
    """构造 CaseIndex（corpus_path 缺省 = rag/corpus/cases.json）；参数语义同 ``build_policy_index``。"""
    from pra.rag.chroma_backend import ChromaCaseIndex

    return _build_index(
        ChromaCaseIndex,
        loader=load_cases,
        corpus_path=corpus_path,
        rows=rows,
        embedding_model=embedding_model,
        mode=mode,
        config=config,
    )
