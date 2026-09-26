"""BM25 检索器：jieba 分词 + ``bm25s`` 自建索引。"""

from __future__ import annotations

import functools
import re
from typing import Any

from pra.rag.deps import llama
from pra.rag.retrieval import _RetrievalContext

__all__ = ["make_bm25_retriever"]

_LATIN_RUN = re.compile(r"[0-9A-Za-z_]+")


def _jieba_tokens(text: str) -> list[str]:
    """jieba 精确模式切词；拉丁/数字串拆出并小写，中文词原样。"""
    import jieba

    tokens: list[str] = []
    for piece in jieba.cut(text, cut_all=False):
        piece = piece.strip()
        if not piece:
            continue
        for latin in _LATIN_RUN.findall(piece):
            tokens.append(latin.lower())
        if _LATIN_RUN.sub("", piece):
            tokens.append(piece)
    return tokens


def _tokenize(texts: Any) -> Any:
    """文本（单条或列表）→ token 二维列表。"""
    items = [texts] if isinstance(texts, str) else list(texts)
    return [_jieba_tokens(str(text)) for text in items]


@functools.cache
def _retriever_class() -> type:
    """返回 BM25 检索器的类对象。"""
    import bm25s

    class _JiebaBM25Retriever(llama().BaseRetriever):
        """候选 node 上的 BM25 检索器（jieba 分词 + ``bm25s``）。"""

        def __init__(self, nodes: list[Any], similarity_top_k: int) -> None:
            self._corpus = [
                llama().node_to_metadata_dict(node) | {"node_id": node.node_id} for node in nodes
            ]
            self._similarity_top_k = max(1, min(similarity_top_k, len(nodes)))
            self._bm25 = bm25s.BM25()
            self._bm25.index(
                _tokenize(
                    [node.get_content(metadata_mode=llama().MetadataMode.EMBED) for node in nodes]
                ),
                show_progress=False,
            )
            super().__init__()

        def _retrieve(self, query_bundle: Any) -> list[Any]:
            """返回 Top-K 命中（``NodeWithScore`` 列表）。"""
            indexes, scores = self._bm25.retrieve(
                _tokenize(query_bundle.query_str),
                k=self._similarity_top_k,
                show_progress=False,
                weight_mask=None,
            )
            return [
                llama().NodeWithScore(
                    node=llama().metadata_dict_to_node(self._corpus[int(idx)]),
                    score=float(score),
                )
                for idx, score in zip(indexes[0], scores[0])
            ]

    return _JiebaBM25Retriever


def make_bm25_retriever(ctx: _RetrievalContext, top_k: int) -> Any:
    """候选 node 上的 BM25 检索器。"""
    return _retriever_class()(list(ctx.nodes), top_k)
