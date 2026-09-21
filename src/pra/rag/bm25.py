"""BM25 检索器 —— jieba 分词 + ``bm25s`` 自建索引，不使用库 ``BM25Retriever``。

分词不吃 ``bm25s.tokenize`` 模块级符号：语料与查询都走本模块的 ``_tokenize``，故无全局替换、
无锁、也无调用顺序约束。
"""

from __future__ import annotations

import functools
import re
from typing import Any

from pra.rag.deps import llama
from pra.rag.retrieval import _RetrievalContext

__all__ = ["make_bm25_retriever"]

_LATIN_RUN = re.compile(r"[0-9A-Za-z_]+")


def _jieba_tokens(text: str) -> list[str]:
    """jieba 精确模式切词（确定性；拉丁/数字串拆出并小写，中文词原样）。

    不做停用词过滤/词干化（中文语料上英文词干器无意义）；单字符词保留（文档频率高、IDF 低，
    对排序影响可忽略）。
    """
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
    """文本（单条或列表）→ ``bm25s`` 的 ``Tokenized(ids, vocab)``；语料与查询共用。

    词表按「首次出现即编号」在本次调用内生成；``bm25s`` 检索时把查询 token 经字符串映回索引
    词表，故查询与索引必为同一分词器，缺失查询词也不会触发越界 token id 报错。
    """
    from bm25s.tokenization import Tokenized

    items = [texts] if isinstance(texts, str) else list(texts)
    vocab: dict[str, int] = {}
    ids: list[list[int]] = []
    for text in items:
        doc_ids: list[int] = []
        for tok in _jieba_tokens(str(text)):
            if tok not in vocab:
                vocab[tok] = len(vocab)
            doc_ids.append(vocab[tok])
        ids.append(doc_ids)
    return Tokenized(ids=ids, vocab=vocab)


@functools.cache
def _retriever_class() -> type:
    """BM25 检索器的类对象（继承库 ``BaseRetriever`` ⇒ 首次调用才 import llama_index）。"""
    import bm25s

    class _JiebaBM25Retriever(llama().BaseRetriever):
        """候选 node 上的 BM25 检索器（jieba 分词 + ``bm25s``；索引与词表随实例自持）。"""

        def __init__(self, nodes: list[Any], similarity_top_k: int) -> None:
            # 索引文本 = node 的 EMBED 正文（metadata 已在 ``chroma_store._build_nodes`` 排除）。
            # 命中节点由 ``_node_content`` 重建，与向量路同源 —— RRF 按 ``node.hash`` 合并两路。
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
            """Top-K 命中（分数原样透传 ``bm25s``，不做量纲适配）。"""
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
    """候选 node 上的 BM25 检索器（Python 侧过滤 = 只喂候选 node）。"""
    return _retriever_class()(list(ctx.nodes), top_k)
