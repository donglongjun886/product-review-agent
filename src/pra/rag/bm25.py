"""BM25 检索桥 —— jieba 分词接入 ``bm25s``，以及候选集内的 BM25 检索器。

``bm25s.tokenize`` 无注入点，故构造与检索期间在锁内换成 jieba 实现；这**不是「天然线程安全」**，
``_TOKENIZER_LOCK`` 必须写在补丁之前（见 :func:`_jieba_tokenizer`）。
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from pra.rag.deps import llama
from pra.rag.retrieval import _RetrievalContext

__all__ = ["bm25_retrieve", "make_bm25_retriever"]

_LATIN_RUN = re.compile(r"[0-9A-Za-z_]+")

#: 全局替换锁：bm25s.tokenize 是模块级符号，构造与检索期间独占。
_TOKENIZER_LOCK = threading.RLock()


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


def _jieba_bm25s_tokenize(texts: Any, **_kwargs: Any) -> Any:
    """``bm25s.tokenize`` 的 jieba 替身（签名兼容：忽略 token_pattern/stemmer/stopwords 等）。

    逐文档切词 → 词表「首次出现即编号」（跨文档共享，确定性）；查询与索引走同一函数 →
    缺失查询词也进词表，不会触发 bm25s 的越界 token id 报错。
    """
    from bm25s.tokenization import Tokenized

    single = isinstance(texts, str)
    items = [texts] if single else list(texts)
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


@contextmanager
def _jieba_tokenizer() -> Iterator[None]:
    """在上下文内把 ``bm25s.tokenize`` 换成 jieba 实现（退出即恢复原符号）。

    ``BM25Retriever`` 的构造与 ``retrieve`` 都硬编码调用 ``bm25s.tokenize`` 且没有注入点，故索引
    与查询都必须在此上下文内完成 —— 保证两者是同一个分词器。

    🔴 **调用方必须写成 ``with _TOKENIZER_LOCK, _jieba_tokenizer():``（锁在前、补丁在后）**：
    上下文管理器按「左→右 __enter__、右→左 __exit__」执行；写成 ``with _jieba_tokenizer(),
    _TOKENIZER_LOCK:`` 则补丁在取锁**之前**、恢复在放锁**之后** —— 临界区不覆盖补丁的安装/
    撤销，并发下必然出错：线程 B 会把「A 打的补丁」当 ``original`` 存下，A 退出即恢复真身 →
    B 在自以为的 jieba 上下文里用真分词器检索 jieba 建的索引（``ValueError: The maximum token
    ID in the query ... is higher than the number of tokens in the index.``）；B 退出再把补丁
    写回 → ``bm25s.tokenize`` 进程级永久泄漏。
    """
    import bm25s

    original = bm25s.tokenize
    bm25s.tokenize = _jieba_bm25s_tokenize
    try:
        yield
    finally:
        bm25s.tokenize = original


def bm25_retrieve(retriever: Any, query_bundle: Any) -> list[Any]:
    """BM25 检索（**在 jieba 上下文内** —— 查询与索引必须同一分词器）。

    ⚠️ 顺序不可颠倒：锁**在**补丁之前（见 :func:`_jieba_tokenizer`）。
    """
    with _TOKENIZER_LOCK, _jieba_tokenizer():
        return retriever.retrieve(query_bundle)


def make_bm25_retriever(ctx: _RetrievalContext, top_k: int) -> Any:
    """BM25 检索器（``bm25s`` + jieba）：**只喂候选 node**（Python 侧过滤）。

    ``similarity_top_k=top_k`` 取候选集内 Top-K；``skip_stemming=True`` / ``language=""`` 关掉
    英文词干器与停用词（中文语料无意义）；``token_pattern=""`` 只作显式标注（jieba 替身忽略它）。
    """
    # ⚠️ 顺序不可颠倒：锁**在**补丁之前（见 :func:`_jieba_tokenizer` docstring，R6 实测）。
    with _TOKENIZER_LOCK, _jieba_tokenizer():
        return llama().BM25Retriever(
            nodes=list(ctx.nodes),
            similarity_top_k=min(top_k, len(ctx.nodes)),
            skip_stemming=True,
            language="",
            token_pattern="",
            verbose=False,
        )
