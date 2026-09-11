"""Embedder → LlamaIndex ``BaseEmbedding`` 适配层。

本项目**自带** embedding provider（``pra.rag.embedder.Embedder``），而 LlamaIndex 的
Retriever 只认 ``BaseEmbedding``。本模块只做**形状适配**，不引入任何 LlamaIndex 模型集成：
查询/文本向量都直接委托本项目的 ``Embedder.embed``，无第二套编码逻辑 —— 故向量口径与 local
numpy 后端**完全同源**，换后端只换存储/检索，不换编码。

本适配层**零 LLM、零网络、零随机**；构造不加载模型（``BgeEmbedder`` 的懒加载语义原样保留，
首次 ``embed`` 才可能下载/加载）。模块顶层只 import ``llama_index.core``，**不 import
chromadb、不建客户端、不发请求**。
"""

from __future__ import annotations

from typing import Any

from llama_index.core.base.embeddings.base import BaseEmbedding
from pydantic import PrivateAttr

from pra.rag.embedder import Embedder

__all__ = ["LlamaIndexEmbeddingAdapter", "as_llama_embedding"]


class LlamaIndexEmbeddingAdapter(BaseEmbedding):
    """把本项目的 ``Embedder``（Mock / BGE）包装成 LlamaIndex ``BaseEmbedding``。

    单入口适配：``get_query_embedding`` / ``get_text_embedding`` / ``get_text_embedding_batch``
    全部委托 ``self._embedder.embed`` —— **不区分 query/doc 编码**（与 ``Embedder`` Protocol 的
    窄接口一致；bge-zh query instruction 未引入）。构造本适配器**不加载模型**。

    :param embedder: 本项目 ``Embedder`` 实现（``MockHashEmbedder`` / ``BgeEmbedder`` / 任意
        满足 ``embed(text) -> list[float]`` 的对象）。
    :param embed_batch_size: 批量接口批大小（仅作为 ``BaseEmbedding`` 的声明字段，本适配层
        逐条委托，不做批内并行）。
    """

    _embedder: Any = PrivateAttr()

    def __init__(self, embedder: Embedder, embed_batch_size: int = 10) -> None:
        super().__init__(embed_batch_size=embed_batch_size)
        self._embedder = embedder

    @classmethod
    def class_name(cls) -> str:
        return "PraEmbedderAdapter"

    @property
    def embedder(self) -> Embedder:
        return self._embedder

    # -- LlamaIndex 抽象方法（query） -----------------------------------------

    def _get_query_embedding(self, query: str) -> list[float]:

        return self._embedder.embed(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._embedder.embed(query)

    # -- LlamaIndex 抽象方法（text） ------------------------------------------

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embedder.embed(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [self._embedder.embed(t) for t in texts]


def as_llama_embedding(embedder: Embedder, *, embed_batch_size: int = 10) -> LlamaIndexEmbeddingAdapter:
    return LlamaIndexEmbeddingAdapter(embedder, embed_batch_size=embed_batch_size)
