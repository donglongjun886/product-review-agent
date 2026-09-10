"""Embedder → LlamaIndex ``BaseEmbedding`` 适配层（rag/llama_embedding.py）。

定位（docs/10-rag-upgrade-spec.md §2 复用边界 / §4 新增文件）：本项目**自带** embedding
provider（``pra.rag.embedder.Embedder`` Protocol：``MockHashEmbedder`` / ``BgeEmbedder``），
而 LlamaIndex 的 Retriever 只认 ``BaseEmbedding``。本模块只做**形状适配**，不引入任何
LlamaIndex 模型集成（不 import ``llama-index-embeddings-openai`` / ``-fastembed`` 等）：

- 查询/文本向量都直接委托本项目的 ``Embedder.embed``（单一实现，无第二套编码逻辑）；
- 因此向量口径与 ``rag/index.py``（local numpy 后端）**完全同源**：同一 ``embedder``
  实例产出的向量逐位一致 —— 换后端只换存储/检索，不换编码。

确定性契约（docs/10 §3「确定性」）：本适配层**零 LLM、零网络、零随机**；构造不加载
模型（``BgeEmbedder`` 的懒加载语义原样保留，首次 ``embed`` 才可能下载/加载）。

离线安全：模块顶层只 import ``llama_index.core``（已在本项目 rag extra 中固定），
**不 import chromadb、不建客户端、不发请求**。
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
    全部委托 ``self._embedder.embed`` —— **不区分 query/doc 编码**（与
    ``Embedder`` Protocol 的窄接口一致；bge-zh query instruction 未引入，docs/06 P2-6）。
    BGE 的懒加载语义不变：构造本适配器**不加载模型**。

    :param embedder: 本项目 ``Embedder`` 实现（``MockHashEmbedder`` / ``BgeEmbedder`` / 任意
        满足 ``embed(text) -> list[float]`` 的对象）。
    :param embed_batch_size: 批量接口的批大小（仅作为 ``BaseEmbedding`` 的声明字段，
        本适配层逐条委托，不做批内并行）。
    """

    _embedder: Any = PrivateAttr()

    def __init__(self, embedder: Embedder, embed_batch_size: int = 10) -> None:
        super().__init__(embed_batch_size=embed_batch_size)
        self._embedder = embedder

    @classmethod
    def class_name(cls) -> str:
        """LlamaIndex 序列化标识（本类不落盘，仅供日志/调试辨识）。"""
        return "PraEmbedderAdapter"

    @property
    def embedder(self) -> Embedder:
        """被包装的本项目 embedder（测试/调试可见）。"""
        return self._embedder

    # -- LlamaIndex 抽象方法（query） -----------------------------------------

    def _get_query_embedding(self, query: str) -> list[float]:
        """查询向量 = 本项目 ``Embedder.embed(query)``（与 doc 编码同一入口）。"""
        return self._embedder.embed(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """异步查询向量 —— 委托同一同步实现（本项目 embedder 为纯计算，无 IO）。"""
        return self._embedder.embed(query)

    # -- LlamaIndex 抽象方法（text） ------------------------------------------

    def _get_text_embedding(self, text: str) -> list[float]:
        """文本向量 = 本项目 ``Embedder.embed(text)``。"""
        return self._embedder.embed(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """批量文本向量（逐条委托，顺序与入参一致；确定性无随机）。"""
        return [self._embedder.embed(t) for t in texts]


def as_llama_embedding(embedder: Embedder, *, embed_batch_size: int = 10) -> LlamaIndexEmbeddingAdapter:
    """``Embedder`` → LlamaIndex ``BaseEmbedding`` 的便捷构造函数（语义同直接实例化）。"""
    return LlamaIndexEmbeddingAdapter(embedder, embed_batch_size=embed_batch_size)
