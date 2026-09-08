"""Embedding Provider（rag/embedder.py）—— Embedder 抽象 + 确定性 mock 实现。

定位（对齐 rag-implementation-plan.md R-2，**诚实标注，勿包装**）：

- ``Embedder`` Protocol：``embed(text) -> list[float]`` —— 上层检索（rag/retrieval.py
  与 rag/index.py）只依赖该窄接口；Phase 2 换本地模型（如 BGE）+ Qdrant 时，
  provider 替换即可，RAG 上层检索代码不动。
- ``MockHashEmbedder``：**确定性 mock（hash 特征），不是语义检索**。同输入同输出、
  离线、维度固定 —— 用途仅是：验证「query/doc → 向量 → 余弦 → 混合融合」链路
  通与评测可重放。文档/查询向量刻画的是**词面特征**（CJK 字符 bigram / 拉丁词，
  与 bm25.tokenize 同口径 → bm25 与 vector 两路在词面层可比），**不承诺**语义
  相似（"增高鞋"与"瘦身鞋"是否相似不由此向量保证）——语义质量留 Phase 2
  本地 embedding 模型验证，勿把 hash 结果当语义相似度解读。

确定性：特征哈希用 ``hashlib.sha256``（非内置 ``hash()`` —— 后者受
PYTHONHASHSEED 影响会跨进程漂移，破坏评测逐字节重放）。维度默认 256（常量
``MOCK_DIM``）。向量为词特征计数（counts），cosine 见 rag/vectors.py。
"""

from __future__ import annotations

import hashlib
from typing import Protocol

from pra.rag.bm25 import tokenize

__all__ = ["MOCK_DIM", "Embedder", "MockHashEmbedder"]

MOCK_DIM = 256  # mock 特征维度（Phase 2 换真实模型后由其自身决定维度）


class Embedder(Protocol):
    """文本 → 定长向量的 provider 窄接口（Phase 2 本地模型的替换位）。"""

    def embed(self, text: str) -> list[float]:
        """对一段文本编码为定长 float 向量（离线确定性）。"""
        ...


class MockHashEmbedder:
    """确定性 mock embedding：词特征哈希到固定维度（模块 docstring 口径）。

    :param dim: 特征维度（默认 256）；同输入同输出、跨进程稳定。
    """

    def __init__(self, dim: int = MOCK_DIM) -> None:
        if dim < 1:
            raise ValueError(f"embedding 维度须为正: {dim}")
        self.dim = dim

    def _feature_index(self, token: str) -> int:
        """token → [0, dim) 稳定哈希位（sha256 前缀 8 字节 mod dim）。"""
        digest = hashlib.sha256(token.encode("utf-8")).digest()[:8]
        return int.from_bytes(digest, "big") % self.dim

    def embed(self, text: str) -> list[float]:
        """词特征计数向量（词面层；语义质量不承诺，见模块 docstring）。"""
        vec = [0.0] * self.dim
        for tok in tokenize(text):
            vec[self._feature_index(tok)] += 1.0
        return vec
