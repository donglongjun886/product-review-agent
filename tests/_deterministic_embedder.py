"""测试自持的确定性编码器 —— 替代已移除的 ``MockHashEmbedder``。

chroma 链的编码器由后端**构造参数** ``embedding_model`` 注入（一个「编码器对象」，不是 ``kind``
字符串），故测试可以自持一个**同接口**的编码器，而不必依赖 ``pra.rag.embedder`` 里那个已被移除的
mock：

- ``get_text_embedding(text) -> list[float]``：建库路径（``_ChromaIndexBase`` 逐行编码 corpus）。
- ``get_query_embedding(query) -> list[float]``：检索路径（向量路把查询编码后做 cosine）。
- ``dim`` / ``embed_dim``：``_resolve_dim`` 解析维度用（声明维度 == 实测向量长度）。

**为什么是纯标准库实现（不继承 LlamaIndex ``BaseEmbedding``）**：本模块被 CI 下（``uv sync
--frozen``，**不装任何 extra**）的 ``tests/test_rag.py`` 顶层 import —— 一旦在此拉起
``llama_index``，CI 会直接 ImportError，原有的「离线确定性覆盖」就退化成「没装 rag extra 就全
skip」。故这里只依赖标准库 + ``pra.rag.bm25.tokenize``（零第三方分词依赖），离线、零下载。

**算法与旧 ``MockHashEmbedder`` 逐字节一致**（词面 sha256 特征哈希：``sha256(token)[:8]`` 大端
取模落到特征桶 +1），故既有数值断言（256 维 collection 名 / cosine 单位向量探针 / 候选完整性）
全部保持；向量刻画**词面特征**（CJK bigram / 拉丁词），同输入同输出、跨进程逐位可重放 —— 但它
**不是语义模型**，不得据此解读「语义检索」能力。
"""

from __future__ import annotations

import hashlib

from pra.rag.bm25 import tokenize

__all__ = ["DETERMINISTIC_EMBED_DIM", "DeterministicHashEmbedder"]

#: 默认特征维度（与旧 mock 一致；多处断言写死 collection 名 ``*_256``）。
DETERMINISTIC_EMBED_DIM = 256


class DeterministicHashEmbedder:
    """确定性词面编码器：同输入同输出、跨进程稳定、离线零下载（非语义）。

    :param dim: 特征维度（默认 :data:`DETERMINISTIC_EMBED_DIM` = 256）。
    """

    def __init__(self, dim: int = DETERMINISTIC_EMBED_DIM) -> None:
        if dim < 1:
            raise ValueError(f"embedding 维度须为正: {dim}")
        self.dim = dim

    @property
    def embed_dim(self) -> int:
        """``BaseEmbedding`` 风格的维度别名（``_resolve_dim`` 依次读 ``dim`` / ``embed_dim``）。"""
        return self.dim

    def _feature_index(self, token: str) -> int:
        # sha256 而非内置 hash()：后者受 PYTHONHASHSEED 影响跨进程漂移，破坏逐字节重放。
        digest = hashlib.sha256(token.encode("utf-8")).digest()[:8]
        return int.from_bytes(digest, "big") % self.dim

    def embed(self, text: str) -> list[float]:
        """词面特征哈希 → 稀疏计数向量（与旧 ``MockHashEmbedder.embed`` 逐字节一致）。"""
        vec = [0.0] * self.dim
        for tok in tokenize(text):
            vec[self._feature_index(tok)] += 1.0
        return vec

    # -- chroma 链接口（与 LlamaIndex ``BaseEmbedding`` 的公开方法同名）--
    def get_text_embedding(self, text: str) -> list[float]:
        return self.embed(text)

    def get_query_embedding(self, query: str) -> list[float]:
        return self.embed(query)
