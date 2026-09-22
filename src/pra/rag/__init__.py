"""RAG 真实检索链路（Policy KB + Case KB）：corpus 加载 + BM25 / 向量 / RRF hybrid 检索。

chroma / llama_index / bm25s / jieba / fastembed 全部延迟 import，故 import 本包零额外依赖；
缺 rag extra 时抛带指引的 ``RuntimeError``，不静默降级成空检索。
"""

from __future__ import annotations
