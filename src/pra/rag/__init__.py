"""RAG 真实检索链路（Policy KB + Case KB）。

- ``corpus/``：静态 corpus 数据 + 校验 schema；
- ``retrieval.py``：检索模式枚举与 BM25 分归一化等公共件；
- ``chroma_backend.py``：ChromaDB + LlamaIndex 检索后端（唯一后端），向量路 + BM25(jieba) 路 RRF 融合；
- ``lazy_index.py``：把索引构建推迟到首次检索的代理；
- ``factory.py``：装配入口 ``build_policy_index`` / ``build_case_index``。

chroma / llama_index / bm25s / jieba 全部**延迟 import**（在装配或检索时才拉起），故 import 本包
零额外依赖；缺 rag extra 时抛出带指引的 ``RuntimeError``，不静默降级成无关结果的空检索。
"""

from __future__ import annotations
