"""RAG 真实检索链路（Policy KB + Case KB）—— 包导览。

- ``corpus/``：静态 corpus 数据 + 校验 schema；
- ``deps.py``：第三方重依赖（chromadb / llama_index）的延迟 import 边界；
- ``embedding.py``：BGE 编码器构造点（``production_embedder``）；
- ``retrieval.py``：检索公共件（候选集检索上下文）；
- ``chroma_store.py``：Chroma 连接、collection 与 corpus 行 → Node 的装配；
- ``bm25.py``：BM25 检索器（jieba 分词 + bm25s 自建索引）；
- ``index.py``：``ChromaPolicyIndex`` / ``ChromaCaseIndex``（候选过滤 + hybrid 检索）；
- ``lazy_index.py``：把索引构建推迟到首次检索的代理；
- ``factory.py``：装配入口 ``build_policy_index`` / ``build_case_index``。

chroma / llama_index / bm25s / jieba / fastembed 全部**延迟 import**（装配或检索时才拉起），故
import 本包零额外依赖；缺 rag extra 时抛出带指引的 ``RuntimeError``，不静默降级成无关结果的空检索。
"""

from __future__ import annotations
