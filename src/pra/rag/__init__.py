"""RAG 真实检索链路（Policy KB + Case KB）。

- ``corpus/``：静态 corpus 数据 + 校验 schema；
- ``embedder.py`` / ``bm25.py`` / ``vectors.py`` / ``retrieval.py``：检索内核 —— embedder
  provider + 自写 BM25 + 纯 Python 余弦 + 三模式（bm25/vector/hybrid）编排；
- ``index.py``：``RagPolicyIndex`` / ``RagCaseIndex``，实现 tools 层检索 Protocol；
- ``factory.py``：装配入口 ``build_policy_index`` / ``build_case_index``。

默认 embedding 是**确定性 mock（hash）**：只验证链路与可重放、**不是语义检索**；真语义
模型与向量库是可选后端（``backend="qdrant"`` / ``"chroma"``），默认路径不装任何向量库依赖。
"""

from __future__ import annotations
