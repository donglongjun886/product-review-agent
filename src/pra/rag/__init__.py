"""RAG 真实检索链路（Policy KB + Case KB）—— MVP（numpy/BM25/确定性 mock embedding）。

范围（docs/00-system-design.md §6 + rag-implementation-plan.md v1 定稿）：
- ``corpus/``：Policy KB / Case KB 静态数据 + 校验 schema（M1）；
- ``embedder.py`` / ``bm25.py`` / ``vectors.py`` / ``retrieval.py``：
  检索内核 —— Embedder provider + 自写 BM25 + 纯 Python 余弦 + 三模式
  （bm25/vector/hybrid）编排（M2）；
- ``index.py``：``RagPolicyIndex`` / ``RagCaseIndex`` —— 真实实现 tools 层
  ``PolicyIndex`` / ``CaseIndex`` Protocol（Tool 层零改动）（M3）；
- ``factory.py``：装配入口 build_policy_index / build_case_index（M3/M4 注入点）。

MVP 定位（勿偏移）：可替换 / 可复现 / 可评测 / 与 Evaluation 隔离。embedding 为
**确定性 mock（hash）**，只验证链路与可重放、不包装成语义检索；Qdrant + 本地
embedding（BGE）是 Phase 2 路线（本期不装任何向量库依赖，见 pyproject 可选
``rag = ["qdrant-client"]`` 未启用）。
"""

from __future__ import annotations
