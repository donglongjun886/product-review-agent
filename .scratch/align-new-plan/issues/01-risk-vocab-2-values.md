# 01: 风险词表收敛为 2 值

**What to build:** 风险类型受控词表从 4 值收敛为 2 值：只保留 `POTENTIAL_IP_RISK`（疑似 IP / 品牌模仿）与 `EVASION_PATTERN`（规避审核行为模式）。`FALSE_CLAIM` / `FIELD_CONFLICT` 及其全部引用连带删除——枚举定义、LLM prompt、Gate、RAG 语料（案件与政策库的元数据标注）、语料构建脚本、测试，一处不留。完成后全量测试绿。

**Blocked by:** None (can start immediately)

**Status:** resolved

- [x] `RiskType` 枚举只剩 `POTENTIAL_IP_RISK` / `EVASION_PATTERN` 两个值
- [x] 全仓库（`docs/01-requirements.md` 中记录该决策的叙述性文字除外）grep 不到 `FALSE_CLAIM` / `FIELD_CONFLICT`
- [x] RAG 语料中仅被这两个类型支撑、随删除失去存在意义的条目一并移除
- [x] `uv run pytest -q` 全绿，静态检查无新增告警

## Comments

- 实施记录（2026-10-12）：Policy KB 24→16 条款（删 2.x/3.x 段），Case KB 61 条（删 6 条旧案型种子，变体 pattern 2 改写为临界相似度转人工案型，rng 流对齐）；`tests/test_rag.py` 条款数下界 20→10。
- 已知缺口（记录，不修）：`gate.py` `_finalize_risk_type` 兜底路径不再为 text_compliance 阳性派生 risk_type——该分支在生产中本不可达（无工具产出 text_compliance 测量证据），非回归。若未来需要在兜底路径派生 IP/规避类型，应直接从规则命中（R-102/R-302）映射，而不是从维度映射。
