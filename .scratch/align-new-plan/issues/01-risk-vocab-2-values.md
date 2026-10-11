# 01: 风险词表收敛为 2 值

**What to build:** 风险类型受控词表从 4 值收敛为 2 值：只保留 `POTENTIAL_IP_RISK`（疑似 IP / 品牌模仿）与 `EVASION_PATTERN`（规避审核行为模式）。`FALSE_CLAIM` / `FIELD_CONFLICT` 及其全部引用连带删除——枚举定义、LLM prompt、Gate、RAG 语料（案件与政策库的元数据标注）、语料构建脚本、测试，一处不留。完成后全量测试绿。

**Blocked by:** None (can start immediately)

**Status:** ready-for-agent

- [ ] `RiskType` 枚举只剩 `POTENTIAL_IP_RISK` / `EVASION_PATTERN` 两个值
- [ ] 全仓库（`docs/01-requirements.md` 中记录该决策的叙述性文字除外）grep 不到 `FALSE_CLAIM` / `FIELD_CONFLICT`
- [ ] RAG 语料中仅被这两个类型支撑、随删除失去存在意义的条目一并移除
- [ ] `uv run pytest -q` 全绿，静态检查无新增告警
