# 03: 评测执行层改造

**What to build:** 评测执行层适配 v3 数据集：dataset schema / loader 读取生成器输出（真值来自打标函数，不再是人工标签字段），判卷全确定性（禁止 LLM-as-judge），`run_evaluation` 对 v3 数据集端到端跑通并产出逐案判卷结果。新旧数据契约不兼容时显式失败，不得静默混用。

**Blocked by:** 02（评测生成器 v3 —— 构造即标签）

**Status:** ready-for-agent

- [ ] `run_evaluation` 对 v3 数据集端到端跑通，产出逐案判卷结果
- [ ] 判卷路径无任何 LLM 调用（有测试守护）
- [ ] 旧格式数据喂入新执行层时显式报错而非静默降级
- [ ] `uv run pytest -q` 全绿
