# 04: 指标层重写

**What to build:** 评测报告指标层重写：头条改为 selective prediction 口径——coverage（自动裁决覆盖率）、selective risk（自动裁决错误率）、错误拒绝率；第二层为蜕变测试、pass^k、过程断言（证据引用、Gate 触发）；Agent 专用指标（工具选择准确率等）降级为诊断层保留，不进头条。

**Blocked by:** 03（评测执行层改造）

**Status:** ready-for-agent

- [ ] 评测报告头条输出 coverage / selective risk / 错误拒绝率
- [ ] 蜕变测试与 pass^k 可运行并出数
- [ ] 过程断言纳入判卷：无证据引用的拒绝、应触发的 Gate 未触发均判错
- [ ] Agent 诊断指标保留在报告诊断区，不出现在头条
- [ ] 指标口径与 `docs/02-evaluation.md` 一致
