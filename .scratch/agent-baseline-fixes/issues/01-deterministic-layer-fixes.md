# 01: 起点成绩三个确定性层缺陷修复

**Labels:** ready-for-agent

**What to build:** 修复全量 320 案起点成绩暴露的三个确定性层缺陷（详见同目录 spec.md）：
R1 硬规则拒绝补政策引用；listing_registry 测量做真实快照比对；规则层词命中写入
text_compliance 测量证据供模型决策。全程不动 prompt、不动预算/阈值常量。

**Blocked by:** 无

**Status:** resolved

- [x] R1 硬规则 REJECT 自动携带 POLICY_1.1 引用（POLICY_REF 证据 + policy 字段）
- [x] ProductTool listing_registry 测量按快照×在库比对发 verdict（不一致/真空缺 → 阳性）
- [x] 图入口确定性写入 text_compliance 测量证据（R-102/R-302 命中可见、幂等）
- [x] 生产常量与 eval 规则书映射一致性测试
- [x] `uv run pytest -q` 全绿 + ruff 无新增告警
- [x] v3-dev 160 案跑分对比：REJECT_WITHOUT_CITATION 清零、false_reject 保持 0
