# Spec: 起点成绩暴露的三个确定性层缺陷修复

## Problem Statement

全量 320 案起点成绩（2026-10-15 跑分）暴露 agent 臂的三类缺陷，根因都在确定性代码层，不在模型：

1. **26 案 REJECT_WITHOUT_CITATION**：R1 硬规则（品牌黑名单）强制 REJECT 时，policy 只从
   POLICY_REF 证据派生；模型未检索政策 → policy 为空 → 违反"自动拒绝必须有可引用依据"。
   26 案裁决全部正确（truth=REJECT），错在手续。
2. **12 案真判错（boundary 场景）**：`ProductTool` 的 listing_registry 测量永远报 NEGATIVE
   —— 只念库中事实，从不与案件快照比对。快照缺字段 / 快照与注册不一致（真值均为
   HUMAN_REVIEW）被 PASS Gate 放行。
3. **32 案 evasion 全弃权**：规则层 R-302 明明扫到描述中的规避词，但命中从不写入证据通道，
   模型在 plan/decide 上下文中看到 text_compliance 恒为"阴性"，不敢 REJECT，只能弃权。

## Solution

三处全部为确定性代码修复，**不修改任何 prompt 文本**：

1. R1 硬规则命中时，由确定性代码把命中对应的规则书条款作为 POLICY_REF 证据写入结论
   （黑名单 → POLICY_1.1）。
2. listing_registry 测量真正比对快照与在库事实：不一致 / 关键字段真空缺 → verdict 阳性，
   PASS Gate 确定性拦截 → 转人工。
3. agent 图入口把规则层词命中（R-102/R-302）确定性写成 text_compliance 测量证据
   （阳性含命中明细），模型上下文经既有 coverage_gap_lines 自动可见。

## User Stories

1. As 审核平台运营方，I want 硬规则拒绝自动携带规则书条款引用，so that 自动拒绝满足可引用依据契约。
2. As 审核平台运营方，I want 快照与注册信息不一致的商品被转人工而非放行，so that 边界案不被误判为安全。
3. As 审核平台运营方，I want 决策模型能看到文本合规扫描的阳性命中，so that 该拒的案子敢拒、不再空转人工。
4. As 评测维护者，I want 修复只动确定性层不动 prompt，so that 不存在针对评测集的 prompt 过拟合。
5. As 评测维护者，I want 修复前后在练习册（v3-dev）上对比指标，so that 改进可量化且考卷不被污染。

## Implementation Decisions

- **Fix 1（gate.py R1 分支）**：hard rule 命中时构造 POLICY_REF 证据（ref_id=POLICY_1.1，
  source 确定性标识）追加进证据通道，policy 字段由既有 `_build_decision` 派生逻辑自然带出。
  黑名单→POLICY_1.1 的映射放生产侧（`pra/screening/rule_engine/terms.py` 常量）；
  与 eval 规则书 `policy_refs["blacklist_brand"]` 的一致性由测试守护（环境隔离双份，
  只对可能导致静默错误的部分校验，符合 AGENTS.md）。
- **Fix 2（ProductTool）**：ToolContext 增加 `case`（案件快照），ProductTool.call 比对
  快照 brand/category 与在库行，ProductResult 增加一致性字段；to_evidence 按比对结果发
  verdict：一致 → NEGATIVE；不一致或真空缺 → POSITIVE（value 说明缺口细节）。
  其它工具的 to_evidence 签名不变。
- **Fix 3（图入口）**：在 agent 图首个 LLM 节点之前（hypothesize 入口或装配层）确定性执行
  `triage(case)`，把 R-102/R-302 命中写成 text_compliance MEASUREMENT 证据（阳性带命中词与
  字段明细；无命中不追加——ALWAYS_COVERED 已标记该维度已覆盖，阴性由覆盖清单自然呈现）。
  幂等：重入不重复追加。
- 判定语义不变：PASS/REJECT/HUMAN_REVIEW 三分类、硬规则不可覆盖、证据不足转人工等契约
  全部保持；修复只让确定性层"把已知事实如实呈现"。
- 预算与阈值常量不动。

## Testing Decisions

- 好测试 = 只断言外部可观测契约（verdict/verdict 文本/policy 引用/证据幂等），不锁内部实现。
- 落点（全部沿用现有缝，不新增）：
  - `tests/test_gate.py`：R1 命中 → policy 含 POLICY_1.1 且有对应 POLICY_REF 证据。
  - `tests/test_measurements.py` 或工具测试既有缝：registry 比对三态（一致/不一致/真空缺）。
  - 图入口注入：无命中不追加、命中追加且幂等；coverage_gap_lines 呈现阳性。
  - 生产常量与 eval 规则书映射一致性测试。
- 端到端验证：v3-dev 160 案跑分，对比修复前基线（冒烟：rule cov 1.0 / single 1.0 /
  agent 0.3 干净题口径），预期 REJECT_WITHOUT_CITATION 清零、evasion 弃权大幅下降、
  boundary 误放行清零，false_reject 保持 0。

## Out of Scope

- prompt 文本调优（本轮一律不动）。
- evasion 场景中 truth=HUMAN_REVIEW 的 merchant_dirty_only 案的行为变化不做指标承诺。
- 考卷（eval_data/v3）重跑与最终报告——属后续里程碑，不在本 spec。
- 疑问篮子 63 案的处置（交叉检查设计内分歧，已留档）。

## Further Notes

- 起点成绩留档：/tmp/eval_v3_full.json（agent cov 0.444 / risk 0.268 / false_reject 0）。
- 修复完成后如需归因 decide 弃权理由，可用 .scratch/probe_decide.py 探针。
