# 04: 指标层重写

**What to build:** 评测报告指标层重写：头条改为 selective prediction 口径——coverage（自动裁决覆盖率）、selective risk（自动裁决错误率）、错误拒绝率；第二层为蜕变测试、pass^k、过程断言（证据引用、Gate 触发）；Agent 专用指标（工具选择准确率等）降级为诊断层保留，不进头条。

**Blocked by:** 03（评测执行层改造）

**Status:** resolved

- [x] 评测报告头条输出 coverage / selective risk / 错误拒绝率
- [x] 蜕变测试与 pass^k 可运行并出数
- [x] 过程断言纳入判卷：无证据引用的拒绝、应触发的 Gate 未触发均判错
- [x] Agent 诊断指标保留在报告诊断区，不出现在头条
- [x] 指标口径与 `docs/02-evaluation.md` 一致

## Comments

- 实施记录（2026-10-12）：头条 selective 口径在 `scripts/evaluation/metrics/selective.py`
  （`evaluate_selective(gradings, exclude=...)` → `SelectiveMetrics`：coverage=auto_decided/total、
  selective_risk=auto_wrong/auto_decided、false_reject_rate=false_rejects/truth_pass_total；
  直接消费判卷结果 `CaseGrading`，失败案天然不进分母，疑问篮子经 `exclude` 集合从全部分母剔除）。
  诊断层在 `scripts/evaluation/metrics/diagnostics.py`（工具选择准确率=应调全调/标注期望工具的案、
  多余调用率=调了期望外工具/同分母、风险类型命中率、有效取证率=带来新证据的 ok 调用/成功调用，
  工具与风险类型两格复用判卷器的集合差，不重现比对逻辑）。过程断言在
  `scripts/evaluation/metrics/assertions.py`（四码：REJECT_WITHOUT_CITATION / GATE_NOT_TRIGGERED /
  BUDGET_EXCEEDED / CITATION_UNRESOLVABLE；黑名单直接引 `pra.screening.rule_engine.terms`，
  引用解析经 `load_citation_index` 直读 corpus JSON，政策按 policy_id 或 clause_id 解析、
  先例按 CASE_PRECEDENT 证据 ref_id 解析）。蜕变与 pass^k 在 `scripts/evaluation/metamorphic.py`
  （保语义扰动只改价格/SKU ID/上架时间/图片地址，风险增强扰动标题前缀植入规避词；
  严格度序 REJECT>HUMAN_REVIEW>PASS；`pass_k_consistency` 分母=有 ≥2 次运行记录的案）。
- `scripts/run_evaluation.py`：报告 `_render_report` 重排为 头条三指标表（值(分子/分母)）→ 诊断区
  → 过程断言清单 → 可选蜕变/pass^k → 失败案明细 → 工程指标（EngineeringEvaluator 口径未动）；
  CLI 新增 `--exclude-ids`（JSON 数组文件，id 不在数据集显式报错）、`--metamorphic`（agent 臂
  加跑 2 类扰动）、`--pass-k`（agent 臂重复 k 次）；payload 的 metrics 改挂 selective+engineering，
  新增 diagnostics / assertions / metamorphic / pass_k / excluded_ids 字段。
- 删除的旧口径：`scripts/evaluation/metrics/business.py`（DecisionEvaluator 的
  accuracy/precision/recall/TP-FP/reject_unhandled/automation_coverage）、
  `scripts/evaluation/metrics/agent.py`（旧 AgentMetricsBundle 三件套，含 docs/02 §5.2 之外的
  risk_level 一致率与 decision_changed/avg_evidence_added 等无消费者字段）、
  `tests/test_evaluation_recall_definition.py`（旧 recall 定义守护）。
  `tests/test_metrics.py` 未动：它是 guardrails/gate_probe 的生产护栏探针测试，与评测指标层无关
  （任务书点名它只是因为文件名）。v2 数据专用测试 `test_evaluation_dataset_v2.py` 留票 06。
- 偏离说明：① 过程断言未并入判卷器（grade_case 签名未动）：黑名单常量与语料库都在 pra 侧，
  而 `tests/test_eval_grading.py` 的 AST 守卫禁止 grading 包 import pra；断言放 metrics 层独立模块，
  经 truth 行拿 manifest，效果等同"纳入判卷"（失败清单进报告与 payload）。② 无证据引用拒绝的判定
  按任务书字面取"或"：REJECT ∧（policy 为空 ∨ 证据链无可引用依据）即判错 —— rule 臂 REJECT 只有
  规则命中没有政策引用，会系统性命中此断言；这是口径语义（自动拒绝须同时具备可引用依据），
  不是实现疏漏。③ 蜕变/pass^k 只接线 agent 臂（实施时的口头指引是"至少覆盖 agent 臂"，票原文无此句；
  rule 确定性无需重复，single 臂未覆盖——如有需要同一开关可扩展）。
- 验证：`uv run pytest -q` 391 passed / 7 skipped；`uv run ruff check` 维持基线 51 个存量 error，
  本票触碰的文件 0 告警。新增测试：test_eval_selective_metrics / test_eval_diagnostics /
  test_eval_process_assertions / test_eval_metamorphic / test_eval_report，重写
  test_eval_failure_isolation，test_eval_run_v3_e2e 增加第二层 CLI 接线格（fake backend 下
  --exclude-ids/--metamorphic/--pass-k 出数）。
- 修复记录（2026-10-12，code-review 后续修）：
  - 判卷补证据类型比对（docs/02 §4.1）：`CaseGrading` 新增 expected/actual/missing/extra/
    match 五格（期望空 → None 不进比对，语义同工具格）；诊断区**不**新增证据命中率指标
    （docs/02 §5.2 未列，无 spec 依据不造指标）。
  - 删除无消费者字段 `CaseGrading.truth_abstain_label`（abstain 可由 truth_decision 推导）。
  - 过程断言口径修正：`REJECT_WITHOUT_CITATION` 收敛为只评 agent 臂（rule 臂可引用依据是
    规则命中、single 臂无证据通道是基线设计），与 `GATE_NOT_TRIGGERED` 臂 scope 对齐；
    `check_process_assertions` 签名加 `scheme` 臂参数；新增 `SILENT_DEGRADATION`
    （auto 决策带 detail.error，或 agent 臂 PASS 但证据链为空）。
  - "过程断言失败计入判错"兑现：selective risk 分子 = 自动决策 ∧（裁决与真值不一致 ∨
    命中过程断言），coverage 分母不变；`evaluate_selective` 加 `assertion_failed` 参数，
    run_evaluation 接线，报告头条注记判错口径。
  - 跨源一致性：断言的黑名单与蜕变规避词统一改读冻结规则书（真值口径单一来源）；
    新增 `tests/test_eval_rulebook_drift.py` 守规则书词表/阈值与生产 terms/
    MERCHANT_DIRTY_MIN 及证据词表与生产 type 常量的一致性，漂移即显式失败。
  - 蜕变补"换表述"扰动（docs/02 §6）：`perturb_paraphrase` 用生成器模板同义改写映射，
    匹配不到回退追加中性从句；`check_metamorphic` 收保语义变体列表；--metamorphic
    接线为 3 类扰动。
