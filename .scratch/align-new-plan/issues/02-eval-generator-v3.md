# 02: 评测生成器 v3 —— 构造即标签

**What to build:** 从零新建评测数据集生成器，落地"构造即标签"：生成器按构造参数合成案件，同时输出**埋点清单**（植入了什么风险证据、埋在哪）；**独立打标函数**按**冻结规则书**从埋点清单推导真值（decision + abstention + 标签字段），打标函数与生成器代码解耦。难例不再是人工筛选结果，而是生成器的构造维度。真值全程零人工参与、可复现。

**Blocked by:** 01（风险词表收敛为 2 值）

**Status:** resolved

- [x] 生成器一次运行产出 v3 数据集：案件 + 埋点清单 + 由打标函数推导的真值
- [x] 打标函数是独立函数：输入埋点清单、输出真值，不依赖生成器内部状态
- [x] 规则书冻结为数据文件，同输入必得同真值（可复现性有测试守护）
- [x] 难例（强规避、弱证据、近似边界等）可通过构造参数配置生成
- [x] 词表只含 `POTENTIAL_IP_RISK` / `EVASION_PATTERN`

## Comments

- 实施记录（2026-10-12）：320 案（seed=20261012，场景 64/64/96/64/32，真值 PASS 88 / REJECT 122 / HUMAN_REVIEW 110）。
  布局：`scripts/evaluation/generator/builder.py`（构造）、`scripts/evaluation/labeling/`（`labeler.py` 纯函数 +
  冻结 `rulebook.json`，sha256 在 `tests/test_eval_labeling_v3.py` 钉死）、v3 契约模型在
  `scripts/evaluation/dataset/schema.py`（v2 模型未动）、CLI `scripts/build_eval_dataset.py`（自检 +
  `--no-write` + 退出码）。产物 `eval_data/v3/`：cases/truth 双文件（埋点只在真值侧）+ `seed_v3.sql`
  （P_V3_* / M_V3_* 各 320 行，幂等约定同 migrations/003、004）+ manifest.json（含规则书版本/哈希）。
  `build_rag_corpus.py` 的 EVAL_FILES 纳入 v3，隔离自检继续通过。
- 打标判定优先级：黑名单品牌 > 规避词（商家脏从重 POLICY_4.5 / 干净 POLICY_1.2） > 快照缺口与在库不一致 >
  品牌词（可叠加商家脏）> 商家脏 > 干净放行。文本命中归类、商家脏判定均由打标函数从原始事实 + 规则书推导
  （埋点清单只记原始事实，不记结论），构造期"文本自证"校验（重扫 == 植入清单）会拦住模板词表污染。
- 偏离说明：HUMAN_REVIEW 案的 risk_level 统一 MEDIUM、expected_policies 为空（弃权不需要引用依据，
  正确性契约只要求 REJECT 具备可引用依据，该不变量已进 `EvalExpectedV3` 校验器）。
- 修复记录（2026-10-12，code-review 后续修）：
  - 真值补"期望证据类型"（docs/02 §3.1）：`EvalExpectedV3` 新增 `expected_evidence`
    （受控词表 RULE_HIT / MERCHANT_HISTORY / PRODUCT_FACT / POLICY_REF / CASE_PRECEDENT /
    MEASUREMENT，对齐生产证据 type 常量）；规则书新增 `evidence_rules` 推导数据
    （规则命中→RULE_HIT、快照缺口→MEASUREMENT、REJECT→POLICY_REF、期望工具→对应证据类型），
    打标函数据此推导，320 案全量标注；规则书版本升 v3-rulebook-2，sha256 钉值同步更新。
  - 删除无消费者字段 `EvalExpectedV3.risk_level`（判卷不比对、指标层不消费、docs/02 §3.1
    本无此项），连同 schema 校验器中涉及它的不变量一起删。
  - 数据集按同默认 seed 重生成：规模 320 案不变，真值分布 PASS 88 / REJECT 122 /
    HUMAN_REVIEW 110 不变（risk_level 删除与 expected_evidence 新增不影响裁决推导）。
