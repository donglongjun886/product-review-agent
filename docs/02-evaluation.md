# 电商平台商品内容治理 · 复杂风险调查 Agent —— 评测方案执行细化（02-evaluation）

> 本文档把《00》评测相关章节（§11 Evaluation Dataset、§12 三方案对比、§13 Hard Case Benchmark，及 §7.6 数值口径）
> 落成**实现层唯一依据**，服务对象是写 `src/pra/evaluation/**` 与 `scripts/` 评测脚本的人。
> 引用约定：总设计写作 **《00》§x.y**（即 `docs/00-system-design.md`）；本文内部节号直接写 §x.y。
> 字段级契约以代码为准（`src/pra/agent/state.py`、`src/pra/domain/models.py`、`src/pra/agent/guardrails/schemas.py`）。
>
> **本文档不包含业务代码**：只写 eval_case schema、模块划分、伪代码与指标口径（与 01 同一原则），它们是"契约"。
> **版本状态：v1（P-1~P-5 已拍板）**：评测口径按拍板结果定稿，正文不再标 [待拍板]（逐条决策记录见 §9，供追溯）；
> 正文残留的〔细化待定〕均为**实现者可自行收敛**的落地细节，不阻塞实现；命名/模块结构以 §3.5 为准（**已实现部分以 `src/pra/evaluation/` 实际结构为准**，见该节"命名注"）。
> 本文不评审、不引用 `pra/screening` 的具体规则动作（并行修正集进行中，见 §7），评测框架与具体规则解耦，只按"运行时当前行为"取数并记录行为快照。

---

## 1. 定位与目标

### 1.1 本文档回答什么（职责）

《00》§11–13 只给了评测的"设计意图"，未到可执行层。本文逐项细化为实现者可照做的内容：

| 《00》章节 | 设计意图 | 本文细化 |
|---|---|---|
| §11.1/§11.2 | 评测集规模分布 + Case 结构化标签 | §2 eval_dataset 文件格式、schema、构造与版本管理 |
| §11.3 | 业务/Agent/工程三组指标 | §4 每个指标的计算口径（分子/分母/取数字段） |
| §11.4 | 三方案同一 harness、可比 metrics、确定性重放 | §3 harness 模块划分与 SchemeRunner 契约 |
| §11.5 | threshold sweep、只动配置 | §5 sweep 脚本、曲线与 operating point 选取 |
| §12.0–12.4 | 三方案公平性前提与预期结论 | §3.1–3.4 每方案的执行语义与装配 |
| §13.1–13.4 | Hard Case 定义/构造/Ablation | §2.4（占比与挑选）、§6（Ablation 二期） |

### 1.2 评测要回答的四个问题（本文所有设计围绕它们）

- **Q1** Agent 相比 Rule / Single-call LLM 到底提升在哪：按案件类型（scene）分层看指标差与代价差，而不是只报一个均值。
- **Q2** 哪类 case 必须进 Agent：用"案型 × 方案"增益矩阵（对齐《00》§12.4 预期结论表）给出分流边界的实验依据。
- **Q3** Screening 确定性直判边界是否合理：三分流中 PASS/REJECT 直判在评测集上的错判率是否低于红线、哪些规则动作应改 COMPLEX（含 §7 的品牌词命中实验点）。
- **Q4** 能否用实验证明 Agent 价值：可复现（确定性重放）+ 结论边界诚实标注（工具数据源 / LLM 模式 / 数据集版本，见 §3.4/§7）。

### 1.3 与其它文档的关系

| 文档 | 关系 |
|---|---|
| 00-system-design | 本文件是其 §7.6/§11–13 的可执行细化；sweep 校准结果**只作评测内部实验记录，不写回《00》§7.6 口径表**（§5.3） |
| Agent 契约（代码） | Agent 指标取数依据：`tool_call_history` 边际增益 4 字段、Tool Selection Accuracy 评测接口、`ReviewDecision` 终形状 —— 实现见 `src/pra/agent/state.py`、`src/pra/agent/tools_node.py`、`src/pra/domain/models.py` |
| 参数口径（代码） | 权威值在代码常量：预算 10/15、`EVIDENCE_MIN_SIM=0.70` / `EVIDENCE_STRONG=0.85`（`src/pra/tools/image_analysis/tool.py`）、`CONFIDENCE_ABSTAIN_THRESHOLD=0.7`（`src/pra/agent/guardrails/gate.py`）；校准结果**不回写生产常量**（§5.3） |
| 图装配（代码） | Agent scheme 的图装配与 Ablation 变体构造依据（工具注册层裁剪，见 §6）：`src/pra/agent/graph.py` |

---

## 2. 评测集 eval_dataset

### 2.1 规模与分布（分阶段落地；引用《00》§11.1）

评测集**分两阶段**落地，规模与目标各不相同（P-4 已拍板，记录见 §9）：

- **Smoke 集（≤10 条，先行）**：复用 demo case（P_88231 复古运动鞋等）作冒烟最小集，只验证 loader → harness → EvalRecord 链路可跑；**不混入正式统计**（见 §2.3）。
- **Phase 1 集（30–50 条）**：目标是**跑通完整框架**：Golden Dataset → Rule → Single-call LLM → Agent → Evaluator → Metrics → Console Report，验证三方案可比口径与指标模块正确性——**不是统计显著**。每条必须有明确 Ground Truth（expected.decision ∈ {PASS, REJECT}，见 §2.2/§4.1）；scene 仍按五类标注，便于分层冒烟，但数量小**不按比例验收**（分布达标自 Phase 2 起）。
- **Phase 2 正式集（300+ 条）**：作为出数与校准的正式集，五类分布如下（标注时按此比例分层抽样，避免案型偏斜；分布容差每类 ±5 个百分点内）：

| scene（场景标签） | 占比 | 说明 |
|---|---|---|
| `normal` 明确正常 | 20% | 规则与 Agent 都应 PASS |
| `violation` 明确违规 | 20% | 规则与 Agent 都应 REJECT |
| `boundary` 边界案件 | 30% | 规则拿不准、单信号弱 |
| `multi-signal` 多信号组合 | 20% | 需要多源交叉验证 |
| `evasion` 对抗/规避 | 10% | 刻意规避审核（核心 Hard Case） |

> HUMAN_REVIEW 期望案（Phase 2 起标注的 `SHOULD_ABSTAIN` 案，见 §4.4）**不强制占比**：每个 case 必须有明确 Ground Truth，**不为凑比例塞数据**（P-4）。

> **数据集局限（2026-09-11 实测，如实声明）**：
> - **单一标注者**：全部真值由生成器按与评测审查员同源的规则程序化标注（`annotation.labelers=["eval-phase2"]`），**无第二标注者交叉校验** → §3.3 同口径耦合的直接来源，只衡量实现一致性，不外推调查能力。
> - **无可见内容重复案**：`_dedupe_visible_rows` 保证 v2 不存在只差 `product_id/version/listing_time` 的重复行（改前 18 组、改后 0 组，20 行标题后缀改写）；但标题核心词仍跨案复用（表观多样性局限，非重复案）。
> - **`expected_tools` 空列表 = 未标注**（干净案、三方案一致 PASS），**不计入 Tool Selection 指标分母**，不得解读为「应调用 0 个工具」；「需调查才能判」的 AUTO 案已给非空期望（brand / category 空缺核验 30 案 = `ProductTool + MerchantTool`）。
> - **真 LLM 对照仅 v1 35 案单次抽样**（acc 0.200 / human_review_rate 0.771）：只作「真实 LLM 链路已跑通」的证明，**不是模型水平**；未跑 v2、未重复采样（重复采样仅限小 subset 看稳定性）。

### 2.2 eval_case 结构化标签 schema（引用《00》§11.2，细化字段清单）

文件格式建议：**JSONL，每行一条 eval_case**；版本由所在目录 / manifest 声明（见 2.3），不在每行重复全量元数据。

```jsonc
{
  "eval_case_id": "EC_00042",              // 唯一；建议 EC_<4位序号>，与业务 case_id 解耦
  "schema_version": 1,
  "scene": "evasion",                       // normal|violation|boundary|multi-signal|evasion（五类标签，2.1）
  "source_type": "REAL_DESENSITIZED",       // SYNTHETIC | REAL_DESENSITIZED | VARIANT（00 §13.2 三来源）
  "lineage": { "seed_case_id": "CASE_…", "mutation": "similarity:0.72→0.91" },  // 程序化变异的溯源，可选
  "hard_case": true,                        // 是否入选 Hard（00 §13.1 三选一，见 2.4）
  "hard_reason": ["rule_cannot_judge"],     // rule_cannot_judge | llm_evidence_gap | agent_can_discover（00 §13.1 三选一，可多）
  "input": { "...": "ProductReviewCase" },  // 《00》§2.1 快照；含 images[].ocr_text 与 screening_signals（基础输入，00 §12.0）
  "expected": {
    "decision": "HUMAN_REVIEW",             // PASS | REJECT（Phase 1 真值只用这两类）| HUMAN_REVIEW（Phase 2 的 SHOULD_ABSTAIN 案，见 P-3/§4.4）
    "risk_level": "HIGH",                   // LOW|MEDIUM|HIGH|NONE（《00》§7.5，独立于 decision）
    "risk_type": ["POTENTIAL_IP_RISK"],     // 受控词表（00 §7.3），供 risk_type 命中类指标
    "evidence": ["image_similarity>=0.85", "merchant_history>=5_removals"],  // 证据类型 + 标注时阈值口径（00 §11.2 注）
    "expected_tools": ["ImageAnalysisTool", "MerchantTool"],  // Agent 应调用的工具集合（Tool Selection Accuracy 真值）
    "applicable_policy": ["POLICY_3.2"]     // 政策条款 / 先例（REJECT 案必填，对齐 REJECT Gate 的"可引用依据"）
  },
  "annotation": { "labelers": ["A", "B"], "agreed": true, "notes": "" }  // 人工标注与交叉校验记录（2.3-4）
}
```

要点：
- **每条 eval_case 必须有明确 Ground Truth**：Phase 1 只允许 `expected.decision ∈ {PASS, REJECT}`；Phase 2 才允许 HUMAN_REVIEW（`SHOULD_ABSTAIN` 案，§4.4），不强制占比、不为凑比例塞数据（P-4）。
- 标签**不只是 PASS/REJECT**：`expected.evidence / risk_type / expected_tools` 使评测能区分"结论对但理由错"（《00》§11.2）。
- `expected.evidence` 里的阈值（如 `>=0.85`）是**标注时的证据口径**，须与运行时 `EVIDENCE_MIN_SIM/STRONG` 口径一致；阈值经 §5 sweep 校准变更后**同步修订标签**（《00》§11.2 注）。
- `input.images` 的素材引用（URL vs 本地 asset）与 OCR 文本是否内联，〔细化待定：实现者可自行收敛〕：影响 ImageAnalysis/OCR 工具的评测可复现性，建议 eval 专用 asset 目录 + URL 占位，避免外网依赖。

### 2.3 构造流程（对应《00》§13.2，落到"谁产、什么格式、存哪、版本怎么管"）

| 步骤 | 产出 | 责任人 / 工具 | 输出格式与存放 |
|---|---|---|---|
| 1 人工构造对抗样本 | 合成 case（核心场景：品牌模仿/规避，无 Logo 无品牌词但外观高度相似 + 规避史） | 标注者 + 脚本辅助生成 JSON 骨架 | 进 `scripts/` 生成器，产物写 eval JSONL |
| 2 真实案例改写脱敏 | 历史人工裁决案件 → 脱敏 benchmark 条目 | 标注者改写（PII/商家联系方式不落库，见《00》§8.2-3） | 手工维护 JSONL，git 入库前过脱敏校验 |
| 3 程序化变异 | 对模板做字段变异（改相似度/商家历史/OCR 冲突）制造边界 | `scripts/eval_dataset_gen.py`（确定性随机种子） | 脚本入库，产物可重复生成；变异参数记入 `lineage` |
| 4 人工标注 + 交叉校验 | 每条 `expected_decision + evidence + policy + expected_tools` | ≥2 人标注，冲突 case 修正至一致 | `annotation.agreed`；一致性统计写 manifest |

版本管理建议：
- 目录 `eval_data/v<N>/`：`eval_cases_v<N>.jsonl`（全量）+ `manifest.json`（schema_version、五类分布统计、Hard 占比、标注时阈值口径快照 `EVIDENCE_MIN_SIM/STRONG`、生成命令与 git commit、标注一致性）→ 评测报告必须记录所用数据集版本。
- 合成/变异由**脚本生成**（可复现、diff 友好）；真实脱敏改写以 **JSONL 入库**（可评审）；是否两者分文件（`synthetic/` 与 `handcrafted/`）〔细化待定：实现者可自行收敛〕。
- **Smoke 集（已拍板，P-4）**：复用已有 demo case（P_88231 复古运动鞋等）作冒烟最小集 ≤10 条，仅快速验证 loader/harness/EvalRecord 链路，**不混入正式分布统计**；真实改写脱敏种子随 Phase 2 正式集扩充。

### 2.4 Hard Case 占比与挑选（《00》§13.1）

- 三选一即算 Hard（标注 `hard_case=true` + `hard_reason`）：① Rule 无法判断/易误判（无硬规则命中但真实风险）；② Single-call LLM 证据不足/不稳定（决策依赖输入里**不存在**的信息）；③ Agent 多步调查可获得额外证据（工具调用能显著改变结论）。
- **验证方法**：①③需在 Rule / Agent 冒烟跑分后回填确认（先粗标、跑分后复核 hard_reason 是否成立），②由标注者按证据缺口判断。
- 占比终值（已拍板，P-4）：Phase 2 正式集目标 **≥15%**（`evasion` 10% + `boundary` 中符合三选一者）；Phase 1（30–50 条）不设硬性占比，以覆盖五类、跑通框架为准。

### 2.5 数据集划分（sweep 校准与报告隔离）

- 一次划分成 **report 集（主跑分报告）** 与 **validation 集（§5 sweep 校准 operating point）**，按 scene 分层抽样，比例建议 70/30。
- 报告必须声明 operating point 取自 validation；report 集只在定稿后跑一次全量，避免"调参调到报告集"。
- 划分自 **Phase 2（300+）** 起执行；Phase 1（30–50 条）以全集跑通框架与指标口径即可，不在 Phase 1 集上做 sweep 选点（sweep 排在 Phase 1 框架跑通之后，§5/§8）。

---

## 3. 三方案定义与 Harness 结构

### 3.1 共享基础输入（《00》§12.0，公平性前提，不可违背）

```
基础输入 = ProductReviewCase 的商品事实快照：
  标题 / 描述 / 属性 / 类目 / 品牌 / SKU / 图片 / 机审已产出 OCR 文本（images[].ocr_text）/ screening_signals
```

- Rule：只允许对基础输入跑确定性规则。
- Single-call LLM：基础输入一次性全部交给 LLM（一次调用输出决策 JSON），**不给**商家历史/案例库/政策库（那是 Agent 经工具"调查"得来的，否则作弊，《00》§12.2）。
- Agent：允许按证据缺口动态调查——经 Tools 获取基础输入之外的证据。
- harness 必须**逐 scheme 校验**未越权（如 Single-call LLM 的 prompt 里不注入工具检索结果），防止"看到材料不一样"的作弊质疑。

### 3.2 Rule baseline（P-1 已拍板：选项 (a) 为主，(b) 本期不做）

- **主 baseline = 复用线上 `pra.screening` 三分流**：PASS→PASS、REJECT→REJECT、**COMPLEX→HUMAN_REVIEW**（评测语义：无 Agent 时复杂案只能人工，见 §4.5）。
  - 评测对象 = 线上真实判定器，结论可直接外推；
  - **口径差异必须注明**：评测里的 "COMPLEX→人工" 与线上 "COMPLEX→Agent" 是**不同口径**——线上复杂案交给 Agent 终裁，Rule baseline 代表"没有 Agent 时复杂案只能转人工"，报告须写明（§4.5/§7.1）；
  - Rule 行为随 screening 修正集变动：报告记录 screening 行为快照（git commit / 规则语义版本），正式基线结论在修正集合入后重跑（§7.1/§8）。
- **选项 (b) 独立二分 rule（命中→REJECT、否则 PASS）本期不做**（P-1 拍板）；后续如需作为"纯规则上限"的敏感性补充，再单独加跑，**不进主对比表**。

### 3.3 Single-call LLM（Phase 1 必跑中层方案 + 消融变体）

- 定位：三方案对照的**中层**（Rule Baseline → Single-call LLM → Multi-step Review Agent，Phase 1 三方案全部进框架），用来回答核心问题：**"Agent 的收益来自 LLM 本身，还是来自多步调查 / Tool / RAG / Evidence Aggregation？"**
- 执行语义：一次调用，基础输入全量（§3.1）+ 少量背景 prompt → 结构化决策 JSON（decision/risk_level/risk_type/decision_confidence/evidence 摘要/policy 候选）；**不给**商家历史/案例库/政策库（那是 Agent 经工具"调查"得来的，否则作弊，《00》§12.2）。
- 输出 schema 复用 ReviewDecision 子集（`src/pra/domain/models.py` 形状的子集）；`budget_used/overrides` 恒空/不适用；决策 JSON 自带置信度后处理（REJECT 候选 `decision_confidence < CONFIDENCE_ABSTAIN_THRESHOLD` → HUMAN_REVIEW，见 §4.5），即 **Single-call 也有 HUMAN_REVIEW 语义**。
- Phase 1 实现：scripted/mock LLM + InMemory；三方案统一归一为 `ReviewDecision` 形状的 EvalRecord（§3.6），进**同一 Evaluator / Metrics**，保证可比。
- **消融变体（后续 Ablation，Phase 2，见 §6）**——回答"给 Single-call 更多上下文能提升多少？Agent 的额外收益是否来自主动调查、而非只是看到更多文本"：
  - **2a** Single-call + Raw Input（仅商品原始数据，不给任何预塞知识）；
  - **2b** Single-call + RAG-in-prompt（政策/先例**预塞 prompt**，仍不给工具）；
  - **2c** Multi-step Agent + 主动调查（标准 Agent，§3.4）。

### 3.4 Agent（真实执行语义；Phase 1 运行模式已拍板）

- 走 **`build_agent_graph`**（hypothesize→plan→tools→reevaluate→decide，状态 `AgentState`，预算 10/15/40000/30000 Guardrail）；终态以确定性 overlay 后的 `ReviewDecision`（`src/pra/domain/models.py`）为判决策略真值；HUMAN_REVIEW 是三个 Decision Gate 之一（abstention 清单语义），即 **Agent 有 HUMAN_REVIEW 语义**。
- **运行模式（P-2 已拍板）**：
  - **Phase 1 默认：scripted Agent + InMemory 种子数据**（Case/Policy/Merchant 现状）——确定性、可重复、**CI 可回归**（同 case 重跑同结果，《00》§11.4）；
  - **Real LLM Evaluation 排后续阶段（Phase 3，§8）**：真实 LLM + 真实工具数据源（RAG 未接前不可用），复核 scripted 结论（尤其 Tool Selection / Evidence Sufficiency 等依赖 LLM 行为的指标）。
- **结论边界（报告必声明，P-2 口径）**："当前结果主要验证 **Agent Workflow、规则协同与 Evaluation Framework**，不代表真实 LLM 最终能力；**InMemory 种子覆盖有限，可能低估 Agent 上限**。"（另见 §7.2）

### 3.5 Harness 代码落点（src/pra/evaluation/ 模块划分）

```
src/pra/evaluation/
├── dataset/
│   ├── schema.py          # EvalCase Pydantic 模型 + 校验（§2.2 JSON 对齐；标签完整性/受控词表）
│   └── loader.py          # JSONL 读取、manifest 解析、report/validation 划分、scene 分层统计
├── harness/
│   ├── base.py            # EvalContext（阈值常量/预算/LLM 模式/工具数据源）+ SchemeRunner 抽象 + EvalRecord（见 3.6）
│   ├── rule_scheme.py     # P-1 已拍板：薄封装 pra.screening 三分流（PASS/REJECT/COMPLEX→HUMAN_REVIEW）；(b) 独立二分 rule 本期不做（RuleBaseline）
│   ├── single_call_scheme.py  # prompt 构造 + 单次调用 + JSON 校验降级（与 Agent 的 LLM 调用壳同款：重试 1 次，失败→HUMAN_REVIEW；低置信 REJECT 候选→HUMAN_REVIEW）（SingleCallScheme）
│   └── agent_scheme.py    # build_agent_graph + scripted/real LLM + 工具数据源注入 + 结果转录（AgentScheme）
├── metrics/
│   ├── business.py        # §4.1：Accuracy/Precision/Recall/FPR/FNR + human_review_rate/automation_coverage（DecisionMetrics/DecisionEvaluator；命名以 §4 为准，无 HRR 缩写）
│   ├── abstention.py      # §4.4：abstention_rate/abstention_recall/wrong_auto_decision_rate（AbstentionMetrics/AbstentionEvaluator）
│   ├── agent.py           # §4.2：Tool Selection Accuracy/Evidence Sufficiency/…/Budget Utilization —— **规划未实现**（本模块尚不存在）
│   └── engineering.py     # §4.3：llm_calls/tool_calls/tokens/latency 均值与分位数 —— **规划未实现**（本模块尚不存在）
├── runner.py              # 数据集遍历 + 方案调度 + 结果汇总（EvaluationRunner/EvaluationResult，含 cost 均值）
├── ablation.py            # §6：Ablation（方案级 2a/2b/2c + 组件级；AblationRunner）
├── regression.py          # §8：确定性回归（baseline digest 记录与比对；RegressionReport）
├── report.py              # 汇总 → markdown/json 报告；含按 scene 分层表 + 《00》§12.4 预期结论对照 + 结论边界声明
└── sweep.py               # §5：threshold sweep 驱动（只改配置；Phase 2，见 §5；ThresholdSweepRunner）

scripts/
├── eval_dataset_gen.py    # §2.3 程序化变异/合成入口（确定性种子）
└── run_evaluation.py      # 跑分入口：load → N schemes → metrics → report（子命令 --scheme/--split）
```

> **命名注**：本节是**设计期规划**；**已实现部分以 `src/pra/evaluation/` 的实际结构为准**（上面目录树已按实现对齐：
> `EvalRecord` 与 `EvalContext` / `SchemeRunner` 同落 `harness/base.py`，abstention 指标落 `metrics/abstention.py`，
> 另含本节最初未列的 `runner.py` / `ablation.py` / `regression.py` / `report.py` / `sweep.py` / `dataset/`）。
> 契约标识符 `eval_case` / `SchemeRunner`×3 / `EvalRecord` / `metrics.business` 的语义不变；实现层实际使用的名字为
> `EvalCase` / `RuleBaseline`·`SingleCallScheme`·`AgentScheme` / `EvalRecord` / `metrics.business`
> （`DecisionEvaluator` / `AbstentionEvaluator`），另有 `EvaluationRunner`（`runner.py`）、`AblationRunner`（`ablation.py`）等。
> **不得引入与本节平行的新抽象命名**；`§4.2/§4.3` 的 agent / engineering 指标（`metrics/agent.py`、`metrics/engineering.py`）
> **属规划项、尚未实现**，落地时按本节与 §4.2/§4.3 口径补。

SchemeRunner 契约（伪代码；实现者照此写，不强求框架）：

```python
# harness/base.py
class SchemeRunner(ABC):
    name: str                       # "rule" | "single_call_llm" | "agent"
    async def run(self, case: EvalCase, ctx: EvalContext) -> EvalRecord: ...

# EvalContext：注入配置（阈值/预算/LLM 模式/工具数据源/随机种子）——sweep 只改这里的常量（§5）
# EvalRecord：见 3.6 —— metrics 层只吃 EvalRecord，不直接吃 DB/State
```

### 3.6 同一次运行产出可比 metrics（《00》§11.4）

- 三个 scheme 跑**同一份 eval_dataset**，每个 (case, scheme) 产出**统一 EvalRecord**，metrics 层只吃它 → 保证可比：

```jsonc
{
  "eval_case_id": "EC_00042", "scheme": "agent",
  "decision": "REJECT",            // PASS|REJECT|HUMAN_REVIEW（三分类归一，映射见 4.5）
  "risk_level": "HIGH", "risk_type": ["POTENTIAL_IP_RISK"],
  "decision_confidence": 0.87,
  "evidence": [ { "type": "IMAGE_SIMILARITY", "value": "similarity=0.91…", "extra": {"similarity": 0.91} } ],
  "policy": ["POLICY_3.2"],
  "tool_calls_actual": ["ImageAnalysisTool", "MerchantTool"],   // agent 用；rule/llm 为 []
  "trace": { "plan_outputs": [ … ], "tool_call_history": [ … ] },  // 边际增益 4 字段原样转录（scripted 模式）
  "cost": { "llm_calls": 8, "tool_calls": 5, "tokens": 18000, "latency_ms": 9200 }
}
```

- Agent 的 EvalRecord 由 **AgentState/review_trace/review_result 字段转录**（review_trace(PLAN).output_json → plan_outputs；AgentState.tool_call_history → trace；decision_json.budget_used → cost）；rule/llm 的 EvalRecord 由各自执行结果构造。
- **DB 落库非必需**：评测以内存 EvalRecord 为主（快、可并行、不污染业务表）；DB 版（走 run_and_persist/SCREENING_DIRECT 真落库）作集成测试可选路径〔细化待定：实现者可自行收敛〕。
- **确定性重放**：scripted 模式要求同 case 重跑产出**逐字节一致**的 EvalRecord（mock/录制工具结果，随机种子固定），作为 Phase 1 验收断言（§8）。

---

## 4. 指标口径（每个指标给分子/分母/取数字段）

> 取数一律以 **EvalRecord** 为准（3.6）；与 DB 字段（review_trace / review_result.decision_json / decision_json.budget_used）
> 的映射仅用于集成抽查。字段口径见 `src/pra/agent/state.py`（tool_call_history）、`src/pra/domain/models.py`（ReviewDecision）、《00》§10.3。

### 4.1 业务指标（Phase 1 二分类五指标 + 转人工观测量；abstention 语义见 §4.4）

Phase 1 Golden Dataset 只有 PASS/REJECT 真值（P-3/P-4），故业务主指标是**二分类五指标**，另报转人工观测量：

| 指标 | 计算口径（Phase 1） | 取数字段 | 备注 |
|---|---|---|---|
| Accuracy（决策准确率） | 二值真值案（expected∈{PASS,REJECT}）上 decision==expected 的占比；**预测 HUMAN_REVIEW 计为错** | decision × expected.decision | 实现口径（2026-09-09 Q1 拍板 (b)），口径注见下表 |
| Precision（精确率） | 预测 REJECT 且 expected=REJECT / 预测 REJECT | 同上 | 误拒直接伤害商家 |
| Recall（违规召回） | 预测 REJECT 且 expected=REJECT / expected=REJECT | 同上 | 违规漏放伤害平台 |
| False Positive Rate（FPR，误杀率） | expected=PASS（正常）中自动终裁为 REJECT 的比例 | 同上 | **防误伤红线、Phase 1 重点观察**（《00》§7.2-2）；sweep 主观察曲线之一 |
| False Negative Rate（FNR，漏放率） | expected=REJECT（违规）中自动终裁为 PASS 的比例 | 同上 | 与 Recall 互补 |
| human_review_rate | 输出 HUMAN_REVIEW 的 case 占全部 case 的比例 | decision | 转人工占用（人工负担）；Rule 的 COMPLEX 映射与 Agent/Single-call 的 abstention 都计入 |
| automation_coverage | 1 − human_review_rate（自动终裁占比） | decision | 自动化覆盖率；与 FPR/FNR **必须并读**（§4.4 核心口径） |

> **口径注（2026-09-09 Q1 拍板 (b)，实现为准）**：二值真值案 = expected∈{PASS,REJECT} 的案
> （v2 = 274；46 条 SHOULD_ABSTAIN 真值不参与本表分母，其质量由 §4.4 abstention 指标承接）。
> Accuracy 分母 = 全部二值真值案，**预测 HUMAN_REVIEW 计为错**——这是比"auto 子集口径"更严的
> 工程口径：保守转人工与决策错误同罚，迫使"自动化 + abstention 指标并读"（§4.4 核心口径），
> 否则"大量转人工"的方案会在 Accuracy 上显得准。对照参考（auto 子集口径，即分母排除 HUMAN 预测、
> 只算 decision∈{PASS,REJECT} 的案）：v2 rule 0.825 / single 0.866 / agent 1.000 —— 两口径的
> 差别就是 abstain 惩罚量（code 口径 rule 0.380 / single_call_llm 0.518 / agent 0.964；v2 320 案，
> 二值真值 274，出处 `scripts/run_evaluation.py --data eval_data/v2/cases_v2.jsonl`），报告已并排披露（abstention 区
> abstention_rate + wrong_auto_decision_rate 量化该惩罚）。Precision/Recall/FPR/FNR 分母只含
> 对应真值类、HUMAN 预测不计入（保持 §4.4 "HUMAN 不当第三真值类"语义）。abstention 质量评估
> （该转人工是否转了、自动决策是否安全）见 §4.4（AUTO_DECIDABLE / SHOULD_ABSTAIN）。

### 4.2 Agent 指标（仅 agent scheme 有意义）

> ⚠️ **本节指标为规划口径，代码尚未实现**（`src/pra/evaluation/` 内无对应实现；§3.5 目录树
> 已将 `metrics/agent.py` 标为「规划未实现」）——表中「计算口径」是设计约定，不是当前产出。

| 指标 | 计算口径（建议） | 取数字段 | 备注 |
|---|---|---|---|
| Tool Selection Accuracy | 每 case：`expected_tools ⊆ 实际调用集合` 且实际调用尽量少；聚合 = 命中案 / 总案 | expected.expected_tools × tool_calls_actual | 真值来自 eval_case 标签；取数接口见 `src/pra/agent/guardrails/schemas.py`（PlanOutput）+ `src/pra/agent/state.py`（tool_call_history） |
| Evidence Sufficiency | ① expected.evidence 的**证据类型**被实际 evidence 覆盖比例；② REJECT 案是否满足 REJECT Gate 前置（有可引用依据 `CITABLE_TYPES`、无关键矛盾） | expected.evidence × evidence | 精确公式〔细化待定：实现者可自行收敛〕 |
| Reasoning Correctness | 结论对但推理错：risk_type 命中率（expected.risk_type ⊆ 输出）+ risk_level 档位一致 + 抽样人工复核结构化理由 | risk_type/risk_level/trace | 自动化近似无法全覆盖 → 抽样人工复核子集（标注） |
| Marginal Evidence Gain / Investigation Efficiency | 逐 tool_call 记 `after_confidence−before_confidence` 与 `evidence_added` 非空、`decision_changed`；效率 = Σ(新增证据数 + 决策翻转权重) / 有效 Tool Calls | trace.tool_call_history 边际增益 4 字段（`src/pra/agent/state.py`） | 暴露"为调查而调查"；加权口径〔细化待定：实现者可自行收敛〕 |
| Budget Utilization | 四组占用率：llm_calls/tool_calls/tokens/latency 各 ÷ 上限（10/15/40000/30000），报均值与分布 | cost + decision_json.budget_used | 证明 Budget 是 Guardrail 非目标（《00》§10.3/§11.3） |

### 4.3 工程指标（成本与效率，三方案同口径）

> ⚠️ 同 §4.2：**规划口径，尚未实现**（`metrics/engineering.py` 为规划件）；当前报告中的
> 成本信息来自 `cost` 字段的均值直出，不含本节设想的分位数/占用率分布。

- LLM Calls（平均/P95）、Tool Calls（平均/P95）、Token Usage、P50/P95 Latency、单 Case 成本。
- 一律报**分位数与分布**（不只均值），用于回答"Agent 贵在哪、是否值得"（《00》§11.3）；按 scene 分层报。

### 4.4 HUMAN_REVIEW / Abstention 口径（P-3 已拍板；分阶段语义）

- **总口径**：三方案统一三分类输出 PASS/REJECT/HUMAN_REVIEW（§4.5），但 **HUMAN_REVIEW 不当普通第三分类**混进 Acc 等混合指标——它表示 **abstention / 转人工能力**，按两阶段评估：
  - **Phase 1**：真值只有 PASS/REJECT → 只在二分类真值上算 Accuracy / Precision / Recall / FPR / FNR（§4.1，**重点看 FPR（误杀）**）；无 abstention 质量指标（没有 SHOULD_ABSTAIN 真值可评价"转得对不对"）。
  - **Phase 2**：真值引入两类 abstention 语义——
    - `AUTO_DECIDABLE`（expected=PASS/REJECT，本可自动判）：评价**正确自动决策**（自动终裁且与真值一致）与 **wrong_auto_decision_rate** = 自动终裁（输出 PASS/REJECT）中与真值不符的比例——回答"自动判断是否准确、安全"；
    - `SHOULD_ABSTAIN`（expected=HUMAN_REVIEW，应转人工）：评价**正确转人工**与 **abstention_recall** = 输出 HUMAN_REVIEW 的 SHOULD_ABSTAIN 案 / 全部 SHOULD_ABSTAIN 案——"该转人工的克制地转了"；其中被自动终裁的 SHOULD_ABSTAIN 案（漏转人工）即**危险误自动**，是 abstention_recall 的分子缺口。
- **指标命名（不用 HRR 缩写；代码/文档统一用下列五个名字）**：
  - `human_review_rate`：输出 HUMAN_REVIEW 占全部 case 的比例（§4.1）——人工占用；
  - `automation_coverage`：1 − human_review_rate（§4.1）——自动化覆盖面；
  - `abstention_rate`：AUTO_DECIDABLE 案上输出 HUMAN_REVIEW 的比例——"本该自动判却转人工"的**过度保守 abstention**（越高说明方案越保守）；
  - `abstention_recall`：见上（SHOULD_ABSTAIN 正确转人工的召回）；
  - `wrong_auto_decision_rate`：见上（自动终裁中的错误占比，安全/准确侧）。
- **核心价值口径**：评测**不是为降低转人工而牺牲安全**；真正的目标是——"**在风险可控、自动判断准确（FPR/FNR/wrong_auto_decision_rate 可控）的前提下，Agent 能否识别 Rule 无法判断的复杂案、并经调查把其中一部分安全自动化**"。因此**自动化（转人工降低 / automation_coverage 上升）必须与 FPR/FNR 一起看**：允许方案把复杂案克制地转人工（human_review_rate 高、FPR 低），但只有当它能安全地把其中一部分自动化（automation_coverage 上升且 FPR/FNR/wrong_auto_decision_rate 不恶化）时，才是 Agent 价值的证据。
- SHOULD_ABSTAIN 案标注原则见 §2.1/§2.2（每个 case 有明确 Ground Truth；不强制占比、不为凑比例塞数据）；Phase 2 按本节语义实现 abstention 指标评测（落 `metrics/abstention.py`，§8/§3.5；模块落点见 §3.5 命名注）。

### 4.5 三分类输出空间（映射已定稿）

三方案统一 **PASS/REJECT/HUMAN_REVIEW 三分类输出**、进同一 EvalRecord（§3.6）与同一 Evaluator——映射是**各方案的执行语义**而非事后补对齐，报告随 EvalRecord 记录每 case 的映射来源：

- **Rule baseline（P-1(a)）**：PASS→PASS、REJECT→REJECT、**COMPLEX→HUMAN_REVIEW**（评测语义 = "不可自动判"：无 Agent 时复杂案只能人工）；报告须注明该 "COMPLEX→人工" 与线上 "COMPLEX→Agent" 是**不同口径**（§3.2/§7.1）。
- **Single-call LLM**：REJECT 候选 `decision_confidence < CONFIDENCE_ABSTAIN_THRESHOLD(0.7)` 经确定性后处理记 HUMAN_REVIEW（与 Agent 的 REJECT Gate 同源口径；§3.3）。
- **Agent**：三个 Decision Gate 的终态之一（PASS/REJECT Gate + HUMAN_REVIEW abstention 清单），无额外映射。
- 映射影响 FPR / human_review_rate trade-off 的形状，报告须写明所用映射；转人工与安全并读（§4.4 核心口径）。

---

## 5. Threshold Sweep（P-5 已拍板：单参数、排在 Phase 1 之后）

> sweep **排在 Phase 1 框架跑通之后**（先有三方案可比指标与 §4 口径，再谈校准；落 Phase 2，§8）。
> P-5 已拍板：第一轮**只 sweep Evidence 阈值、单参数**，不做多参数联合 Grid Search。

### 5.1 扫哪些常量（只动配置，不动判定逻辑；《00》§11.5）

| 常量 | 第一轮口径（P-5） | 实际影响路径（2026-09-09 Q4 拍板 (b)，实证收窄） |
|---|---|---|
| `EVIDENCE_MIN_SIM` / `EVIDENCE_STRONG` | **只 sweep Evidence 阈值一维**：0.60/0.65/0.70/0.75/0.80/0.85/0.90；每次只动一个常量、另一个取当前默认（0.70/0.85）固定 | **只作用于评测确定性审查员（EvalScriptedLLMBackend）读证据视图的阈值**（sweep.py 经 EvalContext 注入）；**生产侧 tools_node quality_filter（0.70）/ REJECT Gate 强档（0.85）不随 sweep 变化**，Rule baseline 不读相似度（无图片相关路径）——sweep **不校准生产常量**（§5.3 已修订）；另见下方数据带限制注 |
| `CONFIDENCE_ABSTAIN_THRESHOLD` | **固定 0.7，第一轮不扫** | REJECT Gate 安全门槛 → 主要影响 Agent 与 Single-call LLM 的 abstention |

> **数据带限制（2026-09-09 实证）**：当前种子世界图片相似度权重仅两簇——{0.72,0.73}（弱相似）与
> {0.90,0.91,0.93,0.95}（强相似），**(0.73,0.90) 区间无任何数据**。实测 v2 320 案：EVIDENCE_STRONG
> 全网格 0.60–0.90 **0 决策差异**、EVIDENCE_MIN_SIM 仅 0.90 档 14 案变化（v1 仅 2 案）。因此
> **曲线平不代表生产阈值不敏感**——是数据带没覆盖；任何 operating point 结论都受此限制，报告必须
> 注明。若要把 sweep 变成生产阈值的校准工具，须先把注入下沉到 tools_node/gate 实际常量读取点并补
> 0.75–0.88 区间 family（另行立项，Q4 本轮未做）。

- **不做多参数联合 Grid Search**（EVIDENCE×CONFIDENCE 乃至 MIN×STRONG 全组合都不做）——变量过多、无法判断效果来源（P-5）。
- 第一轮观察 **Accuracy / Precision / Recall / FPR / FNR / human_review_rate / automation_coverage（自动化覆盖率）** 随 Evidence 阈值的变化，判断效果来源后，再决定是否联合校准 Confidence（P-5）。
- **实现约束**：阈值全部经配置层注入（EvalContext），评测代码与判定逻辑**不得内联阈值常量**；sweep 只是换 EvalContext 重跑 agent/rule scheme，Rule 与 Agent 共用同一份配置快照。
- sweep 第一轮两个 Evidence 常量各自的扫描次序等细节〔细化待定：实现者可自行收敛〕（其余常量取默认固定值即可）。

### 5.2 观察哪些曲线、在哪定 operating point

- 主曲线：§5.1 七项指标对 Evidence 阈值的 trade-off（每 scheme 一组；重点看 Agent 与 Rule baseline）；自动化相关（human_review_rate / automation_coverage）须与 FPR/FNR 同图呈现（§4.4 核心口径）。
- 只允许在 **validation 集**（§2.5）上选取 operating point；选点优先级建议：先压 FPR（防误伤商家红线，《00》§7.2-2），再保 Recall（违规召回），human_review_rate / automation_coverage 作为可接受成本。
- operating point 定义 = (`EVIDENCE_MIN_SIM`, `EVIDENCE_STRONG`) 一组值（第一轮 `CONFIDENCE_ABSTAIN_THRESHOLD` 固定 0.7），记录选点理由与所选点的观测指标值。

### 5.3 校准结果回写（2026-09-09 Q4 拍板 (b) 修订）

- **当前 sweep 只观测评测确定性审查员的读证据视图，不写回生产常量**（tools_node quality_filter /
  gate 强档未被 sweep 观测过——把选点写进生产属于"写进未测层级"，禁止）。旧版"回写《00》§7.6 /
  注入下沉到生产常量读取点并补
  中间相似度带数据后**才可启用（另行立项）。
- sweep 结论（曲线/选点）作为**评测内部实验记录**，附 §5.1 数据带限制注后，可与 docs §8 一并引用。
- 报告必须附 **sweep 曲线**而不是只报最终点（《00》§11.5 精神：证明阈值是"选"出来的，不是拍脑袋
  ——在当前数据带下曲线近平，如实呈现并注明限制）。

---

## 6. Ablation Evaluation（Phase 2；引用《00》§13.4）

- 回答"Agent 每个组件/每层能力是否真的必要"，两类消融跑**同一 eval_dataset**：
  - **方案级（Single-call 的上下文 vs Agent 的主动调查，§3.3 变体）**：2a（Raw Input）→ 2b（+RAG-in-prompt 预塞政策）→ 2c（Multi-step Agent + 主动调查）——先量化"给 Single-call 更多文本能提升多少"，再看"Agent 额外收益是否来自主动调查而非只是看到更多文本"；
  - **组件级（同图结构逐组件去掉）**：Full Agent（基线）/ −RAG（无 CaseSearch+PolicySearch）/ −MerchantTool / −CaseTool / −ImageTool。
- **Phase 2 做，不阻塞 Phase 1 三方案可比跑分**（三方案主对比之后；真实 RAG 接入后组件级消融更可信）。
- 实现只做装配层裁剪，不动判定逻辑与评测集（差异唯一归因）：方案级只换装配（Single-call prompt 是否预塞政策 / Agent 是否给工具）；组件级只做"图装配层不给该工具注册 / plan prompt 不注入该工具描述"，基于 04-graph-design 的注册表。
- 判定规则：去掉后指标几乎不变 → 组件必要性存疑；显著变差 → 必要能力（《00》§13.4）。
- 前置依赖：工具数据源至少达"能区分有/无该工具证据"的种子覆盖（InMemory 种子里查不到先例时，−CaseTool 必然无差异，结论失效——见 §7.2）。

---

## 7. 与 screening 修正集 / RAG 现状的关系

### 7.1 Rule baseline 依赖 screening（修正集进行中）

- Rule baseline 选项 (a) 复用 `pra.screening`，其行为随**并行修正集**变动（空规则集不得静默 PASS、brand 空缺不得直判 PASS、品牌词加词边界、品牌词命中 REJECT→COMPLEX 等）。
- **本文档刻意不与具体规则对齐**：harness 不做规则动作断言，只按"运行时当前行为"取数；报告必须记录 **screening 行为快照**（git commit / 规则语义版本）。
- 修正集未完成前 Rule baseline 结论会失真（漏放/误杀 → "Agent 比 Rule 强"可能是假象）：Phase 1 的三方案可比跑分照常执行，但把该失真列入**结论边界**（§3.4）；**正式基线结论在修正集合入后重跑并纳入 Regression**（§8 Phase 2）。
- Fix 5（品牌词命中 REJECT vs COMPLEX→Agent 上下文终裁）本身是评测实验点：评测集需含"品牌词命中但可能合法"样本，报告给出两种规则动作下的对比行；其与评测口径 "COMPLEX→人工"（§3.2/§4.5）的差异须一并注明。

### 7.2 Agent 工具数据源的结论边界（评测世界 InMemory vs 生产真链路）

- **评测世界**：Agent scheme 的 CaseSearch / PolicySearch / Merchant 以 **InMemory 种子数据**运行（scripted 模式，§3.4 已拍板为 Phase 1 默认）；评测侧可经 `EvalContext.tool_world="rag"`（或显式装配 `data_source="rag"`）切到真实检索，向量库为 ChromaDB + LlamaIndex（见 [docs/00-system-design.md](00-system-design.md) 的 RAG 节与 `src/pra/rag/chroma_backend.py`）。
- **生产 / HTTP 入口已接真实链路**：`build_production_tools()` 注入商品/商家 MySQL 与案例/政策的真实 RAG（`Lazy*Index` 惰性构建：装配期零 import/零 IO，首次检索才建库连 Chroma）。故**本文所有数字仍是 InMemory 评测世界口径**，不得读成生产链路成绩。
- **工具集口径**：本文所有 Agent 数字均出自**评测世界的 5 个工具**（`make_eval_world_tools()`：Product / ImageAnalysis / Merchant / CaseSearch / PolicySearch），比生产 `build_tools()` 的 6 个**少一个 `OCRTool`** —— 评测集 `expected_tools` 从不含它、机审 OCR 文本近乎全空，补进去只是多一个拿不到数据的工具。两者清单各自维护、**无"同构"约束**，Tool Selection Accuracy 真值按该 5 工具集合计。
- **报告必带边界声明（P-2 口径）**："当前结果主要验证 **Agent Workflow、规则协同与 Evaluation Framework**，不代表真实 LLM 最终能力；InMemory 种子覆盖有限，**可能低估 Agent 上限**。"（与 §3.4 同文）
- 局限：种子里缺失的真实先例/完整规避史 Agent 取不到 → 多信号与对抗类案的 Evidence Sufficiency、Marginal Evidence Gain 等指标会**低估上限**。
- 真实 RAG/Merchant 接入后以 real 模式复核（Phase 3，§8）。

---

## 8. 验收标准与里程碑（分阶段主线，P-4/P-5 拍板后）

| 阶段 | 内容 | 交付 / 验收 |
|---|---|---|
| **Phase 1**（当前） | eval_data schema + **30~50 条 golden cases**（PASS/REJECT 真值，§2.1）→ Rule / Single-call / Agent 三 **SchemeRunner**（§3.2–3.4）→ 决策指标评测（Accuracy/Precision/Recall/FPR/FNR，§4.1）→ Console Report → **确定性重放断言** | **三方案在同一数据集上可比跑分**（含 metrics 金标准小样本单测）；scripted 同 case 重跑**逐字节一致**；报告含结论边界声明（§3.4/§7.2）与 screening 行为快照（§7.1）；FPR 为 Phase 1 重点指标 |
| **Phase 2** | 300+ 正式集（§2.1 五类分布）；Evidence 侧指标评测（Evidence Sufficiency / Marginal Evidence Gain，§4.2）；**Ablation**（2a/2b/2c + 组件级，§3.3/§6）；**abstention 评测**（AUTO_DECIDABLE / SHOULD_ABSTAIN，§4.4）；**Threshold Sweep**（Evidence 单参数、CONFIDENCE 固定 0.7，§5）；**Regression** | sweep 曲线与 operating point（validation 集）**只作评测内部实验记录，不写回生产常量**（§5.3）；abstention 五指标在正式集出数；修正集合入后重跑 Rule 基线并纳入回归（§7.1） |
| **Phase 3** | **Real LLM Evaluation**；LLM-as-a-Judge（如必要）；Regression Report | real 模式复核 scripted 结论（§3.4/§7.2）；完整报告逐行对照《00》§12.4 预期结论、明确回答 Q1–Q4（§1.2），附全部结论边界标注 |

> **Phase 3 实现状态（2026-09-08 已落地首次复核）**：`LiteLLMBackend`（`pra/agent/litellm_backend.py`，
> 四节点完整 prompt 见 `pra/agent/llm_prompts.py`）经 `AgentScheme(llm=..., budget_limits=...)`
> 接入评测；`scripts/run_evaluation_real.py` 跑 real vs scripted 对照（需 API key；real 臂评测侧
> 放宽 max_latency_ms 墙钟护栏 —— 生产 30s 对真实 LLM 过紧，见脚本 docstring 与 README「评测与
> 结论」）。**v1 35 案首次 real 单次抽样已出**（deepseek 网关）：human_review_rate 0.771 / acc 0.200，27/35 转人工
> 全部由确定性 Gate 归因（R3_BUDGET_EXHAUSTED×19 收敛效率、R3_HYPOTHESES_INDISTINGUISHABLE×7）；
> 实证 scripted 高分含「标注-审查员同口径」耦合，real 显著更保守且暴露 1 例无视觉证据的幻觉性
> SUPPORTED 误杀（EC_0007）。**该抽样只验证链路与暴露迭代方向，不代表模型固定水平**；剩余为
> prompt/约束迭代（禁重复假设、外观类假设须引用 IMAGE_SIMILARITY），LLM-as-a-Judge 按需。

> **最终主线**：**Rule Baseline → Single-call LLM → Multi-step Agent → Ablation → Abstention → Threshold Sweep → Regression**。
> Smoke 集（≤10 条 demo case）先行验证 loader/harness/EvalRecord 链路，不混入正式统计（§2.3）。
> Phase 1 是"能跑且可比"的门槛，Phase 2/3 才回答 Q1–Q4；Phase 1 与 Phase 2 数据构造可并行推进。

---

## 9. 拍板记录（P-1~P-5 已拍板，供追溯）

> 本节把 P-1~P-5 的**最终决策**与**原建议**对照存档；正文已按"最终决策"落实为定稿口径，不再标 [待拍板]。

| # | 议题 | 原建议 | 最终决策（权威口径） | 落点 |
|---|---|---|---|---|
| P-1 | Rule baseline 语义 | (a) 复用 `pra.screening` 三分流为主；(b) 独立二分 rule 作"纯规则上限"敏感性补充（可选、不进主对比表） | **拍板 (a)**：主 baseline = 线上 `pra.screening` 三分流——PASS→PASS、REJECT→REJECT、**COMPLEX→HUMAN_REVIEW**（评测语义：无 Agent 时复杂案只能人工；报告注明与线上 "COMPLEX→Agent" 是不同口径）；**(b) 本期不做**，后续如需作敏感性补充再单独加、不进主对比表 | §3.2/§3.5/§4.5/§7.1 |
| P-2 | Agent 评测模式与数据源 | scripted Agent + InMemory 为默认；real 模式二期复核 | **Phase 1 默认 scripted Agent + InMemory 种子数据**（确定性、可重复、CI 可回归）；**Real LLM Evaluation 排后续阶段**；报告必声明结论边界（验证 Agent Workflow/规则协同/Evaluation Framework，不代表真实 LLM 最终能力；InMemory 种子覆盖有限可能低估 Agent 上限） | §3.4/§7.2/§8 |
| P-3 | HUMAN_REVIEW / Abstention 语义与指标 | 三分类统一 + HRR 细分三率 + HUMAN 期望案单列 abstention 质量 | 三分类统一但 **HUMAN_REVIEW 不当普通第三分类**（评估 Agent 的 abstention/转人工能力）；指标命名**去 HRR 缩写**，统一 `human_review_rate` / `abstention_rate` / `automation_coverage` / `abstention_recall` / `wrong_auto_decision_rate`；**Phase 1 只有 PASS/REJECT 真值案**，指标 = Accuracy/Precision/Recall/FPR/FNR（重点 FPR）；**Phase 2 引入 AUTO_DECIDABLE / SHOULD_ABSTAIN** 两类语义；核心口径：**不是为降转人工牺牲安全——自动化必须与 FPR/FNR 并读** | §4.1/§4.4/§4.5 |
| P-4 | 数据集规模/来源/阶段 | 规模 300 起步；demo case 作 smoke ≤10 条不混正式分布；含少量 HUMAN 期望案（≤10%） | **推翻"300 起步"**：Phase 1 **30~50 条**（跑通 Golden Dataset → Rule → Single-call LLM → Agent → Evaluator → Metrics → Console Report 全链路）；Phase 2 **300+ 条**正式集（normal 20% / violation 20% / boundary 30% / multi-signal 20% / evasion 10%）；Smoke ≤10 条（P_88231 等）仅快速验证、不混正式统计；HUMAN 期望案不强制占比；**每个 case 必须有明确 Ground Truth，不为凑比例塞数据** | §2.1–§2.4/§8 |
| P-5 | sweep 范围与顺序 | EVIDENCE 双阈值网格全扫、CONFIDENCE 先固定 0.7 观察 | **第一轮只 sweep Evidence 阈值**（0.60/0.65/0.70/0.75/0.80/0.85/0.90 单参数），`CONFIDENCE_ABSTAIN_THRESHOLD` **固定 0.7**；**不做多参数联合 Grid Search**（变量过多无法判断效果来源）；观察 Accuracy/Precision/Recall/FPR/FNR/human_review_rate/自动化覆盖率后再决定是否联合校准 Confidence；**sweep 排在 Phase 1 框架跑通之后** | §5/§8 |

> 其余〔细化待定〕（不含 P 编号，均为**实现者可自行收敛**的落地细节；如与实现冲突以本文口径为准）：
> evidence 引用素材形态（§2.2）、数据文件分卷方式（§2.3）、Marginal Evidence Gain 加权口径（§4.2）、Evidence Sufficiency 精确公式（§4.2）、
> sweep 第一轮两个 Evidence 常量的扫描次序与其余常量取值（§5.1）、harness 是否含 DB 真落库集成路径（§3.6）。

---

## 10. 检索层与可观测性实测数字（快照，供追溯）

> 本节收集原先散落在设计文档里的**实测数字**，避免随文档收敛而丢失。数值口径与限制条件必须同框阅读；
> 绝对值随语料 / 模型 / 服务状态变化，**勿照抄为结论**。

### 10.1 RAG 三路 Recall@3（小 probe；如实并排，不预设 hybrid 最优）

**A. 真实 BGE + Qdrant 进程内（2026-09-09；`bge-small-zh-v1.5`，hybrid 权重 0.5/0.5，Policy/Case 各 8 条 probe = 5 keyword + 3 同义改写）**

| KB | bm25 | vector | hybrid |
|---|---|---|---|
| Policy (8) | 6/8 (75%) | 8/8 (100%) | 8/8 (100%) |
| Case (8) | 6/8 (75%) | 8/8 (100%) | 8/8 (100%) |

- 语义 vs 词面（同义改写 query 与目标文本零/近零共享关键词，已用 `bm25.tokenize` 程序化验证 token 交集）：
  bm25 **全漏**、vector 命中 @1~@2、hybrid 基本同 vector（`RAG_CASE_0011`「冒用双 G logo 先例」hybrid 漏，如实呈现）。
- 本 probe 下 **hybrid 未单独优于 vector**（N=8 小样本定向观测）。

**B. MockHash 8+8（2026-09-10；`scripts/run_rag_eval.py --probe`，粒度 12.5%）**

| KB | backend | bm25 / vector / hybrid |
|---|---|---|
| Policy | local | 6 / 5 / **6** |
| Policy | chroma | 6 / 5 / **7** |
| Case | local | 6 / 4 / **7** |
| Case | chroma | 6 / 4 / **6** |

- **两个 KB 上 hybrid 的相对位置相反** → 实证「不预设 RRF 最优」。
- ⚠️ 该组用 **MockHash 而非 BGE**：para 类失败**不得**解读为「语义检索不行」；`--probe` **缺省关闭**
  （诊断能力不进默认评测路径）。

### 10.2 Chroma 臂客户端等价性（A/B 隔离）

- 16 probe × 3 模式 × 2 KB + 4 组过滤组合：`ephemeral` 与 `http` 两种客户端**报告数据行 diff 为空**，
  vector parity 分差 `0.000e+00`。
- `--backend chroma`（`EphemeralClient`）35 案 A/B 报告与 `local` 臂**逐字节一致**（digest `50351888fd8bd605`）。
- ⚠️ `ephemeral`（进程内、随进程消失）与 `http`（服务端）**不是同一份存储**；生产 RAG 仍用服务端，
  该开关只影响评测脚本。

### 10.3 可观测性实测（Langfuse）

- 本地 Docker 自托管 Langfuse **4.32.0**，**6 个常驻容器**（另加一次性 `minio-init`）；health 200；project `pra-local`。
- 埋点覆盖 root / node / generation / tool / gate：单案真实路径 **≈26 observation**；评测侧 `--langfuse`
  实测 **3 case → 3 条 root trace / 69 条 observation**，全部 `sessionId=eval-demo-1`，每 case 带 `eval_case_id`。
- **口径**：scripted 桩路径下 token=0 / cost 空 / latency≈0 是真实情况，不得伪造（详见 docs/00 可观测性节）。

### 10.4 Qdrant 历史实测（`url=` 远端 server，2026-09-10）

- 阻断缺陷：`_point_id()` 取 sha256 前 16 字节 → **128 位整数**，而 Qdrant 服务端只接受 u64/UUID；
  qdrant-client 进程内模式不校验上界，**只在真 server 上以 400 暴露**（复盘见 `src/pra/rag/qdrant_index.py`）。
- 修复（截为 u64）后：真 server 落库 **24 / 67 点**，与 `local` 后端 top3 逐条一致。
- 该路线**已被 ChromaDB 取代**，此处数字仅作历史对照。
