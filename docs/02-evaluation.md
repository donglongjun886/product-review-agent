# 电商平台商品内容治理 · 复杂风险调查 Agent —— 评测方案执行细化（02-evaluation）

> 本文档把《00》评测相关章节（§11 Evaluation Dataset、§12 三方案对比、§13 Hard Case Benchmark，及 §7.6 数值口径）
> 落成**可执行口径与决策依据**，服务对象是写 `src/pra/evaluation/**` 与 `scripts/` 评测脚本的人；
> 实现规格（模块、字段、配置取值、进度状态）以代码与 README 为准。
> 引用约定：总设计写作 **《00》§x.y**（即 `docs/00-system-design.md`）；本文内部节号直接写 §x.y。
> 字段级契约以代码为准（`src/pra/agent/state.py`、`src/pra/domain/models.py`、`src/pra/agent/guardrails/schemas.py`）。
>
> **本文档不包含业务代码**：只写评测口径与决策依据 —— 数据集口径、三方案定义与输入边界、每个指标的分子/分母/取数字段、拍板记录与结论边界声明；
> 具体实现（模块划分、字段、配置取值、进度状态）一律以代码与 README 为准，本文不镜像。
> **版本状态：v1（P-1~P-5 已拍板）**：评测口径按拍板结果定稿，正文不再标 [待拍板]（逐条决策记录见 §9，供追溯）；
> 正文残留的〔细化待定〕均为**实现者可自行收敛**的落地细节，不阻塞实现；harness 代码落点见 §3.5（**以代码实际结构为准**）。
> **结果只保留最近一次运行**（§11/§12）：重跑即整节覆盖，历次数字由版本库承担。
> 本文不评审、不引用 `pra/screening` 的具体规则动作（并行修正集进行中，见 §7），评测框架与具体规则解耦，只按当次运行的 screening 行为取数并记录行为快照。

---

## 1. 定位与目标

### 1.1 本文档回答什么（职责）

《00》§11–13 只给了评测的"设计意图"，未到可执行层。本文逐项细化为实现者可照做的内容：

| 《00》章节 | 设计意图 | 本文细化 |
|---|---|---|
| §11.1/§11.2 | 评测集规模分布 + Case 结构化标签 | §2 eval_dataset 文件格式、schema、构造与版本管理 |
| §11.3 | 业务/Agent/工程三组指标 | §4 每个指标的计算口径（分子/分母/取数字段） |
| §11.4 | 三方案同一 harness、可比 metrics、确定性重放 | §3 harness 模块划分与 SchemeRunner 契约 |
| §11.5 | threshold sweep（**能力已移除**） | §5 移除决策记录 |
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
| 00-system-design | 本文件是其 §7.6/§11–13 的可执行细化；阈值校准工具已移除（§5），《00》§7.6 口径表保持代码常量 |
| Agent 契约（代码） | Agent 指标取数依据：`tool_call_history` 边际增益 4 字段、Tool Selection Accuracy 评测接口、`ReviewDecision` 终形状 —— 实现见 `src/pra/agent/state.py`、`src/pra/agent/tools_node.py`、`src/pra/domain/models.py` |
| 参数口径（代码） | 阈值/预算等**具体取值以代码常量为准**（生产侧 `src/pra/tools/image_analysis/tool.py`、`src/pra/agent/guardrails/gate.py`；预算上限见 guardrails 配置），本文不复制；口径约定：阈值不随数据集调参（校准工具已移除，见 §5） |
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
> - **真 LLM 对照是单次运行、无重复采样**：故真 LLM 数字只证明真实链路已跑通、暴露迭代方向，
>   **不代表模型固定水平**；real 臂非确定性、不可重放。

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
- `expected.evidence` 里的阈值（如 `>=0.85`）是**标注时的证据口径**，须与运行时 `EVIDENCE_MIN_SIM/STRONG` 口径一致；阈值变更须同步修订标签（《00》§11.2 注）。
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

### 2.5 数据集划分（报告隔离）

- 一次划分成 **report 集（主跑分报告）** 与 **validation 集**（供人工复核与后续校准实验），按 scene 分层抽样，比例建议 70/30。
- 报告只报 report 集数字，避免"看着 validation 改到 report 集"。
- 划分自 **Phase 2（300+）** 起执行；Phase 1（30–50 条）以全集跑通框架与指标口径即可。

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

- 走 **`build_agent_graph`**（hypothesize→plan→tools→reevaluate→decide，状态 `AgentState`；决策预算与 Guardrail 上限由图装配配置给定，取值以代码为准）；终态以确定性 overlay 后的 `ReviewDecision`（`src/pra/domain/models.py`）为判决策略真值；HUMAN_REVIEW 是三个 Decision Gate 之一（abstention 清单语义），即 **Agent 有 HUMAN_REVIEW 语义**。
- **运行模式（P-2 已拍板）**：
  - **Phase 1 默认：scripted Agent + InMemory 种子数据**（Case/Policy/Merchant 现状）——确定性、可重复、**CI 可回归**（同 case 重跑同结果，《00》§11.4）；
  - **Real LLM Evaluation 排后续阶段（Phase 3，§8）**：真实 LLM + 真实工具数据源（RAG 未接前不可用），复核 scripted 结论（尤其 Tool Selection / Evidence Sufficiency 等依赖 LLM 行为的指标）。
- **结论边界（报告必声明，P-2 口径）**："当前结果主要验证 **Agent Workflow、规则协同与 Evaluation Framework**，不代表真实 LLM 最终能力；**InMemory 种子覆盖有限，可能低估 Agent 上限**。"（另见 §7.2）

### 3.5 Harness 代码落点（src/pra/evaluation/ 模块划分）

**代码落点**：`src/pra/evaluation/`（数据集加载、harness、metrics、runner、报告、消融与回归驱动）与 `scripts/` 下的跑分入口；**实际模块划分与文件名以代码为准**，本文不镜像目录树。

契约标识符（语义固定，实现层名字以代码为准）：`EvalCase`（数据集条目）/ `RuleBaseline`·`SingleCallScheme`·`AgentScheme`（三个 SchemeRunner）/ `EvalRecord`（统一结果记录）/ `metrics.business`。三条不变：

- **EvalContext** 承载注入配置（工具数据源 / 检索模式 / abstention 门槛）——判定逻辑与评测代码**不得内联阈值常量**；
- **SchemeRunner** 契约：`async run(case: EvalCase, ctx: EvalContext) -> EvalRecord`，`name ∈ {rule, single_call_llm, agent}`；
- **metrics 层只吃 EvalRecord**，不直接吃 DB / AgentState。

> **不得引入与本节平行的新抽象命名。**

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

- **转录口径**：Agent 的 EvalRecord 由 AgentState / review_trace / review_result 字段转录（review_trace(PLAN).output_json → plan_outputs；AgentState.tool_call_history → trace；decision_json.budget_used → cost）；rule/llm 的 EvalRecord 由各自执行结果构造。
- **DB 落库非必需**：评测以内存 EvalRecord 为主（快、可并行、不污染业务表）。
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
| False Positive Rate（FPR，误杀率） | expected=PASS（正常）中自动终裁为 REJECT 的比例 | 同上 | **防误伤红线、Phase 1 重点观察**（《00》§7.2-2） |
| False Negative Rate（FNR，漏放率） | expected=REJECT（违规）中自动终裁为 PASS 的比例 | 同上 | 与 Recall 互补 |
| human_review_rate | 输出 HUMAN_REVIEW 的 case 占全部 case 的比例 | decision | 转人工占用（人工负担）；Rule 的 COMPLEX 映射与 Agent/Single-call 的 abstention 都计入 |
| automation_coverage | 1 − human_review_rate（自动终裁占比） | decision | 自动化覆盖率；与 FPR/FNR **必须并读**（§4.4 核心口径） |

> **口径注（2026-09-09 Q1 拍板 (b)，实现为准）**：二值真值案 = expected∈{PASS,REJECT} 的案
> （v2 = 274；46 条 SHOULD_ABSTAIN 真值不参与本表分母，其质量由 §4.4 abstention 指标承接）。
> Accuracy 分母 = 全部二值真值案，**预测 HUMAN_REVIEW 计为错**——这是比"auto 子集口径"更严的
> 工程口径：保守转人工与决策错误同罚，迫使"自动化 + abstention 指标并读"（§4.4 核心口径），
> 否则"大量转人工"的方案会在 Accuracy 上显得准。对照参考（auto 子集口径，即分母排除 HUMAN 预测、
> 只算 decision∈{PASS,REJECT} 的案）：v2 rule 0.921 / single 0.866 / agent 1.000 —— 两口径的
> 差别就是 abstain 惩罚量（code 口径 rule 0.423 / single_call_llm 0.518 / agent 0.964；v2 320 案，
> 二值真值 274，出处 `scripts/run_evaluation.py --data eval_data/v2/cases_v2.jsonl`），报告已并排披露（abstention 区
> abstention_rate + wrong_auto_decision_rate 量化该惩罚）。Precision/Recall/FPR/FNR 分母只含
> 对应真值类、HUMAN 预测不计入（保持 §4.4 "HUMAN 不当第三真值类"语义）。abstention 质量评估
> （该转人工是否转了、自动决策是否安全）见 §4.4（AUTO_DECIDABLE / SHOULD_ABSTAIN）。

### 4.2 Agent 指标（仅 agent scheme 有意义）

> **取数与报告口径**：① **空真值不进分母** —— `expected_tools=[]` / `expected.evidence=[]` /
> `expected.risk_type=[]` 的案被排除，案数与无法映射的期望标签数在报告行内显式给出；②覆盖口径为
> `expected_tools ⊆ actual_tools`，「调用少」另立 `redundant_tool_rate`；③证据覆盖按**可映射标签**
> 计算（不硬猜映射）；④推理正确性是**自动代理**（无人工第二标注者），只能近似。
> 各指标的**实现状态以代码为准**（未实现项不以其名义出数，如实留空）。

| 指标 | 计算口径（建议） | 取数字段 | 备注 |
|---|---|---|---|
| Tool Selection Accuracy | 每 case：`expected_tools ⊆ 实际调用集合` 且实际调用尽量少；聚合 = 命中案 / 总案 | expected.expected_tools × tool_calls_actual | 真值来自 eval_case 标签；取数接口见 `src/pra/agent/guardrails/schemas.py`（PlanOutput）+ `src/pra/agent/state.py`（tool_call_history） |
| Evidence Sufficiency | ① expected.evidence 的**证据类型**被实际 evidence 覆盖比例；② REJECT 案是否满足 REJECT Gate 前置（有可引用依据 `CITABLE_TYPES`、无关键矛盾） | expected.evidence × evidence | 精确公式〔细化待定：实现者可自行收敛〕 |
| Reasoning Correctness | 结论对但推理错：risk_type 命中率（expected.risk_type ⊆ 输出）+ risk_level 档位一致 + 抽样人工复核结构化理由 | risk_type/risk_level/trace | 自动化近似无法全覆盖 → 抽样人工复核子集（标注） |
| Marginal Evidence Gain / Investigation Efficiency | 逐 tool_call 记 `after_confidence−before_confidence` 与 `evidence_added` 非空、`decision_changed`；效率 = Σ(新增证据数 + 决策翻转权重) / 有效 Tool Calls | trace.tool_call_history 边际增益 4 字段（`src/pra/agent/state.py`） | 暴露"为调查而调查"；加权口径〔细化待定：实现者可自行收敛〕 |
| Budget Utilization | 四组占用率：llm_calls/tool_calls/tokens/latency 各 ÷ 上限（上限取值以代码配置为准），报均值与分布（**未实现**，见本节取数与报告口径） | cost + decision_json.budget_used | 证明 Budget 是 Guardrail 非目标（《00》§10.3/§11.3） |

### 4.3 工程指标（成本与效率，三方案同口径）

> **报告口径**：每方案报 llm_calls / tool_calls / tokens 的分位数（均值、P50、P95、max）。两条如实口径：
> ①scripted 路径 `tokens=0` 是真实情况（确定性桩不烧 token），报告如实显示、**不伪造估算值**；
> ②**墙钟延迟不落 `EvalRecord`**（评测逐字节可重放红线），仅 real 臂在进程内计时后传入其报告，
> 故主评测报告的延迟一栏为 `-`。各指标的**实现状态以代码为准**（未实现项如实留空，不用填充值代替）。

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
- **Evidence 三态口径（决定 HUMAN_REVIEW 怎么归因）**：`MEASURED_POSITIVE` / `MEASURED_NEGATIVE` 由证据承载
  （含阴性结论）；`NOT_MEASURED` / `UNMEASURABLE` **不是证据**，由 Gate 按「required 维度 × 证据存在性 ×
  环境能力」推导 —— **不为"缺席"造证据**，以保住"查不到 ≠ 证明无"。二者在报告里必须分开：
  **未测是可补救的（路由先回环补测）**，**环境不可测不回环**（如生产视觉桩）。
- 映射影响 FPR / human_review_rate trade-off 的形状，报告须写明所用映射；转人工与安全并读（§4.4 核心口径）。

---

## 5. Threshold Sweep（**已移除**）

> 本节标题保留为稳定锚点（外部按节号引用）；内容记录**移除决策**，不描述已不存在的实现。

- 该能力曾在评测侧实现为「按一组 Evidence 阈值重跑 agent scheme 并出曲线」。**结论（2026-09 复核）**：
  被扫的阈值只作用于评测确定性审查员读证据的视图，**与生产 tools_node quality_filter / REJECT Gate
  的实际常量不是同一处**（把选点写回生产属于"写进未测层级"，本就禁止），且评测世界相似度数据只有
  {0.72,0.73} 与 {0.90+} 两簇、中间带无 case → 曲线恒定或近平，**对任何决策都不可行动**。
- **决策**：删除阈值注入与 sweep 入口，连同其配置字段、脚本与测试。生产阈值保持代码常量不变；
  三方案决策序列不变（决策序列回归守护）。
- **若未来确需校准生产阈值**：须先把可配置点下沉到生产实际读常量的位置，并补齐中间相似度带数据 ——
  届时按新需求重新设计，不复活本节的旧形态。
- 生产侧相似度分界（`EVIDENCE_MIN_SIM` / `EVIDENCE_STRONG`）仍是 **v1 工程初始值**，未做数据集拟合。

---

## 6. Ablation Evaluation（Phase 2；引用《00》§13.4）

- 回答"给 Single-call 更多文本、再多步主动调查各能提升多少"，方案级消融跑**同一 eval_dataset**：
  **2a**（Raw Input）→ **2b**（+RAG-in-prompt 预塞政策）→ **2c**（Multi-step Agent + 主动调查）。
- **Phase 2 做，不阻塞 Phase 1 三方案可比跑分**（三方案主对比之后）。
- 实现只换装配（Single-call prompt 是否预塞政策），不动判定逻辑与评测集（差异唯一归因）。
- 判定规则：2b 与 2a 无差异 → "更多文本"增益在本数据上不可测；2c 相对 2a/2b 的差异是
  Single-call→Agent 的**整体**差异（工具 + 多步 + mock 变体一并替换），不可分解归因于"主动调查"。
- **组件级消融（逐工具裁剪）已移除**：其能力只服务一次性的组件必要性论证，需要图装配层裁剪参数与
  评测侧裁剪口径，属一次性实验脚手架；报告保留**工具证据覆盖**标注（哪些 case 的
  `expected_tools` 含该工具），覆盖为空的工具不做"不必要"判定（见 §7.2）。

---

## 7. 与 screening 修正集 / RAG 现状的关系

> 标题是**稳定锚点**（外部按节号引用），不随本文"不镜像实现现状"的原则改写；本节正文只写**本文数字基于什么世界口径**，不描述代码现状。

### 7.1 Rule baseline 依赖 screening（行为以运行时为准）

- Rule baseline 复用 screening 层的三值裁决；规则语义一变，该 baseline 数字随之变动。
- **本文档刻意不与具体规则对齐**：harness 不做规则动作断言，只按**当次运行的 screening 行为**取数；报告必须记录 **screening 行为快照**（git commit / 规则语义版本）。
- 因此 Rule baseline 的数字**只在声明的行为快照下成立**：词表或规则动作变更后必须重跑并纳入 Regression（§11/§8）；两次快照之间的差异属**结论边界**（§3.4），不得与旧数字混读。
- Fix 5（品牌词命中 REJECT vs COMPLEX→Agent 上下文终裁）本身是评测实验点：评测集需含"品牌词命中但可能合法"样本，报告给出两种规则动作下的对比行；其与评测口径 "COMPLEX→人工"（§3.2/§4.5）的差异须一并注明。

### 7.2 Agent 工具数据源的结论边界（评测世界 InMemory vs 生产真链路）

- **评测世界（本文数字的世界口径）**：Agent scheme 的 CaseSearch / PolicySearch / Merchant 以 **InMemory 种子数据**运行（scripted 模式，§3.4 已拍板为 Phase 1 默认）；评测侧另可切到真实检索（ChromaDB + LlamaIndex），两种世界的结果不得混读。
- **生产 / HTTP 入口走真实链路**：工具侧注入商品/商家 MySQL 与案例/政策的真实 RAG，LLM 侧经**组合根 `pra.wiring.build_llm_backend()`** 装配真实 litellm 网关（缺 `DEEPSEEK_API_KEY` **显式抛错、不回落 scripted 桩**）。故**本文所有数字仍是 InMemory 评测世界口径**，不得读成生产链路成绩。
- **工具集口径**：本文所有 Agent 数字均出自**评测世界的 5 个工具**（Product / ImageAnalysis / Merchant / CaseSearch / PolicySearch），比生产工具集**少一个 `OCRTool`** —— 评测集 `expected_tools` 从不含它、机审 OCR 文本近乎全空，补进去只是多一个拿不到数据的工具。两者清单各自维护、**无"同构"约束**，Tool Selection Accuracy 真值按该 5 工具集合计。
- **报告必带边界声明（P-2 口径）**："当前结果主要验证 **Agent Workflow、规则协同与 Evaluation Framework**，不代表真实 LLM 最终能力；InMemory 种子覆盖有限，**可能低估 Agent 上限**。"（与 §3.4 同文）
- 局限：种子里缺失的真实先例/完整规避史 Agent 取不到 → 多信号与对抗类案的 Evidence Sufficiency、Marginal Evidence Gain 等指标会**低估上限**。
- **环境能力声明（生产与评测不同，如实声明）**：生产的 `image_analysis` / `ocr` 仍是**冻结 Mock 桩**
  （真实商品图永远空命中）⇒ 生产装配显式把 `image_appearance` 声明为 **不可测（UNMEASURABLE）**，
  不把"桩测不出"误判成"测过且阴性"。**后果**：生产入口对带图案件在拿到真实视觉数据源之前不会自动放行 ——
  这是已声明的覆盖缺口，不是静默降级；评测世界的工具则声明了可测能力（故评测结果里该归因码为 0）。
- **"仅商家行为脏"不授权自动拒绝**（需本 listing 的外观或文本信号佐证）—— 收严来自 reviewer 语义
  （"疑似规避但图/文本无确证 → 克制转人工"），由判定侧显式实现并有单测锁定；业务若改判该族，
  必须先改这条语义再改实现。
- 真实 RAG/Merchant 接入后以 real 模式复核（Phase 3，§8）。

---

## 8. 验收标准与里程碑（分阶段主线，P-4/P-5 拍板后）

| 阶段 | 内容 | 交付 / 验收 |
|---|---|---|
| **Phase 1** | eval_data schema + **30~50 条 golden cases**（PASS/REJECT 真值，§2.1）→ Rule / Single-call / Agent 三 **SchemeRunner**（§3.2–3.4）→ 决策指标评测（Accuracy/Precision/Recall/FPR/FNR，§4.1）→ Console Report → **确定性重放断言** | **三方案在同一数据集上可比跑分**（含 metrics 金标准小样本单测）；scripted 同 case 重跑**逐字节一致**；报告含结论边界声明（§3.4/§7.2）与 screening 行为快照（§7.1）；FPR 为 Phase 1 重点指标 |
| **Phase 2** | 300+ 正式集（§2.1 五类分布）；Evidence 侧指标评测（Evidence Sufficiency / Marginal Evidence Gain，§4.2）；**Ablation**（方案级 2a/2b/2c，§3.3/§6）；**abstention 评测**（AUTO_DECIDABLE / SHOULD_ABSTAIN，§4.4）；**Regression** | abstention 五指标在正式集出数；修正集合入后重跑 Rule 基线并纳入回归（§7.1） |
| **Phase 3** | **Real LLM Evaluation**；LLM-as-a-Judge（如必要）；Regression Report | real 模式复核 scripted 结论（§3.4/§7.2）；完整报告逐行对照《00》§12.4 预期结论、明确回答 Q1–Q4（§1.2），附全部结论边界标注 |

> **本节只留验收口径**：各阶段的落地进度不写在此处（结果见 §11/§12 的最近两次运行）。

> **real 臂的接入方式（口径）**：评测侧经 `AgentScheme(llm=...)` 接 `LiteLLMBackend`，入口为
> `scripts/run_evaluation_real.py`（需 API key；real 臂评测侧放宽墙钟护栏 —— 生产护栏对真实 LLM 过紧）；
> **生产入口另经组合根 `pra.wiring.build_llm_backend()` 直接装配真实 LLM 网关**（缺凭据显式抛错，
> 不回落 scripted 桩），与评测侧的接法相互独立。

> **最终主线**：**Rule Baseline → Single-call LLM → Multi-step Agent → Ablation → Abstention → Regression**。
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
| P-5 | ~~sweep 范围与顺序~~（**该能力已移除**，见 §5） |

> 其余〔细化待定〕（不含 P 编号，均为**实现者可自行收敛**的落地细节；如与实现冲突以本文口径为准）：
> evidence 引用素材形态（§2.2）、数据文件分卷方式（§2.3）、Marginal Evidence Gain 加权口径（§4.2）、Evidence Sufficiency 精确公式（§4.2）、
> harness 是否含 DB 真落库集成路径（§3.6）。

---

> 以下为**最近一次运行的结果**；重跑实验后**整节覆盖**，不追加历史（历次数字见版本库）。
> 两块结果分属不同世界，**不可混读**：§11 为 scripted 桩 + InMemory 种子世界（可逐字节重放）；
> §12 为真实 LLM（单次运行、非确定性、不可重放，**不代表模型固定水平**）。

## 11. 封板结果表（v2 320 案；2026-09-22 重放）

> **本节为最近一次 scripted 三方案封板运行的结果**：运行锚点 = 日期 **2026-09-22** + 代码 `237f03b`
> + 工作区未提交改动（`src/` `tests/` `eval_data/`）+ 数据集 `eval_data/v2/cases_v2.jsonl`（320 案；
> 该次重生成把 `blackbrand_field` 家族 brand 改用 `terms.BLACKLISTED_BRANDS` 成员，使 R-101 黑名单
> 直判 REJECT 名副其实）。
> 重放命令：`uv run python scripts/run_evaluation.py --data eval_data/v2/cases_v2.jsonl`（§11.1/11.2/11.5）、
> `uv run python scripts/run_error_analysis.py --data eval_data/v2/cases_v2.jsonl`（§11.3/11.4）、
> `uv run python scripts/run_regression.py --data eval_data/v2/cases_v2.jsonl` → 该次 **REGRESSION PASS**
> （决策序列与 `eval_data/v2/regression_baseline.json` 一致，digest `93f3e988…`）。
> 指标**口径定义**见 §4.1/§4.2/§4.4；本节只承载该口径下的当次数值，两者冲突时以 §4 定口径、以本节记数。

### 11.1 主结果（业务分母 = 二值真值 274：PASS 134 / REJECT 140）

| Strategy | Accuracy | Precision | Recall | FPR | FNR | 漏放 | 误杀 | LLM 调用/案 | Tool 调用/案 | tokens/案 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rule baseline | 0.423 | 1.000 | 0.545 | 0.000 | 0.455 | 10 | 0 | 0.000 | 0.000 | 0.000 |
| Single-call LLM | 0.518 | 1.000 | 0.633 | 0.000 | 0.367 | 22 | 0 | 1.000 | 0.000 | 0.000 |
| **Agent** | **0.964** | **1.000** | **1.000** | **0.000** | **0.000** | **0** | **0** | 5.450 | 4.190 | 0.000 |

> 口径：Accuracy 分母 = 274，**预测 HUMAN_REVIEW 计为错**；Precision/Recall/FPR/FNR 只在自动判出
> （pred ∈ {PASS,REJECT}）子集上计算——预测 HUMAN 不入其分母（HUMAN 不当第三真值类，§4.1 口径注）。
> 漏放 = truth REJECT ∧ pred PASS；误杀 = truth PASS ∧ pred REJECT。
> tokens = 0 是 scripted 桩路径的真实值（不伪造，§4.3）。

### 11.2 转人工 / abstention（分母：全量 320；AUTO_DECIDABLE 274 / SHOULD_ABSTAIN 46）

| Strategy | human_review_rate | automation_coverage | abstention_rate | abstention_recall | wrong_auto_decision_rate |
|---|---:|---:|---:|---:|---:|
| Rule | 0.606 | 0.394 | 0.540 | 1.000 | 0.079 |
| Single-call LLM | 0.487 | 0.512 | 0.401 | 1.000 | 0.134 |
| **Agent** | **0.175** | **0.825** | **0.036** | **1.000** | **0.000** |

> 本节 human_review_rate / automation_coverage 分母为**全量 320**，与 §11.1 的 274 分母**不同**，勿混读
> （§4.4 核心口径：自动化率必须与 FPR/FNR/wrong_auto_decision_rate 并读）。

### 11.3 三分类混淆矩阵（行 = pred，列 = truth；对角线即命中，含 HUMAN 真值）

| Rule | PASS | REJECT | HUMAN |
|---|---:|---:|---:|
| PASS | 104 | 10 | 0 |
| REJECT | 0 | 12 | 0 |
| HUMAN_REVIEW | 30 | 118 | 46 |

| Single-call LLM | PASS | REJECT | HUMAN |
|---|---:|---:|---:|
| PASS | 104 | 22 | 0 |
| REJECT | 0 | 38 | 0 |
| HUMAN_REVIEW | 30 | 80 | 46 |

| **Agent** | PASS | REJECT | HUMAN |
|---|---:|---:|---:|
| PASS | **124** | 0 | 0 |
| REJECT | 0 | **140** | 0 |
| HUMAN_REVIEW | 10 | 0 | **46** |

> 对角线命中 rule 162/320、single_call_llm 188/320、agent 310/320。**该全量三分类口径不参与方案排名**
> ——它会把"多转人工"算成命中，读排名只看 §11.1（§4.1 口径注）。

### 11.4 决策迁移与 Agent 剩余失败（仅二值真值 274 案）

| baseline → Agent | 修好 | 其中从转人工修回 | 其中从判反修回 | 退步 | 同错 | 同对 |
|---|---:|---:|---:|---:|---:|---:|
| Rule → Agent | **158** | 148 | 10 | 10 | 0 | 106 |
| Single-call → Agent | **132** | 110 | 22 | 10 | 0 | 132 |

> **Agent 剩余失败合计 10/320**，全部为 `scene=boundary`、`truth=PASS`、`pred=HUMAN_REVIEW`（过度保守）：
> `EC_V2_0275` `EC_V2_0279` `EC_V2_0283` `EC_V2_0287` `EC_V2_0289` `EC_V2_0290` `EC_V2_0291`
> `EC_V2_0292` `EC_V2_0293` `EC_V2_0294`。
> 其中 **0 漏放（truth REJECT 未自动放行）、0 误杀、0 该转人工却自动终裁**；10 例"退步"即这 10 例。

### 11.5 Agent 级指标与成本分布（仅 agent 臂；空真值不进分母）

| 指标 | 值 | 计数 |
|---|---|---|
| Tool Selection Accuracy（覆盖口径 `expected_tools ⊆ actual_tools`） | 0.870 | 188/216 案 |
| redundant_tool_rate（有期望外调用的案占比） | 0.176 | 期望外调用案 38 |
| Evidence Type Coverage（micro） | 1.000 | 345/345 期望类型 |
| REJECT 依据前置（可引用依据近似） | 1.000 | 140/140 预测 REJECT 案 |
| risk_type_coverage | 0.914 | 128/140 案 |
| risk_level_agreement | 0.775 | 248/320 案 |
| evidence_gain_rate（带来新增证据的调用占比） | 1.000 | 1228/1228 调用 |
| decision_changed_rate（触发 Gate 判定翻转） | 0.332 | 408/1228 调用 |

> 成本分布（均值/P50/P95）：LLM 调用 5.450/6.000/10.000；Tool 调用 4.190/5.000/6.000；tokens 0.000/0.000/0.000
> （脚本路径恒 0 为真实值）。延迟仅 real 臂进程内墙钟，scripted 无此项。
> **本次运行的计数口径**：`blackbrand_field` 12 案由 R-101 直判，scripted Agent 对这 12 案不输出
> `EVASION_PATTERN` ⇒ risk_type_coverage 0.914（128/140）、触发翻转的调用 408。两者均为**只读审计字段**，
> 三方案决策与成本不受影响。
> 未实现的 Agent 级口径（Budget Utilization、按 scene 分层分布、单案成本折算）见 §4.2 取数与报告口径。
> 未映射期望标签 98 个实例、仅含未映射标签被剔除的案 10（缺口显式化，不硬猜映射）。

### 11.6 封板结论与对外表述边界

- **Agent 的增益集中在复杂案型**：`violation` 上 Rule 只自动裁决 12/64 案（R-101 黑名单直判，
  acc 0.188、hrr 0.812）→ Single-call acc 0.594 → Agent **1.000**；`multi-signal` / `evasion` 上
  Rule 与 Single-call acc 均 0.000 → Agent **1.000**；`boundary` 0.571 → **0.857**；`normal` 三方案
  均已 1.000（Agent 的收益不来自简单案）。
- **代价与收益同框**：Agent 5.450 LLM + 4.190 Tool 调用/案，换得转人工率 0.606 → **0.175**、
  自动化覆盖率 0.394 → **0.825**、`wrong_auto_decision_rate` 0.079/0.134 → **0.000**。
- **结论边界（必须同框引用）**：本表为 **scripted 桩 + InMemory 种子世界**（衡量实现一致性与工作流，
  §3.4/§7.2）；真值由**单一标注者**按与审查员同源规则构造，SHOULD_ABSTAIN 无对抗负例
  → 不得读成真实 LLM 能力，也不得外推为线上成绩。真实 LLM 世界见 §12（**单次、不可重放、不可外推**）。
- **封板判定**：本表 + §11.4 归因 + 回归守护（`tests/test_regression_v2.py`）构成 P0 全集；
  Ablation / real 冒烟为 P1（不做不影响 eval 完整性）；P2 不做项见 §4.2、§7.2
  与 README「当前实现边界」。**重跑这三条命令（数据集 / 判定逻辑 / 指标口径 / RAG 组件任一改变后）
  即整节覆盖本表，不追加历史。**

---

## 12. real-LLM 全量结果（v2 320 案；单次、不可重放）

> **本节为最近一次 real 臂 320 全量运行的结果**：运行锚点 = 日期 **2026-09-11** + 代码 `ba97fda`
> + 数据集 `eval_data/v2/cases_v2.jsonl`（320 案）+ world=eval（与 §11 同一 InMemory 种子世界，
> LLM 是唯一变量）。与 §11 的 scripted 封板表**分属不同世界，不可混读**：§11 是确定性桩世界
> （可逐字节重放，衡量实现一致性）；本节是真实 LLM 世界（**单次运行、非确定性、不可重放，
> 不代表模型固定水平**）。重跑即整节覆盖。
>
> ⚠ **数据集范围注记（2026-09-22）**：本节 real 臂与 §12.5 的 scripted 配对列取自 **2026-09-22 数据集
> 重生成之前**的 `cases_v2.jsonl`（该次重生成仅改 `blackbrand_field` 12 案的 brand → R-101 直判 REJECT）。
> real 未重跑（不可重放、单次 5.17M token）—— 与 §11（新数据集）比较只可用于**方向性**结论，
> 逐项差异见 §12.5 注。
>
> - 命令：`uv run python scripts/run_evaluation_real.py --data eval_data/v2 --concurrency 8 --out .cache/real_v2_gate.json`
> - 模型：配置串 `deepseek/deepseek-chat` → 网关实服 **DeepSeek-V4.1-Flash**
> - 预算：**生产护栏未动**（llm / tool / token 上限取代码配置值）；仅按既有评测口径放宽墙钟
> - 320/320 全部产出裁决，墙钟 ≈15 分钟，token **5,173,433**（估算 ≈$2.56 峰价 / $1.29 谷价，
>   单价口径见 §12.5 —— **金额是估算，token 总量是硬数**）
> - 产物（gitignored）：`.cache/real_v2_gate.json`（real 全量记录 + 两臂指标）、`.cache/real_v2_gate_console.txt`
>   （Console 报告）、`.cache/real_v2_gate_diff.tsv`（320 行逐案 truth/scripted/real/归因）、
>   `.cache/real_v2_gate_analysis.txt`（事后复算：abstention 五指标 / 安全面 / 分场景 / 归因汇总）
> - **交叉验证**：同一次运行的 scripted 配对臂与 §11 **Agent 主结果逐项一致**（0.964 / 1.000 / 1.000 /
>   hrr 0.036 / 140-0-124-0；仅审计字段有两处差异，见 §12.5 注）⇒ 两臂同 harness、同世界，LLM 是唯一变量。

### 12.1 主结果（业务分母 = 二值真值 274，口径同 §4.1）

| Strategy | Accuracy | Precision | Recall | FPR | FNR | 漏放 | 误杀 | TP/FP/TN/FN |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| scripted（= §11） | 0.964 | 1.000 | 1.000 | 0.000 | 0.000 | 0 | 0 | 140/0/124/0 |
| **real-LLM（单次）** | **0.850** | **1.000** | 1.000 | **0.000** | 0.000 | **0** | **0** | **118/0/115/0** |

### 12.2 全量 320 口径（口径同 §4.4）

| Strategy | human_review_rate | automation_coverage | abstention_rate | abstention_recall | wrong_auto_decision_rate | pred 分布 |
|---|---:|---:|---:|---:|---:|---|
| scripted（= §11） | 0.175 (56) | 0.825 (264) | 0.036 (10/274) | 46/46 = 1.000 | 0.000 (0/264) | PASS 124 / REJECT 140 / HUMAN 56 |
| **real-LLM（单次）** | **0.272 (87)** | **0.728 (233)** | 0.150 (41/274) | 46/46 = 1.000 | **0.000 (0/233)** | PASS **115** / REJECT 118 / HUMAN 87 |

### 12.3 三分类混淆矩阵（行 = pred，列 = truth；含 HUMAN 真值）

| real | truth PASS | truth REJECT | truth HUMAN |
|---|---:|---:|---:|
| pred PASS | **115** | 0 | 0 |
| pred REJECT | 0 | **118** | 0 |
| pred HUMAN_REVIEW | 19 | 22 | **46** |

> 两臂的 HUMAN 行不同：scripted 会把 10 例 PASS 真值刻意过度转人工（既有观测点），real 在该 10 例上判对；
> real 的 41 例 HUMAN 全部落在安全侧（无 PASS/REJECT 误终裁）。

### 12.4 按 scene（二值真值案）

| scene | scripted acc | **real acc** | real TP | real FP | real hrr |
|---|---:|---:|---:|---:|---:|
| normal | 1.000 | **0.953** | 0 | 0 | 0.047 |
| violation | 1.000 | **0.891** | 57 | 0 | 0.109 |
| boundary | 0.857 | **0.771** | 0 | 0 | 0.438 |
| multi-signal | 1.000 | **0.786** | 44 | 0 | 0.312 |
| evasion | 1.000 | **0.850** | 17 | 0 | 0.469 |

> `normal` 上 real 的 hrr 为 0.047（64 案里 3 案过度保守）；全场景 **FP = 0**。

### 12.5 调用 / token / 工具 / 证据链（两臂同口径）

| 统计项 | scripted（配对臂） | **real** |
|---|---|---|
| LLM 调用 均值 / P50 / P95 / max | 5.45 / 6 / 10 / 10 | **5.93 / 6 / 10 / 10** |
| tool 调用 均值 / P50 / P95 / max | 4.19 / 5 / 6 / 6 | **3.83 / 4 / 6 / 8** |
| tokens 均值 / P50 / P95 / max | 0（桩不烧 token） | **16,167 / 16,436 / 30,360 / 31,787** |
| tokens 合计 | 0 | **5,173,433** |
| tool_selection_accuracy（覆盖口径） | 0.870 (188/216) | **0.296 (64/216)** |
| redundant_tool_rate | 0.176 | 0.245 |
| evidence_type_coverage（micro） | 1.000 (345/345) | **1.000 (345/345)** |
| reject_evidence_gate_pass_rate | 1.000 (140/140) | **1.000 (118/118)** |
| risk_type_coverage | 1.000 (140/140) | 0.971 (136/140) |
| risk_level_agreement | 0.775 (248/320) | **0.816 (261/320)** |
| evidence_gain_rate | 1.000 | 0.999 |
| decision_changed_rate | 0.352 | 0.377 |

> **成本口径**：`EvalRecord.cost` 只有 token **总量**（`usage.total_tokens`，含缓存命中），
> **input/output 拆分不落 record**（只进 Langfuse），**金额完全未采集**。金额估算口径：用一次性诊断
> （运行时包装 `litellm.acompletion`，1 案 10 次调用）实测拆分为 **input 78.2% / output 21.8%**，
> 据此按 Flash 官方价（输入按最贵档 cache-miss）估算。**金额是估算，token 总量是硬数。**
> **scripted 配对臂（本表第 2 列）与 §11 的差异**：表内 scripted 列取自**同一次 real 运行的配对臂**
> （2026-09-22 数据集重生成前的版本），与刷新后的 §11.5 有两行审计字段差异
> （risk_type_coverage 0.914 对 1.000、decision_changed_rate 0.332 对 0.352，源于 `blackbrand_field`
> 12 案的 R-101 直判），**主结果（决策）一致**。
> **real 侧工具选择准确率偏低但证据链指标全满**：真实 LLM 会多调"期望外"的工具去补判断，
> 属行为差异而非判定缺陷（覆盖口径见 §4.2）。

### 12.6 overrides 归因与异常审计

| 项 | scripted | **real** |
|---|---:|---:|
| 带 overrides 案数 | 28/320 | **66/320** |
| R3_BUDGET_EXHAUSTED（先撞维度） | 28（全 `LLM_CALLS`） | **50（全 `LLM_CALLS`）** |
| R3_MEASUREMENT_MISSING | 28 | 28 |
| R2_REJECT_GATE_FAIL | 0 | 8 |
| R3_POSITIVE_INSUFFICIENT | 0 | 8 |
| R4_PASS_GATE_FAIL | 0 | 8 |
| R3_EVIDENCE_CONFLICT | 4 | 4 |
| R3_DIMENSION_UNMEASURABLE / R3_KEY_TOOL_FAILED | 0 / 0 | **0 / 0** |
| R5_DEGRADED_OR_FAILED_STEP | 0 | **0** |

> **异常审计（可证伪）**：R5 降级 **0 案**、`token=0` 案 **0**（无静默回退）、320/320 全部产出裁决、
> 日志无 traceback / 超时 / 连接错误。`R3_DIMENSION_UNMEASURABLE = 0` 是**预期**：评测世界 5 个工具都
> 声明了可测能力，该码只在生产视觉桩（见 §7.2 环境能力声明）下才会出现。
> **口径限制（不谎称 0 重试）**：`llm_calls` / `tokens` 按 attempts 累计（一次 schema 重试占 2 格），
> 单次重试次数无法从 record 分离 —— 能证明的只有"无最终降级、无静默回退、无异常中止"。

### 12.7 结论与不可外推边界

- **两个结构性目标达成**：① 干净案 **PASS 可达**（GT=PASS→PASS = **115/134**，hrr **0.272**）；
  ② 弱相似不授权自动拒绝，**误杀 0**。
- **安全面全清**：漏放 0、误杀 0、错误自动终裁 0.000、SHOULD_ABSTAIN 被自动终裁 0/46、R5 0、`token=0` 0。
- **剩余失败 41 案全部在安全侧**（19 自转人工 + 22 预算截胡），没有一项是危险终裁；boundary 是最保守族
  （hrr 0.438）。
- **成本**：real 每案 LLM 调用 5.93、tool 调用 3.83、tokens 16,167，总 token **5.17M**。
- **不可外推**：单次运行、单模型、非确定性、InMemory 工具世界、生产视觉不可测；real 数字**只代表这一次运行**，
  不得当模型固定水平，也不得与 §11 的 scripted 数字混算（§11 衡量实现一致性）。**不重复采样。**

### 12.8 已知边界与后续演进（本轮明确不做）

| 项 | 状态 |
|---|---|
| 生产预算档位调整（抬高 `LLM_CALLS` 上限） | 未做（50 案仍撞墙；`--llm-budget` 对照实验可选，不改规则） |
| `GT=REJECT → HUMAN_REVIEW` 的回收（回环/取证） | 未做（本轮 22 例同族） |
| `listing_registry` **阳性路径**（声明与在库事实确定性比对） | 未实现（现只判"事实取到了没有"） |
| 真实视觉 / OCR 数据源 | 冻结（生产 `image_appearance` 声明为 UNMEASURABLE） |
| Ablation 在评测世界跑 | 未做（agent 逐案决策零变化 ⇒ 预期不变，未实测） |
| MQ / 异步 / Observability 扩展 / MySQL Checkpointer 等生产化增强 | 未做（§「不做什么」清单） |
| 其他功能扩展 | 不做 |

