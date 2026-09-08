# 电商平台商品内容治理 · 复杂风险调查 Agent —— 评测方案执行细化（02-evaluation）

> 本文档把《00》评测相关章节（§11 Evaluation Dataset、§12 三方案对比、§13 Hard Case Benchmark，及 §7.6 数值口径）
> 落成**实现层唯一依据**，服务对象是写 `src/pra/evaluation/**` 与 `scripts/` 评测脚本的人。
> 引用约定：总设计写作 **《00》§x.y**，契约细化写作 **01 §x.y**，拍板记录写作 **03 T-x / 03 §x**。
>
> **本文档不包含业务代码**：只写 eval_case schema、模块划分、伪代码与指标口径（与 01 同一原则），它们是"契约"。
> **版本状态：v0.1 骨架**：多处标 **〔细化待定〕** 或列入 §9 待拍板清单——主 agent 拍板前不得当作已定稿；
> 本文不评审、不引用 `pra/screening` 的具体规则动作（并行修正集进行中，见 §7），评测框架与具体规则解耦。

---

## 1. 定位与目标

### 1.1 本文档回答什么（职责）

《00》§11–13 只给了评测的"设计意图"，未到可执行层。本文逐项细化为实现者可照做的内容：

| 《00》章节 | 设计意图 | 本文细化 |
|---|---|---|
| §11.1/11.2 | 评测集规模分布 + Case 结构化标签 | §2 eval_dataset 文件格式、schema、构造与版本管理 |
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
| 00-system-design | 本文件是其 §7.6/§11–13 的可执行细化；sweep 校准结果回写《00》§7.6 口径表 |
| 01-agent-loop | Agent 指标取数依据：tool_call_history 边际增益 4 字段（01 §2.4/§5.8）、Tool Selection Accuracy 评测接口（01 §4.2）、ReviewDecision 终形状（01 §7.6） |
| 03-decisions | 参数权威值来源：T-7 预算 10/15、T-11 `EVIDENCE_MIN_SIM=0.70/`EVIDENCE_STRONG=0.85`、`CONFIDENCE_ABSTAIN_THRESHOLD=0.7`（03 §5 常量表）；校准结果回写 03 §5 |
| 04-graph-design | Agent scheme 的图装配与 Ablation 变体构造依据（工具注册层裁剪，见 §6） |

---

## 2. 评测集 eval_dataset

### 2.1 规模与分布（引用《00》§11.1）

v1 目标 **300–500 Case**，五类分布如下（标注时按此比例分层抽样，避免案型偏斜）：

| scene（场景标签） | 占比 | 说明 |
|---|---|---|
| `normal` 明确正常 | 20% | 规则与 Agent 都应 PASS |
| `violation` 明确违规 | 20% | 规则与 Agent 都应 REJECT |
| `boundary` 边界案件 | 30% | 规则拿不准、单信号弱 |
| `multi-signal` 多信号组合 | 20% | 需要多源交叉验证 |
| `evasion` 对抗/规避 | 10% | 刻意规避审核（核心 Hard Case） |

〔待拍板 P-4〕规模终值（下限建议 300，质量优先于数量）；分布容差（建议每类 ±5 个百分点内）。

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
    "decision": "HUMAN_REVIEW",             // PASS | REJECT | HUMAN_REVIEW（三分类真值；是否含 HUMAN 案见 P-4）
    "risk_level": "HIGH",                   // LOW|MEDIUM|HIGH|NONE（《00》§7.5，独立于 decision）
    "risk_type": ["POTENTIAL_IP_RISK"],     // 受控词表（00 §7.3），供 risk_type 命中类指标
    "evidence": ["image_similarity>=0.85", "merchant_history>=5_removals"],  // 证据类型 + 标注时阈值口径（00 §11.2 注）
    "expected_tools": ["ImageAnalysisTool", "MerchantTool"],  // Agent 应调用的工具集合（01 §4.2 Tool Selection Accuracy 真值）
    "applicable_policy": ["POLICY_3.2"]     // 政策条款 / 先例（REJECT 案必填，对齐 REJECT Gate 的"可引用依据"）
  },
  "annotation": { "labelers": ["A", "B"], "agreed": true, "notes": "" }  // 人工标注与交叉校验记录（2.3-4）
}
```

要点：
- 标签**不只是 PASS/REJECT**：`expected.evidence / risk_type / expected_tools` 使评测能区分"结论对但理由错"（《00》§11.2）。
- `expected.evidence` 里的阈值（如 `>=0.85`）是**标注时的证据口径**，须与运行时 `EVIDENCE_MIN_SIM/STRONG` 口径一致；阈值经 §5 sweep 校准变更后**同步修订标签**（《00》§11.2 注）。
- `input.images` 的素材引用（URL vs 本地 asset）与 OCR 文本是否内联，〔细化待定〕：影响 ImageAnalysis/OCR 工具的评测可复现性，建议 eval 专用 asset 目录 + URL 占位，避免外网依赖。

### 2.3 构造流程（对应《00》§13.2，落到"谁产、什么格式、存哪、版本怎么管"）

| 步骤 | 产出 | 责任人 / 工具 | 输出格式与存放 |
|---|---|---|---|
| 1 人工构造对抗样本 | 合成 case（核心场景：品牌模仿/规避，无 Logo 无品牌词但外观高度相似 + 规避史） | 标注者 + 脚本辅助生成 JSON 骨架 | 进 `scripts/` 生成器，产物写 eval JSONL |
| 2 真实案例改写脱敏 | 历史人工裁决案件 → 脱敏 benchmark 条目 | 标注者改写（PII/商家联系方式不落库，见《00》§8.2-3） | 手工维护 JSONL，git 入库前过脱敏校验 |
| 3 程序化变异 | 对模板做字段变异（改相似度/商家历史/OCR 冲突）制造边界 | `scripts/eval_dataset_gen.py`（确定性随机种子） | 脚本入库，产物可重复生成；变异参数记入 `lineage` |
| 4 人工标注 + 交叉校验 | 每条 `expected_decision + evidence + policy + expected_tools` | ≥2 人标注，冲突 case 修正至一致 | `annotation.agreed`；一致性统计写 manifest |

版本管理建议：
- 目录 `eval_data/v<N>/`：`eval_cases_v<N>.jsonl`（全量）+ `manifest.json`（schema_version、五类分布统计、Hard 占比、标注时阈值口径快照 `EVIDENCE_MIN_SIM/STRONG`、生成命令与 git commit、标注一致性）→ 评测报告必须记录所用数据集版本。
- 合成/变异由**脚本生成**（可复现、diff 友好）；真实脱敏改写以 **JSONL 入库**（可评审）；是否两者分文件（`synthetic/` 与 `handcrafted/`）〔细化待定〕。
- 〔待拍板 P-4〕是否复用已有 demo case（P_88231 复古运动鞋等）作为首批真实改写种子与冒烟最小集（建议：复用为 smoke 集 ≤10 条，不混入正式分布统计）。

### 2.4 Hard Case 占比与挑选（《00》§13.1）

- 三选一即算 Hard（标注 `hard_case=true` + `hard_reason`）：① Rule 无法判断/易误判（无硬规则命中但真实风险）；② Single-call LLM 证据不足/不稳定（决策依赖输入里**不存在**的信息）；③ Agent 多步调查可获得额外证据（工具调用能显著改变结论）。
- **验证方法**：①③需在 Rule / Agent 冒烟跑分后回填确认（先粗标、跑分后复核 hard_reason 是否成立），②由标注者按证据缺口判断。
- 占比建议 ≥15%（`evasion` 10% + `boundary` 中符合三选一者）；〔待拍板 P-4〕终值。

### 2.5 数据集划分（sweep 校准与报告隔离）

- 一次划分成 **report 集（主跑分报告）** 与 **validation 集（§5 sweep 校准 operating point）**，按 scene 分层抽样，比例建议 70/30。
- 报告必须声明 operating point 取自 validation；report 集只在定稿后跑一次全量，避免"调参调到报告集"。

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

### 3.2 Rule baseline（[待拍板 P-1]：两种候选语义）

- **选项 (a)：复用 `pra.screening` 三分流**。PASS/REJECT 为确定性规则**直接终裁**、COMPLEX 进 Agent（已拍板语义）；评测里把 **COMPLEX 等价映射为 HUMAN_REVIEW**（无 Agent 时复杂案只能人工，见 §4.5）。
  - 优点：评测对象 = 线上真实判定器，结论可直接外推；三分类输出空间天然对齐（§4.5）。
  - 代价：Rule 行为随 screening 修正集变动（§7）；"COMPLEX→人工"是**评测语义**，与线上"COMPLEX→Agent"不同，报告须写明口径。
- **选项 (b)：独立二分 rule**（《00》§12.1 原述：命中→REJECT、否则 PASS、无 HUMAN_REVIEW 语义；或仅低置信即转人工），在 evaluation 内新写或薄封装，不碰线上 screening。
  - 优点：与《00》§12.1 文字一致、实现最小、完全隔离修正集影响。
  - 代价：输出空间少一档，指标可比性需映射（§4.5）；结论外推线上能力弱。
- **[建议默认]** 主 baseline 用 **(a)**；**(b)** 作"无转人工选项的纯规则上限"敏感性补充（可选运行、不进主对比表）。P-1 影响 §3.5 的 rule_scheme 装配与 §4.5 映射，是**实现前必拍项**。

### 3.3 Single-call LLM（《00》§12.2 + 变体）

- 一次调用：基础输入全量 + 少量背景 prompt → 结构化决策 JSON（decision/risk_level/risk_type/decision_confidence/evidence 摘要/policy 候选）。
- 无工具、无多步；输出 schema 复用 ReviewDecision 子集（01 §7.6 形状的子集，`budget_used/overrides` 恒空/不适用）。
- 变体（可选，二期消融）：2a 仅商品原始数据；2b 政策+案例预塞 prompt（RAG-in-prompt），隔离"缺证据 vs 缺多步推理"（《00》§12.2）。

### 3.4 Agent（真实执行语义 + 两种运行模式）

- 走 **`build_agent_graph`**（hypothesize→plan→tools→reevaluate→decide，状态 `AgentState`，预算 10/15/40000/30000 Guardrail）；终态以确定性 overlay 后的 `ReviewDecision`（01 §7.6）为判决策略真值。
- **[待拍板 P-2] 两种运行模式**：
  - **scripted 模式（[建议默认]，CI 可跑）**：LLM 节点注入 scripted 桩（按 case 预录/规则化的结构化输出），工具数据源为 **InMemory 种子数据**（Case/Policy/Merchant 现状）→ **确定性重放**（同 case 重跑同结果，《00》§11.4）。
  - **real 模式（二期）**：真实 LLM + 真实工具数据源（RAG 未接前不可用），复核 scripted 结论（尤其 Tool Selection / Evidence Sufficiency 等依赖 LLM 行为的指标）。
- **结论边界（必标注）**：scripted + InMemory 的结论，报告须标注"**工具为 InMemory 种子数据、LLM 为桩**"——种子里查不到的先例/规避史会**低估** Agent 上限（见 §7.2）。

### 3.5 Harness 代码落点（src/pra/evaluation/ 建议模块划分）

```
src/pra/evaluation/
├── dataset/
│   ├── schema.py          # EvalCase Pydantic 模型 + 校验（§2.2 JSON 对齐；标签完整性/受控词表）
│   └── loader.py          # JSONL 读取、manifest 解析、report/validation 划分、scene 分层统计
├── harness/
│   ├── base.py            # EvalContext（阈值常量/预算/LLM 模式/工具数据源）+ SchemeRunner 抽象
│   ├── rule_scheme.py     # P-1 (a) 薄封装 pra.screening 入口 | (b) 独立二分 rule（待拍板定）
│   ├── single_call_scheme.py  # prompt 构造 + 单次调用 + JSON 校验降级（01 §3.0 壳同款：重试 1 次，失败→HUMAN_REVIEW）
│   ├── agent_scheme.py    # build_agent_graph + scripted/real LLM + 工具数据源注入 + 结果转录
│   └── record.py          # 统一 EvalRecord（见 3.6），各 scheme 输出归一化落点
├── metrics/
│   ├── business.py        # §4.1：Recall/Precision/FPR/Decision Accuracy/HRR/Automation
│   ├── agent.py           # §4.2：Tool Selection Accuracy/Evidence Sufficiency/…/Budget Utilization
│   └── engineering.py     # §4.3：llm_calls/tool_calls/tokens/latency 均值与分位数
├── report.py              # 汇总 → markdown/json 报告；含按 scene 分层表 + 《00》§12.4 预期结论对照
└── sweep.py               # §5：threshold sweep 驱动（只改配置，见 §5）

scripts/
├── eval_dataset_gen.py    # §2.3 程序化变异/合成入口（确定性种子）
└── run_evaluation.py      # 跑分入口：load → N schemes → metrics → report（子命令 --scheme/--split）
```

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
  "decision": "REJECT",            // PASS|REJECT|HUMAN_REVIEW（三分类归一，见 4.5）
  "risk_level": "HIGH", "risk_type": ["POTENTIAL_IP_RISK"],
  "decision_confidence": 0.87,
  "evidence": [ { "type": "IMAGE_SIMILARITY", "value": "similarity=0.91…", "extra": {"similarity": 0.91} } ],
  "policy": ["POLICY_3.2"],
  "tool_calls_actual": ["ImageAnalysisTool", "MerchantTool"],   // agent 用；rule/llm 为 []
  "trace": { "plan_outputs": [ … ], "tool_call_history": [ … ] },  // 01 §2.4/§5.8 字段原样转录（scripted 模式）
  "cost": { "llm_calls": 8, "tool_calls": 5, "tokens": 18000, "latency_ms": 9200 }
}
```

- Agent 的 EvalRecord 由 **AgentState/review_trace/review_result 字段转录**（review_trace(PLAN).output_json → plan_outputs；AgentState.tool_call_history → trace；decision_json.budget_used → cost）；rule/llm 的 EvalRecord 由各自执行结果构造。
- **DB 落库非必需**：评测以内存 EvalRecord 为主（快、可并行、不污染业务表）；DB 版（走 run_and_persist/SCREENING_DIRECT 真落库）作集成测试可选路径〔细化待定〕。
- **确定性重放**：scripted 模式要求同 case 重跑产出**逐字节一致**的 EvalRecord（mock/录制工具结果，随机种子固定），作为 M2 验收断言（§8）。

---

## 4. 指标口径（每个指标给分子/分母/取数字段）

> 取数一律以 **EvalRecord** 为准（3.6）；与 DB 字段（review_trace / review_result.decision_json / decision_json.budget_used）
> 的映射仅用于集成抽查。字段口径见 01 §2.4（tool_call_history）、01 §7.6（ReviewDecision）、《00》§10.3。

### 4.1 业务指标

| 指标 | 计算口径（建议） | 取数字段 | 备注 |
|---|---|---|---|
| Risk Recall（违规召回） | 在"可自动判"子集上：预测 REJECT 且 expected=REJECT / expected=REJECT | decision × expected.decision | HUMAN_REVIEW 期望/输出是否入混淆，见 §4.4 口径与 P-3 |
| Precision（精确率） | 预测 REJECT 中 expected=REJECT 的比例 | 同上 | 误拒直接伤害商家 |
| False Positive Rate | expected=PASS（正常）中预测 REJECT 的比例 | 同上 | **防误伤红线**（《00》§7.2-2）；sweep 主观察曲线之一 |
| Decision Accuracy | 三分类正确率：expected∈{PASS,REJECT} 上逐案精确匹配；expected=HUMAN 案单独报 abstention 命中 | decision | 期望含 HUMAN 案时的处理见 P-4 |
| Human Review Rate | 输出 HUMAN_REVIEW 的 case 占比（细分见 §4.4） | decision | 三方案可比前提见 §4.5 |
| Automation Rate | 1 − Human Review Rate | decision | |

### 4.2 Agent 指标（仅 agent scheme 有意义）

| 指标 | 计算口径（建议） | 取数字段 | 备注 |
|---|---|---|---|
| Tool Selection Accuracy | 每 case：`expected_tools ⊆ 实际调用集合` 且实际调用尽量少；聚合 = 命中案 / 总案 | expected.expected_tools × tool_calls_actual | 真值来自 eval_case 标签；取数接口见 01 §4.2（PlanOutput + tool_call_history） |
| Evidence Sufficiency | ① expected.evidence 的**证据类型**被实际 evidence 覆盖比例；② REJECT 案是否满足 REJECT Gate 前置（有可引用依据 `CITABLE_TYPES`、无关键矛盾） | expected.evidence × evidence | 精确公式〔细化待定〕 |
| Reasoning Correctness | 结论对但推理错：risk_type 命中率（expected.risk_type ⊆ 输出）+ risk_level 档位一致 + 抽样人工复核结构化理由 | risk_type/risk_level/trace | 自动化近似无法全覆盖 → 抽样人工复核子集（标注） |
| Marginal Evidence Gain / Investigation Efficiency | 逐 tool_call 记 `after_confidence−before_confidence` 与 `evidence_added` 非空、`decision_changed`；效率 = Σ(新增证据数 + 决策翻转权重) / 有效 Tool Calls | trace.tool_call_history 边际增益 4 字段（01 §2.4/§5.8） | 暴露"为调查而调查"；加权口径〔细化待定〕 |
| Budget Utilization | 四组占用率：llm_calls/tool_calls/tokens/latency 各 ÷ 上限（10/15/40000/30000），报均值与分布 | cost + decision_json.budget_used | 证明 Budget 是 Guardrail 非目标（《00》§10.3/§11.3） |

### 4.3 工程指标（成本与效率，三方案同口径）

- LLM Calls（平均/P95）、Tool Calls（平均/P95）、Token Usage、P50/P95 Latency、单 Case 成本。
- 一律报**分位数与分布**（不只均值），用于回答"Agent 贵在哪、是否值得"（《00》§11.3）；按 scene 分层报。

### 4.4 Human Review Rate 口径注意（[待拍板 P-3]）

- HUMAN_REVIEW 语义在三方案里**不一致**（§4.5），且评测案里"应转人工的真值"（expected=HUMAN_REVIEW）如何构造是开放问题（P-4）。
- 建议把 HRR 细分为三率，避免一个数字混口径：
  - `HRR_total` = 输出 HUMAN_REVIEW / 全部；
  - `HRR_on_auto_decidable` = expected∈{PASS,REJECT} 却输出 HUMAN_REVIEW / 该子集（"本该自动判却转人工"，越高说明方案越保守）；
  - `abstention_recall` = expected=HUMAN_REVIEW 且输出 HUMAN_REVIEW / 该子集（"该转人工的克制地转了"）。
- 口径细节（二分类混淆矩阵是否剔除 HUMAN 期望案、PASS 案如何构造"干净但低置信"样本）〔细化待定〕，但**必须先于跑分定稿**，否则三方案 HRR 不可比。

### 4.5 三分类输出空间对齐（Rule 无 HUMAN_REVIEW 的问题）

- 方案输出空间：Agent/Single-call LLM（建议）三分类；Rule (a) COMPLEX→HUMAN_REVIEW 映射；Rule (b) 二分无 HUMAN。
- **[建议]** 三方案统一按 **PASS/REJECT/HUMAN_REVIEW 三分类**对齐后进 metrics：
  - Rule (a)：COMPLEX 记 HUMAN_REVIEW（评测语义 = "不可自动判"；《00》§12.1 注"或仅低置信即转人工"即此意）；
  - Rule (b)：二分输出，HUMAN_REVIEW 恒为 0 → HRR 天然 0，其"转人工代价"体现在 FPR/漏放上，需在报告里注明比较基准差异；
  - Single-call LLM：输出 decision_confidence，`< CONFIDENCE_ABSTAIN_THRESHOLD` 的 REJECT 候选按确定性后处理记 HUMAN_REVIEW（与 Agent 的 REJECT Gate 口径一致）。
- 对齐映射是 **[待拍板 P-1/P-3]** 的组成部分：不同映射会改变 FPR/HRR trade-off 曲线的形状，报告须写明所用映射。

---

## 5. Threshold Sweep

### 5.1 扫哪些常量（只动配置，不动判定逻辑；《00》§11.5/03 T-11）

| 常量 | 扫描网格（《00》§11.5） | 影响路径 |
|---|---|---|
| `EVIDENCE_MIN_SIM` / `EVIDENCE_STRONG` | 0.60/0.65/0.70/0.75/0.80/0.85/0.90 | 产 IMAGE_SIMILARITY 证据的三档分界 → 影响含图片证据案（agent tools_node quality_filter / Rule 图片相关路径） |
| `CONFIDENCE_ABSTAIN_THRESHOLD` | 二次扫或先固定 0.7（[建议]先固定，随主扫观察，P-5） | REJECT Gate 安全门槛 → 主要影响 Agent 与 Single-call LLM 的 abstention |

- 网格组合量级与是否全组合（EVIDENCE_MIN ≤ EVIDENCE_STRONG 约束下采样）〔细化待定〕，v1 建议先全组合小跑 validation 子集定粗区间再加密。
- **实现约束**：阈值全部经配置层注入（EvalContext），评测代码与判定逻辑**不得内联阈值常量**；sweep 只是换 EvalContext 重跑 agent/rule scheme，Rule 与 Agent 共用同一份配置快照。

### 5.2 观察哪些曲线、在哪定 operating point

- 主曲线：**Risk Recall / Precision / False Positive Rate / Human Review Rate** 对阈值的 trade-off（每 scheme 一组；重点看 Agent 与 Rule(a)）。
- 只允许在 **validation 集**（§2.5）上选取 operating point；选点优先级建议：先压 FPR（防误伤商家红线，《00》§7.2-2），再保 Risk Recall，HUMAN Review Rate 作为可接受成本。
- operating point 定义 = (EVIDENCE_MIN_SIM, EVIDENCE_STRONG[, CONFIDENCE_ABSTAIN_THRESHOLD]) 一组值，记录选点理由与所选点的四条指标值。

### 5.3 校准结果回写

- 回写《00》§7.6 数值口径表、03-decisions.md §5 常量表（新增决策条目，注明"经 02-evaluation §5 sweep 校准"）。
- 同步修订 eval_case 标签中受影响的证据阈值口径（§2.2/《00》§11.2 注）。
- 报告必须附 **sweep 曲线**而不是只报最终点（《00》§11.5：证明阈值是"选"出来的，不是拍脑袋）。

---

## 6. Ablation Evaluation（可选，二期；引用《00》§13.4）

- 回答"Agent 每个组件是否真的必要"，同一 eval_dataset + 同一图结构逐组件去掉：
  Full Agent（基线）/ −RAG（无 CaseSearch+PolicySearch）/ −MerchantTool / −CaseTool / −ImageTool。
- **二期做，不阻塞三方案主对比**（M2 之后、RAG 接入后更可信）。
- 实现只做"图装配层不给该工具注册 / plan prompt 不注入该工具描述"，不动判定逻辑与评测集（差异唯一归因）；装配裁剪基于 04-graph-design 的注册表。
- 判定规则：去掉后指标几乎不变 → 组件必要性存疑；显著变差 → 必要能力（《00》§13.4）。
- 前置依赖：工具数据源至少达"能区分有/无该工具证据"的种子覆盖（InMemory 种子里查不到先例时，−CaseTool 必然无差异，结论失效——见 §7）。

---

## 7. 与 screening 修正集 / RAG 现状的关系

### 7.1 Rule baseline 依赖 screening（修正集进行中）

- Rule baseline 选项 (a) 复用 `pra.screening`，其行为随**并行修正集**变动（空规则集不得静默 PASS、brand 空缺不得直判 PASS、品牌词加词边界、品牌词命中 REJECT→COMPLEX 等）。
- **本文档刻意不与具体规则对齐**：harness 不做规则动作断言，只按"运行时当前行为"取数；报告必须记录 **screening 行为快照**（git commit / 规则语义版本）。
- 修正集未完成前跑 Rule baseline 会失真（漏放/误杀 → "Agent 比 Rule 强"可能是假象）→ **Rule baseline 正式跑分排在修正集合入之后**（与 code-review-backlog"与 02-evaluation 的耦合"一致）。
- Fix 5（品牌词命中 REJECT vs COMPLEX→Agent 上下文终裁）本身是评测实验点：评测集需含"品牌词命中但可能合法"样本，报告给出两种规则动作下的对比行。

### 7.2 RAG 未接 → Agent 工具数据源的结论边界

- RAG（Policy KB / Case KB 真实检索）尚未接入，Agent scheme 的 CaseSearch / PolicySearch / Merchant 以 **InMemory 种子数据**运行（scripted 模式，§3.4）。
- 局限：种子里缺失的真实先例/完整规避史 Agent 取不到 → 多信号与对抗类案的 Evidence Sufficiency、Marginal Evidence Gain 等指标会**低估上限**。
- 报告必带边界标注："工具为 InMemory 种子数据时的结论边界"；真实 RAG/Merchant 接入后以 real 模式复核（二期）。

---

## 8. 验收标准与里程碑

| 里程碑 | 内容 | 验收标准（骨架级） |
|---|---|---|
| **M1** | 评测集 v1 可用 | N ≥ 300（P-4 定终值）；五类分布达标（±5pp）；每条含 expected 三字段 + scene + source_type + 标注交叉校验；Hard 标记与 hard_reason 齐；loader/schema 校验通过；生成脚本可复现（同 commit 同产物） |
| **M2** | 三方案 harness 跑通同一数据集产出 metrics | rule/single_call_llm/agent 三 scheme 在全集（或 validation 子集）产出 EvalRecord 无异常；scripted 模式同 case 重跑**逐字节一致**（确定性重放断言）；metrics 模块有金标准小样本单测；Rule 正式跑分在 screening 修正集合入后执行（§7.1） |
| **M3** | sweep 曲线与 operating point | validation 上产出四条曲线（Recall/Precision/FPR/HRR）；记录 operating point 与选点理由；回写《00》§7.6 与 03 §5；标签阈值口径同步修订（§5.3） |
| **M4** | 评测报告 | 指标汇总 + 按 scene 分层 + 与《00》§12.4 预期结论表逐行对照（哪些符合/哪些反例及其解释）；明确回答 Q1–Q4；附结论边界标注（数据集版本 / LLM 模式 / 工具数据源 / 对齐映射） |

> M2 是"能跑"的门槛，M3/M4 才回答 Q1–Q4；M1 与 M2 可并行推进（loader + smoke 集先行）。

---

## 9. 待拍板清单汇总（实现前必拍；[建议] 为推荐默认，正文未按已定稿处理）

| # | 待拍板项 | 选项 | [建议] 默认 |
|---|---|---|---|
| P-1 | Rule baseline 用哪种 | (a) `pra.screening` 三分流，COMPLEX→HUMAN_REVIEW 映射；(b) 独立二分 rule（命中→REJECT 否则 PASS） | **(a) 为主 baseline**（结论可外推线上，输出空间天然对齐）；(b) 作"纯规则上限"敏感性补充。影响 §3.2/§3.5/§4.5 |
| P-2 | 评测时 Agent 用 scripted 桩还是真实 LLM、工具数据源 | scripted + InMemory（确定性、CI 可跑）vs real LLM + 真实 RAG/Merchant | **scripted + InMemory 为默认**；real 模式二期复核；报告标注结论边界（§3.4/§7.2） |
| P-3 | HUMAN_REVIEW 语义与对齐口径 | HRR 细分三率（§4.4）；二分类混淆矩阵是否剔除 HUMAN 期望案；Rule/Single-call 的 HUMAN 映射（§4.5） | **三分类统一对齐 + HRR 细分三率**；PASS/REJECT 真值案上算二分类指标，HUMAN 期望案单列 abstention 质量 |
| P-4 | 评测集规模/来源/HUMAN 期望案 | 规模 300 vs 500；复用 demo case（P_88231 等）与否；是否含 expected=HUMAN_REVIEW 案及其占比 | **规模先取下限 300（质量优先）；demo case 仅作 smoke 集 ≤10 条不混正式分布；含少量 HUMAN 期望案（建议 ≤10%）** 用于 abstention 真值（占比与口径随 P-3 联动） |
| P-5 | sweep 是否含 `CONFIDENCE_ABSTAIN_THRESHOLD` 联合扫描 | 主扫 EVIDENCE 双阈值网格；CONFIDENCE 二次扫或先固定 0.7 | **EVIDENCE 网格全扫、CONFIDENCE 先固定 0.7 观察**，主扫完成后再决定是否联合校准（§5.1） |

> 其余〔细化待定〕（不含 P 编号、可由实现者在写码前自行收敛或回主 agent 确认）：evidence 引用素材形态（§2.2）、数据文件分卷方式（§2.3）、Marginal Evidence Gain 加权口径（§4.2）、Evidence Sufficiency 精确公式（§4.2）、sweep 网格组合采样（§5.1）、harness 是否含 DB 真落库集成路径（§3.6）。
