# 电商平台商品内容治理 · 复杂风险调查 Agent —— 设计口径与决策依据

> 主线：**业务价值 → 决策难点 → 系统设计 → 验证结果**
> 原则：能用规则解决的，不交给 LLM；传统机审发现异常，Agent 调查复杂异常；Agent 不是全量审核系统，而是机审链路里的"复杂案件处理节点"。
>
> **本文只承载口径与决策依据**：为什么这么设计、要满足什么不变量、什么算合规、什么明确不做。
> 字段 / 表结构 / 目录 / 工具 schema / 配置常量等实现规格一律以代码为准，本文只给指针；
> 实现状态（已落地 / 规划中）见 [README](../README.md) 的「当前实现边界」与「Roadmap」，
> 评测数字、数据集规模与评测口径见 [docs/02-evaluation.md](02-evaluation.md)。

---

## 0. 一句话定位

> 传统机审负责**确定性异常**（黑名单/关键词/Logo/类目/OCR/分类模型），Agent 负责**开放性、上下文依赖强、需要多源交叉验证的复杂风险案件**，产出 `PASS / REJECT / HUMAN_REVIEW`，并在证据不足时主动**克制地转人工**。

整个系统的核心叙事：

```
商品上架/变更
   ↓
传统规则 + 模型 初筛（同步、快、便宜）
   ↓
三分流：明确正常 / 明确违规 / 复杂低置信
   ↓
复杂低置信 → Complex Risk Investigation Agent（异步、预算受限）
   ↓
PASS / REJECT / HUMAN_REVIEW
   ↓
HUMAN_REVIEW → 人工裁决 → 回流案例库 + 策略库 + 评测集
```

---

## 1. 项目整体业务架构

### 1.1 分层架构（体现后端工程能力的关键层）

```
┌─────────────────────────────────────────────────────────────┐
│  接入层：商品上架 / 改标题 / 改属性 / 改图片 事件  →  MQ Topic │
│  （product_review_request）                                    │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────┐
│  同步机审层 Screening（确定性，低延迟，全量过）                  │
│  · Rule Engine（黑名单/关键词/敏感词/类目规则/禁止属性）          │
│  · 传统模型（图片分类/Logo检测/OCR/风险分模型）                  │
│  · 输出：signal 列表 + 风险分 + 三分类建议                       │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────┐
│  分流层 Triage（确定性决策，不是 LLM）                          │
│  · 命中硬规则且高分 → REJECT                                   │
│  · 无任何信号且低分   → PASS                                   │
│  · 中间带/低置信/多弱信号 → 投递复杂队列 complex_review          │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────┐
│  复杂调查层（异步 Worker Pool，预算受限）                        │
│  · Complex Risk Investigation Agent（LangGraph）             │
│  · 假设生成 → 计划 → 动态选工具 → 多源取证 → 证据综合 → 决策      │
│  · 工具：Product / ImageAnalysis / OCR / Merchant              │
│  · RAG：CaseSearch / PolicySearch                             │
└──────────────────────────────┬──────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────┐
│  决策输出 + 人工层                                              │
│  · PASS / REJECT → 落库 + 通知商家                             │
│  · HUMAN_REVIEW → 审核工作台队列 → 人工裁决 → 回流              │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 每一层的技术载体（Python 技术栈落点）

| 层 | 技术载体 | 体现的能力 |
|---|---|---|
| 接入 | 同步 HTTP `POST /api/v1/reviews`（MQ Topic 为后续扩展位） | 异步解耦、削峰、事件溯源 |
| 同步机审 | FastAPI 服务 + 规则引擎 | 高吞吐、低延迟、确定性 |
| 分流 | 状态机 + 阈值规则 | 明确性决策，不滥用 LLM |
| 复杂调查 | LangGraph StateGraph（asyncio） | 预算控制、可恢复状态机 |
| 工具/RAG | 内部服务 + 向量检索（ChromaDB） | 多源数据访问 |
| 人工回流 | 审核工作台 + 反馈 Topic | 闭环、知识沉淀 |
| 全链路 | MySQL 状态机 + Langfuse 观测（Redis 幂等/限流、OTel 链路为后续扩展位，见 §10.4） | 可靠性、可观测性 |

### 1.3 关键架构决策：为什么"同步 + 异步"两段式

- 规则/传统模型是**同步确定性**链路，要求毫秒~秒级、可并行、成本为零（无 LLM）。
- Agent 是**慢、贵、有不确定性**的链路，只处理分流出来的复杂案件（预计 < 15% 全量），异步消费 + 预算限制 + 失败重试。
- 这样保证：Agent 的引入**不拖慢全量审核吞吐**，只对复杂案件追加一个"调查节点"。

---

## 2. 典型审核 Case 模型

Case 是系统的核心数据对象，**输入是商品事实，输出是结构化裁决 + 证据链**。

### 2.1 输入：ProductReviewCase（商品事实快照）

字段与序列化形态以 `src/pra/domain/models.py` 为准，本文不再镜像字段清单。设计上这条输入必须自带
Agent 调查的**起点信息**：商品事实（标题 / 描述 / 属性 / 类目 / 品牌 / SKU / 图片 / 版本）、机审已产出的
OCR 文本与 `screening_signals`（即 §12.0 的"基础输入"）。

### 2.2 输出：ReviewDecision（结构化裁决）

字段与取值域以 `src/pra/domain/models.py` 为准，本文不再镜像。两条口径不变：`decision_confidence` 是
**自动决策的安全门槛**，与 `risk_level` 相互独立（§7.1 / §7.4）；支撑判定的证据必须带可追溯引用（§7.2）。

### 2.3 Case 模型的设计要点

1. **输入 = 事实，输出 = 裁决 + 证据链**。证据链是核心，需要解释"为什么这么判"时，答案在证据与假设轨迹里。
2. `screening_signals` 记录传统机审已做过什么、结果是什么，作为 Agent 的**起点信息**（避免 Agent 重复劳动）。
3. 结构化的 `risk_type` 使用**受控词表**（§7.3），保证可统计、可评测。

---

## 3. Agent State 模型

Agent 的状态**必须是显式、可序列化、可持久化、可恢复**的，而不是藏在 LLM 的上下文里。这是"工程化 Agent"和"Prompt 循环"的分水岭。

> 字段清单、合并（reducer）语义与数量上限以代码为准：`src/pra/agent/state.py`、
> `src/pra/agent/guardrails/schemas.py`。本文不再镜像 State 字段表。

### 3.1 设计要点

1. **Hypothesis 是状态的核心**：Agent 不是"分类器"，而是"假设验证器"。每个假设有 `prior → posterior` 的演变，这是可解释的。
2. **状态持久化**：`AgentState` 即 LangGraph 的 State，由 LangGraph **Checkpointer** 每步后持久化——「崩溃可恢复 / 断点续跑」以持久化落地为前提。
3. **预算（budget）是状态的硬字段**：条件边路由函数在每轮进入节点前检查预算，超限即路由到转人工止损。
4. **"结论依据"与"过程审计"分离**：证据是前者，工具调用历史是后者，两者都进 trace。
5. **图的终态只有一个**：收敛即终态；`PASS / REJECT / HUMAN_REVIEW` 是 `ReviewDecision.decision` 的取值，不是图终态；
   预算耗尽 / 工具失败 / 降级通过 decision 的 overrides 记录（§7 / §8）。`run_id` / `case_id` 等价于 LangGraph thread_id，
   运行状态由 DB `review_run.status` 承载。

### 3.2 AgentState 实现契约（结论）

- **只装调查记忆与图内控制通道**：调查记忆（商品事实 / 假设 / 证据 / 待验证问题 / 工具调用历史 / 预算 / 决策）外加图内控制通道；
  `run_id` / `case_id` 不进 state（等价于 thread_id），运行状态不进 state（落 DB）。
- **证据一旦收集不可篡改**：证据合并按去重键丢弃新增而非覆盖；工具调用历史与失败记录为追加。
- **不累积对话历史**：每个 LLM 节点都从 state 重新组装 prompt，单次调用内部的对话只存在于该次节点调用内 —— 可序列化、token 可预算、重放确定。
- **边界由入参 schema 约束，判定不读模型量**：假设数 / 调查队列长度 / 单轮计划工具数由 schema 限定；
  两道 Gate 与收敛判定都不消费 LLM 产出的 `prior` / `posterior`（§7.2）。
- **LLM 步失败降级**：结构化输出校验失败自动重试一次，仍失败则该节点返回降级结果并标记降级，后续 LLM 节点不再调用 LLM，
  统一按「证据不足」路由到 decide，由确定性 overlay 产出 `HUMAN_REVIEW`（硬规则命中除外）；重试上限与降级标记以
  `src/pra/agent/guardrails/llm_shell.py` 为准。
- **假设生命周期**：`PENDING → SUPPORTED / REFUTED / UNRESOLVED`（`UNRESOLVED` = 已查证但未能证实也未证伪，与「还没查」区分）；
  假设状态**只引导调查与留痕**，终裁由证据事实决定（§7.2），不由假设状态推导；假设在入口生成一次，运行中新出现的假设由再评估节点追加。

---

## 4. Agent Workflow（业务能力 → LangGraph StateGraph）

### 4.1 Loop 抽象（业务视角，框架无关）

```
Observe（读商品事实 + 已有信号 + 已有证据）
   ↓
Hypothesize（生成/更新风险假设）
   ↓
Plan（决定：验证哪个假设？还缺什么证据？）
   ↓
Act（动态选择 Tool 调用）
   ↓
Observe（读 Tool 结果 → 合并进证据）
   ↓
Re-evaluate（更新假设 posterior / 关停假设）
   ↓
Decision（充分 → PASS/REJECT/HUMAN_REVIEW；不足 → 回到 Plan 或转人工）
```

### 4.2 从 Agent 能力到 LangGraph StateGraph 的映射

> 设计顺序：**先明确"审核员需要哪些 Agent 能力"，再把每个能力映射为 LangGraph 的节点 / 边**——不是先有图再填业务。

| Agent 能力（业务） | LangGraph 落点 | 说明 |
|---|---|---|
| 1. Risk Hypothesis Generation | `hypothesize` 节点 | LLM 结构化输出初始/更新假设 |
| 2. Investigation Planning | `plan` 节点 | LLM 输出"下一步查什么、调哪个工具" |
| 3. Dynamic Tool Selection | `plan` 输出 + 条件边 → `tools` | `ToolNode` 按工具名 + 参数分发执行 |
| 4. Multi-source Investigation | `tools` 节点（6 个 Tool） | 商品/图片/OCR/商家/案例/政策 |
| 5. RAG | `case_search` / `policy_search` 两个 Tool | 在 `tools` 节点内调用 |
| 6. Evidence Synthesis | `reevaluate` 节点 | LLM 依据新证据更新假设 posterior |
| 7. Uncertainty / Abstention | `decide` 节点 + 确定性 Decision Gate | required 测量维度未覆盖（可补救）/ 维度不可测 / 阳性不足以自动拒绝 / Gate 不通过（含 decision_confidence<0.7、关键矛盾、关键 Tool 失败、预算耗尽、降级）→ HUMAN_REVIEW（§7.2） |
| 8. Cost / Latency Budget | 条件边的确定性 `budget_check` | 超限 → 直接路由到转人工 |

> 节点 / 边装配与条件边路由以 `src/pra/agent/graph.py` 为准，本文不再镜像代码草图。

### 4.3 关键分工：LangGraph 只做"编排骨架"，其余是确定性代码 + LLM 节点

这是本项目最重要的工程决策。

| 步骤 | 谁来做 | 理由 |
|---|---|---|
| 图编排、条件边路由、预算记账 | LangGraph 图 + 确定性 Python 路由函数 | 可靠、可测、可控成本 |
| Tool 执行、结果反序列化、证据去重合并 | ToolNode + 确定性 Python | 不浪费 LLM token |
| 假设生成 / 计划 / 证据综合 / 决策推理 | LLM 节点（结构化 JSON + Schema 校验） | 需要语义推理 |
| 硬规则兜底（黑名单必 REJECT） | `decide` 节点的确定性 overlay | 安全红线 |
| State 持久化 / 恢复 / 重放 | LangGraph Checkpointer（持久化后端可替换） | eval 重放与崩溃恢复 |

**为什么选 LangGraph（而非自研 Loop）**：LangGraph 的核心模型（显式 State + Graph + Checkpointer + 条件边）恰好就是本项目 Loop 需要的骨架，直接复用能省掉自写状态机 / 持久化 / 重放的重复劳动。但 LangGraph **只负责编排骨架**，以下仍是自研：预算护栏、硬规则兜底、证据去重、Tool 的 JSON Schema、决策 guardrail、trace 落库——既吃到框架红利，又保留工程可控性。

> 关键边界：**让 LLM 做语义推理（节点内部），让确定性代码做路由与安全（条件边 + guardrail）**。如果整张图都由 LLM 决定怎么走，会带来成本失控、不可控、不可测三个问题。

### 4.4 一次完整调查的推演（硬 Case：复古运动鞋）

| 轮次 | 节点 | 内容 | 结果 |
|---|---|---|---|
| 0 | hypothesize | 读商品事实 + 机审信号：全部 PASS、brand=null、无品牌词 | 建立 H1~H4 四个初始假设 |
| 1 | plan | 最高优先级疑点：外观是否对应某品牌？ | 选 ImageAnalysisTool |
| 1 | tools | 调 ImageAnalysisTool | similarity=0.91 → 证据 E_01 |
| 1 | reevaluate | H1 被削弱，H2 被支持 | H2.posterior ↑ |
| 2 | plan | 需确认商品字段与商家历史 | 选 ProductTool + MerchantTool |
| 2 | tools | 调 ProductTool | brand=null、标题/描述无品牌 → E_02 |
| 2 | tools | 调 MerchantTool | 23 相似 / 5 下架 / 3 改标题重上架 → E_03 |
| 2 | reevaluate | H3、H4 被支持 | H3.posterior 0.88 |
| 3 | plan | 需要"先例 + 政策"支撑 | 选 CaseSearchTool + PolicySearchTool |
| 3 | tools | RAG 查类似案例 | CASE_1832 高度相似 → REJECT → E_04 |
| 3 | tools | RAG 查政策 | POLICY_3.2 支持高风险转人工 → E_05 |
| 3 | reevaluate | 证据链完整 | 收敛 |
| 4 | decide | 证据充分但涉及"仿冒"主观判定 → 转人工 | `HUMAN_REVIEW / HIGH / 0.95` |

> 注意第 4 步：Agent 的价值不仅是"自动判掉"，更是"知道什么时候证据足够、什么时候该人介入"。

### 4.5 为什么是"动态选工具"而不是固定流水线

- 固定 `A→B→C` 会对每个案件无差别调用所有工具，**浪费成本**（比如商品字段无冲突时没必要调 OCR）。
- 动态选工具的依据是：**"当前最值得验证的假设，需要哪条证据？"**——由 `plan` 节点决策、条件边路由到 `tools`。
- 例：若 OCR 已显示"100% Polyester"而标题写"真丝"，则 `plan` 应优先调 ProductTool 做字段交叉，而不是 ImageAnalysis。
- 这是 Agent 相比"固定 Workflow"的核心增量之一，也是评测里的 `Tool Selection Accuracy` 指标来源。

### 4.6 四个 LLM 节点契约要点（结论）

| 节点 | 契约要点 |
|---|---|
| `hypothesize` | 入口执行 1 次；读商品事实 + 机审信号，产出「初始假设集（允许少提，不强制产出"正常假设"）+ 初始调查队列」 |
| `plan` | 每轮输出「下一步验证哪条假设、调哪个工具、为什么」（≤3 条/轮）；没有值得做的动作时输出 conclude，直接转 decide |
| `tools` | 确定性执行：按工具名 + 参数分发，做结果反序列化与证据去重合并（不消耗 LLM）；单批调用按上限截断 |
| `reevaluate` | 依据新证据更新假设 posterior 与状态、可追加新假设；证据不足且预算未超 → 回 plan，否则转 decide（`posterior`/`status` 只用于引导调查与留痕，**不参与终裁**，见 §7.2） |
| `decide` | LLM 只产出 `DecisionProposal`（提案），随后由确定性 overlay 按 §7.2 顺序收口，改判与归因码写入 `overrides` |

- **重复动作防护**：plan 若反复提议同一调用 → 去重 guardrail 第 2 次即清空计划并视同 conclude，不存在"空转烧预算"的死循环。

---

## 5. Tool 列表及职责

每个 Tool 必须回答"审核员为什么需要这个信息"。**v1 只做这 6 个，不为数量堆工具**；
字段级输入 / 输出 schema 以 `src/pra/tools/`（`base.py` 与各工具目录）为准，本文只写职责。

| Tool | 回答的业务问题（职责） |
|---|---|
| **ProductTool** | 这个商品的事实是什么？（尤其 brand 是否为空、字段是否冲突） |
| **ImageAnalysisTool** | 商品外观是否与某品牌 / 违禁视觉高度相似？ |
| **OCRTool** | 图片里到底写了什么？（用于与标题 / 描述交叉验证） |
| **MerchantTool** | 这个商家是否系统性地做类似行为？ |
| **CaseSearchTool** | 有没有类似且已有人工裁决的先例？结论是什么？ |
| **PolicySearchTool** | 当前有效政策对这类情况怎么说？ |

### 5.1 每个 Tool 的"为什么需要"一句话

- ProductTool：**事实锚点**——判断"规避品牌"必须先确认 brand 字段是否真空缺。
- ImageAnalysisTool：**多模态核心**——外观相似是本案最大的、规则无法覆盖的证据缺口。
- OCRTool：**交叉验证**——发现"标题/描述"与"图片实际内容"的冲突（如真丝 vs Polyester）。
- MerchantTool：**行为模式**——单商品看不出问题，商家的历史行为才是"规避"的关键信号。
- CaseSearchTool：**先例**——同类案件人是怎么判的，提供决策参照。
- PolicySearchTool：**政策依据**——当前规则下到底能不能判、判到什么程度。

### 5.2 Tool 的工程抽象（为未来 MCP 留口，但不提前引入）

统一 `Tool` 接口（协议与工具 schema 以 `src/pra/tools/base.py` 为准，本文不再镜像）：

- 所有 Tool 注册到 `ToolRegistry`，Agent 的 Plan 步骤只输出 `{tool, args}`，由 Controller 调度执行。
- **MCP 的定位**：v1 的 6 个工具都是**内部服务**，用统一 `Tool` 接口 + 注册表即可，**不需要 MCP**。
- **什么时候才引入 MCP**：当需要接入**外部/第三方/跨语言**的工具（如外部图片检索服务、外部 OCR 厂商、公司其他团队的 Tool）时，MCP 的标准化价值才显现。v1 明确不引入，避免为技术而技术。

### 5.3 六个 Tool 的实现现状（重要边界）

装配层决定每个 Tool 的数据源：默认装配路径与评测世界使用**固定种子数据**（评测确定性红线 —— CI 不连外部数据源），
生产 / HTTP 入口读真库与真实 RAG；具体实现类与装配入口以 `src/pra/tools/`、`src/pra/wiring.py` 为准。

**真链路不可用时（库 / 索引 / 依赖不可达）必须在首次工具调用显式失败并记为失败，不静默回退种子数据** ——
这是"禁止静默降级"的系统契约（§8.2）。

---

## 6. RAG 知识库设计

### 6.1 两类知识，作用不同

| 知识库 | 内容 | 作用（关键：不是教 LLM 规则） |
|---|---|---|
| **Policy KB（政策库）** | 审核规则、平台政策、违规判定标准（带版本/生效日期） | 在**具体案件调查中**提供"当前适用政策依据" |
| **Case KB（案例库）** | 历史人工审核案例（商品、证据、最终裁决、依据政策） | 提供"相似案件先例"作为决策参照 |

> **定位澄清**：RAG 不是"让 LLM 背规则"（那是规则引擎的活），而是**在 Agent 调查过程中，把"政策依据"和"历史案例依据"作为证据注入**，让决策有据可依、可引用、可审计。

### 6.2 数据结构

**Policy KB：**

```
Policy (policy_id, 版本, 标题, 类目, 风险类型, 状态, 生效日期, 失效日期)
  └── Clause (clause_id, 条款原文)   ← 检索/分块的最小单元
        └── Chunk + Embedding
```

**Case KB：**

```
CasePrecedent (case_id, 商品摘要, 商家摘要, 证据摘要, decision, risk_type, risk_level, 适用policy)
  └── 检索文本 = 结构化字段拼接（"无品牌标识 + 外观高度模仿 + 商家多次类似上架"）
        └── Chunk + Embedding
```

### 6.3 检索策略

- **混合检索**：BM25（关键词，如"无品牌""模仿""重上架"）+ 向量相似度（语义），融合后取 Top-K。
- **元数据过滤**：按 `类目`、`risk_type`、`政策有效性（当前生效版本）` 过滤，避免检索到过期政策或不相关类目案例。
  **两路的过滤机制不同、语义必须等价**：向量路把过滤**下推给 Chroma `where`**（在库侧收窄打分域）；
  BM25 路是 `bm25s` 内存索引、**没有 `where`**，只能在 Python 候选集上建索引打分。两侧等价性由
  `tests/test_rag_retrieval.py` 的过滤器全组合用例锁住（漏召回只在带过滤时暴露）。
- **融合与截断**：向量 + BM25 混合召回经 **RRF（Reciprocal Rank Fusion）** 融合后取 Top-K，控制注入上下文的量；**v1 不做重排序（rerank）环节**。
- **引用格式**：检索结果必须带 `policy_id + 版本 + 条款原文` / `case_id + 决策`，进 `evidence[]` 时保留可追溯引用。

### 6.4 向量库选型

- 向量库选型为 **ChromaDB**（Docker 服务端 + HttpClient，LlamaIndex 装配）；语料属小语料量级，**v1 明确不做 ES / Milvus / 知识图谱**，不引入重型向量库，降低工程复杂度。
- 向量模型：文本在**生产 / 测试 / 回归 / 评测中用同一真语义模型**（BGE 系）—— 不得为测试引入确定性 mock 编码器，
  具体模型与维度以 `src/pra/rag/embedding.py` 为准。**图片向量单独存**（ImageAnalysisTool 的品牌款相似度检索可与文本向量库分开）。

### 6.5 知识回流（闭环）

人工裁决结果 → 沉淀为新 CasePrecedent → 重新 embedding → 入 Case KB；政策更新 → 版本化 → 生效后替换检索范围。**这是系统"越用越准"的机制**，形成系统的知识闭环。

### 6.6 RAG 实现现状与硬约束（结论）

**链路**：`Query → 双路召回（向量（ChromaDB, cosine）+ BM25）→ RRF 融合 → Top-K → CaseSearch / PolicySearch Tool → Evidence → Agent`。
索引装配与两路实现以 `src/pra/rag/` 为准（`chroma_store.py` / `index.py` / `bm25.py` / `factory.py`），本文只写口径。

- **唯一检索后端是 chroma**：不保留后端开关；生产 / HTTP 入口即用 chroma + BGE + hybrid，默认装配路径与评测世界用种子数据（§5.3）。
- 🔴 **建库必须显式按 cosine 度量**：缺省度量会让"像不像"的**间距语义整体偏离**（排名与分数一起失真，**且不报错**）。
- 🔴 **向量取数的分直接采库口径 `exp(-distance)`，不做量纲换算**：① 排名只依赖单调性，任何 distance 的单调降函数给出同一顺序；
  ② 该分往下只变成检索分（渲染给 LLM 的证据行 + 落库审计），**不参与任何 Gate 判定**（Gate 对案例先例 / 政策引用只判存在性、不读分值）；
  ③ `1 − distance` 在 `d > 1` 时被 clamp 塌成 0，低分区区分度反而更差。
- 🔴 **检索分（`retrieval_score`）不设取值域约束，也不做量纲适配**：它是**后端口径的检索分**，取值域由检索后端定义
  （BM25 = 原始分**无界**；vector = `exp(-distance)` ⊂ (0,1]；hybrid = RRF ⊂ (0, `2/60`]）。schema 不为一个既不归它管、
  也不参与决策的量 policing 取值域，代码也不把三种量纲"拉平"；用 min-max 把等值集抬成满分属**伪造**，不做。
  下游证据 `weight` 同理只要求非负、**不设上界**——`weight` 的上界从来不是为检索分设的（它源自外观相似度的三档阈值语义），
  且其全部读点都是 `>= 常量` 比较与 `max()`。
- 🔴 **检索分是检索分，任何场合不得称为「语义相似度」**，也不参与 Gate 判定（Gate 对案例先例 / 政策引用只要求**存在**）。
  外观相似度是另一类证据（`IMAGE_SIMILARITY`），两者量纲不同。
- **三模式（vector / bm25 / hybrid）分数量纲互不可比**，跨模式只断言「候选完整 + 可复现 + 案例库与评测真值零交集」。
- **过滤语义的两处实现必须逐条等价**：向量路把过滤下推到库侧 `where`，BM25 路只能在 Python 候选集上过滤；
  改任一侧都要同步另一侧，并由 `tests/test_rag_retrieval.py` 的全组合用例锁死（§6.3）。
- **写入与 metadata 形状**：同 id 记录不会被覆盖，写入须先删后加；metadata 形状变更必须换 collection 名，
  否则旧库被原样复用、新过滤条件**静默零命中**。
- **取回多少就是多少**：`top-k` 不保证返回数量等于 `top_k`；检索**真失败**（服务端不可达 / collection 不存在 / 编码失败）直接上抛，
  不另补一套检索系统兜底。
- **BM25 路自持分词与索引**：不替换第三方库的全局符号，各检索器实例互不共享状态 —— 并发下互不干扰，无锁、无调用顺序约束。
- **不得声称「CI 覆盖 chroma」**：CI 不安装 RAG extra，chroma 相关用例在 CI 上不执行。
- 🔴 **进程内向量库模式（内存 / 临时 client）对 id / 维度等约束比真服务端宽松**，缺陷可能在单测全绿下潜伏、**只在真服务端暴露**
  ⇒ 真服务端集成用例不可省。

---

## 7. PASS / REJECT / HUMAN_REVIEW 决策机制

### 7.1 两个概念 + 三分类语义

> **Abstention 的核心不是"置信度低于阈值就转人工"，而是"证据是否足以支持安全的自动决策"**。
> 为此把两个易混的量分开（§7.2 / §7.4 展开）：

| 概念 | 一句话定义 | 说明 |
|---|---|---|
| **decision_confidence** | 对"自动决策（不放人工）"的安全性把握 —— **安全门槛量，不是模型判"是否违规"的真实概率** | 输出 `ReviewDecision.decision_confidence` 即此值；只回答"如果自动判，判错风险够不够低"，不回答"风险有多高" |
| **risk_level / risk confidence** | 风险本身的高低（LOW/MEDIUM/HIGH）与风险强度（如最高支持假设的 posterior） | **仅供展示与人工队列排序**，独立于决策结论：HIGH risk + 证据不足 = HUMAN_REVIEW，不是 REJECT（见 §7.5） |

| 决策 | 语义 | 触发条件（概要，完整 Gate 见 §7.2） |
|---|---|---|
| **PASS** | 放行 | **PASS Gate**（全读事实通道）：无维度匹配的**阳性证据** AND 无**规则侧阳性**（R-102/R-302）AND 本案 required 测量维度**全覆盖**（无 NOT_MEASURED / UNMEASURABLE）AND 无关键工具失败 AND 无关键矛盾 |
| **REJECT** | 违规，拒绝上架 | **REJECT Gate**：证据链中存在与风险维度匹配的**硬阳性**（强相似 ≥0.85 / Logo / 商家行为脏且本 listing 有外观信号 / R-302 规避词命中）AND 存在带 `ref_id` 的可引用依据 AND `decision_confidence ≥ 0.7` AND 无关键矛盾 |
| **HUMAN_REVIEW** | 转人工（克制地 abstain） | 不满足任一自动 Gate，或命中弃权清单：必需测量维度缺失（可补救）/ 维度在本环境不可测 / 阳性不足以自动拒绝 / 关键证据冲突 / 关键 Tool 失败 / Budget Exhausted / 任一步降级 |

### 7.2 决策规则（确定性兜底：LLM 只"提案"，Gate 做最终校验）

决策分两层：**LLM 提案 → 确定性 Decision Gate 校验**。LLM 提案给出
`decision / risk_level / risk_type / decision_confidence / evidence / policy`；
确定性 overlay 依次执行下列规则，任一不满足即改写为 `HUMAN_REVIEW` 并记录 `overrides` 原因码
（overlay 实现在 `src/pra/agent/guardrails/gate.py`；归因码 `R1_*` ~ `R5_*` 的**全表定义也在该文件**，此处不复制）。

1. **硬规则优先（确定性代码，不可被 LLM 覆盖）**：
   - 调查中发现黑名单品牌 / 硬违规 → 强制 `REJECT`。
   - 即便 LLM 说 PASS，只要硬规则命中，以 REJECT 为准（防止漏放）。
2. **REJECT Gate**：可自动 REJECT 当且仅当 **全部满足**：
   - 存在**足以授权自动拒绝的阳性**（`measurements.reject_positive_dims`）：本 listing 直接测量的硬阳性（IMAGE_SIMILARITY ≥0.85 / IMAGE_LOGO），或**商家行为脏且有本 listing 外观信号佐证**（弱相似 ≥0.70 亦算），或平台规则命中**规避词**（R-302，文本自证）；
   - 存在**可引用依据**——至少一条带 `ref_id` 的政策条款或高度相似先例证据（**防止误伤商家**，过审拒审也是资损/商誉损失）；
   - `decision_confidence ≥ 0.7`（**安全门槛**，非模型真实概率，见 §7.4/§7.6）；
   - 无**关键矛盾**（如相似度极高但商家历史干净）。

   > 说明：**弱相似（0.70~0.85）不算硬阳性**——单独不足以自动拒绝（历史上正是高置信误杀的来源），
   > 其应然处置是与"商品事实"维度交叉，在库不可核验时该维度落 NOT_MEASURED、案件自然转人工；
   > **"仅商家行为脏"也不授权自动拒绝**——商家画像是针对该商家的统计，不能当作本 listing 违规的确证
   > （reviewer 语义：疑似规避但图/文本无确证 → 克制转人工），R-102 品牌词同理只阻塞 PASS。
   > 两道 Gate **不读 LLM 生成量**（`prior` / `posterior` / `Hypothesis.status` / `evidence_for` /
   > 提案里的 `confidence`）；`proposal` 只决定走哪道 Gate，以及非判定性的展示字段。
3. **HUMAN_REVIEW 触发条件（abstention）**——下列**任一**成立即转人工（即便 LLM 提案为 PASS/REJECT）：
   - **关键测量缺口**：required 维度中本环境**可测却没测**（归因码 `R3_MEASUREMENT_MISSING`，属**可补救**缺口 → 路由先回环补测）；
   - **维度不可测**：required 维度在本环境**没有可用测量来源**（`R3_DIMENSION_UNMEASURABLE`，重跑无用、不回环）；
   - **阳性不足以自动拒绝**：只有弱信号 / 仅商家画像 / 无可引用依据（`R3_POSITIVE_INSUFFICIENT`）；
   - **关键证据冲突**（`R3_EVIDENCE_CONFLICT`）：决定性证据互相矛盾；
   - **关键 Tool 失败**导致证据缺失（`R3_KEY_TOOL_FAILED`，如 ImageAnalysis 调用失败且无法重试）；
   - 预算耗尽（Budget Exhausted，§8.1）→ 带上已收集的部分证据转人工；
   - 任一步降级（`R5_DEGRADED_OR_FAILED_STEP`）；
   - PASS 提案过不了 PASS Gate（`R4_PASS_GATE_FAIL`）/ REJECT 提案过不了 REJECT Gate（`R2_REJECT_GATE_FAIL`）。
4. **PASS Gate**：可自动 PASS 当且仅当 **全部满足**：
   - 无任何维度匹配的**阳性证据**（含仅商家画像）；
   - 无**规则侧阳性**：R-102 品牌词 / R-302 规避词命中（确定性规则层即时求值）→ 阻塞 PASS；
   - 本案 **required 测量维度全部覆盖**，且结论为阴性 —— required 由**案件可观测事实**导出（商品/商家/文本恒必需，带图案件另有外观维度；实现见 `src/pra/agent/guardrails/measurements.py`），**不由 LLM 的 plan 或假设决定**；
   - 无关键 Tool 失败、无关键矛盾。
   - 三态的意义：`MEASUREMENT` 证据承载"测过（含**阴性**）"；`NOT_MEASURED` / `UNMEASURABLE` **不是证据**，由 Gate 用"required × 证据存在性 × 环境能力"推导 —— 不为"缺席"造证据，从而保住区分"证明无风险"（可 PASS）与"没查到风险"（应 HUMAN_REVIEW）。

### 7.3 风险类型受控词表（v1）

```
POTENTIAL_IP_RISK     疑似 IP / 品牌模仿
EVASION_PATTERN       规避审核行为模式
FALSE_CLAIM           虚假 / 无依据宣传
FIELD_CONFLICT        商品字段信息冲突
```

> 受控词表的意义：评测集的 `risk_type` 标签、`Policy KB` 的元数据、决策输出三者用同一套词表，才能做召回/精确率统计。

### 7.4 decision_confidence 的含义（与 risk confidence 分离）

- **`decision_confidence`**（输出 `ReviewDecision.decision_confidence`，即安全门槛）不是 LLM 拍脑袋的数字，也不是"违规概率"，而是**确定性函数**（可解释、可单测），可作为自动决策是否安全的一个计算来源：

  ```
  decision_confidence = 0.40*coverage + 0.30*strength + 0.20*citation + 0.10 - 0.20*conflict
  ```

  `coverage` = required 维度的覆盖比例；`strength` = 各已覆盖维度**决定性证据**的平均强度（有阳性取阳性最大，否则取阴性测量的可信度）；`citation` = 是否存在带 `ref_id` 的可引用依据；`conflict` = 是否命中关键矛盾。**四项全部来自证据事实，不读 LLM 的 `posterior`**；系数为结构性给定、未用任何数据集拟合，阈值 0.7 亦未动。

  它只回答"如果自动判（PASS/REJECT），判错风险是否足够低"——**自动 REJECT 的安全门槛是 0.7**（§7.2-2 / §7.6），低于门槛 → 转人工。
- **risk confidence / risk_level** 与决策结论**分离**：风险强度由最高支持假设的 `posterior`（见 `hypothesis_trace`）与 `risk_level` 表达，**只用于展示与人工队列排序**；它回答"风险有多高"，**不**参与路由或 Gate 判定（HIGH risk + 证据不足 → HUMAN_REVIEW，见 §7.5）。
- Abstention（克制转人工）的判断落点是 **Decision Gate**（§7.2）：`decision_confidence < 0.7` 只约束**自动 REJECT 侧**；PASS 侧由 PASS Gate 判定（无阳性 + 无规则阳性 + required 维度全覆盖且阴性 + 无关键失败/矛盾），**不以低风险置信度转人工**——干净商品低风险置信是正常态，不是 abstention 信号。

### 7.5 risk_level ≠ decision（不参与路由）

- `risk_level`（LOW/MEDIUM/HIGH/NONE）只用于**展示、人工队列排序、审核优先级与统计**；**不作为路由或 overlay 的判定输入**（避免把展示口径变成判定逻辑）。
- 典型反例（面试可讲）：**risk_level = HIGH 且证据不足 → HUMAN_REVIEW**，而不是 HIGH → REJECT。风险高 ≠ 可以自动判；能否自动判取决于 §7.2 的 Decision Gate（证据 + 政策依据 + decision_confidence）。

### 7.6 数值口径总览（v1 工程初始值，非理论最优，未做数据集校准）

> 本节三个数值都是 **v1 工程初始值**（第一版可跑的起点，不是调参后的最优值），以**代码常量为单一来源**（不写死在分支逻辑里）；
> **未做数据集校准**：曾有的评测侧阈值扫点能力已移除（§11.5），这些值不参与拟合，也不因评测结果回改。

| 数值 | 语义 | 取值依据 |
|---|---|---|
| 相似度 `0.70 / 0.85` | ImageAnalysis 三档证据分界：`<0.70` 不作证据 / `0.70~0.85` 普通证据 / `≥0.85` **Strong Evidence**（代码常量 `EVIDENCE_MIN_SIM` / `EVIDENCE_STRONG`，见 §11.5） | 工程初始值，未做数据集拟合（§11.5） |
| `decision_confidence 0.7` | 自动 REJECT 的**安全门槛**（非模型真实概率） | 工程初始值：门槛不达标即转人工，未按误伤 / 漏放校准 |
| LLM `10` / Tool `15` 上限 | **Budget 是 Guardrail 上界、不是目标调用次数**（§8.1）；余量用于 schema 重试 1 次、工具失败恢复与防无限循环 | 观测 Budget Utilization（§11.3）——正常案件应明显低于上限 |

---

## 8. Guardrail / Budget 设计

### 8.1 Budget（成本 / 延迟 Guardrail）

四组上限（LLM 调用次数 / Tool 调用次数 / Token / 执行时间）为**可配置的 Guardrail 上界**，默认值是 v1 工程初始值
（见 §7.6），具体常量与配置项以 `src/pra/agent/guardrails/budget.py` 为准。超限行为按维度定义：

| 维度 | 超限行为 |
|---|---|
| LLM 调用次数 / Tool 调用次数 | 停止调查 → 输出已收集的部分证据 + HUMAN_REVIEW（归因码 `R3_BUDGET_EXHAUSTED`）；工具节点同批执行也按上限截断 |
| Token | 触发上下文压缩 / 停止 |
| 执行时间 | 超时 → 转人工 |

- **语义（重要）**：Budget 是 **Guardrail 上界，不是目标调用次数**。正常案件的实际调用应**明显低于上限**——
  主链路（§4.4 走查）的调用量远低于上限，多出的余量只用于覆盖 schema 校验失败重试、工具失败恢复与防无限循环。
  评测用 **Budget Utilization**（§11.3）证明 Agent 不是为耗完预算而调工具。
- Budget 在**条件边路由函数里、每次进入节点前**检查（确定性代码），不是"跑完才发现超了"。
- 上限**运行时可由配置覆盖**；Trace 层记录四组占用率（§10.3）。
- 超限的语义是：**"调查成本已超过可接受范围，证据不足以自动判，转人工最稳妥"**——这本身就是正确的业务行为，不是失败。

### 8.2 安全 / 业务 Guardrail

> 下列为与实现进度无关的设计约束；哪些已落地见 [README](../README.md) 的「当前实现边界」。

1. **硬规则不可被 LLM 覆盖**（黑名单、硬违禁）→ 防漏放。
2. **REJECT 必须有可引用依据** → 防误伤商家。
3. **PII / 敏感信息**：工具返回给 LLM 前做脱敏（商家联系方式等），LLM 输出不落地敏感字段。
4. **决策审计**：每个决策必须带完整 `evidence[] + hypothesis_trace[] + tool_call_history[]`，可回溯到"谁（哪个工具）提供的哪条证据导致这个结论"。
5. **幂等 / 去重**：同一商品同一版本只审一次（幂等键 + 唯一索引），防止重复消费导致重复计费。

### 8.3 终止性（结论）

- 图里**唯一的回环**是 `plan → tools → reevaluate → plan`；`decide` 无出边。离开回环只有三个出口：收敛判定、plan 侧 conclude、预算超限。
- 每一轮回环至少消耗 1 次 plan + 1 次 reevaluate 的 LLM 调用 → 由 LLM 调用上限可推出**回环轮数上界约 4 轮**（实际因早停更少）。
- 不烧预算的"空转"也被堵死：plan 反复提议同一动作 → 去重 guardrail 第 2 次即清空并视同 conclude；工具只有 6 个且 Tool 调用上限兜底；任一节点降级即短路进 decide。
- 预算在**每个节点入口与每次条件边路由**都检查（确定性纯函数），任一维度超限 → 带部分证据转人工（`overrides=["R3_BUDGET_EXHAUSTED"]`）。
- 路由 / 预算 / 收敛 / Gate 全为纯函数（无随机、无 LLM）→ 同 state 必同后继，不存在"同 state 走不同分支"的非确定性死循环。

---

## 9. 数据库核心表设计（MySQL 8 / InnoDB / utf8mb4）

> 设计约定：主键统一 bigint（雪花），版本字段做乐观锁，状态字段加索引；JSON 列用于半结构化数据。
> ORM 用 SQLAlchemy 2.0（async），迁移为 `migrations/` 下的**手写 SQL**（不用 Alembic 自动生成），Pydantic 负责模型校验与序列化。

### 9.1 核心表清单

表的划分与字段清单**一律以 `migrations/*.sql` 为准**（ORM 映射在 `src/pra/infra/`），本文不再镜像字段表。
划分口径：商品事实（product / product_sku / product_image）、商家画像与行为事件（merchant / merchant_event）、
案件与运行轨迹（review_case / review_run / review_trace / review_evidence / review_result）。
其中商品图片表**不存 OCR 文本与向量列** —— OCR 归 OCRTool、向量归向量库，同一事实不两处存储。

### 9.2 关键 DDL 示例（代表性强，非全量）

DDL 真身在 `migrations/*.sql`，本文不再镜像建表语句。

### 9.3 Redis / MQ 使用（**规划，未实现** —— 现状为同步 HTTP + 进程内状态）

> v1 不引入 Redis / MQ：核心闭环不依赖它们（可选项不得成为默认路径的前置条件）。下列是出现并发与异步需求时的落点。

- **Redis**：幂等去重（setnx）、LLM/Tool 限流（令牌桶）、worker 抢占 case 的分布式锁、政策版本/案例索引热点缓存、agent_state 热快照（落库为准）。
- **MQ**：`product_review_request`（接入）、`complex_review`（分流投递）、`review_feedback`（人工裁决回流）、重试 + 死信队列。

---

## 10. Trace / Observability 设计

### 10.1 观测分层与现状

| 层 | 内容 | 载体 |
|---|---|---|
| **Agent 内部 Trace** | Agent 每一步（Hypothesize/Plan/Tool/Re-evaluate/Decide）的 tool 名、args、结果、tokens、latency | 落 MySQL `review_trace` 表（业务审计的真相所在） |
| **LLM 调用级 Trace** | 每次 LLM 调用的 prompt / 输出 / tokens / 成本 | Langfuse（§10.4） |
| **链路 Trace** | 全审核链路（接入→机审→分流→Agent→决策）一个 traceId 贯穿 | OpenTelemetry / Jaeger |
| **业务指标** | 决策分布、转人工率、自动化率、各风险类型占比 | 指标表 / Prometheus |

> 各层的落地状态见 [README](../README.md) 的「当前实现边界」。

> 职责分离：**Langfuse 管 LLM 调用级可观测性**，**MySQL `review_trace` 管业务步骤审计**——两者互补，不互相替代。

### 10.2 为什么 Agent Trace 落库

- **可复现**：eval 重放、回归测试、申诉调查（"这个商品为什么被拒"）。
- **成本核算**：每 case 的 llm_calls / tokens / latency 精确到步。
- **意义**：这是"工程化 Agent"和"调 API 的 Demo"的本质区别。

### 10.3 核心指标（见 §11 完整列表）

成本侧：P50/P95 latency、平均/P95 LLM 调用次数、Tool 调用次数、Token、单 case 成本。
预算侧：**Budget Utilization**（四组占用率 `llm_calls/max_llm_calls`、`tool_calls/max_tool_calls`、
`tokens/max_tokens`、`latency/max_latency`，来自 review_trace（逐步 tokens/latency）与 decision_json.budget_used）——正常案件占用率应明显低于 1，
用于证明"Budget 是 Guardrail 而非目标"（§8.1）。
可靠侧：失败率、重试率、超时/超限转人工率。

### 10.4 Langfuse 可观测性（现状与口径）

- **本地 Docker 自托管**（`deploy/langfuse`，UI :3000，端口全部绑回环）；适配层在 `src/pra/observability/`。
- **无凭据 → 回落 `NullTracer`（全 no-op）**：**观测路径不因 Langfuse 联网** —— 缺凭据时 Langfuse 侧不产生网络调用、不阻塞主流程。
  这条只约束观测：默认开发路径（起 HTTP 服务）本身会连真实 LLM 网关。
- **开关语义（最容易说错的一条）**：**未设 `PRA_LANGFUSE_ENABLED` = 启用**，但缺凭据即实际 no-op；只有 `PRA_LANGFUSE_ENABLED=0` 才是显式强制关闭。
- **口径红线（不得伪造观测）**：scripted / 桩路径下 **token=0 / cost 为空 / latency≈0 是真实情况**（没有真实 provider 调用），**绝不填充**；
  真实 token / latency 出现在任何走真实 LLM 的路径（生产 HTTP 入口与 real LLM 评测）。两个世界的数字不得混算或互相暗示。
- 埋点覆盖 root / 节点 / generation / tool / gate；评测可带 `eval_case_id` 把 trace 与评测用例关联。

---

## 11. Evaluation Dataset 设计

> 评测集真身在 `eval_data/`（JSONL + manifest + schema 校验），可执行口径见 [docs/02-evaluation.md](02-evaluation.md)；本文只写设计口径。

### 11.1 数据集规模与分布（实测规模：v1 35 Case；v2 320 Case）

| 类型 | 占比 | 说明 |
|---|---|---|
| 明确正常 | 20% | 规则和 Agent 都应 PASS |
| 明确违规 | 20% | 规则和 Agent 都应 REJECT |
| 边界案件 | 30% | 规则拿不准、单信号弱 |
| 多信号组合 | 20% | 需要多源交叉验证 |
| 对抗/规避 | 10% | 刻意规避审核（核心 Hard Case） |

### 11.2 每个 Case 的结构化标签

每条 case = 输入快照 + 结构化期望标签；字段与序列化形态以 `eval_data/`（JSONL + schema）与
`src/pra/evaluation/dataset/` 为准，本文不再镜像。

> 关键：标签**不是只有 PASS/REJECT**，而是结构化地包含 期望决策 / 风险类型 / 风险等级 / 证据 / 适用政策。
> 这样评测能区分"结论对但理由错"（Decision Correct vs Reasoning Correct）。
> 标注时的证据阈值标签（如 `image_similarity>=0.85`）是标注口径，必须与**运行时证据阈值**口径一致
> （v1 工程初始值，§7.6 / §11.5）；阈值变更后同步修订标签。

### 11.3 评测指标

**业务指标**
- 二分类五指标（真值 `expected.decision ∈ {PASS, REJECT}`）：Accuracy（决策准确率）、Precision（精确率）、Recall（违规召回）、False Positive Rate（FPR，误伤率）、False Negative Rate（FNR，漏放率）
- Accuracy 口径：**预测 HUMAN_REVIEW 计为错**（HUMAN_REVIEW 不作第三分类混入 Accuracy，只以转人工观测量单列）
- `human_review_rate`（转人工率，输出 HUMAN_REVIEW 的 case 占比）、`automation_coverage`（自动化覆盖率，= 1 − human_review_rate）

**Agent 指标**
- Tool Selection Accuracy（选对了工具吗）
- Evidence Sufficiency（证据是否足以支撑结论）
- Reasoning Correctness（推理过程是否正确，即使结论对）
- **Marginal Evidence Gain / Investigation Efficiency**（每次 Tool Call 带来多少新有效信息）——Investigation Efficiency =
  Σ(单次调用后 decision_confidence 增量或新增关键证据数) / Tool Calls，用于暴露"为调查而调查"的低效调用
- **Budget Utilization**：四组占用率（llm_calls / tool_calls / tokens / latency 各 ÷ 对应上限），
  证明 Budget 是 Guardrail 而非目标（§8.1）

**工程指标（成本与调查效率）**
- LLM Calls（平均/P95）、Tool Calls（平均/P95）、Token Usage、P50/P95 Latency、单 Case 成本（Cost）
- 上述指标的**分位数与分布**（不只均值），用于回答"Agent 贵在哪、是否值得"

### 11.4 评测 Harness

- 同一份 `eval_dataset`，三个 scheme（rule / single-call-llm / agent）跑同一 harness，产出可比 metrics。
- Agent 评测支持**确定性重放**（工具返回 mock 或录制的结果），保证可重复。

### 11.5 阈值与口径校准（**阈值扫描能力已移除**）

- 相似度分界与决策门槛都是 **v1 工程初始值**，未在数据集上拟合；生产判定读的就是这些常量。
- **曾有的评测侧阈值扫点（threshold sweep）已移除**：被扫的阈值只作用于评测确定性审查员读证据的视图，
  与生产实际读常量的位置不是同一处，且评测世界相似度数据只有 {0.72,0.73} 与 {0.90+} 两簇 ——
  曲线恒定或近平，对任何决策都不可行动。移除决策与边界见 [docs/02-evaluation.md](02-evaluation.md) §5。
- 要校准生产阈值，须先把可配置点下沉到生产实际读常量的位置并补齐中间相似度带数据，届时另行设计。

---

## 12. Rule / Single-call LLM / Agent 三套方案如何实现

> 这是项目的**核心证明**：同一批测试集，三个方案，回答"Agent 到底多解决了什么"。

### 12.0 公平性前提：三方案共享同一"基础输入"

为避免"各方案看到的材料不一样"的作弊质疑，三方案在**同一份 eval_case 上共享同一基础输入**：

```
基础输入 = ProductReviewCase 的商品事实快照：
  标题 / 描述 / 属性 / 类目 / 品牌 / SKU / 图片 / 机审已产出的 OCR 文本（images[].ocr_text，§2.1）/ screening_signals
```

区别只在**谁允许用这份输入之外的信息**：
- **Rule**：只允许对基础输入跑确定性规则（含 OCR 文本关键词）；
- **Single-call LLM**：基础输入一次性全部交给 LLM（一次调用输出决策 JSON）；
- **Agent**：允许按证据缺口**动态调查**——经 Tools/RAG 获取基础输入之外的证据（商家历史、案例库、政策库等，见 §12.3）。

### 12.1 Baseline 1：Rule Engine（规则引擎）

- **输入**：仅基础输入（§12.0），纯确定性：黑名单、关键词、敏感词、类目规则、Logo 检测、OCR 关键词、风险分阈值。
- **输出**：复用 `pra.screening` 三分流——PASS→PASS、REJECT→REJECT、**COMPLEX→HUMAN_REVIEW**（评测语义：无 Agent 时复杂案只能人工；与线上 "COMPLEX→Agent" 是不同口径，docs/02 §3.2/§4.5）。
- **特点**：快、零 LLM 成本、确定性，但**无法处理"多源交叉验证 + 上下文依赖"的复杂案件**。

### 12.2 Baseline 2：Single-call LLM（单次 LLM）

- **一次调用**：输入 = 全部基础输入（§12.0：标题/描述/属性/图片/机审 OCR 文本/机审信号）+ 少量背景，直接输出结构化决策 JSON。
- **关键：不给它商家历史、案例库、政策库**——因为那些正是 Agent 通过工具"调查"获取的证据。否则等于作弊。
- 变体（可选消融实验）：
  - **2a**：仅商品原始数据（隔离变量 = 多步调查）
  - **2b**：把政策+案例预塞进 prompt（RAG-in-prompt）——用来隔离"是缺证据，还是缺多步推理"

### 12.3 System 3：Agent + RAG + Tools

- 完整 Loop：假设生成 → 计划 → 动态选工具 → 多源取证（含 RAG）→ 证据综合 → 三分类决策。

### 12.4 三方案对比要证明的结论（预期，不虚构数据）

| 案件类型 | Rule | Single-call LLM | Agent |
|---|---|---|---|
| 明确正常/违规 | ✅ 快且准 | ✅ | ✅（但没必要，成本高） |
| 边界/单弱信号 | ⚠️ 误判 | ⚠️ 不稳定/证据不足 | ✅ 多步取证 |
| 多信号组合 | ❌ 无法交叉验证 | ⚠️ 缺证据 | ✅ 交叉验证 |
| 对抗/规避（Hard） | ❌ 直接漏过 | ⚠️ 输入里没信号 | ✅ 靠调查取证 |

> 结论落点：**Agent 的增量价值集中在"需要多步取证才能获得决策证据"的案件上**；对确定性案件，Agent 反而是过度设计（这就是为什么 Agent 只做复杂案件节点，不做全量）。

---

## 13. Hard Case Benchmark 如何构造

这是"最重要的 Benchmark"，专门证明 Agent 的必要性。

### 13.1 Hard Case 的定义（三选其一即算 Hard）

1. **Rule Engine 容易误判/无法判断**：没有任何硬规则命中，但存在真实风险。
2. **Single-call LLM 证据不足/判断不稳定**：决策依赖输入里**不存在**的信息（需调查获取）。
3. **Agent 可通过多步调查获得额外证据**：调用工具后能显著改变结论。

### 13.2 构造方法（多源合成 + 人工审核）

1. **人工构造对抗样本**：基于核心场景（品牌模仿/规避），故意设计"无 Logo、无品牌词、但外观高度相似 + 商家有规避史"。
2. **从真实案例改写**：把历史人工裁决案件脱敏改写为 benchmark 条目。
3. **程序化变异**：对模板做字段变异（改相似度、改商家历史、改 OCR 冲突）制造边界。
4. **人工标注 + 交叉校验**：每条由人工给 `expected_decision + evidence + policy`；**质量门槛是至少两人一致性校验**。
   当前数据集为**单一标注者**、未做第二标注者交叉校验，属已知局限（见 `02-evaluation` §2）；标注者构成随数据集 manifest 记录。

### 13.3 Hard Case 的具体形态（对应核心场景）

| 子类型 | 例子 | Agent 增量 |
|---|---|---|
| 品牌模仿/规避 | 复古运动鞋 + 无品牌 + 相似度0.91 + 商家5次下架 | 图像相似 + 商家历史 + 案例先例 |
| 字段冲突 | 标题"真丝" vs OCR"Polyester" | OCR + 商品字段交叉验证 |
| 无依据宣传 | "7天瘦身""医学专家推荐" | Claim 提取 + 政策查询 + 证据查询 |
| 弱信号叠加 | 多个弱信号各自 PASS，组合后高风险 | 多源综合 |

> 注：Hard Case 里的相似度（如 0.91）与运行时证据阈值（0.70/0.85，§7.6）口径一致——0.91 属于
> **Strong Evidence（≥0.85）** 档；阈值是 v1 工程初始值（§11.5）。

### 13.4 Ablation Evaluation（方案级；组件级能力已移除）

回答"给 Single-call 更多文本、再多步主动调查各能提升多少"——在同一 eval_dataset 上跑三个变体：

| 变体 | 装配差异 | 要回答的问题 |
|---|---|---|
| **2a** | Single-call + Raw Input | 一次调用只看基础输入的上限 |
| **2b** | Single-call + RAG-in-prompt（预塞政策/先例文本） | 多给文本能提升多少？（本评测世界实测 2a≡2b，该腿 inert） |
| **2c** | Multi-step Agent（主动调查） | 相对 Single-call 的整体差异 |

- 评价口径：各变体在 **Decision Accuracy / Risk Recall / False Positive Rate / Human Review Rate** 上的差异
  （三变体均保留 Decision Gate，三分类口径一致）。
- **口径边界**：2c 相对 2a/2b 的差异是"工具 + 多步 + mock 变体"的**整体**差异，不可分解归因于"主动调查"。
- **曾有的组件级消融（逐工具裁剪）已移除**：它需要图装配层裁剪参数与评测侧裁剪口径，只服务一次性的
  组件必要性论证；改为在消融报告里附**工具证据覆盖**（哪些 case 的 `expected_tools` 含该工具），
  覆盖为空的工具不做"不必要"判定。

---|---|---|
| **Full Agent**（基线） | 无 | 完整 Agent 的上限 |
| Agent - RAG（无 CaseSearch/PolicySearch） | 案例库 + 政策库检索 | 没有"政策依据/先例"时，REJECT 与 HUMAN_REVIEW 的质量掉多少？ |
| Agent - MerchantTool | 商家历史画像 | 单看商品/图片能否识别"规避模式"（H3/H4 类假设）？ |
| Agent - CaseTool | 案例先例（Case KB 检索） | 没有先例参照时决策是否漂移？（可看作 RAG 消融的细分） |
| Agent - ImageTool | 图像相似度/Logo | 没有多模态外观证据时，IP 风险类 Hard Case 是否漏放？ |

- 评价口径：各变体与 Full Agent 在 **Decision Accuracy / Risk Recall / False Positive Rate / Human Review Rate** 上的差异
  （5 个变体均保留 Decision Gate，三分类口径一致，无需等价映射）。
- 判定规则：**去掉某组件后指标几乎不变 → 该组件（或其在 plan 中的使用策略）需要重新审视是否真有必要**；
  显著变差 → 该组件对某类案件是必要能力。结果同时回答"为什么 6 个工具不多不少"。
- 实现要点：消融只做"图装配层不给该工具注册 / plan prompt 不注入该工具描述"，**不动判定逻辑与评测集**，
  保证差异唯一归因于被去掉的组件。

---

## 14. MVP 范围：做什么 / 不做什么

### 14.1 MVP 必做（能跑通核心叙事的最小闭环）

1. 核心场景 1 个：**潜在 IP / 品牌模仿 / 规避审核**（复古运动鞋）。
2. 传统机审初筛（简版规则引擎，能跑出三分流）。
3. 6 个 Tool（Product / ImageAnalysis / OCR / Merchant / CaseSearch / PolicySearch）。
4. LangGraph StateGraph 编排的 Agent Loop（显式状态机 + 预算护栏 + 动态选工具）。
5. 两类 RAG（Policy KB + Case KB，各几十~几百条种子数据）。
6. 三分类决策 + 证据链输出。
7. 评测集（分阶段：先小集跑通、再扩到含 Hard Case 的正式集）+ 三方案对比 harness。
8. 全链路 Trace + 指标 + 成本统计。

### 14.2 支撑场景（第二阶段再加，不阻塞核心）

- 虚假/无依据宣传（Claim 提取 + 证据推理）。
- 字段信息冲突（多源交叉验证）。

### 14.3 明确不做（避免过度设计）

| 不做 | 为什么 |
|---|---|
| Multi-Agent | 单案件调查是**单智能体多步推理**，一个 Loop 就能解决；拆多 Agent 只会增加协调成本和上下文割裂，业务无必要 |
| MCP | 6 个工具都是内部服务，统一 Tool 接口 + 注册表已够；接入外部/第三方 Tool 时才需要 |
| 微调/模型训练 | 决策逻辑靠证据+政策，不靠"背答案"；评测集还没稳定就微调是过早优化 |
| 知识图谱 | v1 的案例/政策用向量+BM25 检索足够，图谱复杂度收益不成比例 |
| 10 个审核场景 | 先证明 1 个 Hard Case 的闭环，再横向扩展 |
| 全量 Agent 审核 | Agent 定位是复杂节点，全量交给 Agent 既贵又慢还引入不确定性 |

---

## 15. 最终项目目录结构（Python 3.12 + uv + FastAPI，src 布局）

目录结构以仓库实际布局为准（[README](../README.md) 的「项目结构」与 `src/pra/`），本文不再镜像目录树。
分层约定不变：`src/pra/` 是唯一主包，按 `domain / screening / agent / tools / rag / evaluation / api / infra / observability` 分包；
SQL 迁移在 `migrations/`，数据初始化与跑分脚本在 `scripts/`，测试在 `tests/`。

### 15.1 模块划分理由

- **domain** 独立：案件、证据等核心领域契约（Pydantic）位于 `pra.domain`，被 screening / agent / evaluation 三处复用；Agent 的 State 归 `pra.agent`。
- **agent / tools / rag 分层**：StateGraph 依赖 Tool 接口，不依赖具体实现（依赖倒置，便于 pytest mock 工具、未来替换）。
- **evaluation 独立**：评测是独立关注点，不污染业务代码。
