# 电商平台商品内容治理 · 复杂风险调查 Agent —— 系统设计拆解（v1）

> 主线：**业务价值 → 决策难点 → 系统设计 → 验证结果**
> 原则：能用规则解决的，不交给 LLM；传统机审发现异常，Agent 调查复杂异常；Agent 不是全量审核系统，而是机审链路里的"复杂案件处理节点"。

---

## 0. 一句话定位

> 传统机审负责**确定性异常**（黑名单/关键词/Logo/类目/OCR/分类模型），Agent 负责**开放性、上下文依赖强、需要多源交叉验证的复杂风险案件**，产出 `PASS / REJECT / HUMAN_REVIEW`，并在证据不足时主动**克制地转人工**。

整个系统的核心叙事（面试主线）：

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
| 接入 | MQ（Kafka / RabbitMQ） | 异步解耦、削峰、事件溯源 |
| 同步机审 | FastAPI 服务 + 规则引擎 | 高吞吐、低延迟、确定性 |
| 分流 | 状态机 + 阈值规则 | 明确性决策，不滥用 LLM |
| 复杂调查 | LangGraph StateGraph（asyncio + Worker） | 预算控制、可恢复状态机 |
| 工具/RAG | 内部服务 + 向量检索（pgvector / Qdrant） | 多源数据访问 |
| 人工回流 | 审核工作台 + 反馈 Topic | 闭环、知识沉淀 |
| 全链路 | MySQL 状态机 + Redis 幂等/限流 + OTel | 可靠性、可观测性 |

### 1.3 关键架构决策：为什么"同步 + 异步"两段式

- 规则/传统模型是**同步确定性**链路，要求毫秒~秒级、可并行、成本为零（无 LLM）。
- Agent 是**慢、贵、有不确定性**的链路，只处理分流出来的复杂案件（预计 < 15% 全量），异步消费 + 预算限制 + 失败重试。
- 这样保证：Agent 的引入**不拖慢全量审核吞吐**，只对复杂案件追加一个"调查节点"。

---

## 2. 典型审核 Case 模型

Case 是系统的核心数据对象，**输入是商品事实，输出是结构化裁决 + 证据链**。

### 2.1 输入：ProductReviewCase（商品事实快照）

```json
{
  "case_id": "CASE_20240907_001",
  "product": {
    "product_id": "P_88231",
    "title": "新款厚底复古跑鞋 女士百搭运动鞋",
    "description": "经典复古跑鞋设计，轻量缓震，适合日常通勤和运动。",
    "category": "女鞋/运动鞋",
    "brand": null,
    "attributes": { "材质": "PU", "鞋底": "橡胶", "适用人群": "女士" },
    "sku_list": [
      { "sku_id": "S_1", "color": "米白", "size": "36-40", "price": 129.0 }
    ],
    "images": [
      { "url": ".../img1.jpg", "ocr_text": null, "source": "主图" }
    ],
    "listing_time": "2024-09-06 14:00:00",
    "version": 3
  },
  "merchant_id": "M_5512",
  "event_type": "NEW_LISTING",
  "screening_signals": [
    { "name": "KEYWORD", "result": "PASS", "score": 0.0 },
    { "name": "LOGO_DETECT", "result": "PASS", "score": 0.0 },
    { "name": "CATEGORY_RULE", "result": "PASS", "score": 0.0 },
    { "name": "IMAGE_NSFW", "result": "PASS", "score": 0.0 }
  ]
}
```

### 2.2 输出：ReviewDecision（结构化裁决）

```json
{
  "decision": "HUMAN_REVIEW",
  "risk_level": "HIGH",
  "risk_type": ["POTENTIAL_IP_RISK", "EVASION_PATTERN"],
  "confidence": 0.91,
  "evidence": [
    { "type": "IMAGE_SIMILARITY", "source": "ImageAnalysisTool", "value": "similarity=0.91, match=某品牌经典鞋款", "weight": 0.9 },
    { "type": "MERCHANT_HISTORY", "source": "MerchantTool", "value": "23 similar / 5 removals / 3 relisting", "weight": 0.85 },
    { "type": "CASE_PRECEDENT", "source": "CaseSearchTool", "value": "CASE_1832 → REJECT", "weight": 0.8 }
  ],
  "policy": ["POLICY_3.2"],
  "hypothesis_trace": [
    { "id": "H3", "statement": "刻意规避品牌识别", "prior": 0.2, "posterior": 0.88, "status": "SUPPORTED" }
  ],
  "budget_used": { "llm_calls": 4, "tool_calls": 5, "tokens": 18000, "latency_ms": 9200 }
}
```

### 2.3 Case 模型的设计要点

1. **输入 = 事实，输出 = 裁决 + 证据链**。证据链是核心，面试官问"为什么这么判"时，答案在 `evidence[]` 和 `hypothesis_trace[]` 里。
2. `screening_signals` 记录传统机审已做过什么、结果是什么，作为 Agent 的**起点信息**（避免 Agent 重复劳动）。
3. 结构化的 `risk_type` 使用**受控词表**（见 §7），保证可统计、可评测。

---

## 3. Agent State 模型

Agent 的状态**必须是显式、可序列化、可持久化、可恢复**的，而不是藏在 LLM 的上下文里。这是"工程化 Agent"和"Prompt 循环"的分水岭。

```json
{
  "run_id": "RUN_20240907_001_01",
  "case_id": "CASE_20240907_001",
  "status": "INVESTIGATING",           // PENDING | INVESTIGATING | DECIDED | ESCALATED | BUDGET_EXCEEDED | FAILED
  "hypotheses": [
    {
      "id": "H1", "statement": "普通复古设计，无违规",
      "prior": 0.5, "posterior": 0.05, "status": "REFUTED",
      "evidence_for": [], "evidence_against": ["image_similarity=0.91"]
    },
    {
      "id": "H2", "statement": "参考了某知名品牌经典设计",
      "prior": 0.4, "posterior": 0.9, "status": "SUPPORTED",
      "evidence_for": ["image_similarity=0.91"], "evidence_against": []
    },
    {
      "id": "H3", "statement": "刻意规避品牌识别",
      "prior": 0.2, "posterior": 0.88, "status": "SUPPORTED",
      "evidence_for": ["brand=null", "标题/描述无品牌", "商家3次改标题重上架"],
      "evidence_against": []
    },
    {
      "id": "H4", "statement": "商家系统性类似行为",
      "prior": 0.15, "posterior": 0.85, "status": "SUPPORTED",
      "evidence_for": ["23 similar / 5 removals"], "evidence_against": []
    }
  ],
  "evidence": [],                      // 已收集证据（去重、合并）
  "investigation_queue": [             // 待验证问题（优先级排序）
    { "q": "商品外观是否对应某品牌?", "priority": 1, "status": "DONE" },
    { "q": "商家是否系统性类似行为?", "priority": 2, "status": "DONE" }
  ],
  "tool_call_history": [               // 调用审计
    { "seq": 1, "tool": "ImageAnalysisTool", "args": "{...}", "result_ref": "E_01", "latency_ms": 1200, "tokens": 800 }
  ],
  "budget": {
    "llm_calls": 0, "tool_calls": 0, "tokens": 0,
    "start_time": "2024-09-07T10:00:00Z",
    "limits": { "max_llm_calls": 8, "max_tool_calls": 12, "max_tokens": 40000, "max_latency_ms": 30000 }
  },
  "decision": null                     // 收敛后写入 ReviewDecision
}
```

### 3.1 设计要点

1. **Hypothesis 是状态的核心**：Agent 不是"分类器"，而是"假设验证器"。每个假设有 `prior → posterior` 的演变，这是可解释的。
2. **状态持久化到 MySQL**：`AgentState` 即 LangGraph 的 State（`TypedDict`/Pydantic），由 LangGraph **Checkpointer** 每步后落库，worker 崩溃可恢复、可断点续跑、可复现（eval 重放）。
3. **`budget` 是状态的硬字段**：条件边路由函数在每轮进入节点前检查预算，超限即路由到转人工止损。
4. **`evidence` 与 `tool_call_history` 分离**：前者是"结论依据"，后者是"过程审计"，两者都进 trace。

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
| 7. Uncertainty / Abstention | `decide` 节点 + 确定性 guardrail | 证据不足 → HUMAN_REVIEW |
| 8. Cost / Latency Budget | 条件边的确定性 `budget_check` | 超限 → 直接路由到转人工 |

**StateGraph 草图：**

```python
class AgentState(TypedDict):
    case: dict                 # ProductReviewCase 事实快照
    hypotheses: list[dict]     # 风险假设（prior / posterior / status）
    evidence: list[dict]       # 已收集证据
    investigation_queue: list[dict]
    tool_call_history: list[dict]
    budget: dict               # 已用 + 限额
    decision: dict | None      # 收敛后的 ReviewDecision

graph = StateGraph(AgentState)

graph.add_node("hypothesize", hypothesize_node)   # LLM：生成/更新假设
graph.add_node("plan", plan_node)                 # LLM：决定下一步调查
graph.add_node("tools", ToolNode(tools))          # 确定性：执行工具
graph.add_node("reevaluate", reevaluate_node)     # LLM：证据综合、更新后验
graph.add_node("decide", decide_node)             # LLM 提案 + 硬规则校验

graph.set_entry_point("hypothesize")
graph.add_edge("hypothesize", "plan")

# plan 之后：有工具要调且预算够 → tools；否则 → decide
graph.add_conditional_edges("plan", route_after_plan,
    {"tools": "tools", "decide": "decide"})
graph.add_edge("tools", "reevaluate")

# reevaluate 之后：证据不足且预算够 → 回 plan；否则 → decide
graph.add_conditional_edges("reevaluate", route_after_reevaluate,
    {"continue": "plan", "decide": "decide"})

app = graph.compile(checkpointer=mysql_checkpointer)  # State 持久化、可恢复、可重放
```

### 4.3 关键分工：LangGraph 只做"编排骨架"，其余是确定性代码 + LLM 节点

这是本项目最重要的工程决策，面试必问。

| 步骤 | 谁来做 | 理由 |
|---|---|---|
| 图编排、条件边路由、预算记账 | LangGraph 图 + 确定性 Python 路由函数 | 可靠、可测、可控成本 |
| Tool 执行、结果反序列化、证据去重合并 | ToolNode + 确定性 Python | 不浪费 LLM token |
| 假设生成 / 计划 / 证据综合 / 决策推理 | LLM 节点（结构化 JSON + Schema 校验） | 需要语义推理 |
| 硬规则兜底（黑名单必 REJECT） | `decide` 节点的确定性 overlay | 安全红线 |
| State 持久化 / 恢复 / 重放 | LangGraph Checkpointer（MySQL） | 崩溃可恢复、eval 重放 |

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
| 4 | decide | 证据充分但涉及"仿冒"主观判定 → 转人工 | `HUMAN_REVIEW / HIGH / 0.91` |

> 注意第 4 步：Agent 的价值不仅是"自动判掉"，更是"知道什么时候证据足够、什么时候该人介入"。

### 4.5 为什么是"动态选工具"而不是固定流水线

- 固定 `A→B→C` 会对每个案件无差别调用所有工具，**浪费成本**（比如商品字段无冲突时没必要调 OCR）。
- 动态选工具的依据是：**"当前最值得验证的假设，需要哪条证据？"**——由 `plan` 节点决策、条件边路由到 `tools`。
- 例：若 OCR 已显示"100% Polyester"而标题写"真丝"，则 `plan` 应优先调 ProductTool 做字段交叉，而不是 ImageAnalysis。
- 这是 Agent 相比"固定 Workflow"的核心增量之一，也是评测里的 `Tool Selection Accuracy` 指标来源。

---

## 5. Tool 列表及职责

每个 Tool 必须回答"审核员为什么需要这个信息"。**v1 只做这 6 个，不为数量堆工具。**

| Tool | 回答的业务问题 | 输入 | 输出 | 实现类型 |
|---|---|---|---|---|
| **ProductTool** | 这个商品的事实是什么？（尤其 brand 是否为空、字段是否冲突） | product_id | 结构化商品快照（标题/描述/属性/品牌/SKU/图片/版本） | 确定性（MySQL 读） |
| **ImageAnalysisTool** | 商品外观是否与某品牌/违禁视觉高度相似？ | image url(s) | 相似度 Top-K、Logo 检测、视觉风险描述 | 混合（向量检索 + 视觉 LLM） |
| **OCRTool** | 图片里到底写了什么？ | image | 文字 + 坐标 + 语言 | 确定性（OCR 服务） |
| **MerchantTool** | 这个商家是否系统性地做类似行为？ | merchant_id | 商品总数、相似商品数、历史违规/下架/改标题重上架数、信用分 | 确定性（聚合 + 向量扫描） |
| **CaseSearchTool** | 有没有类似且已有人工裁决的先例？结论是什么？ | 自然语言/结构化查询 | Top-K 相似案例 + 元数据（决策/风险类型/关键证据/政策） | RAG（案例库检索） |
| **PolicySearchTool** | 当前有效政策对这类情况怎么说？ | 查询 + 类目/风险类型过滤 | Top-K 政策条款 + 引用（policy_id/版本/生效日期） | RAG（政策库检索） |

### 5.1 每个 Tool 的"为什么需要"一句话

- ProductTool：**事实锚点**——判断"规避品牌"必须先确认 brand 字段是否真空缺。
- ImageAnalysisTool：**多模态核心**——外观相似是本案最大的、规则无法覆盖的证据缺口。
- OCRTool：**交叉验证**——发现"标题/描述"与"图片实际内容"的冲突（如真丝 vs Polyester）。
- MerchantTool：**行为模式**——单商品看不出问题，商家的历史行为才是"规避"的关键信号。
- CaseSearchTool：**先例**——同类案件人是怎么判的，提供决策参照。
- PolicySearchTool：**政策依据**——当前规则下到底能不能判、判到什么程度。

### 5.2 Tool 的工程抽象（为未来 MCP 留口，但不提前引入）

统一 `Tool` 接口：

```python
class Tool(Protocol):
    name: str
    description: str            # 给 LLM 的 tool schema（JSON Schema）
    async def call(self, args: ToolArgs, ctx: ToolContext) -> ToolResult: ...
```

- 所有 Tool 注册到 `ToolRegistry`，Agent 的 Plan 步骤只输出 `{tool, args}`，由 Controller 调度执行。
- **MCP 的定位**：v1 的 6 个工具都是**内部服务**，用统一 `Tool` 接口 + 注册表即可，**不需要 MCP**。
- **什么时候才引入 MCP**：当需要接入**外部/第三方/跨语言**的工具（如外部图片检索服务、外部 OCR 厂商、公司其他团队的 Tool）时，MCP 的标准化价值才显现。v1 明确不引入，避免为技术而技术。

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

- **混合检索**：BM25（关键词，如"无品牌""模仿""重上架"）+ 向量相似度（语义），加权融合。
- **元数据过滤**：先按 `类目`、`risk_type`、`政策有效性（当前生效版本）` 过滤，再检索——避免检索到过期政策或不相关类目案例。
- **Rerank**：粗召回 Top-50 → 精排 Top-5（用重排模型或 LLM 打分），控制注入上下文的量。
- **引用格式**：检索结果必须带 `policy_id + 版本 + 条款原文` / `case_id + 决策`，进 `evidence[]` 时保留可追溯引用。

### 6.4 向量库选型

- 数据量小（政策几百条、案例几千条），**优先用 MySQL + pgvector 或 ES 的 kNN**，不引入重型向量库，降低工程复杂度。
- 向量模型：文本用通用 embedding（BGE/OpenAI text-embedding 均可）；**图片向量单独存**（ImageAnalysisTool 用于品牌款相似度检索的向量库，可与文本向量库分开）。

### 6.5 知识回流（闭环）

人工裁决结果 → 沉淀为新 CasePrecedent → 重新 embedding → 入 Case KB；政策更新 → 版本化 → 生效后替换检索范围。**这是系统"越用越准"的机制**，也是面试里能讲的"闭环设计"。

---

## 7. PASS / REJECT / HUMAN_REVIEW 决策机制

### 7.1 三分类语义

| 决策 | 语义 | 触发条件（概要） |
|---|---|---|
| **PASS** | 放行 | 所有高优先级假设被证伪，或风险低于阈值 |
| **REJECT** | 违规，拒绝上架 | 证据链达到"充分"且命中**可引用的违规依据**（政策/先例） |
| **HUMAN_REVIEW** | 转人工 | 证据不足 / 置信度低于阈值 / 预算耗尽 / 政策模糊 / 新型风险 |

### 7.2 决策规则（确定性兜底，LLM 只"提案"）

决策分两层：**LLM 提案 → 确定性规则校验**。

1. **硬规则优先（确定性 Python 代码，不可被 LLM 覆盖）**：
   - 调查中发现黑名单品牌 / 硬违规 → 强制 `REJECT`。
   - 即便 LLM 说 PASS，只要硬规则命中，以 REJECT 为准（防止漏放）。
2. **REJECT 需要"可引用依据"**：LLM 不能仅凭"感觉像"判 REJECT，必须至少一条证据指向明确政策条款或高度相似先例。**防止误伤商家**（过审拒审也是资损/商誉损失）。
3. **HUMAN_REVIEW 触发条件（abstention）**：
   - `confidence < 阈值`（如 0.7）；
   - 证据相互矛盾（如相似度高但商家历史干净）；
   - 预算耗尽（`BUDGET_EXCEEDED`）→ 带上已收集的部分证据转人工；
   - 政策模糊 / 无先例（novel risk）。
4. **PASS 需要**：所有高优先级假设被**证伪**（而非"没有证据"）——区分"证明无风险"和"没查到风险"。

### 7.3 风险类型受控词表（v1）

```
POTENTIAL_IP_RISK     疑似 IP / 品牌模仿
EVASION_PATTERN       规避审核行为模式
FALSE_CLAIM           虚假 / 无依据宣传
FIELD_CONFLICT        商品字段信息冲突
```

> 受控词表的意义：评测集的 `risk_type` 标签、`Policy KB` 的元数据、决策输出三者用同一套词表，才能做召回/精确率统计。

### 7.4 Confidence 的含义

`confidence` 不是 LLM 拍脑袋的数字，而是**证据完整性 + 假设后验 + 依据强度**的函数（可解释）：

```
confidence = f(最高假设 posterior, 证据链完整性, 是否存在可引用依据, 证据是否矛盾)
```

面试时可以讲：confidence 低 → 转人工，这就是"不确定性 / Abstention"能力的落地。

---

## 8. Guardrail / Budget 设计

### 8.1 Budget（成本 / 延迟护栏）

| 维度 | v1 阈值 | 超限行为 |
|---|---|---|
| 最大 LLM 调用次数 | 8 | 停止调查 → 输出部分证据 + HUMAN_REVIEW |
| 最大 Tool 调用次数 | 12 | 同上 |
| 最大 Token | 40,000 | 触发上下文压缩 / 停止 |
| 最大执行时间 | 30s | 超时 → 转人工 |

- Budget 在**条件边路由函数里、每次进入节点前**检查（确定性代码），不是"跑完才发现超了"。
- 超限的语义是：**"调查成本已超过可接受范围，证据不足以自动判，转人工最稳妥"**——这本身就是正确的业务行为，不是失败。

### 8.2 安全 / 业务 Guardrail

1. **硬规则不可被 LLM 覆盖**（黑名单、硬违禁）→ 防漏放。
2. **REJECT 必须有可引用依据** → 防误伤商家。
3. **PII / 敏感信息**：工具返回给 LLM 前做脱敏（商家联系方式等），LLM 输出不落地敏感字段。
4. **决策审计**：每个决策必须带完整 `evidence[] + hypothesis_trace[] + tool_call_history[]`，可回溯到"谁（哪个工具）提供的哪条证据导致这个结论"。
5. **幂等 / 去重**：同一商品同一版本只审一次（Redis setnx + MySQL 唯一索引），防止重复消费 MQ 重复计费。

---

## 9. 数据库核心表设计（MySQL 8 / InnoDB / utf8mb4）

> 主键统一 bigint（雪花），版本字段做乐观锁，状态字段加索引。JSON 列用于半结构化（如 attributes、agent_state）。ORM 用 SQLAlchemy 2.0（async）+ Alembic 管理迁移，Pydantic 负责模型校验与序列化。

### 9.1 核心表清单

| 表 | 职责 | 关键字段 |
|---|---|---|
| `product` | 商品主表 | product_id, merchant_id, title, description, category, brand, status, version, listing_time |
| `product_sku` | SKU | sku_id, product_id, color, size, price, status |
| `product_image` | 商品图片 | image_id, product_id, url, ocr_text, embedding(向量), is_primary |
| `merchant` | 商家主表 | merchant_id, name, credit_score, status |
| `merchant_event` | 商家行为事件 | event_id, merchant_id, event_type(违规/下架/改标题重上架), product_id, ts |
| `review_case` | 审核案件（一次事件一个） | case_id, product_id, event_type, triage_result, status, version |
| `review_signal` | 传统机审信号 | signal_id, case_id, signal_name, result, score |
| `agent_run` | Agent 运行 | run_id, case_id, status, budget_json, agent_state_json, final_decision |
| `agent_step` | Agent 步骤 trace | step_id, run_id, seq, step_type, input_json, output_json, tokens, latency_ms |
| `evidence` | 收集的证据 | evidence_id, run_id, type, source_tool, value, weight, ref_id |
| `decision` | 最终裁决 | decision_id, case_id, decision, risk_level, risk_type, confidence, policy_refs, evidence_json |
| `policy` / `policy_clause` | 政策库 | policy_id, version, category, risk_type, status, effective_date / clause_id, policy_id, text |
| `policy_chunk` / `policy_embedding` | 政策分块+向量 | chunk_id, clause_id, chunk_text, embedding |
| `case_precedent` | 历史案例库 | case_id, summary, decision, risk_type, risk_level, policy_refs |
| `case_chunk` / `case_embedding` | 案例分块+向量 | chunk_id, precedent_id, chunk_text, embedding |
| `eval_dataset` / `eval_case` | 评测集 | dataset_id, name, version / eval_case_id, dataset_id, case_json, expected_json |
| `eval_run` / `eval_result` | 评测运行 | run_id, scheme(rule/llm/agent), dataset_id, metrics_json / result_id, run_id, eval_case_id, actual_json, is_correct |

### 9.2 关键 DDL 示例（代表性强，非全量）

```sql
CREATE TABLE review_case (
  case_id       BIGINT PRIMARY KEY,
  product_id    BIGINT NOT NULL,
  merchant_id   BIGINT NOT NULL,
  event_type    VARCHAR(32) NOT NULL,
  triage_result VARCHAR(16) NOT NULL,          -- PASS / REJECT / COMPLEX
  status        VARCHAR(16) NOT NULL,          -- PENDING/SCREENED/INVESTIGATING/DECIDED
  version       INT NOT NULL,
  created_at    DATETIME(3) NOT NULL,
  KEY idx_status (status, created_at),
  KEY idx_product_version (product_id, version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE agent_step (
  step_id    BIGINT PRIMARY KEY,
  run_id     BIGINT NOT NULL,
  seq        INT NOT NULL,
  step_type  VARCHAR(24) NOT NULL,             -- HYPOTHESIZE/PLAN/TOOL_CALL/REEVALUATE/DECIDE
  tool_name  VARCHAR(64) NULL,
  input_json JSON NULL,
  output_json JSON NULL,
  tokens     INT NOT NULL DEFAULT 0,
  latency_ms INT NOT NULL DEFAULT 0,
  created_at DATETIME(3) NOT NULL,
  KEY idx_run (run_id, seq)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

### 9.3 Redis / MQ 使用

- **Redis**：幂等去重（setnx）、LLM/Tool 限流（令牌桶）、worker 抢占 case 的分布式锁、政策版本/案例索引热点缓存、agent_state 热快照（落库为准）。
- **MQ**：`product_review_request`（接入）、`complex_review`（分流投递）、`review_feedback`（人工裁决回流）、重试 + 死信队列。

---

## 10. Trace / Observability 设计

### 10.1 三层观测

| 层 | 内容 | 载体 |
|---|---|---|
| **链路 Trace** | 全审核链路（接入→机审→分流→Agent→决策）一个 traceId 贯穿 | OpenTelemetry（Jaeger/自建） |
| **Agent 内部 Trace** | Agent 每一步（Observe/Hypothesize/Plan/Tool/Re-evaluate/Decide）作为一个 span，记录 tool 名、args、结果、tokens、latency | 落 `agent_step` 表（JSON） + OTel span |
| **LLM 调用级 Trace** | 每次 LLM 调用的 prompt/输出/tokens/成本 | Langfuse（Python 生态 LLM 追踪事实标准） |
| **业务指标** | 决策分布、转人工率、自动化率、各风险类型占比 | Prometheus + 指标表 |

> 分工：**Langfuse** 管 LLM 调用级可观测性（自动记录每次调用的 prompt/输出/tokens/成本），**OTel** 管业务链路 trace，两者在 traceId 上对齐。

### 10.2 为什么 Agent Trace 落库

- **可复现**：eval 重放、回归测试、申诉调查（"这个商品为什么被拒"）。
- **成本核算**：每 case 的 llm_calls / tokens / latency 精确到步。
- **面试亮点**：这是"工程化 Agent"和"调 API 的 Demo"的本质区别。

### 10.3 核心指标（见 §11 完整列表）

成本侧：P50/P95 latency、平均/P95 LLM 调用次数、Token、单 case 成本。可靠侧：失败率、重试率、超时转人工率。

---

## 11. Evaluation Dataset 设计

### 11.1 数据集规模与分布（v1：300–500 Case）

| 类型 | 占比 | 说明 |
|---|---|---|
| 明确正常 | 20% | 规则和 Agent 都应 PASS |
| 明确违规 | 20% | 规则和 Agent 都应 REJECT |
| 边界案件 | 30% | 规则拿不准、单信号弱 |
| 多信号组合 | 20% | 需要多源交叉验证 |
| 对抗/规避 | 10% | 刻意规避审核（核心 Hard Case） |

### 11.2 每个 Case 的结构化标签

```json
{
  "eval_case_id": "EC_001",
  "input": { "...": "ProductReviewCase" },
  "expected": {
    "decision": "HUMAN_REVIEW",
    "risk_level": "HIGH",
    "risk_type": ["POTENTIAL_IP_RISK", "EVASION_PATTERN"],
    "evidence": ["image_similarity>=0.85", "merchant_history>=5_removals"],
    "applicable_policy": ["POLICY_3.2"]
  }
}
```

> 关键：标签**不是只有 PASS/REJECT**，而是结构化地包含 `expected_decision / risk_type / risk_level / evidence / applicable_policy`。这样评测能区分"结论对但理由错"（Decision Correct vs Reasoning Correct）。

### 11.3 评测指标

**业务指标**
- Risk Recall（违规召回）、Precision（精确率）、False Positive Rate（误伤率）
- Decision Accuracy（三分类正确率）
- Human Review Rate（转人工率）、Automation Rate（自动化率，= 1 - 转人工率）

**Agent 指标**
- Tool Selection Accuracy（选对了工具吗）
- Evidence Sufficiency（证据是否足以支撑结论）
- Reasoning Correctness（推理过程是否正确，即使结论对）

**工程指标**
- 平均/P95 LLM 调用次数、Tool 调用次数、Token Usage、P50/P95 Latency、单 Case 成本

### 11.4 评测 Harness

- 同一份 `eval_dataset`，三个 scheme（rule / single-call-llm / agent）跑同一 harness，产出可比 metrics。
- Agent 评测支持**确定性重放**（工具返回 mock 或录制的结果），保证可重复。

---

## 12. Rule / Single-call LLM / Agent 三套方案如何实现

> 这是项目的**核心证明**：同一批测试集，三个方案，回答"Agent 到底多解决了什么"。

### 12.1 Baseline 1：Rule Engine（规则引擎）

- 纯确定性：黑名单、关键词、敏感词、类目规则、Logo 检测、OCR 关键词、风险分阈值。
- 输出：命中规则 → REJECT，否则 PASS（**没有 HUMAN_REVIEW 语义**，或仅"低置信"即转人工）。
- 特点：快、零 LLM 成本、确定性，但**无法处理"多源交叉验证 + 上下文依赖"的复杂案件**。

### 12.2 Baseline 2：Single-call LLM（单次 LLM）

- **一次调用**：输入 = 商品原始数据（标题/描述/属性/OCR/图片）+ 少量背景，直接输出结构化决策 JSON。
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
4. **人工标注 + 交叉校验**：每条由人工给 `expected_decision + evidence + policy`，至少两人一致性校验。

### 13.3 Hard Case 的具体形态（对应核心场景）

| 子类型 | 例子 | Agent 增量 |
|---|---|---|
| 品牌模仿/规避 | 复古运动鞋 + 无品牌 + 相似度0.91 + 商家5次下架 | 图像相似 + 商家历史 + 案例先例 |
| 字段冲突 | 标题"真丝" vs OCR"Polyester" | OCR + 商品字段交叉验证 |
| 无依据宣传 | "7天瘦身""医学专家推荐" | Claim 提取 + 政策查询 + 证据查询 |
| 弱信号叠加 | 多个弱信号各自 PASS，组合后高风险 | 多源综合 |

---

## 14. MVP 范围：做什么 / 不做什么

### 14.1 MVP 必做（能跑通核心叙事的最小闭环）

1. 核心场景 1 个：**潜在 IP / 品牌模仿 / 规避审核**（复古运动鞋）。
2. 传统机审初筛（简版规则引擎，能跑出三分流）。
3. 6 个 Tool（Product / ImageAnalysis / OCR / Merchant / CaseSearch / PolicySearch）。
4. LangGraph StateGraph 编排的 Agent Loop（显式状态机 + 预算护栏 + 动态选工具）。
5. 两类 RAG（Policy KB + Case KB，各几十~几百条种子数据）。
6. 三分类决策 + 证据链输出。
7. 评测集 v1（300–500 Case，含 Hard Case）+ 三方案对比 harness。
8. 全链路 Trace + 指标 + 成本统计。

### 14.2 支撑场景（第二阶段再加，不阻塞核心）

- 虚假/无依据宣传（Claim 提取 + 证据推理）。
- 字段信息冲突（多源交叉验证）。

### 14.3 明确不做（反炫技，面试可主动讲"为什么不做"）

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

```
content-governance/
├── pyproject.toml                   # uv 管理依赖，统一版本
├── README.md
├── docs/
│   ├── 00-system-design.md          # 本文档
│   ├── 01-agent-loop.md             # StateGraph 节点/边/状态细化
│   └── 02-evaluation.md             # 评测方案细化
│
├── src/cg/                          # 主包
│   ├── common/                      # 通用：雪花ID、JSON工具、错误码
│   ├── domain/                      # 领域模型：Case/AgentState/Evidence/Decision（Pydantic）
│   ├── screening/                   # 传统机审初筛 + 三分流
│   │   ├── rule_engine/             # 规则引擎（黑名单/关键词/类目/阈值）
│   │   └── triage/                  # 三分类分流逻辑
│   ├── agent/                       # Agent 核心（LangGraph StateGraph）
│   │   ├── state.py                 # AgentState（TypedDict / Pydantic）
│   │   ├── graph.py                 # StateGraph：节点 + 边 + 条件边 + 路由
│   │   ├── nodes/                   # hypothesize / plan / reevaluate / decide 节点
│   │   ├── tools_node.py            # ToolNode + 6 个 Tool 注册
│   │   ├── guardrails/              # 预算护栏、硬规则兜底、决策校验（确定性）
│   │   └── checkpointer.py          # MySQL Checkpointer 接入
│   ├── tools/                       # 6 个 Tool + ToolRegistry + 统一 Tool 接口
│   │   ├── product/
│   │   ├── image_analysis/
│   │   ├── ocr/
│   │   ├── merchant/
│   │   ├── case_search/
│   │   └── policy_search/
│   ├── rag/                         # 政策库 + 案例库：分块、embedding、混合检索、rerank
│   ├── evaluation/                  # 评测 harness + 三方案对比 + Hard Case Benchmark
│   ├── api/                         # FastAPI 路由 + 审核工作台接口
│   └── infra/                       # MySQL/Redis/MQ/OTel/Langfuse/向量库 集成
│
├── migrations/                      # Alembic 数据库迁移
├── scripts/                         # 数据初始化、评测集生成、跑分
└── tests/                           # pytest：单测 + 评测（LLM 响应 mock 保证确定性）
```

### 15.1 模块划分理由

- **domain** 独立：Case/AgentState/Evidence 是核心领域模型（Pydantic），被 screening/agent/evaluation 三处复用。
- **agent / tools / rag 分层**：StateGraph 依赖 Tool 接口，不依赖具体实现（依赖倒置，便于 pytest mock 工具、未来替换）。
- **evaluation 独立**：评测是独立关注点，不污染业务代码。

---

## 16. 面试答辩备忘（把设计讲成故事）

1. **为什么需要 Agent**：传统机审解决确定性异常；复杂案件的风险证据**分散在商品/图片/商家历史/案例/政策多处**，必须"调查取证"才能决策，规则覆盖不了、单次 LLM 拿不到证据。
2. **Agent 是什么**：不是全量审核系统，是机审链路里的"复杂案件调查节点"，产出三分类 + 证据链。
3. **工程化体现在哪**：LangGraph StateGraph 编排 + 显式 Agent State（Checkpointer 持久化/可恢复）、条件边路由 + 预算护栏、LLM 只做语义推理、硬规则兜底、全链路 Trace 落库、评测集三方案对比。
4. **工程能力落在哪（语言换 Python，架构思维不变）**：MQ 异步解耦、Redis 幂等/限流/锁、MySQL 状态机 + 乐观锁、DDD 模块划分、asyncio 并发、可观测性——12 年 Java 架构沉淀的并发/分布式/状态机思维，在 Python 里直接复用。
5. **最得意/最难的点**：如何让 Agent "知道什么时候证据不足该转人工"（Abstention），以及如何用 Hard Case Benchmark 证明 Agent 的必要性。

---

> 下一步（进入实现前）：先做 **StateGraph 的节点/状态契约 + 6 个 Tool 的 JSON Schema + 一个可跑的最小闭环（核心场景单链路）**，再铺评测集。不要一上来就写全量代码。
