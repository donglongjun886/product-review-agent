# 复杂风险调查 Agent —— StateGraph 节点 / 边 / 状态契约细化（01-agent-loop v1）

> 本文档是《00-system-design.md》在 **Agent Loop（实现层）** 上的细化与契约化，服务对象是实现者（写 `src/cg/domain`、`src/cg/agent/**`、`src/cg/tools/**` 的人）。
> 引用约定：凡提到总设计原文均写作 **《00》§x.y**（如《00》§4.2），避免与本文编号混淆。
>
> **对齐承诺**：本文档不推翻《00》任何决策；与《00》冲突时以《00》为准，并在第 9 章"待定项"登记，交设计方拍板后再修订本文。
>
> **本文档不包含业务代码**：只写 TypedDict/Pydantic 伪代码、JSON Schema、确定性路由/Guardrail 伪代码；这些是"契约"，不是 `src/cg/` 下可直接运行的实现。

---

## 1. 范围与前置

### 1.1 本文档回答什么

本文档把《00》§4.2 的 StateGraph 草图落成**实现层唯一依据**，逐一定义：

| 契约对象 | 对应《00》章节 | 本文章节 |
|---|---|---|
| AgentState 字段、类型、写入方、reducer | §3 / §4.2 | 2 |
| 四个 LLM 节点（hypothesize / plan / reevaluate / decide）的输入输出与校验降级 | §4.1 / §4.2 / §4.3 | 3 |
| plan 节点的"动态选工具"输出契约 | §4.5 / §5.2 | 4 |
| 6 个 Tool 的 args / result 字段级 Schema 与"结果→Evidence" | §5 | 5 |
| 条件边路由伪代码（含 budget_check） | §4.2 / §8.1 | 6 |
| decide 节点的确定性 overlay | §7.1 / §7.2 / §8.2 | 7 |
| 复古运动鞋案例逐轮对齐 | §4.4 | 8 |
| 需要设计方拍板的待定项 | — | 9 |

### 1.2 前置（已确认，不再讨论）

- 技术栈：Python 3.12、LangGraph `StateGraph`、Pydantic v2（`litellm` 做模型网关，MySQL Checkpointer 持久化）。
- 图只有 5 个节点、2 条条件边，与《00》§4.2 草图**严格一致**：

```
entry ──> hypothesize ──> plan ──(条件)──> tools ──> reevaluate ──(条件)──> decide（终结点）
```

- 目录落点（实现者照此创建文件，本文档只描述契约，不替你创建代码文件）：
  - `src/cg/domain/`：Pydantic 领域模型（Case / Evidence / Hypothesis / Budget / ReviewDecision / 各类 LLM 输出模型）。
  - `src/cg/agent/state.py`：`AgentState` TypedDict + reducer 声明（第 2 章）。
  - `src/cg/agent/graph.py`：节点注册、静态边、`add_conditional_edges`、编译（第 6 章）。
  - `src/cg/agent/nodes/`：hypothesize / plan / reevaluate / decide 四节点（第 3、4、7 章）。
  - `src/cg/agent/tools_node.py`：ToolNode（执行、args 校验、结果→Evidence、预算记账）（第 4、5 章）。
  - `src/cg/agent/guardrails/`：`budget.py`（budget_check）、`hard_rules.py`、`decision_guardrail.py`、`dedup.py`（第 6、7 章确定性代码）。
  - `src/cg/agent/checkpointer.py`：MySQL Checkpointer 接入。
  - `src/cg/tools/<six>/`：6 个工具子包 + `ToolRegistry` + 统一 `Tool` 接口（第 5 章）。

### 1.3 三条贯穿性原则（实现时不可违背）

1. **状态显式化**（《00》§3）：跨步骤信息全部走 `AgentState` 结构化字段，不把推理过程藏在 LLM message 历史里 → **不使用 `MessagesState`**（理由见 2.6）。
2. **分工**（《00》§4.3）：LLM 只做节点内部语义推理（结构化 JSON 输出）；**图怎么走（条件边）、预算、硬规则、去重、决策校验全部是确定性 Python**，不允许 LLM 决定路由。
3. **超限即止损**（《00》§8.1）：Budget 检查发生在**条件边路由函数内、进入下一个节点之前**；超限 → 路由到 decide，由 decide 的确定性 overlay 产出带部分证据的 `HUMAN_REVIEW`（这是正确业务行为，不是失败）。

---

## 2. AgentState 契约

### 2.1 总述

`AgentState` 声明为 `TypedDict`（LangGraph channel 声明用），**字段值**用 `src/cg/domain/` 下的 Pydantic 模型承载（校验、序列化、JSON Schema 生成）。全部字段必须可 JSON 序列化（Checkpointer 每步落库，《00》§3.1）。

各节点只**返回自己负责的字段**（partial update），LangGraph 按各字段的 reducer 语义合并进状态。

```python
# src/cg/agent/state.py 契约（伪代码，不落地）
from typing import TypedDict, Annotated, Literal, NotRequired
from operator import add

class AgentState(TypedDict):
    run_id: str                        # 见 2.3
    case_id: str                       # 见 2.3
    case: ProductReviewCase            # 见 2.3
    status: AgentRunStatus             # PENDING|INVESTIGATING|DECIDED|ESCALATED|BUDGET_EXCEEDED|FAILED
    hypotheses: list[Hypothesis]       # 无 reducer（覆盖写，见 2.5）
    evidence: Annotated[list[Evidence], merge_evidence]      # 自定义去重合并 reducer
    investigation_queue: list[InvestigationItem]             # 无 reducer（覆盖写）
    tool_call_history: Annotated[list[ToolCallRecord], add]  # append 合并
    pending_tool_calls: list[PlannedToolCall]                # 无 reducer（覆盖写）〔细化新增〕
    budget: Budget                     # 无 reducer（整体覆盖写）
    decision: ReviewDecision | None    # 无 reducer（终局写入）
    degraded: bool                     # 〔细化新增〕上一 LLM 步校验失败降级标记
    failures: Annotated[list[StepFailure], add]              # 〔细化新增〕失败审计
```

> 〔细化新增〕带此标记的 4 个字段（`pending_tool_calls / degraded / failures`）在《00》§3 的状态 JSON 示例里没有，是让第 3~7 章契约可落地的**图内部通道**：`pending_tool_calls` 承载 plan→tools 的计划传递；`degraded` 承载 LLM 校验失败的降级信号；`failures` 供 decide overlay 与审计引用。语义与《00》§3 的"显式状态 / 过程审计"一致，非业务矛盾。是否允许保留，见待定项 T-9。

### 2.2 字段级契约表

类型列给出 Pydantic 模型名（域模型定义见 `src/cg/domain/`）；「写入方」列只列**唯一合法写入方**；「读方」列列出会读取该字段的节点/函数。

| # | 字段 | 类型 | 含义 | 写入方 | 读方 |
|---|---|---|---|---|---|
| 1 | `run_id` | `str` | Agent 运行唯一 id（如 `RUN_CASE_20240907_001_01`），一次消费一次运行 | 入口 worker（写一次后只读） | 全部节点（审计/落库用） |
| 2 | `case_id` | `str` | 案件 id，与 `case.case_id` 一致 | 入口 worker | 全部节点 |
| 3 | `case` | `ProductReviewCase` | 商品事实快照：`case_id / product{title,description,category,brand,attributes,sku_list,images,listing_time,version} / merchant_id / event_type / screening_signals[]`（《00》§2.1 原样）。**只读快照**，Agent 不修改 | 入口 worker（由 MQ 负载构造） | hypothesize / plan / reevaluate / decide / 硬规则 |
| 4 | `status` | `AgentRunStatus` | 运行状态枚举：`PENDING / INVESTIGATING / DECIDED / ESCALATED / BUDGET_EXCEEDED / FAILED`（《00》§3 原样） | 入口置 `INVESTIGATING`；decide 节点收尾置终态 | decide / worker / ops |
| 5 | `hypotheses` | `list[Hypothesis]` | **核心推理状态**：假设集合，每条含 `id/statement/prior/posterior/status/evidence_for/evidence_against` | hypothesize（初始化）；reevaluate（更新 posterior/status/证据链接、追加新假设） | plan / reevaluate / decide / is_converged |
| 6 | `evidence` | `list[Evidence]` | **结论依据**：已收集、去重、合并的证据链（与 tool_call_history 分离，《00》§3.1.4） | tools_node（Tool 结果→Evidence 转换器） | reevaluate / decide / overlay / is_converged |
| 7 | `investigation_queue` | `list[InvestigationItem]` | 待验证问题（含已解决项），`{q, priority, status}`，供 plan 排序取用 | hypothesize（初始化）；reevaluate（标记 DONE/新增） | plan / reevaluate |
| 8 | `tool_call_history` | `list[ToolCallRecord]` | **过程审计**：每次工具调用的完整记录 `{seq, tool, args, result_ref, latency_ms, tokens, status}` | tools_node（append，seq 自增）；dedup guardrail 被跳过的调用也记录 `status=skipped` | decide / 审计 / eval 重放 |
| 9 | `pending_tool_calls` | `list[PlannedToolCall]` | 本轮 plan 决定要执行、尚未消费的工具调用 `{tool, args}`（按 priority 排序） | plan（写入）；tools_node（消费后置 `[]`） | route_after_plan / tools_node / dedup |
| 10 | `budget` | `Budget` | `{llm_calls, tool_calls, tokens, start_time, limits{max_llm_calls:8, max_tool_calls:12, max_tokens:40000, max_latency_ms:30000}}`（《00》§3 / §8.1 原样） | 入口初始化（`start_time=now`）；各 LLM 节点壳记账 `llm_calls/tokens`；tools_node 记账 `tool_calls` | budget_check / route_* / decide overlay / prompt 注入剩余预算 |
| 11 | `decision` | `ReviewDecision | None` | 终局裁决，收敛后一次写入（《00》§2.2 形状，含 `hypothesis_trace / budget_used`） | decide（overlay 之后） | worker / 下游落库 |
| 12 | `degraded` | `bool` | 最近一次 LLM 步是否因 schema 校验失败（重试 1 次后仍失败）而降级。True 时后续 LLM 节点**不再调用 LLM**，透传至 decide（见 3.1/6.2） | 各 LLM 节点：失败置 True；成功置 False（默认 False） | 各节点入口 / route_after_* / decide |
| 13 | `failures` | `list[StepFailure]` | 步骤失败审计 `{step_type, reason, ts}`，一次一追加，供 decide overlay（T-8 涉及）与人工/运维追溯 | 各 LLM 节点 / tools_node（工具执行失败） | decide overlay / 审计 |

**只读约定**：入口写入后，`run_id / case_id / case / budget.start_time / budget.limits` 一律只读；任何节点不得修改 `case`（商品事实是历史快照）。

### 2.3 初始化（入口 worker 负责，非任何节点）

```
AgentState = {
  run_id, case_id, case: <MQ 负载构造的 ProductReviewCase>,
  status: "INVESTIGATING",
  hypotheses: [], evidence: [], investigation_queue: [],
  tool_call_history: [], pending_tool_calls: [],
  budget: {llm_calls:0, tool_calls:0, tokens:0, start_time: now,
           limits: <来自配置，默认 8/12/40000/30000>},
  decision: None, degraded: False, failures: []
}
```

### 2.4 Hypothesis / Evidence / 队列等子模型字段契约

```python
class Hypothesis(BaseModel):                 # 《00》§3 示例原字段
    id: str                                  # 形如 "H1".."H4"，同一 run 内唯一
    statement: str                           # 一句话假设（可解释性核心）
    prior: float                             # 初始先验 [0,1]，由 hypothesize 给出（T-1）
    posterior: float = 0.0                   # 当前后验 [0,1]，reevaluate 更新
    status: HypothesisStatus                 # 见下方枚举（T-3）
    evidence_for: list[str] = []             # 支持证据的 evidence_key 列表（不是全文）
    evidence_against: list[str] = []         # 反驳证据的 evidence_key 列表

class Evidence(BaseModel):
    evidence_id: str                         # 如 "E_01"；同 run 唯一（也对应《00》§9.1 evidence 表）
    type: EvidenceType                       # 受控枚举，见 5.x 每工具的映射表
    source_tool: str                         # 来源工具名（ProductTool 等）
    value: str                               # 人类可读证据值，如 "similarity=0.91, match=某品牌经典鞋款"
    weight: float = 0.5                      # 证据强度/可信度 [0,1]，默认按工具类型表（5.7），T-5
    ref_id: str | None = None                # 可追溯引用：policy clause_id / case_id（RAG 必填）
    extra: dict = {}                         # 结构化附加（相似度数值、Top-K、bbox 等），便于确定性函数读取

class InvestigationItem(BaseModel):          # 《00》§3 原字段
    q: str                                   # 待验证问题，如 "商品外观是否对应某品牌?"
    priority: int                            # 越小越优先（1..5）
    status: Literal["OPEN", "DONE"]          # OPEN=待查；DONE=已由证据回答（T-3 相关）

class PlannedToolCall(BaseModel):            # 〔细化新增〕plan 的“下一步动作”
    tool: ToolName                           # 6 工具受控名
    args: dict[str, Any]                     # 与工具 args Schema 对齐（第 5 章）
    reason: str                              # plan 给的选取理由（进审计）
    priority: int                            # 1..5，1 最高（第 4 章）

class ToolCallRecord(BaseModel):             # 《00》§3 原字段 + status
    seq: int                                 # 自增，= 上一 seq + 1
    tool: str
    args: dict[str, Any]                     # 实际入参（含默认值回填后的规范形）
    result_ref: str | None = None            # 命中的 Evidence.evidence_id（若产出）
    latency_ms: int = 0
    tokens: int = 0                          # 本次工具调用估算 token（可选，默认 0）
    status: Literal["ok", "error", "skipped"]  # skipped=被 dedup/预算截断

class StepFailure(BaseModel):                # 〔细化新增〕
    step_type: Literal["HYPOTHESIZE","PLAN","TOOL_CALL","REEVALUATE","DECIDE"]
    reason: str
    ts: str                                  # ISO8601
```

**假设状态枚举**（《00》只出现 SUPPORTED/REFUTED 两个终态示例，本文补齐生命周期，见 T-3）：

```
PENDING   初始待验证（hypothesize 建立）
SUPPORTED 证据链支持（reevaluate 置位）
REFUTED   证据链反驳（reevaluate 置位）
UNRESOLVED 证据不足、未能证实也未证伪（reevaluate 置位，→ 导向 HUMAN_REVIEW）
```

**高优先级假设判定**（PASS/收敛判定的基础，默认 `prior >= 0.3`，见 T-1）：`high_priority(h) = h.prior >= HIGH_PRIOR_THRESHOLD`，阈值放 `guardrails/` 配置常量。

### 2.5 reducer 语义（哪些字段走合并）

| 字段 | reducer | 语义与理由 |
|---|---|---|
| `evidence` | **自定义 `merge_evidence`** | 按 `evidence_key = (type, source_tool, ref_id)` 去重合并：已存在则**丢弃重复项**（不覆盖——证据一旦收集不可篡改）；新增项 append。实现：`sorted(list, key)` 或 dict 归并均可，保证可序列化、幂等（图重放不产生重复证据，对齐《00》§8.2.5 幂等语义的 agent 内版本） |
| `tool_call_history` | `operator.add`（append） | 只追加不合并，审计日志语义 |
| `failures` | `operator.add`（append） | 只追加 |
| `hypotheses` | **无 reducer（覆盖写）** | 单条执行路径上每个时点只有一个合法写入方（hypothesize 或 reevaluate），它返回**计算后的全集**即可；用覆盖写避免合并歧义。注意：写入方必须返回完整假设列表（含未被本次更新的假设），否则丢假设 |
| `investigation_queue` | 无 reducer（覆盖写） | 同上，写入方返回完整队列 |
| `budget` | 无 reducer（覆盖写） | 每次只有一个节点记账，返回整对象 |
| `decision / status / degraded / pending_tool_calls / case / run_id / case_id` | 无 reducer（覆盖写） | 单写方字段 |

> 并发提示：本图是**单路径线性链**，不存在两个节点同轮写同一字段，因此除 3 个 append/merge 字段外都用覆盖写，最简单且无歧义。若未来引入并行子调查（当前明确不做，见《00》§14.3 Multi-Agent），再为 `evidence` 设计更细的并发合并。

### 2.6 是否需要 MessagesState：**不需要**（决策 + 理由）

- 《00》§3 明确定义"状态必须是显式、可序列化、可持久化、可恢复的，而不是藏在 LLM 的上下文里"。
- 因此四个 LLM 节点**每次调用都重新构造 prompt**（把 state 相关字段序列化成上下文注入，见 3.2 上下文组装器），**不累积** message 历史：可序列化、可审计、token 可预算、重放确定。
- 由此 `MessagesState`（message 累积 + `add_messages` reducer）与本设计冲突，明确不使用。LLM 调用级对话（含"重试 1 次"时的修正信息）只存在于**单次节点调用内部**，随 agent_step trace 落库（《00》§9.2 `agent_step`），不进 AgentState。
- 若未来需要让 LLM 引用上轮完整输出，走结构化字段（如 `pending_tool_calls.reason`、`hypotheses.evidence_for`），而不是把 message 堆进 state。

---

## 3. 四个 LLM 节点契约

### 3.0 公共调用壳（四个节点共用的"结构化输出 + 校验 + 记账"壳）

先定义壳，四个节点只填"职责 / 读哪些字段 / prompt 注入什么 / 输出模型 / 降级结果"，避免重复。

```python
# guardrails/llm_shell.py 契约（伪代码）
async def call_structured_llm(*, prompt_builder, OutputModel, state) -> OutputModel | None:
    """返回解析校验通过的模型；重试 1 次仍失败返回 None（不抛异常、不无限重试）。"""
    if budget_exceeded(state["budget"]):
        return None                       # 预算不足不再发起调用（壳层守卫）
    for attempt in (1, 2):                # 首次 + 重试 1 次
        messages = build_messages(state)  # 见 3.2 上下文组装器（截断规则内建）
        resp = await llm.chat(messages, response_format=OutputModel.json_schema())  # litellm
        usage = resp.usage                # prompt+completion tokens
        try:
            return OutputModel.model_validate_json(resp.text)   # Pydantic 强校验
        except ValidationError as e:
            if attempt == 1:
                messages.append(修正提示: "输出不满足 schema，错误如下，请按 JSON Schema 重新输出：" + str(e))
                continue
            return None                   # 第 2 次失败：降级，不再重试
    # 记账由外层节点壳完成：budget.llm_calls += 1; budget.tokens += usage.total_tokens
```

```python
# 节点壳模板（四节点共用；每节点实现 build_prompt / OutputModel / apply）
async def llm_node(state):
    if state["degraded"] or budget_exceeded(state["budget"]):
        return await degraded_short_circuit(state)   # 3.1 降级短路
    out = await call_structured_llm(...)
    if out is None:
        state_updates = degraded_result(state)       # 每节点各自定义，见 3.x
        state_updates.update(degraded=True)
        state_updates["failures"] = [StepFailure(step_type=..., reason="schema 校验重试仍失败")]
        return state_updates
    updates = apply_out(state, out)                  # 每节点各自定义
    updates["degraded"] = False
    updates["budget"] = bump_llm_usage(state["budget"])   # llm_calls+1、tokens 累加
    return updates
```

**校验失败策略（四个节点统一，实现"标记失败 + 按证据不足降级路由，不无限重试"）**：

| 层 | 行为 |
|---|---|
| Pydantic schema 校验失败 | 自动重试 **1 次**（把校验错误原文回喂给 LLM 修正输出） |
| 重试仍失败 | 不再重试：节点返回降级结果并置 `degraded=True` + 追加 `failures`；**后续 LLM 节点不再调用 LLM**（`degraded_short_circuit` 只做必要透传），直到路由进入 decide |
| 降级的业务语义 | 统一视为**"证据不足"**：哪怕之前已收集部分证据，未完成语义综合的链不自动放行 → 路由最终落 decide，由确定性 overlay 产出 `HUMAN_REVIEW`（硬规则命中除外，见第 7 章） |
| 降级短路具体行为 | `plan/reevaluate`：返回空更新 `{}`（不动任何推理字段，degraded 保持 True）→ 各自条件边读 `degraded=True` 直接路由 decide；`decide`：跳过 LLM 提案（见 7.3 的预算/降级分支） |
| 无限重试防护 | 重试上限 1；全局还有 budget（llm_calls≤8）兜底 |

### 3.1 hypothesize 节点（能力 1：Risk Hypothesis Generation）

**图位置**：`set_entry_point("hypothesize")`，静态边 `hypothesize → plan`。**执行次数**：默认仅在入口执行 1 次（初始化假设与调查队列），此后由 reevaluate 承担"更新假设"职责（《00》§4.2 图无回到 hypothesize 的边；假设在运行中是否可重跑，见待定项 T-6）。

**职责一句话**：读商品事实 + 机审信号，建立"待验证的风险假设集"（含至少一条低风险/正常假设）与"初始调查问题队列"。

**输入（读 state 字段）**：`case`（product 字段、merchant_id、event_type）、`case.screening_signals`；不读 evidence（此时为空）。

**prompt 注入上下文**（`build_messages` 组装，均序列化为 JSON）：
1. 商品事实：`product_id / title / description / category / brand / attributes / sku_list 摘要 / images 数量与 source / listing_time / version`；
2. `screening_signals` 完整列表（`name/result/score`，避免 Agent 重复机审已做的事，《00》§2.3.2）；
3. 系统指令：说明这是"假设生成"而非"终判"；要求包含一条"低风险/无违规"假设（PASS 前提需要，见 7.4）；prior 含义 = 未经调查的先验怀疑度（T-1）；每条假设一句话可验证；每轮调查只验证最值得的那一两条（T-2 关联）。

**输出 Pydantic 模型（字段级契约）**：

```python
class HypothesizeOutput(BaseModel):
    hypotheses: list[HypothesisProposal]     # 上限 MAX_HYPOTHESES（默认 5，见 T-2）
    investigation_queue: list[QueueProposal] # 上限 MAX_QUEUE（默认 8）
    rationale: str                           # 一句话说明假设来源（进审计）

class HypothesisProposal(BaseModel):
    statement: str                           # 假设内容，必须一句话可验证
    prior: float = Field(ge=0, le=1)         # 先验（T-1：谁给/约束）
    evidence_hint: list[str] = []            # 想用什么证据验证（供 plan 参考，可选）

class QueueProposal(BaseModel):
    q: str
    priority: int = Field(ge=1, le=5)        # 1 最优先
```

**写入 state**：`hypotheses`（初始化全集，`status=PENDING`、`posterior=0.0`、`evidence_for/against=[]`）；`investigation_queue`（全部 `OPEN`）；`degraded=False`；`budget`（LLM 记账）。

**降级结果（重试后仍失败）**：返回 `hypotheses=[]`、`investigation_queue=[]`、`degraded=True`。下游：plan 短路 → decide → overlay：`failures` 非空且无硬规则 → `HUMAN_REVIEW`（原因：假设生成失败、证据不足），不会因"没有假设"而被误判 PASS（见 7.4 PASS 前置校验）。

**契约约束**：hypotheses 至少 1 条（否则视为异常输出计入校验失败）；id 由 run 内序号 `H1..Hn` 生成，LLM 不写 id（避免冲突）。

### 3.2 plan 节点（能力 2+3：Investigation Planning + Dynamic Tool Selection）

**图位置**：`hypothesize → plan`、条件边 `plan → {tools, decide}`。**每轮循环都可能执行**（经 `reevaluate → plan` 回来）。

**职责一句话**：读当前假设/证据/调查队列，决定"下一步验证哪个假设、调哪个工具、给什么参数"——**工具选择契约见第 4 章**，本节只给节点的输入输出与降级。

**输入（读 state 字段）**：`hypotheses`、`evidence`、`investigation_queue`、`budget`（剩余额度）、`case`（摘要）。

**prompt 注入上下文**：
1. 全部假设的"仪表盘"：`id / statement / prior / posterior / status / 证据数`（证据全文不注入，控制 token）；
2. 已收集证据摘要（按 recency 取最近 ≤20 条：`type/source_tool/value 截断 120 字/ref_id`）；
3. `investigation_queue` 中 `OPEN` 项（按 priority 排序）；
4. 剩余预算（`budget.llm_calls/tool_calls/tokens` vs limits）——提示它克制；
5. **6 个工具的 JSON Schema（name + description + args schema）**，来自 ToolRegistry（`tool.json_schema()`），让 LLM 产出合法 args；
6. 系统指令：只计划"能带来新证据"的工具调用；**不要重复**已经执行过且成功的 (tool, args)（dedup 有确定性兜底，见 4.3，但 prompt 先约束）；工具调不到 → 输出 `conclude`，别硬编。

**输出 Pydantic 模型（字段级契约）**：

```python
class PlanOutput(BaseModel):
    next_action: Literal["call_tools", "conclude"]   # 见 4.1 语义
    tools: list[PlannedToolCall] = []                # 完整定义见 2.4；≤3 条/轮（T-2）
    rationale: str                                   # 一句解释"验证哪个假设→要哪条证据"
```

**写入 state**：`pending_tool_calls`（写入 tools，经 4.3 dedup guardrail 清洗后回填）；`degraded=False`；`budget`。

**降级结果**：`pending_tool_calls=[]` + `degraded=True` → 条件边 `route_after_plan` 读 degraded 路由 decide（证据不足）。

**与"收敛"的关系**：plan 判断"再查无益"时输出 `next_action=conclude`（`tools=[]`），`route_after_plan` 据此路由 decide（见 6.2）；这是循环终止的正常途径之一（另两个：converged、budget 兜底）。

### 3.3 reevaluate 节点（能力 6：Evidence Synthesis）

**图位置**：静态边 `tools → reevaluate`；条件边 `reevaluate → {continue: plan, decide: decide}`。

**职责一句话**：把本批新证据综合进假设（更新 posterior/status、标注支持/反驳证据、关闭/新增调查队列项），并给出"是否可收敛"的语义判定**供确定性路由参考**（路由权威是确定性谓词，见 6.3，LLM 只给数据不给路由）。

**输入（读 state 字段）**：`hypotheses`、`evidence`（重点：自上次 reevaluate 后新增的证据，用 evidence_id 集合差集找出）、`investigation_queue`、`case`（摘要）、上一轮 `pending_tool_calls`（本批查了什么）。

**prompt 注入上下文**：
1. 假设仪表盘（同上）；
2. **本批新增证据**（全文，≤10 条，逐条带 evidence_id/type/value/weight/ref_id）；
3. 历史证据仅给摘要（避免重复推理全量）；
4. 系统指令：
   - 只依据已给证据更新，不许臆造证据（证据 id 必须存在于注入列表，否则校验失败）；
   - `posterior` 更新需给出可解释依据（在 `rationale` 里点名证据 id）；
   - 证据不足无法判定的假设置 `UNRESOLVED`（不要硬给 SUPPORTED/REFUTED）；
   - 证据相互矛盾的（如"相似度高但商家历史干净"）必须在 `conflicts` 里显式标出（T-4 关联）。

**输出 Pydantic 模型（字段级契约）**：

```python
class ReevaluateOutput(BaseModel):
    hypothesis_updates: list[HypothesisUpdate]    # 只列被更新的假设，未列出的保持原状
    queue_updates: list[QueueUpdate] = []         # 队列项状态变更
    new_hypotheses: list[HypothesisProposal] = [] # 运行中新发现的风险维度（可选，默认空）
    evidence_sufficiency: Literal["SUFFICIENT", "INSUFFICIENT"]  # 语义参考量，非路由权威
    conflicts: list[ConflictNote] = []            # 矛盾证据对（进 HUMAN_REVIEW 判定）
    rationale: str

class HypothesisUpdate(BaseModel):
    id: str                                      # 必须命中已有假设 id
    posterior: float = Field(ge=0, le=1)
    status: HypothesisStatus                     # SUPPORTED/REFUTED/UNRESOLVED
    evidence_for: list[str] = []                 # 引用的 evidence_id，必须真实存在
    evidence_against: list[str] = []

class QueueUpdate(BaseModel):
    q: str                                       # 命中已有问题原文
    status: Literal["OPEN", "DONE"]

class ConflictNote(BaseModel):
    between: list[str]                           # 两个 evidence_id
    description: str
```

**写入 state**：`hypotheses`（对 HypothesisUpdate 逐条 apply 后返回**全集**；`new_hypotheses` 以 `PENDING` 追加，同样返回全集）；`investigation_queue`（apply 后返回全集）；`degraded=False`；`budget`。

**降级结果**：不做任何假设更新（保持现状）+ `degraded=True` → `route_after_reevaluate` 读 degraded 路由 decide（本批证据未能综合 → 证据不足 → HUMAN_REVIEW）。

### 3.4 decide 节点（能力 7：Uncertainty / Abstention）

**图位置**：终结点，无出边；可被 `route_after_plan` / `route_after_reevaluate` 两处路由进入。

**职责一句话**：LLM 先产出一份决策**提案**，随后**确定性 overlay**（第 7 章）按《00》§7.2 规则校验/改写，产出最终 `ReviewDecision`；在预算耗尽/降级等情形下跳过 LLM、直接由 overlay 给出 `HUMAN_REVIEW`（§8 语义：超限→部分证据转人工）。

**输入（读 state 字段）**：`hypotheses`（含 posterior/status/证据链接）、`evidence`（含 weight/ref_id）、`case`、`budget`、`failures`、`degraded`；`investigation_queue` 供参考。

**prompt 注入上下文**：
1. 假设仪表盘（同上，含 SUPPORTED 假设的证据全文）；
2. **可引用依据候选**：evidence 中 `type ∈ {POLICY_REF, CASE_PRECEDENT}` 的项全文（policy 条款原文/案例裁决摘要），这是 REJECT 的依据来源（《00》§7.2.2）；
3. 矛盾证据说明（reevaluate 的 conflicts + 确定性矛盾检测结果，见 7.3）；
4. 三分类语义（《00》§7.1 表格原文）与风险类型受控词表（《00》§7.3，4 个枚举原样）；
5. 剩余/已用预算；
6. 系统指令：决策必须能引用 evidence（`evidence_ids` 必须存在）；`policy` 只能填证据中真实出现的 policy_id/条款；不确定就 `HUMAN_REVIEW`（这是"克制地转人工"，不是失败）。

**输出 Pydantic 模型（字段级契约）——LLM 只产"提案"**：

```python
class DecisionProposal(BaseModel):
    decision: Literal["PASS", "REJECT", "HUMAN_REVIEW"]
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]       # 默认映射建议 LOW/MEDIUM/HIGH（T-10）
    risk_type: list[RiskType] = []                     # 受控词表 4 项，可为空
    confidence: float = Field(ge=0, le=1)              # 提案置信度（7.5 会被 overlay 校准）
    evidence_ids: list[str] = []                       # 支撑证据，必须存在
    policy: list[str] = []                             # policy_id / clause 引用，必须存在于证据
    rationale: str

RiskType = Literal["POTENTIAL_IP_RISK", "EVASION_PATTERN", "FALSE_CLAIM", "FIELD_CONFLICT"]  # 《00》§7.3
```

**写入 state**：`decision`（overlay 后的最终 ReviewDecision，字段见第 7 章 7.6）；`status`（终态，取值规则见 T-8）；`degraded`（消费后置 False）；`failures`（若本次因预算/降级没跑 LLM，也可记 note）。

**降级结果（decide 自身 LLM 失败）**：无提案 → overlay 以空提案执行：无硬规则 → `HUMAN_REVIEW`（原因：决策推理失败）。

**节点内部两步结构（LLM → overlay 不可跳过）**：

```
decide_node(state):
  1. if budget_exceeded or degraded: proposal = None        # 7.3 预算/降级分支
     else: proposal = call_structured_llm(DecisionProposal) # 仍失败 → None
  2. final = run_decision_overlay(state, proposal)          # 第 7 章确定性 overlay（永远执行，含硬规则）
  3. return {decision: final, status: 终态, degraded: False, budget: 记账(如调过 LLM)}
```

---

## 4. plan 节点的工具选择契约（能力 3：Dynamic Tool Selection）

### 4.1 输出语义：`next_action` 与 `tools`

`PlanOutput` 即"本轮的调查动作建议"，字段级契约见 3.2。两个枚举的精确语义：

| `next_action` | 条件 | 路由去向（6.2） | 说明 |
|---|---|---|---|
| `call_tools` | 有值得验证的假设缺口，且存在可提供新证据的工具 | `tools` | `tools` 必须非空；每条含 tool/args/reason/priority |
| `conclude` | 再查无益：要么收敛、要么所有工具都查过且无新证据、要么只剩主观判定 | `decide` | `tools` 必须为空（不一致输出→校验失败） |

**`tools[].priority` 语义**（字段级）：
- `1..5` 整数，1 最高；由 plan 依据"该调用对验证当前最高优先假设的边际价值"排序；
- **同一轮 tools 节点访问内按 priority 升序依次执行**（对齐《00》§4.4：第 2 轮同轮执行 ProductTool + MerchantTool）；
- 预算只够执行前 k 个时：执行前 k 个，其余本轮不执行、**不丢弃**——`pending_tool_calls` 本轮置空（tools 节点一次性消费），下一轮 plan 会基于"上轮部分执行、证据仍缺"重新决策（或改选更省预算的工具）。被预算截断的调用**不记** `tool_call_history`（`skipped` 状态只用于 dedup 截断场景，见 4.3），部分执行的情况在 plan 的 rationale 与 budget 记账上自然可见。

**每轮建议条数上限**：`≤3`（默认，T-2）；超出按 priority 截断（提示 LLM 一次只做最值得的 1~2 件事）。

### 4.2 与《00》§4.5"动态选工具"的对应

- **动态性的载体**：plan（LLM，语义判断"当前最值得验证的假设需要哪条证据"）+ 确定性 ToolNode（执行）+ 条件边（是否真的去调）。不是固定 `A→B→C` 流水线：**默认不调任何工具**，只有当 plan 判定存在证据缺口才产生 `call_tools`。
- **每轮工具集合由当轮 state 决定**：同样商品，若 OCR 已显示"100% Polyester"而标题写"真丝"，plan 应优先 ProductTool 做字段交叉而非 ImageAnalysis（《00》§4.5 例）。
- **评测接口**：`PlanOutput` 与执行记录（`agent_step(PLAN)` 的 output_json + `tool_call_history`）就是《00》§11.3 `Tool Selection Accuracy` 的取数来源——预期工具集合来自 eval_case 标签，实际来自 plan 输出。
- **prompt 内工具 Schema 的来源**：ToolRegistry 为每个工具维护 `name/description/args_json_schema`（5.8），plan 的 prompt 注入这些 Schema，保证 `args` 字段合法；合法性的最终裁决在 ToolNode 的确定性 args 校验（5.8），不信任 LLM。

### 4.3 确定性 dedup guardrail（`guardrails/dedup.py`）

plan 可能（重试后、上下文不清时）重复建议已执行的调用。**不靠 prompt 承诺，靠确定性兜底**，在 plan 写入 `pending_tool_calls` 时清洗：

```
dedup_pending(state, planned) -> list[PlannedToolCall]:
    规范化：args 做 canonical 序列化（key 排序 + 类型规整）
    对每条 planned：
        若 tool_call_history 中存在同 (tool, canonical_args) 且 status == "ok"
            → 丢弃，并 append 一条 tool_call_history(status="skipped", reason="duplicate")
        若曾执行但 status == "error"
            → 保留（允许重试 1 次）
        否则保留
    return 清洗后列表
```

清洗后若 `pending_tool_calls` 变空（全部是重复）且 plan 原输出是 `call_tools` → 视为 `conclude`：路由直接进 decide，防止"无新证据死循环"。循环终止三保险：① dedup 截断；② plan 的 conclude 语义；③ budget 兜底（llm_calls/tool_calls 上限）。

---

## 5. 6 个 Tool 的输入 / 输出 Schema（字段级契约）

### 5.0 通用约定

**统一接口**（《00》§5.2 原样，本处细化字段）：

```python
class Tool(Protocol):
    name: ToolName                 # 6 个受控名之一
    description: str               # 给 LLM 的一句话说明（进 plan prompt）
    async def call(self, args: dict, ctx: ToolContext) -> ToolResult: ...
```

- `ToolResult`：`{ok: bool, data: dict | None, error: str | None}`。**确定性错误**（业务无结果）返回 `ok=False + error`，不抛异常；基础设施级瞬态错误（超时/连接）允许 infra 层重试 1 次。
- 每个工具实现**结果→Evidence 转换器**（`to_evidence(result) -> list[Evidence]`），由 ToolNode 在工具返回后调用并入 `state.evidence`（《00》§4.3：转换与去重合并是确定性 Python，不进 LLM）。
- Evidence 的 `source_tool` = 工具名；`evidence_id` 由 ToolNode 按 `E_<nn>` 顺序分配。
- **脱敏**（《00》§8.2.3）：工具返回给 LLM 前（即进 state/进 prompt 前）由 ToolNode 统一过 PII 脱敏过滤器（商家联系方式等字段打码）；脱敏逻辑单测覆盖。
- args 校验失败（LLM 给错参）：工具不执行，记 `tool_call_history{status:"error", error=args 校验错误}`；当轮继续执行其余合法调用；不单独为坏 args 重试 LLM（证据缺口会在下一轮 plan 自然暴露）。
- 本表「→ Evidence」列给出转换规则与 `weight` 默认值（T-5 校准）。

### 5.1 ProductTool —— 事实锚点（确定性 · MySQL 读，《00》§5）

| 项 | 内容 |
|---|---|
| name | `ProductTool` |
| description（给 LLM） | "读取商品在库最新事实快照（标题/描述/属性/品牌/SKU/图片/版本），用于确认 brand 真空缺、字段冲突、版本漂移" |
| args Schema | `{ "product_id": str required, "version": int optional(默认取库中最新) }` |
| result data | `{ product_id, merchant_id, title, description, category, brand: str|null, attributes: dict[str,str], sku_list: [{sku_id,color,size,price}], images: [{url, source}], version: int, listing_time: str, status: str }` |
| 确定性说明 | 读 MySQL `product/product_sku/product_image`；返回库中最新 version；`images` 不含 ocr_text（OCR 归 OCRTool），避免重复劳动 |
| → Evidence | 每商品 1 条：`Evidence{type="PRODUCT_FACT", source_tool="ProductTool", value="brand=null, 标题/描述无品牌词, version=3（与 case 快照一致）", weight=0.6, extra={...关键字段}}`。**版本比对**：库中 version ≠ `case.product.version` 时在 extra 标注 `version_drift=true`（提示决策时案件基于旧快照） |
| 回答的业务问题 | 判断"规避品牌"前先确认 brand 是否真空缺（《00》§5.1） |

### 5.2 ImageAnalysisTool —— 多模态核心（混合：图片向量检索 + 视觉 LLM）

| 项 | 内容 |
|---|---|
| name | `ImageAnalysisTool` |
| description | "分析商品图片外观是否与知名品牌款/违禁视觉高度相似；返回相似度 Top-K、Logo 检测、视觉风险描述" |
| args Schema | `{ "image_urls": [str] required(1..5), "top_k": int optional(默认 5, 1..10), "detect_logo": bool optional(默认 true) }` |
| result data | `{ items: [ { image_url, top_similar: [{brand_ref: str, similarity: float 0..1}], logos: [{brand: str, confidence: float}], visual_risk: str, } ] }`（top_similar 按相似度降序；`brand_ref` 指向图片品牌向量库条目，保留引用） |
| 混合实现说明 | 图片向量库召回（《00》§6.4 图片向量单独存）粗召回 Top-K → 视觉 LLM 复核输出结构化结果 |
| → Evidence | 对**每个品牌命中**（similarity ≥ EVIDENCE_MIN_SIM，默认 0.60，T-11）产 1 条：`type="IMAGE_SIMILARITY"`，`value="similarity=0.91, match=某品牌经典鞋款"`，`weight=similarity 数值`，`extra={image_url, similarity}`；Logo 命中每条产 `type="IMAGE_LOGO"`，`value="logo=某品牌, conf=0.93"`，`weight=confidence`。无命中（全部低于阈值）产 0 条证据，不污染证据链（"没查到"与"证明无"的区分见 7.4） |
| 回答的业务问题 | 外观相似是本案最大、规则无法覆盖的证据缺口（《00》§5.1） |

### 5.3 OCRTool —— 交叉验证（确定性 OCR 服务）

| 项 | 内容 |
|---|---|
| name | `OCRTool` |
| description | "识别图片中的文字内容（含坐标与置信度），用于标题/描述与图片实际内容的交叉验证" |
| args Schema | `{ "image": str required(图片 url 或 base64 data-url) }` |
| result data | `{ full_text: str, blocks: [ { text, bbox:{x,y,w,h}, lang, confidence } ] }` |
| → Evidence | 1 条聚合：`type="OCR_TEXT"`，`value=full_text（截断 ≤500 字，完整进 extra.blocks）`，`weight=0.5`；若识别到与商品描述冲突的关键词（如 "Polyester" vs "真丝"），`extra.conflict_hint=true`（字段冲突判定归 reevaluate/确定性检测，T-12） |
| 回答的业务问题 | 交叉验证：发现"标题/描述"与"图片实际内容"冲突（《00》§5.1） |

### 5.4 MerchantTool —— 行为模式（确定性：聚合 + 向量扫描）

| 项 | 内容 |
|---|---|
| name | `MerchantTool` |
| description | "查询商家的系统性行为画像：在架商品数、相似商品数、历史违规/下架/改标题重上架次数、信用分" |
| args Schema | `{ "merchant_id": str required, "window_days": int optional(默认 90, 1..365) }` |
| result data | `{ merchant_id, product_total, similar_product_count, removals, title_relisting_count, violations: {total, by_type: dict}, credit_score, recent_events: [ {event_type, ts} 最多 20 条 ] }`（《00》§5 输出列："商品总数、相似商品数、历史违规/下架/改标题重上架数、信用分"） |
| → Evidence | 1 条聚合：`type="MERCHANT_HISTORY"`，`value="23 similar / 5 removals / 3 title-relisting, credit=62"`，`weight=0.85`（多信号聚合型证据默认高权重，T-5），`extra={...上述字段}`；`extra.signals` 供确定性"规避行为"判定（违规+下架+改标题重上架组合） |
| 回答的业务问题 | 单商品看不出问题，商家历史行为才是"规避"的关键信号（《00》§5.1） |

### 5.5 CaseSearchTool —— 先例（RAG · Case KB）

| 项 | 内容 |
|---|---|
| name | `CaseSearchTool` |
| description | "检索历史人工裁决的相似案件（先例），返回 Top-K 相似案例及其决策/风险类型/关键证据/适用政策" |
| args Schema | `{ "query": str required(自然语言或结构化描述，如 '无品牌标识+外观高度模仿+商家多次重上架'), "filters": { "category": str optional, "risk_type": [RiskType] optional }, "top_k": int optional(默认 5, 1..10) }` |
| result data | `{ hits: [ { case_id, similarity, decision: PASS/REJECT/HUMAN_REVIEW, risk_level, risk_type: [..], summary, key_evidence: [str], policy_refs: [policy_id], } ] }`（检索：BM25+向量融合、元数据过滤、粗召回 Top-50 → rerank Top-K，《00》§6.3；`case_id` 是回案库引用主键，脱敏文本只含摘要） |
| → Evidence | **每个 hit** 1 条：`type="CASE_PRECEDENT"`，`value="CASE_1832 高度相似 → REJECT（POTENTIAL_IP_RISK）"`，`weight=similarity`，**`ref_id=case_id`（必填，可追溯，《00》§6.3 引用格式）**，`extra={decision, risk_type, policy_refs}` |
| 回答的业务问题 | 同类案件人怎么判的，提供决策参照（REJECT 的可引用依据之一，《00》§7.2.2） |

### 5.6 PolicySearchTool —— 政策依据（RAG · Policy KB）

| 项 | 内容 |
|---|---|
| name | `PolicySearchTool` |
| description | "检索当前有效平台政策条款（按类目/风险类型过滤），返回条款原文与版本引用" |
| args Schema | `{ "query": str required, "filters": { "category": str optional, "risk_type": [RiskType] optional }, "top_k": int optional(默认 5, 1..10), "effective_only": bool optional(默认 true, 只查生效版本) }` |
| result data | `{ hits: [ { policy_id, version, clause_id, title, text, category, risk_type: [..], status, effective_date } ] }`（元数据过滤 + 版本有效性过滤，《00》§6.3） |
| → Evidence | 每个 hit 1 条：`type="POLICY_REF"`，`value="POLICY_3.2 v2 条款：外观高度模仿品牌设计且无授权 → 高风险转人工"`，`weight=0.9`，**`ref_id=clause_id`（必填）**，`extra={policy_id, version, effective_date}` |
| 回答的业务问题 | 当前规则下到底能不能判、判到什么程度（《00》§5.1） |

### 5.7 Evidence 受控类型汇总（供 5.x 映射表与 §7 overlay 使用）

```
EvidenceType = PRODUCT_FACT | IMAGE_SIMILARITY | IMAGE_LOGO | OCR_TEXT
             | MERCHANT_HISTORY | CASE_PRECEDENT | POLICY_REF
可引用依据类型（REJECT/HUMAN 判定的 citable 集合）: CASE_PRECEDENT | POLICY_REF
```

### 5.8 ToolRegistry 与 ToolNode 契约（确定性）

```python
class ToolRegistry:
    register(tool); get(name) -> Tool
    list_descriptions() -> [{name, description, args_json_schema}]   # 供 plan prompt
    validate_args(tool_name, args) -> None | str(错误信息)            # JSON Schema 校验

async def tools_node(state):
    """执行 pending_tool_calls：按 priority 升序，逐个：预算→校验→脱敏→执行→转证据→记账。"""
    updates = {pending_tool_calls: [], budget: copy(state.budget)}
    for call in sorted(state["pending_tool_calls"], key=lambda c: c.priority):
        if updates.budget.tool_calls >= limits.max_tool_calls: break    # 预算截断，剩余不执行
        if err := registry.validate_args(call.tool, call.args):         # args 校验
            record(seq, status="error", error=err); continue
        result = await tool.call(desensitize(call.args), ctx)           # 瞬态错误 infra 重试 1 次
        updates.budget.tool_calls += 1
        if not result.ok: record(seq, status="error", error=result.error); continue
        evidences = tool.to_evidence(result)                             # → Evidence（5.x 表）
        updates.evidence = merge_evidence(state.evidence + evidences)    # 走 reducer 语义
        record(seq, status="ok", result_ref=evidences[0].evidence_id if evidences else None)
    return updates
```

---

## 6. 条件边路由契约（确定性）

### 6.1 图接线（与《00》§4.2 草图逐字一致）

```python
graph.set_entry_point("hypothesize")
graph.add_edge("hypothesize", "plan")
graph.add_conditional_edges("plan", route_after_plan,
                            {"tools": "tools", "decide": "decide"})
graph.add_edge("tools", "reevaluate")
graph.add_conditional_edges("reevaluate", route_after_reevaluate,
                            {"continue": "plan", "decide": "decide"})
app = graph.compile(checkpointer=mysql_checkpointer)   # state.py + checkpointer.py
```

LangGraph 映射要点（实现者注意）：
- `route_*` 函数签名 `(state) -> str`，返回 path_map 的 key；**LangGraph 在每个节点执行完、应用其 state 更新后**，用"最新提交后的 state"调用路由函数——所以路由能读到 plan 刚写入的 `pending_tool_calls`、reevaluate 刚更新的 `hypotheses`；
- 路由函数必须是**纯函数、确定性、可单测**（不许调 LLM、不许有随机）；全部谓词下沉到 `guardrails/`（`budget.py / converge.py`），`graph.py` 里只留一行分发；
- 路由返回的 key 与节点名的映射即上面两个 dict，key 命名沿用《00》§4.2（`tools/decide`、`continue/decide`）。

### 6.2 `route_after_plan` 伪代码

```python
def route_after_plan(state) -> Literal["tools", "decide"]:
    # ① 降级短路：plan 校验失败 → 证据不足 → 直接 decide（第 3 章）
    if state["degraded"]:
        return "decide"
    # ② 预算护栏：进入 tools 前先查预算（《00》§8.1：路由处、进节点前）
    if budget_exceeded(state["budget"]):            # 见 6.4
        return "decide"                             # decide 内 overlay → HUMAN_REVIEW(预算耗尽)
    # ③ 有工具要调才进 tools
    if state["pending_tool_calls"]:                 # 已过 dedup（4.3），非空即真实动作
        return "tools"
    # ④ plan 输出 conclude / 空计划 → 不再调查
    return "decide"
```

### 6.3 `route_after_reevaluate` 伪代码

```python
def route_after_reevaluate(state) -> Literal["continue", "decide"]:
    if state["degraded"]:                    # reevaluate 综合失败 → 证据不足 → decide
        return "decide"
    if budget_exceeded(state["budget"]):     # 预算护栏
        return "decide"
    if is_converged(state):                  # 确定性收敛判定（下方）
        return "decide"
    return "continue"                        # 证据不足且预算够 → 回 plan 再查

# —— 收敛判定（guardrails/converge.py；纯确定性，可单测）——
HIGH_PRIOR_THRESHOLD = 0.3                    # T-1（待定项默认值）
CITABLE_TYPES = {"CASE_PRECEDENT", "POLICY_REF"}

def is_converged(state) -> bool:
    hs = state["hypotheses"]
    high = [h for h in hs if h.prior >= HIGH_PRIOR_THRESHOLD]
    if not high:                              # 无高优先假设（如假设生成失败场景，另有 degraded）
        return False                          # 保守：不收敛 → 走 decide 也会 HUMAN_REVIEW
    open_hp  = [h for h in high if h.status in {"PENDING", "UNRESOLVED"}]
    # 关键：SUPPORTED 假设若无"可引用依据"证据 → 不算收敛，继续查 Case/Policy（对齐《00》§7.2.2）
    uncited  = [h for h in high if h.status == "SUPPORTED"
                and not any(e.type in CITABLE_TYPES for e in citing_evidence(state, h))]
    return not open_hp and not uncited
```

> 语义对齐说明：这条谓词让"证据链可支撑自动判"（假设都定了 + SUPPORTED 的可引用）才路由 decide；否则回 plan 继续取证。《00》§4.4 推演里第 3 轮查 CaseSearch/PolicySearch，正是因为 H2/H3/H4 SUPPORTED 但还没有可引用依据（详见第 8 章逐轮验证）。`citing_evidence(state, h)` 从 `h.evidence_for` 的 evidence_id 取回 Evidence。`is_converged` 的精确形态本身是设计拍板项（T-4）。

### 6.4 `budget_check`（`guardrails/budget.py`，对齐《00》§8.1）

```python
def budget_exceeded(budget) -> Literal["LLM_CALLS", "TOOL_CALLS", "TOKENS", "LATENCY"] | None:
    # 返回首个超限维度（调用方据此记审计）；全部未超限返回 None
    if budget.llm_calls >= budget.limits.max_llm_calls:    return "LLM_CALLS"
    if budget.tool_calls >= budget.limits.max_tool_calls:  return "TOOL_CALLS"
    if budget.tokens   >= budget.limits.max_tokens:        return "TOKENS"
    elapsed_ms = now_ms() - budget.start_time_ms
    if elapsed_ms >= budget.limits.max_latency_ms:         return "LATENCY"
    return None
```

| 维度 | v1 阈值（《00》§8.1 原值） | 超限后的路由行为 |
|---|---|---|
| max_llm_calls | 8 | 路由 decide → overlay 产出 `HUMAN_REVIEW`（带已收集的部分证据） |
| max_tool_calls | 12 | 同上（tools_node 内部也按此截断单批执行） |
| max_tokens | 40000 | 同上；另：llm_shell 在发起调用前若估算超出也直接放弃该调用（壳层守卫） |
| max_latency_ms | 30000 | 同上（wall-clock 从 `budget.start_time` 算） |

超限语义（《00》§8.1 原句，写进注释防误读）：**"调查成本已超过可接受范围，证据不足以自动判，转人工最稳妥"——这是正确的业务行为，不是失败。**

### 6.5 循环终止性论证（写在文档里给评审看）

循环只可能存在于 `plan → tools → reevaluate → (continue) plan`；每个终止途径都是确定性的：① plan 输出 `conclude` → `route_after_plan` → decide；② dedup 把重复动作清空 → 同上；③ `is_converged()` 为真 → decide；④ 预算维度任一超限 → decide。LLM 语义判断（plan/reevaluate）只影响收敛早晚，不影响"最终一定到达 decide"这一性质（budget 是硬后盾）。

---

## 7. decide 节点的确定性 overlay（对齐《00》§7.2）

### 7.0 结构：LLM 提案 + 确定性校验，二层不可合并

decide 节点 = **先**跑 LLM 产出 `DecisionProposal`（第 3.4 章）**后**跑 `run_decision_overlay`。overlay 是普通确定性 Python（`guardrails/decision_guardrail.py`），规则顺序固定，全部可单测。

### 7.1 overlay 输入

- `state`：`hypotheses / evidence / case / budget / failures / degraded / investigation_queue`；
- `proposal: DecisionProposal | None`（None = 预算耗尽 / 降级 / decide 自身 LLM 失败时的空提案）。

### 7.2 overlay 伪代码（规则优先级自上而下，命中即短路 + 记录 override）

```python
def run_decision_overlay(state, proposal) -> ReviewDecision:
    evidence   = state["evidence"]
    failures   = state["failures"]
    budget     = state["budget"]
    hard_hit   = hard_rule_hit(state)                # R1 硬规则（blacklist/硬违禁），guardrails/hard_rules.py

    # ---- R1 硬规则优先：不可被 LLM 覆盖（《00》§7.2.1）----
    if hard_hit:
        return build_decision("REJECT", risk_level="HIGH",
                              risk_type=hard_hit.risk_types,           # 硬规则给出风险类型
                              confidence=1.0, evidence=evidence,
                              overrides=["R1_HARD_RULE"])              # 防漏放，覆盖一切（含 PASS 提案）

    proposal = proposal or DecisionProposal(decision="HUMAN_REVIEW",
                                            risk_level="MEDIUM", confidence=0.0, ...)  # 空提案兜底
    decision, reasons = proposal.decision, []

    # ---- R2 REJECT 必须有可引用依据（《00》§7.2.2）----
    if decision == "REJECT":
        citable = [e for e in evidence if e.type in CITABLE_TYPES and e.ref_id]
        if not citable:
            decision = "HUMAN_REVIEW"; reasons.append("R2_REJECT_NO_CITABLE_BASIS")

    # ---- R3 HUMAN_REVIEW 触发条件（《00》§7.2.3，任一命中即转人工）----
    if budget_exceeded(budget):
        decision = "HUMAN_REVIEW"; reasons.append("R3_BUDGET_EXHAUSTED")     # 预算耗尽
    if proposal.confidence < 0.7:
        decision = "HUMAN_REVIEW"; reasons.append("R3_LOW_CONFIDENCE")       # 阈值默认 0.7（T-4）
    if contradiction_detect(state):                                            # 证据矛盾（如高相似 vs 商家历史干净）
        decision = "HUMAN_REVIEW"; reasons.append("R3_CONTRADICTORY_EVIDENCE")
    if not citable_for_reject(proposal) and decision == "REJECT":            # 政策模糊/无先例 → 不许自动 REJECT
        decision = "HUMAN_REVIEW"; reasons.append("R3_NOVEL_RISK_NO_PRECEDENT")

    # ---- R4 PASS 前置条件：所有高优先级假设必须被证伪（《00》§7.2.4）----
    if decision == "PASS":
        not_refuted = [h for h in high_priority(state) if h.status != "REFUTED"]
        if not_refuted:
            decision = "HUMAN_REVIEW"; reasons.append("R4_PASS_WITHOUT_REFUTATION")  # 区分"证明无风险"和"没查到风险"

    # ---- R5 降级/失败兜底（第 3 章 degrade 语义）----
    if state["degraded"] or failures:
        decision = "HUMAN_REVIEW"; reasons.append("R5_DEGRADED_OR_FAILED_STEP")

    return build_decision(decision=decision,
                          risk_level=finalize_risk_level(proposal, decision),   # 见 7.5
                          risk_type=finalize_risk_type(proposal, decision),
                          confidence=finalize_confidence(proposal, decision),    # 见 7.5，T-4
                          evidence=evidence,
                          policy=citable_policy_ids(evidence),                   # 从 POLICY_REF 收集
                          hypothesis_trace=trace_from(state["hypotheses"]),      # id/statement/prior/posterior/status
                          budget_used=snapshot_budget(budget),
                          overrides=reasons)
```

### 7.3 预算/降级分支（decide 入口，7.0 的第一步）

```
若 budget_exceeded(budget) 或 state["degraded"] 或 failures 非空（且本轮尚未消费）：
    → proposal = None（不调用 LLM，不再烧 token）
否则正常调 LLM（proposal）
两种情况都必须进入 run_decision_overlay —— 因为 R1 硬规则可能把结果改成 REJECT。
```

### 7.4 `hard_rule_hit`（`guardrails/hard_rules.py`，R1 数据来源）

纯确定性扫描以下来源，命中即强制 REJECT：
- `case.product.brand` / `case.product.title/description` 命中品牌黑名单（词表来自规则引擎复用）；
- `case.screening_signals` 中带"硬违禁"语义且 result 非 PASS 的信号（正常情况分流层不会把硬违规投进来，但调查中新证据可能触发）；
- 调查新证据：`IMAGE_LOGO`（识别到黑名单品牌 Logo）、`OCR_TEXT` 命中禁词等——用 `Evidence.extra` 结构化字段判定，不用 LLM。

### 7.5 置信度 / risk_level 的确定性校准（T-4：公式待设计方拍板，先给默认）

- **risk_level**：proposal 给 LOW/MEDIUM/HIGH；overlay 不做语义重判，仅当 overlay 把决策改成 HUMAN_REVIEW 且原为 HIGH 时保持 HIGH，否则沿用 proposal（默认映射 LOW/MEDIUM/HIGH 与 risk_type 的对应见 T-10）。
- **confidence 最终值**（写进 decision 的）：按《00》§7.4"confidence = f(最高假设 posterior, 证据链完整性, 是否存在可引用依据, 证据是否矛盾)"的**确定性可解释默认公式**（待 T-4 定稿）：

```python
def finalize_confidence(proposal, decision) -> float:
    top = max((h.posterior for h in hypotheses if h.status == "SUPPORTED"), default=0.0)
    completeness = len(evidence) / MAX_EXPECTED_EVIDENCE       # 默认 8，见 T-4
    has_citation = 1.0 if any(e.type in CITABLE_TYPES and e.ref_id for e in evidence) else 0.0
    contradiction = 0.0 if not contradiction_detect(state) else -0.2   # 矛盾扣分
    c = 0.45*top + 0.25*min(completeness, 1.0) + 0.20*has_citation + 0.10 + contradiction
    return round(min(max(c, 0.0), 1.0), 2)
```

> 注意：overlay 用 LLM 提案的 confidence 触发 R3_LOW_CONFIDENCE（<0.7 转人工），但**落库的 confidence 用上面的确定性公式重算**——保证"confidence 不是 LLM 拍脑袋"（《00》§7.4 原话）且可解释。

### 7.6 最终 ReviewDecision 形状（对齐《00》§2.2，落库前不变形）

```json
{
  "decision": "HUMAN_REVIEW",
  "risk_level": "HIGH",
  "risk_type": ["POTENTIAL_IP_RISK", "EVASION_PATTERN"],
  "confidence": 0.91,
  "evidence": [ { "evidence_id": "E_01", "type": "IMAGE_SIMILARITY",
                  "source_tool": "ImageAnalysisTool",
                  "value": "similarity=0.91, match=某品牌经典鞋款", "weight": 0.91,
                  "ref_id": null } ],
  "policy": ["POLICY_3.2"],
  "hypothesis_trace": [ { "id": "H3", "statement": "刻意规避品牌识别",
                          "prior": 0.2, "posterior": 0.88, "status": "SUPPORTED" } ],
  "budget_used": { "llm_calls": 8, "tool_calls": 5, "tokens": 18000, "latency_ms": 9200 },
  "overrides": ["R3_LOW_CONFIDENCE"]          // 〔细化新增，可选〕overlay 改判记录，进 trace
}
```

> `overrides` 是〔细化新增〕字段：只有 overlay 实际改写了 LLM 提案（或走了预算/降级分支）才非空，保证"谁（哪条规则）把 PASS 改成了 HUMAN_REVIEW"可审计（《00》§8.2.4 决策审计）。是否接受字段扩展见 T-8。

---

## 8. 与《00》§4.4 核心场景（复古运动鞋）的逐轮对齐

本节点验证第 2~7 章契约能走通主链路。列：设计轮次（《00》§4.4 表的行）→ 图执行序列 → 关键 state 变化 → 走的边与判定依据。

前置事实（《00》§4.4 / §2.1）：机审 4 项 signal 全 PASS、`brand=null`、标题/描述无品牌词、图片待分析、商家 M_5512。

| 设计轮次 | 图执行（节点序列） | 产出 / state 变化 | 走的边（含谓词依据） |
|---|---|---|---|
| 0 | `hypothesize`（入口唯一一次） | 建立 H1（普通复古设计，prior 0.5）/ H2（参考知名品牌经典设计，prior 0.4）/ H3（刻意规避品牌识别，prior 0.2）/ H4（商家系统性类似行为，prior 0.15），均 PENDING；队列置 2 个 OPEN 问题 | 静态边 hypothesize→plan |
| 1 | `plan` | 读仪表盘：H1/H2 高优先（prior≥0.3）且 PENDING；判定"外观是否对应某品牌"最值得查 → `next_action=call_tools, tools=[{ImageAnalysisTool, args:{image_urls:[img1]}, priority:1}]` | 静态边 plan→tools（经 route_after_plan ③：pending 非空、预算 OK、未 degraded） |
| 1 | `tools`（ToolNode） | 执行 ImageAnalysisTool → similarity=0.91 命中品牌款 → **E_01**（IMAGE_SIMILARITY, weight 0.91）；budget.tool_calls=1 | 静态边 tools→reevaluate |
| 1 | `reevaluate` | 综合 E_01：H1 REFUTED（证据 against），H2 SUPPORTED（posterior ↑）；H3/H4 仍 PENDING | route_after_reevaluate：未 degraded、预算 OK、`is_converged=false`（H3/H4 高优先 OPEN）→ **continue → plan** |
| 2 | `plan` | "需确认商品字段真空缺 + 商家历史" → `call_tools, tools=[{ProductTool, priority:1}, {MerchantTool, priority:2}]` | plan→tools（同 ③） |
| 2 | `tools` | ProductTool → brand=null、标题/描述无品牌（**E_02** PRODUCT_FACT）；MerchantTool → 23 similar / 5 removals / 3 title-relisting（**E_03** MERCHANT_HISTORY）；tool_calls=3 | tools→reevaluate |
| 2 | `reevaluate` | E_02/E_03 支持：H3 SUPPORTED（posterior 0.88）、H4 SUPPORTED（posterior 0.85）；队列两问 DONE | route_after_reevaluate：`is_converged=false` —— 高优先假设虽无 OPEN，但 H2/H3/H4 SUPPORTED 且**没有任何 CASE_PRECEDENT/POLICY_REF 可引用依据**（uncited 非空，对齐《00》§7.2.2）→ **continue → plan** |
| 3 | `plan` | "需要先例 + 政策支撑才能判" → `call_tools, tools=[{CaseSearchTool, priority:1}, {PolicySearchTool, priority:2}]` | plan→tools |
| 3 | `tools` | CaseSearchTool → CASE_1832 高度相似 → REJECT（**E_04** CASE_PRECEDENT, ref_id=CASE_1832）；PolicySearchTool → POLICY_3.2"外观高度模仿高风险转人工"（**E_05** POLICY_REF, ref_id=clause）；tool_calls=5 | tools→reevaluate |
| 3 | `reevaluate` | 证据链补全：所有高优先假设已终态，SUPPORTED 假设均有可引用依据；输出 evidence_sufficiency=SUFFICIENT | route_after_reevaluate：`is_converged=true`（open_hp 空、uncited 空）→ **decide** |
| 4 | `decide` | LLM 提案：`HUMAN_REVIEW / HIGH / [POTENTIAL_IP_RISK, EVASION_PATTERN] / confidence 0.91 / evidence E_01..E_05 / policy [POLICY_3.2]`；overlay：R1 硬规则未命中（无黑名单）→ R2 REJECT 依据检查（提案非 REJECT，跳过）→ R3 各触发条件（confidence 0.91≥0.7、无矛盾、预算未耗尽）均不触发 → R4 PASS 前置（非 PASS，跳过）→ 通过，原样落库；`status → DECIDED`，`decision.budget_used={llm_calls:8, tool_calls:5, tokens:..., latency_ms:...}` | 终结点（无出边） |

**"为什么第 4 步是 HUMAN_REVIEW 而不是 REJECT"的契约解释**：证据链已充分（假设全部 SUPPORTED + 可引用依据齐备），但 REJECT 属"仿冒主观判定"且 POLICY_3.2 指引是"高风险转人工"——LLM 提案与《00》§4.4 原结果一致，overlay 无改判；这体现 Agent"知道什么时候该人介入"（《00》§4.4 注意行）。若未来把该案改为可自动判，只动 policy 指引与提案，路由/overlay 结构不变。

**预算核查（对齐 6.4）**：llm_calls=8（hypothesize+plan×3+reevaluate×3+decide）恰抵上限 8、tool_calls=5≤12、tokens/latency 未超 → 主链路在预算内走通。8/8 余量为 0 说明阈值紧贴示例走查（T-7 需要设计方确认是否调阈值）。

---

## 9. 开放问题清单（待设计方拍板；本文只给建议默认值，不作最终决定）

> 每条给出：问题 / 影响面 / 建议默认值。标 🔒 的表示实现可以先按默认值开发，但**定稿前请设计方确认**（因为它们影响 schema 或评测标签，改动会波及 02-evaluation.md）。

| ID | 待定项 | 出处/影响 | 建议默认值 |
|---|---|---|---|
| T-1 | **prior 初始值谁给**（LLM 输出 vs 确定性规则） | hypotheses、is_converged 的高优先阈值 | hypothesize 的 LLM 输出 prior（0..1），overlay 只做范围钳制；`HIGH_PRIOR_THRESHOLD=0.3`；LLM 需在 prompt 里解释 prior 依据 |
| T-2 | **假设数量 / 队列 / 每轮工具数上限** | schema、prompt、成本 | `MAX_HYPOTHESES=5`（走查用 4）、`MAX_QUEUE=8`、每轮 tools≤3 🔒 |
| T-3 | **HypothesisStatus 词表**（《00》只示例 SUPPORTED/REFUTED） | hypotheses 状态机、收敛判定、评测标签 | `PENDING/SUPPORTED/REFUTED/UNRESOLVED`，初始 PENDING 🔒 |
| T-4 | **confidence 精确公式与"证据矛盾/收敛"的确定性定义** | decide overlay、R3/R4、is_converged | 本文 7.5 默认公式（0.45·top_posterior + 0.25·完整性 + 0.20·可引用 + 0.10 − 0.2·矛盾）；`MAX_EXPECTED_EVIDENCE=8`；矛盾检测 v1 只做一条启发式：IMAGE_SIMILARITY(≥0.85) 与 MERCHANT_HISTORY(0 removal & 0 violation) 并存 🔒 |
| T-5 | **Evidence.weight 来源**（转换器默认 vs reevaluate 校准） | evidence 表、评测 | v1：转换器默认权重表（5.1~5.6 各 tool 列），reevaluate 不改 weight（只改 posterior/status）；weight 语义=证据强度供审计/展示 |
| T-6 | **hypothesize 是否可重跑**（图无回到 hypothesize 的边；《00》能力1 写"初始/更新假设"） | 图结构 | 默认只入口跑 1 次；运行中新增假设走 `reevaluate.new_hypotheses`；若评审认为需重生成假设，加 `decide → hypothesize` 条件边再议 |
| T-7 | **预算阈值与走查余量**（§8 走查 llm_calls=8/8 抵上限） | §8.1、示例 | 保留《00》8/12/40000/30000 原值；评估建议按走查实测把 max_llm_calls 提到 10（给 decide 修正留余量）或确认 8 即可 |
| T-8 | **运行终态语义**（DECIDED / ESCALATED / BUDGET_EXCEEDED 何时用）+ decision.overrides 扩展 | status 枚举、落库、审计 | 默认：decide 完成后一律 `DECIDED`（含 HUMAN_REVIEW）；BUDGET_EXCEEDED 仅作 trace 标记不进状态机；ESCALATED 由人工工作台侧置位；`decision.overrides` 作为可选字段保留（不进 core schema 时至少进 agent_step 落库）🔒 |
| T-9 | **4 个〔细化新增〕state 字段是否保留**（pending_tool_calls/degraded/failures + decision.overrides） | state schema、checkpointer 持久化列 | 保留（图内通道，实现必需）；正式 agent_state_json 持久化时与核心字段同库即可 |
| T-10 | **risk_level LOW/MEDIUM/HIGH 的词表与到 risk_type 的默认映射** | decision schema、评测 | 词表 LOW/MEDIUM/HIGH；默认映射：POTENTIAL_IP_RISK/EVASION_PATTERN 建议 HIGH 起步，FALSE_CLAIM/FIELD_CONFLICT 视证据 MEDIUM；仅做展示不做路由依据 🔒 |
| T-11 | **ImageAnalysis 相似度阈值**（多少算"命中品牌款"证据） | EVIDENCE_MIN_SIM、矛盾检测、评测集构造 | `EVIDENCE_MIN_SIM=0.60`（产生证据）、矛盾检测用 ≥0.85；两处阈值都做成配置常量 |
| T-12 | **字段冲突（FIELD_CONFLICT）检测归属**（OCR vs 商品字段的交叉验证放哪层） | 《00》§14.2 支撑场景（二期） | 二期：guardrails 确定性检测器（不占 LLM）；v1 只在 OCRTool extra 里打 conflict_hint，不展开 |

**使用约定**：T-1~T-5 是"决策机制/评测标签"级，必须先于 02-evaluation.md 定稿；T-6~T-12 可并行，实现按默认值推进，变更只影响本文第 2/3/7 章对应小节。

---

> 附：实现顺序建议（与《00》末尾"下一步"呼应）：① `domain/` 各 Pydantic 模型 + `agent/state.py`（第 2 章）→ ② `guardrails/`（budget/dedup/hard_rules/decision_guardrail，第 6/7 章，全部先写单测）→ ③ 4 个 LLM 节点壳 + mock LLM（第 3 章）→ ④ 6 个 Tool 空实现 + ToolRegistry + ToolNode（第 5 章）→ ⑤ graph.py 接线跑通复古运动鞋单链路（第 8 章验证）→ ⑥ Checkpointer 接 MySQL。
