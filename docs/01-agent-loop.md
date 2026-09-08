# 复杂风险调查 Agent —— StateGraph 节点 / 边 / 状态契约细化（01-agent-loop v2）

> 本文档是《00-system-design.md》在 **Agent Loop（实现层）** 上的细化与契约化，服务对象是实现者（写 `src/pra/domain`、`src/pra/agent/**`、`src/pra/tools/**` 的人）。
> 引用约定：凡提到总设计原文均写作 **《00》§x.y**（如《00》§4.2），避免与本文编号混淆。
>
> **v2 修订（review 复核，与 03-decisions v2 / 代码同步）**：§2 AgentState 与落地 `state.py`/`domain/models.py` 对齐（run_id/case_id→thread、status→DB、posterior=None、Evidence.source/extra、dict 元素）；
> §6 收敛谓词改为证据级可引用（不过滤 prior）；§7 decide overlay 升级为 **Decision Gate**（decision_confidence 与 risk 分离、PASS/REJECT Gate、abstention 清单）；
> §2.4/§5.8 增 tool_call_history 边际增益 4 字段（Marginal Evidence Gain 取数）；§6.4 预算按 T-7 为 10/15 Guardrail；§5.2/§5.7 相似度三档 0.70/0.85（EVIDENCE_MIN_SIM/EVIDENCE_STRONG）；§9 改为已拍板索引。
>
> **对齐承诺**：本文档不推翻《00》任何决策；与《00》冲突时以《00》为准，冲突与拍板记录见第 9 章"待定项状态索引"（T-1~T-12 已全部拍板，权威值 docs/03-decisions.md）与 03-decisions.md。
>
> **本文档不包含业务代码**：只写 TypedDict/Pydantic 伪代码、JSON Schema、确定性路由/Guardrail 伪代码；这些是"契约"，不是 `src/pra/` 下可直接运行的实现。

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
  - `src/pra/domain/`：Pydantic 领域模型（Case / Evidence / Hypothesis / Budget / ReviewDecision / 各类 LLM 输出模型）。
  - `src/pra/agent/state.py`：`AgentState` TypedDict + reducer 声明（第 2 章）。
  - `src/pra/agent/graph.py`：节点注册、静态边、`add_conditional_edges`、编译（第 6 章）。
  - `src/pra/agent/nodes/`：hypothesize / plan / reevaluate / decide 四节点（第 3、4、7 章）。
  - `src/pra/agent/tools_node.py`：ToolNode（执行、args 校验、结果→Evidence、预算记账）（第 4、5 章）。
  - `src/pra/agent/guardrails/`：`budget.py`（budget_check）、`hard_rules.py`、`decision_guardrail.py`、`dedup.py`（第 6、7 章确定性代码）。
  - `src/pra/agent/checkpointer.py`：MySQL Checkpointer 接入。
  - `src/pra/tools/<six>/`：6 个工具子包 + `ToolRegistry` + 统一 `Tool` 接口（第 5 章）。

### 1.3 三条贯穿性原则（实现时不可违背）

1. **状态显式化**（《00》§3）：跨步骤信息全部走 `AgentState` 结构化字段，不把推理过程藏在 LLM message 历史里 → **不使用 `MessagesState`**（理由见 2.6）。
2. **分工**（《00》§4.3）：LLM 只做节点内部语义推理（结构化 JSON 输出）；**图怎么走（条件边）、预算、硬规则、去重、决策校验全部是确定性 Python**，不允许 LLM 决定路由。
3. **超限即止损**（《00》§8.1）：Budget 检查发生在**条件边路由函数内、进入下一个节点之前**；超限 → 路由到 decide，由 decide 的确定性 overlay 产出带部分证据的 `HUMAN_REVIEW`（这是正确业务行为，不是失败）。

---

## 2. AgentState 契约

### 2.1 总述

`AgentState` 声明为 `TypedDict`（LangGraph channel 声明用），**字段值**用 `src/pra/domain/` 下的 Pydantic 模型承载（校验、序列化、JSON Schema 生成）。全部字段必须可 JSON 序列化（Checkpointer 每步落库，《00》§3.1）。

各节点只**返回自己负责的字段**（partial update），LangGraph 按各字段的 reducer 语义合并进状态。

```python
# src/pra/agent/state.py 契约（已与落地代码对齐；伪代码示意 reducer）
from typing import TypedDict, Annotated, Literal, NotRequired
from operator import add

class AgentState(TypedDict):
    case: ProductReviewCase            # 输入事实快照（只读，调查起点）
    hypotheses: list[Hypothesis]       # 无 reducer（覆盖写，见 2.5）
    evidence: Annotated[list[Evidence], merge_evidence]      # 自定义去重合并 reducer
    investigation_queue: list[dict]    # 待验证问题 {q, priority, status}；无 reducer（覆盖写）
    tool_call_history: Annotated[list[dict], add]            # 调用审计（含边际增益 4 字段，见 2.4）
    pending_tool_calls: list[dict]     # plan→tools 计划传递 {tool,args,reason,priority}；无 reducer（覆盖写）
    budget: Budget                     # 无 reducer（整体覆盖写）
    decision: ReviewDecision | None    # 无 reducer（终局写入）
    degraded: bool                     # 上一 LLM 步校验失败降级标记（覆盖写）
    failures: Annotated[list[dict], add]                     # 步骤失败审计 {step_type, reason, ts}
```

> **与落地代码/拍板的一致性（本版修订）**：`run_id / case_id` **不在 State**——映射为 LangGraph **thread_id**
> （Checkpointer 线程键，调用方携带）；`status` **不在 State**——由 DB `review_run.status` 承载，**DECIDED 是图内唯一终态**
> （PASS/REJECT/HUMAN_REVIEW 是 `ReviewDecision.decision` 取值；预算耗尽/工具失败/降级通过 `decision.overrides` 记录，
> 见第 7 章）。字段清单与 `src/pra/agent/state.py` 一致（10 字段）。〔细化新增〕通道
> `pending_tool_calls / degraded / failures` 为让第 3~7 章契约可落地的图内部通道，语义与《00》§3 的
> "显式状态 / 过程审计"一致（拍板见 03-decisions.md T-9）。

### 2.2 字段级契约表

类型列给出 Pydantic 模型名（域模型定义见 `src/pra/domain/`）；「写入方」列只列**唯一合法写入方**；「读方」列列出会读取该字段的节点/函数。

| # | 字段 | 类型 | 含义 | 写入方 | 读方 |
|---|---|---|---|---|---|
| 1 | `case` | `ProductReviewCase` | 商品事实快照：`case_id / product{title,description,category,brand,attributes,sku_list,images,listing_time,version} / merchant_id / event_type / screening_signals[]`（《00》§2.1 原样）。**只读快照**，Agent 不修改 | 入口 worker（由 MQ 负载构造） | hypothesize / plan / reevaluate / decide / 硬规则 |
| 2 | `hypotheses` | `list[Hypothesis]` | **核心推理状态**：假设集合，每条含 `id/statement/prior/posterior/status/evidence_for/evidence_against` | hypothesize（初始化）；reevaluate（更新 posterior/status/证据链接、追加新假设） | plan / reevaluate / decide / is_converged |
| 3 | `evidence` | `list[Evidence]` | **结论依据**：已收集、去重、合并的证据链（与 tool_call_history 分离，《00》§3.1.4）；`Evidence{type,source,value,weight,ref_id,extra}`（字段与代码对齐，见 2.4） | tools_node（Tool 结果→Evidence 转换器 + 证据质量过滤） | reevaluate / decide / overlay / is_converged |
| 4 | `investigation_queue` | `list[dict]` | 待验证问题（含已解决项），元素 `{q, priority, status}`（dict 形态，与代码/《00》§3 JSON 一致），供 plan 排序取用 | hypothesize（初始化）；reevaluate（标记 DONE/新增） | plan / reevaluate |
| 5 | `tool_call_history` | `list[dict]` | **过程审计**：每次工具调用的完整记录 `{seq, tool, args, result_ref, latency_ms, tokens, status, before_confidence, after_confidence, evidence_added, decision_changed}`（dict 形态，与代码一致；边际增益 4 字段见 2.4/5.8） | tools_node（append，seq 自增）；dedup guardrail 被跳过的调用也记录 `status=skipped` | decide / 审计 / eval 重放 / Marginal Evidence Gain 指标 |
| 6 | `pending_tool_calls` | `list[dict]` | 本轮 plan 决定要执行、尚未消费的工具调用，元素 `{tool, args, reason, priority}`（按 priority 排序） | plan（写入）；tools_node（消费后置 `[]`） | route_after_plan / tools_node / dedup |
| 7 | `budget` | `Budget` | `{llm_calls, tool_calls, tokens, latency_ms, start_time, limits{max_llm_calls:10, max_tool_calls:15, max_tokens:40000, max_latency_ms:30000}}`（拍板 T-7；默认值与 `models.py BudgetLimits` 一致） | 入口初始化（`start_time=now`）；各 LLM 节点壳记账 `llm_calls/tokens`；tools_node 记账 `tool_calls` | budget_check / route_* / decide overlay / prompt 注入剩余预算 |
| 8 | `decision` | `ReviewDecision | None` | 终局裁决，收敛后一次写入（《00》§2.2 形状，含 `hypothesis_trace / budget_used / overrides`） | decide（overlay 之后） | worker / 下游落库 |
| 9 | `degraded` | `bool` | 最近一次 LLM 步是否因 schema 校验失败（重试 1 次后仍失败）而降级。True 时后续 LLM 节点**不再调用 LLM**，透传至 decide（见 3.1/6.2） | 各 LLM 节点：失败置 True；成功置 False（默认 False） | 各节点入口 / route_after_* / decide |
| 10 | `failures` | `list[dict]` | 步骤失败审计 `{step_type, reason, ts}`，一次一追加，供 decide overlay（T-8 涉及）与人工/运维追溯 | 各 LLM 节点 / tools_node（工具执行失败） | decide overlay / 审计 |

**身份/状态字段不在 State（拍板 03 T-8/T-9）**：`run_id / case_id` → LangGraph thread_id（Checkpointer 线程键）；
`status` → DB `review_run.status`（worker 层维护，DECIDED 为图唯一终态）。**只读约定**：入口写入后 `case / budget.start_time / budget.limits` 一律只读；任何节点不得修改 `case`。

### 2.3 初始化（入口 worker 负责，非任何节点）

```
AgentState = {
  case: <MQ 负载构造的 ProductReviewCase>,
  hypotheses: [], evidence: [], investigation_queue: [],
  tool_call_history: [], pending_tool_calls: [],
  budget: {llm_calls:0, tool_calls:0, tokens:0, latency_ms:0, start_time: now,
           limits: <来自配置/代码默认 10/15/40000/30000，运行时可由配置覆盖>},
  decision: None, degraded: False, failures: []
}
# run_id/case_id 由 worker 以 thread_id 携带；review_run.status 由 worker/DB 维护（03 T-8）
```

### 2.4 子模型字段契约（与落地代码 `domain/models.py` / `agent/state.py` 对齐）

```python
# —— Pydantic 领域模型（domain/models.py 已落地）——
class Hypothesis(BaseModel):
    id: str                                  # 形如 "H1".."H4"，同一 run 内唯一
    statement: str                           # 一句话假设（可解释性核心）
    prior: float | None = None               # 先验 [0,1]；hypothesize 显式给出（T-1），未赋值前 None（代码口径）
    posterior: float | None = None           # 后验 [0,1]；reevaluate 更新，未更新前 None（聚合公式对 None 按 0）
    status: HypothesisStatus                 # PENDING/SUPPORTED/REFUTED/UNRESOLVED（T-3，代码已重命名）
    evidence_for: list[str] = []             # 支持本假设的证据引用/摘要字符串
    evidence_against: list[str] = []         # 反驳证据的引用/摘要字符串

class Evidence(BaseModel):                   # 字段与代码一致：无 evidence_id / 无 source_tool
    type: str                                # 开放文本，运行时收敛到 §5.7 受控集（guardrails 常量，不进枚举）
    source: str                              # 来源工具名（如 ImageAnalysisTool）—— 字段名以《00》§2.2/代码为准
    value: str                               # 人类可读证据值，如 "similarity=0.91, match=某品牌经典鞋款"
    weight: float = 0.5                      # 证据强度 [0,1]，默认按工具类型表（5.7），T-5
    ref_id: str | None = None                # 可追溯引用：稳定业务标识优先（image_url/product_id/merchant_id/case_id/clause_id，O-1）；无稳定 ref 时 None（去重 key 回退 value，见 §2.5）
    extra: dict = {}                         # 结构化附加数值（similarity/removals 等），供确定性函数读取（代码已落地）
    # evidence_id 不进 DTO（O-1）：result_ref / evidence_added / evidence_for 一律用引用串 f"{type} {value}"（≤200 字符）；
    # E_<nn> 运行序号只存在于 DB evidence 主键（evidence_id），DTO/审计不用 E_nn

# —— AgentState 中 list 字段的元素为 dict（与代码/《00》§3 JSON 一致）；字段约束如下 ——
# investigation_queue[] 元素:  {"q": str, "priority": int(1..5), "status": "OPEN"|"DONE"}
# pending_tool_calls[] 元素:  {"tool": ToolName, "args": dict, "reason": str, "priority": int(1..5)}
# failures[] 元素:            {"step_type": "HYPOTHESIZE"|"PLAN"|"TOOL_CALL"|"REEVALUATE"|"DECIDE",
#                               "tool": ToolName?,           # O-3：TOOL_CALL 类失败必填，其余缺省
#                               "severity": "warn"|"critical", # O-3：warn=仅审计；critical=LLM 步失败/关键取证失败
#                               "reason": str, "ts": ISO8601}

# tool_call_history[] 元素（每次工具调用一条，含边际增益 4 字段，见 §5.8；G 项拍板）
{
  "seq": 1, "tool": "ImageAnalysisTool", "args": {...},
  "result_ref": "IMAGE_SIMILARITY similarity=0.91, match=某品牌经典鞋款",   # 本次调用新增首条证据的引用串 f"{type} {value}"（O-1）
  "latency_ms": 1200, "tokens": 800,
  "status": "ok" | "error" | "skipped",     # skipped=被 dedup/预算截断
  "before_confidence": 0.42,                # 本次调用前 decision_confidence 代理值（确定性，见 5.8）
  "after_confidence": 0.73,                 # 本次调用后 decision_confidence 代理值
  "evidence_added": ["IMAGE_SIMILARITY similarity=0.91, match=某品牌经典鞋款"],  # 本次调用新增证据的引用串列表（无新增 = []）
  "decision_changed": false                 # 本次调用是否翻转 Gate 探测结果（gate_probe(before)!=gate_probe(after)）
}
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
| `evidence` | **自定义 `merge_evidence`** | 按 `evidence_key` 去重合并（**O-1 已拍板口径**）：`evidence_key = (type, source, ref_id)` —— `ref_id` 优先填**稳定业务标识**（image_url / product_id / merchant_id / case_id / clause_id，能产的工具转换器都填，不再一律 None）；**`ref_id` 为 None 时 key 回退用 value**：`(type, source, value)`（value 人读可辨，防"同 type/source 的多条无 ref 证据"互相吞并，如多图多品牌命中）。同 key 已存在 → **丢弃新增**（证据一旦收集不可篡改）；新 key → append。实现 `sorted(list, key)` 或 dict 归并均可，保证可序列化、幂等（图重放不产生重复证据，对齐《00》§8.2.5） |
| `tool_call_history` | `operator.add`（append） | 只追加不合并，审计日志语义 |
| `failures` | `operator.add`（append） | 只追加 |
| `hypotheses` | **无 reducer（覆盖写）** | 单条执行路径上每个时点只有一个合法写入方（hypothesize 或 reevaluate），它返回**计算后的全集**即可；用覆盖写避免合并歧义。注意：写入方必须返回完整假设列表（含未被本次更新的假设），否则丢假设 |
| `investigation_queue` | 无 reducer（覆盖写） | 同上，写入方返回完整队列 |
| `budget` | 无 reducer（覆盖写） | 每次只有一个节点记账，返回整对象 |
| `decision / degraded / pending_tool_calls / hypotheses / investigation_queue / budget / case` | 无 reducer（覆盖写） | 单写方字段（`run_id/case_id→thread`、`status→DB review_run`，均不在 State） |

> 并发提示：本图是**单路径线性链**，不存在两个节点同轮写同一字段，因此除 3 个 append/merge 字段外都用覆盖写，最简单且无歧义。若未来引入并行子调查（当前明确不做，见《00》§14.3 Multi-Agent），再为 `evidence` 设计更细的并发合并。

### 2.6 是否需要 MessagesState：**不需要**（决策 + 理由）

- 《00》§3 明确定义"状态必须是显式、可序列化、可持久化、可恢复的，而不是藏在 LLM 的上下文里"。
- 因此四个 LLM 节点**每次调用都重新构造 prompt**（把 state 相关字段序列化成上下文注入，见 3.2 上下文组装器），**不累积** message 历史：可序列化、可审计、token 可预算、重放确定。
- 由此 `MessagesState`（message 累积 + `add_messages` reducer）与本设计冲突，明确不使用。LLM 调用级对话（含"重试 1 次"时的修正信息）只存在于**单次节点调用内部**，随 review_trace 落库（《00》§9.2 `review_trace`），不进 AgentState。
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
| 无限重试防护 | 重试上限 1；全局还有 budget（llm_calls ≤ max_llm_calls，默认 10，见 6.4/T-7）兜底 |

### 3.1 hypothesize 节点（能力 1：Risk Hypothesis Generation）

**图位置**：`set_entry_point("hypothesize")`，静态边 `hypothesize → plan`。**执行次数**：默认仅在入口执行 1 次（初始化假设与调查队列），此后由 reevaluate 承担"更新假设"职责（《00》§4.2 图无回到 hypothesize 的边；假设在运行中是否可重跑，见待定项 T-6）。

**职责一句话**：读商品事实 + 机审信号，建立"待验证的风险假设集"（含至少一条低风险/正常假设）与"初始调查问题队列"。

**输入（读 state 字段）**：`case`（product 字段、merchant_id、event_type）、`case.screening_signals`；不读 evidence（此时为空）。

**prompt 注入上下文**（`build_messages` 组装，均序列化为 JSON）：
1. 商品事实：`product_id / title / description / category / brand / attributes / sku_list 摘要 / images 数量与 source / listing_time / version`；
2. `screening_signals` 完整列表（`name/result/score`，避免 Agent 重复机审已做的事，《00》§2.3.2）；
3. 系统指令：说明这是"假设生成"而非"终判"；要求包含一条"低风险/无违规"假设（PASS 前提需要，见 §7.2 PASS Gate）；prior 含义 = 未经调查的先验怀疑度（T-1）；每条假设一句话可验证；每轮调查只验证最值得的那一两条（T-2 关联）。

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

**写入 state**：`hypotheses`（初始化全集，`status=PENDING`、`prior` 用 LLM 输出、`posterior=None`（未评估）、`evidence_for/against=[]`）；`investigation_queue`（全部 `OPEN`）；`degraded=False`；`budget`（LLM 记账）。

**降级结果（重试后仍失败）**：返回 `hypotheses=[]`、`investigation_queue=[]`、`degraded=True`。下游：plan 短路 → decide → overlay：`failures` 非空且无硬规则 → `HUMAN_REVIEW`（原因：假设生成失败、证据不足），不会因"没有假设"而被误判 PASS（见 §7.2 PASS Gate：高优先假设须全部充分证伪）。

**契约约束**：hypotheses 至少 1 条（否则视为异常输出计入校验失败）；id 由 run 内序号 `H1..Hn` 生成，LLM 不写 id（避免冲突）。

### 3.2 plan 节点（能力 2+3：Investigation Planning + Dynamic Tool Selection）

**图位置**：`hypothesize → plan`、条件边 `plan → {tools, decide}`。**每轮循环都可能执行**（经 `reevaluate → plan` 回来）。

**职责一句话**：读当前假设/证据/调查队列，决定"下一步验证哪个假设、调哪个工具、给什么参数"——**工具选择契约见第 4 章**，本节只给节点的输入输出与降级。

**输入（读 state 字段）**：`hypotheses`、`evidence`、`investigation_queue`、`budget`（剩余额度）、`case`（摘要）。

**prompt 注入上下文**：
1. 全部假设的"仪表盘"：`id / statement / prior / posterior / status / 证据数`（证据全文不注入，控制 token）；
2. 已收集证据摘要（按 recency 取最近 ≤20 条：`type/source/value 截断 120 字/ref_id`）；
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

**输入（读 state 字段）**：`hypotheses`、`evidence`（重点：自上次 reevaluate 后新增的证据，用 E_nn 运行序号集合差集找出）、`investigation_queue`、`case`（摘要）、上一轮 `pending_tool_calls`（本批查了什么）。

**prompt 注入上下文**：
1. 假设仪表盘（同上）；
2. **本批新增证据**（全文，≤10 条，逐条带 E_nn 序号/type/value/weight/ref_id）；
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
    evidence_for: list[str] = []                 # 引用的证据 E_nn 序号，必须真实存在
    evidence_against: list[str] = []

class QueueUpdate(BaseModel):
    q: str                                       # 命中已有问题原文
    status: Literal["OPEN", "DONE"]

class ConflictNote(BaseModel):
    between: list[str]                           # 两个冲突证据的 E_nn 序号
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
3. 矛盾证据说明（reevaluate 的 conflicts + 确定性矛盾检测结果，见 §7.2 abstention 清单 R3_CRITICAL_CONFLICT；启发式见 03-decisions T-4(e)）；
4. 三分类语义（《00》§7.1 表格原文）与风险类型受控词表（《00》§7.3，4 个枚举原样）；
5. 剩余/已用预算；
6. 系统指令：决策必须能引用 evidence（`evidence_ids`（E_nn 序号）必须存在）；`policy` 只能填证据中真实出现的 policy_id/条款；不确定就 `HUMAN_REVIEW`（这是"克制地转人工"，不是失败）。

**输出 Pydantic 模型（字段级契约）——LLM 只产"提案"**：

```python
class DecisionProposal(BaseModel):
    decision: Literal["PASS", "REJECT", "HUMAN_REVIEW"]
    risk_level: Literal["NONE", "LOW", "MEDIUM", "HIGH"]   # 词表含 NONE（PASS）；T-10，仅展示/队列排序
    risk_type: list[RiskType] = []                     # 受控词表 4 项，可为空
    confidence: float = Field(ge=0, le=1)              # decision_confidence 提案值（overlay 会重算并作 Gate 输入，见 §7.2/§7.5）
    evidence_ids: list[str] = []                       # 支撑证据（E_nn 序号），必须存在
    policy: list[str] = []                             # policy_id / clause 引用，必须存在于证据
    rationale: str

RiskType = Literal["POTENTIAL_IP_RISK", "EVASION_PATTERN", "FALSE_CLAIM", "FIELD_CONFLICT"]  # 《00》§7.3
```

**写入 state**：`decision`（overlay 后的最终 ReviewDecision，字段见第 7 章 7.6）；`degraded`（消费后置 False）；`failures`（若本次因预算/降级没跑 LLM，也可记 note）。**终态不在 State 写**：worker 在 invoke 返回且 `decision` 非空后置 DB `review_run.status=DECIDED`（PASS/REJECT/HUMAN_REVIEW 都是 decision 取值；03 T-8）。

**降级结果（decide 自身 LLM 失败）**：无提案 → overlay 以空提案执行：无硬规则 → `HUMAN_REVIEW`（原因：决策推理失败）。

**节点内部两步结构（LLM → overlay 不可跳过）**：

```
decide_node(state):
  1. if budget_exceeded or degraded: proposal = None        # 7.3 预算/降级分支
     else: proposal = call_structured_llm(DecisionProposal) # 仍失败 → None
  2. final = run_decision_overlay(state, proposal)          # 第 7 章确定性 overlay（永远执行，含硬规则）
  3. return {decision: final, degraded: False, budget: 记账(如调过 LLM)}   # 终态由 worker 落 DB（DECIDED，03 T-8）
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
- **评测接口**：`PlanOutput` 与执行记录（`review_trace(PLAN)` 的 output_json + `tool_call_history`）就是《00》§11.3 `Tool Selection Accuracy` 的取数来源——预期工具集合来自 eval_case 标签，实际来自 plan 输出。
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
    args_model: type[ToolArgs]     # 本工具入参模型（O-5 拍板，代码已落地 tools/base.py）
    async def call(self, args: ToolArgs, ctx: ToolContext) -> ToolResult: ...
```

- `ToolResult`：`{ok: bool, data: dict | None, error: str | None}`。**确定性错误**（业务无结果）返回 `ok=False + error`，不抛异常；基础设施级瞬态错误（超时/连接）允许 infra 层重试 1 次。
- 每个工具实现**结果→Evidence 转换器**（`to_evidence(result) -> list[Evidence]`），由 ToolNode 在工具返回后调用并入 `state.evidence`（《00》§4.3：转换与去重合并是确定性 Python，不进 LLM）。
- Evidence 的 `source` = 工具名（字段名以代码/《00》§2.2 为准）；`evidence_id` 不进 DTO，由 ToolNode 按 `E_<nn>` 运行序号分配，用于 `result_ref` / `evidence_for` 引用与 DB evidence 主键。
- **证据质量过滤在确定性层**：相似度下限等过滤在 tools_node/guardrails 层执行（工具本身只返回原始结果，与已落地代码一致，见 5.2/5.8），过滤依据 `EVIDENCE_MIN_SIM / EVIDENCE_STRONG`（T-11 拍板，0.70/0.85）。
- **Evidence.extra 回填（O-8 已拍板）**：工具 `to_evidence` 只产原始事实（type/source/value/weight/ref_id），`extra` 派生数值（similarity / version_drift / conflict 等）统一由 tools_node 的 `backfill_extra` 依据工具 raw result 回填（对齐 04 §4/§9）；下方 5.x 各表 "extra={…}" 是回填的目标内容，非工具转换器产出。
- **ref_id 规则（O-1 已拍板）**：能产稳定业务 ref 的转换器一律填 `ref_id`（image_url / product_id / merchant_id；RAG 工具 case_id / clause_id 必填），不再留 None；去重 key 见 §2.5（ref_id=None 时回退 value）。
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
| → Evidence | 每商品 1 条：`Evidence{type="PRODUCT_FACT", source="ProductTool", value="brand=null, version=3（库中最新）, status=ON_SALE", weight=0.6, ref_id=product_id}`（ref_id 规则见 §5.0，O-1）。**版本漂移（O-4/O-8 已拍板）**：库中 version ≠ `case.product.version` 的比对放 tools_node evidence processing 层（其持有 case 快照），由 `backfill_extra` 在 extra 标注 `version_drift=true`（提示决策时案件基于旧快照）；工具转换器只产原始事实 |
| 回答的业务问题 | 判断"规避品牌"前先确认 brand 是否真空缺（《00》§5.1） |

### 5.2 ImageAnalysisTool —— 多模态核心（混合：图片向量检索 + 视觉 LLM）

| 项 | 内容 |
|---|---|
| name | `ImageAnalysisTool` |
| description | "分析商品图片外观是否与知名品牌款/违禁视觉高度相似；返回相似度 Top-K、Logo 检测、视觉风险描述" |
| args Schema | `{ "image_urls": [str] required(1..5), "top_k": int optional(默认 5, 1..10), "detect_logo": bool optional(默认 true) }` |
| result data | `{ items: [ { image_url, top_similar: [{brand_ref: str, similarity: float 0..1}], logos: [{brand: str, confidence: float}], visual_risk: str, } ] }`（top_similar 按相似度降序；`brand_ref` 指向图片品牌向量库条目，保留引用） |
| 混合实现说明 | 图片向量库召回（《00》§6.4 图片向量单独存）粗召回 Top-K → 视觉 LLM 复核输出结构化结果 |
| → Evidence | 转换器（已落地 `image_analysis/tool.py`）对**每个返回命中**（不做阈值过滤）产 1 条原始 Evidence：`type="IMAGE_SIMILARITY"`，`value="similarity=0.91, match=某品牌经典鞋款"`，`weight=similarity 数值`；Logo 命中每条产 `type="IMAGE_LOGO"`，`value="logo=某品牌, conf=0.93"`，`weight=confidence`，两条均 `ref_id=image_url`（O-1：源图片稳定业务标识）。**阈值裁决在确定性层（tools_node/guardrails）**：`similarity < 0.70`（EVIDENCE_MIN_SIM）不入证据链 / `0.70 ≤ similarity < 0.85` 普通证据 / `≥ 0.85`（EVIDENCE_STRONG，即 03 的 SIM_HIGH_CONTRADICT，同值同义）**Strong Evidence** 档（矛盾启发式的"高相似"判据也用它）；入链时确定性层把 `extra.similarity` 等数值补进 Evidence（供矛盾检测机器读取）。无命中产 0 条证据（"没查到"与"证明无"的区分见 §7.2 PASS/REJECT Gate） |
| 回答的业务问题 | 外观相似是本案最大、规则无法覆盖的证据缺口（《00》§5.1） |

### 5.3 OCRTool —— 交叉验证（确定性 OCR 服务）

| 项 | 内容 |
|---|---|
| name | `OCRTool` |
| description | "识别图片中的文字内容（含坐标与置信度），用于标题/描述与图片实际内容的交叉验证" |
| args Schema | `{ "image": str required(图片 url 或 base64 data-url) }` |
| result data | `{ full_text: str, blocks: [ { text, bbox:{x,y,w,h}, lang, confidence } ] }` |
| → Evidence | 1 条聚合：`type="OCR_TEXT"`，`value=full_text（截断 ≤500 字；完整文本与 blocks 留在 result/tool_call_history 审计）`，`weight=0.5`，`ref_id=image`（O-1，调用侧回填）；冲突关键词判定（conflict_hint，T-12）归 reevaluate/确定性检测，v1 由 LLM 证据综合发现，extra 派生字段如需由 tools_node backfill（O-8） |
| 回答的业务问题 | 交叉验证：发现"标题/描述"与"图片实际内容"冲突（《00》§5.1） |

### 5.4 MerchantTool —— 行为模式（确定性：聚合 + 向量扫描）

| 项 | 内容 |
|---|---|
| name | `MerchantTool` |
| description | "查询商家的系统性行为画像：在架商品数、相似商品数、历史违规/下架/改标题重上架次数、信用分" |
| args Schema | `{ "merchant_id": str required, "window_days": int optional(默认 90, 1..365) }` |
| result data | `{ merchant_id, product_total, similar_product_count, removals, title_relisting_count, violations: {total, by_type: dict}, credit_score, recent_events: [ {event_type, ts} 最多 20 条 ] }`（《00》§5 输出列："商品总数、相似商品数、历史违规/下架/改标题重上架数、信用分"） |
| → Evidence | 1 条聚合：`type="MERCHANT_HISTORY"`，`value="23 similar / 5 removals / 3 title-relisting, credit=62"`，`weight=0.85`（多信号聚合型证据默认高权重，T-5），`ref_id=merchant_id`（O-1）；数值字段由 tools_node backfill 进 extra（O-8），供确定性"规避行为"判定（违规+下架+改标题重上架组合） |
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

> 相似度三档语义（T-11 已拍板，代码常量 `EVIDENCE_MIN_SIM=0.70` / `EVIDENCE_STRONG=0.85`）：
> `<0.70` 不作 IMAGE_SIMILARITY 证据；`0.70~0.85` 普通证据；`≥0.85` **Strong Evidence**（也是矛盾启发式"高相似"判据，
> 代码名 `EVIDENCE_STRONG` 与拍板表旧名 `SIM_HIGH_CONTRADICT` 同值同义，文档统一用 `EVIDENCE_STRONG`）。
> 两个阈值是 **v1 工程初始值、非理论最优**，全部**配置化**（不写死），02-evaluation 在 validation set 上做
> **threshold sweep**（0.60/0.65/0.70/0.75/0.80/0.85/0.90）看 Recall/Precision/FPR/Human Review Rate 选 operating point（《00》§11.5）。

### 5.8 ToolRegistry 与 ToolNode 契约（确定性）

```python
class ToolRegistry:
    register(tool); get(name) -> Tool
    parse_args(tool_name, raw: dict) -> ToolArgs                      # O-5：按名取该工具 args_model 校验/解析（抛 ValidationError，tools_node 捕获记 status="error"）
    list_descriptions() -> [{name, description, args_json_schema}]   # 供 plan prompt（schema 来自 args_model.model_json_schema()）

async def tools_node(state):
    """执行 pending_tool_calls：按 priority 升序，逐个：预算→校验→脱敏→执行→证据质量过滤→转证据→记账→记边际增益。

    全程确定性 Python，不调 LLM。每次调用在 tool_call_history 落 before_confidence /
    after_confidence / evidence_added / decision_changed（Marginal Evidence Gain 取数，见 2.4 与《00》§11.3）。
    """
    updates = {pending_tool_calls: [], budget: copy(state.budget)}
    for call in sorted(state["pending_tool_calls"], key=lambda c: c.priority):
        if updates.budget.tool_calls >= limits.max_tool_calls: break    # 预算截断（Guardrail 上界），剩余不执行
        try:
            parsed = registry.parse_args(call.tool, desensitize(call.args))   # O-5：args_model 校验/解析（坏参抛 ValidationError）
        except ValidationError:
            record(seq, status="error", error="args 校验失败"); continue   # 坏参不执行，当轮继续
        before = decision_conf_probe(updates)                           # ① 调用前 decision_confidence 代理
        result = await tool.call(parsed, ctx)                           # 传强类型 Args；瞬态错误 infra 重试 1 次
        updates.budget.tool_calls += 1
        if not result.ok: record(seq, status="error", error=result.error); continue
        raw = tool.to_evidence(result)                                  # 工具原始结果（5.x 表，不过滤）
        evs = quality_filter(raw)    # ② 确定性证据质量过滤（单参：下限默认 EVIDENCE_MIN_SIM=0.70；Strong 档由 ③ backfill_extra 写 extra.strong 承载）
        evs = backfill_extra(evs)                                       # ③ 把 similarity/removals 等数值写入 Evidence.extra
        updates.evidence = merge_evidence(state.evidence + evs)         # 走 reducer 语义（去重合并）
        after = decision_conf_probe(updates)                            # ④ 调用后 decision_confidence 代理
        refs = [f"{e.type} {e.value}" for e in evs]                   # 引用串（O-1：DTO 无 E_nn，用 type+value 摘要）
        record(seq, status="ok", result_ref=refs[0] if refs else None,
               before_confidence=before, after_confidence=after,
               evidence_added=refs,                                    # 本次新增证据的引用串列表（无新增 = []）
               decision_changed=gate_probe(before 态) != gate_probe(after 态))
    return updates

# decision_conf_probe / gate_probe：确定性轻量函数 —— 用 §7.5 的 decision_confidence 公式与 §7.2 的
# Gate 判定，对"当前证据集/hypotheses"快照算代理值；仅供边际增益审计，不驱动图内路由（路由只认 §6 谓词）。
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

# —— 收敛判定（guardrails/converge.py；纯确定性，可单测；拍板 03 T-4(b)）——
CITABLE_TYPES = {"CASE_PRECEDENT", "POLICY_REF"}

def is_converged(state) -> bool:
    # ① 还有假设未定论（PENDING/UNRESOLVED，含低 prior 假设）→ 未收敛（继续调查）
    open_hp = any(h.status in {"PENDING", "UNRESOLVED"} for h in state["hypotheses"])
    # ② 存在 SUPPORTED 假设但整个证据链没有任何可引用依据（政策条款/先例）→ 未收敛（继续查 Case/Policy）
    supported_wo_citation = any(h.status == "SUPPORTED" for h in state["hypotheses"]) and \
        not any(e.type in CITABLE_TYPES and e.ref_id for e in state["evidence"])
    return not open_hp and not supported_wo_citation
```

> 语义对齐说明（拍板 03 T-4(b)，修订自初版"按 prior≥0.3 过滤"的写法）：收敛判定**不过滤 prior**——走查里的
> H3/H4（prior 0.2/0.15）在第 1 轮后仍 PENDING，若按初版 prior 过滤会被排除出"待收敛集合"导致提前 decide，
> 与《00》§4.4 走查矛盾；"可引用依据"也是**证据级**判断（任一 POLICY_REF/CASE_PRECEDENT 带 ref_id），而非假设级。
> 高优先阈值（0.3）只用于 decide overlay 的 PASS Gate（03 T-1），不用于收敛判定。
> 有 SUPPORTED 假设但检索不到政策/先例时的预期结局：plan 经 dedup 清空重复动作 / 无新工具可查而 `conclude` →
> 路由 decide → overlay 的 REJECT Gate 因无可引用依据不给 REJECT → HUMAN_REVIEW（新型风险）——预期克制行为，非缺陷。

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

| 维度 | v1 上限（Guardrail，拍板 03 T-7；与 `BudgetLimits` 默认一致） | 超限后的路由行为 |
|---|---|---|
| max_llm_calls | 10 | 路由 decide → overlay 产出 `HUMAN_REVIEW`（带已收集的部分证据，overrides=R3_BUDGET_EXHAUSTED） |
| max_tool_calls | 15 | 同上（tools_node 内部也按此截断单批执行） |
| max_tokens | 40000 | 同上；另：llm_shell 在发起调用前若估算超出也直接放弃该调用（壳层守卫） |
| max_latency_ms | 30000 | 同上（wall-clock 从 `budget.start_time` 算） |

超限语义（《00》§8.1 原句，写进注释防误读）：**"调查成本已超过可接受范围，证据不足以自动判，转人工最稳妥"——这是正确的业务行为，不是失败。**
**Guardrail 语义**：上限是"可花费上界"而非目标——主链路走查常态 8 次 LLM / 5 次 Tool，明显低于上限；余量只用于 schema 重试 1 次、工具失败恢复、防无限循环。上限运行时可由配置覆盖；Trace 记录四组占用率（llm/tool/tokens/latency ÷ 上限），评测指标 Budget Utilization（《00》§11.3）。

### 6.5 循环终止性论证（写在文档里给评审看）

循环只可能存在于 `plan → tools → reevaluate → (continue) plan`；每个终止途径都是确定性的：① plan 输出 `conclude` → `route_after_plan` → decide；② dedup 把重复动作清空 → 同上；③ `is_converged()` 为真 → decide；④ 预算维度任一超限 → decide。LLM 语义判断（plan/reevaluate）只影响收敛早晚，不影响"最终一定到达 decide"这一性质（budget 是硬后盾）。

---

## 7. decide 节点的确定性 overlay（对齐《00》§7.2）

### 7.0 结构：LLM 提案 + 确定性 Decision Gate，二层不可合并

decide 节点 = **先**跑 LLM 产出 `DecisionProposal`（第 3.4 章）**后**跑 `run_decision_overlay`。
overlay 是普通确定性 Python（`guardrails/decision_guardrail.py`），实现《00》§7.2 的
**PASS Gate / REJECT Gate 与 HUMAN_REVIEW abstention 清单**；规则顺序固定、全部可单测。

**图终态（D 项拍板）**：decide 产出 `decision` 后图即结束——**DECIDED 是图内唯一终态**（worker 落 DB
`review_run.status=DECIDED`）；PASS/REJECT/HUMAN_REVIEW 是 `ReviewDecision.decision` 的取值而非图终态；
预算耗尽 / 工具失败 / 降级 / Gate 改判一律通过 `decision.overrides` 记录，本图不存在
ESCALATED / BUDGET_EXCEEDED 终结点（03 T-8）。

### 7.1 overlay 输入

- `state`：`hypotheses / evidence / case / budget / failures / degraded / investigation_queue`；
- `proposal: DecisionProposal | None`（None = 预算耗尽 / 降级 / decide 自身 LLM 失败时的空提案）；
- 注意：`proposal.confidence` 只是 LLM 的 **decision_confidence 提案值**，overlay 以**确定性重算值** `dc` 为准（7.5）。

### 7.2 overlay 伪代码（Decision Gate）

```python
CONFIDENCE_ABSTAIN_THRESHOLD = 0.7        # REJECT Gate 安全门槛（拍板 T-4/《00》§7.6，验证集校准）
CITABLE_TYPES = {"CASE_PRECEDENT", "POLICY_REF"}

def run_decision_overlay(state, proposal) -> ReviewDecision:
    evidence, failures, budget = state["evidence"], state["failures"], state["budget"]
    hard_hit = hard_rule_hit(state)                        # R1 硬规则（guardrails/hard_rules.py）

    # ---- R1 硬规则优先：不可被 LLM 覆盖（《00》§7.2-1）----
    if hard_hit:
        return build_decision("REJECT", risk_level="HIGH", risk_type=hard_hit.risk_types,
                              decision_confidence=1.0, evidence=evidence,
                              overrides=["R1_HARD_RULE"])  # 防漏放，覆盖一切（含 PASS 提案）

    proposal = proposal or DecisionProposal(decision="HUMAN_REVIEW",
                                            risk_level="NONE", confidence=0.0, ...)  # 空提案兜底
    dc = finalize_decision_confidence(state)               # 确定性重算 decision_confidence（7.5）

    # ---- HUMAN_REVIEW abstention 清单先行：《00》§7.2-3，任一命中即转人工 ----
    overrides: list[str] = []
    if budget_exceeded(budget):              overrides.append("R3_BUDGET_EXHAUSTED")   # 预算耗尽
    if contradiction_detect(state):          overrides.append("R3_CRITICAL_CONFLICT")   # 关键证据冲突
    if key_tool_failure(state, failures):    overrides.append("R3_KEY_TOOL_FAILED")     # 关键 Tool 失败致证据缺失
    if policy_indeterminate(evidence):       overrides.append("R3_POLICY_UNCERTAIN")    # 政策无法确定/无适用条款
    if indistinguishable_hypotheses(state):  overrides.append("R3_HYPOTHESES_INDISTINGUISHABLE")  # 多假设无法区分
    # R5（O-2 已拍板，对齐 04 §5.2）：failures 非空**不再一律** HUMAN_REVIEW ——
    # 仅 LLM 步失败（state["degraded"]，severity=critical）在此补 R5；未解决的 critical Tool
    # 失败已由上面 key_tool_failure 以 R3_KEY_TOOL_FAILED 计；severity="warn" 的失败只进
    # 审计/trace，不触发转人工（避免任何一次非关键工具抖动推高 Human Review Rate）。
    if state["degraded"]:                      overrides.append("R5_DEGRADED_OR_FAILED_STEP")
    if overrides:
        return build_decision("HUMAN_REVIEW", risk_level=finalize_risk_level(proposal),
                              risk_type=finalize_risk_type(proposal), decision_confidence=dc,
                              evidence=evidence, policy=citable_policy_ids(evidence),
                              hypothesis_trace=trace_from(state["hypotheses"]),
                              budget_used=snapshot_budget(budget), overrides=overrides)

    # ---- PASS / REJECT Gate：校验 LLM 提案（《00》§7.2-2/-4），不满足即降 HUMAN_REVIEW ----
    if proposal.decision == "PASS" and not pass_gate(state):
        return build_decision("HUMAN_REVIEW", ..., decision_confidence=dc, overrides=["R4_PASS_GATE_FAIL"])
    if proposal.decision == "REJECT" and not reject_gate(state, dc):
        return build_decision("HUMAN_REVIEW", ..., decision_confidence=dc, overrides=["R2_REJECT_GATE_FAIL"])

    # 提案 HUMAN_REVIEW，或 Gate 通过 → 采纳（decision_confidence 用确定性 dc）
    return build_decision(decision=proposal.decision, ..., decision_confidence=dc, overrides=[])

# —— PASS Gate：《00》§7.2-4 ——
def pass_gate(state) -> bool:
    return (all(h.status == "REFUTED" and h.evidence_against                      # ① 高优先假设被"充分证据"证伪
                for h in high_priority(state))                                     #    （有反驳证据，非"没查到"）
            and key_evidence_complete(state)                                       # ② 关键证据完整（无关键 Tool 失败缺失）
            and not contradiction_detect(state))                                   # ③ 无未解决关键矛盾

# —— REJECT Gate：《00》§7.2-2 ——
def reject_gate(state, dc) -> bool:
    return (any(h.status == "SUPPORTED" for h in high_priority(state))             # ① 高风险假设成立
            and evidence_sufficient(state)                                         # ② 证据充分
            and any(e.type in CITABLE_TYPES and e.ref_id for e in state["evidence"])  # ③ 明确政策依据/先例
            and dc >= CONFIDENCE_ABSTAIN_THRESHOLD                                 # ④ decision_confidence≥0.7（安全门槛）
            and not contradiction_detect(state))                                   # ⑤ 无关键矛盾
```

> `high_priority(h) = h.prior >= HIGH_PRIOR_THRESHOLD(0.3)`（T-1）：prior 只用于 PASS/REJECT Gate 的"高优先"口径与 prompt 强调，**不用于收敛判定**（6.3）。

### 7.3 预算/降级分支（decide 入口，7.0 的第一步）

```
若 budget_exceeded(budget) 或 state["degraded"]（LLM 步失败）或存在未解决 critical Tool 失败（04 §5.2）：
    → proposal = None（不调用 LLM，不再烧 token）
（O-2 已拍板：仅 severity=warn 的 Tool 失败**不**跳 LLM —— 它只审计，不构成 abstention。）
否则正常调 LLM（proposal）
两种情况都必须进入 run_decision_overlay —— 因为 R1 硬规则可能把结果改成 REJECT。
预算耗尽/降级最终由 overlay 的 abstention 清单产出 HUMAN_REVIEW（overrides=R3_BUDGET_EXHAUSTED/R5_*）。
```

### 7.4 `hard_rule_hit`（`guardrails/hard_rules.py`，R1 数据来源）

纯确定性扫描以下来源，命中即强制 REJECT：
- `case.product.brand` / `case.product.title/description` 命中品牌黑名单（词表来自规则引擎复用）；
- `case.screening_signals` 中带"硬违禁"语义且 result 非 PASS 的信号（正常情况分流层不会把硬违规投进来，但调查中新证据可能触发）；
- 调查新证据：`IMAGE_LOGO`（识别到黑名单品牌 Logo）、`OCR_TEXT` 命中禁词等——用 `Evidence.extra` 结构化字段判定，不用 LLM。

### 7.5 decision_confidence 与 risk 的分离（确定性重算，T-4）

- **`decision_confidence`（安全门槛）**：落库 `ReviewDecision.decision_confidence`（O-7 已改名）的值由确定性函数重算——
  LLM 提案的 confidence 只作参考，不作为最终值（保证可解释、可单测）：

```python
def finalize_decision_confidence(state) -> float:
    # 产出的是 decision_confidence：对"自动决策安全"的把握（非违规概率）——
    # 只回答"如果自动判，判错风险够不够低"，不回答"风险有多高"（后者由 risk_level / 假设 posterior 表达）
    top = max((h.posterior or 0.0 for h in state["hypotheses"] if h.status == "SUPPORTED"), default=0.0)
    completeness = min(len(state["evidence"]) / MAX_EXPECTED_EVIDENCE, 1.0)     # 分母默认 8（T-4）
    citation = 1.0 if any(e.type in CITABLE_TYPES and e.ref_id for e in state["evidence"]) else 0.0
    conflict = 0.2 if contradiction_detect(state) else 0.0
    c = 0.45 * top + 0.25 * completeness + 0.20 * citation + 0.10 - conflict
    return round(min(max(c, 0.0), 1.0), 2)
```

- **与 risk 分离**：`risk_level`（NONE/LOW/MEDIUM/HIGH）与风险强度（最高 SUPPORTED 假设的 `posterior`，见
  `hypothesis_trace`）表达"风险有多高"，**不参与 Gate 判定**（只用于展示/队列排序，《00》§7.5）；`decision_confidence`
  单独表达"能不能安全自动判"。
- **0.7 的用法**：仅作为 **REJECT Gate** 的安全门槛（`reject_gate` 里 `dc >= 0.7`）；**PASS 不因低 decision_confidence
  转人工**——PASS 由 `pass_gate` 判定（高优先假设充分证伪 + 关键证据完整 + 无关键矛盾），干净商品低风险置信是正常态。
- `CONFIDENCE_ABSTAIN_THRESHOLD=0.7` / `MAX_EXPECTED_EVIDENCE=8` 均配置化，validation set 校准（《00》§11.5）。

### 7.6 最终 ReviewDecision 形状（对齐《00》§2.2 与代码 `ReviewDecision`，落库前不变形）

```json
{
  "decision": "HUMAN_REVIEW",
  "risk_level": "HIGH",
  "risk_type": ["POTENTIAL_IP_RISK", "EVASION_PATTERN"],
  "decision_confidence": 0.87,
  "evidence": [ { "type": "IMAGE_SIMILARITY", "source": "ImageAnalysisTool",
                  "value": "similarity=0.91, match=某品牌经典鞋款", "weight": 0.91,
                  "ref_id": null, "extra": {"similarity": 0.91} } ],
  "policy": ["POLICY_3.2"],
  "hypothesis_trace": [ { "id": "H3", "statement": "刻意规避品牌识别",
                          "prior": 0.2, "posterior": 0.88, "status": "SUPPORTED" } ],
  "budget_used": { "llm_calls": 8, "tool_calls": 5, "tokens": 18000, "latency_ms": 9200 },
  "overrides": []
}
```

> `decision_confidence` = 自动决策安全门槛（§7.5，确定性重算）。伪代码里 `build_decision(..., decision_confidence=dc)` 的形参即 DTO 字段 `decision_confidence`（O-7 已拍板改名，与代码一致）。
> 示例 0.87 = §7.5 确定性重算终值（0.91 只是 LLM 提案 confidence 参考值，demo 断言 ≥0.7）。
> `overrides` 记录确定性 overlay 的改判/归因原因码
> （R1_HARD_RULE / R2_REJECT_GATE_FAIL / R3_BUDGET_EXHAUSTED / R3_CRITICAL_CONFLICT / R3_KEY_TOOL_FAILED /
> R3_POLICY_UNCERTAIN / R3_HYPOTHESES_INDISTINGUISHABLE / R4_PASS_GATE_FAIL / R5_DEGRADED_OR_FAILED_STEP）；
> 空 = overlay 未改判（LLM 提案即终值）。`evidence[]` 元素为代码 `Evidence` 形状（无 evidence_id/source_tool，运行序号在 trace）。

---

## 8. 与《00》§4.4 核心场景（复古运动鞋）的逐轮对齐

本节点验证第 2~7 章契约能走通主链路。列：设计轮次（《00》§4.4 表的行）→ 图执行序列 → 关键 state 变化 → 走的边与判定依据。

前置事实（《00》§4.4 / §2.1）：机审 4 项 signal 全 PASS、`brand=null`、标题/描述无品牌词、图片待分析、商家 M_5512。

| 设计轮次 | 图执行（节点序列） | 产出 / state 变化 | 走的边（含谓词依据） |
|---|---|---|---|
| 0 | `hypothesize`（入口唯一一次） | 建立 H1（普通复古设计，prior 0.5）/ H2（参考知名品牌经典设计，prior 0.4）/ H3（刻意规避品牌识别，prior 0.2）/ H4（商家系统性类似行为，prior 0.15），均 PENDING；队列置 2 个 OPEN 问题 | 静态边 hypothesize→plan |
| 1 | `plan` | 读仪表盘：H1/H2 高优先（prior≥0.3）且 PENDING；判定"外观是否对应某品牌"最值得查 → `next_action=call_tools, tools=[{ImageAnalysisTool, args:{image_urls:[img1]}, priority:1}]` | 静态边 plan→tools（经 route_after_plan ③：pending 非空、预算 OK、未 degraded） |
| 1 | `tools`（ToolNode） | 执行 ImageAnalysisTool → similarity=0.91 命中品牌款 → **E_01**（IMAGE_SIMILARITY, weight 0.91）；budget.tool_calls=1 | 静态边 tools→reevaluate |
| 1 | `reevaluate` | 综合 E_01：H1 REFUTED（证据 against），H2 SUPPORTED（posterior ↑）；H3/H4 仍 PENDING | route_after_reevaluate：未 degraded、预算 OK、`is_converged=false`（H3/H4 仍 PENDING，未定论假设存在，见 6.3 收敛谓词——不过滤 prior）→ **continue → plan** |
| 2 | `plan` | "需确认商品字段真空缺 + 商家历史" → `call_tools, tools=[{ProductTool, priority:1}, {MerchantTool, priority:2}]` | plan→tools（同 ③） |
| 2 | `tools` | ProductTool → brand=null、标题/描述无品牌（**E_02** PRODUCT_FACT）；MerchantTool → 23 similar / 5 removals / 3 title-relisting（**E_03** MERCHANT_HISTORY）；tool_calls=3 | tools→reevaluate |
| 2 | `reevaluate` | E_02/E_03 支持：H3 SUPPORTED（posterior 0.88）、H4 SUPPORTED（posterior 0.85）；队列两问 DONE | route_after_reevaluate：`is_converged=false` —— 无未定论假设，但存在 SUPPORTED 假设且**证据链仍无任何 CASE_PRECEDENT/POLICY_REF 可引用依据**（supported_wo_citation=true，对齐《00》§7.2-2）→ **continue → plan** |
| 3 | `plan` | "需要先例 + 政策支撑才能判" → `call_tools, tools=[{CaseSearchTool, priority:1}, {PolicySearchTool, priority:2}]` | plan→tools |
| 3 | `tools` | CaseSearchTool → CASE_1832 高度相似 → REJECT（**E_04** CASE_PRECEDENT, ref_id=CASE_1832）；PolicySearchTool → POLICY_3.2"外观高度模仿高风险转人工"（**E_05** POLICY_REF, ref_id=clause）；tool_calls=5 | tools→reevaluate |
| 3 | `reevaluate` | 证据链补全：无未定论假设，且已存在可引用依据（E_04/E_05） | route_after_reevaluate：`is_converged=true` → **decide** |
| 4 | `decide` | LLM 提案：`HUMAN_REVIEW / HIGH / [POTENTIAL_IP_RISK, EVASION_PATTERN] / decision_confidence 0.91 / evidence E_01..E_05 / policy [POLICY_3.2]`；overlay：R1 硬规则未命中 → abstention 清单（预算/关键冲突/工具失败/政策不确定/多假设不可分/降级）均不成立 → 提案即 HUMAN_REVIEW，PASS/REJECT Gate 不适用 → 采纳；`decision.decision_confidence`（确定性重算）=0.87（0.91 只是 LLM 提案 confidence 参考值，终值=确定性重算 0.87，demo 断言 ≥0.7）、`overrides=[]`；worker 置 DB `review_run.status=DECIDED`（图唯一终态） | 终结点（无出边） |

**"为什么第 4 步是 HUMAN_REVIEW 而不是 REJECT"的契约解释**：overlay 的 REJECT Gate 其实已可满足（H2/H3 SUPPORTED + 证据充分 + E_04/E_05 可引用依据 + decision_confidence 0.87≥0.7 + 无矛盾），但 LLM 提案为 HUMAN_REVIEW 且 POLICY_3.2 指引是"高风险转人工"（仿冒属主观判定）——overlay 只做**下限守卫**（防止不安全自动判），**不把 HUMAN 提案强行升为 REJECT**。这体现 Agent"知道什么时候该人介入"（《00》§4.4 注意行）。若未来该案改为可自动判，只动政策指引与提案，Gate 结构不变。

**预算核查（Guardrail 语义，对齐 6.4）**：llm_calls=8 ≤ **10**（Guardrail 上界，余量 2，主链路常态 8 次明显低于上限）、tool_calls=5 ≤ **15**、tokens/latency 未超 → 主链路在预算内走通；余量保留给 schema 重试/工具失败恢复（T-7 拍板 10/15）。

---

## 9. 待定项状态索引（T-1~T-12 已全部拍板，权威值为 docs/03-decisions.md）

> 初版第 9 章曾列 12 项开放问题；**经拍板（docs/03-decisions.md）全部关闭**。本表只留状态与最终值索引，
> 权威细节（含选项、理由、一致性核查）一律以 03-decisions.md 为准；实现时不要再按本节旧默认值（如 8/12、0.60）开发。

| ID | 状态 | 最终值（权威：03-decisions.md） |
|---|---|---|
| T-1 | 已定 [A] | prior 由 hypothesize LLM 输出（0..1，不归一化）+ 钳制；须含 ≥1 条低风险假设；`HIGH_PRIOR_THRESHOLD=0.3`（仅 PASS Gate / prompt 强调，不用于收敛判定） |
| T-2 | 已定 [A] | `MAX_HYPOTHESES=5` / `MAX_QUEUE=8` / `MAX_TOOLS_PER_PLAN=3`（配置化） |
| T-3 | 已定 [A]（代码已落地） | `PENDING / SUPPORTED / REFUTED / UNRESOLVED`；代码 `models.py` 已用 `UNRESOLVED` |
| T-4 | 已定 [A]（本版按 review 修订为 decision_confidence 口径） | 见 §6.3 收敛谓词 / §7.2 Gate / §7.5 decision_confidence 公式；`CONFIDENCE_ABSTAIN_THRESHOLD=0.7`（仅 REJECT Gate）、`MAX_EXPECTED_EVIDENCE=8` |
| T-5 | 已定 [A] | 工具转换器写默认权重，reevaluate 不改；weight 仅供审计/展示，不参与 v1 公式 |
| T-6 | 已定 [A] | hypothesize 仅入口 1 次；新假设走 `reevaluate.new_hypotheses` |
| T-7 | 已拍板 [B→B] | `10 / 15 / 40000 / 30000`（Guardrail 上界非目标；代码 `BudgetLimits` 默认已改，见 §6.4/§8.1） |
| T-8 | 已定 [A] | DECIDED 为图唯一终态；ESCALATED/BUDGET_EXCEEDED 不作终态（DB/归因）；`ReviewDecision.overrides` 已落地 |
| T-9 | 已定 [A]（代码已落地） | `pending_tool_calls / degraded / failures` 通道保留；run_id/case_id→thread、status→DB |
| T-10 | 已定 [A] | `risk_level=NONE/LOW/MEDIUM/HIGH`（代码已含 NONE，PASS→NONE）；仅展示/队列排序，不参与路由与 Gate |
| T-11 | 已拍板 [B→B] | `EVIDENCE_MIN_SIM=0.70` / `EVIDENCE_STRONG=0.85`（三档语义 + threshold sweep，见 §5.7；代码常量已落地） |
| T-12 | 已定 [A] | FIELD_CONFLICT 检测归二期 guardrails；v1 词表保留 |

**使用约定**：T-1~T-12 实现/评测参数一律取 03-decisions.md §5 常量表；本节与 03 冲突处以 03 为准。

---

> 附：实现顺序建议（与《00》末尾"下一步"呼应）：① `domain/` 各 Pydantic 模型 + `agent/state.py`（第 2 章）→ ② `guardrails/`（budget/dedup/hard_rules/decision_guardrail，第 6/7 章，全部先写单测）→ ③ 4 个 LLM 节点壳 + mock LLM（第 3 章）→ ④ 6 个 Tool 空实现 + ToolRegistry + ToolNode（第 5 章）→ ⑤ graph.py 接线跑通复古运动鞋单链路（第 8 章验证）→ ⑥ Checkpointer 接 MySQL。
