# 复杂风险调查 Agent —— LangGraph StateGraph 正式设计（docs/04-graph-design.md v1）

> 本文档是进入 `src/pra/agent/graph.py` 实现前的**最后一份设计文档**：把《00-system-design.md》(v1.1)、
> 《01-agent-loop.md》(v2)、《03-decisions.md》(v2) 的契约与已落地代码，翻译成**可直接照抄的
> LangGraph 图定义草稿 + 各文件实现落点**。只设计"复杂案件调查子图"（screening 分流在子图外）。
>
> 文中代码草稿的 API 已按**本机实际安装版本**逐一核对/实测（§1.1），import 路径统一 `pra.*`；
> 未做任何 git 操作、未写 `src/` 下任何代码（本文档是唯一交付物）。

---

## 1. 范围与前置

### 1.1 版本基线与依赖事实（实测）

| 基线 | 版本/状态 | 说明 |
|---|---|---|
| 00-system-design.md | v1.1 | 决策机制三 Gate、Budget Guardrail 语义（已 revision） |
| 01-agent-loop.md | v2 | AgentState 10 字段契约、4 LLM 节点、6 Tool、条件边路由、decide Gate overlay |
| 03-decisions.md | v2 | T-1~T-12 全部拍板（含 10/15/40000/30s、0.70/0.85） |
| `langgraph` | **1.2.11**（实测 `importlib.metadata`） | `langgraph.__version__` 不存在，勿用 |
| `langgraph-checkpoint` | **4.2.0**（已装核心） | 提供 `InMemorySaver`；SQLite/Postgres saver 在**未安装**的可选包中 |
| `langsmith` | 0.12.2 | 无关本次图定义 |
| 落地契约 | models.py / state.py / tools/base.py / tools/__init__.py（`build_tools()`） | 见 §1.2 |

**本机 LangGraph API 核对结论（直接决定草稿写法）**：
- `from langgraph.graph import StateGraph, START, END`（`START`/`END` 即 `__start__`/`__end__`）；
- `builder.add_node(name, action)`；`builder.add_edge(start_key | list[str], end_key)`；
  `builder.add_conditional_edges(source, path, path_map: dict[Hashable, str])`；
  `builder.compile(checkpointer=..., interrupt_before=..., interrupt_after=...)`；
- 节点 action 可 `async def node(state, config) -> dict`（第二参 `config: RunnableConfig` 可选，用于读
  `config["configurable"]["thread_id"]`）；条件边 path 可为同步/异步纯函数 `(state) -> key`；
- `compiled.ainvoke(input, {"configurable": {"thread_id": ...}})`；`input` 通常为部分/全量 state dict；
- TypedDict channel 的 reducer 用 `Annotated[T, reducer_fn]` 声明（`Annotated[list[X], operator.add]` 追加、
  自定义 `merge_evidence(left, right)` 去重合并）；
- Checkpointer：核心包内 `langgraph.checkpoint.memory.InMemorySaver` 可用；`BaseCheckpointSaver`（含
  sync/async 双轨 `get_tuple/aget_tuple, put/aput, put_writes/aput_writes, list/alist` 等方法）可用于自研；
  **官方无 MySQL saver**（详 §7）。

### 1.2 已落地契约（本文档直接引用，不再重复定义）

- `pra.domain.models`：`ProductReviewCase/ProductInfo/SkuInfo/ProductImage/ScreeningSignal`、
  `Hypothesis`（`prior/posterior: float|None`、`status: HypothesisStatus{...UNRESOLVED}`、
  `evidence_for/against: list[str]`）、`Evidence{type,source,value,weight,ref_id,extra:dict}`、
  `Budget/BudgetLimits`（默认 10/15/40000/30000）、`ReviewDecision`（含 `overrides: list[str]`）、
  枚举 `Decision/RiskLevel(含 NONE)/RiskType`。
- `pra.agent.state.AgentState`（10 字段）：`case / hypotheses / evidence / investigation_queue /
  tool_call_history / budget / decision / pending_tool_calls / degraded / failures`（后三者 + status 口径见 01 §2）。
- `pra.tools.base`：`ToolArgs / ToolResult / ToolContext{run_id,case_id,budget} / Tool(Protocol) / ToolRegistry`；
  `pra.tools.build_tools()` 返回 6 个带 InMemory 数据源的 Tool。
- 决策语义（00 §7 / 01 §7 / 03 T-4）：decision_confidence（安全门槛，落 `ReviewDecision.decision_confidence`，O-7 已改名）与
  risk 分离；PASS/REJECT Gate + HUMAN_REVIEW abstention 清单；`DECIDED` 是图唯一终态。

### 1.3 子图边界

只设计 **Complex Risk Investigation 子图**：入参=一个已分流进来的 `ProductReviewCase`（含 screening_signals
起点），出参=`AgentState`（含 `decision: ReviewDecision`）。screening/分流、MQ 消费、DB 落库、人工队列都**在子图外**
由 worker/下游系统承担（§3.4）。

---

## 2. 图拓扑（代码级）

### 2.1 节点/边总览

```
START ──> hypothesize ──> plan ──(条件)──> tools ──> reevaluate ──(条件)──> decide ──> END
                ▲────────────────────────────────────┘(continue)   │(decide)
```

| 项 | 值 |
|---|---|
| 节点数 | 5：`hypothesize / plan / tools / reevaluate / decide` |
| 静态边 | `START→hypothesize`、`hypothesize→plan`、`tools→reevaluate`、`decide→END` |
| 条件边 | `plan`：path=`route_after_plan`，map=`{"tools":"tools","decide":"decide"}`；`reevaluate`：path=`route_after_reevaluate`，map=`{"continue":"plan","decide":"decide"}` |
| 可回环节点 | 仅 `plan→tools→reevaluate→plan` 一圈（预算/收敛保证终止，见 §6） |

### 2.2 完整图定义草稿（`src/pra/agent/graph.py`，可照抄）

```python
"""src/pra/agent/graph.py —— 复杂风险调查子图装配（草稿，按 docs/04-graph-design.md 实现）。"""
from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from pra.agent.state import AgentState
from pra.agent.nodes.hypothesize import hypothesize_node
from pra.agent.nodes.plan import plan_node
from pra.agent.nodes.reevaluate import reevaluate_node
from pra.agent.nodes.decide import decide_node
from pra.agent.tools_node import tools_node

# 节点名常量（路由表 key 与 add_node 名一一对应）
N_HYPOTHESIZE = "hypothesize"
N_PLAN = "plan"
N_TOOLS = "tools"
N_REEVALUATE = "reevaluate"
N_DECIDE = "decide"


def route_after_plan(state: AgentState) -> Literal["tools", "decide"]:
    """plan 之后：① degraded（schema 校验失败，见 §5）→ decide；② 预算超限 → decide；
    ③ pending_tool_calls 非空（已过 dedup）→ tools；④ conclude/空计划 → decide。"""
    if state["degraded"]:
        return N_DECIDE                       # 降级：证据不足 → 直接收尾（01 §6.2 ①）
    from pra.agent.guardrails.budget import budget_exceeded
    if budget_exceeded(state["budget"]) is not None:
        return N_DECIDE                       # 预算护栏（01 §6.2 ② / 00 §8.1）
    if state["pending_tool_calls"]:
        return N_TOOLS                        # 有真实动作才进 tools（01 §6.2 ③）
    return N_DECIDE                           # conclude / 空计划（01 §6.2 ④）


def route_after_reevaluate(state: AgentState) -> Literal["continue", "decide"]:
    """reevaluate 之后：① degraded → decide；② 预算超限 → decide；③ is_converged → decide；否则 continue（回 plan）。"""
    if state["degraded"]:
        return N_DECIDE
    from pra.agent.guardrails.budget import budget_exceeded
    if budget_exceeded(state["budget"]) is not None:
        return N_DECIDE
    from pra.agent.guardrails.converge import is_converged
    if is_converged(state):
        return N_DECIDE
    return "continue"                         # map 到 plan


def build_agent_graph(*, tools: list[Any] | None = None,
                      checkpointer: Any | None = None,
                      ) -> CompiledStateGraph:
    """装配并编译调查子图。

    - tools: 6 个 Tool（默认 pra.tools.build_tools()），注册进 ToolRegistry 供 tools_node 使用；
    - checkpointer: 见 §7（MVP 传 InMemorySaver；None 时不持久化，仅调试用）。
    本函数同时负责把 tools/registry 注入 tools_node 可访问的位置（通过闭包工厂或
    模块级 provider，二选一，见 §9 tools_node 说明）。
    """
    builder = StateGraph(AgentState)

    builder.add_node(N_HYPOTHESIZE, hypothesize_node)
    builder.add_node(N_PLAN, plan_node)
    builder.add_node(N_TOOLS, tools_node)
    builder.add_node(N_REEVALUATE, reevaluate_node)
    builder.add_node(N_DECIDE, decide_node)

    builder.add_edge(START, N_HYPOTHESIZE)
    builder.add_edge(N_HYPOTHESIZE, N_PLAN)
    builder.add_conditional_edges(
        N_PLAN, route_after_plan,
        {N_TOOLS: N_TOOLS, N_DECIDE: N_DECIDE},        # key = 路由函数返回值
    )
    builder.add_edge(N_TOOLS, N_REEVALUATE)
    builder.add_conditional_edges(
        N_REEVALUATE, route_after_reevaluate,
        {"continue": N_PLAN, N_DECIDE: N_DECIDE},
    )
    builder.add_edge(N_DECIDE, END)                     # DECIDED 唯一终态：decide 产出 decision 即结束（03 T-8）

    return builder.compile(checkpointer=checkpointer)
```

> 落地注（graph.py 已实现，与 §9 tools_node 行口径一致）：模块**无**模块级 `tools_node` 符号 —— tools 节点由
> `make_tools_node(tools)` 闭包工厂生成（闭包内注册私有 ToolRegistry 并返回 `async tools_node(state, config)`）；
> 装配处 `tools_action = make_tools_node(tools)` 后再 `add_node(N_TOOLS, tools_action)`，上方 import/add_node 为草稿示意。

> 实现提示（LangGraph 1.2.x 行为，已实测）：条件边 path 在每个上游节点写入 state 之后、进入下一节点之前执行，
> 因此 `route_after_plan` 能读到 plan 刚写入的 `pending_tool_calls`、`route_after_reevaluate` 能读到
> reevaluate 更新后的 `hypotheses`。路由函数保持**纯确定性**（预算/收敛谓词下沉 guardrails）。

### 2.3 reducer 语义（代码级实现，落到 `pra/agent/state.py`）

01 §2.5 的 reducer 表在此给出**可执行定义**。注意：`state.py` 目前是"纯 TypedDict + 注释"形态，落地时需按
本表把 `Annotated[...]` 与 reducer 函数补上（属 §9 的 state.py 补充项，非本次改动）。

```python
"""src/pra/agent/state.py —— AgentState + reducer（补充草稿）"""
from __future__ import annotations
from operator import add
from typing import Annotated, TypedDict

from pra.domain.models import Budget, Evidence, Hypothesis, ProductReviewCase, ReviewDecision


def _evidence_key(e: Evidence) -> tuple:
    """证据去重指纹（O-1 已拍板）：ref_id 优先稳定业务标识（image_url / product_id /
    merchant_id / case_id / clause_id）；ref_id 为 None 时回退 value —— 防同 (type, source)
    的多条无 ref 证据（如多图多品牌命中）互相吞并。"""
    return (e.type, e.source, e.ref_id if e.ref_id is not None else e.value)


def merge_evidence(left: list[Evidence], right: list[Evidence]) -> list[Evidence]:
    """evidence channel reducer：按 _evidence_key 去重合并（左=当前 state，右=节点新增）。

    语义：已存在同 key → 丢弃新增（证据一旦收集不可篡改，重放/重试幂等）；新 key → append。
    """
    seen = {_evidence_key(e): e for e in left}
    out = list(left)
    for e in right:
        if _evidence_key(e) not in seen:
            seen[_evidence_key(e)] = e
            out.append(e)
    return out


class AgentState(TypedDict, total=False):
    # 输入事实（只读；worker 构造，一次调查不变）
    case: ProductReviewCase
    # 核心推理状态
    hypotheses: list[Hypothesis]                                   # 覆盖写（hypothesize 初始化 / reevaluate 返回全集）
    evidence: Annotated[list[Evidence], merge_evidence]            # 自定义去重合并 reducer
    investigation_queue: list[dict]                                # 覆盖写；元素 {q, priority, status}
    tool_call_history: Annotated[list[dict], add]                  # append；含边际增益 4 字段（§4）
    # 成本/审计
    budget: Budget                                                 # 覆盖写（整对象）
    decision: ReviewDecision | None                                # 覆盖写（decide 终局写入）
    # 图内通道（03 T-9）
    pending_tool_calls: list[dict]                                 # 覆盖写；plan 写 / tools 消费后置 []
    degraded: bool                                                 # 覆盖写；LLM 步失败降级标记
    failures: Annotated[list[dict], add]                           # append；元素 {step_type, tool?, severity, reason, ts}


def build_initial_state(case: ProductReviewCase) -> AgentState:
    """每次 invoke 的完整初始输入（保证所有 channel 有值，首节点读不炸）。"""
    from pra.domain.models import Budget
    return AgentState(
        case=case,
        hypotheses=[],
        evidence=[],
        investigation_queue=[],
        tool_call_history=[],
        budget=Budget(),                       # 默认 10/15/40000/30000，可由配置覆盖
        decision=None,
        pending_tool_calls=[],
        degraded=False,
        failures=[],
    )
```

> **invoke 约定**：一律 `await app.ainvoke(build_initial_state(case), {"configurable": {"thread_id": run_id}})`——
> 全量键输入保证每个 channel 首读有值、reducer 首写安全；`thread_id = run_id`（03 T-9 映射，O-6）。

**Reducer/覆盖写速查表**（与 01 §2.5 逐字段一致，含代码形态）：

| 字段 | reducer/语义 | 代码形态 | 写入方 |
|---|---|---|---|
| `evidence` | 去重合并 | `Annotated[list[Evidence], merge_evidence]` | tools_node（只返回**新增**证据） |
| `tool_call_history` | append | `Annotated[list[dict], add]` | tools_node / dedup（`skipped`） |
| `failures` | append | `Annotated[list[dict], add]` | LLM 节点 / tools_node |
| `hypotheses` | 覆盖写（返回全集） | 无 reducer | hypothesize / reevaluate |
| `investigation_queue` | 覆盖写（返回全集） | 无 reducer | hypothesize / reevaluate |
| `budget` | 覆盖写（整对象） | 无 reducer | LLM 节点壳 / tools_node |
| `decision` | 覆盖写 | 无 reducer | decide |
| `degraded` | 覆盖写 | 无 reducer | 各 LLM 节点（失败 True / 成功 False） |
| `pending_tool_calls` | 覆盖写 | 无 reducer | plan 写入 / tools 置 `[]` |
| `case` | 覆盖写（只写一次） | 无 reducer | worker（invoke 输入） |

> 单路径线性链上每步只有一个合法写方，故 list 之外全用覆盖写；reducer 只服务于"追加/去重幂等"两类需求
> （checkpointer 断点重放、super-step 重跑时重复执行不产生重复证据/记录）。

### 2.4 预算与收敛谓词（最终版签名，落到 `guardrails/budget.py` / `guardrails/converge.py`）

```python
# guardrails/budget.py —— 对齐 01 §6.4 / 03 T-7
from pra.domain.models import Budget

def budget_exceeded(budget: Budget) -> str | None:
    """返回首个超限维度 'LLM_CALLS' | 'TOOL_CALLS' | 'TOKENS' | 'LATENCY'；全部未超限返回 None。"""
    if budget.llm_calls >= budget.limits.max_llm_calls:
        return "LLM_CALLS"
    if budget.tool_calls >= budget.limits.max_tool_calls:
        return "TOOL_CALLS"
    if budget.tokens >= budget.limits.max_tokens:
        return "TOKENS"
    elapsed_ms = _now_ms() - int(budget.start_time.timestamp() * 1000)
    if elapsed_ms >= budget.limits.max_latency_ms:
        return "LATENCY"
    return None

def bump_llm_usage(budget: Budget, *, llm_calls: int = 1, tokens: int = 0) -> Budget:
    return budget.model_copy(update={"llm_calls": budget.llm_calls + llm_calls,
                                     "tokens": budget.tokens + tokens})

def bump_tool_usage(budget: Budget, *, tokens: int = 0) -> Budget:
    return budget.model_copy(update={"tool_calls": budget.tool_calls + 1,
                                     "tokens": budget.tokens + tokens})

def snapshot_budget(budget: Budget) -> Budget:
    """决策时预算快照（写入 ReviewDecision.budget_used；补 latency_ms）。"""
    return budget.model_copy(update={"latency_ms": _now_ms() - int(budget.start_time.timestamp() * 1000)})
```

```python
# guardrails/converge.py —— 对齐 01 §6.3 / 03 T-4(b)
CITABLE_TYPES = {"CASE_PRECEDENT", "POLICY_REF"}

def is_converged(state) -> bool:
    """收敛 = ①无 PENDING/UNRESOLVED 假设（全部假设，不限 prior）+ ②不存在'SUPPORTED 但证据链无任何
    可引用依据(POLICY_REF/CASE_PRECEDENT 且带 ref_id)'。"""
    hypotheses = state["hypotheses"]
    open_hp = any(h.status in {"PENDING", "UNRESOLVED"} for h in hypotheses)
    supported_wo_citation = (
        any(h.status == "SUPPORTED" for h in hypotheses)
        and not any(e.type in CITABLE_TYPES and e.ref_id for e in state["evidence"])
    )
    return not open_hp and not supported_wo_citation
```

---

## 3. 三 Gate 接线（decide 节点内）

### 3.1 结构：LLM 提案 → 确定性 overlay

decide 节点 = 两层（01 §7.0，此处给出函数级落点）：

```python
# src/pra/agent/nodes/decide.py（草稿）
async def decide_node(state, config) -> dict:
    from pra.agent.guardrails.budget import budget_exceeded, bump_llm_usage, snapshot_budget
    from pra.agent.guardrails.decision_guardrail import run_decision_overlay
    from pra.agent.guardrails.llm_shell import call_structured_llm

    proposal = None
    # ① 预算/降级 → 不调 LLM（proposal=None）；否则 LLM 提案（失败→None）
    if budget_exceeded(state["budget"]) is None and not state["degraded"]:
        proposal = await call_structured_llm(OutputModel=DecisionProposal, state=state, config=config)

    # ② 确定性 overlay 永远执行（R1 硬规则可能强制 REJECT）——产出最终 ReviewDecision
    final = run_decision_overlay(state, proposal)

    updates = {"decision": final, "degraded": False}
    # ③ 预算记账（若调过 LLM）+ 预算快照已含在 decision.budget_used（overlay 内 snapshot_budget）
    return updates
```

> 说明：各节点的 LLM 输出模型（`DecisionProposal / PlanOutput / ReevaluateOutput / HypothesizeOutput`）随节点模块
> 定义或放 `guardrails/schemas.py`（统一导入），本文不重复展开其字段（见 01 §3.2/§3.3/§3.4 契约）。

### 3.2 guardrails 模块函数签名（`pra/agent/guardrails/decision_guardrail.py`）

```python
def run_decision_overlay(state, proposal: DecisionProposal | None) -> ReviewDecision:
    """执行顺序固定（00 §7.2 / 01 §7.2）：
       R1 硬规则强制 REJECT → abstention 清单（预算/关键冲突/关键 Tool 失败/政策不确定/多假设不可分/
       LLM 步降级或失败）任一命中→HUMAN_REVIEW → 校验 LLM 提案：PASS Gate / REJECT Gate 不通过→HUMAN_REVIEW
       → 采纳提案（decision_confidence 用确定性重算值）。改判/归因全部写入 decision.overrides（03 T-8）。"""

def hard_rule_hit(state) -> HardRuleHit | None: ...   # R1：guardrails/hard_rules.py，命中即强制 REJECT

def pass_gate(state) -> bool:
    # ① 高优先假设(h.prior>=0.3)全部 REFUTED 且 evidence_against 非空（充分证伪）
    # ② key_evidence_complete(state) ③ not contradiction_detect(state)

def reject_gate(state, dc: float) -> bool:
    # ① ∃高优先 SUPPORTED ② evidence_sufficient(state) ③ ∃ POLICY_REF/CASE_PRECEDENT 带 ref_id
    # ④ dc >= 0.7 ⑤ not contradiction_detect(state)

def contradiction_detect(state) -> bool: ...   # v1：IMAGE_SIMILARITY extra.similarity>=0.85 ∧ MERCHANT_HISTORY 干净（03 T-4(e)）

def policy_indeterminate(evidence) -> bool: ...
def indistinguishable_hypotheses(state) -> bool: ...
def key_tool_failure(state, failures) -> bool: ...      # §5：存在未解决的 critical TOOL_CALL 失败
def finalize_decision_confidence(state) -> float: ...   # 确定性公式（01 §7.5 / 03 T-4(c)）
def build_decision(...) -> ReviewDecision: ...          # 组装 decision + overrides + budget_used=snapshot_budget
```

**Gate 判定条件（文档可引用版本）**：
- **PASS Gate**：可 PASS ⇔ 高优先风险假设全部被**充分证据**证伪（REFUTED 且 `evidence_against` 非空，
  非"没查到"）AND 关键证据完整 AND 无未解决关键矛盾。
- **REJECT Gate**：可 REJECT ⇔ 高风险假设成立（高优先 SUPPORTED）AND 证据充分 AND 存在明确政策依据/先例
  AND `decision_confidence ≥ 0.7` AND 无关键矛盾。
- **HUMAN_REVIEW**（任一即转人工）：证据不足 / 关键证据冲突 / 政策无法确定 / 多个风险假设无法区分 /
  拟自动 REJECT 而 `decision_confidence < 0.7` / 关键 Tool 失败致证据缺失 / Budget Exhausted。
- R1 硬规则命中 → 强制 REJECT，覆盖一切（含 LLM PASS 提案）。

### 3.3 DECIDED 唯一终态与 decision 语义

- 图结构上 decide 后只有 `END`（§2.2）；**DECIDED 不是图里的节点/值**，而是 worker 在 `ainvoke` 返回后、
  读到 `decision` 非空时置 DB `review_run.status=DECIDED` 的运行态（03 T-8）。PASS/REJECT/HUMAN_REVIEW 全部
  写入 `ReviewDecision.decision`；预算耗尽/工具失败/降级/Gate 改判原因写入 `ReviewDecision.overrides`。
- `review_case.status` 等下游状态机由 worker/人工流程维护，不在子图内。

### 3.4 HUMAN_REVIEW 之后的图外流转（v1 不用 `interrupt()`）

```
decide 产出 decision=HUMAN_REVIEW
   → worker: review_run.status=DECIDED; decision/evidence/tool_call_history 落 MySQL（review_result/review_trace/review_evidence 表）
   → 投 MQ human_review（审核工作台队列）→ 人工裁决 → 回流（review_feedback → 案例库/策略库/评测集，00 §6.5/§9.3）
```

**v1 明确不用 LangGraph `interrupt()` 的理由**：
1. 本系统是**异步批量 Worker**：一条 case 从入队到裁决应在**单次 `ainvoke` 内跑完**并落库；
   `interrupt()` 会把图"挂起等外部输入"，要求 worker 长期持有线程 + 依赖外部在 `Command(resume=...)` 语义下
   唤醒，与"MQ 批量消费 + MySQL 状态机 + 幂等"的现架构耦合重、恢复链路复杂。
2. 人审发生在**图已产出 DECIDED 之后**，本质是下游队列/工作台流程，不是"调查中途需要人给数据"；
   Graph 的"中断点"语义（等人在中间步骤注入内容）在本案中不存在。
3. `interrupt()` 的价值场景是**实时交互式人审**（人在调查中给指示、图随后续跑）——留作未来方案，
   图结构不受影响（interrupt 位置不改变节点/边拓扑）。若将来要实时人审，只需在 decide 前后包一层
   带 interrupt 的"审查壳"，核心子图不动。

---

## 4. 边际证据增益记录（Marginal Evidence Gain）

### 4.1 谁写、何时写

| 字段（tool_call_history 元素） | 写入位置 | 写入方 |
|---|---|---|
| `before_confidence` / `after_confidence` | **tools_node 单次调用前后**（同一函数内先探针后执行） | tools_node（`guardrails/metrics.py` 探针） |
| `evidence_added` | 该次调用产出并经 `quality_filter/merge_evidence` 后实际新增的 evidence `E_nn` 列表 | tools_node |
| `decision_changed` | 同上（`gate_probe(before) != gate_probe(after)`） | tools_node |

> **O-10 已拍板**：边际增益 4 字段（before_confidence / after_confidence / evidence_added / decision_changed）
> 仅以 JSON 承载于 `tool_call_history` / `review_trace.output_json`，**DB `review_trace` 不加列**；二期如需 SQL 分析再加列。

### 4.2 代码级位置（tools_node 主循环）

```python
# src/pra/agent/tools_node.py 主循环骨架（伪代码级，具体落点见 §9）
# 每次调用记录 {seq, tool, args, result_ref, latency_ms, tokens, status,
#              before_confidence, after_confidence, evidence_added, decision_changed}
async def tools_node(state, config) -> dict:
    budget = state["budget"].model_copy()
    working = {**state, "budget": budget}          # 探测快照：本 visit 内只演进 evidence/budget
    records: list[dict] = []
    added_all: list[Evidence] = []
    seq = len(state["tool_call_history"]) + 1      # 调用审计 seq 自增

    for call in sorted(state["pending_tool_calls"], key=lambda c: c["priority"]):
        if budget_exceeded(budget) is not None:
            break                                   # Guardrail 截断（§6.4）
        before_state = dict(working)                # 调用前快照（探测用，浅拷贝即可）
        before = decision_conf_probe(before_state)  # ① 调用前 decision_confidence 代理
        outcome = await execute_one(call, budget)   # args 校验 → infra 重试 1 次 → 执行（§5.2）
        if not outcome["ok"]:
            # ② 错误：按 §5.2 记 failures（severity 依 call 是否 required=True），跳过本调用继续
            records.append(_error_record(seq, call, outcome))   # status="error"
            seq += 1
            continue
        budget = bump_tool_usage(budget)
        raw = outcome["raw_evidences"]              # 工具原始 Evidence（不过滤，§5.2 任务边界）
        evs = backfill_extra(quality_filter(raw))   # ③ EVIDENCE_MIN_SIM/STRONG 过滤 + extra 回填
        added = [e for e in evs if _evidence_key(e) not in {_evidence_key(x) for x in working["evidence"]}]
        if added:
            working["evidence"] = merge_evidence(working["evidence"], added)   # 只演进探测快照
        after = decision_conf_probe(working)        # ④ 调用后 decision_confidence 代理
        evi_ids = _assign_evidence_seqs(state, added)    # ⑤ E_nn 运行序号（DTO 无 evidence_id，§10 O-1）
        records.append({
            "seq": seq, "tool": call["tool"], "args": call["args"],
            "result_ref": evi_ids[0] if evi_ids else None,
            "latency_ms": outcome["latency_ms"], "tokens": 0, "status": "ok",
            "before_confidence": before, "after_confidence": after,
            "evidence_added": evi_ids,
            "decision_changed": gate_probe(before_state) != gate_probe(working),
        })
        added_all.extend(added)
        seq += 1

    return {"pending_tool_calls": [],               # 一次性消费（覆盖写）
            "evidence": added_all,                  # 只返回新增，state.py merge_evidence 去重合并（幂等）
            "tool_call_history": records,
            "budget": budget}
```

> 语义与限定（务必写进实现注释）：
> - `decision_conf_probe` / `gate_probe` 是 `guardrails/metrics.py` 里的**轻量纯函数**：前者对"当前证据集 +
>   已知 posterior"跑 §7.5 确定性公式得 decision_confidence 代理；后者用同一套 Gate 谓词对 state 快照探测
>   `PASS|REJECT|HUMAN|UNDECIDED`（`UNDECIDED`=证据不足尚无法通过任何自动 Gate）。**两者只用于边际增益审计，
>   不驱动路由**（路由只认 §2.4 谓词）。
> - `decision_changed = gate_probe(调用前快照) != gate_probe(调用后快照)`，即本次调用是否翻转 Gate 探测结论。
> - posterior 在 reevaluate 才刷新，因此 tools_node 内 before/after 差值主要反映"证据量/可引用/矛盾"变化，
>   精确"后验增益"由 reevaluate 步承担；探针可单测，不改变任何业务字段。

---

## 5. 错误与降级路径

### 5.1 LLM 节点 schema 校验失败（4 个 LLM 节点统一）

规则（01 §3.0）：`call_structured_llm` 内 pydantic 校验失败 → 携带错误重试 **1 次** → 仍失败返回 `None`，
由各节点壳统一处理：

| 场景 | 行为 |
|---|---|
| hypothesize 失败 | 返回 `{hypotheses: [], investigation_queue: [], degraded: True}` + append failure；plan 入口见 degraded → 短路（§5.3）→ decide → HUMAN_REVIEW（R5） |
| plan 失败 | `{pending_tool_calls: [], degraded: True}` + failure → `route_after_plan` ① → decide |
| reevaluate 失败 | `{degraded: True}`（hypotheses 不变，保留现有证据）→ `route_after_reevaluate` ① → decide |
| decide 失败 | `proposal=None` → overlay 兜底 HUMAN_REVIEW（无硬规则时） |

节点统一入口守卫：

```python
# guardrails/llm_shell.py 内嵌守卫（草稿）
async def _llm_node_guarded(state, config, *, OutputModel, node_name, on_failure):
    if state["degraded"] or budget_exceeded(state["budget"]) is not None:
        return on_failure(state)                      # 已降级/预算超限 → 不再调 LLM（短路）
    out = await call_structured_llm(OutputModel=OutputModel, state=state, config=config)
    if out is None:                                   # 重试 1 次仍失败
        f = {"step_type": node_name, "severity": "critical", "reason": "schema 校验重试仍失败",
             "ts": now_iso()}
        return {**on_failure(state), "degraded": True,
                "failures": [f],                       # append reducer，只返回新增
                "budget": bump_llm_usage(state["budget"])}
    # 成功：由各节点把结构化输出 apply 回 state（degraded=False）
```

### 5.2 Tool 抛错 / 返回 ok=False

策略：**记 failures → 决定"降级转 decide"还是"可跳过继续"**（任务边界要求在本设计里拍板，细化 01 口径）：

| 场景 | 判定 | 行为 |
|---|---|---|
| infra 瞬态错误（超时/连接，非业务） | 可重试 | ToolNode 内重试 **1 次**；仍失败走下行 |
| 业务 `ok=False` 或重试仍失败 | **非关键**（默认） | 记 `failures`：`{step_type:"TOOL_CALL", tool, severity:"warn", reason, ts}`；**跳过继续**执行本 visit 其余调用；plan 下一轮可见证据缺口（dedup 允许对 `error` 再试 1 次，01 §4.3） |
| 同上，但该调用是 **plan 标记 `required=True`** 的关键取证（如 IP 类案件必须的 ImageAnalysis，无替代工具） | **关键** | 记 `failures`：`severity:"critical"`；**不立即中断**——仍走完本 visit；overlay 侧 `key_tool_failure(state, failures)` 若发现"critical 失败且其后无同 tool 成功调用"→ R3_KEY_TOOL_FAILED → HUMAN_REVIEW（证据缺失，不自动判） |

```python
# guardrails/errors.py（草稿）：关键失败判定
def key_tool_failure(state, failures) -> bool:
    for f in failures:
        if f.get("step_type") == "TOOL_CALL" and f.get("severity") == "critical":
            # 该 tool 其后（含 dedup 允许的 1 次重试）是否有成功调用
            tool = f["tool"]
            later_ok = any(r["tool"] == tool and r["status"] == "ok"
                           for r in state["tool_call_history"] if r["seq"] > f["seq"])
            if not later_ok:
                return True
    return False
```

> **与 01 §7.2 的差异（O-2/O-3 已拍板，01 §7.2 R5 已同步）**：01 旧版把"`failures` 非空"一律记
> `R5_DEGRADED_OR_FAILED_STEP`；本文口径 —— **仅 LLM 步失败（critical，degraded）与未解决的 critical Tool 失败才触发
> HUMAN**（前者 R5、后者 R3_KEY_TOOL_FAILED）；`severity:"warn"` 的 Tool 失败只进审计与 trace，不强制转人工
> （否则任何一次非关键工具抖动都会人为推高 Human Review Rate）。

### 5.3 降级短路（plan/reevaluate）

plan/reevaluate 入口若 `degraded=True` 或预算已超 → 直接返回**最小更新**（不动推理字段、不调 LLM）：
`plan` 返回 `{"pending_tool_calls": []}`，`reevaluate` 返回 `{}`；随后条件边①读 `degraded=True` → decide。
decide 永远执行（overlay），保证图有且只有一个出口（§6）。

---

## 6. 终止性论证（一页）

**断言**：从任意合法初始 state 出发，图必然在有限步内到达 `decide → END`，且每次运行 LLM 调用 ≤ 10、
Tool 调用 ≤ 15、墙钟 ≤ 30s（Guardrail 上界）。

论证（按结构分解）：

1. **回环唯一且收敛受控**：唯一的回环是 `plan → tools → reevaluate → plan`。离开回环只有两个条件出口
   （都在 `route_after_reevaluate`）：`is_converged(state)=True`（早停）或超限；plan 侧另有 `conclude`
   出口（`route_after_plan` ④）与 dedup 截断（01 §4.3：重复动作清空 → 视同 conclude）。**decide 无出边**。
2. **每轮回环必须消耗预算**：一轮完整回环至少触发 1 次 `plan` LLM + 1 次 `reevaluate` LLM（每次 +1 llm_calls）；
   若 tools 执行则另 +tool_calls。于是回环轮数 `R` 满足 `R ≤ floor((max_llm_calls − hypothesize_1次 − decide_≤1次) / 2)`
   = 由 01/03 拍板 `max_llm_calls=10` 得 **R ≤ 4**（理想上界；实际常因 is_converged/conclude 早停更少）。
3. **不消耗预算的反复也不会死循环**：若 plan 反复提议同一调用 → dedup guardrail 第 2 次即清空并视同
   `conclude` → `route_after_plan` ④ → decide；若 plan 每次提议**不同**工具 → 工具只有 6 个且
   `max_tool_calls=15` 兜底；若 plan/reevaluate 因 degraded 短路 → 直接 decide。
4. **预算四维全有硬上界**：`budget_exceeded` 在**每次条件边路由、每个节点入口**都检查（§2.4），任一维度
   超限 → decide → overlay 产出带部分证据的 HUMAN_REVIEW（overrides=R3_BUDGET_EXHAUSTED）。LLM/Tool 上限
   之外，tokens ≤ 40k、墙钟 ≤ 30s（latency 从 `budget.start_time` 计时）。
5. **确定性保证**：路由/预算/收敛/Gate 全为纯函数（无随机、无 LLM），同 state 必同后继；故不存在
   "同样的 state 反复走不同分支"的非确定性死循环。
6. 步数上界（粗算）：1 (hypothesize) + ≤4×(1 plan + ≤N tools 次调用 + 1 reevaluate) + ≤1 (decide)；
   总节点执行步数 ≤ ~20，其中 LLM ≤10、Tool ≤15——与 §2.4 Guardrail 上限一致。

---

## 7. Checkpointer 选型（研究结论）

### 7.1 事实（本机实测 + 官方生态）

- 已装：`langgraph 1.2.11` + `langgraph-checkpoint 4.2.0`（核心，含 `InMemorySaver`）。
  `langgraph-checkpoint-sqlite`、`langgraph-checkpoint-postgres` **未安装**（官方可选包）。
- 官方 checkpoint 后端（LangGraph 1.x 文档）：`InMemorySaver`（`langgraph.checkpoint.memory`，仅调试/测试）、
  `SqliteSaver`/`AsyncSqliteSaver`（`langgraph-checkpoint-sqlite`）、`PostgresSaver`/`AsyncPostgresSaver`
  （`langgraph-checkpoint-postgres`，面向生产）。
- **LangGraph 官方没有 MySQL saver**。自研需继承 `BaseCheckpointSaver`，实现（实测其方法清单）
  `get_tuple/aget_tuple, put/aput, put_writes/aput_writes, list/alist` 等 sync+async 双轨方法，
  并理解其内部 checkpoint 序列化/版本（channel deltas）语义——**随 langgraph-checkpoint 版本演进维护成本高**。
- **Pydantic 状态序列化（实测结论）**：默认 `JsonPlusSerializer` 能序列化 `pra.domain.models` 的 Pydantic
  对象，但会告警 `Deserializing unregistered type ... will be blocked in a future version`；
  通过 `JsonPlusSerializer(allowed_msgpack_modules=[("pra.domain.models", "Budget"), ...])` 显式注册模块
  可消除告警（已实测无警告、断点续跑正常）。未来严格模式（`LANGGRAPH_STRICT_MSGPACK`）下 allowlist 是正道。
  参考：LangGraph persistence 文档
  （https://langchain-ai.github.io/langgraph/concepts/persistence/ ，
  https://langchain-ai.github.io/langgraph/how-tos/persistence/ ）；
  BaseCheckpointSaver API（https://langchain-ai.github.io/langgraph/reference/checkpoints/ ）。

### 7.2 选型表

| 选项 | 做法 | 优点 | 缺点 | 结论 |
|---|---|---|---|---|
| **A（推荐）** | **业务状态自建表落 MySQL**（`review_run`/`review_trace`/`review_evidence`/`review_result`，已落地 migration 001）；**线程状态（checkpoint）用 InMemorySaver（MVP）/ AsyncSqliteSaver（本地与集成）/ 未来 PostgresSaver 或自研** | ① trace/eval/审计/申诉依据是 MySQL 业务表（review_trace 逐 step 记 tokens/latency 与步骤 input/output 摘要；线程中间 state 全量快照由 checkpointer 承担）——checkpointer 只服务"断点续跑/重放"；② 线程 checkpoint 与业务 schema 解耦，langgraph-checkpoint 升级不阻塞业务表；③ 契合团队"MySQL + SQLAlchemy async"栈，不引入第二数据库 | 崩溃恢复粒度=checkpoint 线程；若线程 checkpoint 在内存则断点仅对当次进程有效——生产需把线程 checkpoint 落到 Sqlite/Postgres（或自研） | **采纳** |
| B | 自研 MySQL saver（继承 `BaseCheckpointSaver`） | 单库单技术栈 | 需维护 checkpoint 内部格式/版本/并发写入，随 langgraph-checkpoint 升级持续跟进；成本高、收益低（业务表已在 A 承担） | 不推荐 v1；作为二期可选（把 checkpoint 表并进 MySQL） |
| C | 业务状态也换 Postgres（`PostgresSaver` + SQLAlchemy/PG） | 官方生产级 saver 开箱 | 推翻 MySQL 选型（00 §9/§15、pyproject `aiomysql`），改动面大 | 不采用 |

**推荐 A + 理由（写入实现注释）**：`review_run/review_trace/review_evidence/review_result` 才是 trace/eval/审计的真相，
checkpointer 的职责仅是"把 LangGraph 线程（thread_id=run）的中间 state 可恢复"，二者分离让"业务持久化稳定"
与"框架持久化自由演进"互不拖累。MVP/联调用 `InMemorySaver`；本地/集成测试升级 `AsyncSqliteSaver`（需装
`langgraph-checkpoint-sqlite`）；生产评估 `PostgresSaver` 或自研 MySQL saver（B）后再定。

### 7.3 代码落点（`pra/agent/checkpointer.py`）

```python
# pra/agent/checkpointer.py（草稿）
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

ALLOWED_MSG_PACK_MODULES = [
    ("pra.domain.models", "Budget"), ("pra.domain.models", "BudgetLimits"),
    ("pra.domain.models", "Evidence"), ("pra.domain.models", "Hypothesis"),
    ("pra.domain.models", "HypothesisStatus"), ("pra.domain.models", "Decision"),
    ("pra.domain.models", "RiskLevel"), ("pra.domain.models", "RiskType"),
    ("pra.domain.models", "ProductReviewCase"), ("pra.domain.models", "ProductInfo"),
    ("pra.domain.models", "ProductImage"), ("pra.domain.models", "SkuInfo"),
    ("pra.domain.models", "ScreeningSignal"), ("pra.domain.models", "ReviewDecision"),
]

def make_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSG_PACK_MODULES)

def make_memory_checkpointer():
    return InMemorySaver(serde=make_serde())          # MVP / 联调

# 未来：
# def make_sqlite_checkpointer(path=":memory:"): from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver; ...
# def make_postgres_checkpointer(dsn): from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver; ...
```

### 7.4 文档一致性注记（00/01 措辞修订建议，本轮不擅改 00/01/03）

- 00 §3.1"由 LangGraph Checkpointer（MySQL）每步后落库"、00 §4.2 草图 `compile(checkpointer=mysql_checkpointer)`、
  01 §1.2/§6.1 `checkpointer=mysql_checkpointer`、01 §9 与 03 §4.2 关于 "MySQL Checkpointer" 的表述，
  **最终口径**：`compile(checkpointer=<线程状态 saver，见 §7.2 选型 A>)`；MySQL 承载业务表 review_run/review_trace/review_evidence/review_result——逐步 token·latency（run 级汇总）由 review_trace 承载、终局 budget_used/overrides 随 `review_result.decision_json` 落库（worker 层显式落库，不依赖 langgraph checkpoint 写 MySQL）；线程中间 state 的恢复靠 checkpointer。
  建议 00/01 后续修订为同一口径（列于 03 §7 遗留/待办）。

---

## 8. MVP 裁剪

**第一版最小闭环路径（必须首先跑通）**：`hypothesize → plan → tools → reevaluate → … → decide → DECIDED`
（复古运动鞋案例，00 §4.4 走查），即**只实现一条 happy path + 三条收尾路径**（plan conclude / converged /
budget 超限各走向 decide）。

| 维度 | MVP 做法 | 后补 |
|---|---|---|
| Tools | 已就绪：6 个 Tool 默认 InMemory/Mock 数据源（`build_tools()`），只跑 ImageAnalysis/Product/Merchant/Case/Policy 五种命中组合 | 真数据源接入（infra 层 Repository/Provider） |
| Checkpointer | `InMemorySaver`（含 pydantic allowlist serde，§7.3） | Sqlite/Postgres/自研 MySQL saver |
| LLM | 节点经 `call_structured_llm` 走 litellm；**测试用确定性 JSON 桩**（同输入同输出，保证 eval 可重放） | Langfuse 埋点 |
| 路由/Guardrails | 只实现：budget 4 维 + is_converged + dedup + 三 Gate overlay + hard_rules 最小黑名单 | 更多硬规则、矛盾启发式扩充 |
| 边际增益 | tools_node 内实现 4 字段（§4）；**先行单测**（探针是纯函数） | 聚合指标仪表（00 §11.3） |
| 人审流转 | 不实现：decide 后由 worker 打印/记录 decision 即可 | MQ human_review + 工作台 + 回流 |
| 持久化 | 不实现 DB 落库（用 checkpointer + 日志观察 state） | review_run/review_trace/review_evidence/review_result 落 MySQL |
| 观测 | 预留 trace hook 点（tools_node/LLM 壳内注释位），不接线 | Langfuse / OTel |
| screening 接入 | 子图外（输入即为 case），先不接 | 机审 → 分流 → worker 消费接线 |

---

## 9. 实现落点清单（每文件职责 + 主要函数签名）

| 文件 | 职责 | 主要函数/符号 |
|---|---|---|
| `pra/agent/state.py` | AgentState + reducer（按 §2.3 补 `Annotated`）+ 初始状态构造 | `merge_evidence(left,right)`；`AgentState(TypedDict)`；`build_initial_state(case)`；`_evidence_key(e)` |
| `pra/agent/graph.py` | 图装配、条件边路由、编译 | `build_agent_graph(*, tools=None, checkpointer=None) -> CompiledStateGraph`；`route_after_plan(state)`；`route_after_reevaluate(state)`；节点名常量 |
| `pra/agent/nodes/hypothesize.py` | 假设生成（LLM→HypothesizeOutput→hypotheses/queue） | `async hypothesize_node(state, config) -> dict`；内部 `apply_hypothesize(...)` |
| `pra/agent/nodes/plan.py` | 调查计划（LLM→PlanOutput→pending_tool_calls，经 dedup） | `async plan_node(state, config) -> dict` |
| `pra/agent/nodes/reevaluate.py` | 证据综合（LLM→ReevaluateOutput→hypotheses/queue 更新） | `async reevaluate_node(state, config) -> dict` |
| `pra/agent/nodes/decide.py` | 提案 + overlay 收口 | `async decide_node(state, config) -> dict` |
| `pra/agent/tools_node.py` | 执行 pending_tool_calls（预算/校验/重试/过滤/转证据/记账/边际增益） | `make_tools_node(tools) -> Callable`（闭包工厂：注册私有 ToolRegistry 后返回 `async tools_node(state, config) -> dict`；tools 经 `build_agent_graph(tools=...)` 注入） |
| `pra/agent/guardrails/budget.py` | 预算四维检查与记账 | `budget_exceeded(budget)`；`bump_llm_usage`；`bump_tool_usage`；`snapshot_budget` |
| `pra/agent/guardrails/converge.py` | 收敛判定 | `is_converged(state)` |
| `pra/agent/guardrails/decision_guardrail.py` | 三 Gate overlay 与 decision 组装 | `run_decision_overlay`；`pass_gate`；`reject_gate`；`finalize_decision_confidence`；`build_decision` |
| `pra/agent/guardrails/hard_rules.py` | R1 硬规则 | `hard_rule_hit(state) -> HardRuleHit | None` |
| `pra/agent/guardrails/dedup.py` | plan 输出去重 | `dedup_pending(state, planned) -> list[dict]` |
| `pra/agent/guardrails/metrics.py` | 边际增益探针（纯函数） | `decision_conf_probe(state)`；`gate_probe(state)` |
| `pra/agent/guardrails/errors.py` | 错误分级与关键失败判定 | `key_tool_failure(state, failures)` |
| `pra/agent/guardrails/llm_shell.py` | LLM 结构化调用壳（重试 1 次/记账/降级短路） | `call_structured_llm(*, OutputModel, state, config)`；`_llm_node_guarded(...)` |
| `pra/agent/guardrails/evidence.py` | 证据质量过滤与 extra 回填 | `quality_filter(raw, *, evid_min_sim=0.70)`；`backfill_extra(evs, *, case=None)`（Strong 档非过滤参数：由 `backfill_extra` 写 `extra.strong` 键、gate 谓词消费） |
| `pra/agent/checkpointer.py` | saver/serde 工厂 | `make_serde()`；`make_memory_checkpointer()`；（未来 `make_sqlite/postgres`） |
| `pra/tools/base.py`（O-5 已拍板落地） | Tool 暴露 args schema 供 tools_node 解析 | ✅ `Tool.args_model: type[ToolArgs]` + `ToolRegistry.parse_args(name, raw)`（tools_node 校验/解析入口，见 §5/§10 O-5） |

> 依赖注入提示：tools 与 registry 通过 `build_agent_graph(tools=...)` 注入 tools_node 可访问的 provider
> （闭包工厂或 `graph.py` 内构造 ToolNode 闭包），避免模块级单例，便于 pytest mock（00 §15.1 依赖倒置）。

---

## 10. 开放问题（O-1~O-10 —— 均已拍板，本表为决策记录；实现按"拍板结果"列执行）

| ID | 问题 | 现状矛盾/出处 | 拍板结果（O-1~O-10 已拍板，2025 用户决策） |
|---|---|---|---|
| O-1 | **evidence 去重 key** | 01 §2.5 / 03 §4.2 漂移 3 写 `(type, source, ref_id)`；但 ImageAnalysis 对同一品牌、不同图片会产出多条 `ref_id=None` 且 value 不同的证据——纯 `(type,source,ref_id)` 会把第二条丢弃（信息丢失）。落地代码 DTO 无 `evidence_id`，无法在 key 里用稳定业务 id | 已拍板：evidence 去重防丢 —— ref 优先稳定业务标识（image_url / product_id / merchant_id；RAG 工具 case_id / clause_id），无稳定 ref 时去重 key 回退 value。落地：6 个工具 to_evidence 已填稳定 ref_id（工具层不写 extra，见 O-8）；去重口径已同步 01 §2.5 / 03 §4.2 / 本文 §2.3（`_evidence_key` 见 §2.3） |
| O-2 | **failures 触发 HUMAN 的口径** | 01 §7.2 overlay 写"`failures` 非空一律 R5→HUMAN"；本设计 §5.2 细化：仅 LLM 步失败(critical)与未解决 critical Tool 失败触发，warn 只审计 | 已拍板：failures 非空不一律 HUMAN_REVIEW —— 仅 LLM 步失败（degraded，critical）与未解决的 critical Tool 失败（R3_KEY_TOOL_FAILED）触发；`severity:"warn"` 只审计。01 §7.2 R5 / §7.3 已同步 |
| O-3 | **failures 元素字段扩展** | 01 §2.4 定义 `{step_type, reason, ts}`；本设计需 `{step_type, tool?, severity: warn/critical, reason, ts}`（dedup/tools_node 要写 severity 与 tool） | 已拍板：failures 元素 = `{step_type, tool?, severity: "warn"|"critical", reason, ts}`（01 §2.4 / 03 §2.7 / state.py 注释已同步） |
| O-4 | **ToolContext 缺 case/版本** | 落地 `product/tool.py` 注释已提出：`version_drift`（库中 version vs case.product.version）需在工具内比对，但 `ToolContext` 只有 run_id/case_id/budget、不带 case | 已拍板：version_drift 比对放 tools_node evidence processing 层（其持有 case 快照），**不扩 ToolContext**（tools_node 的 backfill_extra 实现时落位） |
| O-5 | **Tool args 解析模型缺失** | `ToolRegistry` 只注册 name→Tool，无 `args_model`；plan 给的是 dict，`tool.call` 期望 `ToolArgs` 子类实例 | 已拍板：Tool 增 `args_model: type[ToolArgs]`，6 个工具声明各自 Args 子类；`ToolRegistry.parse_args(tool_name, raw)` 校验/解析（tools/base.py 已落地，01 §5.8 已同步） |
| O-6 | **run_id 来源** | 落地 state.py 已把 run_id/case_id 迁出为 thread_id；节点/工具需要 run_id（ToolContext 必填） | 已拍板：graph 层 `thread_id = run_id`；节点从 `config["configurable"]["thread_id"]` 读 run_id（见 §2.3 invoke 约定与 §9 节点签名注释） |
| O-7 | **decision_confidence 字段名** | 语义=decision_confidence，但 DTO/代码字段名是 `ReviewDecision.confidence` | 已拍板：`ReviewDecision.confidence` → `decision_confidence`（models.py 已改名；00 §2.2/§7/§9.1、01 §7.5/§7.6/§8、03 T-4 已同步；DB `decision` 列同口径） |
| O-8 | **Evidence.extra 填充时机** | 落地 `image_analysis/tool.py::to_evidence` 未填 extra（任务边界：工具不裁决）；矛盾检测（03 T-4(e)）读 `extra.similarity` | 已拍板：Evidence.extra 派生数值（similarity / version_drift 等）由 tools_node `backfill_extra` 回填，工具只给原始事实 —— 工具代码不含 extra（已如此，§4/§9 evidence.py 实现时落位） |
| O-9 | **Checkpointer 严格序列化** | 默认 JsonPlus 对 pydantic 状态会告警（未来阻塞）；已实测 allowlist 消除 | 已拍板：保留 §7.3 allowlist serde 方案（MVP InMemorySaver + JsonPlusSerializer allowlist；升级 langgraph 后回归一次含 STRICT_MSGPACK 预检） |
| O-10 | **review_trace/DB 无边际增益列** | 边际增益 4 字段在 tool_call_history/review_trace.output_json（JSON），DB 无专列 | 已拍板：边际增益 4 字段仅 JSON 承载（tool_call_history / review_trace.output_json），DB 不加列（§4.1 注记） |

---

> 结论一句话：**5 节点、7 条边（2 条条件边）的单回环子图**，decide 是唯一出口；MVP 用 InMemorySaver +
> pydantic allowlist serde，业务真相走 MySQL 自建表；三 Gate 全在 decide 的确定性 overlay 内，LLM 只提案。
