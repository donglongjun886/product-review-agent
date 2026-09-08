"""AgentState 类型定义（TypedDict）—— 复杂风险调查 Agent 的状态契约。

AgentState 即 LangGraph 的 State（§3.1 设计要点 2 / §4.2 草图）：以 TypedDict 承载
调查主线的**显式、可序列化、可持久化、可恢复**记忆 —— 而不是藏在 LLM 上下文里。
LangGraph 的**线程 State Checkpointer**（MVP 用 InMemorySaver；Sqlite/Postgres/自研
见 docs/04-graph-design.md §7 选型 A）每步保存线程中间状态用于断点续跑/eval 重放；
业务真相（agent_step / evidence / decision 等）由 worker 层显式落 MySQL（非 checkpointer
职责，04 §7.4 口径）。本文件只定契约，接线在 graph.py / checkpointer.py 阶段。

案件身份（run_id / case_id）**不放进本 State**：接入层映射为 LangGraph 的
**thread 维度（thread_id = run_id，O-6 拍板）**，由 Checkpointer 与调用方携带；
节点/工具需 run_id 时从节点第二参 ``config["configurable"]["thread_id"]`` 读取。
（后续 graph.py / checkpointer.py 接线时再落地该映射，本文件只定契约。）

字段即 §4.1 Loop 每一环的产物：
- ``case``:            输入事实快照（起点，调查过程中不变；§2.1 ProductReviewCase）
- ``hypotheses``:      假设验证器状态（prior→posterior→status，§3.1 要点 1）
- ``evidence``:        结论依据（去重、合并后的证据链；与 tool_call_history 分离，§3.1 要点 4）
- ``investigation_queue``: 待验证问题队列（按优先级排序）
- ``tool_call_history``:   过程审计（谁、何时、调了哪个工具、tokens/latency）
- ``budget``:          成本/延迟硬字段 —— 条件边路由函数每轮进入节点前检查，超限转人工（§3.1 要点 3 / §8.1）
- ``decision``:        收敛后写入的 ReviewDecision（调查中为 None）
- 图内通道（T-9 拍板保留，docs/03-decisions.md §2.7；01 §2.1 标〔细化新增〕）：
  ``pending_tool_calls`` 承载 plan→tools 的"本轮待执行工具调用"传递（覆盖写）；
  ``degraded`` 承载"上一 LLM 步 schema 校验失败"的降级信号（覆盖写）；
  ``failures`` 为步骤失败审计（append reducer；元素含 tool?/severity，O-3 拍板，见
  03-decisions §2.7 注记）——三者是让第 3~7 章契约可落地的图内部通道，不进最终决策
  输出；``run_id/case_id`` 按代码方案迁出为 thread 维度。

reducer 说明：LangGraph 中 list 类字段跨步**追加**（而非覆盖）需在装配阶段用
``Annotated[list[...], reducer]`` 声明；本文件先保留纯 TypedDict 形态（字段级
reducer 语义见 03-decisions §2.7 与 01 §2.5：``evidence``=自定义去重合并（key 规则
见 01 §2.5，O-1 拍板）、``tool_call_history``/``failures``=append、
``pending_tool_calls``/``hypotheses``/``investigation_queue``/``budget``/``degraded``/
``decision``=覆盖写），待 graph.py 落地时再挂 reducer，避免过早耦合框架细节。
"""

from __future__ import annotations

from typing import TypedDict

from pra.domain import (
    Budget,
    Evidence,
    Hypothesis,
    ProductReviewCase,
    ReviewDecision,
)


class AgentState(TypedDict):
    """LangGraph State 的状态契约（§4.2 Python 草图，字段类型提升为 domain 模型）。

    值均为可 JSON 序列化的 domain 模型 / 原始 dict，供 Checkpointer 落库与 eval 重放。
    """

    case: ProductReviewCase  # 输入商品事实快照（调查起点，不改写）
    hypotheses: list[Hypothesis]  # 风险假设（可解释的 prior→posterior→status 演变）
    evidence: list[Evidence]  # 已收集证据（结论依据，去重/合并）
    investigation_queue: list[dict]  # 待验证问题，如 {"q", "priority", "status"}
    tool_call_history: list[dict]  # 调用审计，如 {"seq", "tool", "args", "result_ref", "latency_ms", "tokens"}
    budget: Budget  # 已用 + 限额（§8.1）；条件边路由的确定性检查对象
    decision: ReviewDecision | None  # 收敛后的裁决；调查中为 None

    # ---- 图内通道（T-9 拍板保留；reducer 接线在 graph 阶段落地）----
    pending_tool_calls: list[dict]  # plan 写、tools 消费后置 []；覆盖写；元素 {tool, args, reason, priority}
    degraded: bool  # 上一 LLM 步 schema 校验失败降级标记；覆盖写（True 后不再调 LLM，透传 decide）
    failures: list[dict]  # 步骤失败审计 {step_type, tool?, severity: "warn"|"critical", reason, ts}（O-3）；append reducer

    # 待实现（graph.py / checkpointer.py 阶段）：reducer 接线（append/merge 语义见模块
    # docstring）、run_id/case_id → thread_id 映射、State 序列化落库。
