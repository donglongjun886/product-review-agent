"""AgentState 类型定义（TypedDict）—— 复杂风险调查 Agent 的状态契约。

AgentState 即 LangGraph 的 State（§3.1 设计要点 2 / §4.2 草图）：以 TypedDict 承载
调查主线的**显式、可序列化、可持久化、可恢复**记忆 —— 而不是藏在 LLM 上下文里。
LangGraph 的 **Checkpointer** 在每步执行后把整个 State 落库（MySQL，见
``checkpointer.py``，待实现）：worker 崩溃可恢复、断点可续跑、eval 可重放。

案件身份（run_id / case_id）**不放进本 State**：§3 JSON 里的 ``run_id`` / ``case_id``
在接入层映射为 LangGraph 的 **thread 维度（thread_id = run/案件）**，由 Checkpointer
与调用方携带 —— 状态体只装"调查记忆"，身份归线程键，避免双份冗余与不一致。
（后续 graph.py / checkpointer.py 接线时再落地该映射，本文件只定契约。）

字段即 §4.1 Loop 每一环的产物：
- ``case``:            输入事实快照（起点，调查过程中不变；§2.1 ProductReviewCase）
- ``hypotheses``:      假设验证器状态（prior→posterior→status，§3.1 要点 1）
- ``evidence``:        结论依据（去重、合并后的证据链；与 tool_call_history 分离，§3.1 要点 4）
- ``investigation_queue``: 待验证问题队列（按优先级排序）
- ``tool_call_history``:   过程审计（谁、何时、调了哪个工具、tokens/latency）
- ``budget``:          成本/延迟硬字段 —— 条件边路由函数每轮进入节点前检查，超限转人工（§3.1 要点 3 / §8.1）
- ``decision``:        收敛后写入的 ReviewDecision（调查中为 None）

reducer 说明：LangGraph 中 list 类字段跨步**追加**（而非覆盖）需在装配阶段用
``Annotated[list[...], reducer]`` 声明；本文件先保留与 §4.2 草图一致的纯 TypedDict
形态，待 graph.py 落地时再按各节点语义挂 reducer，避免过早耦合框架细节。
"""

from __future__ import annotations

from typing import TypedDict

from cg.domain import (
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

    # 待实现（graph.py / checkpointer.py 阶段）：reducer 接线（追加式更新 hypothesis /
    # evidence / queue / tool_call_history）、run_id/case_id → thread_id 映射、State 序列化落库。
