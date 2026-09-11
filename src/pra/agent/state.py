"""AgentState 类型定义（TypedDict + reducer）—— 复杂风险调查 Agent 的状态契约。

AgentState 即 LangGraph 的 State：以 TypedDict 承载调查主线的**显式、可序列化、
可持久化、可恢复**记忆，而不是藏在 LLM 上下文里。线程 Checkpointer（MVP 用
InMemorySaver）每步保存线程中间状态用于断点续跑 / eval 重放；业务真相
（review_trace / review_evidence / review_result 等）由 worker 层显式落 MySQL ——
checkpointer 不写业务表。每次 invoke 一律走 ``build_initial_state``，保证每个
channel 首读有值、reducer 首写安全。

案件身份（run_id / case_id）**不放进本 State**：接入层映射为 thread 维度
（thread_id = run_id），由 Checkpointer 与调用方携带；节点/工具需 run_id 时从节点第二参
``config["configurable"]["thread_id"]`` 读取。

字段即调查循环每一环的产物，各 channel 的语义与 reducer 见 ``AgentState`` 的字段注释。
要点：``budget`` 是条件边路由每轮进节点前检查的硬限额，超限转人工；
``pending_tool_calls`` / ``degraded`` / ``failures`` 是图内通道，不进最终决策输出。

reducer 只服务于追加/去重幂等两类需求：``evidence`` 走自定义 merge（ref_id 优先，无则
回退 value）；``tool_call_history`` / ``failures`` 走 append；其余字段均为覆盖写（单路径
线性链上每步只有一个合法写方），断点重放 / super-step 重跑不产生重复记录。
"""

from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict

from pra.domain import (
    Budget,
    Evidence,
    Hypothesis,
    ProductReviewCase,
    ReviewDecision,
)


def _evidence_key(e: Evidence) -> tuple:
    """证据去重指纹：ref_id 优先稳定业务标识（image_url / product_id / merchant_id /
    case_id / clause_id），为 None 时回退 value —— 防同 (type, source) 的多条无 ref
    证据（如多图多品牌命中）互相吞并。"""
    return (e.type, e.source, e.ref_id if e.ref_id is not None else e.value)


def merge_evidence(left: list[Evidence], right: list[Evidence]) -> list[Evidence]:
    """evidence channel reducer：按 _evidence_key 去重合并（左=state，右=节点新增）。

    已存在同 key → 丢弃新增（证据不可篡改，重放/重试幂等）；新 key → append；
    left 可能为空/None（首写或空输入）。
    """
    seen: dict[tuple, Evidence] = {_evidence_key(e): e for e in left or []}
    out: list[Evidence] = list(left or [])
    for e in right or []:
        if _evidence_key(e) not in seen:
            seen[_evidence_key(e)] = e
            out.append(e)
    return out


class AgentState(TypedDict, total=False):
    """LangGraph State 的状态契约（字段类型为 domain 模型）。

    值均为可 JSON 序列化的 domain 模型 / 原始 dict，供 Checkpointer 落库与 eval 重放。
    ``total=False``：所有 channel 可选，首读依赖 ``build_initial_state`` 全量初始化。
    """

    case: ProductReviewCase  # 输入商品事实快照（调查起点，不改写）
    hypotheses: list[Hypothesis]  # 风险假设（prior→posterior→status 演变）
    evidence: Annotated[list[Evidence], merge_evidence]  # 已收集证据（自定义去重合并 reducer）
    investigation_queue: list[dict]  # 待验证问题，如 {"q", "priority", "status"}
    tool_call_history: Annotated[list[dict], add]  # 调用审计（append；含边际增益 4 字段）
    budget: Budget  # 已用 + 限额；条件边路由的确定性检查对象（覆盖写，整对象）
    decision: ReviewDecision | None  # 收敛后的裁决；调查中为 None

    # ---- 图内通道 ----
    pending_tool_calls: list[dict]  # plan 写、tools 消费后置 []；元素 {tool, args, reason, priority}
    degraded: bool  # 上一 LLM 步 schema 校验失败降级标记；覆盖写（True 后不再调 LLM）
    failures: Annotated[list[dict], add]  # 步骤失败审计 {step_type, tool?, severity, reason, ts}


def build_initial_state(case: ProductReviewCase) -> AgentState:
    """每次 invoke 的完整初始输入。

    保证所有 channel 有值（首节点读不炸、reducer 首写安全）；``budget`` 取
    BudgetLimits 默认值 10/15/40000/30000（接线时勿再覆盖回 8/12）。
    """
    return AgentState(
        case=case,
        hypotheses=[],
        evidence=[],
        investigation_queue=[],
        tool_call_history=[],
        budget=Budget(),
        decision=None,
        pending_tool_calls=[],
        degraded=False,
        failures=[],
    )
