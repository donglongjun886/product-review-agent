"""AgentState 类型定义（TypedDict + reducer）—— 调查 Agent 的状态契约。"""

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
    """证据去重指纹：``(type, source, ref_id)``；``ref_id`` 为 None 时用 ``value``。"""
    return (e.type, e.source, e.ref_id if e.ref_id is not None else e.value)


def merge_evidence(left: list[Evidence], right: list[Evidence]) -> list[Evidence]:
    """evidence channel reducer：合并 ``right`` 到 ``left``，按 ``_evidence_key`` 去重。

    同 key 丢弃新增、新 key 追加；``left``/``right`` 可为空或 None。
    """
    seen: dict[tuple, Evidence] = {_evidence_key(e): e for e in left or []}
    out: list[Evidence] = list(left or [])
    for e in right or []:
        if _evidence_key(e) not in seen:
            seen[_evidence_key(e)] = e
            out.append(e)
    return out


class AgentState(TypedDict, total=False):
    """LangGraph State 状态契约：字段值为可 JSON 序列化的 domain 模型或原始 dict；``total=False``（全部可选）。"""

    case: ProductReviewCase  # 输入商品事实快照
    hypotheses: list[Hypothesis]  # 风险假设清单
    evidence: Annotated[list[Evidence], merge_evidence]  # 已收集证据（去重合并 reducer）
    tool_call_history: Annotated[list[dict], add]  # 工具调用审计记录（append）
    budget: Budget  # 已用 + 限额（覆盖写）
    decision: ReviewDecision | None  # 收敛后的裁决；调查中为 None

    # ---- 图内通道 ----
    pending_tool_calls: list[dict]  # 待执行工具调用（元素 {tool, args, reason, priority}）；消费后置 []
    degraded: bool  # 上一 LLM 步 schema 校验失败降级标记
    failures: Annotated[list[dict], add]  # 步骤失败审计（元素 {step_type, severity, reason}）
    measurement_capabilities: dict[str, bool]  # 测量环境能力（维度 → 是否可测）


def build_initial_state(case: ProductReviewCase) -> AgentState:
    """每次 invoke 的完整初始输入：所有 channel 有值，``budget`` 取 ``BudgetLimits`` 默认值。"""
    return AgentState(
        case=case,
        hypotheses=[],
        evidence=[],
        tool_call_history=[],
        budget=Budget(),
        decision=None,
        pending_tool_calls=[],
        degraded=False,
        failures=[],
        measurement_capabilities={},
    )
