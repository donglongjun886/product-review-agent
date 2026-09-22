"""reevaluate 节点：把本轮新证据综合进假设状态（"LLM 提案 + 确定性 apply"两层）。

1. 依据 ``state=`` 中**已给的 evidence**（全量字段）更新各条 hypothesis 的
   posterior/status/evidence_for/evidence_against。status 只允许 SUPPORTED / REFUTED /
   UNRESOLVED —— **证据不足置 UNRESOLVED**（"查了没结论" ≠ "证伪"；PENDING 只留给新假设）；
   禁止臆造未出现的事实/数值/来源。
2. **新发现的风险维度**走 ``new_hypotheses``（hypothesize 不重跑），apply 追加为
   status=PENDING、posterior=None 的新假设，id 从现用最大 H 序号 +1 续号。
3. ``conflicts`` / ``evidence_sufficiency`` **仅供审计/参考**：AgentState 无对应 channel。

契约要点：
- hypotheses 为**覆盖写** → 成功路径返回**全集**；evidence 走
  merge reducer、tool_call_history/failures 走 append —— 本节点不动它们。
- 入口短路：``state["degraded"]`` 或 ``budget_exceeded(state["budget"])`` 非 None →
  **不调 LLM**，返回 ``{}``（不动推理字段、不动 degraded）。
- LLM 记账：成功/失败均按 ``outcome.attempts`` 次数 bump。
- LLM 失败（outcome.model=None）→ 返回 degraded=True + 一条 critical failure（reason 取
  固定字面值 ``_DEGRADE_REASON``，不拼 outcome.error）+ bump 后 budget，**不动
  hypotheses**（degraded=True 后路由直接进 decide，overlay 按降级归因转人工）。
"""

from __future__ import annotations

import re

from pra.agent.guardrails.budget import budget_exceeded, bump_llm_usage
from pra.agent.guardrails.errors import SEV_CRITICAL, STEP_REEVALUATE, make_failure
from pra.agent.guardrails.llm_shell import LLMBackend, call_structured_llm
from pra.agent.guardrails.schemas import ReevaluateOutput
from pra.domain.models import Hypothesis, HypothesisStatus

__all__ = ["reevaluate_node"]

# 假设 id 序号匹配：H1..Hn（hypothesize 生成 / reevaluate.new_hypotheses 续号共用）。
_H_ID_RE = re.compile(r"^H(\d+)$")

# LLM 步降级 failure 文案：与 hypothesize/plan 对齐（固定字面值、不拼 outcome.error）。
_DEGRADE_REASON = "schema 校验重试仍失败"


def _state_payload(state: dict) -> dict:
    """LLM 入参 state 子集：hypotheses / evidence 全量 + 上一轮 pending_tool_calls 摘要。"""
    return {
        "hypotheses": [h.model_dump(mode="json") for h in state.get("hypotheses") or []],
        "evidence": [e.model_dump(mode="json") for e in state.get("evidence") or []],
        # 上一轮 pending_tool_calls 只取 tool/priority/reason，不带 args 全量。
        "pending_tool_calls": [
            {
                "tool": c.get("tool"),
                "priority": c.get("priority"),
                "reason": c.get("reason"),
            }
            for c in (state.get("pending_tool_calls") or [])
        ],
    }


def _max_h_seq(hypotheses: list[Hypothesis]) -> int:
    """现用假设的最大 H 序号（H 后数字）；无匹配假设时返回 0（续号从 H1 起）。"""
    seq = 0
    for h in hypotheses:
        m = _H_ID_RE.match(h.id)
        if m:
            seq = max(seq, int(m.group(1)))
    return seq


def _apply(state: dict, out: ReevaluateOutput) -> dict:
    """确定性应用 ReevaluateOutput → hypotheses **全集**（覆盖写）。"""
    hypotheses = list(state.get("hypotheses") or [])
    by_id = {h.id: h for h in hypotheses}
    updated: dict[str, Hypothesis] = {}
    for u in out.hypothesis_updates:
        # 目标 = 原对象 model_copy（不就地改原对象）；id 必须命中，否则跳过。
        base = updated.get(u.id) if u.id in updated else by_id.get(u.id)
        if base is None:
            continue  # LLM 幻觉/失效 id → 跳过（id 必须命中现用假设）
        updated[u.id] = base.model_copy(
            update={
                "posterior": u.posterior,
                "status": HypothesisStatus(u.status),  # Literal 字符串 → 受控词表枚举
                "evidence_for": list(u.evidence_for),
                "evidence_against": list(u.evidence_against),
            }
        )
    # 保持原顺序：命中更新的替换为副本，未列出的原样保留。
    result: list[Hypothesis] = [updated.get(h.id, h) for h in hypotheses]
    # new_hypotheses：从现用最大 H 序号 +1 续号；status=PENDING、posterior=None。
    seq = _max_h_seq(hypotheses)
    for proposal in out.new_hypotheses:
        seq += 1
        result.append(
            Hypothesis(
                id=f"H{seq}",
                statement=proposal.statement,
                prior=proposal.prior,
                posterior=None,
                status=HypothesisStatus.PENDING,
                evidence_for=[],
                evidence_against=[],
            )
        )
    return {"hypotheses": result}


async def reevaluate_node(state: dict, config, *, llm: LLMBackend) -> dict:
    """reevaluate 图节点 action（graph.py 按 ``pra.agent.nodes.reevaluate`` import）。

    ``llm`` 由 ``build_agent_graph`` 装配期显式注入（本节点不持有/不查找任何默认后端）。
    """
    # 入口短路：degraded 或预算超限 → 不调 LLM，返回最小更新 {}（不动推理字段、
    # 不动 degraded —— 路由据此直接进 decide，overlay 兜底）。
    if state.get("degraded") or budget_exceeded(state["budget"]) is not None:
        return {}
    outcome = await call_structured_llm(
        OutputModel=ReevaluateOutput,
        node="reevaluate",
        state=_state_payload(state),
        llm=llm,
    )
    # 记账：按实际尝试次数（含 schema 重试）bump。
    budget = bump_llm_usage(
        state["budget"], llm_calls=outcome.attempts, tokens=outcome.tokens
    )
    if outcome.model is None:
        # 降级：不动 hypotheses（保留现有证据/假设，由路由进 decide 兜底）；
        # failures 只 append 本次新增。
        return {
            "degraded": True,
            "failures": [
                make_failure(
                    step_type=STEP_REEVALUATE,
                    severity=SEV_CRITICAL,
                    reason=_DEGRADE_REASON,
                )
            ],
            "budget": budget,
        }
    updates = _apply(state, outcome.model)
    updates["degraded"] = False
    updates["budget"] = budget
    return updates
