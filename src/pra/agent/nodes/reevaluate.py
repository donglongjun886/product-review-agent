"""reevaluate 节点：把本轮新证据综合进假设状态（docs/04-graph-design.md §2.2 主链路
"tools → reevaluate → route → plan/decide"环；《01》§3.3，graph MVP 契约 §4.2/§4.3）。

本节点 = "LLM 提案（ReevaluateOutput）+ apply 确定性应用"两层：
1. 依据 ``__STATE__`` 中**已给的 evidence**（type/source/value/weight/ref_id/extra
   全量）综合更新各条 hypothesis 的 posterior/status/evidence_for/evidence_against，
   并把已解答的 investigation_queue 项置 DONE。status 只允许 SUPPORTED / REFUTED /
   UNRESOLVED —— **证据不足置 UNRESOLVED**（"查了没结论" ≠ "证伪"；PENDING 只留给
   新假设）。prompt 指令层约束：只依据已给证据、禁止臆造任何未出现的事实/数值/来源。
2. 运行中**新发现的风险维度**走 ``ReevaluateOutput.new_hypotheses``（T-6：hypothesize
   不重跑），apply 追加为 status=PENDING、posterior=None 的新假设，id 从现用最大
   H 序号（H 后数字）+1 续号。
3. ``conflicts`` / ``evidence_sufficiency`` **仅供审计/参考**：AgentState 无对应
   channel，apply 不写入 state（本节点 docstring 注明 —— 02-evaluation 再决定是否随
   trace 落库）。

契约要点（对齐 graph MVP 契约 §2.1/§4.2）：
- hypotheses / investigation_queue 为**覆盖写** channel → 成功路径返回**全集**；
  evidence 走 merge reducer、tool_call_history/failures 走 append —— 本节点不动它们。
- 入口短路：``state["degraded"]`` 或 ``budget_exceeded(state["budget"])`` 非 None →
  **不调 LLM**，返回 ``{}``（不动推理字段、不动 degraded —— 超限/降级交由路由与
  decide overlay 收口）。
- LLM 记账：成功/失败均按 ``outcome.attempts`` 次数 bump（§2.1）。
- LLM 失败（schema 校验重试仍失败，outcome.model=None）→ 返回 degraded=True + 一条
  critical failure（STEP_REEVALUATE，reason 取固定字面值 ``_DEGRADE_REASON``，不拼
  outcome.error —— 与 hypothesize/plan 的 failure 文案约定一致，保持可断言）+
  bump 后 budget，**不动 hypotheses / queue**（04 §5.1：保留现有证据与假设；
  degraded=True 后由路由直接进 decide，overlay 以 R5_DEGRADED_OR_FAILED_STEP 归因
  转人工）。
- ``_build_messages``：MVP 简短事实注入（真实 litellm 的完整 prompt 后续补，§4.1）；
  **首条 user 消息固定 ``"__STATE__ {json}"``**（§4.3：scripted_llm 靠该 state 子集做
  确定性决策/可重放；state 子集用 pydantic ``model_dump(mode="json")`` 保证可序列化）。

语法/import 约定：顶部 ``from __future__ import annotations``；import 一律 ``pra.*``。
"""

from __future__ import annotations

import json
import re

from pra.agent.guardrails.budget import budget_exceeded, bump_llm_usage
from pra.agent.guardrails.errors import SEV_CRITICAL, STEP_REEVALUATE, make_failure
from pra.agent.guardrails.llm_shell import call_structured_llm
from pra.agent.guardrails.schemas import QueueUpdate, ReevaluateOutput
from pra.domain.models import Hypothesis, HypothesisStatus

__all__ = ["reevaluate_node"]

# 假设 id 序号匹配：H1..Hn（hypothesize 生成 / reevaluate.new_hypotheses 续号共用）。
_H_ID_RE = re.compile(r"^H(\d+)$")

# 系统指令 —— 证据综合（MVP 注入版；真实 litellm 的完整 prompt 后续补，§4.1）。
_SYSTEM_PROMPT = (
    "你是商品审核 Agent 的证据综合步骤（reevaluate）：把 __STATE__ 中本轮已收集的 "
    "evidence 综合进各条 hypothesis，并关闭已解答的队列问题。\n"
    "硬性约束：\n"
    "1. 只依据 __STATE__ 已给的 evidence 判断，禁止臆造任何未出现的事实/数值/来源；\n"
    "2. 证据不足、既无法证实也无法证伪的假设 → status=UNRESOLVED（不要把'没查到'当"
    "'证伪'）；\n"
    "3. hypothesis_updates 的 id 必须命中现有假设 id，status 只能取 SUPPORTED / "
    "REFUTED / UNRESOLVED（PENDING 只留给新增假设）；\n"
    "4. evidence_for / evidence_against 只填证据引用串：type + 空格 + value"
    "（value 为 ≤200 字符的人读摘要）；\n"
    "5. 运行中新发现的风险维度放 new_hypotheses（后续由 plan 决定取证），不要塞进 "
    "hypothesis_updates；\n"
    "6. 已解答的队列问题经 queue_updates 置 DONE。\n"
    "输出 JSON 必须严格符合给定 Schema。"
)

# LLM 步降级 failure 文案（契约 §2.1 允许用默认文案；与 hypothesize/plan 对齐，取
# 固定字面值、不拼 outcome.error —— 保持失败路径 reason 稳定、可断言）。
_DEGRADE_REASON = "schema 校验重试仍失败"


def _build_messages(state: dict) -> list[dict]:
    """构造 LLM 消息（§4.3 __STATE__ 机制）。

    首条 user 消息 = ``"__STATE__ " + json``，携带 hypotheses 全量 + evidence 全量
    （model_dump(mode="json")，含 type/weight/ref_id/extra）+ 上一轮 pending_tool_calls
    摘要（tools_node 消费后通常为空，仅审计/参考占位）。
    """
    hypotheses = [h.model_dump(mode="json") for h in state.get("hypotheses") or []]
    evidence = [e.model_dump(mode="json") for e in state.get("evidence") or []]
    pending = state.get("pending_tool_calls") or []
    queue = [dict(item) for item in (state.get("investigation_queue") or [])]
    state_json = json.dumps(
        {
            "hypotheses": hypotheses,
            "evidence": evidence,
            # investigation_queue 全量（scripted_llm 据此产 queue_updates 置 DONE；
            # code review R-B P1：此前漏注入导致队列项永不 DONE）。
            "investigation_queue": queue,
            # 上一轮 pending_tool_calls 摘要：只取 tool/priority/reason，不携带 args 全量。
            "pending_tool_calls": [
                {
                    "tool": c.get("tool"),
                    "priority": c.get("priority"),
                    "reason": c.get("reason"),
                }
                for c in pending
            ],
        },
        ensure_ascii=False,
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "__STATE__ " + state_json},
    ]


def _max_h_seq(hypotheses: list[Hypothesis]) -> int:
    """现用假设的最大 H 序号（H 后数字）；无匹配假设时返回 0（续号从 H1 起）。"""
    seq = 0
    for h in hypotheses:
        m = _H_ID_RE.match(h.id)
        if m:
            seq = max(seq, int(m.group(1)))
    return seq


def _apply_queue(queue: list[dict], queue_updates: list[QueueUpdate]) -> list[dict]:
    """队列项按 ``q`` 原文匹配 QueueUpdate 改 status（未命中保持原状）。

    返回新列表、不就地修改入参（纯函数；顺序与原有键保留）。
    """
    wanted = {qu.q: qu.status for qu in queue_updates}
    if not wanted:
        return [dict(item) for item in queue]
    result: list[dict] = []
    for item in queue:
        q = item.get("q")
        if q in wanted:
            result.append({**item, "status": wanted[q]})
        else:
            result.append(dict(item))
    return result


def _apply(state: dict, out: ReevaluateOutput) -> dict:
    """确定性应用 ReevaluateOutput → 返回 hypotheses / investigation_queue **全集**
    （覆盖写 channel 语义；本函数返回的 dict 只含这两个 channel，degraded/budget 由
    reevaluate_node 补上）。
    """
    hypotheses = list(state.get("hypotheses") or [])
    by_id = {h.id: h for h in hypotheses}
    updated: dict[str, Hypothesis] = {}
    for u in out.hypothesis_updates:
        # 目标 Hypothesis = 原对象 model_copy（不就地改原对象）；id 必须命中，否则跳过。
        base = updated.get(u.id) if u.id in updated else by_id.get(u.id)
        if base is None:
            continue  # LLM 幻觉/失效 id → 跳过（契约：id 必须命中现用假设）
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
    # new_hypotheses：从现用最大 H 序号 +1 续号；status=PENDING、posterior=None（未评估）。
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
    queue = _apply_queue(state.get("investigation_queue") or [], out.queue_updates)
    return {"hypotheses": result, "investigation_queue": queue}


async def reevaluate_node(state: dict, config) -> dict:
    """reevaluate 图节点 action（模块级导出名，graph.py 按 ``pra.agent.nodes.reevaluate``
    import，契约 §9.1）。

    - ``state``：AgentState（build_initial_state 已全量初始化，invoke 约定）；
    - ``config``：LangGraph 运行时配置（thread_id 本节点不消费，预留签名）。
    """
    # §2.1 入口短路：degraded 或预算超限 → 不调 LLM，返回最小更新 {}（不动推理字段、
    # 不动 degraded —— 路由据此直接进 decide，overlay 兜底）。
    if state.get("degraded") or budget_exceeded(state["budget"]) is not None:
        return {}
    outcome = await call_structured_llm(
        OutputModel=ReevaluateOutput,
        node="reevaluate",
        messages=_build_messages(state),
    )
    # 记账：按实际尝试次数（含 schema 重试）bump。
    budget = bump_llm_usage(
        state["budget"], llm_calls=outcome.attempts, tokens=outcome.tokens
    )
    if outcome.model is None:
        # 降级：不动 hypotheses / investigation_queue（04 §5.1：保留现有证据/假设，
        # 由路由进 decide，R5 归因）；failures 只 append 本次新增。
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
