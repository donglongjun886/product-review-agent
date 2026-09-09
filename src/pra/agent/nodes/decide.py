"""decide 节点：最终裁决 —— 图中唯一"LLM 提案 + 确定性 overlay"双层节点
（docs/00-system-design.md §7.2 / docs/01-agent-loop.md §3.4 / 04 §7.2；graph MVP 契约
§5.5 / §4.2 decide 例外）。

职责：LLM 只产 **DecisionProposal 提案**；``run_decision_overlay``（gate.py）做确定性
收口 —— R1 硬规则强制 REJECT、预算超限/矛盾/关键工具失败/政策不确定/假设不可区分/
degraded 等 abstention 兜底转人工（R3_*/R5_* 写进 overrides）、PASS/REJECT 只有过
Gate（pass_gate/reject_gate）才被采纳 → 产出唯一终态 ``ReviewDecision``。
**DECIDED 是图内唯一终态**：worker 在 invoke 返回后把 decision 落 DB（03 T-8）；
本节点不落库、不写额外状态字段（只返回 decision channel 覆盖写）。

契约要点（§5.5，逐条对齐）：
1. ``can_llm`` = not degraded ∧ 预算未超限（budget_exceeded is None）∧ 无未解决关键
   工具失败（errors.key_tool_failure）；否则不调 LLM：proposal=None、
   budget=state["budget"]、decide_llm_failed=False。
2. can_llm 时：``call_structured_llm(OutputModel=DecisionProposal, node="decide")``；
   记账按 outcome.attempts/tokens ``bump_llm_usage``。
3. LLM 失败（decide_llm_failed，outcome.model=None）→ proposal=None；overlay_state
   置 degraded=True 且 failures 追加一条 critical failure（STEP_DECIDE，reason=
   "decide LLM schema 校验重试仍失败"），**让 overlay 的 R5_DEGRADED_OR_FAILED_STEP
   落进 decision.overrides**（可审计的"谁降级了"）。
4. ``final = run_decision_overlay(overlay_state, proposal)`` —— budget_used 快照在
   overlay 的 build_decision 内 snapshot_budget 完成，节点不重复快照。
5. **decide 恒返回 degraded=False**：降级/失败被消费进 decision / overrides /
   failures 审计，不再把 degraded 透传出去（§5.5 注）。

- ``_build_messages``：MVP 简短事实注入（真实 litellm 的完整 prompt 后续补，§4.1）；
  首条 user 消息固定 ``"__STATE__ {json}"``（§4.3：scripted_llm 的确定性决策输入 ——
  hypotheses/evidence 全量 + degraded + failures + budget 摘要，model_dump(mode="json")）。

语法/import 约定：顶部 ``from __future__ import annotations``；import 一律 ``pra.*``。
"""

from __future__ import annotations

import json

from pra.agent.guardrails.budget import budget_exceeded, bump_llm_usage
from pra.agent.guardrails.errors import SEV_CRITICAL, STEP_DECIDE, key_tool_failure, make_failure
from pra.agent.guardrails.gate import run_decision_overlay
from pra.agent.guardrails.llm_shell import call_structured_llm
from pra.agent.guardrails.schemas import DecisionProposal
from pra.observability.tracing import get_tracer

__all__ = ["decide_node"]

# 系统指令 —— 决策提案（MVP 注入版；真实 litellm 的完整 prompt 后续补，§4.1）。
_SYSTEM_PROMPT = (
    "你是商品审核 Agent 的最终决策步骤（decide）：基于 __STATE__ 中的 hypotheses 与 "
    "evidence 给出**裁决提案**（仅提案 —— 确定性 overlay 还会做 Gate 校验与兜底）。\n"
    "三分类语义：\n"
    "1) PASS：所有高优先假设均被证据证伪（REFUTED）且无未解决的证据缺口 —— 放行；\n"
    "2) REJECT：有高优先假设被证据支持（SUPPORTED）、证据链充分且可引用政策/先例条款"
    "支撑 —— 拒绝上架；无据可依时宁可转人工，也不无依据拒绝；\n"
    "3) HUMAN_REVIEW：证据不足 / 置信不足 / 存在矛盾 / 政策模糊 / 关键取证失败 —— "
    "克制转人工。\n"
    "硬性约束：\n"
    "1. decision / risk_level 只取给定受控词表取值；risk_type 只能从 "
    "POTENTIAL_IP_RISK / EVASION_PATTERN / FALSE_CLAIM / FIELD_CONFLICT 中选择且必须与"
    "证据一致；\n"
    "2. evidence_ids 必须引用 __STATE__ 中真实存在的证据（type + value 引用串）；\n"
    "3. policy 只能填 __STATE__ 证据中真实出现的 policy_id / 条款号，禁止臆造；\n"
    "4. confidence ∈ [0,1]，仅表示'自动判定出错风险低'的把握（overlay 会确定性重算为 "
    "decision_confidence）；\n"
    "5. 任何不确定 → HUMAN_REVIEW，不要硬判 PASS / REJECT。\n"
    "输出 JSON 必须严格符合给定 Schema。"
)


def _build_messages(state: dict) -> list[dict]:
    """构造 LLM 消息（§4.3 __STATE__ 机制）。

    首条 user 消息 = ``"__STATE__ " + json``，携带 hypotheses + evidence 全量
    （model_dump(mode="json")）+ degraded + failures + budget 摘要。
    """
    hypotheses = [h.model_dump(mode="json") for h in state.get("hypotheses") or []]
    evidence = [e.model_dump(mode="json") for e in state.get("evidence") or []]
    budget = state.get("budget")
    if budget is not None:
        budget_summary = {
            "llm_calls": budget.llm_calls,
            "tool_calls": budget.tool_calls,
            "tokens": budget.tokens,
            "limits": {
                "max_llm_calls": budget.limits.max_llm_calls,
                "max_tool_calls": budget.limits.max_tool_calls,
                "max_tokens": budget.limits.max_tokens,
                "max_latency_ms": budget.limits.max_latency_ms,
            },
        }
    else:
        budget_summary = None
    payload = {
        "hypotheses": hypotheses,
        "evidence": evidence,
        "degraded": bool(state.get("degraded")),
        "failures": state.get("failures") or [],
        "budget": budget_summary,
    }
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "__STATE__ " + json.dumps(payload, ensure_ascii=False)},
    ]


def _gate_input_summary(proposal: DecisionProposal | None) -> dict:
    """Gate 子 span 的入参摘要（LLM 提案：decision / risk_level / confidence）。

    只读摘要，不参与任何判定；``proposal is None``（预算/降级/关键工具失败）如实记 None。
    """
    if proposal is None:
        return {"proposal": None}
    return {
        "proposal": {
            "decision": proposal.decision,
            "risk_level": proposal.risk_level,
            "confidence": proposal.confidence,
            "risk_type": [getattr(t, "value", t) for t in proposal.risk_type],
        }
    }


def _gate_output_summary(final) -> dict:
    """Gate 子 span 的出参摘要（overlay 后的终裁 + overrides 原因码，只读）。"""
    return {
        "decision": getattr(final.decision, "value", final.decision),
        "risk_level": getattr(final.risk_level, "value", final.risk_level),
        "decision_confidence": final.decision_confidence,
        "overrides": list(final.overrides),
    }


async def decide_node(state: dict, config) -> dict:
    """decide 图节点 action（模块级导出名，graph.py 按 ``pra.agent.nodes.decide``
    import，契约 §9.1）。

    - ``state``：AgentState（build_initial_state 已全量初始化，invoke 约定）；
    - ``config``：LangGraph 运行时配置（thread_id 本节点不消费，预留签名）。

    返回 {"decision": ReviewDecision, "degraded": False, "budget": Budget,
    "failures": [新 failure] 或 []}（failures 为 append reducer，只返回本次新增）。
    """
    can_llm = (
        not state["degraded"]
        and budget_exceeded(state["budget"]) is None
        and not key_tool_failure(state, state["failures"])
    )
    if can_llm:
        outcome = await call_structured_llm(
            OutputModel=DecisionProposal,
            node="decide",
            messages=_build_messages(state),
        )
        # 记账：按实际尝试次数（含 schema 重试）bump。
        budget = bump_llm_usage(
            state["budget"], llm_calls=outcome.attempts, tokens=outcome.tokens
        )
        proposal = outcome.model
        decide_llm_failed = outcome.model is None
    else:
        # 预算/降级/关键工具失败 → 不再烧一次 LLM；overlay 用 proposal=None 兜底。
        proposal = None
        budget = state["budget"]
        decide_llm_failed = False

    # overlay 只读的本地快照（浅拷贝）：预算换为记账后对象；decide LLM 失败时把
    # degraded=True + critical failure 注进去，让 R5_DEGRADED_OR_FAILED_STEP 落进
    # decision.overrides（节点自身返回仍为 degraded=False，§5.5 注）。
    overlay_state = dict(state)
    overlay_state["budget"] = budget
    failure = None
    if decide_llm_failed:
        failure = make_failure(
            step_type=STEP_DECIDE,
            severity=SEV_CRITICAL,
            reason="decide LLM schema 校验重试仍失败",
        )
        overlay_state["degraded"] = True
        overlay_state["failures"] = [*state["failures"], failure]

    # Gate 子 span（docs/09 §4.5）：Gate 不是图节点，这里只给它一层**只读**子观测 ——
    # 不新增 Graph Node、不改路由、不改 gate.py 的判定顺序与结果。
    with get_tracer().node_span("gate", input=_gate_input_summary(proposal)) as gate_span:
        final = run_decision_overlay(overlay_state, proposal)
        gate_span.update(output=_gate_output_summary(final))
    return {
        "decision": final,
        "degraded": False,
        "budget": budget,
        "failures": [failure] if decide_llm_failed else [],
    }
