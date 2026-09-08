"""plan 节点 —— 决定本轮取证工具（LLM 语义步；确定性 dedup 在 guardrails/dedup）。

职责（docs/04-graph-design.md §2.2 /《01》§3.2；graph-mvp-contracts §3 G5）：
对照当前证据缺口与仍存疑（PENDING/UNRESOLVED）的假设，决定本轮是否调用取证工具
（``PlanOutput.next_action`` ∈ call_tools / conclude），为 tools → reevaluate →
（plan）循环的每一轮选择"下一步查什么"。

- **入口短路**（§2.1 / §4.2）：``state["degraded"]`` 为真或
  ``budget_exceeded(state["budget"])`` 非 None → 不调 LLM，返回最小更新
  ``{"pending_tool_calls": []}``，**不动 degraded**（后续路由转 decide 止损）。
- 假设/证据/工具调用选择来自 LLM（``PlanOutput`` 由 llm_shell 强校验：失败重试 1
  次仍失败 → 本节点降级 + ``degraded=True``，§2.1）；**是否真的执行**由 tools_node
  经 ``ToolRegistry.parse_args`` 确定性校验（不信任 LLM 参数）。
- 确定性 apply：``next_action=="conclude"`` 或 ``tools`` 为空 → 计划为空；
  否则 ``PlannedToolCall → dict {tool, args, reason, priority}``（model_dump）。
- **确定性 dedup**（guardrails/dedup，§6.2）：``dedup_pending`` 同轮自去重 + 过滤
  已执行成功调用（曾 error 的同 tool+args 允许重试）；被过滤的调用进 ``skipped``
  （带审计 seq 的 tool_call_history 记录，append reducer —— 本节点只返回本次新增）。
  清洗后 ``pending_tool_calls`` 为空（含全被 dedup）→ 路由自动 decide（§6.2：
  pending 空 → decide，无需额外标记）。
- ``_build_messages`` MVP 只做"首条 user 消息以 ``__STATE__ {json}`` 注入 state 子集
  （hypotheses 仪表盘 + evidence 摘要 + case 子集；domain 对象经
  ``model_dump(mode="json")`` 转 JSON 形状；scripted_llm 桩据此决策）"；真实
  litellm 的完整 prompt（规划约束展开）在接入真实 LLM 时补充。

分工声明：本节点只做"计划 + dedup"（计划是 LLM 语义步，dedup 调确定性护栏）；
预算/降级判定、failures 审计、确定性工具执行分别在 guardrails/{budget,errors} 与
tools_node —— 不在本节点重复实现。
"""

from __future__ import annotations

import json

from pra.agent.guardrails.budget import budget_exceeded, bump_llm_usage
from pra.agent.guardrails.dedup import dedup_pending
from pra.agent.guardrails.errors import SEV_CRITICAL, STEP_PLAN, make_failure
from pra.agent.guardrails.llm_shell import call_structured_llm
from pra.agent.guardrails.schemas import PlanOutput

__all__ = ["plan_node"]

# LLM 步降级 failure 文案（契约 §2.1 允许用默认文案；与 hypothesize 保持一致，
# 不拼 outcome.error —— 失败路径 reason 稳定、可断言）。
_DEGRADE_REASON = "schema 校验重试仍失败"

_SYSTEM_PROMPT = (
    "你是调查取证的计划器。根据当前已收集证据与仍存疑（PENDING/UNRESOLVED）的假设，"
    "决定本轮是否调用取证工具：\n"
    "1. 只计划能带来「新证据」的工具调用（对照证据缺口：如 IMAGE_SIMILARITY / "
    "PRODUCT_FACT / MERCHANT_HISTORY / CASE_PRECEDENT / POLICY_REF 尚未收集时安排"
    "对应工具取证）；\n"
    "2. 不重复已成功执行过的调用（同 tool + 同 args；确定性 dedup 会兜底过滤）；\n"
    "3. 若已无能带来新证据的工具 → next_action=conclude（tools 必须为空）；\n"
    "4. tools 每轮 ≤3 条、priority 1 最优先，reason 说明验证哪条假设 / 要哪条证据。\n"
    "只输出符合 PlanOutput JSON Schema 的 JSON。"
)


def _case_subset(case) -> dict:
    """plan 视角的 case 子集：案件身份 + 商品核心字段 + 图片（取证起点）。

    保持 ProductReviewCase / ProductInfo / ProductImage 的**原字段名**（值已 JSON
    化），便于脚本后端与真实 LLM 直接按 ``state["case"][...]`` 读取 merchant_id /
    product_id / category / images（url 列表等）做确定性分支。
    """
    product = case.product
    return {
        "case_id": case.case_id,
        "merchant_id": case.merchant_id,
        "event_type": case.event_type,
        "product": {
            "product_id": product.product_id,
            "title": product.title,
            "description": product.description,
            "category": product.category,
            "brand": product.brand,
            "version": product.version,
            "attributes": dict(product.attributes),
            "sku_list": [s.model_dump(mode="json") for s in product.sku_list],
            "images": [i.model_dump(mode="json") for i in product.images],
            "listing_time": product.listing_time.isoformat(),
        },
    }


def _build_messages(state: dict) -> list[dict]:
    """组装 LLM 消息：system=规划指令；首条 user 以 "__STATE__ " 开头携带 state 子集。

    state 子集 = hypotheses 仪表盘（id/status/prior/posterior/证据引用，model_dump
    全量即仪表盘）+ evidence 摘要 + case 子集 —— 供 scripted_llm 桩按 evidence 类型
    存在性与 case 字段做确定性分支。MVP 简短注入即可，完整 prompt 后续补。
    """
    payload = {
        "hypotheses": [
            h.model_dump(mode="json") for h in (state.get("hypotheses") or [])
        ],
        "evidence": [e.model_dump(mode="json") for e in (state.get("evidence") or [])],
        "case": _case_subset(state["case"]),
    }
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "__STATE__ " + json.dumps(payload, ensure_ascii=False),
        },
    ]


def _apply_plan(out: PlanOutput) -> list[dict]:
    """LLM 提案 → 待执行工具列表（确定性 apply）。

    ``conclude`` 或 ``tools`` 为空（含 next_action=call_tools 但 tools 空的不一致
    输出，按 conclude 容错）→ 计划为空；否则 PlannedToolCall → dict {tool, args,
    reason, priority}（``model_dump``）。
    """
    if out.next_action == "conclude" or not out.tools:
        return []
    return [call.model_dump() for call in out.tools]


async def plan_node(state: dict, config) -> dict:
    """plan 图节点：决定本轮取证工具（LLM 语义步）+ 确定性 dedup。

    返回 dict 只含 AgentState channel 键：pending_tool_calls / tool_call_history /
    degraded / failures / budget。cleaned 为空（conclude 或全被 dedup）时
    pending_tool_calls=[] → 路由自动 decide（契约 §6.2，无需额外标记）。
    """
    # §2.1 入口短路：degraded 或预算超限 → 不调 LLM、最小更新、不动 degraded。
    if state.get("degraded") or budget_exceeded(state["budget"]) is not None:
        return {"pending_tool_calls": []}

    outcome = await call_structured_llm(
        OutputModel=PlanOutput, node="plan", messages=_build_messages(state)
    )
    # LLM 记账：按实际尝试次数 bump（成功 1 次 / 重试后成功 2 次 / 两次失败仍 2 次）
    budget = bump_llm_usage(
        state["budget"], llm_calls=outcome.attempts, tokens=outcome.tokens
    )
    if outcome.model is None:
        return {
            "pending_tool_calls": [],
            "degraded": True,
            "failures": [
                make_failure(
                    step_type=STEP_PLAN,
                    severity=SEV_CRITICAL,
                    reason=_DEGRADE_REASON,
                )
            ],
            "budget": budget,
        }

    planned = _apply_plan(outcome.model)
    # 确定性 dedup：同轮自去重 + 过滤已执行成功调用（曾 error 的允许重试）。
    # skipped = 被过滤的调用审计记录（append reducer：本次新增），与 pending 分离。
    cleaned, skipped = dedup_pending(state, planned)
    return {
        "pending_tool_calls": cleaned,
        "tool_call_history": skipped,
        "degraded": False,
        "budget": budget,
    }
