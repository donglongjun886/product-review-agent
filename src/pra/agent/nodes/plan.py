"""plan 节点 —— 决定本轮取证工具（LLM 语义步 + 确定性 dedup）。

对照当前证据缺口与仍存疑（PENDING/UNRESOLVED）的假设，决定本轮是否调用取证工具：
``PlanOutput.next_action`` ∈ call_tools / conclude。

- **入口短路**：``state["degraded"]`` 为真或 ``budget_exceeded(state["budget"])`` 非
  None → 不调 LLM，返回 ``{"pending_tool_calls": []}``，且**不动 degraded**（后续路由
  转 decide 止损）。
- 假设/证据/工具选择来自 LLM（``PlanOutput`` 由 llm_shell 强校验：失败重试 1 次仍失败
  → 降级 + ``degraded=True``）；**是否真的执行**由 tools_node 经 ``ToolRegistry.parse_args``
  确定性校验（不信任 LLM 参数）。
- 确定性 apply：``next_action=="conclude"`` 或 ``tools`` 为空 → 计划为空；否则
  ``PlannedToolCall → dict {tool, args, reason, priority}``。
- **确定性 dedup**（guardrails/dedup）：同轮自去重 + 过滤已执行成功调用（曾 error 的
  同 tool+args 允许重试）；被过滤的调用进 ``skipped`` 审计（带 seq 的 tool_call_history
  记录，append reducer）。清洗后 pending 为空 → 路由自动 decide。
- ``_build_messages`` 首条 user 消息以 ``__STATE__ {json}`` 注入 state 子集
  （hypotheses 仪表盘 + evidence 摘要 + case 子集）。
"""

from __future__ import annotations

import json

from pra.agent.guardrails.budget import budget_exceeded, bump_llm_usage
from pra.agent.guardrails.dedup import dedup_pending
from pra.agent.guardrails.errors import SEV_CRITICAL, STEP_PLAN, make_failure
from pra.agent.guardrails.llm_shell import call_structured_llm
from pra.agent.guardrails.schemas import PlanOutput

__all__ = ["plan_node"]

# LLM 步降级 failure 文案：与 hypothesize 一致，取固定字面值、不拼 outcome.error。
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
    """plan 视角的 case 子集：案件身份 + 商品核心字段 + 图片（保持原字段名）。"""
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


def _coverage_gap_lines(state: dict) -> list[str]:
    """本案必需测量的覆盖清单（人读行，注入 plan 上下文）。

    在**节点侧**用真实 state 计算（case 是 domain 对象、能力表齐备），渲染层只负责打印 ——
    避免渲染层拿 ``__STATE__`` 里的 case 子集去反序列化（缺字段会炸）。

    除 PASS 必需的四维外，**有阳性证据时额外列出 ``policy_citation`` 缺口**：
    它不是 PASS 的必要条件（放行不需要引用依据），却是自动 REJECT 的必要条件。
    不列出来，plan 会因为"必需覆盖已满"而提前 conclude，导致案件以
    ``R3_POSITIVE_INSUFFICIENT`` 转人工、白白丢掉可自动拒绝的案。
    """
    from pra.agent.guardrails.measurements import coverage_report
    from pra.domain.measurement import DIM_POLICY_CITATION

    evidence = list(state.get("evidence") or [])
    cov = coverage_report(
        state.get("case"),
        evidence,
        state.get("measurement_capabilities"),
    )
    covered = [d for d in cov.required if d in cov.covered]
    lines = [f"- 必需测量覆盖：{len(covered)}/{len(cov.required)}"]
    for dim in cov.required:
        if dim in cov.covered:
            verdict = "阳性" if dim in cov.positive else "阴性"
            lines.append(f"  - {dim}：已测（{verdict}）")
        elif dim in cov.unmeasurable:
            lines.append(f"  - {dim}：本环境不可测（不要再安排该类工具，重跑无用）")
        else:
            lines.append(f"  - {dim}：**尚未取得** —— 优先安排能补齐它的工具")
    has_citable = any(
        e.type in ("POLICY_REF", "CASE_PRECEDENT") and e.ref_id for e in evidence
    )
    if cov.positive and not has_citable:
        if (state.get("measurement_capabilities") or {}).get(DIM_POLICY_CITATION, True):
            lines.append(
                "  - policy_citation：**尚未取得** —— 已存在阳性证据，自动拒绝必须能引用政策"
                "条款或同类先例，**请安排先例/政策检索**"
            )
        else:
            lines.append("  - policy_citation：本环境不可测（无检索数据源）")
    return lines


def _build_messages(state: dict) -> list[dict]:
    """组装 LLM 消息：system=规划指令；首条 user 以 "__STATE__ " 携带 state 子集
    （hypotheses 仪表盘 + evidence 摘要 + case 子集 + 环境测量能力 + 必需测量缺口，
    供 scripted 桩做确定性分支）。

    ``measurement_capabilities`` 与覆盖缺口一并注入：让 plan 优先安排能补齐缺口的工具，
    并区分"没测"（可补）与"本环境不可测"（补不了，别空转）。
    """
    payload = {
        "hypotheses": [
            h.model_dump(mode="json") for h in (state.get("hypotheses") or [])
        ],
        "evidence": [e.model_dump(mode="json") for e in (state.get("evidence") or [])],
        "case": _case_subset(state["case"]),
        "measurement_capabilities": dict(state.get("measurement_capabilities") or {}),
        "required_measurement_coverage": _coverage_gap_lines(state),
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

    ``conclude`` 或 ``tools`` 为空（含 call_tools 但 tools 空，按 conclude 容错）→ 计划
    为空；否则 PlannedToolCall → dict ``model_dump()``。
    """
    if out.next_action == "conclude" or not out.tools:
        return []
    return [call.model_dump() for call in out.tools]


async def plan_node(state: dict, config) -> dict:
    """plan 图节点：决定本轮取证工具（LLM 语义步）+ 确定性 dedup。

    返回 pending_tool_calls / tool_call_history / degraded / failures / budget；cleaned
    为空 → 路由自动 decide。
    """
    # 入口短路：degraded 或预算超限 → 不调 LLM、最小更新、不动 degraded。
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
    # 确定性 dedup：同轮自去重 + 过滤已执行成功调用（曾 error 的允许重试）；
    # skipped 为被过滤调用的审计记录（append reducer），与 pending 分离。
    cleaned, skipped = dedup_pending(state, planned)
    return {
        "pending_tool_calls": cleaned,
        "tool_call_history": skipped,
        "degraded": False,
        "budget": budget,
    }
