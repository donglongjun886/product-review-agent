"""plan 节点：决定取证工具（LLM 语义步 + 确定性 dedup）。

``PlanOutput.next_action`` ∈ ``call_tools`` / ``conclude``；返回 pending_tool_calls /
tool_call_history / degraded / failures / budget。
"""

from __future__ import annotations

from pra.agent.guardrails.budget import budget_exceeded, bump_llm_usage
from pra.agent.guardrails.dedup import dedup_pending
from pra.agent.guardrails.errors import SEV_CRITICAL, STEP_PLAN, make_failure
from pra.agent.guardrails.llm_shell import LLMBackend, call_structured_llm
from pra.agent.guardrails.schemas import PlanOutput

__all__ = ["plan_node"]

# LLM 步降级 failure 文案（固定字面值）
_DEGRADE_REASON = "schema 校验重试仍失败"


def _coverage_gap_lines(state: dict) -> list[str]:
    """本案必需测量的覆盖清单（人读行，注入 plan 上下文）；有阳性证据且无可引用依据时额外列出 ``policy_citation`` 缺口。"""
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


def _state_payload(state: dict) -> dict:
    """LLM 入参 state 子集：hypotheses / evidence / case 全量 + ``required_measurement_coverage`` 覆盖缺口。"""
    return {
        "hypotheses": [
            h.model_dump(mode="json") for h in (state.get("hypotheses") or [])
        ],
        "evidence": [e.model_dump(mode="json") for e in (state.get("evidence") or [])],
        "case": state["case"].model_dump(mode="json"),
        "required_measurement_coverage": _coverage_gap_lines(state),
    }


def _apply_plan(out: PlanOutput) -> list[dict]:
    """LLM 提案 → 待执行工具列表；``conclude`` 或 ``tools`` 为空返回空列表。"""
    if out.next_action == "conclude" or not out.tools:
        return []
    return [call.model_dump() for call in out.tools]


async def plan_node(state: dict, config, *, llm: LLMBackend) -> dict:
    """plan 图节点：决定取证工具并做确定性 dedup；``llm`` 由装配期显式注入。"""
    if state.get("degraded") or budget_exceeded(state["budget"]) is not None:
        return {"pending_tool_calls": []}

    outcome = await call_structured_llm(
        OutputModel=PlanOutput, node="plan", state=_state_payload(state), llm=llm
    )
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
    cleaned, skipped = dedup_pending(state, planned)
    return {
        "pending_tool_calls": cleaned,
        "tool_call_history": skipped,
        "degraded": False,
        "budget": budget,
    }
