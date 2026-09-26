"""tools_node —— 图内「调查工具执行」节点：执行本 visit 的 pending_tool_calls。"""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable

from pydantic import ValidationError

from pra.agent.guardrails.budget import bump_tool_usage
from pra.agent.guardrails.errors import SEV_WARN, STEP_TOOL_CALL, make_failure
from pra.agent.state import _evidence_key, merge_evidence
from pra.domain.models import Budget, Evidence
from pra.tools.base import Tool, ToolContext

MAX_REF_LEN = 200

__all__ = ["make_tools_node"]


def _ref_str(e: Evidence) -> str:
    """证据引用/摘要串：``f"{e.type} {e.value}"``，超长截到 ``MAX_REF_LEN``。"""
    ref = f"{e.type} {e.value}"
    return ref if len(ref) <= MAX_REF_LEN else ref[:MAX_REF_LEN]


def _elapsed_ms(t0: float) -> int:
    return int((time.perf_counter() - t0) * 1000)


def _error_record(seq: int, tool: str, args: Any, error: str, latency_ms: int = 0) -> dict[str, Any]:
    """错误 audit record（``status="error"``，无边际增益字段）。"""
    return {
        "seq": seq,
        "tool": tool,
        "args": args,
        "status": "error",
        "error": error,
        "latency_ms": latency_ms,
        "tokens": 0,
    }


def make_tools_node(tools: list[Tool]) -> Callable[[dict, dict], Awaitable[dict]]:
    """构建 tools 节点（闭包工厂）：按 ``tool.name`` 建私有映射，重名直接抛 ``ValueError``。"""
    tools_by_name: dict[str, Tool] = {}
    for tool in tools:
        if tool.name in tools_by_name:
            raise ValueError(f"工具重名：{tool.name!r}（装配缺陷，按名调度会静默失配）")
        tools_by_name[tool.name] = tool

    from pra.agent.guardrails.metrics import gate_probe

    async def tools_node(state: dict, config) -> dict:
        """执行本 visit 的 pending_tool_calls。"""
        budget_w = state.get("budget") or Budget()
        case = state.get("case")
        working_evidence = list(state.get("evidence") or [])
        tool_history = state.get("tool_call_history") or []
        pending = sorted(
            state.get("pending_tool_calls") or [], key=lambda c: c.get("priority", 5)
        )
        records: list[dict[str, Any]] = []
        added_all: list[Evidence] = []
        warn_failures: list[dict[str, Any]] = []
        seq = len(tool_history) + 1
        run_id = ((config or {}).get("configurable") or {}).get("thread_id", "unknown-run")
        case_id = case.case_id if case is not None else "unknown-case"

        def _snapshot() -> dict[str, Any]:
            """当前 working 快照。"""
            snap = dict(state)
            snap["evidence"] = list(working_evidence)
            snap["budget"] = budget_w
            return snap

        for call in pending:
            if budget_w.tool_calls >= budget_w.limits.max_tool_calls:
                break

            tool_name = call.get("tool")
            args_raw = call.get("args", {})
            if not isinstance(tool_name, str) or not tool_name:
                records.append(
                    _error_record(seq, tool_name or "", args_raw,
                                  "pending_tool_calls 元素缺少 tool 字段")
                )
                seq += 1
                continue

            tool = tools_by_name.get(tool_name)
            if tool is None:
                records.append(
                    _error_record(
                        seq,
                        tool_name,
                        args_raw,
                        f"tool 不可用: 未装配的 tool {tool_name!r}；"
                        f"已装配: {sorted(tools_by_name)}",
                    )
                )
                seq += 1
                continue
            try:
                parsed = tool.args_model.model_validate(args_raw)
            except ValidationError as exc:
                records.append(_error_record(seq, tool_name, args_raw, f"args 校验失败: {exc}"))
                seq += 1
                continue

            snap_before = _snapshot()
            before_gate = gate_probe(snap_before)

            ctx = ToolContext(run_id=run_id, case_id=case_id, budget=budget_w)

            t0 = time.perf_counter()
            try:
                result = await tool.call(parsed, ctx)
            except Exception:
                result = await tool.call(parsed, ctx)
            latency_ms = _elapsed_ms(t0)

            if not result.ok:
                budget_w = bump_tool_usage(budget_w)
                reason = result.error or f"{tool.name} 业务失败（无错误详情）"
                records.append(_error_record(seq, tool.name, args_raw, reason, latency_ms))
                warn_failures.append(
                    make_failure(step_type=STEP_TOOL_CALL, severity=SEV_WARN,
                                 tool=tool.name, seq=seq, reason=reason)
                )
                seq += 1
                continue

            budget_w = bump_tool_usage(budget_w)
            evs = tool.to_evidence(result)
            existing = {_evidence_key(x) for x in working_evidence}
            added = [e for e in evs if _evidence_key(e) not in existing]
            working_evidence = merge_evidence(working_evidence, added)
            added_all.extend(added)

            snap_after = _snapshot()
            after_gate = gate_probe(snap_after)
            refs = [_ref_str(e) for e in added]
            records.append(
                {
                    "seq": seq,
                    "tool": tool.name,
                    "args": args_raw,
                    "result_ref": refs[0] if refs else None,
                    "latency_ms": latency_ms,
                    "tokens": 0,
                    "status": "ok",
                    "evidence_added": refs,
                    "decision_changed": before_gate != after_gate,
                }
            )
            seq += 1

        return {
            "pending_tool_calls": [],
            "evidence": added_all,     # 本 visit 新增（reducer 去重合并）
            "tool_call_history": records,
            "failures": warn_failures,
            "budget": budget_w,
        }

    return tools_node
