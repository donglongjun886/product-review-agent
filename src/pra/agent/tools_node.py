"""tools_node —— 图内"调查工具执行"节点：执行本轮待办的 pending_tool_calls。

逐条走：预算截断 → ``parse_args`` 校验 → before 边际探针 → 工具执行（infra 失败重试
1 次）→ 记账 → ``to_evidence`` → 去重合并（只写回本 visit 新增）
→ after 边际探针 → 审计 record（ok 含边际增益字段：evidence_added / decision_changed，
JSON 承载、不进 DB 列）。

确定性口径：全程纯 Python，不调 LLM；infra 异常在重试 1 次后上抛（不写 failure、不落
record），业务失败（``ok=False``）仍 ``severity=warn``（warn 只进审计与 trace，不触发
转人工 —— 只有 critical 才触发）。预算语义：args 校验失败不计 tool_calls、不入 failures；
真正执行 call（成功 / 业务失败）计 1 次；业务失败不中断本 visit。边际探针同样只进
record，不驱动路由。

依赖注入：``make_tools_node(tools)`` 闭包持有私有 ``{name: tool}`` 映射，不建模块级单例；
run_id 从节点 config 的 ``configurable.thread_id`` 读取（缺省 ``"unknown-run"``）。
``evidence`` / ``metrics`` 的 import 延迟到图构建时刻，故本模块 import 阶段不依赖它们。
"""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable

from pydantic import ValidationError

from pra.agent.guardrails.budget import bump_tool_usage
from pra.agent.guardrails.errors import SEV_WARN, STEP_TOOL_CALL, make_failure
from pra.agent.state import _evidence_key, merge_evidence
from pra.domain.models import Budget, Evidence
from pra.observability.tracing import get_tracer
from pra.tools.base import Tool, ToolContext

# 引用/摘要串长度上限：f"{type} {value}" 截到 200 字符
MAX_REF_LEN = 200

__all__ = ["make_tools_node"]


def _ref_str(e: Evidence) -> str:
    """证据引用/摘要串：``f"{e.type} {e.value}"``，超长截到 200。"""
    ref = f"{e.type} {e.value}"
    return ref if len(ref) <= MAX_REF_LEN else ref[:MAX_REF_LEN]


def _elapsed_ms(t0: float) -> int:
    """距 ``t0`` 的墙钟耗时（毫秒）。"""
    return int((time.perf_counter() - t0) * 1000)


def _error_record(seq: int, tool: str, args: Any, error: str, latency_ms: int = 0) -> dict[str, Any]:
    """错误 audit record（不带边际增益 4 字段）。"""
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
    """构建 tools 节点（闭包工厂）。

    ``tools``（如 ``pra.tools.build_production_tools()`` 或测试替身）按 ``tool.name`` 建私有映射
    （重名 = 装配缺陷，直接抛错，避免按名调度静默失配）。
    """
    tools_by_name: dict[str, Tool] = {}
    for tool in tools:
        if tool.name in tools_by_name:
            raise ValueError(f"工具重名：{tool.name!r}（装配缺陷，按名调度会静默失配）")
        tools_by_name[tool.name] = tool

    # 延迟 import：evidence/metrics 到"图构建"这一刻才真正需要，
    # 且避免模块导入期就拉起 metrics 依赖。
    from pra.agent.guardrails.metrics import gate_probe

    async def tools_node(state: dict, config) -> dict:
        """执行本 visit 的 pending_tool_calls（确定性编排）。"""
        # 本 visit 的演进工作区（budget/evidence/seq 各自演进，不改 state）
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
        seq = len(tool_history) + 1  # 审计 seq 自增（含 plan 阶段已入 history 的 skipped）
        run_id = ((config or {}).get("configurable") or {}).get("thread_id", "unknown-run")
        case_id = case.case_id if case is not None else "unknown-case"
        # 观测（旁路）：进程级单例；无凭据 = NullTracer（全 no-op，零开销）。
        # 每次节点调用取一次 —— 注入的假 tracer（测试）即时生效。
        tracer = get_tracer()

        def _snapshot() -> dict[str, Any]:
            """当前 working 快照（边际探针用；只读、不驱动路由）。"""
            snap = dict(state)
            snap["evidence"] = list(working_evidence)
            snap["budget"] = budget_w
            return snap

        for call in pending:
            # ① Guardrail 截断（每轮开头）：预算只够前 k 个时只执行前 k 个 ——
            #    剩余不执行、不记 history。
            if budget_w.tool_calls >= budget_w.limits.max_tool_calls:
                break

            tool_name = call.get("tool")
            args_raw = call.get("args", {})
            if not isinstance(tool_name, str) or not tool_name:
                # 结构畸形（plan 缺陷）：预执行失败 —— error record，不入 failures
                records.append(
                    _error_record(seq, tool_name or "", args_raw,
                                  "pending_tool_calls 元素缺少 tool 字段")
                )
                seq += 1
                continue

            # ② 取工具 + args 校验：用工具自带 args_model 强校验 plan 给的 dict（不信任 LLM）
            tool = tools_by_name.get(tool_name)
            if tool is None:
                # 未装配的 tool（plan 缺陷）：预执行失败
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
                # 校验失败：error record；不入 failures、不 bump 预算
                records.append(_error_record(seq, tool_name, args_raw, f"args 校验失败: {exc}"))
                seq += 1
                continue

            # ③ before 边际探针（working 快照；只进审计，不驱动路由）
            snap_before = _snapshot()
            before_gate = gate_probe(snap_before)

            ctx = ToolContext(run_id=run_id, case_id=case_id, budget=budget_w)

            # ④ 执行：失败 infra 重试 1 次；重试仍抛 → 异常上抛
            # 观测（旁路）：一次工具调用 = 一个 tool span（覆盖 infra 重试；只读）。
            t0 = time.perf_counter()
            with tracer.tool_span(
                name=tool.name, input=args_raw, metadata={"seq": seq}
            ) as obs:
                try:
                    result = await tool.call(parsed, ctx)
                except Exception:
                    # 首次异常 → infra 重试 1 次
                    try:
                        result = await tool.call(parsed, ctx)
                    except Exception as exc:
                        obs.record_error(exc)  # 观测：异常（原 record 不变）
                        raise
                latency_ms = _elapsed_ms(t0)

                if not result.ok:
                    # 业务失败（工具正常返回但 ok=False）：reason=result.error
                    budget_w = bump_tool_usage(budget_w)
                    reason = result.error or f"{tool.name} 业务失败（无错误详情）"
                    records.append(_error_record(seq, tool.name, args_raw, reason, latency_ms))
                    warn_failures.append(
                        make_failure(step_type=STEP_TOOL_CALL, severity=SEV_WARN,
                                     tool=tool.name, seq=seq, reason=reason)
                    )
                    # 观测：业务失败（无异常对象，用人读 reason 构造）
                    obs.record_error(RuntimeError(reason))
                    seq += 1
                    continue

                # ⑤ 成功：记账 → to_evidence → 去重合并
                budget_w = bump_tool_usage(budget_w)
                evs = tool.to_evidence(result)
                existing = {_evidence_key(x) for x in working_evidence}
                added = [e for e in evs if _evidence_key(e) not in existing]
                working_evidence = merge_evidence(working_evidence, added)
                added_all.extend(added)

                # ⑥ after 边际探针 + ok record（含边际增益字段）
                snap_after = _snapshot()
                after_gate = gate_probe(snap_after)
                refs = [_ref_str(e) for e in added]
                # 观测：成功 —— output = 本 visit 新增证据引用串
                obs.update(
                    output={"evidence": refs, "status": "ok"},
                    metadata={"latency_ms": latency_ms, "seq": seq},
                )
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
            "pending_tool_calls": [],  # 消费完本 visit 待办（截断剩余由下轮 plan 重排）
            "evidence": added_all,     # 只返回本 visit 新增（reducer 去重合并）
            "tool_call_history": records,
            "failures": warn_failures,
            "budget": budget_w,
        }

    return tools_node
