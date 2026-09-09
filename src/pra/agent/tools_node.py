"""tools_node —— 图内"调查工具执行"节点（G7 覆盖重写 stub；graph MVP 契约 §8）。

职责（契约 §8.2 主循环规格）：执行本 visit 的 ``pending_tool_calls``，逐条完成
预算 Guardrail 截断 → ``parse_args`` 参数校验 → before 边际探针 → 工具执行（失败
infra 重试 1 次）→ 记账 → ``to_evidence`` → 质量过滤（T-11）→ extra 回填（O-8）→
``_evidence_key`` 去重合并（只写回本 visit 新增）→ after 边际探针 → 审计 record
（ok 含边际增益 4 字段：before_confidence / after_confidence / evidence_added /
decision_changed，04 §4 / O-10 —— JSON 承载，不进 DB 列）。

确定性口径：**全程纯 Python，不调 LLM**（04 §4/§5/§9）；预算/校验/过滤/去重全部
走已落盘或契约签名的确定性函数。MVP 无 ``required`` 标记 → 工具失败一律
``severity=warn``（O-2：warn 只进审计与 trace、不触发转人工 —— R3_KEY_TOOL_FAILED
只认 critical）。预算语义：args 校验失败不计 tool_calls、不入 failures；一旦真正
执行 call（成功 / 业务失败 / 异常重试后）计 1 次；失败不中断本 visit（证据缺口由
下一轮 plan 自然暴露重查）。

依赖注入（04 §9）：``make_tools_node(tools)`` 闭包持有私有 ``ToolRegistry``，不建
模块级单例；run_id 从节点 config 的 ``configurable.thread_id`` 读取（缺省
``"unknown-run"``）。

并行实现注记：``quality_filter/backfill_extra``（G3 evidence.py，契约 §6.1）与
``decision_conf_probe/gate_probe``（G1 metrics.py，契约 §5.6）可能晚于本文件落盘，
故延迟到 ``make_tools_node``（图构建时刻）才 import —— 本模块 import / py_compile
阶段不依赖它们。边际探针为纯审计（只进 record），不驱动路由。
"""

from __future__ import annotations

import time
from typing import Any, Awaitable, Callable

from pydantic import ValidationError

from pra.agent.guardrails.budget import bump_tool_usage
from pra.agent.guardrails.errors import SEV_WARN, STEP_TOOL_CALL, make_failure
from pra.agent.state import _evidence_key, merge_evidence
from pra.domain.models import Budget, Evidence
from pra.observability.tracing import Observation, get_tracer
from pra.tools.base import Tool, ToolContext, ToolRegistry

# 引用/摘要串长度上限（契约 §2.2：f"{type} {value}"，≤200 字符）
MAX_REF_LEN = 200

__all__ = ["make_tools_node"]


def _observe_update(obs: Observation, **kwargs: Any) -> None:
    """观测旁路：``update`` 绝不抛（适配层已吞异常，此处双保险）。"""
    try:
        obs.update(**kwargs)
    except Exception:  # noqa: BLE001 - 观测失败不得影响业务
        return


def _observe_record_error(obs: Observation, exc: BaseException) -> None:
    """观测旁路：``record_error`` 绝不抛。"""
    try:
        obs.record_error(exc)
    except Exception:  # noqa: BLE001 - 观测失败不得影响业务
        return


def _ref_str(e: Evidence) -> str:
    """证据引用/摘要串（§2.2 统一格式 ``f"{e.type} {e.value}"``，超长截到 200）。"""
    ref = f"{e.type} {e.value}"
    return ref if len(ref) <= MAX_REF_LEN else ref[:MAX_REF_LEN]


def _elapsed_ms(t0: float) -> int:
    """距 ``t0`` 的墙钟耗时（毫秒，int）。"""
    return int((time.perf_counter() - t0) * 1000)


def _error_record(seq: int, tool: str, args: Any, error: str, latency_ms: int = 0) -> dict[str, Any]:
    """错误 audit record（契约 §8.2：不带边际 4 字段）。"""
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
    """构建 tools 节点（闭包工厂，§8.1；依赖注入见模块 docstring）。

    ``tools``：外部注入的 Tool 列表（如 ``pra.tools.build_tools()`` 或测试替身），
    逐个注册进本闭包私有的 ``ToolRegistry``（重名/非 Tool 由 register 防呆抛错）。
    返回的 ``tools_node(state, config)`` 执行 pending_tool_calls（预算 / parse_args
    校验 / 执行 / 质量过滤 / extra 回填 / 去重合并 / 记账 / 边际增益 4 字段），
    全程确定性 Python、不调 LLM（04 §4/§5/§9）。
    """
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)

    # 延迟 import：并行实现期间 evidence/metrics 可能未落盘；且防模块导入期拉起依赖。
    # 到"图构建"这一刻才真正需要它们 —— 详见模块 docstring 注记。
    from pra.agent.guardrails.evidence import backfill_extra, quality_filter
    from pra.agent.guardrails.metrics import decision_conf_probe, gate_probe

    async def tools_node(state: dict, config) -> dict:
        """执行本 visit 的 pending_tool_calls（确定性编排，不调 LLM）。"""
        # ---- 本 visit 演进用工作区（budget/evidence/seq 各自演进，不改 state）----
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
        # 观测（S3，旁路）：进程级单例；无凭据 = NullTracer（全 no-op，零开销）。
        # 每次节点调用取一次 —— 注入的假 tracer（测试）即时生效。
        tracer = get_tracer()

        def _snapshot() -> dict[str, Any]:
            """当前 working 快照（before/after 边际探针用；只读、不驱动路由）。"""
            snap = dict(state)
            snap["evidence"] = list(working_evidence)
            snap["budget"] = budget_w
            return snap

        for call in pending:
            # ① Guardrail 截断（每轮开头）：预算只够前 k 个时只执行前 k 个 ——
            #    剩余不执行、不记 history（budget.py 检查点口径）。
            if budget_w.tool_calls >= budget_w.limits.max_tool_calls:
                break

            tool_name = call.get("tool")
            args_raw = call.get("args", {})
            if not isinstance(tool_name, str) or not tool_name:
                # 结构畸形（plan 缺陷）：预执行失败 —— error record，不入 failures、不 bump
                records.append(
                    _error_record(seq, tool_name or "", args_raw,
                                  "pending_tool_calls 元素缺少 tool 字段")
                )
                seq += 1
                continue

            # ② args 校验（O-5）：registry 用工具自带 args_model 强校验 plan 给的 dict
            try:
                parsed = registry.parse_args(tool_name, args_raw)
            except ValidationError as exc:
                # 校验失败：error record；不入 failures、不 bump（契约 §8.2 ①）
                records.append(_error_record(seq, tool_name, args_raw, f"args 校验失败: {exc}"))
                seq += 1
                continue
            except KeyError as exc:
                # 未注册 tool（plan 缺陷，registry.get 内抛）：同属预执行失败
                detail = exc.args[0] if exc.args else str(exc)
                records.append(_error_record(seq, tool_name, args_raw, f"tool 不可用: {detail}"))
                seq += 1
                continue

            tool = registry.get(tool_name)

            # ③ before 边际探针（working 快照；仅供审计，不驱动路由 —— metrics.py 同口径）
            snap_before = _snapshot()
            before_conf = decision_conf_probe(snap_before)
            before_gate = gate_probe(snap_before)

            ctx = ToolContext(run_id=run_id, case_id=case_id, budget=budget_w)

            # ④ 执行：失败 infra 重试 1 次；重试仍抛 → error 分支（不中断本 visit）
            # 观测（S3，旁路）：一次工具调用 = 一个 tool span（覆盖 infra 重试；
            # 只读、不驱动路由；适配层契约保证观测调用绝不抛）。
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
                        latency_ms = _elapsed_ms(t0)
                        budget_w = bump_tool_usage(budget_w)  # 已执行（含重试）计 1 次
                        reason = f"工具执行异常（infra 重试 1 次后仍失败）: {type(exc).__name__}: {exc}"
                        records.append(_error_record(seq, tool.name, args_raw, reason, latency_ms))
                        warn_failures.append(
                            make_failure(step_type=STEP_TOOL_CALL, severity=SEV_WARN,
                                         tool=tool.name, seq=seq, reason=reason)
                        )
                        _observe_record_error(obs, exc)  # 观测：异常（原 record 不变）
                        seq += 1
                        continue
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
                    # 观测：业务失败（无异常对象 → 用人读 reason 构造，status_message 可读）
                    _observe_record_error(obs, RuntimeError(reason))
                    seq += 1
                    continue

                # ⑤ 成功：记账 → to_evidence → 质量过滤（T-11）→ extra 回填（O-8）→ 去重合并
                budget_w = bump_tool_usage(budget_w)
                raw = tool.to_evidence(result)
                evs = quality_filter(raw)
                evs = backfill_extra(evs, case=case)  # case=None 时跳过 version_drift 回填
                existing = {_evidence_key(x) for x in working_evidence}
                added = [e for e in evs if _evidence_key(e) not in existing]
                working_evidence = merge_evidence(working_evidence, added)
                added_all.extend(added)

                # ⑥ after 边际探针 + ok record（含边际增益 4 字段）
                snap_after = _snapshot()
                after_conf = decision_conf_probe(snap_after)
                after_gate = gate_probe(snap_after)
                refs = [_ref_str(e) for e in added]
                # 观测：成功 —— output = 本 visit 新增证据引用串；latency_ms 复用既有值
                _observe_update(
                    obs,
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
                        "before_confidence": before_conf,
                        "after_confidence": after_conf,
                        "evidence_added": refs,
                        "decision_changed": before_gate != after_gate,
                    }
                )
            seq += 1

        return {
            "pending_tool_calls": [],  # 消费完本 visit 待办（截断剩余由下轮 plan 重排）
            "evidence": added_all,     # 只返回本 visit 新增（state reducer 去重合并）
            "tool_call_history": records,
            "failures": warn_failures,
            "budget": budget_w,
        }

    return tools_node
