"""plan 输出确定性去重（01 §4.3，防重复调用死循环的兜底之一）。

plan（LLM）可能因重试/上下文不清重复建议**已执行成功**的工具调用；不靠 prompt
承诺，靠确定性 Python 兜底（01 §4.3）：``dedup_pending`` 在 tools_node 消费
``pending_tool_calls`` 前清洗 ——

- **已执行集合** = ``state["tool_call_history"]`` 中 ``status == "ok"`` 的
  ``(tool, canonical_args(args))``；曾 ``status == "error"`` 的 (tool, args)
  **不进**已执行集合 → 允许重试（01 §4.3：error 保留，重试 1 次窗口）；
  ``skipped`` 状态（本 guardrail 自产）也不进已执行集合。
- 对 planned 逐条单趟扫描，**同轮内先自去重**（同一 (tool, canonical) 保留先出现，
  后现重复静默丢弃 —— 该 (tool, args) 的去向已由首个出现记录，不再重复产审计行）；
  命中已执行 → 从 cleaned 剔除并**进 skipped**（保留，供审计：status="skipped",
  reason="duplicate"，latency_ms/tokens 为 0 —— 未真实执行不记账）。
- skipped 元素 shape 固定：
  ``{"seq", "tool", "args", "status": "skipped", "reason": "duplicate",
    "latency_ms": 0, "tokens": 0}``；``seq`` 从 ``len(state["tool_call_history"]) + 1``
  起**顺延递增**（与 tools_node 的调用审计 seq 体系一致 —— 循环三保险之一
  dedup 截断后，路由 ④ 见 pending 为空即转 decide，见 graph-mvp-contracts §6.2 /
  01 §4.3）。实现对齐 docs/01-agent-loop.md §4.3。

``canonical_args``：args 的 canonical 序列化 —— ``json.dumps(args, sort_keys=True,
default=str)``（key 排序 + 非 JSON 原生类型按 str 规整，01 §4.3"key 排序 + 类型规整"；
对同一参数图恒同串 → 可确定性比对）。
"""

from __future__ import annotations

import json

# ---------------------------------------------------------------------------
# 参数 canonical 序列化
# ---------------------------------------------------------------------------


def canonical_args(args: dict) -> str:
    """args 的 canonical 序列化指纹：key 排序 + 非 JSON 原生类型按 ``str`` 规整。

    ``None``/空按 ``{}`` 处理，保证与历史 record 里 ``args: {}`` 可比。输入对象不变。
    """
    return json.dumps(args if args is not None else {}, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# pending_tool_calls 去重
# ---------------------------------------------------------------------------


def dedup_pending(state: dict, planned: list[dict]) -> tuple[list[dict], list[dict]]:
    """清洗 plan 输出（planned）→ ``(cleaned, skipped)``。

    - ``cleaned``：保留先出现、且未命中已执行集合的调用（原 dict 浅拷贝、保序）——
      tools_node 据此执行；
    - ``skipped``：命中"已执行成功"的调用审计行（shape 见模块 docstring）——
      已带 ``seq``，tools_node 直接并入 ``tool_call_history`` 后从其后 seq 续号。

    防御：``tool_call_history`` 缺失/None 按空处理；planned 缺失/None → 空结果；
    planned 内非 dict 元素跳过；元素缺 tool/args 按空串/{} 归一（PlanOutput 保证
    必有，仅防御）。已 error 的 (tool, args) 不进已执行集合 → 允许重试。
    """
    history = state.get("tool_call_history") if isinstance(state, dict) else None
    history = history or []
    # 已执行集合：只认 status=="ok"（error/skipped 均不拦，error 允许重试 1 次）
    executed: set[tuple] = set()
    for rec in history:
        if isinstance(rec, dict) and rec.get("status") == "ok":
            executed.add((str(rec.get("tool")), canonical_args(rec.get("args") or {})))

    cleaned: list[dict] = []
    skipped: list[dict] = []
    seen_round: set[tuple] = set()  # 同轮内已决断的 (tool, canonical) —— 自去重
    seq = len(history) + 1  # 审计 seq 从 history 尾顺延（与 tools_node 体系一致）

    for call in planned or []:
        if not isinstance(call, dict):
            continue
        tool = str(call.get("tool", ""))
        raw_args = call.get("args")
        key = (tool, canonical_args(raw_args))
        if key in seen_round:
            continue  # 同轮重复：先出现已决断，静默丢弃（不再产重复审计行）
        seen_round.add(key)
        if key in executed:
            # 已执行成功 → 不重跑，产出审计行（供"为什么没执行"回溯）
            skipped.append(
                {
                    "seq": seq,
                    "tool": tool,
                    "args": dict(raw_args) if isinstance(raw_args, dict) else {},
                    "status": "skipped",
                    "reason": "duplicate",
                    "latency_ms": 0,
                    "tokens": 0,
                }
            )
            seq += 1
        else:
            cleaned.append(dict(call))  # 保留（含 reason/priority 等原键；浅拷贝防别名）

    return cleaned, skipped
