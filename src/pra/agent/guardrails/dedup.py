"""plan 输出确定性去重：不靠 prompt 承诺、靠 Python 兜底，防重复调用死循环。

plan（LLM）可能因重试/上下文不清重复建议**已执行成功**的调用。``dedup_pending``
在 tools_node 消费 ``pending_tool_calls`` 前清洗：

- **已执行集合** = ``tool_call_history`` 中 ``status == "ok"`` 的
  ``(tool, canonical_args(args))``；``error``（允许重试 1 次）与 ``skipped`` 不进
  该集合。
- planned 逐条单趟扫描，**同轮内先自去重**（同一 (tool, canonical) 保留先出现的，
  后现的静默丢弃，不再重复产审计行）；命中已执行 → 从 cleaned 剔除并**进 skipped**
  （未真实执行，latency_ms/tokens 为 0）。
- skipped 元素 shape 固定为 ``{"seq", "tool", "args", "status": "skipped",
  "reason": "duplicate", "latency_ms": 0, "tokens": 0}``；``seq`` 从
  ``len(tool_call_history) + 1`` 起顺延递增，与 tools_node 的审计 seq 体系一致。
  dedup 截断后 pending 为空，路由即转 decide。
"""

from __future__ import annotations

import json


def canonical_args(args: dict) -> str:
    """args 的 canonical 序列化指纹：key 排序 + 非 JSON 原生类型按 ``str`` 规整。"""
    return json.dumps(args if args is not None else {}, sort_keys=True, default=str)


def dedup_pending(state: dict, planned: list[dict]) -> tuple[list[dict], list[dict]]:
    """清洗 plan 输出（planned）→ ``(cleaned, skipped)``。

    ``cleaned`` 是未命中已执行集合的调用（原 dict 浅拷贝、保序）；``skipped`` 是命中
    「已执行成功」的审计行（shape 见模块 docstring）。history/planned 缺失或 None 按
    空处理，planned 内非 dict 跳过。
    """
    history = state.get("tool_call_history") if isinstance(state, dict) else None
    history = history or []
    # 只认 status=="ok"（error 允许重试 1 次、skipped 不拦）
    executed: set[tuple] = set()
    for rec in history:
        if isinstance(rec, dict) and rec.get("status") == "ok":
            executed.add((str(rec.get("tool")), canonical_args(rec.get("args") or {})))

    cleaned: list[dict] = []
    skipped: list[dict] = []
    seen_round: set[tuple] = set()  # 同轮内已决断的 (tool, canonical)
    seq = len(history) + 1  # 审计 seq 从 history 尾顺延

    for call in planned or []:
        if not isinstance(call, dict):
            continue
        tool = str(call.get("tool", ""))
        raw_args = call.get("args")
        key = (tool, canonical_args(raw_args))
        if key in seen_round:
            continue  # 同轮重复：先出现已决断，静默丢弃
        seen_round.add(key)
        if key in executed:
            # 已执行成功 → 不重跑，产出审计行
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
            cleaned.append(dict(call))  # 浅拷贝防别名，保留 reason/priority 等原键

    return cleaned, skipped
