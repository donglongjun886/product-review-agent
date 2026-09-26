"""plan 输出确定性去重：清洗重复的工具调用，产出 cleaned / skipped 两份。"""

from __future__ import annotations

import json


def canonical_args(args: dict) -> str:
    """args 的 canonical 序列化指纹：key 排序 + 非 JSON 原生类型按 ``str`` 规整。"""
    return json.dumps(args if args is not None else {}, sort_keys=True, default=str)


def dedup_pending(state: dict, planned: list[dict]) -> tuple[list[dict], list[dict]]:
    """清洗 plan 输出（planned）→ ``(cleaned, skipped)``。

    ``cleaned`` 为未命中「已执行成功」集合的调用（浅拷贝、保序）；``skipped`` 为命中的审计行，``seq`` 从 ``len(tool_call_history) + 1`` 起顺延。
    """
    history = state.get("tool_call_history") if isinstance(state, dict) else None
    history = history or []
    executed: set[tuple] = set()
    for rec in history:
        if isinstance(rec, dict) and rec.get("status") == "ok":
            executed.add((str(rec.get("tool")), canonical_args(rec.get("args") or {})))

    cleaned: list[dict] = []
    skipped: list[dict] = []
    seen_round: set[tuple] = set()
    seq = len(history) + 1

    for call in planned or []:
        if not isinstance(call, dict):
            continue
        tool = str(call.get("tool", ""))
        raw_args = call.get("args")
        key = (tool, canonical_args(raw_args))
        if key in seen_round:
            continue
        seen_round.add(key)
        if key in executed:
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
            cleaned.append(dict(call))

    return cleaned, skipped
