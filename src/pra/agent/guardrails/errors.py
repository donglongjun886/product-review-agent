"""步骤失败分级与关键失败判定。

failures 元素 shape ``{step_type, tool?, severity, reason, ts}``：

- ``step_type`` ∈ HYPOTHESIZE / PLAN / TOOL_CALL / REEVALUATE / DECIDE；
- ``severity="warn"`` 只进审计与 trace（如非关键工具失败 —— 不让工具抖动推高
  Human Review Rate）；``severity="critical"`` = LLM 步 schema 校验失败（degraded）
  或 plan 标记 required 的关键取证失败（当前无 required 标记）。
- **failures 非空不等于一律 HUMAN_REVIEW**：只有 LLM 步失败（``degraded``，
  ``R5_DEGRADED_OR_FAILED_STEP``）与未解决的 critical Tool 失败
  （``key_tool_failure``，``R3_KEY_TOOL_FAILED``）触发；``warn`` 不触发。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# 受控 step_type / severity

STEP_HYPOTHESIZE = "HYPOTHESIZE"
STEP_PLAN = "PLAN"
STEP_TOOL_CALL = "TOOL_CALL"
STEP_REEVALUATE = "REEVALUATE"
STEP_DECIDE = "DECIDE"

SEV_WARN = "warn"
SEV_CRITICAL = "critical"


def now_iso() -> str:
    """UTC ISO8601（failures.ts）。"""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def make_failure(
    *,
    step_type: str,
    reason: str,
    severity: str = SEV_WARN,
    tool: str | None = None,
    seq: int | None = None,
) -> dict[str, Any]:
    """构造一条 failures 元素（``seq`` 为可选审计序号，供 ``key_tool_failure`` 用）。"""
    entry: dict[str, Any] = {
        "step_type": step_type,
        "severity": severity,
        "reason": reason,
        "ts": now_iso(),
    }
    if tool is not None:
        entry["tool"] = tool
    if seq is not None:
        entry["seq"] = seq
    return entry


def key_tool_failure(state: dict[str, Any], failures: list[dict[str, Any]]) -> bool:
    """是否存在「未解决的关键 Tool 失败」。

    任一 ``severity=critical`` 的 TOOL_CALL 失败，且 ``tool_call_history`` 中 seq 更大
    的记录里没有同 tool 的 ``status="ok"`` 调用（含 dedup 允许的 1 次重试窗口）→
    关键证据缺失 → overlay 记 ``R3_KEY_TOOL_FAILED``。
    """
    history = state.get("tool_call_history") or []
    later_ok_seqs: dict[str, int] = {}  # tool -> 最近一次成功 seq
    for r in history:
        if r.get("status") == "ok" and r.get("tool"):
            later_ok_seqs[r["tool"]] = max(later_ok_seqs.get(r["tool"], 0), r.get("seq", 0))
    for f in failures:
        if f.get("step_type") == STEP_TOOL_CALL and f.get("severity") == SEV_CRITICAL:
            tool = f.get("tool")
            if not tool:
                continue
            fail_seq = f.get("seq") or 0
            if later_ok_seqs.get(tool, 0) <= fail_seq:
                return True
    return False
