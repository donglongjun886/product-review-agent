"""步骤失败分级。

failures 元素 shape ``{step_type, tool?, severity, reason, ts}``：

- ``step_type`` ∈ HYPOTHESIZE / PLAN / TOOL_CALL / REEVALUATE / DECIDE；
- ``severity="warn"`` 只进审计与 trace（业务性失败（``ok=False``）记 warn —— 不让工具抖动推高
  Human Review Rate；工具 infra 异常不上抛成 failure，直接抛出）；``severity="critical"`` =
  LLM 步 schema 校验失败（degraded）。
- **failures 非空不等于一律 HUMAN_REVIEW**：只有 LLM 步失败（``degraded``，
  ``R5_DEGRADED_OR_FAILED_STEP``）触发；``warn`` 不触发。
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
    """构造一条 failures 元素（``seq`` 为可选审计序号，与 tool_call_history 对齐）。"""
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

