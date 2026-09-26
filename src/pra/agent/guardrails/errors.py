"""步骤失败分级：``warn`` 为非致命失败，``critical`` 为 LLM 步 schema 校验失败。

failures 元素 shape ``{step_type, tool?, severity, reason, ts}``（``tool`` / ``seq`` 可选）。
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
    """构造一条 failures 元素（``seq`` 为可选审计序号）。"""
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

