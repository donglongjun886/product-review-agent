"""步骤失败分级与关键失败判定（docs/04-graph-design.md §5 /《01》§2.4，O-2/O-3 已拍板）。

failures 元素（O-3 拍板字段）：``{step_type, tool?, severity, reason, ts}``
- ``step_type``: HYPOTHESIZE / PLAN / TOOL_CALL / REEVALUATE / DECIDE（节点名常量）；
- ``severity``:  **warn** = 仅审计/trace（如非关键工具失败 —— 不触发转人工，
  避免任何一次非关键工具抖动推高 Human Review Rate）；
  **critical** = LLM 步 schema 校验失败（degraded）｜plan 标记 required 的关键
  取证失败（MVP 无 required 标记，见 tools_node 注记）。
- O-2 拍板口径：**failures 非空不再一律 HUMAN_REVIEW** —— 仅
  ① LLM 步失败（degraded，R5_DEGRADED_OR_FAILED_STEP）与
  ② 未解决的 critical Tool 失败（``key_tool_failure``，R3_KEY_TOOL_FAILED）
  触发；``severity="warn"`` 只进审计与 trace。

``key_tool_failure(state, failures)``：critical TOOL_CALL 失败且其后（含 dedup 允许
的 1 次重试窗口，04 §5.2）无同 tool 成功调用 → 视为"关键证据缺失"。MVP 工具几乎
不会失败，该判定主要服务错误注入测试与真实 infra 接入后。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# ---- 受控 step_type / severity ---------------------------------------------

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
    """构造一条 failures 元素（O-3 形状）。

    ``seq`` 为可选审计序号（TOOL_CALL 失败时对应该次调用的 tool_call_history.seq，
    供 key_tool_failure 判断"其后是否有同 tool 成功调用"）。
    """
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
    """是否存在"未解决的关键 Tool 失败"（docs/04-graph-design.md §5.2，O-2 口径）。

    判定：任一 ``severity=critical`` 的 TOOL_CALL 失败，且 tool_call_history 中
    seq 大于该失败的记录里没有同 tool 的 ``status="ok"`` 调用（含 dedup 允许的
    下一轮重试窗口）→ 关键证据缺失 → overlay 记 R3_KEY_TOOL_FAILED → HUMAN_REVIEW。
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
