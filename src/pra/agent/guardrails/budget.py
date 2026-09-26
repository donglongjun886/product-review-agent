"""预算护栏：LLM 调用 / Tool 调用两维的检查与记账。"""

from __future__ import annotations

from datetime import datetime, timezone

from pra.domain.models import Budget

# 超限维度返回值
DIM_LLM_CALLS = "LLM_CALLS"
DIM_TOOL_CALLS = "TOOL_CALLS"


def elapsed_ms(budget: Budget) -> int:
    """已耗时（毫秒）：从 ``start_time`` 按墙钟推算。"""
    start = budget.start_time
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    return max(int((datetime.now(timezone.utc) - start).total_seconds() * 1000), 0)


def budget_exceeded(budget: Budget) -> str | None:
    """返回首个超限维度（LLM_CALLS / TOOL_CALLS）；全部未超限返回 None。"""
    if budget.llm_calls >= budget.limits.max_llm_calls:
        return DIM_LLM_CALLS
    if budget.tool_calls >= budget.limits.max_tool_calls:
        return DIM_TOOL_CALLS
    return None


def bump_llm_usage(budget: Budget, *, llm_calls: int = 1, tokens: int = 0) -> Budget:
    """LLM 记账：返回累加 llm_calls / tokens 后的新对象。"""
    return budget.model_copy(
        update={
            "llm_calls": budget.llm_calls + llm_calls,
            "tokens": budget.tokens + tokens,
        }
    )


def bump_tool_usage(budget: Budget, *, tokens: int = 0) -> Budget:
    """Tool 记账：返回累加 tool_calls / tokens 后的新对象。"""
    return budget.model_copy(
        update={"tool_calls": budget.tool_calls + 1, "tokens": budget.tokens + tokens}
    )


def snapshot_budget(budget: Budget) -> Budget:
    """预算快照：补 latency_ms 后返回副本。"""
    return budget.model_copy(update={"latency_ms": elapsed_ms(budget)})
