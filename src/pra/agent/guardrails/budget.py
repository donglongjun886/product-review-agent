"""预算护栏：四维检查与记账。

**上限是 Guardrail 上界、不是目标调用次数**。正常案件（常态 8 次 LLM / 5 次 Tool）
远低于上限 10/15/40000/30000；余量只用于 schema 校验失败重试、工具失败恢复与
防无限循环。超限 = 调查成本超可接受范围、证据不足以自动判，转人工最稳妥 ——
是正确业务行为而非失败（由 decide overlay 以 ``R3_BUDGET_EXHAUSTED`` 落地）。

检查点只有三处：条件边路由进下一节点前、tools_node 每批执行前（预算只够前 k 个
就只执行前 k 个）、decide 入口。

检查只在**节点入口/路由**，单节点内可再 bump —— schema/transport 重试把
attempts 抬到 2，可把 ``llm_calls`` 推高 2 格，故生效截胡点可为**上限 + 1**
（至多越 1 次）。cap 是节点级护栏，不是全案精确调用上限。

token 记账：tokens = 后端 ``usage.total_tokens``（input+output 合计，含 provider
缓存命中；schema 校验失败的尝试也全额累计；transport 失败无响应不计）。

超限返回**首个**超限维度名，全部未超限返回 None。
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from pra.domain.models import Budget

# 超限维度返回值（稳定字符串，供 overrides/审计引用）
DIM_LLM_CALLS = "LLM_CALLS"
DIM_TOOL_CALLS = "TOOL_CALLS"
DIM_TOKENS = "TOKENS"
DIM_LATENCY = "LATENCY"


def _now_ms() -> int:
    """当前 UTC 毫秒（与 ``Budget.start_time`` 同口径）。"""
    return int(time.time() * 1000)


def elapsed_ms(budget: Budget) -> int:
    """已耗时（毫秒）：从 ``start_time`` 按墙钟推算。"""
    start = budget.start_time
    if start.tzinfo is None:  # naive（防御）：按 UTC 解释
        start = start.replace(tzinfo=timezone.utc)
    return max(int((datetime.now(timezone.utc) - start).total_seconds() * 1000), 0)


def budget_exceeded(budget: Budget) -> str | None:
    """返回首个超限维度（LLM_CALLS/TOOL_CALLS/TOKENS/LATENCY）；全部未超限返回 None。

    阈值语义 ``>=``；只在节点入口/路由调用 —— 节点内重试可把计数器推到 cap+1 后才在
    下一次入口截胡。
    """
    if budget.llm_calls >= budget.limits.max_llm_calls:
        return DIM_LLM_CALLS
    if budget.tool_calls >= budget.limits.max_tool_calls:
        return DIM_TOOL_CALLS
    if budget.tokens >= budget.limits.max_tokens:
        return DIM_TOKENS
    if elapsed_ms(budget) >= budget.limits.max_latency_ms:
        return DIM_LATENCY
    return None


def bump_llm_usage(budget: Budget, *, llm_calls: int = 1, tokens: int = 0) -> Budget:
    """LLM 记账：llm_calls + n（含 schema 重试的每次尝试）、tokens 累加。返回新对象。"""
    return budget.model_copy(
        update={
            "llm_calls": budget.llm_calls + llm_calls,
            "tokens": budget.tokens + tokens,
        }
    )


def bump_tool_usage(budget: Budget, *, tokens: int = 0) -> Budget:
    """Tool 记账：tool_calls + 1（含业务失败/异常重试后）、tokens 累加。返回新对象。"""
    return budget.model_copy(
        update={"tool_calls": budget.tool_calls + 1, "tokens": budget.tokens + tokens}
    )


def snapshot_budget(budget: Budget) -> Budget:
    """决策时预算快照（写入 ``ReviewDecision.budget_used``）：补 latency_ms 后返回副本。"""
    return budget.model_copy(update={"latency_ms": elapsed_ms(budget)})
