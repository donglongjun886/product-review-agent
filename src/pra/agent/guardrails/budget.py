"""预算护栏（Guardrail）—— 四维检查与记账（docs/04-graph-design.md §2.4 /《01》§6.4）。

对齐《00》§8.1 语义：**预算上限是 Guardrail 上界、不是目标调用次数**。正常案件
（走查常态 8 次 LLM / 5 次 Tool）明显低于上限（10/15/40000/30000，T-7 拍板），
余量只用于"schema 校验失败→重试 1 次"、工具失败恢复与防无限循环。超限语义 =
"调查成本已超过可接受范围，证据不足以自动判，转人工最稳妥" —— 正确的业务行为，
不是失败（该语义由 decide overlay 以 ``R3_BUDGET_EXHAUSTED`` 归因落地，见 gate.py）。

检查点（确定性代码，全部纯函数可单测）：
- 条件边路由（route_after_plan / route_after_reevaluate）在进入下一节点前检查；
- tools_node 主循环按单批执行前检查（预算只够前 k 个时执行前 k 个）；
- decide 入口决定是否还值得再烧一次 LLM 提案。

**节点级护栏边界（P2-17，语义未改、仅文档化）**：检查只发生在**节点入口/路由**，
单次节点调用内部可再 bump —— 例如 schema 校验重试 / transport 重试把单节点
attempts 抬到 2（节点按 ``LLMCallOutcome.attempts`` 记账，见 nodes/*.py），一次节点
调用可把 ``llm_calls`` 推高 2 格。故生效截胡点可为 **上限 + 1（至多越 1 次）**：
入口 ``llm_calls == cap-1`` 的节点耗尽 2 次尝试后，要到**下一节点入口**才截胡
（最终 ``llm_calls == cap+1``）。**cap 是节点级护栏而非全案精确调用上限**；如需
严格上限须在每次 bump 前检查（当前刻意不做 —— 语义是上界，正常案件远低于上限）。

token 记账口径（P2-16 注记，本模块不改口径）：``bump_llm_usage(tokens=…)`` 收到的
tokens 由节点转交 ``LLMCallOutcome.tokens`` = 后端 ``usage.total_tokens``
（**input+output 合计**，含 provider 缓存命中 token；**schema 校验失败的尝试也全额
累计**；transport 失败无响应不计）—— 详见 llm_shell / litellm_backend 注释。

超限返回**首个**超限维度名（LLM_CALLS / TOOL_CALLS / TOKENS / LATENCY），供审计
与 overrides 记录；全部未超限返回 None。``latency`` 从 ``budget.start_time``
（UTC）按墙钟推算。
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
    """当前 UTC 时间毫秒（与 Budget.start_time 同口径：datetime 墙钟）。"""
    return int(time.time() * 1000)


def elapsed_ms(budget: Budget) -> int:
    """本轮调查已耗时（毫秒）—— 从 start_time 推算（幂等，不依赖逐步累加）。"""
    start = budget.start_time
    if start.tzinfo is None:  # naive（防御）：按 UTC 解释
        start = start.replace(tzinfo=timezone.utc)
    return max(int((datetime.now(timezone.utc) - start).total_seconds() * 1000), 0)


def budget_exceeded(budget: Budget) -> str | None:
    """返回首个超限维度（LLM_CALLS/TOOL_CALLS/TOKENS/LATENCY）；全部未超限返回 None。

    阈值语义：``>=``（达到上限即视为超限停止 —— Guardrail 上界）。
    检查时点：只在节点入口/路由调用 —— 单节点内 attempts=2 的重试可把计数器
    bump 到 cap+1 才在下一次入口截胡（节点级护栏，至多越 1 次，见模块 docstring
    P2-17）。
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
    """Tool 记账：tool_calls + 1（一次已执行的工具调用，含业务失败/异常重试后）、tokens 累加。"""
    return budget.model_copy(
        update={"tool_calls": budget.tool_calls + 1, "tokens": budget.tokens + tokens}
    )


def snapshot_budget(budget: Budget) -> Budget:
    """决策时预算快照（写入 ReviewDecision.budget_used）：补 latency_ms 后返回副本。

    不修改运行期对象 —— decide overlay 在 ``build_decision`` 内调用本函数取快照。
    """
    return budget.model_copy(update={"latency_ms": elapsed_ms(budget)})
