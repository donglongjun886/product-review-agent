"""预算护栏（guardrails/budget.py）单测：四维 >= 语义、记账不可变、快照补 latency。

阈值语义：**达到上限即视为超限**（>=，Guardrail 上界）；超限返回首个超限维度名，
未超限返回 None。
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pra.agent.guardrails.budget import (
    DIM_LLM_CALLS,
    DIM_LATENCY,
    DIM_TOKENS,
    DIM_TOOL_CALLS,
    budget_exceeded,
    bump_llm_usage,
    bump_tool_usage,
    snapshot_budget,
)
from pra.domain.models import Budget, BudgetLimits


def _old_budget(start_age_seconds: int = 90) -> Budget:
    """start_time 置于过去 start_age_seconds 秒的 Budget（其余字段默认 0/默认限额）。"""
    start = datetime.now(timezone.utc) - timedelta(seconds=start_age_seconds)
    return Budget(start_time=start)


def test_not_exceeded_when_below_limits():
    b = Budget(llm_calls=8, tool_calls=5, tokens=1000)
    assert budget_exceeded(b) is None


def test_llm_calls_at_limit_exceeded():
    """llm_calls == max（10）→ LLM_CALLS（>= 语义：达限即超）。"""
    assert budget_exceeded(Budget(llm_calls=BudgetLimits().max_llm_calls)) == DIM_LLM_CALLS


def test_llm_calls_above_limit_exceeded():
    assert budget_exceeded(Budget(llm_calls=99)) == DIM_LLM_CALLS


def test_tool_calls_at_limit_exceeded():
    """llm 未超、tool_calls == max（15）→ TOOL_CALLS。"""
    b = Budget(tool_calls=BudgetLimits().max_tool_calls)
    assert budget_exceeded(b) == DIM_TOOL_CALLS


def test_tokens_at_limit_exceeded():
    b = Budget(tokens=BudgetLimits().max_tokens)
    assert budget_exceeded(b) == DIM_TOKENS


def test_latency_at_limit_exceeded_from_start_time():
    """elapsed_ms >= max_latency_ms（30000）→ LATENCY（从 start_time 墙钟推算）。"""
    assert budget_exceeded(_old_budget(start_age_seconds=90)) == DIM_LATENCY


def test_latency_naive_start_time_treated_utc():
    """naive start_time 按 UTC 解释（防御分支）—— 同样按墙钟超限。"""
    naive_old = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(seconds=90)
    assert budget_exceeded(Budget(start_time=naive_old)) == DIM_LATENCY


def test_first_exceeded_dimension_wins():
    """多维同时超限 → 返回首个（固定顺序 LLM→TOOL→TOKENS→LATENCY）。"""
    b = Budget(llm_calls=12, tool_calls=20, tokens=999999)
    assert budget_exceeded(b) == DIM_LLM_CALLS


def test_bump_llm_usage_returns_new_object_no_mutation():
    b = Budget(llm_calls=1, tokens=100)
    bumped = bump_llm_usage(b, llm_calls=2, tokens=50)
    assert bumped is not b
    assert bumped.llm_calls == 3 and bumped.tokens == 150
    assert b.llm_calls == 1 and b.tokens == 100  # 原对象不被改动


def test_bump_tool_usage_increments_tool_and_tokens():
    b = Budget(tool_calls=4, tokens=10)
    bumped = bump_tool_usage(b, tokens=30)
    assert bumped is not b
    assert bumped.tool_calls == 5 and bumped.tokens == 40
    assert b.tool_calls == 4 and b.tokens == 10


def test_cap_one_exceeded_at_first_attempt():
    """cap=1：已用 1 次即超限（>= 语义）—— cap 为 1 时没有"先跑一次"的余量。"""
    limits = BudgetLimits(max_llm_calls=1, max_tool_calls=1)
    assert budget_exceeded(Budget(limits=limits)) is None
    assert budget_exceeded(Budget(llm_calls=1, limits=limits)) == DIM_LLM_CALLS
    assert budget_exceeded(Budget(tool_calls=1, limits=limits)) == DIM_TOOL_CALLS


def test_node_retry_can_push_counter_to_cap_plus_one():
    """cap 是**节点级**护栏：节点内 attempts=2 可把计数推到 cap+1，下一次入口才截胡。

    即 ``budget_exceeded`` 保证的是"每个节点入口处 ≤ cap"，不是"整案 ≤ cap" —— 至多越 1 次
    （见 ``budget.py`` 模块 docstring）。本用例把这条不变量钉住：入口未超 → 节点内两次
    bump → cap+1 → 下一次入口超限。
    """
    limits = BudgetLimits(max_llm_calls=2)
    b = Budget(llm_calls=1, limits=limits)  # 节点入口：1 < 2 → 允许进入
    assert budget_exceeded(b) is None
    b = bump_llm_usage(b)  # 节点内第 1 次尝试 → 达限
    assert b.llm_calls == limits.max_llm_calls
    assert budget_exceeded(b) == DIM_LLM_CALLS
    b = bump_llm_usage(b)  # 重试：节点内不复查预算 → 越到 cap+1
    assert b.llm_calls == limits.max_llm_calls + 1
    assert budget_exceeded(b) == DIM_LLM_CALLS


def test_snapshot_budget_fills_latency_ms_without_mutating():
    """快照 = 补 latency_ms 的副本；运行期对象不变（latency_ms 仍为初始 0）。"""
    b = _old_budget(start_age_seconds=90)  # latency_ms 字段默认 0
    snap = snapshot_budget(b)
    assert snap is not b
    assert snap.latency_ms >= b.limits.max_latency_ms  # 已按墙钟推算填入
    assert snap.llm_calls == b.llm_calls and snap.tool_calls == b.tool_calls
    assert b.latency_ms == 0  # 原对象未被写入 latency
    assert snap.limits is b.limits  # 限额引用共享（同一配置对象）
