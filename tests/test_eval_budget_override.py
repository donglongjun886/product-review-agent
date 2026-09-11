"""评测装配层可配 LLM 调用预算（``max_llm_calls``）的装配 / 默认 / 边界 / 端到端 / CLI 测试。

生产护栏固定 10/15/40000/30000（``BudgetLimits`` 契约不改），只动评测装配层。覆盖：
覆盖走 ``model_copy`` 只改注入臂 limits（原对象不动），None/{} 原样返回；``>=`` 达限即
超，cap=12 时第 12 次才截胡、默认仍第 10 次，cap=2 第 2 次截胡；v2 不收敛案
（EC_V2_0260）在 cap 10/12/15 下恰好打满对应上限；覆盖不跨实例泄漏到对照臂；
CLI ``--llm-budget`` 缺省 None。

未知键（拼错）在构造与 ``_apply_budget_limits`` 都抛 ValueError —— pydantic
``model_copy`` 不校验未知键，会静默挂属性而覆盖不生效；cap 是节点级护栏，单节点
schema 重试（attempts=2）时至多越 1 次；R3 案在 ``detail["budget_hit_dim"]`` 附先撞
限维度，overrides 码字面不变。全程离线、确定性。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from helpers import hypothesize_json
from pra.agent.guardrails.budget import DIM_LLM_CALLS, budget_exceeded, bump_llm_usage
from pra.agent.guardrails.llm_shell import LLMBackendError, LLMResponse
from pra.agent.state import build_initial_state
from pra.domain.models import Budget, BudgetLimits
from pra.evaluation.dataset.loader import load_dataset
from pra.evaluation.harness.agent_scheme import AgentScheme
from pra.evaluation.harness.base import EvalContext, EvalRecord

DATA_PATH = Path(__file__).resolve().parents[1] / "eval_data" / "v1" / "cases_v1.jsonl"
DATA_PATH_V2 = Path(__file__).resolve().parents[1] / "eval_data" / "v2" / "cases_v2.jsonl"
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_evaluation_real.py"

_REAL_SCRIPT_MODULE = None


def _real_script():
    """把 scripts/run_evaluation_real.py 当模块加载（只取解析/装配函数，不跑 main）。"""
    global _REAL_SCRIPT_MODULE
    if _REAL_SCRIPT_MODULE is None:
        spec = importlib.util.spec_from_file_location("_pra_real_cli_under_test", SCRIPT_PATH)
        assert spec is not None and spec.loader is not None, f"无法加载 {SCRIPT_PATH}"
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # 模块级仅常量/函数定义；LLM 后端为延迟 import
        _REAL_SCRIPT_MODULE = mod
    return _REAL_SCRIPT_MODULE


def _v1_case(case_id: str = "EC_0001"):
    """v1 确定性收敛案（默认 EC_0001：自然 7 次 LLM 调用收敛 PASS，实测值）。"""
    cases = load_dataset(DATA_PATH)
    for c in cases:
        if c.eval_case_id == case_id:
            return c
    return cases[0]  # id 变更兜底：任一确定性收敛案均可（全 v1 自然 6-7 次 < 10）


def test_override_sets_initial_state_llm_cap_and_preserves_other_limits():
    case = _v1_case()
    state = build_initial_state(case.input)
    original_budget = state["budget"]
    original_limits = state["budget"].limits
    assert original_limits.max_llm_calls == 10  # 契约默认（生产护栏）不变

    out = AgentScheme._apply_budget_limits(state, {"max_llm_calls": 12})
    limits = out["budget"].limits
    assert limits.max_llm_calls == 12
    assert (limits.max_tool_calls, limits.max_tokens, limits.max_latency_ms) == (
        15,
        40000,
        30000,
    )
    # 不改生产 Budget/BudgetLimits 对象：覆盖走 model_copy 逐层拷贝，原对象未被改动
    assert original_limits.max_llm_calls == 10
    assert original_budget.limits.max_llm_calls == 10
    assert out["budget"] is not original_budget
    assert out["budget"].limits is not original_limits


def test_default_none_or_empty_keeps_production_10_and_returns_same_object():
    scheme = AgentScheme()
    assert scheme._budget_limits is None  # 缺省 = 不注入覆盖 → 行为逐字节不变
    case = _v1_case()
    state = build_initial_state(case.input)
    assert state["budget"].limits.max_llm_calls == BudgetLimits().max_llm_calls == 10
    assert AgentScheme._apply_budget_limits(state, None) is state  # 零覆盖 → 原样返回
    assert AgentScheme._apply_budget_limits(state, {}) is state


def test_guardrail_fires_on_12th_call_not_10th_with_cap_12():
    limits12 = BudgetLimits(max_llm_calls=12)
    assert budget_exceeded(Budget(llm_calls=10, limits=limits12)) is None  # 第 10 次不截胡
    assert budget_exceeded(Budget(llm_calls=11, limits=limits12)) is None  # 第 11 次不截胡
    assert budget_exceeded(Budget(llm_calls=12, limits=limits12)) == DIM_LLM_CALLS  # 第 12 次才截胡
    # 无覆盖（生产默认 10）仍在第 10 次截胡 —— 覆盖不漂移默认语义
    assert budget_exceeded(Budget(llm_calls=10)) == DIM_LLM_CALLS


async def test_e2e_interception_happens_at_override_cap_not_at_10():
    ctx = EvalContext()
    case = _v1_case()
    default = await AgentScheme().run(case, ctx)  # 对照臂：无覆盖
    wide = await AgentScheme(budget_limits={"max_llm_calls": 12}).run(case, ctx)
    tight = await AgentScheme(budget_limits={"max_llm_calls": 2}).run(case, ctx)

    # 案件自然收敛需求（EC_0001 = 7 次）< 10 → 默认 10 与 cap 12 都不截胡、结果一致
    assert default.cost["llm_calls"] == wide.cost["llm_calls"]
    assert default.cost["llm_calls"] < 10
    assert default.decision == wide.decision
    assert default.detail["overrides"] == [] and wide.detail["overrides"] == []

    # cap=2（≪ 收敛需求）→ 恰好第 2 次 LLM 调用后被确定性护栏截胡转人工（不在 10）
    assert tight.decision == "HUMAN_REVIEW"
    assert tight.cost["llm_calls"] == 2
    assert "R3_BUDGET_EXHAUSTED" in tight.detail["overrides"]


async def test_override_isolated_to_injected_arm_scripted_control_stays_default():
    """模拟 run_evaluation_real.py 两臂装配：real 臂注入覆盖，scripted 臂 AgentScheme()。"""
    ctx = EvalContext()
    case = _v1_case()
    real_arm = AgentScheme(budget_limits={"max_llm_calls": 2})  # "real 臂"（预算档 2）
    control = AgentScheme()  # "scripted 对照臂"（恒默认）

    r_real = await real_arm.run(case, ctx)
    r_ctl = await control.run(case, ctx)

    assert r_real.cost["llm_calls"] == 2 and r_real.decision == "HUMAN_REVIEW"
    assert r_real.detail["overrides"] == ["R3_BUDGET_EXHAUSTED"]
    # 对照臂跑满自然调查（无 R3、调用数 > 被截胡臂）—— 覆盖不跨实例泄漏
    assert r_ctl.detail["overrides"] == []
    assert r_ctl.cost["llm_calls"] > r_real.cost["llm_calls"]
    assert r_ctl.decision in {"PASS", "REJECT"}


async def test_e2e_nonconverging_case_interception_follows_cap_12_not_10():
    """确定性不收敛案（v2 EC_V2_0260：脚本桩下持续烧预算）—— 截胡次数 == 生效上限。

    v2 全量 320 案中恰有 28 案在默认预算 10 下确定性打满转人工
    （R3_BUDGET_EXHAUSTED）；cap 抬到 12/15 后仍 28/28 烧满新上限 ⇒ 归因是
    "收敛逻辑差"而非"预算紧"。此处用 EC_V2_0260 证明**覆盖后的上限才是截胡边界**：
    默认 10 → 恰好第 10 次；cap 12 → 第 12 次；cap 15 → 第 15 次。
    """
    ctx = EvalContext()
    cases = load_dataset(DATA_PATH_V2)
    case = next(c for c in cases if c.eval_case_id == "EC_V2_0260")

    at10 = await AgentScheme().run(case, ctx)  # 生产默认 10
    at12 = await AgentScheme(budget_limits={"max_llm_calls": 12}).run(case, ctx)
    at15 = await AgentScheme(budget_limits={"max_llm_calls": 15}).run(case, ctx)

    assert at10.cost["llm_calls"] == 10
    assert at10.decision == "HUMAN_REVIEW"
    assert "R3_BUDGET_EXHAUSTED" in at10.detail["overrides"]
    assert at12.cost["llm_calls"] == 12
    assert at12.decision == "HUMAN_REVIEW"
    assert "R3_BUDGET_EXHAUSTED" in at12.detail["overrides"]
    assert at15.cost["llm_calls"] == 15
    assert at15.decision == "HUMAN_REVIEW"


def test_cli_default_parses_none_and_assembly_byte_identical_to_before():
    mod = _real_script()
    args = mod._parse_args([])
    assert args.llm_budget is None  # 缺省 → 不覆盖 → 生产默认 10
    assert args.max_latency_ms == 600000
    # 缺省装配 = 改动前唯一的 {"max_latency_ms": ...}（逐字节一致，只放宽墙钟）
    assert mod._build_budget_limits(
        max_latency_ms=args.max_latency_ms, llm_budget=args.llm_budget
    ) == {"max_latency_ms": 600000}


def test_cli_llm_budget_flag_maps_to_max_llm_calls_and_feeds_agent_scheme():
    mod = _real_script()
    assert mod._parse_args(["--llm-budget", "12"]).llm_budget == 12
    assert mod._parse_args(["--llm-budget", "10"]).llm_budget == 10  # 生产默认档显式化（sweep 锚点）

    limits = mod._build_budget_limits(max_latency_ms=600000, llm_budget=12)
    assert limits == {"max_latency_ms": 600000, "max_llm_calls": 12}
    assert mod._build_budget_limits(max_latency_ms=600000, llm_budget=15)["max_llm_calls"] == 15

    # 装配产物直接投 AgentScheme（real 臂构造路径，同 _main）→ 覆盖被持有
    scheme = AgentScheme(budget_limits=limits)
    assert scheme._budget_limits == {"max_latency_ms": 600000, "max_llm_calls": 12}


def test_budget_limits_unknown_key_raises_value_error():
    """未知键（拼错字）在装配与 ``_apply_budget_limits`` 都抛 ValueError。

    pydantic v2 ``model_copy(update=...)`` 对未知键不校验：会静默挂成实例多余属性而
    覆盖不生效 —— 白名单校验让装配层失败响亮，而不是以生产默认 10 跑完实验。
    """
    case = _v1_case()
    state = build_initial_state(case.input)
    with pytest.raises(ValueError) as ei:
        AgentScheme._apply_budget_limits(state, {"max_llm_call": 12})
    assert "max_llm_call" in str(ei.value)
    assert "max_llm_calls" in str(ei.value)  # 报错信息带合法键提示
    # 构造期同样拦截（装配层失败要响亮，不等 run 才暴露）
    with pytest.raises(ValueError, match="max_llm_call"):
        AgentScheme(budget_limits={"max_llm_call": 12})
    with pytest.raises(ValueError, match="未知键"):
        AgentScheme(budget_limits={"max_llm_calls": 12, "max_tool_call": 3})
    # 合法键（部分/全部四键组合）不受影响，仍只覆盖给定键
    out = AgentScheme._apply_budget_limits(state, {"max_llm_calls": 12, "max_tokens": 50000})
    assert out["budget"].limits.max_llm_calls == 12
    assert out["budget"].limits.max_tokens == 50000
    assert out["budget"].limits.max_latency_ms == 30000  # 未给键保持默认


def test_guardrail_cap_is_node_level_can_overshoot_by_one():
    """cap 只在节点入口检查 → attempts=2 的单节点可把计数推到 cap+1。

    cap=1：入口 llm_calls=0 放行 → 单节点 bump 2（schema/transport 重试，attempts=2）
    → 下一节点入口才截胡。语义 = "节点级护栏、至多越 1 次"。
    """
    caps = BudgetLimits(max_llm_calls=1)
    b0 = Budget(llm_calls=0, limits=caps)
    assert budget_exceeded(b0) is None  # 节点入口检查：0 < 1 放行
    b1 = bump_llm_usage(b0, llm_calls=2)  # 单节点两次尝试（schema 修正/transport 重试）
    assert b1.llm_calls == 2  # == cap + 1（越过 1 次）
    assert budget_exceeded(b1) == DIM_LLM_CALLS  # 下一节点入口才截胡
    # 对照：attempts=1 时恰好 cap 截胡（不越界）
    b2 = bump_llm_usage(Budget(llm_calls=0, limits=caps), llm_calls=1)
    assert b2.llm_calls == 1
    assert budget_exceeded(b2) == DIM_LLM_CALLS


class _CapOverrunBackend:
    """cap=1 e2e 用替身（实现 LLMBackend Protocol）：hypothesize 第 1 次非法 JSON
    （触发 llm_shell schema 修正重试 → 单节点 attempts=2）、第 2 次合法；
    其它 node 永不应被调用（预算在 hypothesize 后已超限，各节点入口短路）。"""

    name = "cap-overrun-test"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, *, node, messages, json_schema):
        self.calls += 1
        if node != "hypothesize":
            raise LLMBackendError(f"测试后端不应被调用的 node: {node}（第 {self.calls} 次）")
        if self.calls == 1:
            return LLMResponse(content='{"not_valid": true', tokens=0)  # 非法 JSON
        if self.calls == 2:
            return LLMResponse(content=hypothesize_json(), tokens=0)
        raise LLMBackendError(f"hypothesize 被意外多调（第 {self.calls} 次）")


async def test_e2e_node_guardrail_cap1_attempts2_intercepts_at_cap_plus_1():
    """cap=1 时单节点 schema 重试（attempts=2）越过 cap → 截胡点 cap+1。

    确定性全图 + 替身（仅 hypothesize 生效）：hypothesize bump llm_calls=2，
    plan/decide 入口见预算已超（>= cap 1）短路 → HUMAN + R3，llm_calls==2==cap+1。
    """
    ctx = EvalContext()
    case = _v1_case()
    backend = _CapOverrunBackend()
    rec = await AgentScheme(llm=backend, budget_limits={"max_llm_calls": 1}).run(case, ctx)
    assert backend.calls == 2  # 恰好 hypothesize 两次尝试；其余节点被预算短路
    assert rec.decision == "HUMAN_REVIEW"
    assert rec.cost["llm_calls"] == 2  # == cap(1) + 1 —— 节点级护栏"至多越 1 次"
    assert "R3_BUDGET_EXHAUSTED" in rec.detail["overrides"]


async def test_record_detail_reports_budget_hit_dimension():
    """R3 命中案在 ``EvalRecord.detail["budget_hit_dim"]`` 附先撞限维度。

    维度只作记录侧附加审计（真实跑分 token 是第二截胡源时区分
    llm_calls/tokens/latency），overrides 码字面不变。
    """
    ctx = EvalContext()
    cases = load_dataset(DATA_PATH_V2)
    nonconv = next(c for c in cases if c.eval_case_id == "EC_V2_0260")
    rec = await AgentScheme().run(nonconv, ctx)  # 默认预算 10 打满截胡（确定性）
    assert rec.decision == "HUMAN_REVIEW"
    assert "R3_BUDGET_EXHAUSTED" in rec.detail["overrides"]  # 码字面不变
    assert rec.detail["budget_hit_dim"] == "LLM_CALLS"  # llm_calls 先撞限
    ok = await AgentScheme().run(_v1_case(), ctx)
    assert ok.detail["overrides"] == []
    assert ok.detail["budget_hit_dim"] is None


def test_real_overrides_summary_counts_r5_r3_and_mixed():
    """real 报告 overrides 汇总函数（R5 降级案数 / R3 案数 / 混合案 / R3 先撞维度）计数正确。"""
    mod = _real_script()

    def _rec(overrides: list, dim: str | None = None) -> EvalRecord:
        return EvalRecord(
            eval_case_id="EC_SUM",
            scheme="agent",
            decision="HUMAN_REVIEW",
            detail={"overrides": list(overrides), "budget_hit_dim": dim},
        )

    records = [
        _rec(["R5_DEGRADED_OR_FAILED_STEP"]),
        _rec(["R5_DEGRADED_OR_FAILED_STEP", "R3_BUDGET_EXHAUSTED"], dim="TOKENS"),  # 混合案
        _rec(["R3_BUDGET_EXHAUSTED"], dim="LLM_CALLS"),
        _rec(["R3_BUDGET_EXHAUSTED"], dim=None),  # 缺维度 → UNKNOWN（不静默丢案）
        _rec(["R2_REJECT_GATE_FAIL"]),
        _rec([]),  # 无码案不计数
    ]
    s = mod._overrides_summary(records)
    assert s["cases_with_any"] == 5
    assert s["R5_DEGRADED_OR_FAILED_STEP"] == 2
    assert s["R3_BUDGET_EXHAUSTED"] == 3
    assert s["r3_r5_mixed"] == 1
    # token 与 llm_calls 必须分维度可见（真实跑分里两者都可能是截胡源）
    assert s["budget_hit_dims"] == {"LLM_CALLS": 1, "TOKENS": 1, "UNKNOWN": 1}
    assert s["other_codes"] == {"R2_REJECT_GATE_FAIL": 1}
    empty = mod._overrides_summary([_rec([]), _rec([])])
    assert empty["cases_with_any"] == 0 and empty["R5_DEGRADED_OR_FAILED_STEP"] == 0
    assert empty["budget_hit_dims"] == {}
