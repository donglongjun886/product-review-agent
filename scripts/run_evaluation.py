"""Evaluation 正式跑分（单脚本 · 三臂）：rule（确定性初筛）/ single（单次 LLM 直判）/ agent（完整调查 Agent）。

数据集为 v3 双文件契约（cases_v3.jsonl + truth_v3.jsonl，真值独立成行）；
判卷是纯代码逐案比对（evaluation.grading），与跑分路径的 LLM 调用完全隔离。
"""

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evaluation.cli_utils import (
    load_dotenv_keys,
    ratio_cell,
    resolve_v3_data_paths,
    write_json_file,
)
from evaluation.dataset.loader import load_dataset_v3
from evaluation.dataset.schema import EvalCaseV3
from evaluation.grading.grader import grade_records
from evaluation.labeling.labeler import load_rulebook
from evaluation.metamorphic import (
    MetamorphicFailure,
    PassKMetrics,
    check_metamorphic,
    pass_k_consistency,
    perturb_paraphrase,
    perturb_risk_amplifying,
    perturb_semantics_preserving,
)
from evaluation.metrics.assertions import (
    AssertionFailure,
    check_process_assertions,
    load_citation_index,
)
from evaluation.metrics.diagnostics import AgentDiagnostics, evaluate_diagnostics
from evaluation.metrics.engineering import (
    DistributionMetrics,
    EngineeringEvaluator,
    EngineeringMetrics,
)
from evaluation.metrics.selective import SelectiveMetrics, evaluate_selective
from evaluation.record import EvalRecord
from pydantic import BaseModel, ConfigDict, Field

from pra import tools as tools_pkg
from pra.agent.graph import build_agent_graph
from pra.agent.guardrails.budget import budget_exceeded
from pra.agent.guardrails.llm_shell import LLMBackend, call_structured_llm
from pra.agent.state import build_initial_state
from pra.domain import Decision, ReviewDecision, RiskLevel, RiskType
from pra.domain.models import BudgetLimits
from pra.screening.engine import Verdict, triage

DEFAULT_DATA = "eval_data/v3"
DEFAULT_MODEL = "deepseek/deepseek-flash"
ENV_API_KEY = "DEEPSEEK_API_KEY"
ENV_BASE_URL = "DEEPSEEK_BASE_URL"
ARMS: tuple[str, ...] = ("rule", "single", "agent")
NOTE = "单次运行产物：真实 LLM 非确定性、不可重放（同数据重跑结果可不同），数字只代表本次抽样"
CORPUS_DIR = Path(__file__).resolve().parents[1] / "src" / "pra" / "rag" / "corpus"

__all__ = [
    "DEFAULT_DATA",
    "DEFAULT_MODEL",
    "NOTE",
    "SingleCallOutput",
    "main",
]


# ---------------------------------------------------------------------------
# 数据集 / 凭据 / 后端装配
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ArmFailure:
    """一臂内单个失败案：承载哪一案与什么异常，不产出 EvalRecord。"""

    eval_case_id: str
    error_type: str
    reason: str


def _resolve_api_key(cli_value: str | None) -> str | None:
    """API key 解析：``--api-key`` 优先，否则读环境变量 ``DEEPSEEK_API_KEY``。"""
    if cli_value:
        return cli_value
    env_value = os.environ.get(ENV_API_KEY, "")
    return env_value.strip() or None


def _make_real_backend(
    *,
    model: str,
    api_key: str,
    base_url: str | None,
    tools: list,
    thinking: str | None = None,
    reasoning_effort: str | None = None,
) -> LLMBackend:
    """构造真实 LLM 后端。

    :param tools: 生产工具列表（后端据此提取 function schema / 工具提示）；
    :param thinking: None → 不下发，用网关默认（enabled）；``reasoning_effort`` 同理（默认 high）。
    """
    try:
        from pra.agent.litellm_backend import LiteLLMBackend
    except ImportError as exc:
        raise RuntimeError(
            "real 模式需要 pra.agent.litellm_backend（含 litellm 依赖，见 pyproject）："
            "`uv sync` 后重试"
        ) from exc
    return LiteLLMBackend(
        model=model,
        api_key=api_key,
        base_url=base_url,
        tools=tools,
        thinking=thinking,  # type: ignore[arg-type]
        reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
    )


def _load_exclude_ids(raw: str | None, known_ids: set[str]) -> set[str]:
    """读入疑问篮子清单（JSON 字符串数组文件），校验 id 都在当前数据集中。

    :raises ValueError: 文件缺失 / 非 JSON 字符串数组 / 含数据集外的 id（显式失败，不静默跳过）。
    """
    if not raw:
        return set()
    path = Path(raw)
    if not path.is_file():
        raise ValueError(f"--exclude-ids 文件不存在: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"--exclude-ids 不是合法 JSON 数组文件: {path}（{exc}）") from exc
    if not isinstance(data, list) or not all(isinstance(x, str) and x.strip() for x in data):
        raise ValueError(f"--exclude-ids 必须是 JSON 字符串数组（eval_case_id 列表）: {path}")
    excluded = {x.strip() for x in data}
    unknown = excluded - known_ids
    if unknown:
        raise ValueError(f"--exclude-ids 有 {len(unknown)} 个不在当前数据集中: {sorted(unknown)}")
    return excluded


# ---------------------------------------------------------------------------
# 三臂
# ---------------------------------------------------------------------------

VERDICT_TO_DECISION: dict[Verdict, str] = {
    "PASS": "PASS",
    "REJECT": "REJECT",
    "COMPLEX": "HUMAN_REVIEW",
}


async def _run_rule(case: EvalCaseV3) -> EvalRecord:
    """rule 臂：``triage(case.input)`` 三分流映射为 EvalRecord。"""
    result = triage(case.input)
    hits = [{"rule_id": hit.rule_id, "name": hit.name, "detail": hit.detail} for hit in result.hits]
    evidence = [
        {
            "type": "RULE_HIT",
            "value": f"{hit.rule_id} {hit.name}: {hit.detail}",
            "extra": {"rule_id": hit.rule_id},
        }
        for hit in result.hits
    ]
    decision = VERDICT_TO_DECISION[result.verdict]
    return EvalRecord(
        eval_case_id=case.eval_case_id,
        scheme="rule",
        decision=decision,
        risk_level="NONE" if decision == "PASS" else None,
        risk_type=[],
        decision_confidence=None,
        evidence=evidence,
        policy=[],
        tool_calls_actual=[],
        cost={"llm_calls": 0, "tool_calls": 0, "tokens": 0},
        detail={"verdict": result.verdict, "hits": hits},
    )


class SingleCallOutput(BaseModel):
    """single 臂的一次 LLM 输出（直判，无证据链 / 无政策引用）。"""

    model_config = ConfigDict(extra="forbid")

    decision: Decision = Field(description="三分类裁决：PASS / REJECT / HUMAN_REVIEW")
    risk_level: RiskLevel = Field(description="风险等级（PASS 为 NONE）")
    risk_type: list[RiskType] = Field(default_factory=list, description="风险类型受控词表（PASS 为 []）")
    decision_confidence: float = Field(ge=0.0, le=1.0, description="直判自评把握（0..1，观测字段）")


async def _run_single(case: EvalCaseV3, *, llm: LLMBackend) -> EvalRecord:
    """single 臂：只喂 case 快照的一次 LLM 直判；不调工具、不过 Gate。

    调用失败 → 记为 HUMAN_REVIEW，``detail`` 带 ``error`` 与 ``attempts``。
    """
    outcome = await call_structured_llm(
        OutputModel=SingleCallOutput,
        node="single_call",
        state={"case": case.input.model_dump(mode="json")},
        llm=llm,
    )
    cost = {"llm_calls": outcome.attempts, "tool_calls": 0, "tokens": outcome.tokens}
    if outcome.model is None:
        return EvalRecord(
            eval_case_id=case.eval_case_id,
            scheme="single",
            decision="HUMAN_REVIEW",
            risk_level=None,
            risk_type=[],
            decision_confidence=None,
            evidence=[],
            policy=[],
            tool_calls_actual=[],
            cost=cost,
            detail={"error": outcome.error, "attempts": outcome.attempts},
        )
    model = outcome.model
    return EvalRecord(
        eval_case_id=case.eval_case_id,
        scheme="single",
        decision=model.decision.value,
        risk_level=model.risk_level.value,
        risk_type=[t.value for t in model.risk_type],
        decision_confidence=model.decision_confidence,
        evidence=[],
        policy=[],
        tool_calls_actual=[],
        cost=cost,
        detail={"attempts": outcome.attempts, "error": None},
    )


def _transcribe_agent(case: EvalCaseV3, final_state: dict, decision: ReviewDecision) -> EvalRecord:
    """agent 臂终态 → EvalRecord（只读 ReviewDecision / state 摘要）。"""
    history = final_state.get("tool_call_history") or []
    ok_calls = [r for r in history if isinstance(r, dict) and r.get("status") == "ok"]
    tool_names = list(dict.fromkeys(str(r.get("tool")) for r in ok_calls))
    evidence = [
        {
            "type": e.type,
            "source": e.source,
            "value": e.value,
            "weight": e.weight,
            "ref_id": e.ref_id,
            "extra": dict(e.extra or {}),
        }
        for e in decision.evidence
    ]
    trace = [
        {
            "id": h.id,
            "status": h.status.value,
            "prior": h.prior,
            "posterior": h.posterior,
            "statement": h.statement,
        }
        for h in decision.hypothesis_trace
    ]
    budget = decision.budget_used
    return EvalRecord(
        eval_case_id=case.eval_case_id,
        scheme="agent",
        decision=decision.decision.value,
        risk_level=decision.risk_level.value,
        risk_type=[t.value for t in decision.risk_type],
        decision_confidence=decision.decision_confidence,
        evidence=evidence,
        policy=list(decision.policy),
        tool_calls_actual=tool_names,
        cost={
            "llm_calls": budget.llm_calls,
            "tool_calls": budget.tool_calls,
            "tokens": budget.tokens,
        },
        detail={
            "overrides": list(decision.overrides),
            "budget_hit_dim": budget_exceeded(budget),
            "hypothesis_trace": trace,
            "tool_history": [
                {
                    "tool": h.get("tool"),
                    "status": h.get("status"),
                    "evidence_added": list(h.get("evidence_added") or []),
                    "decision_changed": bool(h.get("decision_changed")),
                }
                for h in history
                if isinstance(h, dict)
            ],
        },
    )


async def _run_agent(case: EvalCaseV3, *, graph: Any) -> EvalRecord:
    """agent 臂：完整调查图跑一案，终态 ``ReviewDecision`` 转录 EvalRecord。"""
    initial_state = build_initial_state(case.input)
    final_state = await graph.ainvoke(
        initial_state,
        {"configurable": {"thread_id": f"eval-agent-{case.eval_case_id}"}},
    )
    decision: ReviewDecision | None = final_state.get("decision")
    if decision is None:
        raise RuntimeError(
            f"Agent 图执行完成但终态缺少 decision（eval_case_id={case.eval_case_id}）"
        )
    return _transcribe_agent(case, final_state, decision)


# ---------------------------------------------------------------------------
# 调度
# ---------------------------------------------------------------------------


async def _run_arm(
    cases: list[EvalCaseV3],
    run_one: Callable[[EvalCaseV3], Awaitable[EvalRecord]],
    *,
    concurrency: int,
    progress: bool,
) -> tuple[list[tuple[EvalRecord, float]], list["_ArmFailure"]]:
    """并发跑一臂，返回 ``(按用例原序的 (record, 进程内墙钟毫秒) 列表, 失败案列表)``。"""
    sem = asyncio.Semaphore(max(1, concurrency))
    total = len(cases)
    done = 0

    async def _one(case: EvalCaseV3) -> tuple[EvalRecord, float] | _ArmFailure:
        nonlocal done
        async with sem:
            t0 = time.monotonic()
            try:
                rec = await run_one(case)
            except Exception as exc:  # noqa: BLE001
                elapsed_ms = (time.monotonic() - t0) * 1000
                done += 1
                if progress:
                    print(
                        f"  [{done}/{total}] {case.eval_case_id:<12} → FAILED "
                        f"({type(exc).__name__}) ({elapsed_ms / 1000:.1f}s)",
                        flush=True,
                    )
                return _ArmFailure(
                    eval_case_id=case.eval_case_id,
                    error_type=type(exc).__name__,
                    reason=str(exc),
                )
            elapsed_ms = (time.monotonic() - t0) * 1000
            done += 1
            if progress:
                print(
                    f"  [{done}/{total}] {case.eval_case_id:<12} → {rec.decision:<12} "
                    f"({elapsed_ms / 1000:.1f}s)",
                    flush=True,
                )
            return rec, elapsed_ms

    settled = list(await asyncio.gather(*[_one(c) for c in cases]))
    ok: list[tuple[EvalRecord, float]] = [
        item for item in settled if not isinstance(item, _ArmFailure)
    ]
    failures: list[_ArmFailure] = [item for item in settled if isinstance(item, _ArmFailure)]
    return ok, failures


# ---------------------------------------------------------------------------
# 报告渲染
# ---------------------------------------------------------------------------


def _num(v: float | None) -> str:
    """分布单元格：None → ``-``，否则紧凑数值。"""
    return "-" if v is None else f"{v:g}"


def _headline_row(
    arm: str, m: SelectiveMetrics, eng: EngineeringMetrics, failed: int
) -> list[str]:
    """头条表的一行：selective prediction 三指标（带分子/分母）+ 失败案数 + 成本均值。"""
    return [
        arm,
        ratio_cell(m.coverage, m.auto_decided, m.total),
        ratio_cell(m.selective_risk, m.auto_wrong, m.auto_decided),
        ratio_cell(m.false_reject_rate, m.false_rejects, m.truth_pass_total),
        f"{failed}",
        f"llm={_num(eng.llm_calls.mean)} tool={_num(eng.tool_calls.mean)} tok={_num(eng.tokens.mean)}",
    ]


_HEADLINE_HEADERS = [
    "arm",
    "coverage",
    "selective_risk",
    "false_reject_rate",
    "failed",
    "成本均值(llm/tool/tok)",
]


def _table(headers: list[str], rows: list[list[str]]) -> str:
    """渲染左对齐文本表：列宽 = 该列最大单元格宽度，按传入顺序输出表头与数据行。"""
    widths = [max([len(head), *(len(row[i]) for row in rows)]) for i, head in enumerate(headers)]
    lines = ["  ".join(head.ljust(w) for head, w in zip(headers, widths))]
    lines.extend("  ".join(cell.ljust(w) for cell, w in zip(row, widths)) for row in rows)
    return "\n".join(lines)


def _latency_cell(d: DistributionMetrics | None) -> str:
    """进程内墙钟三元组 ``mean/p50/p95``（毫秒；None → ``-``）。"""
    if d is None:
        return "-"
    return f"{_num(d.mean)}/{_num(d.p50)}/{_num(d.p95)}"


def _render_report(
    *,
    data_path: str,
    model: str,
    cases: list[EvalCaseV3],
    selective: dict[str, SelectiveMetrics],
    diagnostics: AgentDiagnostics,
    assertions: dict[str, list[AssertionFailure]],
    engineering: dict[str, EngineeringMetrics],
    failures: dict[str, list["_ArmFailure"]],
    excluded: set[str],
    metamorphic: dict | None,
    pass_k: PassKMetrics | None,
    pass_k_runs: int,
    llm_config: dict,
    out_path: str | None,
) -> str:
    """渲染 Console 报告：头条（selective prediction 三指标）+ 诊断区 + 过程断言 +
    可选蜕变/pass^k + 工程指标 + 失败案明细。"""
    out: list[str] = []
    add = out.append
    m0 = selective["rule"]
    add("=" * 100)
    add("商品审核 Agent · Evaluation 跑分：rule（确定性初筛）/ single（单次 LLM 直判）/ agent（完整调查）")
    add("=" * 100)
    add(f"数据集: {data_path}（{len(cases)} 条）| 模型: {model}")
    add("工具世界: 生产装配（真 MySQL + 真 RAG）—— rule / single 臂不使用工具")
    if any(llm_config.values()):
        add(
            "思考配置: "
            f"thinking={llm_config.get('thinking') or '默认'} / "
            f"reasoning_effort={llm_config.get('reasoning_effort') or '默认'}"
            "（非网关默认档，判定分布与成本/墙钟可能不同，勿与默认档结果混读）"
        )
    else:
        add("思考配置: 网关默认（thinking=enabled / reasoning_effort=high，本次未覆盖）")
    add("-" * 100)
    add("头条指标（selective prediction 口径，docs/02 §5.1；比率形如 值(分子/分母)，分母为 0 → '-'）:")
    add(f"  coverage = 自动决策案 / 全部案（剔除疑问篮子 {len(excluded)} 案）")
    add("  selective_risk = 自动决策中判错案 / 被自动决策的案（转人工不进分母，只压低 coverage；")
    add("      判错 = 裁决与真值不一致 ∨ 命中过程断言 —— 无证据引用的拒绝、应触发的 Gate 未触发均判错）")
    add("  false_reject_rate = 真值 PASS 被自动 REJECT 案 / 真值 PASS 案")
    add("  失败案（单案基础设施异常）不产出记录：全部分母均**不含失败案**")
    rows = [
        _headline_row(arm, selective[arm], engineering[arm], len(failures[arm])) for arm in ARMS
    ]
    add(_table(_HEADLINE_HEADERS, rows))
    add(
        f"  分母注记: 全部案 {m0.total} / 被自动决策案（各臂不同）/ 真值 PASS 案 "
        f"{m0.truth_pass_total} —— 三组分母不同，勿混读（均不含失败案、已剔除疑问篮子）"
    )
    add("-" * 100)
    add("诊断区（agent 臂；不进头条 —— 回答'结论对了，理由对不对'，docs/02 §5.2）:")
    add(
        "  工具选择准确率 "
        f"{ratio_cell(diagnostics.tool_selection_accuracy, diagnostics.tool_covered_cases, diagnostics.tool_expected_cases)}"
        "（分母=标注期望工具的案）  多余调用率 "
        f"{ratio_cell(diagnostics.redundant_tool_rate, diagnostics.tool_redundant_cases, diagnostics.tool_expected_cases)}"
        "（同分母）"
    )
    add(
        "  风险类型命中率 "
        f"{ratio_cell(diagnostics.risk_type_hit_rate, diagnostics.risk_type_hit_cases, diagnostics.risk_type_labeled_cases)}"
        "（分母=标注期望风险类型的案）  有效取证率 "
        f"{ratio_cell(diagnostics.evidence_gain_rate, diagnostics.calls_with_new_evidence, diagnostics.ok_tool_calls)}"
        "（分母=成功的工具调用）"
    )
    add("-" * 100)
    add("过程断言（无证据引用的拒绝 / 应触发 Gate 未触发 / 超预算 / 引用不可解析 / 静默降级；命中计入 selective_risk 判错）:")
    for arm in ARMS:
        arm_failures = assertions[arm]
        if not arm_failures:
            add(f"  {arm:<7} 无断言失败")
            continue
        counts: dict[str, int] = {}
        for failure in arm_failures:
            counts[failure.code] = counts.get(failure.code, 0) + 1
        add(
            f"  {arm:<7} 断言失败 {len(arm_failures)} 条: "
            + ", ".join(f"{code}×{n}" for code, n in sorted(counts.items()))
        )
        for failure in arm_failures:
            add(f"    {failure.eval_case_id:<12} {failure.code}: {failure.detail[:80]}")
    if metamorphic is not None:
        add("-" * 100)
        add("蜕变测试（agent 臂；保语义扰动（改无关字段 / 换表述）裁决不得翻转 / 风险增强扰动裁决不得变松）:")
        mm_failures: list[MetamorphicFailure] = metamorphic["failures"]
        add(f"  比对案数 {metamorphic['compared_cases']}，违例 {len(mm_failures)} 条")
        for failure in mm_failures:
            add(f"    {failure.eval_case_id:<12} {failure.kind}: {failure.original}→{failure.perturbed}")
    if pass_k is not None:
        add("-" * 100)
        add(f"pass^k（agent 臂重复 {pass_k_runs} 次，裁决一致率）:")
        add(
            f"  consistency_rate "
            f"{ratio_cell(pass_k.consistency_rate, pass_k.consistent_cases, pass_k.compared_cases)}"
            "（分母=有 ≥2 次运行记录的案）"
        )
    add("-" * 100)
    add("失败案明细（单案异常只废该案；不写成任何业务判定）:")
    for arm in ARMS:
        arm_failures = failures[arm]
        if not arm_failures:
            add(f"  {arm:<7} 无失败案")
            continue
        for failure in arm_failures:
            add(
                f"  {arm:<7} {failure.eval_case_id:<12} {failure.error_type}: "
                f"{failure.reason[:80]}"
            )
    add("-" * 100)
    add("工程指标（进程内墙钟：均值/P50/P95 毫秒 —— 不写进 record，跨 run 比对会漂移）:")
    for arm in ARMS:
        eng = engineering[arm]
        add(
            f"  {arm:<7} llm_calls={_latency_cell(eng.llm_calls)} "
            f"tool_calls={_latency_cell(eng.tool_calls)} tokens={_latency_cell(eng.tokens)} "
            f"latency_ms={_latency_cell(eng.latency_ms)} failed={len(failures[arm])}"
        )
    add("-" * 100)
    out_note = f" | JSON 已写入: {out_path}" if out_path else " | 未写文件（--out 可落盘）"
    add(f"[NOTE] {NOTE}{out_note}")
    add("=" * 100)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluation 跑分：rule / single / agent 三臂同一数据集；agent 走生产装配"
            "（真 MySQL + 真 RAG），single / agent 需 API key 且非确定性（建议先 --limit 冒烟）"
        )
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help=f"评测集：v3 目录或 cases_v3.jsonl 文件（默认 {DEFAULT_DATA} → cases_v3.jsonl + truth_v3.jsonl）",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"LLM 模型（默认 {DEFAULT_MODEL}）"
    )
    parser.add_argument(
        "--thinking",
        choices=("enabled", "disabled"),
        default=None,
        help=(
            "思考模式开关（默认 None = 不下发，用网关默认 enabled）。disabled = 非思考模式："
            "输出与墙钟大幅下降，但 temperature 才生效、判定分布可能变 —— 结果不得与默认口径混读"
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "high", "max"),
        default=None,
        help=(
            "思考强度（默认 None = 不下发，用网关默认 high）。none = 等价关闭思考模式；"
            "low 比 high 想得少（更快更省）"
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=f"API key（默认 None → 读环境变量 {ENV_API_KEY}）",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="API base URL 覆盖（默认 None → 用后端默认端点）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只跑数据集前 N 条（文件行序稳定；默认 None = 全量）",
    )
    parser.add_argument(
        "--ids",
        default=None,
        help=(
            '只跑指定 eval_case_id（逗号分隔，如 "EC_V3_0007,EC_V3_0101"；默认 None = 不过滤）。'
            "加载后按 eval_case_id 过滤，可与 --limit 叠加（先 --ids 过滤、再 --limit 截断）；"
            "id 不在数据集中会报错提示"
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="结果 JSON 写出路径（目录需已存在；默认不写文件只打印）",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "并发度（默认 1 = 逐案串行）。>1 时三臂并发跑：用例之间天然隔离（每案独立初始"
            "state、thread_id 唯一，图无 checkpointer、无跨案可变状态），只改调度、不改判定/"
            "指标/数据"
        ),
    )
    parser.add_argument(
        "--exclude-ids",
        default=None,
        help=(
            "疑问篮子清单文件（JSON 字符串数组，元素为 eval_case_id；默认 None = 空）。"
            "清单内案从头条指标全部分母剔除；id 不在数据集中会报错"
        ),
    )
    parser.add_argument(
        "--metamorphic",
        action="store_true",
        help=(
            "蜕变测试：agent 臂对每案加跑保语义（改无关字段 / 换表述）与风险增强三类扰动，"
            "比对裁决是否翻转/变松（3× 案数的真实 LLM 调用，费用自负；rule 臂确定性无需重复）"
        ),
    )
    parser.add_argument(
        "--pass-k",
        type=int,
        default=1,
        help=(
            "pass^k：agent 臂共重复运行 k 次，统计逐案裁决一致率"
            "（默认 1 = 不重复；k>1 时真实 LLM 调用费用 ×k，费用自负）"
        ),
    )
    return parser.parse_args(argv)


def _select_cases(cases: list[EvalCaseV3], *, ids: str | None, limit: int | None) -> list[EvalCaseV3]:
    """按 ``--ids`` 过滤、按 ``--limit`` 截断（先过滤后截断）；结果为空即报错。"""
    if ids:
        wanted = {s.strip() for s in ids.split(",") if s.strip()}
        if not wanted:
            raise ValueError('--ids 为空：请用逗号分隔的 eval_case_id，如 --ids "EC_V3_0007,EC_V3_0101"')
        missing = wanted - {c.eval_case_id for c in cases}
        if missing:
            raise ValueError(f"--ids 有 {len(missing)} 个不在当前数据集中: {sorted(missing)}")
        cases = [c for c in cases if c.eval_case_id in wanted]
    if limit is not None and limit > 0:
        cases = cases[:limit]
    if not cases:
        raise ValueError("评测运行无有效 case（数据集为空或 --limit/--ids 截成空）")
    return cases


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    load_dotenv_keys((ENV_API_KEY, ENV_BASE_URL))
    cases_path, truth_path = resolve_v3_data_paths(args.data)
    dataset = load_dataset_v3(cases_path, truth_path)
    cases = _select_cases(dataset.cases, ids=args.ids, limit=args.limit)
    selected_ids = {c.eval_case_id for c in cases}
    truth = [t for t in dataset.truth if t.eval_case_id in selected_ids]

    api_key = _resolve_api_key(args.api_key)
    base_url = args.base_url or os.environ.get(ENV_BASE_URL, "") or None
    if api_key is None:
        raise ValueError(
            f"未检测到 API key（--api-key 或环境变量 {ENV_API_KEY} / 仓库根 .env）。"
            "single / agent 臂必须配置 API key：无 key 本地网关（仅 --base-url）当前不支持"
            "（LiteLLMBackend 强制 api_key）——请配置真实 key 后重跑"
        )

    tools = tools_pkg.build_production_tools()
    backend = _make_real_backend(
        model=args.model,
        api_key=api_key,
        base_url=base_url,
        tools=tools,
        thinking=args.thinking,
        reasoning_effort=args.reasoning_effort,
    )
    graph = build_agent_graph(tools=tools, llm=backend)
    llm_config = {"thinking": args.thinking, "reasoning_effort": args.reasoning_effort}

    print("=" * 100)
    print("商品审核 Agent · Evaluation 跑分（rule / single / agent 三臂）")
    print("=" * 100)
    print(f"数据集: {cases_path}（{len(cases)} 条）| 模型: {args.model}")
    print("工具世界: 生产装配（真 MySQL + 真 RAG；rule / single 臂不使用工具）")
    print(
        "思考配置: "
        + (
            f"thinking={args.thinking or '默认'} / reasoning_effort={args.reasoning_effort or '默认'}"
            + ("（非网关默认档，判定分布可能与默认档不同）" if any(llm_config.values()) else "")
        )
    )
    print("API key: 已配置（--api-key / 环境变量 / .env）（值不入日志/报告/JSON）")
    print(
        "[NOTE] single / agent 臂真实调用 LLM（有费用、非确定性、不可重放）；"
        f"建议先 --limit 10 冒烟 —— 本次跑 {len(cases)} 条"
    )
    print(
        "[NOTE] 三臂预算恒为生产默认档（评测不覆盖 Guardrail）：agent 臂超限 → HUMAN_REVIEW "
        "是生产语义，归因见 R3 截胡维度"
    )
    print("-" * 100)

    excluded = _load_exclude_ids(args.exclude_ids, {c.eval_case_id for c in dataset.cases})
    if args.pass_k < 1:
        raise ValueError(f"--pass-k 必须 ≥ 1（实际 {args.pass_k}）")
    budget_limits = BudgetLimits()
    citations = load_citation_index(CORPUS_DIR)
    arm_runs: dict[str, tuple[list[tuple[EvalRecord, float]], list[_ArmFailure]]] = {}

    async def _run_and_report(
        arm: str, run_one: Callable[[EvalCaseV3], Awaitable[EvalRecord]], *, progress: bool
    ) -> None:
        """跑一臂并打印完成/失败计数与失败原因（含异常类型短名，同时进报告）。"""
        results, failures = await _run_arm(
            cases, run_one, concurrency=args.concurrency, progress=progress
        )
        arm_runs[arm] = (results, failures)
        print(f"  → {len(results)} 案完成 / {len(failures)} 案失败")
        for failure in failures:
            print(
                f"  !! {failure.eval_case_id:<12} {failure.error_type}: {failure.reason}",
                file=sys.stderr,
            )

    print("① rule（pra.screening 确定性初筛 · 零 LLM 零工具）")
    await _run_and_report("rule", _run_rule, progress=False)
    print("-" * 100)
    print(f"② single（{args.model} · 一次调用 · 只喂 case 快照 · 不调工具）")
    await _run_and_report(
        "single", lambda case: _run_single(case, llm=backend), progress=True
    )
    print("-" * 100)
    print(f"③ agent（{args.model} · 完整调查图 · 生产装配：真 MySQL + 真 RAG）")
    await _run_and_report(
        "agent", lambda case: _run_agent(case, graph=graph), progress=True
    )

    records = {arm: [rec for rec, _ in arm_runs[arm][0]] for arm in ARMS}
    latencies = {arm: [ms for _, ms in arm_runs[arm][0]] for arm in ARMS}
    failures = {arm: arm_runs[arm][1] for arm in ARMS}
    gradings = {arm: grade_records(records[arm], truth) for arm in ARMS}
    blacklisted_brands = [str(b) for b in load_rulebook()["blacklisted_brands"]]
    assertions = {
        arm: check_process_assertions(
            records[arm],
            truth,
            scheme=arm,
            citations=citations,
            blacklisted_brands=blacklisted_brands,
            max_llm_calls=budget_limits.max_llm_calls,
            max_tool_calls=budget_limits.max_tool_calls,
        )
        for arm in ARMS
    }
    selective = {
        arm: evaluate_selective(
            gradings[arm],
            exclude=excluded,
            assertion_failed={f.eval_case_id for f in assertions[arm]},
        )
        for arm in ARMS
    }
    diagnostics = evaluate_diagnostics(gradings["agent"], records["agent"])
    engineering = {
        arm: EngineeringEvaluator.evaluate(records[arm], latency_ms=latencies[arm]) for arm in ARMS
    }

    metamorphic_result: dict | None = None
    if args.metamorphic:
        agent_orig = {rec.eval_case_id: rec for rec in records["agent"]}
        perturbed_cases = [c for c in cases if c.eval_case_id in agent_orig]
        print("-" * 100)
        print(f"④ 蜕变测试（agent 臂 · {len(perturbed_cases)} 案 × 3 类扰动 · 真实 LLM 调用）")
        neutral_runs, _ = await _run_arm(
            [perturb_semantics_preserving(c) for c in perturbed_cases],
            lambda case: _run_agent(case, graph=graph),
            concurrency=args.concurrency,
            progress=True,
        )
        paraphrase_runs, _ = await _run_arm(
            [perturb_paraphrase(c) for c in perturbed_cases],
            lambda case: _run_agent(case, graph=graph),
            concurrency=args.concurrency,
            progress=True,
        )
        amplified_runs, _ = await _run_arm(
            [perturb_risk_amplifying(c) for c in perturbed_cases],
            lambda case: _run_agent(case, graph=graph),
            concurrency=args.concurrency,
            progress=True,
        )
        neutral_by_id = {rec.eval_case_id: rec for rec, _ in neutral_runs}
        paraphrase_by_id = {rec.eval_case_id: rec for rec, _ in paraphrase_runs}
        amplified_by_id = {rec.eval_case_id: rec for rec, _ in amplified_runs}
        mm_failures: list[MetamorphicFailure] = []
        compared = 0
        for cid, original in agent_orig.items():
            neutral = neutral_by_id.get(cid)
            paraphrased = paraphrase_by_id.get(cid)
            amplified = amplified_by_id.get(cid)
            if neutral is None or paraphrased is None or amplified is None:
                continue
            compared += 1
            mm_failures.extend(check_metamorphic(original, [neutral, paraphrased], amplified))
        metamorphic_result = {"compared_cases": compared, "failures": mm_failures}
        print(f"  → 比对 {compared} 案 / 违例 {len(mm_failures)} 条")

    pass_k_metrics: PassKMetrics | None = None
    if args.pass_k > 1:
        print("-" * 100)
        print(f"⑤ pass^k（agent 臂再重复 {args.pass_k - 1} 次 · 真实 LLM 调用）")
        decisions: dict[str, list[str]] = {
            rec.eval_case_id: [rec.decision] for rec in records["agent"]
        }
        for round_i in range(2, args.pass_k + 1):
            rerun, _ = await _run_arm(
                cases,
                lambda case: _run_agent(case, graph=graph),
                concurrency=args.concurrency,
                progress=True,
            )
            for rec, _ in rerun:
                decisions.setdefault(rec.eval_case_id, []).append(rec.decision)
            print(f"  → 第 {round_i}/{args.pass_k} 次运行完成（{len(rerun)} 案）")
        pass_k_metrics = pass_k_consistency(decisions)

    payload = {
        "data": str(cases_path),
        "model": args.model,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "count": len(cases),
        "real_llm_config": llm_config,
        "excluded_ids": sorted(excluded),
        "failed": {arm: len(failures[arm]) for arm in ARMS},
        "records": {arm: [rec.model_dump() for rec in records[arm]] for arm in ARMS},
        "grading": {arm: [g.model_dump() for g in gradings[arm]] for arm in ARMS},
        "metrics": {
            arm: {
                "selective": selective[arm].model_dump(),
                "engineering": engineering[arm].model_dump(),
            }
            for arm in ARMS
        },
        "diagnostics": diagnostics.model_dump(),
        "assertions": {arm: [f.model_dump() for f in assertions[arm]] for arm in ARMS},
        "metamorphic": (
            {
                "compared_cases": metamorphic_result["compared_cases"],
                "failures": [f.model_dump() for f in metamorphic_result["failures"]],
            }
            if metamorphic_result is not None
            else None
        ),
        "pass_k": (
            {"k": args.pass_k, **pass_k_metrics.model_dump()}
            if pass_k_metrics is not None
            else None
        ),
        "note": NOTE,
    }

    out_path: str | None = None
    if args.out:
        out_path = str(args.out)
        write_json_file(Path(out_path), payload)
    print()
    print(
        _render_report(
            data_path=str(cases_path),
            model=args.model,
            cases=cases,
            selective=selective,
            diagnostics=diagnostics,
            assertions=assertions,
            engineering=engineering,
            failures=failures,
            excluded=excluded,
            metamorphic=metamorphic_result,
            pass_k=pass_k_metrics,
            pass_k_runs=args.pass_k,
            llm_config=llm_config,
            out_path=out_path,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_main(argv))
    except Exception as exc:
        print(f"[FAIL] 跑分失败: {exc!r}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
