"""real LLM vs scripted 对比跑分：真实 LLM 接进评测 harness。

把 ``pra.evaluation.harness.agent_scheme.AgentScheme(llm=...)`` 的 real 模式与确定性 scripted
基线（``EvalScriptedLLMBackend`` 审查员桩）在同一 eval 数据 / 同一工具世界上逐案对比，输出：
逐案一致性（scripted vs real + 差异明细）、决策业务指标（``DecisionEvaluator`` 口径：
Accuracy/Precision/Recall/FPR/FNR + HRR/自动化率；真值含 HUMAN_REVIEW 的案自动跳过并注明）、
``--out PATH`` 落盘的 real EvalRecord 全量 + 差异摘要，以及 overrides 归因码汇总（R5 降级 /
R3 预算截胡 / R3+R5 混合案）—— 「整卷全 HUMAN 是链路降级」一眼可见。

CLI：``--limit N`` / ``--ids "EC_0007,EC_0101"`` 定向取案子集；``--data`` 给 JSONL 或目录
（eval_data/v1 → cases_v1.jsonl）；``--world {eval,rag}`` 选工具数据源世界（rag = 真实 KB
检索，mode=hybrid）；``--model`` / ``--api-key`` / ``--base-url`` 配 LLM；``--max-latency-ms``
放宽 real 臂墙钟护栏（默认 600000=10min，生产护栏 30s 对真实 LLM 过紧，每案 ~9 次串行调用
天然 >30s，不放宽会每案 LATENCY 截胡转人工）；``--llm-budget N`` 覆盖 real 臂 max_llm_calls
档（默认 None = 生产默认 10）；``--out`` 写结果 JSON（父目录需已存在）。

成本与结论边界：

- 真实 API 有费用、非确定性：real 侧每次运行都调真实 LLM，同 case 重跑输出可能不同（不可重
  放）。先 ``--limit 10`` 冒烟确认链路与成本量级再跑全量；report / JSON 已如实标注，real 数字
  只代表单次运行抽样，勿当模型固定水平。
- scripted 结果、指标计算、JSON 结构、差异统计全程可复现（回归基线永远以 scripted =
  ``AgentScheme()`` 默认行为为准，real 只观测对照）。
- API key 读取顺序：``--api-key`` > 环境变量 ``DEEPSEEK_API_KEY`` > 仓库根 ``.env``（脚本开头
  以 setdefault 语义注入）；base-url 同理（``--base-url`` > ``DEEPSEEK_BASE_URL``）。API key
  必填 —— 缺 key 预检即报错。
- 工具数据源 = 评测种子世界（eval / RAG，与 scripted 同一世界）→ 两臂差异只归因于 LLM。
  EvalRecord 不含墙钟 latency；real 墙钟只进进程内进度打印，不落 JSON。
- ``pra.agent.litellm_backend`` 为延迟 import：缺失时本模块仍可 import，scripted / fake 干跑
  可用，只有 real 运行需要它就绪。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pra.agent.guardrails.gate import (  # overrides 归因码（汇总用，只读）
    R3_BUDGET_EXHAUSTED,
    R5_DEGRADED_OR_FAILED_STEP,
)
from pra.evaluation.dataset.loader import load_dataset, scene_stats, smoke_subset
from pra.evaluation.dataset.schema import EvalCase
from pra.evaluation.harness.agent_scheme import (
    EVAL_WORLD_LABEL,
    RAG_WORLD_LABEL,
    AgentScheme,
    make_eval_world_tools,
    make_rag_world_tools,
)
from pra.evaluation.harness.base import EvalContext, EvalRecord
from pra.evaluation.metrics.business import DecisionEvaluator, DecisionMetrics
from pra.evaluation.runner import expected_index

DEFAULT_DATA = "eval_data/v1"
DEFAULT_MODEL = "deepseek/deepseek-chat"
ENV_API_KEY = "DEEPSEEK_API_KEY"
ENV_BASE_URL = "DEEPSEEK_BASE_URL"
WORLDS = ("eval", "rag")
SCENES = ("normal", "violation", "boundary", "multi-signal", "evasion")
REAL_NOTE = "real LLM 非确定性，不可重放"

__all__ = ["DEFAULT_DATA", "DEFAULT_MODEL", "REAL_NOTE", "_resolve_data_path", "run_comparison"]


# ---------------------------------------------------------------------------
# 数据集 / 世界 / 后端装配
# ---------------------------------------------------------------------------


def _resolve_data_path(raw: str) -> Path:
    """把 ``--data`` 解析为评测 JSONL 路径（直接 JSONL，或按目录名拼
    ``cases_<目录名>.jsonl``：eval_data/v1 → cases_v1.jsonl）。"""
    p = Path(raw)
    if p.is_file():
        return p
    if p.is_dir():
        if re.fullmatch(r"v\d+", p.name) is not None:
            cand = p / f"cases_{p.name}.jsonl"
            if cand.is_file():
                return cand
        raise ValueError(
            f"评测数据目录 {p} 无法定位 cases JSONL：目录名按 v1/v2 → "
            f"cases_<目录名>.jsonl（{p.name} 不是 v<数字> 目录名或该文件缺失）"
        )
    raise ValueError(f"评测数据路径不存在: {p}（支持 JSONL 文件，或 eval_data/v1 / v2 目录）")


def _world_tools(world: str):
    """按 world 取工具列表，给 real 后端构造 ``tools`` 参数；图侧工具由
    ``AgentScheme.run`` 按 ctx.tool_world 自行装配，两处同一世界。"""
    if world == "rag":
        return make_rag_world_tools(mode="hybrid")  # rag_mode=None → hybrid（与 ctx 默认一致）
    return make_eval_world_tools()


def _world_label(world: str) -> str:
    return RAG_WORLD_LABEL if world == "rag" else EVAL_WORLD_LABEL


def _build_ctx(world: str) -> EvalContext:
    """ctx = EvalContext(tool_world=world)；rag 时 rag_mode=None → hybrid。"""
    if world == "rag":
        return EvalContext(tool_world="rag", rag_mode=None)
    return EvalContext(tool_world="eval")


def _make_real_backend(
    *, model: str, api_key: str | None, base_url: str | None, world: str
) -> Any:
    """构造真实 LLM 后端（延迟 import ``pra.agent.litellm_backend``）。

    按 ``LLMBackend`` Protocol 交给 ``AgentScheme(llm=...)``；tools = 与 world 相同的
    工具列表（供后端输出 function schema / 工具提示）。
    """
    try:
        from pra.agent.litellm_backend import LiteLLMBackend
    except ImportError as exc:  # 模块或 litellm 依赖缺失 → 明确报错而非半路 ImportError
        raise RuntimeError(
            "real 模式需要 pra.agent.litellm_backend（含 litellm 依赖，见 pyproject）："
            "`uv sync` 后重试；scripted / fake 干跑不受影响"
        ) from exc
    return LiteLLMBackend(
        model=model, api_key=api_key, base_url=base_url, tools=_world_tools(world)
    )


def _resolve_api_key(cli_value: str | None) -> str | None:
    """API key 解析：``--api-key`` 优先，否则读环境变量 ``DEEPSEEK_API_KEY``。"""
    if cli_value:
        return cli_value
    env_value = os.environ.get(ENV_API_KEY, "")
    return env_value.strip() or None


def _load_dotenv() -> None:
    """把仓库根 ``.env`` 的 DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL 注入进程环境。

    以 setdefault 语义注入（真实环境变量优先，不覆盖 CLI/既有 env），key 值不入日志/
    报告/JSON。定位方式同 ``pra.infra.db._repo_root_env_file``：从本文件上溯到含
    pyproject.toml 的仓库根；找不到 .env 或键缺失 → 静默跳过。
    """
    here = Path(__file__).resolve()
    env_file: Path | None = None
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            env_file = parent / ".env"
            break
    if env_file is None or not env_file.is_file():
        return
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key in (ENV_API_KEY, ENV_BASE_URL) and value:
            os.environ.setdefault(key, value)  # setdefault：真实 env / CLI 优先


# ---------------------------------------------------------------------------
# 核心对比流程（scripted 可复现 + real 注入后端；两臂仅 LLM 不同）
# ---------------------------------------------------------------------------


async def _run_scheme_records(
    scheme: AgentScheme,
    cases: list[EvalCase],
    ctx: EvalContext,
    *,
    progress_prefix: str | None = None,
) -> list[EvalRecord]:
    """顺序串行跑一遍 scheme；progress_prefix 非 None 时逐条打印进度。"""
    records: list[EvalRecord] = []
    total = len(cases)
    for i, case in enumerate(cases, start=1):
        t0 = time.monotonic()
        rec = await scheme.run(case, ctx)
        if progress_prefix is not None:
            print(
                f"  [{i}/{total}] {case.eval_case_id:<9} → {rec.decision:<12} "
                f"({time.monotonic() - t0:.1f}s)"
            )
        records.append(rec)
    return records


def _compare_rows(
    cases: list[EvalCase],
    scripted_records: list[EvalRecord],
    real_records: list[EvalRecord],
    exp: dict,
) -> tuple[list[dict], list[dict], dict]:
    """逐案对齐（按 eval_case_id）→ 全量行 + 差异行 + 按 scene 一致性计数。"""
    s_by_id = {r.eval_case_id: r for r in scripted_records}
    r_by_id = {r.eval_case_id: r for r in real_records}
    rows: list[dict] = []
    disagree: list[dict] = []
    by_scene: dict[str, dict] = {}
    for case in cases:
        s, r = s_by_id[case.eval_case_id], r_by_id[case.eval_case_id]
        truth = (exp.get(case.eval_case_id) or {}).get("decision", "?")
        row = {
            "eval_case_id": case.eval_case_id,
            "scene": case.scene,
            "truth": truth,
            "scripted_decision": s.decision,
            "real_decision": r.decision,
            "agree": s.decision == r.decision,
            "real_risk_level": r.risk_level,
            "real_risk_type": list(r.risk_type),
            "real_decision_confidence": r.decision_confidence,
            "real_overrides": list((r.detail or {}).get("overrides") or []),
        }
        rows.append(row)
        counter = by_scene.setdefault(case.scene, {"total": 0, "agree": 0})
        counter["total"] += 1
        if row["agree"]:
            counter["agree"] += 1
        else:
            disagree.append(row)
    return rows, disagree, by_scene


async def run_comparison(
    *,
    cases: list[EvalCase],
    ctx: EvalContext,
    real_backend: Any,
    model_label: str,
    world: str = "eval",
    data_path: str = "",
    budget_limits: dict | None = None,
) -> tuple[dict, dict]:
    """核心对比：scripted（确定性桩，先行、可复现）→ real（注入后端，逐案串行）。

    返回 ``(payload, extra)``：``payload`` 可直接落 JSON（real 侧含 REAL_NOTE）；
    ``extra`` 是报告渲染用中间物（rows / by_scene / 两臂 DecisionMetrics /
    scripted_records / truth_human 计数等，不进 JSON）。

    ``budget_limits`` 只作用于 real 臂的评测侧预算覆盖；scripted 臂恒为默认预算
    （毫秒级跑完，不触发墙钟护栏）。
    """
    exp = expected_index(cases)
    scripted = AgentScheme()
    real = AgentScheme(llm=real_backend, budget_limits=budget_limits)

    print("-" * 100)
    print("① scripted（确定性审查员桩 EvalScriptedLLMBackend · 可复现基线）")
    scripted_records = await _run_scheme_records(scripted, cases, ctx)
    print("-" * 100)
    print(f"② real（{model_label} · 真实 LLM · 非确定性 · 逐案串行）")
    real_records = await _run_scheme_records(real, cases, ctx, progress_prefix="real")

    rows, disagree, by_scene = _compare_rows(cases, scripted_records, real_records, exp)
    scripted_metrics = DecisionEvaluator.evaluate(scripted_records, exp)
    real_metrics = DecisionEvaluator.evaluate(real_records, exp)
    truth_human = sum(1 for e in exp.values() if e.get("decision") == "HUMAN_REVIEW")

    payload = {
        "data": data_path,
        "model": model_label,
        "world": world,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "count": len(cases),
        "agree": len(rows) - len(disagree),
        "disagree": disagree,
        "real_records": [r.model_dump() for r in real_records],
        "scripted_decisions": [
            {
                "eval_case_id": r.eval_case_id,
                "decision": r.decision,
                "risk_level": r.risk_level,
                "risk_type": list(r.risk_type),
                "decision_confidence": r.decision_confidence,
            }
            for r in scripted_records
        ],
        # overrides 汇总：R5 降级 / R3 截胡 / 混合案 —— 「整卷全 HUMAN = 链路降级」
        # 在 JSON 里也一眼可见，不只在 Console 报告。
        "overrides_summary": {
            "real": _overrides_summary(real_records),
            "scripted": _overrides_summary(scripted_records),
        },
        "note": REAL_NOTE,
    }
    extra = {
        "rows": rows,
        "by_scene": by_scene,
        "scripted_metrics": scripted_metrics,
        "real_metrics": real_metrics,
        "scripted_records": scripted_records,
        "truth_human": truth_human,
        "scene_stats": scene_stats(cases),
        "scripted_cost": _cost_summary(scripted_records),
        "real_cost": _cost_summary(real_records),
        "real_overrides": _overrides_summary(real_records),  # 渲染用
        "scripted_overrides": _overrides_summary(scripted_records),
    }
    return payload, extra


# ---------------------------------------------------------------------------
# 报告渲染（纯文本中文，report.py 风格：口径注记 + 一致性 + 逐案差异 + 总体指标）
# ---------------------------------------------------------------------------


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.3f}"


def _cost_summary(records: list[EvalRecord]) -> dict:
    """成本均值摘要（确定性；无 case 时全 0）—— 与 runner._cost_summary 同口径。"""
    n = len(records)
    if n == 0:
        return {"llm_calls": 0.0, "tool_calls": 0.0, "tokens": 0.0}
    return {
        "llm_calls": round(sum(r.cost.get("llm_calls") or 0 for r in records) / n, 2),
        "tool_calls": round(sum(r.cost.get("tool_calls") or 0 for r in records) / n, 2),
        "tokens": round(sum(r.cost.get("tokens") or 0 for r in records) / n, 2),
    }


def _overrides_summary(records: list[EvalRecord]) -> dict:
    """overrides 汇总：每案归因码计数 + R3/R5 混合案数，链路降级一眼可见。

    只读 ``EvalRecord.detail["overrides"]``（gate overlay 写入的归因码：
    R5_DEGRADED_OR_FAILED_STEP = LLM 步降级兜底转人工、R3_BUDGET_EXHAUSTED = 预算截胡、
    R2_*/R4_*/R1_* = Gate 改判）；无 case → 全 0。确定性函数。
    """
    counts: Counter = Counter()
    cases_with_any = 0
    r3_r5_mixed = 0
    for r in records:
        ovs = list((r.detail or {}).get("overrides") or [])
        if not ovs:
            continue
        cases_with_any += 1
        counts.update(ovs)
        if R3_BUDGET_EXHAUSTED in ovs and R5_DEGRADED_OR_FAILED_STEP in ovs:
            r3_r5_mixed += 1
    return {
        "cases_with_any": cases_with_any,
        R3_BUDGET_EXHAUSTED: counts.get(R3_BUDGET_EXHAUSTED, 0),
        R5_DEGRADED_OR_FAILED_STEP: counts.get(R5_DEGRADED_OR_FAILED_STEP, 0),
        "r3_r5_mixed": r3_r5_mixed,
        "other_codes": {
            k: v
            for k, v in sorted(counts.items())
            if k not in (R3_BUDGET_EXHAUSTED, R5_DEGRADED_OR_FAILED_STEP)
        },
    }


def _metrics_line(label: str, m: DecisionMetrics, cost: dict) -> str:
    return "  ".join(
        [
            f"{label:<14}",
            _fmt(m.accuracy),
            _fmt(m.precision),
            _fmt(m.recall),
            _fmt(m.fpr),
            _fmt(m.fnr),
            _fmt(m.human_rate),
            _fmt(m.automation),
            f"{m.tp}/{m.fp}/{m.tn}/{m.fn}",
            f"llm={cost.get('llm_calls')} tool={cost.get('tool_calls')} tok={cost.get('tokens')}",
        ]
    )


def _risk_cell(row: dict) -> str:
    types = ",".join(row["real_risk_type"]) or "-"
    conf = "-" if row["real_decision_confidence"] is None else f"{row['real_decision_confidence']:.2f}"
    return f"{row['real_risk_level'] or '-'}/{types}/conf={conf}"


def _overrides_cell(row: dict) -> str:
    """逐案 real overrides 缩写（R3=R3_BUDGET_EXHAUSTED / R5=R5_DEGRADED_OR_FAILED_STEP）。"""
    ovs = row.get("real_overrides") or []
    if not ovs:
        return "-"
    short = {
        R3_BUDGET_EXHAUSTED: "R3",
        R5_DEGRADED_OR_FAILED_STEP: "R5",
    }
    return ",".join(short.get(c, c) for c in ovs)


def _overrides_line(ov: dict, total: int) -> str:
    """一行 overrides 汇总：带码案数 / R5 降级 / R3 截胡 / 混合 / 其它码。"""
    parts = [
        f"带 overrides {ov['cases_with_any']}/{total} 案",
        f"R5 降级 {ov[R5_DEGRADED_OR_FAILED_STEP]} 案",
        f"R3 预算截胡 {ov[R3_BUDGET_EXHAUSTED]} 案",
        f"R3+R5 混合 {ov['r3_r5_mixed']} 案",
    ]
    if ov.get("other_codes"):
        parts.append("其它码 " + ",".join(f"{k}={v}" for k, v in ov["other_codes"].items()))
    return " ｜ ".join(parts)


def render_report(payload: dict, extra: dict, *, out_path: str | None = None) -> str:
    """渲染整份 Console Report（纯文本；real 侧非确定性如实标注）。"""
    out: list[str] = []
    add = out.append

    add("=" * 100)
    add("商品审核 Agent · Evaluation Phase 3 real（真实 LLM）vs scripted（确定性桩）对比")
    add("=" * 100)
    stats = extra["scene_stats"]
    by_scene = stats.get("by_scene", {})
    scene_n = {s: int(by_scene.get(s, {}).get("total", 0)) for s in SCENES}
    dist = " | ".join(f"{s}={scene_n[s]}" for s in SCENES)
    add(f"数据集: {payload['data']}（{payload['count']} 条）| world={payload['world']} | real 模型: {payload['model']}")
    add(f"scene 分布: {dist}")

    add("-" * 100)
    add("结论边界 / 口径注记:")
    add("  · scripted = EvalScriptedLLMBackend（确定性审查员桩：同 case 同 ctx → 同输出，可重放）")
    add(f"  · real = {payload['model']}（真实 LLM —— 非确定性、不可重放、需 API key 与费用；")
    add("    本报告 real 数字 = 单次运行抽样，不代表模型固定水平；回归基线恒以 scripted 为准）")
    add(f"  · 工具数据源: {_world_label(payload['world'])}（两臂同一世界 → LLM 是唯一变量）")
    add("  · 一致性口径: scripted.decision == real.decision 判为一致（risk/evidence 差异不参与）")
    add("  · 指标口径（DecisionEvaluator）: Accuracy=(TP+TN)/真值总数，预测 HUMAN_REVIEW 计为未命中")
    add("    真值(判错，入分母不入分子)；Precision/Recall/FPR/FNR 只在自动判出(pred∈{PASS,REJECT})子集计算")
    add("  · REJECT 为正类: Recall=TP/(TP+FN) 违规召回 / FPR=FP/(FP+TN) 误杀红线 / FNR=FN/(TP+FN) 漏放")
    add("  · HRR=转人工率 / auto=自动化率；EvalRecord 不含墙钟 latency（real 墙钟仅进程内进度打印）")
    add(
        "  · cost.tokens 口径（P2-16）= usage.total_tokens：input+output 合计、含 provider"
        " 缓存命中 token；schema 校验失败的尝试也全额累计 —— R3 按 tokens 维度归因时"
        "按此口径解读（EvalRecord.detail.budget_hit_dim 记录哪一维先撞限）"
    )
    if extra["truth_human"]:
        add(
            f"  · 真值含 HUMAN_REVIEW 的案 {extra['truth_human']} 条（v2 SHOULD_ABSTAIN）："
            "DecisionEvaluator 二值口径自动跳过，不计入上表指标 —— 如实呈现，不硬算"
        )

    add("-" * 100)
    total, agree = payload["count"], payload["agree"]
    diff_n = total - agree
    share = f"{agree / total:.1%}" if total else "-"
    add(f"real vs scripted 决策一致性: 一致 {agree}/{total}（{share}）· 差异 {diff_n} 条")
    scene_parts = []
    for s in SCENES:
        c = extra["by_scene"].get(s)
        if c and c["total"]:
            scene_parts.append(f"{s}={c['agree']}/{c['total']}")
    add("按 scene 一致数: " + (" | ".join(scene_parts) if scene_parts else "-"))

    add("-" * 100)
    if diff_n == 0:
        add("差异 case 列表: （无 —— 两臂逐案裁决完全一致）")
    else:
        add(f"差异 case 列表（共 {diff_n} 条 · 各行含 real risk 摘要 + overrides）:")
        for row in payload["disagree"]:
            ovr_s = _overrides_cell(row)
            ovr_note = f" | real ovr: {ovr_s}" if ovr_s != "-" else ""
            add(
                f"  · {row['eval_case_id']} [{row['scene']}] truth={row['truth']} | "
                f"scripted {row['scripted_decision']} → real {row['real_decision']} "
                f"（{_risk_cell(row)}）{ovr_note}"
            )

    add("-" * 100)
    add("逐案对比明细  case         scene         truth      scripted   real         一致  real risk/type/conf      real ovr")
    for row in extra["rows"]:
        mark = "是" if row["agree"] else "否"
        add(
            f"  {row['eval_case_id']:<11} [{row['scene']:<11}] truth={row['truth']:<6} "
            f"{row['scripted_decision']:<9} {row['real_decision']:<12} {mark:<4} "
            f"{_risk_cell(row):<34} {_overrides_cell(row)}"
        )

    add("-" * 100)
    add("总体指标   acc   prec  recall  fpr   fnr   hrr   auto    TP/FP/TN/FN   成本均值(llm/tool/tok)")
    add(_metrics_line("scripted", extra["scripted_metrics"], extra["scripted_cost"]))
    add(_metrics_line("real", extra["real_metrics"], extra["real_cost"]))
    if extra["truth_human"]:
        add(
            f"（上表两行均只覆盖二值真值案 {extra['scripted_metrics'].total} 条；"
            f"HUMAN_REVIEW 真值 {extra['truth_human']} 条被跳过）"
        )

    add("-" * 100)
    add("overrides 汇总（审计：R5=LLM 步降级兜底转 HUMAN、R3=预算截胡 —— P1-6c）:")
    total_cases = payload["count"]
    add(f"  · real:     {_overrides_line(extra['real_overrides'], total_cases)}")
    add(f"  · scripted: {_overrides_line(extra['scripted_overrides'], total_cases)}")
    real_ov = extra["real_overrides"]
    if real_ov[R5_DEGRADED_OR_FAILED_STEP]:
        add(
            f"    ⚠ real 有 {real_ov[R5_DEGRADED_OR_FAILED_STEP]} 案触发 R5 降级 —— 这些案"
            "的 real 裁决来自降级兜底（HUMAN），**不是模型行为**；解读全卷差异/指标须扣除"
        )
    if total_cases and real_ov[R5_DEGRADED_OR_FAILED_STEP] == total_cases:
        add(
            "    ⚠⚠ real 全部案均 R5 降级：本卷 real 结果 = 100% 链路降级（典型原因：无"
            "key/网关/base-url 配置问题或逐节点连续失败）—— 请勿把本卷 HUMAN 当模型结论"
        )

    add("-" * 100)
    out_note = f" | JSON 已写入: {out_path}" if out_path else " | 未写文件（--out 可落盘）"
    add(f"[OK] 对比完成: {total} 条 · 一致 {agree} · 差异 {diff_n}{out_note}")
    add(f"[NOTE] {payload['note']} —— real 侧输出不可用于逐字节回归比对")
    add("=" * 100)
    return "\n".join(out)


def print_report(payload: dict, extra: dict, *, out_path: str | None = None) -> None:
    """打印 Console Report 到 stdout。"""
    print(render_report(payload, extra, out_path=out_path))


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluation Phase 3 real：agent real（真实 LLM）vs scripted（确定性桩）"
            "对比跑分（同一数据/同一工具世界；real 非确定性且需 API key，建议先 --limit）"
        )
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help=f"评测集：JSONL 路径或目录（默认 {DEFAULT_DATA} → cases_v1.jsonl；v2 目录 → cases_v2.jsonl）",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help=f"real LLM 模型（默认 {DEFAULT_MODEL}）"
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
        help="只跑数据集前 N 条（loader smoke_subset 确定性取法；默认 None = 全量）",
    )
    parser.add_argument(
        "--ids",
        default=None,
        help=(
            "只跑指定 eval_case_id（逗号分隔，如 \"EC_0007,EC_0101\"；定向 real smoke 用；"
            "默认 None = 不按 id 过滤）。加载后按 eval_case_id 过滤，可与 --limit 叠加"
            "（先按 --ids 过滤、再按 --limit 截断）；id 不在数据集中会报错提示"
        ),
    )
    parser.add_argument(
        "--world",
        default="eval",
        choices=list(WORLDS),
        help="Agent 工具数据源世界（默认 eval；rag = RAG 世界真实 KB 检索，mode=hybrid）",
    )
    parser.add_argument(
        "--max-latency-ms",
        type=int,
        default=600000,
        help=(
            "real 评测的预算墙钟护栏上限（毫秒；默认 600000=10min）—— 生产护栏 30s "
            "对真实 LLM 太紧（每案 ~9 次串行调用天然 >30s），不放宽则每案都被 "
            "LATENCY 超限截胡转人工、测不到决策质量；scripted 毫秒级不受影响。llm/"
            "tool/token 护栏默认 10/15/40000（--llm-budget 可覆盖 llm 档）。报告注明"
            "本口径差异"
        ),
    )
    parser.add_argument(
        "--llm-budget",
        type=int,
        default=None,
        help=(
            "real 臂的 LLM 调用预算上限（max_llm_calls 覆盖；默认 None = 生产默认 10"
            " 不变）—— 预算档位对照实验：10/12/15 档跑同一批数据，回答真实案件打满 10 被"
            "截胡转人工是预算太紧还是 Agent 收敛差（档位抬高仍打满 ⇒ 收敛问题；涨到"
            "收敛即止 ⇒ 预算紧）。只作用于 real 臂，scripted 对照臂恒默认，生产护栏"
            "不受影响"
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="结果 JSON 写出路径（目录需已存在；默认不写文件只打印）",
    )
    return parser.parse_args(argv)


def _build_budget_limits(*, max_latency_ms: int, llm_budget: int | None) -> dict:
    """real 臂评测侧预算覆盖装配（键 = ``BudgetLimits`` 字段名）。

    ``llm_budget=None`` → 只放宽 ``max_latency_ms``（与改动前逐字节一致）；
    ``--llm-budget N`` → 追加 ``max_llm_calls=N``（本覆盖只作用于 real 臂，生产护栏
    仍固定 10）。
    """
    limits = {"max_latency_ms": max_latency_ms}
    if llm_budget is not None:
        limits["max_llm_calls"] = llm_budget
    return limits


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _load_dotenv()  # 仓库根 .env 的 DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL 注入（setdefault）
    data_path = _resolve_data_path(args.data)
    cases = load_dataset(data_path)
    # --ids 定向过滤（按 eval_case_id；id 拼错宁可快速报错，避免带着空/错集烧 API 费用）
    if args.ids:
        wanted = {s.strip() for s in args.ids.split(",") if s.strip()}
        if not wanted:
            raise ValueError('--ids 为空：请用逗号分隔的 eval_case_id，如 --ids "EC_0007,EC_0101"')
        missing = wanted - {c.eval_case_id for c in cases}
        if missing:
            raise ValueError(f"--ids 有 {len(missing)} 个不在当前数据集中: {sorted(missing)}")
        cases = [c for c in cases if c.eval_case_id in wanted]
    if args.limit is not None and args.limit > 0:
        cases = smoke_subset(cases, args.limit)  # 确定性取前 N 条
    if not cases:
        raise ValueError("评测运行无有效 case（数据集为空或 --limit/--ids 截成空）")

    api_key = _resolve_api_key(args.api_key)
    base_url = args.base_url or os.environ.get(ENV_BASE_URL, "") or None
    # real 后端（LiteLLMBackend）强制要求 api_key：仅给 --base-url 的本地网关模式不支持，
    # 缺 key 预检即报错（否则会整卷 R5 降级 HUMAN、exit 0，静默产出假 real 结果）。
    # base_url 仍可指向自定义网关端点，但必须配真实 key。
    if api_key is None:
        raise ValueError(
            f"未检测到 API key（--api-key 或环境变量 {ENV_API_KEY} / 仓库根 .env）。"
            "real 模式必须配置 API key：无 key 本地网关（仅 --base-url）当前不支持"
            "（LiteLLMBackend 强制 api_key）——请配置真实 key 后重跑"
        )

    ctx = _build_ctx(args.world)
    real_backend = _make_real_backend(
        model=args.model, api_key=api_key, base_url=base_url, world=args.world
    )
    # real 评测只测 LLM 决策质量：放宽墙钟护栏（默认 10min）避免 LATENCY 截胡，
    # --llm-budget N 再覆盖 LLM 调用预算档；scripted 臂保持默认预算。
    budget_limits = _build_budget_limits(
        max_latency_ms=args.max_latency_ms, llm_budget=args.llm_budget
    )

    # 头部：跑分前先亮明成本与可重放边界
    stats = scene_stats(cases)
    by_scene = stats.get("by_scene", {})
    scene_n = {s: int(by_scene.get(s, {}).get("total", 0)) for s in SCENES}
    print("=" * 100)
    print("商品审核 Agent · Evaluation Phase 3 real 跑分（真实 LLM vs scripted 对照）")
    print("=" * 100)
    print(f"数据集: {data_path}（{len(cases)} 条）| world={args.world} | real 模型: {args.model}")
    print("scene 分布: " + " | ".join(f"{s}={scene_n[s]}" for s in SCENES))
    key_state = "已配置（--api-key / 环境变量 / .env）"
    print(f"API key: {key_state}（值不入日志/报告/JSON）")
    print(
        f"[NOTE] real 侧真实调用 LLM（有费用、非确定性、不可重放）；"
        f"建议先 --limit 10 冒烟 —— 本次跑 {len(cases)} 条"
    )
    print(
        f"[NOTE] real 臂预算墙钟护栏放宽至 {args.max_latency_ms}ms（默认 600000；"
        "生产护栏 30s 对真实 LLM 过紧会截胡转人工）；llm/tool/token 护栏保持默认"
    )
    if args.llm_budget is not None:
        print(
            f"[NOTE] B-2 对照：real 臂 LLM 调用预算上限覆盖为 {args.llm_budget}"
            "（默认 None = 生产默认 10）；scripted 臂与生产护栏不受影响"
        )

    payload, extra = await run_comparison(
        cases=cases,
        ctx=ctx,
        real_backend=real_backend,
        model_label=args.model,
        world=args.world,
        data_path=str(data_path),
        budget_limits=budget_limits,
    )

    out_path: str | None = None
    if args.out:
        out_path = str(args.out)
        # 目录需已存在；父目录缺失时明确报错
        out_file = Path(out_path)
        if not out_file.parent.exists():
            raise ValueError(f"--out 父目录不存在: {out_file.parent}（请先创建目录）")
        with out_file.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    print()
    print_report(payload, extra, out_path=out_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_main(argv))
    except Exception as exc:  # 任何失败 → 非零退出（CI 可捕获）
        print(f"[FAIL] real 跑分失败: {exc!r}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
