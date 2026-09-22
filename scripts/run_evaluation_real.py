"""正式 Evaluation 跑分：rule（确定性初筛）vs real（真实 LLM），工具世界固定为 Eval World。

两臂：① ``RuleBaseline``（复用 ``pra.screening`` 的确定性三分流，零 LLM、零工具）；②
``AgentScheme(llm=...)`` 接真实 LLM。同一数据集、同一 Eval World 工具、同为生产默认 Guardrail
预算档 → 两臂差异只归因于 LLM。输出：逐案对比明细、业务指标（``DecisionEvaluator`` 两套分母）、
工程分布、Agent 级指标、overrides 归因汇总（R5 降级 / R3 预算截胡 / R3+R5 混合 —— 「整卷全
HUMAN 是链路降级」一眼可见），以及 ``--out`` 落盘的 real EvalRecord 全量 + 逐案 rule 决策 +
Agent 指标 + overrides 汇总（``run_error_analysis.py`` 的输入）。

CLI：``--data`` JSONL 或目录（eval_data/v2 → cases_v2.jsonl）；``--model`` / ``--api-key`` /
``--base-url`` 配 LLM；``--limit N`` / ``--ids "EC_V2_0007,EC_V2_0101"`` 定向取案子集；
``--concurrency N`` 并发跑 real 臂；``--out PATH`` 写结果 JSON（父目录需已存在）。

成本与结论边界：真实 API 有费用、非确定性（同案重跑输出可能不同，不可重放）——先
``--limit 10`` 冒烟确认链路与成本量级再跑全量；report / JSON 已如实标注，real 数字只代表单次
运行抽样。API key 读取顺序：``--api-key`` > 环境变量 ``DEEPSEEK_API_KEY`` > 仓库根 ``.env``
（脚本开头以 setdefault 语义注入）；base-url 同理。API key 必填 —— 缺 key 预检即报错，避免整卷
R5 降级后静默产出假 real 结果。EvalRecord 不含墙钟 latency（进程相关量会让跨 run 比对漂移），real
墙钟只进进程内进度打印与报告工程区。预算恒为生产默认档（评测不覆盖 Guardrail）：real 臂超限转
HUMAN_REVIEW 是生产语义，归因走 ``detail.budget_hit_dim``。``pra.agent.litellm_backend`` 为
延迟 import：缺失时本模块仍可 import，只有 real 运行需要它就绪。
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
from datetime import UTC, datetime
from pathlib import Path

from pra.agent.guardrails.gate import (  # overrides 归因码（汇总用，只读）
    R3_BUDGET_EXHAUSTED,
    R5_DEGRADED_OR_FAILED_STEP,
)
from pra.evaluation.dataset.loader import load_dataset, scene_stats, smoke_subset
from pra.evaluation.dataset.schema import EvalCase
from pra.evaluation.harness.agent_scheme import (
    EVAL_WORLD_LABEL,
    AgentScheme,
    make_eval_world_tools,
)
from pra.evaluation.harness.base import EvalRecord, SchemeRunner
from pra.evaluation.harness.rule_scheme import RuleBaseline
from pra.evaluation.metrics.agent import AgentMetricsBundle
from pra.evaluation.metrics.business import DecisionEvaluator
from pra.evaluation.metrics.engineering import EngineeringEvaluator
from pra.evaluation.report import print_report
from pra.evaluation.runner import expected_index

DEFAULT_DATA = "eval_data/v2"
DEFAULT_MODEL = "deepseek/deepseek-flash"
ENV_API_KEY = "DEEPSEEK_API_KEY"
ENV_BASE_URL = "DEEPSEEK_BASE_URL"
SCENES = ("normal", "violation", "boundary", "multi-signal", "evasion")
REAL_NOTE = "real LLM 非确定性，不可重放"

__all__ = ["DEFAULT_DATA", "DEFAULT_MODEL", "REAL_NOTE", "_resolve_data_path", "run_comparison"]


# ---------------------------------------------------------------------------
# 数据集 / 后端装配
# ---------------------------------------------------------------------------


def _resolve_data_path(raw: str) -> Path:
    """把 ``--data`` 解析为评测 JSONL 路径（直接 JSONL，或按目录名拼
    ``cases_<目录名>.jsonl``：eval_data/v2 → cases_v2.jsonl）。"""
    p = Path(raw)
    if p.is_file():
        return p
    if p.is_dir():
        if re.fullmatch(r"v\d+", p.name) is not None:
            cand = p / f"cases_{p.name}.jsonl"
            if cand.is_file():
                return cand
        raise ValueError(
            f"评测数据目录 {p} 无法定位 cases JSONL：目录名按 v<数字> → "
            f"cases_<目录名>.jsonl（{p.name} 不是 v<数字> 目录名或该文件缺失）"
        )
    raise ValueError(f"评测数据路径不存在: {p}（支持 JSONL 文件，或 eval_data/v2 目录）")


def _make_real_backend(*, model: str, api_key: str | None, base_url: str | None) -> object:
    """构造真实 LLM 后端（延迟 import ``pra.agent.litellm_backend``）。

    按 ``LLMBackend`` Protocol 交给 ``AgentScheme(llm=...)``；``tools`` = Eval World 工具列表
    （供后端提取 function schema / 工具提示），与图侧装配的世界同一份。
    """
    try:
        from pra.agent.litellm_backend import LiteLLMBackend
    except ImportError as exc:  # 模块或 litellm 依赖缺失 → 明确报错而非半路 ImportError
        raise RuntimeError(
            "real 模式需要 pra.agent.litellm_backend（含 litellm 依赖，见 pyproject）："
            "`uv sync` 后重试"
        ) from exc
    return LiteLLMBackend(
        model=model, api_key=api_key, base_url=base_url, tools=make_eval_world_tools()
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
# 跑分流程（rule 臂确定性先行 + real 臂注入后端）
# ---------------------------------------------------------------------------


async def _run_scheme_records(
    scheme: SchemeRunner,
    cases: list[EvalCase],
    *,
    progress_prefix: str | None = None,
    latencies_ms: list[float] | None = None,
) -> list[EvalRecord]:
    """顺序串行跑一遍 scheme；progress_prefix 非 None 时逐条打印进度。

    ``latencies_ms`` 非 None 时逐案追加墙钟耗时 —— **只进渲染层**（real 臂的 P50/P95），
    不写进 EvalRecord（跨 run 比对会因进程相关量漂移，见 harness/base.py 口径）。
    """
    records: list[EvalRecord] = []
    total = len(cases)
    for i, case in enumerate(cases, start=1):
        t0 = time.monotonic()
        rec = await scheme.run(case)
        elapsed_ms = (time.monotonic() - t0) * 1000
        if latencies_ms is not None:
            latencies_ms.append(elapsed_ms)
        if progress_prefix is not None:
            print(
                f"  [{i}/{total}] {case.eval_case_id:<9} → {rec.decision:<12} "
                f"({elapsed_ms / 1000:.1f}s)"
            )
        records.append(rec)
    return records


async def _run_scheme_records_concurrent(
    scheme: SchemeRunner,
    cases: list[EvalCase],
    *,
    concurrency: int,
    progress_prefix: str | None = None,
    latencies_ms: list[float] | None = None,
) -> list[EvalRecord]:
    """并发跑同一 scheme（仅 real 臂用；结果按用例原序返回）。

    为什么安全：用例之间天然隔离 —— 每案在 ``AgentScheme.run`` 内独立
    ``build_agent_graph`` + ``compile``（checkpointer=InMemory、thread_id 唯一），
    工具每案新建，LLM 后端每案经 ``llm=`` 显式注入（图持有自己的后端，不共享可变状态）。
    """
    sem = asyncio.Semaphore(concurrency)
    total = len(cases)
    done = 0

    async def _one(case: EvalCase) -> EvalRecord:
        nonlocal done
        async with sem:
            t0 = time.monotonic()
            rec = await scheme.run(case)
            elapsed_ms = (time.monotonic() - t0) * 1000
            if latencies_ms is not None:
                latencies_ms.append(elapsed_ms)  # list.append 原子；并发下顺序不定，只影响分位渲染
            done += 1
            if progress_prefix is not None:
                print(
                    f"  [{done}/{total}] {case.eval_case_id:<9} → {rec.decision:<12} "
                    f"({elapsed_ms / 1000:.1f}s)",
                    flush=True,
                )
            return rec

    return list(await asyncio.gather(*[_one(c) for c in cases]))


def _compare_rows(
    cases: list[EvalCase],
    rule_records: list[EvalRecord],
    real_records: list[EvalRecord],
    exp: dict,
) -> tuple[list[dict], list[dict], dict]:
    """逐案对齐（按 eval_case_id）→ 全量行 + 差异行 + 按 scene 一致性计数。"""
    rule_by_id = {r.eval_case_id: r for r in rule_records}
    real_by_id = {r.eval_case_id: r for r in real_records}
    rows: list[dict] = []
    disagree: list[dict] = []
    by_scene: dict[str, dict] = {}
    for case in cases:
        rule, real = rule_by_id[case.eval_case_id], real_by_id[case.eval_case_id]
        truth = (exp.get(case.eval_case_id) or {}).get("decision", "?")
        row = {
            "eval_case_id": case.eval_case_id,
            "scene": case.scene,
            "truth": truth,
            "rule_decision": rule.decision,
            "real_decision": real.decision,
            "agree": rule.decision == real.decision,
            "real_risk_level": real.risk_level,
            "real_risk_type": list(real.risk_type),
            "real_decision_confidence": real.decision_confidence,
            "real_overrides": list((real.detail or {}).get("overrides") or []),
        }
        rows.append(row)
        counter = by_scene.setdefault(case.scene, {"total": 0, "agree": 0})
        counter["total"] += 1
        if row["agree"]:
            counter["agree"] += 1
        else:
            disagree.append(row)
    return rows, disagree, by_scene


def _overrides_summary(records: list[EvalRecord]) -> dict:
    """overrides 汇总：每案归因码计数 + R3/R5 混合案数，链路降级一眼可见。

    只读 ``EvalRecord.detail["overrides"]``（gate overlay 写入的归因码：
    R5_DEGRADED_OR_FAILED_STEP = LLM 步降级兜底转人工、R3_BUDGET_EXHAUSTED = 预算截胡、
    R2_*/R4_*/R1_* = Gate 改判）；无 case → 全 0。确定性函数。
    """
    counts: Counter = Counter()
    hit_dims: Counter = Counter()
    cases_with_any = 0
    r3_r5_mixed = 0
    for r in records:
        detail = r.detail or {}
        ovs = list(detail.get("overrides") or [])
        if not ovs:
            continue
        cases_with_any += 1
        counts.update(ovs)
        if R3_BUDGET_EXHAUSTED in ovs:
            # 截胡维度只可能是 llm_calls / tool_calls，拆分便于归因（见 budget.py 口径）
            hit_dims[str(detail.get("budget_hit_dim") or "UNKNOWN")] += 1
        if R3_BUDGET_EXHAUSTED in ovs and R5_DEGRADED_OR_FAILED_STEP in ovs:
            r3_r5_mixed += 1
    return {
        "cases_with_any": cases_with_any,
        R3_BUDGET_EXHAUSTED: counts.get(R3_BUDGET_EXHAUSTED, 0),
        R5_DEGRADED_OR_FAILED_STEP: counts.get(R5_DEGRADED_OR_FAILED_STEP, 0),
        "r3_r5_mixed": r3_r5_mixed,
        "budget_hit_dims": dict(sorted(hit_dims.items())),
        "other_codes": {
            k: v
            for k, v in sorted(counts.items())
            if k not in (R3_BUDGET_EXHAUSTED, R5_DEGRADED_OR_FAILED_STEP)
        },
    }


async def run_comparison(
    *,
    cases: list[EvalCase],
    real_backend: object,
    model_label: str,
    data_path: str = "",
    real_concurrency: int = 1,
) -> tuple[dict, dict]:
    """核心跑分：rule（确定性零成本，先行）→ real（注入后端）。

    ``real_concurrency > 1`` 时 real 臂并发跑（rule 臂恒串行 —— 无 LLM 调用且是确定性
    基线，不需要并发）；并发只改**调度**，不改判定 / 指标 / 数据。

    返回 ``(payload, extra)``：``payload`` 可直接落 JSON（含 REAL_NOTE）；``extra`` 是报告
    渲染中间物（逐案行 / by_scene / 两臂 DecisionMetrics / EngineeringMetrics /
    AgentMetricsBundle / overrides 汇总 / scene 统计），不进 JSON。

    两臂预算恒为生产默认档（评测不覆盖 Guardrail）；real 臂超限按生产语义转 HUMAN_REVIEW。
    """
    exp = expected_index(cases)
    rule = RuleBaseline()
    real = AgentScheme(llm=real_backend)

    print("-" * 100)
    print("① rule（RuleBaseline · pra.screening 确定性初筛 · 零 LLM 零工具）")
    rule_records = await _run_scheme_records(rule, cases)
    print(f"  → {len(rule_records)} 案完成")
    print("-" * 100)
    real_mode = f"并发 {real_concurrency}" if real_concurrency > 1 else "逐案串行"
    print(f"② real（{model_label} · 真实 LLM · 非确定性 · {real_mode}）")
    real_latencies_ms: list[float] = []
    if real_concurrency > 1:
        real_records = await _run_scheme_records_concurrent(
            real,
            cases,
            concurrency=real_concurrency,
            progress_prefix="real",
            latencies_ms=real_latencies_ms,
        )
    else:
        real_records = await _run_scheme_records(
            real, cases, progress_prefix="real", latencies_ms=real_latencies_ms
        )

    rows, disagree, by_scene = _compare_rows(cases, rule_records, real_records, exp)
    rule_overrides = _overrides_summary(rule_records)
    real_overrides = _overrides_summary(real_records)

    payload = {
        "data": data_path,
        "model": model_label,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "count": len(cases),
        "real_concurrency": real_concurrency,
        "agree": len(rows) - len(disagree),
        "disagree": disagree,
        # 逐案 rule 决策（Error Analysis 的决策迁移基线；只留 id + 三分类裁决）
        "rule_decisions": [
            {"eval_case_id": r.eval_case_id, "decision": r.decision} for r in rule_records
        ],
        "real_records": [r.model_dump() for r in real_records],
        # Agent 级指标（两臂）：JSON 侧与 Console 报告同数，供事后审计/复算
        "agent_metrics": {
            "rule": AgentMetricsBundle.evaluate(rule_records, exp).model_dump(),
            "real": AgentMetricsBundle.evaluate(real_records, exp).model_dump(),
        },
        # overrides 汇总：R5 降级 / R3 截胡 / 混合案 —— 「整卷全 HUMAN = 链路降级」
        # 在 JSON 里也一眼可见，不只在 Console 报告。
        "overrides_summary": {"rule": rule_overrides, "real": real_overrides},
        "note": REAL_NOTE,
    }
    extra = {
        "rows": rows,
        "by_scene": by_scene,
        "rule_metrics": DecisionEvaluator.evaluate(rule_records, exp),
        "real_metrics": DecisionEvaluator.evaluate(real_records, exp),
        "rule_engineering": EngineeringEvaluator.evaluate(rule_records),
        # 延迟只在 real 臂有（进程内墙钟，不落 record —— 见 _run_scheme_records docstring）
        "real_engineering": EngineeringEvaluator.evaluate(
            real_records, latency_ms=real_latencies_ms
        ),
        "rule_agent_metrics": AgentMetricsBundle.evaluate(rule_records, exp),
        "real_agent_metrics": AgentMetricsBundle.evaluate(real_records, exp),
        "rule_overrides": rule_overrides,
        "real_overrides": real_overrides,
        "scene_stats": scene_stats(cases),
    }
    return payload, extra


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluation 正式跑分：rule（确定性初筛）vs real（真实 LLM），同一数据同一 Eval World"
            "（real 非确定性且需 API key，建议先 --limit 冒烟）"
        )
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help=f"评测集：JSONL 路径或目录（默认 {DEFAULT_DATA} → cases_v2.jsonl）",
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
            '只跑指定 eval_case_id（逗号分隔，如 "EC_0007,EC_0101"；定向 real smoke 用；'
            "默认 None = 不按 id 过滤）。加载后按 eval_case_id 过滤，可与 --limit 叠加"
            "（先按 --ids 过滤、再按 --limit 截断）；id 不在数据集中会报错提示"
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
            "real 臂并发度（默认 1 = 逐案串行）。>1 时并发跑 real 臂：用例之间天然隔离"
            "（每案独立 build+compile 图、thread_id 唯一、工具与 LLM 后端每案新建并显式注入），"
            "只改调度、不改判定/指标/数据；rule 臂恒串行"
        ),
    )
    return parser.parse_args(argv)


def _write_payload(path: Path, payload: dict) -> None:
    """把结果 JSON 写入 ``path``（父目录须已存在，缺失即报错）。"""
    if not path.parent.exists():
        raise ValueError(f"--out 父目录不存在: {path.parent}（请先创建目录）")
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


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
    # real 后端（LiteLLMBackend）强制要求 api_key：缺 key 预检即报错（否则会整卷 R5 降级
    # HUMAN、exit 0，静默产出假 real 结果）。base_url 仍可指向自定义网关端点，但必须配真实 key。
    if api_key is None:
        raise ValueError(
            f"未检测到 API key（--api-key 或环境变量 {ENV_API_KEY} / 仓库根 .env）。"
            "real 模式必须配置 API key：无 key 本地网关（仅 --base-url）当前不支持"
            "（LiteLLMBackend 强制 api_key）——请配置真实 key 后重跑"
        )

    real_backend = _make_real_backend(model=args.model, api_key=api_key, base_url=base_url)
    # 头部：跑分前先亮明成本与可重放边界
    stats = scene_stats(cases)
    by_scene = stats.get("by_scene", {})
    scene_n = {s: int(by_scene.get(s, {}).get("total", 0)) for s in SCENES}
    print("=" * 100)
    print("商品审核 Agent · Evaluation 正式跑分（rule 确定性初筛 vs real 真实 LLM）")
    print("=" * 100)
    print(f"数据集: {data_path}（{len(cases)} 条）| 工具世界: {EVAL_WORLD_LABEL} | real 模型: {args.model}")
    print("scene 分布: " + " | ".join(f"{s}={scene_n[s]}" for s in SCENES))
    print("API key: 已配置（--api-key / 环境变量 / .env）（值不入日志/报告/JSON）")
    print(
        f"[NOTE] real 侧真实调用 LLM（有费用、非确定性、不可重放）；"
        f"建议先 --limit 10 冒烟 —— 本次跑 {len(cases)} 条"
    )
    print(
        "[NOTE] 两臂预算恒为生产默认档（LLM_CALLS=10 / TOOL_CALLS=15，评测不覆盖 Guardrail）："
        "real 臂超限 → HUMAN_REVIEW 是生产语义，归因见 R3 截胡维度"
    )

    payload, extra = await run_comparison(
        cases=cases,
        real_backend=real_backend,
        model_label=args.model,
        data_path=str(data_path),
        real_concurrency=args.concurrency,
    )

    out_path: str | None = None
    if args.out:
        out_path = str(args.out)
        _write_payload(Path(out_path), payload)
    print()
    print_report(payload, extra, out_path=out_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_main(argv))
    except Exception as exc:  # 任何失败 → 非零退出（CI 可捕获）
        print(f"[FAIL] 跑分失败: {exc!r}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
