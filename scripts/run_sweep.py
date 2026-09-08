"""run_sweep.py —— Evaluation Phase 2 Evidence 阈值 sweep 入口（docs/02-evaluation.md §5）。

用法::

    uv run python scripts/run_sweep.py                    # 双变量全网格 + 全三方案
    uv run python scripts/run_sweep.py --variable EVIDENCE_MIN_SIM
    uv run python scripts/run_sweep.py --schemes agent rule --csv sweep_out.csv
    uv run python scripts/run_sweep.py --smoke --csv /tmp/sweep_smoke.csv

P-5 口径：只 sweep Evidence 阈值、单参数（0.60..0.90 步 0.05），另一常量取当前默认
（min_sim=0.70 / strong=0.85）固定；CONFIDENCE_ABSTAIN_THRESHOLD 固定 0.7 不参与扫描。
只动 EvalContext（配置注入），判定逻辑零改动 —— 生效范围见 sweep.py 模块 docstring
（agent 的评测侧相似度证据视图；rule / single_call 不读相似度 → 行恒定）。

退出码 0 = 全部档点跑通；任何异常 → 非零退出。CSV 输出含 docs §5.2 七项指标。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from pra.evaluation.dataset.loader import load_dataset, smoke_subset
from pra.evaluation.runner import ALL_SCHEMES
from pra.evaluation.sweep import (
    EVIDENCE_GRID,
    SWEEP_VARIABLES,
    ThresholdSweepRunner,
    curve_highlights,
    write_csv,
)

DEFAULT_DATA = "eval_data/v1/cases_v1.jsonl"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluation Phase 2：Evidence 阈值单参数 sweep（EVIDENCE_MIN_SIM / EVIDENCE_STRONG）"
    )
    parser.add_argument("--data", default=DEFAULT_DATA, help=f"评测集 JSONL（默认 {DEFAULT_DATA}）")
    parser.add_argument(
        "--variable",
        nargs="*",
        default=list(SWEEP_VARIABLES),
        choices=list(SWEEP_VARIABLES),
        help="要扫的变量（默认两个 Evidence 常量都扫）",
    )
    parser.add_argument(
        "--schemes",
        nargs="*",
        default=list(ALL_SCHEMES),
        choices=list(ALL_SCHEMES),
        help="要观测的方案子集（默认全三方案；rule/single_call 对相似度阈值不敏感）",
    )
    parser.add_argument("--smoke", action="store_true", help="冒烟：只跑前 N 条")
    parser.add_argument("--smoke-limit", type=int, default=10, help="smoke 上限（默认 10）")
    parser.add_argument("--csv", default=None, help="把每档七项指标写 CSV（默认不写）")
    return parser.parse_args(argv)


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cases = load_dataset(args.data)
    if args.smoke:
        cases = smoke_subset(cases, args.smoke_limit)

    runner = ThresholdSweepRunner(data_path=args.data)
    result = await runner.run(
        cases=cases,
        variables=tuple(args.variable),
        schemes=tuple(args.schemes),
    )

    print("=" * 100)
    print("商品审核 Agent · Evaluation Phase 2 Evidence Threshold Sweep")
    print("=" * 100)
    print(f"数据集: {args.data}" + (f"   [smoke {args.smoke_limit} 条]" if args.smoke else ""))
    print(f"案数: {result.total_cases}    网格: {[f'{v:.2f}' for v in EVIDENCE_GRID]}")
    print(f"扫变量: {list(args.variable)}    schemes: {list(args.schemes)}")
    print("观察指标（docs §5.2 七项）: Accuracy/Precision/Recall/FPR/FNR/human_review_rate/automation_coverage")
    for n in result.notes:
        print(f"注记: {n}")
    print("-" * 100)

    # 每档一张紧凑行（按变量分组；每 scheme 一行：value→七项）
    for variable in args.variable:
        points = result.points_of(variable)
        print(f"\n[{variable}] 各档指标（行 = scheme；列为该档值的 7 项指标）")
        print("（列格式: value=acc prec recall fpr fnr human_review_rate automation_coverage）")
        for scheme in args.schemes:
            parts: list[str] = []
            for p in points:
                m = p.metrics_by_scheme.get(scheme)
                if m is None:
                    parts.append("-")
                    continue
                cells = [
                    f"{p.value:.2f}",
                    *(("–" if v is None else f"{v:.3f}")
                      for v in (
                          m.accuracy, m.precision, m.recall, m.fpr, m.fnr,
                          m.human_rate, m.automation,
                      )),
                ]
                parts.append("/".join(cells))
            print(f"{scheme:<16} " + "  ".join(parts))

    print("\n" + "-" * 100)
    print("四曲线要点（FPR / Recall / human_review_rate / Accuracy vs 阈值）:")
    for line in curve_highlights(result, schemes=tuple(args.schemes)):
        print("  " + line)

    if args.csv:
        csv_path = Path(args.csv)
        write_csv(result, csv_path, schemes=tuple(args.schemes))
        print(f"\n[CSV] 已写 {csv_path}")

    n_vars = len(result.points_of(args.variable[0])) if args.variable else 0
    print(
        f"\n[OK] Sweep 完成: 档点数={len(result.points)} "
        f"（{len(args.variable)} 变量 × {n_vars} 档）"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_main(argv))
    except Exception as exc:  # 任何失败 → 非零退出（CI 可捕获）
        print(f"[FAIL] Sweep 运行失败: {exc!r}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
