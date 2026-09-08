"""run_regression.py —— Evaluation 决策序列 Regression（docs/02-evaluation.md §8 里程碑）。

用途：对 Phase 1 集（eval_data/v1/cases_v1.jsonl，35 条）跑 rule / single_call_llm /
agent 三方案，把 EvalRecord 决策序列 hash 与**已记录的基线快照**比对 —— 后续任何改动
（screening 修正 / RAG / LLM 接入）若改变三方案在该集上的决策 → 回归报错（退出码 1），
防静默行为漂移。

用法::

    uv run python scripts/run_regression.py                  # 比对（基线缺失 → 自动记录并 PASS）
    uv run python scripts/run_regression.py --record         # 强制重录基线（升级/有意变更后）
    uv run python scripts/run_regression.py --baseline <path> # 自定义基线路径

基线快照默认存 ``eval_data/v1/regression_baseline.json``（任务契约口径；--baseline 覆盖）。
首次运行（基线不存在）自动记录并报告 "RECORDED"；之后比对报 REGRESSION PASS/FAIL。
退出码：PASS=0 / FAIL=1 / 异常=非零（脚本内部 raise 后由 main 转非零）。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from pra.evaluation.regression import (
    ALL_SCHEMES,
    compute_current_snapshot,
    run_regression,
    write_baseline,
)

DEFAULT_DATA = "eval_data/v1/cases_v1.jsonl"
DEFAULT_BASELINE = "eval_data/v1/regression_baseline.json"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluation Regression：三方案决策序列 hash vs 基线快照（确定性重放扩展）"
    )
    parser.add_argument("--data", default=DEFAULT_DATA, help=f"评测集 JSONL（默认 {DEFAULT_DATA}）")
    parser.add_argument(
        "--baseline",
        default=DEFAULT_BASELINE,
        help=f"基线快照 JSON 路径（默认 {DEFAULT_BASELINE}；缺失自动记录）",
    )
    parser.add_argument(
        "--schemes",
        nargs="*",
        default=list(ALL_SCHEMES),
        choices=list(ALL_SCHEMES),
        help="要回归的方案子集（默认全三方案）",
    )
    parser.add_argument("--record", action="store_true", help="强制（重新）记录基线，不比对")
    return parser.parse_args(argv)


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    baseline_file = Path(args.baseline)
    schemes = tuple(args.schemes)

    record = args.record or not baseline_file.exists()
    if record:
        snapshot = await compute_current_snapshot(args.data, schemes=schemes)
        write_baseline(snapshot, baseline_file)
        print(f"[RECORDED] 基线快照已写入 {baseline_file}（cases={snapshot['total_cases']}）")
        print(f"            digest = {snapshot['digest']}")
        return 0

    report = await run_regression(args.data, baseline_file, schemes=schemes)
    print("=" * 90)
    print("商品审核 Agent · Evaluation Regression")
    print("=" * 90)
    print(f"数据集: {args.data}（基线 {report.baseline_cases} 条 vs 当前 {report.current_cases} 条）")
    print(f"比对结果: REGRESSION {report.status}")
    if report.ok:
        print("三方案决策序列与基线快照一致 —— 无静默行为漂移。")
    else:
        print("检测到行为漂移（相对基线快照）:")
        for line in report.summary:
            print(f"  · {line}")
        for scheme, diffs in sorted(report.mismatches.items()):
            head = diffs[:5]
            for i, cid, base, cur in head:
                print(f"    scheme={scheme} case={cid} 基线 {base} → 当前 {cur}")
            if len(diffs) > 5:
                print(f"    … 其余 {len(diffs) - 5} 处差异（见 summary）")
    print("=" * 90)
    print(f"退出码: {0 if report.ok else 1}（PASS=0 / FAIL=1）")
    return 0 if report.ok else 1


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_main(argv))
    except Exception as exc:  # 任何失败 → 非零退出（CI 可捕获）
        print(f"[FAIL] Regression 运行失败: {exc!r}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
