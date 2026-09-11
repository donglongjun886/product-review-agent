"""决策序列 Regression：对评测集跑三方案，把 EvalRecord 决策序列 hash 与基线快照比对。

任何改动（screening 修正 / RAG / LLM 接入）若改变三方案在该集上的决策 → 回归报错（退出码
1），防静默行为漂移。支持 v1（Phase 1，35 案）与 v2（Phase 2 正式集，320 案）两条路径；
v2 基线 ``eval_data/v2/regression_baseline.json`` 由确定性跑分录制、git 入库（见
tests/test_regression_v2.py 的守护断言）。``--data`` 给 v1/v2 键或显式 JSONL 路径，
``--record`` 强制重录基线，``--baseline`` 覆盖基线路径，``--schemes`` 选回归方案子集。

基线默认存 ``<数据目录>/regression_baseline.json``；首次运行（基线不存在）自动记录并报
"RECORDED"，之后比对报 REGRESSION PASS/FAIL。退出码：PASS=0 / FAIL=1 / 异常=非零。
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
# 已知数据集（键 → (数据 JSONL, 基线 JSON)）：--data 可给键（v1/v2）或显式路径
KNOWN_DATASETS: dict[str, tuple[str, str]] = {
    "v1": (
        "eval_data/v1/cases_v1.jsonl",
        "eval_data/v1/regression_baseline.json",
    ),
    "v2": (
        "eval_data/v2/cases_v2.jsonl",
        "eval_data/v2/regression_baseline.json",
    ),
}


def _resolve_dataset(data_arg: str) -> tuple[str, str]:
    """把 --data 解析为 (数据 JSONL 路径, 推断基线路径)。

    已知键（v1/v2）→ 键内 (data, baseline)；显式 JSONL 路径 → 基线取同目录
    ``regression_baseline.json``，防「指 v2 数据却比 v1 基线」。
    """
    if data_arg in KNOWN_DATASETS:
        return KNOWN_DATASETS[data_arg]
    return data_arg, str(Path(data_arg).parent / "regression_baseline.json")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluation Regression：三方案决策序列 hash vs 基线快照"
        "（确定性重放扩展；v1/v2 两路径，--data 给键或显式路径）"
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help=(
            "评测集：v1/v2 键或 JSONL 路径"
            f"（默认 {DEFAULT_DATA}；键见 {list(KNOWN_DATASETS)}）"
        ),
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="基线快照 JSON 路径（默认取 --data 同目录 regression_baseline.json；缺失自动记录）",
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
    data_path, baseline_default = _resolve_dataset(args.data)
    baseline_file = Path(args.baseline) if args.baseline else Path(baseline_default)
    schemes = tuple(args.schemes)
    label = Path(data_path).parent.name  # v1 / v2（输出标明数据集）

    record = args.record or not baseline_file.exists()
    if record:
        snapshot = await compute_current_snapshot(data_path, schemes=schemes)
        write_baseline(snapshot, baseline_file)
        print(f"[RECORDED] {label} 基线快照已写入 {baseline_file}（cases={snapshot['total_cases']}）")
        print(f"            digest = {snapshot['digest']}")
        return 0

    report = await run_regression(data_path, baseline_file, schemes=schemes)
    print("=" * 90)
    print(f"商品审核 Agent · Evaluation Regression（数据集: {label}）")
    print("=" * 90)
    print(f"评测集: {data_path}")
    print(f"基线:   {baseline_file}（基线 {report.baseline_cases} 条 vs 当前 {report.current_cases} 条）")
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
