"""run_ablation.py —— Evaluation Phase 2 Ablation 跑分入口（docs/02-evaluation.md §6）。

用法::

    uv run python scripts/run_ablation.py                       # 方案级 + 组件级 + 全量数据
    uv run python scripts/run_ablation.py --scheme-level        # 只跑方案级（2a/2b/2c）
    uv run python scripts/run_ablation.py --component-level     # 只跑组件级（full/−rag/…）
    uv run python scripts/run_ablation.py --data <path> --smoke # 冒烟（≤10 条）

确定性：无真 LLM / 网络 / 随机；退出码 0 = 全部变体跑通（任何异常 → 非零退出）。
报告含：并排指标表（含 abstention 五指标列）、2b−2a / 2c−2b 差异、组件相对 Full 的
决策差异、工具证据覆盖标注（docs §6 空消融风险）。

数据版本注记（Q11 低风险半）：默认数据为 v1（Phase 1，35 案）；v1 相对 Phase 2 正式集
属 smoke 级，正式口径为 v2 320 案（docs/02-evaluation.md §8）—— 运行默认 v1 时输出首行
会打印该标注，防 v1 数字被当正式结论（不翻默认数据：翻默认属行为变更，另行拍板）。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from pra.evaluation.ablation import (
    AblationRunner,
    print_ablation_report,
)
from pra.evaluation.dataset.loader import load_dataset, smoke_subset

DEFAULT_DATA = "eval_data/v1/cases_v1.jsonl"


def _is_default_v1(data_path: str) -> bool:
    """数据是否 = 脚本默认 v1 文件（Q11：需首行标注 v1 属 smoke 级，勿当正式结论）。"""
    return os.path.normpath(data_path) == os.path.normpath(DEFAULT_DATA)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluation Phase 2 Ablation：方案级（2a/2b/2c）+ 组件级消融（确定性）"
    )
    parser.add_argument("--data", default=DEFAULT_DATA, help=f"评测集 JSONL（默认 {DEFAULT_DATA}）")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--scheme-level", action="store_true", help="只跑方案级（2a/2b/2c）")
    group.add_argument("--component-level", action="store_true", help="只跑组件级（full/−rag/…）")
    parser.add_argument("--smoke", action="store_true", help="冒烟：只跑前 N 条")
    parser.add_argument("--smoke-limit", type=int, default=10, help="smoke 上限（默认 10）")
    return parser.parse_args(argv)


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cases = load_dataset(args.data)
    if args.smoke:
        cases = smoke_subset(cases, args.smoke_limit)
    runner = AblationRunner(data_path=args.data)
    result = await runner.run(
        cases=cases,
        scheme_level=args.scheme_level or not args.component_level,
        component_level=args.component_level or not args.scheme_level,
    )
    if _is_default_v1(args.data):
        # Q11 低风险半：v1 默认数字易被当正式结论 —— 输出首行标注数据版本
        #（v1 = Phase 1 集，相对 Phase 2 正式集属 smoke 级；不翻默认数据）。
        print(
            f"[数据版本] {DEFAULT_DATA} = Phase 1 v1 集（35 案，本次跑 {result.total_cases} 案），"
            "相对 Phase 2 正式集属 smoke 级 —— 正式口径为 v2 320 案"
            "（docs/02-evaluation.md §8）；以下 v1 数字仅对该数据集成立，勿当正式结论。"
        )
    print_ablation_report(result)
    print(
        f"\n[OK] Ablation 完成: cases={result.total_cases} "
        f"(smoke={args.smoke}) 方案级={bool(result.scheme_outcomes)} "
        f"组件级={bool(result.component_outcomes)} —— 全程确定性，无真 LLM/网络"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_main(argv))
    except Exception as exc:  # 任何失败 → 非零退出（CI 可捕获）
        print(f"[FAIL] Ablation 运行失败: {exc!r}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
