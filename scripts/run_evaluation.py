"""run_evaluation.py —— Evaluation Phase 1 三方案对比跑分入口。

用法::

    uv run python scripts/run_evaluation.py                 # 默认全三方案 + 全量数据
    uv run python scripts/run_evaluation.py --smoke         # 冒烟（≤10 条）
    uv run python scripts/run_evaluation.py --schemes rule agent
    uv run python scripts/run_evaluation.py --data <path> --smoke-limit 5

全链路确定性：无真 LLM / 无网络 / 无随机；退出码 0 = 全部 case 跑通（任何 scheme
抛异常 → 非零退出并打印 traceback，供 CI 捕获）。
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from pra.evaluation.harness.base import EvalContext
from pra.evaluation.report import print_report
from pra.evaluation.runner import ALL_SCHEMES, EvaluationRunner

DEFAULT_DATA = "eval_data/v1/cases_v1.jsonl"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluation Phase 1：rule / single_call_llm / agent 三方案对比（确定性、不落 DB）"
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help=f"评测集 JSONL 路径（默认 {DEFAULT_DATA}）",
    )
    parser.add_argument(
        "--schemes",
        nargs="*",
        default=list(ALL_SCHEMES),
        choices=list(ALL_SCHEMES),
        help="要跑的方案子集（默认全三方案）",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="冒烟模式：只跑数据集前 N 条（确定性子集）",
    )
    parser.add_argument(
        "--smoke-limit",
        type=int,
        default=10,
        help="smoke 子集上限（默认 10）",
    )
    parser.add_argument(
        "--abstain-threshold",
        type=float,
        default=0.7,
        help="Single-call REJECT 候选转人工的置信门槛（默认 0.7）",
    )
    return parser.parse_args(argv)


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ctx = EvalContext(abstain_confidence_threshold=args.abstain_threshold)
    runner = EvaluationRunner(data_path=args.data, ctx=ctx)
    result = await runner.run(
        include=args.schemes,
        smoke=args.smoke,
        smoke_limit=args.smoke_limit,
    )
    print_report(result)
    print(
        f"\n[OK] 跑分完成: schemes={list(args.schemes)} "
        f"cases={result.total_cases} (smoke={result.smoke}) —— 全程确定性，无真 LLM/网络"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_main(argv))
    except Exception as exc:  # 任何失败 → 非零退出（CI 可捕获）
        print(f"[FAIL] EvaluationRunner 运行失败: {exc!r}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
