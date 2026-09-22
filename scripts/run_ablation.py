"""Ablation 跑分入口：方案级（2a/2b/2c）确定性消融。

``--data`` 指定评测集（默认 v1），``--smoke`` 只跑前 ``--smoke-limit`` 条。无真 LLM /
网络 / 随机；退出码 0 = 全部变体跑通（任何异常 → 非零退出）。报告含并排指标表（含
abstention 五指标列）、2b−2a / 2c−2b 差异与工具证据覆盖标注。

数据版本注记：默认数据是 v1（Phase 1，35 案），相对 Phase 2 正式集属 smoke 级，正式口径为
v2 320 案；跑默认 v1 时输出首行会打印该标注，防 v1 数字被当正式结论。不翻默认数据。
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
    """数据路径是否等于脚本默认的 v1 文件。"""
    return os.path.normpath(data_path) == os.path.normpath(DEFAULT_DATA)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluation Phase 2 Ablation：方案级（2a/2b/2c）消融（确定性）"
    )
    parser.add_argument("--data", default=DEFAULT_DATA, help=f"评测集 JSONL（默认 {DEFAULT_DATA}）")
    parser.add_argument("--smoke", action="store_true", help="冒烟：只跑前 N 条")
    parser.add_argument("--smoke-limit", type=int, default=10, help="smoke 上限（默认 10）")
    return parser.parse_args(argv)


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cases = load_dataset(args.data)
    if args.smoke:
        cases = smoke_subset(cases, args.smoke_limit)
    runner = AblationRunner(data_path=args.data)
    result = await runner.run(cases=cases)
    if _is_default_v1(args.data):
        # 默认 v1 数字易被当正式结论：输出首行标注数据版本。
        print(
            f"[数据版本] {DEFAULT_DATA} = Phase 1 v1 集（35 案，本次跑 {result.total_cases} 案），"
            "相对 Phase 2 正式集属 smoke 级 —— 正式口径为 v2 320 案"
            "；以下 v1 数字仅对该数据集成立，勿当正式结论。"
        )
    print_ablation_report(result)
    print(
        f"\n[OK] Ablation 完成: cases={result.total_cases} "
        f"(smoke={args.smoke}) 方案级={bool(result.scheme_outcomes)} "
        "—— 全程确定性，无真 LLM/网络"
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
