"""Evaluation 三方案（rule / single_call_llm / agent）对比跑分入口。

``--data`` 指定评测集（默认 v1）、``--schemes`` 选方案子集（默认全三方案）、``--smoke`` /
``--smoke-limit`` 跑确定性子集、``--abstain-threshold`` 改 Single-call REJECT 候选转人工的
置信门槛（默认 0.7）。全链路确定性：无真 LLM / 无网络 / 无随机；退出码 0 = 全部 case 跑通
（任何 scheme 抛异常 → 非零退出并打印 traceback，供 CI 捕获）。

观测开关（只影响观测，不改判定与 metrics）：

- ``--experiment NAME``：实验版本名 → ``PRA_LANGFUSE_EXPERIMENT``，是评测 root trace 的
  ``version`` 与 trace_id 派生键，必须在跑评测之前设好；缺省取该环境变量，再缺省 ``baseline``；
- ``--session ID``：``PRA_LANGFUSE_SESSION``，缺省自动 ``eval-<UTC>-<4hex>``，整轮 trace 聚成
  一个 session，UI 按 session 过滤即「这一轮评测的全部案件」；
- ``--tag TAG``（可重复）：额外标签，仅打印在最终报告行，不进 EvalContext / 不改 metrics；
- ``--langfuse``：显式开启观测 —— 无凭据 / SDK 未装时只提示并照常跑完（绝不因观测失败中断
  评测），生效时跑前打印 experiment/session，收尾 ``flush_tracer()`` 一次。

不传 ``--langfuse`` 时 stdout 与加这些参数之前逐字节一致（NullTracer 全程 no-op、零网络）。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import UTC, datetime
from uuid import uuid4

from pra.evaluation.harness.base import EvalContext
from pra.evaluation.report import print_report
from pra.evaluation.runner import ALL_SCHEMES, EvaluationRunner
from pra.observability.tracing import flush_tracer, get_tracer

DEFAULT_DATA = "eval_data/v1/cases_v1.jsonl"
#: 缺省实验版本名（与 pra.observability.tracing 同口径）。
DEFAULT_EXPERIMENT = "baseline"
#: Langfuse UI 地址缺省值（与 tracing.make_tracer 同口径）。
DEFAULT_LANGFUSE_HOST = "http://localhost:3000"


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
    parser.add_argument(
        "--experiment",
        default=None,
        help=(
            "实验版本名（写入 PRA_LANGFUSE_EXPERIMENT；缺省取该环境变量，再缺省 "
            f"{DEFAULT_EXPERIMENT}）—— 决定评测 root trace 的 version/trace_id 派生键"
        ),
    )
    parser.add_argument(
        "--session",
        default=None,
        help=(
            "Langfuse 会话分组（PRA_LANGFUSE_SESSION；缺省自动 eval-<UTC时间戳>-<4位随机>，"
            "把整轮 trace 聚成一个 session）"
        ),
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=None,
        metavar="TAG",
        help="额外标签（可重复；仅打印在最终报告行，不进评测上下文 / 不改 metrics）",
    )
    parser.add_argument(
        "--langfuse",
        action="store_true",
        help="显式开启 Langfuse 观测（无凭据时只提示并照常跑完；收尾 flush 一次）",
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------------------
# 观测开关辅助（只读环境变量 / 只打印；绝不参与判定与 metrics）
# --------------------------------------------------------------------------------------

def _configure_experiment(name: str | None) -> str:
    """把实验版本名写进 ``PRA_LANGFUSE_EXPERIMENT`` 并返回生效值。

    ``AgentScheme`` 构造 trace_id 时读该环境变量（``uuid5(experiment:case:agent)``），
    因此必须在跑评测之前设好。缺省取环境变量，再缺省 ``baseline``。
    """
    if name is None:
        current = (os.environ.get("PRA_LANGFUSE_EXPERIMENT") or "").strip()
        if current:
            return current
        name = DEFAULT_EXPERIMENT
    os.environ["PRA_LANGFUSE_EXPERIMENT"] = name
    return name


def _configure_session(name: str | None) -> str:
    """把会话 ID 写进 ``PRA_LANGFUSE_SESSION`` 并返回生效值。

    未显式指定且环境变量为空 → 自动 ``eval-<UTC 时间戳>-<4 位随机>``。仅影响观测关联
    字段，不影响判定 / metrics。
    """
    resolved = (name or "").strip() or (os.environ.get("PRA_LANGFUSE_SESSION") or "").strip()
    if not resolved:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        resolved = f"eval-{stamp}-{uuid4().hex[:4]}"
    os.environ["PRA_LANGFUSE_SESSION"] = resolved
    return resolved


def _langfuse_host() -> str:
    """Langfuse 服务地址（环境变量 > ``Settings``（读 .env）> 缺省）；异常绝不抛。"""
    try:
        from pra.observability.tracing import _env_or_settings

        return _env_or_settings("LANGFUSE_HOST", "langfuse_host") or DEFAULT_LANGFUSE_HOST
    except Exception:  # noqa: BLE001 - 观测旁路：配置来源失败不得影响评测
        return DEFAULT_LANGFUSE_HOST


def _announce_observability(
    *, want_langfuse: bool, experiment: str, session: str
) -> bool:
    """跑前打印观测状态，返回「观测实际生效」与否（纯提示，不改评测行为）。

    - 传了 ``--langfuse`` 且观测生效 → 打印 experiment / session / host；
    - 传了但未生效（无凭据 / SDK 未装 / 显式关闭）→ 打印 ``NullTracer`` 原因 + 启用
      方式后继续跑（绝不因观测中断评测）；
    - 未传 → 不打印任何东西（默认路径零噪音）。
    """
    if not want_langfuse:
        return False
    try:
        tracer = get_tracer()
    except Exception as exc:  # noqa: BLE001 - 观测旁路：装配失败不得中断评测
        print(f"[langfuse] tracing disabled（tracer 装配失败: {exc!r}）—— 评测照常跑")
        return False
    if getattr(tracer, "enabled", False):
        print(
            f"[langfuse] tracing enabled | experiment={experiment} session={session} "
            f"host={_langfuse_host()}"
        )
        return True
    reason = getattr(tracer, "reason", "disabled")
    print(f"[langfuse] tracing disabled（NullTracer: {reason}）—— 评测照常跑，本次无 trace 上报")
    print(
        "[langfuse] 启用方式：仓库根 .env 设 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY"
        " / LANGFUSE_HOST（PRA_LANGFUSE_ENABLED 勿为 0），并安装可选依赖 "
        "uv sync --extra observability"
    )
    return False


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    experiment = _configure_experiment(args.experiment)
    session = _configure_session(args.session)
    tracing_on = _announce_observability(
        want_langfuse=args.langfuse, experiment=experiment, session=session
    )
    try:
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
        if args.tag:
            print(f"[tag] 额外标签: {', '.join(args.tag)}")
        if tracing_on:
            print(
                f"[langfuse] experiment={experiment} session={session} "
                f"host={_langfuse_host()} —— 在 Langfuse UI 按 session 过滤本轮评测"
            )
        return 0
    finally:
        # 整轮只 flush 一次（评测不 per-case flush）；异常路径同样 flush。
        flush_tracer()


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_main(argv))
    except Exception as exc:  # 任何失败 → 非零退出（CI 可捕获）
        print(f"[FAIL] EvaluationRunner 运行失败: {exc!r}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
