"""demo_langfuse_trace.py —— 真实案件跑一遍 Agent 并给出 Langfuse trace 链接（演示/证据）。

用途：面试演示与端到端证据 —— 走 **HTTP 主流程同一入口** ``pra.api.service.run_review``
跑一个真实案件（默认复古运动鞋 ``P_88231`` / 商家 ``M_5512``，与 README 快速开始、
``scripts/demo_walkthrough.py`` 同源构造，但本脚本**自包含**、不 import 其内部函数），
跑完打印 ``run_id`` / ``trace_id`` / 决策值与 UI 链接 ``{LANGFUSE_HOST}/trace/{trace_id}``。

**无凭据 / SDK 未装时的行为（刻意如此，docs/09 §2）**：打印
``tracing disabled (NullTracer: <reason>)`` + 如何启用，**仍然照常跑完 Agent 并打印决策**
（证明观测关闭时业务不受影响），退出码 **0**。

``run_id`` 为 32-hex（``uuid4().hex``）→ ``trace_id_from_run_id`` 原样返回 → Langfuse
trace 与 MySQL ``review_run.run_id`` 一一对应；``--run-id`` 可复现同一条 trace。

只依赖标准库 + 项目内模块；不联网（除非凭据已配置且 SDK 已装）。

用法::

    uv run python scripts/demo_langfuse_trace.py
    uv run python scripts/demo_langfuse_trace.py --run-id RUN_CASE_1
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from uuid import uuid4

from pra.api.service import run_review
from pra.domain.models import (
    ProductImage,
    ProductInfo,
    ProductReviewCase,
    ScreeningSignal,
    SkuInfo,
)
from pra.observability.tracing import (
    flush_tracer,
    get_tracer,
    trace_id_from_run_id,
)

__all__ = ["build_demo_case", "main"]

#: 演示商品图片（与 README / demo_walkthrough 同一 CDN 路径）。
_IMG_URL = "https://cdn.example.com/products/P_88231/img1.jpg"
#: Langfuse UI 地址缺省值（与 tracing.make_tracer 一致）。
DEFAULT_LANGFUSE_HOST = "http://localhost:3000"


def build_demo_case() -> ProductReviewCase:
    """构造真实案件（复古运动鞋 P_88231 / M_5512 / NEW_LISTING）。

    与 README 快速开始、``demo_walkthrough.build_demo_case`` 同口径（品牌空缺 + 复古
    风格词 + 四路机审信号 PASS），但**本脚本自包含**，不 import 其内部函数。
    """
    product = ProductInfo(
        product_id="P_88231",
        title="新款厚底复古跑鞋 女士百搭运动鞋",
        description="复古厚底设计，舒适百搭，适合日常通勤与运动。",
        category="女鞋/运动鞋",
        brand=None,
        sku_list=[SkuInfo(sku_id="S_1", color="米白", size="38", price=219.0)],
        images=[ProductImage(url=_IMG_URL, source="主图")],
        listing_time=datetime(2024, 9, 6, 14, 0, 0),  # naive datetime（DB DATETIME 口径，同 demo_walkthrough）
        version=3,
    )
    signals = [
        ScreeningSignal(name="KEYWORD", result="PASS", score=0.80),
        ScreeningSignal(name="LOGO_DETECT", result="PASS", score=0.20),
        ScreeningSignal(name="CATEGORY_RULE", result="PASS", score=0.95),
        ScreeningSignal(name="DUPLICATE_CHECK", result="PASS", score=0.10),
    ]
    return ProductReviewCase(
        case_id="CASE_20240907_001",
        product=product,
        merchant_id="M_5512",
        event_type="NEW_LISTING",
        screening_signals=signals,
    )


def _langfuse_host() -> str:
    """Langfuse UI 地址（环境变量 > ``Settings``（读 .env）> 缺省）；异常绝不抛。"""
    try:
        from pra.observability.tracing import _env_or_settings

        return _env_or_settings("LANGFUSE_HOST", "langfuse_host") or DEFAULT_LANGFUSE_HOST
    except Exception:  # noqa: BLE001 - 观测旁路：配置来源失败不得影响演示
        return DEFAULT_LANGFUSE_HOST


def _tracing_state() -> tuple[bool, str]:
    """返回 (观测是否生效, 说明)。未生效说明形如 ``NullTracer: <reason>``。"""
    try:
        tracer = get_tracer()
    except Exception as exc:  # noqa: BLE001 - 观测旁路
        return False, f"NullTracer: tracer 装配失败 {exc!r}"
    if getattr(tracer, "enabled", False):
        return True, type(tracer).__name__
    return False, f"NullTracer: {getattr(tracer, 'reason', 'disabled')}"


def _has_credentials() -> bool:
    """凭据是否已配置（只用于 UI 链接旁提示语的措辞；异常绝不抛）。"""
    try:
        from pra.observability.tracing import _env_or_settings

        return bool(
            _env_or_settings("LANGFUSE_PUBLIC_KEY", "langfuse_public_key")
            and _env_or_settings("LANGFUSE_SECRET_KEY", "langfuse_secret_key")
        )
    except Exception:  # noqa: BLE001 - 观测旁路
        return False


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="真实案件跑一遍 Agent，打印 run_id / trace_id / 决策 + Langfuse UI 链接"
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="指定 run_id（32-hex 时 trace_id 原样等于它；缺省 uuid4().hex）",
    )
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    """跑一个真实案件 → 打印关联信息与决策 → flush；退出码恒 0（观测关闭也照跑）。"""
    args = _parse_args(argv)
    case = build_demo_case()
    run_id = args.run_id or uuid4().hex
    tracing_on, tracing_note = _tracing_state()

    print("=" * 76)
    print(
        f"demo_langfuse_trace: case={case.case_id} | product={case.product.product_id} "
        f"| merchant={case.merchant_id} | event={case.event_type}"
    )
    if tracing_on:
        print(f"[langfuse] tracing enabled ({tracing_note})")
    else:
        print(f"tracing disabled ({tracing_note})")
        print(
            "[langfuse] 启用方式：仓库根 .env 设 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY"
            " / LANGFUSE_HOST，并安装可选依赖 uv sync --extra observability"
        )
        print("[langfuse] 本次仍照常跑完 Agent —— 观测关闭不影响业务判定")
    print("=" * 76)

    result = await run_review(case, run_id=run_id)
    decision = result.review_decision
    trace_id = trace_id_from_run_id(result.run_id)

    print(f"run_id:              {result.run_id}")
    print(f"trace_id:            {trace_id}")
    print(f"decision:            {decision.decision.value}")
    print(f"risk_level:          {decision.risk_level.value}")
    print(f"risk_type:           {[t.value for t in decision.risk_type]}")
    print(f"decision_confidence: {decision.decision_confidence:.2f}")
    print(f"policy:              {decision.policy}")
    print(f"evidence:            {len(decision.evidence)} 条")
    budget = decision.budget_used
    print(
        f"budget:              llm_calls={budget.llm_calls} tool_calls={budget.tool_calls} "
        f"tokens={budget.tokens} latency_ms={budget.latency_ms}"
    )

    if tracing_on:
        print(f"UI:                  {_langfuse_host()}/trace/{trace_id}")
    elif _has_credentials():
        print(
            f"UI:                  {_langfuse_host()}/trace/{trace_id}"
            "   ← 凭据已配置但观测被关闭（见上），本次未上报"
        )
    else:
        print(
            f"UI（配置凭据后）:     {_langfuse_host()}/trace/{trace_id}"
            "   ← 当前无凭据，未上报"
        )
    flush_tracer()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
