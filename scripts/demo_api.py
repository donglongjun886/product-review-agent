"""直接调 ``run_review`` 验收执行器入口（不经 HTTP）。

用例与 ``scripts/demo_walkthrough.py`` 的 ``build_demo_case`` 字段一致（复古运动鞋 P_88231 /
商家 M_5512）：默认 6 个 InMemory Tool + scripted LLM 桩，无需 API key，打印 run_id + 裁决
摘要，正常路径退出码 0。

预期结局：decision=HUMAN_REVIEW、risk_level=HIGH、risk_type 覆盖 POTENTIAL_IP_RISK +
EVASION_PATTERN、decision_confidence>=0.7、overrides=[]（scripted 桩确定性产出，可复现）。

HTTP 接入面另行验收：``uv run uvicorn pra.api.app:app --reload``（默认 http://127.0.0.1:8000）
后 ``POST /api/v1/reviews``（请求体即 ``build_demo_case()`` 的 ``model_dump(mode="json")``，
返回 ``{"run_id": ..., "review_decision": {...}}``），存活探针 ``GET /api/v1/health`` →
``{"status":"ok"}``。case 构造逻辑复制自 demo_walkthrough.py（不 import 它）。
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime

from pra.api.service import run_review
from pra.domain.models import (
    ProductImage,
    ProductInfo,
    ProductReviewCase,
    ScreeningSignal,
    SkuInfo,
)

# 图片 URL 与 demo_walkthrough 对齐（img1 主图）。
_IMG_URL = "https://cdn.example.com/products/P_88231/img1.jpg"


def build_demo_case() -> ProductReviewCase:
    """构造走查输入 case（字段对齐 demo_walkthrough.build_demo_case）。"""
    product = ProductInfo(
        product_id="P_88231",
        title="新款厚底复古跑鞋 女士百搭运动鞋",
        description="复古厚底设计，舒适百搭，适合日常通勤与运动。",
        category="女鞋/运动鞋",
        brand=None,  # 品牌真空缺 —— "规避品牌识别"调查的起点信号
        sku_list=[SkuInfo(sku_id="S_1", color="米白", size="38", price=219.0)],
        images=[ProductImage(url=_IMG_URL, source="主图")],
        listing_time=datetime(2024, 9, 6, 14, 0, 0),  # naive datetime（DB DATETIME 口径）
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


def _print_summary(result) -> None:
    """打印 run_id + 裁决摘要（含预算占用）。"""
    d = result.review_decision
    b = d.budget_used
    print("=" * 76)
    print(f"run_id:            {result.run_id}")
    print(f"decision:          {d.decision.value}")
    print(f"risk_level:        {d.risk_level.value}")
    print(f"risk_type:         {[t.value for t in d.risk_type]}")
    print(f"decision_confidence: {d.decision_confidence:.2f}")
    print(f"policy:            {d.policy}")
    print(f"overrides:         {d.overrides}")
    print(f"evidence:          {len(d.evidence)} 条")
    print(f"budget:            llm_calls={b.llm_calls} tool_calls={b.tool_calls} "
          f"tokens={b.tokens} latency_ms={b.latency_ms}")
    print("=" * 76)


async def main() -> int:
    case = build_demo_case()
    print(f"demo_api 走查: case={case.case_id} | product={case.product.product_id} "
          f"| merchant={case.merchant_id} | event={case.event_type}")
    result = await run_review(case, run_id=f"RUN_{case.case_id}")
    _print_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
