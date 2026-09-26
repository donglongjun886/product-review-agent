"""FastAPI 应用工厂 —— 只装配应用壳（元信息 + 路由），不持有图/DB 等重对象。"""

from __future__ import annotations

from fastapi import FastAPI

from pra.api.routes import router

__all__ = ["create_app", "app"]


def create_app() -> FastAPI:
    """装配并返回 FastAPI 应用（路由 + 元信息；不含图/DB 等重对象）。"""
    application = FastAPI(
        title="product-review-agent API",
        description=(
            "电商平台商品内容治理 · 复杂风险调查 Agent 的 HTTP 接入面（总链路 A·1）。\n\n"
            "POST /api/v1/reviews 受理一次审核案件（商品快照 + 商家 + 事件类型），"
            "同步执行 LangGraph 复杂风险调查子图（hypothesize → plan → tools → reevaluate "
            "×N → decide，生产工具世界（商品/商家读 MySQL、案例/政策读真实 RAG）+ 组合根 "
            "pra.wiring 注入的真实 LLM 后端（读 .env 的 DEEPSEEK_API_KEY / DEEPSEEK_MODEL，"
            "缺 API key 时装配期显式失败）），返回最终裁决：PASS / REJECT / HUMAN_REVIEW + "
            "风险等级/类型 + 置信度 + 证据链 + 假设轨迹 + 预算快照。"
        ),
        version="0.1.0",
    )
    application.include_router(router)
    return application


# uvicorn 入口实例：`uv run uvicorn pra.api.app:app --reload`。
app = create_app()
