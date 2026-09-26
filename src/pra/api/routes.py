"""HTTP 路由层 —— 端点装配。"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from pra.api.schemas import ReviewRunResult
from pra.domain.models import ProductReviewCase
from pra.infra.persist_service import process_review

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

__all__ = ["router"]


@router.post(
    "/reviews",
    response_model=ReviewRunResult,
    status_code=status.HTTP_200_OK,
    summary="受理一次审核案件：Screening 三分流 → COMPLEX 走 Agent 调查 / PASS·REJECT 规则直判（同步执行 + 落库，返回最终裁决）",
    tags=["reviews"],
)
async def create_review(case: ProductReviewCase) -> ReviewRunResult:
    """POST /api/v1/reviews —— 受理即分流，执行并落库后返回最终裁决。"""
    try:
        summary = await process_review(case)
        return ReviewRunResult(
            run_id=summary["run_id"],
            review_decision=summary["decision"],
        )
    except Exception as exc:
        logger.exception("process_review 执行失败 case_id=%s", case.case_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"案件审核执行失败（{type(exc).__name__}），请稍后重试或联系管理员",
        ) from exc


@router.get(
    "/health",
    summary="存活探针",
    tags=["health"],
)
async def health() -> dict:
    """GET /api/v1/health —— 存活探针。"""
    return {"status": "ok"}
