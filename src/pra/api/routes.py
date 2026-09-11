"""HTTP 路由层 —— 端点装配。

``POST /api/v1/reviews``：请求体即 domain ``ProductReviewCase``（``extra="forbid"``），经
``persist_service.process_review`` 受理即分流 —— COMPLEX 走 ``run_and_persist`` 执行调查图并
落库，PASS/REJECT 走 ``run_screening_direct`` 规则直判并落库（表：review_case / review_run /
review_trace / review_evidence / review_result；直判路径无 review_trace 行），再按
``ReviewRunResult`` 形状返回 HTTP 200。

错误语义：请求体不合法由 FastAPI 校验层自动 422；triage/图执行/落库异常统一转
``HTTPException 500``，detail 为固定人读文案（仅附异常类型短名）。**绝不拼 str(exc) 或
堆栈** —— 防 SQLAlchemy 等内部异常把 SQL/表列名/绑定值外泄；完整异常与堆栈只进
``logger.exception``。

``GET /api/v1/health`` 存活探针返回 ``{"status": "ok"}``；``service.run_review``（纯执行、
不落库）保留供无 DB 场景与单测复用。
"""

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
    """POST /api/v1/reviews —— 受理即分流，执行并落库后返回最终裁决。

    COMPLEX 执行调查图（Agent run）；PASS/REJECT 规则直判（SCREENING_DIRECT run，
    确定性终裁，不再进 Agent）。响应固定为 ``ReviewRunResult{run_id,
    review_decision}``；额外落库的 review_case.triage_result 与 run/result/trace/
    evidence 行不对 HTTP 暴露。run_id 属运行上下文而非案件事实，请求体不提供，由
    ``process_review`` 各分支生成 uuid4 hex（Agent 路径 = LangGraph thread_id）。
    """
    try:
        summary = await process_review(case)
        return ReviewRunResult(
            run_id=summary["run_id"],
            review_decision=summary["decision"],
        )
    except Exception as exc:  # triage/图执行/落库期异常：固定文案转 500，异常消息/堆栈留日志不外泄
        logger.exception("process_review 执行失败 case_id=%s", case.case_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            # 只附异常类型短名（供排障）；绝不拼 str(exc) —— 防内部异常细节外泄。
            detail=f"案件审核执行失败（{type(exc).__name__}），请稍后重试或联系管理员",
        ) from exc


@router.get(
    "/health",
    summary="存活探针",
    tags=["health"],
)
async def health() -> dict:
    """GET /api/v1/health —— 存活探针（不触图构建，恒定轻量返回）。"""
    return {"status": "ok"}
