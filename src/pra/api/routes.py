"""API 路由层 —— HTTP 接入面的端点装配（总链路 A·1「HTTP 接入」节点；00 §15 api 目录）。

端点：
- ``POST /api/v1/reviews``：受理一次审核案件 —— 请求体即 domain ``ProductReviewCase``
  （extra="forbid"，字段即 OpenAPI 文档，见 schemas.py 说明），调用 ``run_review`` 同步
  执行完整调查图后返回 ``ReviewRunResult``（HTTP 200）。错误语义：
  - 请求体不合法（缺字段/未知字段/类型错）→ FastAPI 校验层自动 422（不进本路由）；
  - case 语义/图执行异常 → 统一捕获转 ``HTTPException 500``：detail 为**人读信息**
    （异常类型 + 消息），**不暴露堆栈**（堆栈仅打日志，防内部细节泄漏给调用方）。
- ``GET /api/v1/health``：存活探针，返回 ``{"status": "ok"}``（负载均衡/容器健康检查用）。

演进路径：async 端点语义保持"同步执行完再返回"；未来 MQ/worker 化后本路由退化为
**受理口**（校验 + 投 ``product_review_request`` topic + 立即返回受理回执），真正执行
移交给消费同一 ``run_review`` 的 worker（service.py docstring 演进说明）—— 届时本文件
新增查询/回调端点，POST 语义与响应信封同步调整。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from pra.api.schemas import ReviewRunResult
from pra.api.service import run_review
from pra.domain.models import ProductReviewCase

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")

__all__ = ["router"]


@router.post(
    "/reviews",
    response_model=ReviewRunResult,
    status_code=status.HTTP_200_OK,
    summary="受理并执行一次复杂风险调查（同步返回最终裁决）",
    tags=["reviews"],
)
async def create_review(case: ProductReviewCase) -> ReviewRunResult:
    """POST /api/v1/reviews —— 总链路 A·1 主端点。

    请求体 = ``ProductReviewCase``（P_88231/M_5512 等真实案件快照），响应 =
    ``ReviewRunResult{run_id, review_decision}``。run_id 未在 body 中提供（它属运行
    上下文而非案件事实），由 service 自动生成 uuid4 hex 并作为 LangGraph thread_id（O-6）。
    """
    try:
        return await run_review(case)
    except Exception as exc:  # 图执行期异常：人读信息转 500，堆栈留日志不外泄
        logger.exception("run_review 执行失败 case_id=%s", case.case_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"案件调查执行失败（{type(exc).__name__}）：{exc}",
        ) from exc


@router.get(
    "/health",
    summary="存活探针",
    tags=["health"],
)
async def health() -> dict:
    """GET /api/v1/health —— 存活探针（不触图构建，恒定轻量返回）。"""
    return {"status": "ok"}
