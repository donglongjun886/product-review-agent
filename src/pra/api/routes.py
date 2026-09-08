"""API 路由层 —— HTTP 接入面的端点装配（总链路 A·1「HTTP 接入」节点；00 §15 api 目录）。

端点：
- ``POST /api/v1/reviews``：受理一次审核案件 —— 请求体即 domain ``ProductReviewCase``
  （extra="forbid"，字段即 OpenAPI 文档，见 schemas.py 说明），调用
  ``persist_service.process_review`` **受理即分流**：先做 Screening 三分流
  （``pra.screening.engine.triage``）—— COMPLEX 走 ``run_and_persist`` 同步执行完整
  调查图**并落库**；PASS/REJECT 走 ``run_screening_direct`` 规则直判**并落库**
  （五表：review_case/review_run/review_trace/review_evidence/review_result，见
  pra/infra/persist_service.py），随后按同构 ``ReviewRunResult`` 形状返回
  （HTTP 200）——**HTTP 响应形状与切换前一致**（run_id + review_decision），现有
  TestClient 断言不受影响。错误语义：
  - 请求体不合法（缺字段/未知字段/类型错）→ FastAPI 校验层自动 422（不进本路由）；
  - triage/图执行/落库异常 → 统一捕获转 ``HTTPException 500``：detail 为**人读信息**
    （异常类型 + 消息），**不暴露堆栈**（堆栈仅打日志，防内部细节泄漏给调用方）。
- ``GET /api/v1/health``：存活探针，返回 ``{"status": "ok"}``（负载均衡/容器健康检查用）。

演进路径（2026-09 接线说明）：原 ``service.run_review``（纯执行、不落库）**保留**，
供无 DB 场景 / 单测 / 未来 MQ worker 消费复用 —— 本端点已切到
``persist_service.process_review``（triage 分流 + 执行 + 落库闭环）；MQ/worker 化后本
路由退化为**受理口**（校验 + 投 ``product_review_request`` topic + 立即返回受理回执），
真正执行移交给消费同一落库入口的 worker（persist_service.py docstring 演进说明）——
届时本文件新增查询/回调端点，POST 语义与响应信封同步调整。
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
    """POST /api/v1/reviews —— 总链路 A·1 主端点（受理即分流 → 接入 → 落库闭环）。

    请求体 = ``ProductReviewCase``（P_88231/M_5512 等真实案件快照）。受理后先做
    Screening 三分流：verdict=COMPLEX → 执行调查图并落库（Agent run）；PASS/REJECT →
    规则直判落库（SCREENING_DIRECT run，确定性终裁，不再进 Agent）。随后按原响应形状
    返回 ``ReviewRunResult{run_id, review_decision}``（HTTP 形状不变；额外落库的
    review_case.triage_result / review_run / result / trace / evidence 行不对 HTTP
    暴露）。run_id 未在 body 中提供（它属运行上下文而非案件事实），由
    ``process_review`` 各分支自动生成 uuid4 hex（Agent 路径 = LangGraph thread_id，O-6）。
    """
    try:
        summary = await process_review(case)
        return ReviewRunResult(
            run_id=summary["run_id"],
            review_decision=summary["decision"],
        )
    except Exception as exc:  # triage/图执行/落库期异常：人读信息转 500，堆栈留日志不外泄
        logger.exception("process_review 执行失败 case_id=%s", case.case_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"案件审核执行失败（{type(exc).__name__}）：{exc}",
        ) from exc


@router.get(
    "/health",
    summary="存活探针",
    tags=["health"],
)
async def health() -> dict:
    """GET /api/v1/health —— 存活探针（不触图构建，恒定轻量返回）。"""
    return {"status": "ok"}
