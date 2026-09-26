"""HTTP 接入面的请求/响应数据契约。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from pra.domain.models import ReviewDecision

__all__ = ["ReviewRunResult"]


class ReviewRunResult(BaseModel):
    """HTTP 响应包装 —— 一次调查执行的最小结果信封。"""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(description="本次调查运行 ID（= LangGraph thread_id）")
    review_decision: ReviewDecision = Field(description="图终态最终裁决（ReviewDecision 全量快照）")
