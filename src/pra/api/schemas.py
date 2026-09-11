"""HTTP 接入面的请求/响应数据契约。

请求体直接复用 domain 的 ``ProductReviewCase``（Pydantic v2 + ``extra="forbid"``，
字段描述即 OpenAPI 文档），不派生同构请求模型 —— 逐字段复制只会制造同步漂移点，
领域层是请求契约的单一事实源。

本层新增的是响应包装 ``ReviewRunResult``：把执行上下文（run_id，即 LangGraph
thread_id）与图终态裁决 ``ReviewDecision`` 打包成最小输出契约。``extra="forbid"``
保证响应不随实现悄悄多出字段。本文件只声明结构与类型，不含执行语义与路由装配。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from pra.domain.models import ReviewDecision

__all__ = ["ReviewRunResult"]


class ReviewRunResult(BaseModel):
    """HTTP 响应包装 —— 一次调查执行的最小结果信封。

    MVP 为同步执行（请求内 await 完整调查后返回），无异步 job 语义；未来 worker
    消费 MQ 后本信封演进为受理回执 + 异步查询/回调地址。
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(description="本次调查运行 ID（= LangGraph thread_id）")
    review_decision: ReviewDecision = Field(description="图终态最终裁决（ReviewDecision 全量快照）")
