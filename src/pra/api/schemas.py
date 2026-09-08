"""API DTO 层 —— HTTP 接入面的请求/响应数据契约（docs/00-system-design.md §15 api 目录；
总链路 A·1「HTTP 接入」节点）。

职责与范围：
- **请求体直接复用 domain 的 ``ProductReviewCase``**，本模块**不派生**同构请求模型 ——
  理由：``ProductReviewCase`` 已是 Pydantic v2 + ``extra="forbid"`` 的契约 DTO
  （字段描述即 OpenAPI 文档、拒收未声明字段，见 src/pra/domain/models.py 模块 docstring），
  与领域层共享**单一事实源**；另建逐字段复制的请求模型只会制造同步漂移点
  （领域加字段而接入层漏同步）。故请求契约的"定义"落在 domain，此处只引用其类型。
- 本层真正新增的是**响应包装** ``ReviewRunResult``：把"执行上下文（run_id，即 LangGraph
  thread_id，O-6）"与"图产出的最终裁决（ReviewDecision）"打包为对前端/人工工作台的
  最小输出契约，对应总链路 A·1 的返回语义。``extra="forbid"`` 继承 domain 的契约精神
  （拒收未声明字段），保证响应不随实现悄悄多出字段。

演进路径（本层何时需要派生请求模型）：当传输层出现"领域字段之外"的关注点时 —— 如
traceId 注入、鉴权主体、幂等键、版本协商、分页 —— 在此层新增请求模型做显式映射，
把传输字段与领域字段解耦；届时 ``ReviewRunResult`` 也可按需追加运行元数据（耗时、
预算占用）而**不动** domain 模型。MVP 阶段不引入，避免空架子。

本文件只声明结构与类型，不含执行语义（在 service.py）与路由装配（在 routes.py / app.py）。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from pra.domain.models import ReviewDecision

__all__ = ["ReviewRunResult"]


class ReviewRunResult(BaseModel):
    """HTTP 响应包装（A·1 输出契约）—— 一次调查执行的最小结果信封。

    字段：
    - ``run_id``：本次调查运行的唯一标识（LangGraph 线程维度 thread_id=run_id，O-6）；
      由调用方显式传入或 service 自动生成（uuid4 hex）。
    - ``review_decision``：图终态产出的最终结构化裁决（三分类 + 风险等级/类型 +
      置信度 + 证据链 + 假设轨迹 + 预算快照），即 domain 的 ``ReviewDecision``。

    为何不含 process 状态/队列回执：MVP 为**同步执行**（请求内 await 完整调查后返回），
    无异步 job 语义；未来 worker 消费 MQ 后，本信封演进为"受理回执 + 异步查询/回调
    地址"，届时经同一 ``pra.api.service.run_review`` 执行器产出并落库后由查询接口返回。
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(description="本次调查运行 ID（= LangGraph thread_id，O-6）")
    review_decision: ReviewDecision = Field(description="图终态最终裁决（ReviewDecision 全量快照）")
