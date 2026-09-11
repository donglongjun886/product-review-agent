"""可观测性适配层（``pra.observability``）—— 与具体 tracing SDK 解耦的薄接口。

**只解决研发观测**（执行链路 / LLM 调用 / Tool 调用 / latency / token / 实验版本）；业务审计
与结果溯源仍由 MySQL `review_trace` 负责，二者职责分离。**Null Object 优先**：未配置凭据或
显式关闭时返回 `NullTracer`，全部方法 no-op（不联网、不 import SDK、不改 AgentState/路由/
决策）。**旁路**：埋点只读状态、异常全吞（观测不得影响业务）。
"""

from __future__ import annotations

from pra.observability.tracing import (
    NullTracer,
    Observation,
    TraceContext,
    Tracer,
    flush_tracer,
    get_tracer,
    make_tracer,
    set_tracer,
    should_sample,
)

__all__ = [
    "NullTracer",
    "Observation",
    "TraceContext",
    "Tracer",
    "flush_tracer",
    "get_tracer",
    "make_tracer",
    "set_tracer",
    "should_sample",
]
