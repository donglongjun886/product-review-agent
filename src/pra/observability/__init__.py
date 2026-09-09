"""可观测性适配层（`pra.observability`）—— 与具体 tracing SDK 解耦的薄接口。

设计要点（docs/09-langfuse-observability.md）：

- **只解决研发观测**：Agent 执行链路 / LLM 调用 / Tool 调用 / latency / token / 实验版本；
  业务审计与结果溯源仍由 MySQL `review_trace` 负责，二者职责分离，互不替代。
- **Null Object 优先**：未配置 Langfuse 凭据（或显式关闭）时返回 `NullTracer`，
  全部方法为 no-op —— 不联网、不 import SDK、不改 AgentState、不改路由与决策，
  既有测试与 CI 无需任何 key。
- **旁路**：埋点只读状态、只写观测，任何异常都被吞掉（观测不得影响业务）。
"""

from __future__ import annotations

from pra.observability.tracing import (
    NullTracer,
    Observation,
    TraceContext,
    Tracer,
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
    "get_tracer",
    "make_tracer",
    "set_tracer",
    "should_sample",
]
