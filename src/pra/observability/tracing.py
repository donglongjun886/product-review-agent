"""tracing.py —— 可观测性薄接口 + Null Object（与 SDK 解耦；docs/09）。

本模块**不 import 任何 tracing SDK**：真实接线在 `langfuse_backend.py`（惰性 import）。
这样默认路径（无 key / 未装 optional 依赖）零开销、零网络、零日志噪音。

四类观测（对应 docs/09 §3 埋点位置）：

| 接口 | 埋点位置 | 记录内容 |
|---|---|---|
| ``trace_root`` | 三个图调用点（api/service、infra/persist_service、evaluation/agent_scheme） | case_id / run_id / experiment / tags |
| ``node_span`` | `graph.py` 的 5 个 add_node 包装 | 节点名 + 入参摘要 |
| ``llm_generation`` | `llm_shell.py` 内层 ``backend.complete()``（**每次真实调用一条**） | model / input / output / usage / latency / error |
| ``tool_span`` | `tools_node.py` 的 ``await tool.call()`` | tool 名 / args / output / latency |

**不变式**（测试守护）：

1. 无凭据 → `NullTracer`，且**不 import** `langfuse`（`sys.modules` 无该键）；
2. 所有观测调用**绝不抛异常**（观测失败不得影响业务）；
3. 采样判定**确定性**（同 key 同结果），保证评测可重放；
4. 埋点不写 AgentState / 不参与路由。
"""

from __future__ import annotations

import hashlib
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, runtime_checkable

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


@dataclass
class TraceContext:
    """一次 root trace 的关联信息（trace_id 与业务主键对齐，docs/09 §4）。

    :param trace_id: W3C 32-hex trace id。**HTTP/落库路径 = ``run_id``**（与 MySQL
        ``review_run.run_id`` 一一对应）；**评测路径 = uuid5(experiment:case:scheme)**
        （确定性，重跑落同一条 trace）。
    :param name: root observation 名（默认 ``review``）。
    :param session_id: 会话分组 —— 一次 evaluation run 用同一个，便于 UI 按 session 过滤。
    :param version: 实验版本（``experiment``），如 baseline / prompt-v2。
    :param metadata: 结构化关联字段（case_id / eval_case_id / scene / scheme / source …）。
    :param tags: UI 过滤标签（``env:local`` / ``scheme:agent`` / ``source:evaluation`` …）。
    :param input: root observation 的输入（Langfuse v4 中 trace 级 input 已废弃，
        须放在 root observation 上）。
    """

    trace_id: str
    name: str = "review"
    session_id: str | None = None
    version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    input: Any = None


@runtime_checkable
class Observation(Protocol):
    """一个已打开的观测节点（span / generation / tool）—— 支持中途补充字段。"""

    def update(self, **kwargs: Any) -> None: ...

    def record_error(self, exc: BaseException) -> None: ...


@runtime_checkable
class Tracer(Protocol):
    """可观测性适配层（真实实现见 ``langfuse_backend.LangfuseTracer``）。"""

    enabled: bool

    def trace_root(self, ctx: TraceContext) -> AbstractContextManager[Observation]: ...

    def node_span(
        self, name: str, *, input: Any = None, metadata: dict[str, Any] | None = None
    ) -> AbstractContextManager[Observation]: ...

    def llm_generation(
        self,
        *,
        name: str,
        model: str,
        input: Any,
        model_parameters: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AbstractContextManager[Observation]: ...

    def tool_span(
        self,
        *,
        name: str,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> AbstractContextManager[Observation]: ...

    def flush(self) -> None: ...


class _NullObservation:
    """no-op 观测节点（`NullTracer` 用）。"""

    __slots__ = ()

    def update(self, **kwargs: Any) -> None:  # noqa: D102 - 见类 docstring
        return None

    def record_error(self, exc: BaseException) -> None:  # noqa: D102
        return None


class NullTracer:
    """空实现：全部 no-op（未配置凭据 / 显式关闭 / SDK 未安装时使用）。

    :param reason: 关闭原因（仅用于日志与测试断言，不参与业务）。
    """

    enabled = False

    def __init__(self, reason: str = "disabled") -> None:
        self.reason = reason

    @contextmanager
    def trace_root(self, ctx: TraceContext) -> Iterator[Observation]:
        """no-op root（返回共享的空观测节点）。"""
        yield _NullObservation()

    @contextmanager
    def node_span(
        self, name: str, *, input: Any = None, metadata: dict[str, Any] | None = None
    ) -> Iterator[Observation]:
        """no-op 节点 span。"""
        yield _NullObservation()

    @contextmanager
    def llm_generation(
        self,
        *,
        name: str,
        model: str,
        input: Any,
        model_parameters: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Observation]:
        """no-op LLM generation。"""
        yield _NullObservation()

    @contextmanager
    def tool_span(
        self,
        *,
        name: str,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Observation]:
        """no-op tool span。"""
        yield _NullObservation()

    def flush(self) -> None:
        """no-op（无缓冲）。"""
        return None


# --------------------------------------------------------------------------------------
# 采样（确定性）
# --------------------------------------------------------------------------------------


def should_sample(key: str, sample: float) -> bool:
    """确定性采样判定：同 ``key`` 同 ``sample`` 恒同结果（评测可重放）。

    ``sample <= 0`` → 恒 False；``sample >= 1`` → 恒 True；否则按 sha256(key) 均匀落桶。
    """
    if sample <= 0.0:
        return False
    if sample >= 1.0:
        return True
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
    return (int(digest, 16) / 0xFFFFFFFF) < sample


# --------------------------------------------------------------------------------------
# 进程级单例（默认从环境变量装配）
# --------------------------------------------------------------------------------------

_tracer: Tracer | None = None


def set_tracer(tracer: Tracer | None) -> None:
    """注入/重置进程级 tracer（测试与显式装配用；``None`` → 下次 ``get_tracer`` 重建）。"""
    global _tracer
    _tracer = tracer


def get_tracer() -> Tracer:
    """取当前生效 tracer（懒装配；无凭据 → `NullTracer`）。"""
    global _tracer
    if _tracer is None:
        _tracer = make_tracer()
    return _tracer


def make_tracer(
    *,
    public_key: str | None = None,
    secret_key: str | None = None,
    host: str | None = None,
    enabled: bool | None = None,
    sample: float | None = None,
) -> Tracer:
    """按配置装配 tracer：凭据齐全且启用 → Langfuse；否则 `NullTracer`。

    显式参数优先，其次读环境变量（``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` /
    ``LANGFUSE_HOST`` / ``PRA_LANGFUSE_ENABLED`` / ``PRA_LANGFUSE_SAMPLE``）。
    **真 SDK 只在 `langfuse_backend` 内惰性 import** —— 未启用路径不会拉起 SDK。
    """
    import os

    public_key = public_key if public_key is not None else os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = secret_key if secret_key is not None else os.environ.get("LANGFUSE_SECRET_KEY")
    host = host or os.environ.get("LANGFUSE_HOST") or "http://localhost:3000"
    if enabled is None:
        raw = (os.environ.get("PRA_LANGFUSE_ENABLED") or "").strip().lower()
        enabled = raw not in {"0", "false", "no", "off"}
    if sample is None:
        try:
            sample = float(os.environ.get("PRA_LANGFUSE_SAMPLE") or 1.0)
        except ValueError:
            sample = 1.0

    if not enabled:
        return NullTracer(reason="PRA_LANGFUSE_ENABLED=0")
    if not public_key or not secret_key:
        return NullTracer(reason="missing LANGFUSE_PUBLIC_KEY/SECRET_KEY")

    # 惰性 import：仅真正启用时才拉起 SDK（见 langfuse_backend 模块 docstring）
    from pra.observability.langfuse_backend import build_langfuse_tracer

    return build_langfuse_tracer(
        public_key=public_key, secret_key=secret_key, host=host, sample=sample
    )


def null_observation() -> AbstractContextManager[Observation]:
    """空观测上下文（供不想分支调用的场景）。"""
    return nullcontext(_NullObservation())
