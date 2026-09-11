"""可观测性薄接口 + Null Object（与具体 tracing SDK 解耦）。

本模块**不 import 任何 tracing SDK**：真实接线在 ``langfuse_backend.py``（惰性 import），
默认路径（无 key / 未装 optional 依赖）零开销、零网络、零日志噪音。

四类观测：``trace_root``（图调用点：case_id / run_id / experiment / tags）、``node_span``
（graph.py 的 add_node 包装）、``llm_generation``（llm_shell.py 内层 backend.complete()，
每次真实调用一条）、``tool_span``（tools_node.py 的 await tool.call()）。不变式（测试守护）：
无凭据 → NullTracer 且不 import langfuse；观测调用绝不抛异常；采样判定确定性（同 key 同结果，
评测可重放）；埋点不写 AgentState、不参与路由。
"""

from __future__ import annotations

import hashlib
import os
import re
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, runtime_checkable
from uuid import NAMESPACE_URL, uuid5

__all__ = [
    "NullTracer",
    "Observation",
    "TraceContext",
    "Tracer",
    "experiment_name",
    "flush_tracer",
    "get_tracer",
    "make_tracer",
    "session_id",
    "set_tracer",
    "should_sample",
    "trace_id_from_run_id",
]


@dataclass
class TraceContext:
    """一次 root trace 的关联信息（trace_id 与业务主键对齐）。

    ``trace_id`` 为 W3C 32-hex：HTTP/落库路径 = ``run_id``（与 MySQL review_run.run_id 一一
    对应）；评测路径 = uuid5(experiment:case:scheme)，确定性可重放。``input`` 放 root
    observation 上（Langfuse v4 中 trace 级 input 已废弃）。
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
    """no-op 观测节点（``NullTracer`` 用）。"""

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


# ---- 采样（确定性）----


def should_sample(key: str, sample: float) -> bool:
    """确定性采样：同 ``key`` 同 ``sample`` 恒同结果（评测可重放）。

    ``sample <= 0`` → 恒 False；``sample >= 1`` → 恒 True；否则按 sha256(key) 均匀落桶。
    """
    if sample <= 0.0:
        return False
    if sample >= 1.0:
        return True
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
    return (int(digest, 16) / 0xFFFFFFFF) < sample


# ---- trace_id / 实验版本 / 会话（纯函数，绝不抛）----

#: W3C trace id 形状（32 位小写 hex）—— 只有这种 run_id 才原样当 trace_id。
_TRACE_ID_HEX = re.compile(r"\A[0-9a-f]{32}\Z")

#: 未设 ``PRA_LANGFUSE_EXPERIMENT`` 时的缺省实验名。
_DEFAULT_EXPERIMENT = "baseline"


def trace_id_from_run_id(run_id: str) -> str:
    """把 ``run_id`` 映射为 32-hex ``trace_id``（**确定性**，绝不抛）。

    - 已是 32 位小写 hex（HTTP 路径的 ``uuid4().hex``）→ **原样返回**：Langfuse trace 与
      MySQL review_run.run_id 一一对应，可直接反查审计链；
    - 其余形态（``RUN_CASE_1``、``eval-agent-EC_0123`` …）→ ``uuid5(NAMESPACE_URL, run_id)``：
      同一 run_id **恒同** trace_id（可重放）；
    - 非法输入（``None`` / 空串 / 非字符串）→ 同样走 uuid5 兜底（观测旁路，不抛）。
    """
    text = run_id if isinstance(run_id, str) else str(run_id or "")
    if _TRACE_ID_HEX.match(text):
        return text
    return uuid5(NAMESPACE_URL, text).hex


def _env_or_settings(env_key: str, settings_attr: str) -> str | None:
    """取配置：**真实环境变量优先**，其次 ``pra.infra.db.Settings``（它读仓库根 `.env`）。

    为什么需要第二来源：pydantic-settings 读 `.env` 时**不会**写进 ``os.environ``，而本模块
    只读 ``os.environ`` —— 凭据只写在 `.env`（本项目推荐做法）时适配层会**永远看不到**，静默
    走 NullTracer。``Settings`` 本身的优先级语义即「环境变量 > .env」，两条路径一致。任何异常
    （Settings 未装/校验失败/属性缺失）→ 视为无配置，绝不抛。
    """
    value = os.environ.get(env_key)
    if value:
        return value
    try:
        from pra.infra.db import Settings

        got = getattr(Settings(), settings_attr, None)
    except Exception:  # noqa: BLE001 - 观测旁路：配置来源失败不得抛
        return None
    return got if isinstance(got, str) and got else None


def experiment_name() -> str:
    """实验版本名（``PRA_LANGFUSE_EXPERIMENT``，缺省 ``baseline``）。

    同时作为 root trace 的 ``version`` 与 tag ``experiment:<name>``，用于多实验对比。
    """
    return (_env_or_settings("PRA_LANGFUSE_EXPERIMENT", "pra_langfuse_experiment") or "").strip() or _DEFAULT_EXPERIMENT


def session_id() -> str | None:
    """一次 evaluation run 的会话 ID（``PRA_LANGFUSE_SESSION``，缺省 ``None``）。

    评测入口为整轮评测设同一个值 → 全部 trace 聚成一个 session，UI 按 session 过滤即「这一轮
    评测的全部案件」；HTTP 路径留空。
    """
    return (_env_or_settings("PRA_LANGFUSE_SESSION", "pra_langfuse_session") or "").strip() or None


def flush_tracer() -> None:
    """刷出 tracer 缓冲（CLI / 短生命周期进程退出前调一次）；NullTracer 下 no-op，绝不抛。

    **签名与名字已冻结**（评测 CLI 直接 ``from pra.observability import flush_tracer``）。
    **不要 per-case 调用**（数百次 flush 太慢）。
    """
    try:
        get_tracer().flush()
    except Exception:  # noqa: BLE001 - 观测旁路：flush 失败不得影响业务
        return


# ---- 进程级单例（默认从环境变量装配）----

_tracer: Tracer | None = None


def set_tracer(tracer: Tracer | None) -> None:
    """注入/重置进程级 tracer（测试与显式装配用；``None`` → 下次 ``get_tracer`` 重建）。"""
    global _tracer
    _tracer = tracer


def get_tracer() -> Tracer:
    """取当前生效 tracer（懒装配；无凭据 → ``NullTracer``）。"""
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
    """按配置装配 tracer：凭据齐全且启用 → Langfuse；否则 ``NullTracer``。

    显式参数优先，其次「真实环境变量 > ``pra.infra.db.Settings``（仓库根 `.env`）」。开关
    语义：**未设 ``PRA_LANGFUSE_ENABLED`` = 启用**，但缺凭据回落 ``NullTracer``（实际 no-op）；
    只有 ``PRA_LANGFUSE_ENABLED=0``（或 false/no/off）才是显式强制关闭。**真 SDK 只在
    ``langfuse_backend`` 内惰性 import** —— 未启用路径不会拉起 SDK。
    """
    public_key = public_key if public_key is not None else _env_or_settings(
        "LANGFUSE_PUBLIC_KEY", "langfuse_public_key"
    )
    secret_key = secret_key if secret_key is not None else _env_or_settings(
        "LANGFUSE_SECRET_KEY", "langfuse_secret_key"
    )
    host = host or _env_or_settings("LANGFUSE_HOST", "langfuse_host") or "http://localhost:3000"
    if enabled is None:
        raw = (_env_or_settings("PRA_LANGFUSE_ENABLED", "pra_langfuse_enabled") or "").strip().lower()
        enabled = raw not in {"0", "false", "no", "off"}
    if sample is None:
        raw_sample = _env_or_settings("PRA_LANGFUSE_SAMPLE", "pra_langfuse_sample")
        try:
            sample = float(raw_sample) if raw_sample else 1.0
        except (TypeError, ValueError):
            sample = 1.0

    if not enabled:
        return NullTracer(reason="PRA_LANGFUSE_ENABLED=0")
    if not public_key or not secret_key:
        return NullTracer(reason="missing LANGFUSE_PUBLIC_KEY/SECRET_KEY")

    # 惰性 import：仅真正启用时才拉起 SDK
    from pra.observability.langfuse_backend import build_langfuse_tracer

    return build_langfuse_tracer(
        public_key=public_key, secret_key=secret_key, host=host, sample=sample
    )


def null_observation() -> AbstractContextManager[Observation]:
    """空观测上下文（供不想分支调用的场景）。"""
    return nullcontext(_NullObservation())
