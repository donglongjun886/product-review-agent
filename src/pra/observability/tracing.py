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
# trace_id / 实验版本 / 会话（docs/09 §4.1 / §5；纯函数，绝不抛）
# --------------------------------------------------------------------------------------

#: W3C trace id 形状（32 位小写 hex）—— 只有这种 run_id 才原样当 trace_id。
_TRACE_ID_HEX = re.compile(r"\A[0-9a-f]{32}\Z")

#: 未设 ``PRA_LANGFUSE_EXPERIMENT`` 时的缺省实验名。
_DEFAULT_EXPERIMENT = "baseline"


def trace_id_from_run_id(run_id: str) -> str:
    """把 ``run_id`` 映射为 32-hex ``trace_id``（**确定性**，绝不抛；docs/09 §4.1）。

    - ``run_id`` 已是 32 位小写 hex（HTTP 路径的 ``uuid4().hex``）→ **原样返回**：
      Langfuse trace 与 MySQL ``review_run.run_id`` 一一对应，可直接反查审计链；
    - 其余形态（demo 脚本 ``RUN_CASE_1``、评测 ``eval-agent-EC_0123`` …）→
      ``uuid5(NAMESPACE_URL, run_id).hex``：同一 run_id **恒同** trace_id（可重放）；
    - 非法输入（``None`` / 空串 / 非字符串）→ 同样走 uuid5 兜底（观测旁路，不抛）。
    """
    text = run_id if isinstance(run_id, str) else str(run_id or "")
    if _TRACE_ID_HEX.match(text):
        return text
    return uuid5(NAMESPACE_URL, text).hex


def _env_or_settings(env_key: str, settings_attr: str) -> str | None:
    """取配置：**真实环境变量优先**，其次 ``pra.infra.db.Settings``（它读仓库根 `.env`）。

    为什么必须有第二来源：pydantic-settings 读 `.env` 时**不会**写进 ``os.environ``，
    而本模块只读 ``os.environ`` —— 若凭据只写在 `.env`（本项目推荐做法），适配层将
    **永远看不到**、静默走 NullTracer（"配了 key 却没有 trace"，且不报错，极难排查）。

    ``Settings`` 的优先级语义本身就是「环境变量 > .env」，因此两条路径都生效且一致。
    任何异常（Settings 未装/校验失败/属性缺失）→ 视为无配置，绝不抛。
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
    """实验版本名（``PRA_LANGFUSE_EXPERIMENT``，缺省 ``baseline``；docs/09 §5）。

    同时作为 root trace 的 ``version`` 与 tag ``experiment:<name>``，用于多实验对比。
    """
    return (_env_or_settings("PRA_LANGFUSE_EXPERIMENT", "pra_langfuse_experiment") or "").strip() or _DEFAULT_EXPERIMENT


def session_id() -> str | None:
    """一次 evaluation run 的会话 ID（``PRA_LANGFUSE_SESSION``，缺省 ``None``）。

    评测入口（S5 CLI）为整轮评测设同一个值 → 320 条 trace 聚成一个 session，
    UI 按 session 过滤即"这一轮评测的全部案件"；HTTP 路径留空（None）。
    """
    return (_env_or_settings("PRA_LANGFUSE_SESSION", "pra_langfuse_session") or "").strip() or None


def flush_tracer() -> None:
    """刷出 tracer 缓冲（CLI / 短生命周期进程退出前调用）；NullTracer 下 no-op，绝不抛。

    **签名与名字已冻结**（S5b 评测 CLI 直接 ``from pra.observability import flush_tracer``，
    在整轮评测结束后调一次）—— 实现只取当前单例的 ``flush()`` 并吞异常：不新建 client、
    不改单例。**不要 per-case 调用**（320 次 flush 太慢）。
    """
    try:
        get_tracer().flush()
    except Exception:  # noqa: BLE001 - 观测旁路：flush 失败不得影响业务
        return


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

    显式参数优先，其次「真实环境变量 > ``pra.infra.db.Settings``（即仓库根 `.env`）」
    （``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` / ``LANGFUSE_HOST`` /
    ``PRA_LANGFUSE_ENABLED`` / ``PRA_LANGFUSE_SAMPLE``）。
    **真 SDK 只在 `langfuse_backend` 内惰性 import** —— 未启用路径不会拉起 SDK。
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

    # 惰性 import：仅真正启用时才拉起 SDK（见 langfuse_backend 模块 docstring）
    from pra.observability.langfuse_backend import build_langfuse_tracer

    return build_langfuse_tracer(
        public_key=public_key, secret_key=secret_key, host=host, sample=sample
    )


def null_observation() -> AbstractContextManager[Observation]:
    """空观测上下文（供不想分支调用的场景）。"""
    return nullcontext(_NullObservation())
