"""Langfuse SDK 的真实接线（**唯一 import SDK 的模块**）。

业务侧只依赖 ``pra.observability.tracing`` 的薄接口；本模块负责惰性 import（``langfuse`` 属
optional 依赖 ``observability``，未安装 → ``NullTracer``）、``TraceContext`` → Langfuse v4
调用映射、确定性采样，以及**吞掉一切观测异常**。

API 事实（本机 spike 实测，langfuse==4.15.1）：入口是 **client 实例**
``get_client().start_as_current_observation(...)``（**模块级**同名函数不存在，网上 v3 示例会
误导）；签名含 ``trace_context`` / ``as_type`` / ``name`` / ``input`` / ``output`` /
``metadata`` / ``version`` / ``level`` / ``status_message`` / ``model`` /
``model_parameters`` / ``usage_details`` / ``cost_details`` / ``prompt``。
``trace_context={"trace_id": <32-hex>}`` 实测生效 → 可与 MySQL ``review_run.run_id`` 硬对齐；
子观测靠 OTel contextvar 自动嵌套，``await`` / ``asyncio.create_task`` / ``asyncio.gather``
三种调度均继承同一 trace；trace 级属性（session_id / metadata / tags / version）经
``propagate_attributes(...)`` 下发（v4 中 trace 级 input/output 已废弃，整体 input/output 放
root observation）；``as_type="tool"`` 受支持。

注意：服务端不可达时 SDK 后台会打印 ``Transient error ...`` 重试日志 —— 这是**启用后**的预期
行为；未启用路径不构造 client，不产生噪音。
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Any, Iterator

from pra.observability.tracing import (
    NullTracer,
    Observation,
    TraceContext,
    Tracer,
    should_sample,
)

__all__ = ["LangfuseTracer", "build_langfuse_tracer"]

#: 当前 trace 是否被采样抑制（True = 本次 root 未采样，子观测全部 no-op）
_SUPPRESSED: ContextVar[bool] = ContextVar("pra_langfuse_suppressed", default=False)


class _LfObservation:
    """Langfuse 观测节点包装：把 ``update`` 收敛为一次 SDK 调用并吞异常。"""

    __slots__ = ("_obs",)

    def __init__(self, obs: Any) -> None:
        self._obs = obs

    def update(self, **kwargs: Any) -> None:
        """更新观测字段（output / usage_details / metadata / model …）；失败静默。"""
        if self._obs is None:
            return None
        try:
            self._obs.update(**kwargs)
        except Exception:  # noqa: BLE001 - 观测旁路：任何异常都不得影响业务
            return None
        return None

    def record_error(self, exc: BaseException) -> None:
        """记录失败（level=ERROR + status_message）；失败静默。"""
        if self._obs is None:
            return None
        try:
            self._obs.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
        except Exception:  # noqa: BLE001
            return None
        return None


class LangfuseTracer:
    """Langfuse v4 实现（由 `build_langfuse_tracer` 构造，业务侧只依赖 `Tracer` 协议）。"""

    enabled = True

    def __init__(self, *, client: Any, propagate_attributes: Any, sample: float = 1.0) -> None:
        self._client = client
        self._propagate = propagate_attributes
        self._sample = sample

    # -- 内部工具 ---------------------------------------------------------------------

    def _start(self, *, as_type: str, name: str, **kwargs: Any) -> Any:
        """打开一个观测（失败返回 None → 包装成 no-op 节点）。"""
        try:
            return self._client.start_as_current_observation(
                as_type=as_type, name=name, **kwargs
            )
        except Exception:  # noqa: BLE001 - 观测旁路
            return None

    # -- 内部：安全进入/退出 SDK 观测 ------------------------------------------------

    @contextlib.contextmanager
    def _guarded(self, *cms: Any) -> Iterator[Observation]:
        """按序进入若干 SDK 上下文管理器，并保证**业务异常原样传播**。

        **关键**：`@contextmanager` 生成器里**绝不能**用 ``try: yield ... except Exception:
        yield ...`` 吞业务异常 —— 业务体抛异常时 Python 走 ``gen.throw(exc)``，若生成器又执行到
        ``yield``，会抛 ``RuntimeError: generator didn't stop after throw()``，**原始异常被
        替换**（违反 tracing.py 不变式 2）。正确做法：进入失败 → 本次观测 no-op；业务体异常 →
        ``finally`` 只负责关闭栈且关闭异常被 suppress，不掩盖原异常。
        """
        stack = contextlib.ExitStack()
        obs: Any = None
        for cm in cms:
            if cm is None:
                continue
            try:
                entered = stack.enter_context(cm)
            except Exception:  # noqa: BLE001 - 观测旁路：SDK 进入失败不改业务语义
                with contextlib.suppress(Exception):
                    stack.close()
                yield _LfObservation(None)
                return
            # 只取**第一个**成功进入的上下文对象作为观测节点（``propagate_attributes`` 只提供
            # trace 级属性副作用，其 __enter__ 返回 None，不能覆盖观测节点）。
            if obs is None:
                obs = entered
        try:
            yield _LfObservation(obs)
        finally:
            with contextlib.suppress(Exception):
                stack.close()

    def _propagate_cm(self, attrs: dict[str, Any]) -> Any:
        """构造 ``propagate_attributes`` 上下文管理器（构造失败 → None，观测降级）。"""
        try:
            return self._propagate(**attrs)
        except Exception:  # noqa: BLE001 - 观测旁路
            return None

    # -- Tracer 协议 -------------------------------------------------------------------

    @contextlib.contextmanager
    def trace_root(self, ctx: TraceContext) -> Iterator[Observation]:
        """开 root observation（显式 trace_id + trace 级属性下发）。

        未采样 → 整棵子树 no-op（置 contextvar 抑制，子 span 不再创建）。业务体异常**原样
        抛出**（见 `_guarded`）。
        """
        if not should_sample(ctx.trace_id, self._sample):
            token = _SUPPRESSED.set(True)
            try:
                yield _LfObservation(None)
            finally:
                _SUPPRESSED.reset(token)
            return

        attrs: dict[str, Any] = {}
        if ctx.session_id:
            attrs["session_id"] = ctx.session_id
        if ctx.metadata:
            attrs["metadata"] = ctx.metadata
        if ctx.tags:
            attrs["tags"] = ctx.tags
        if ctx.version:
            attrs["version"] = ctx.version

        cms: list[Any] = [
            self._start(
                as_type="span",
                name=ctx.name,
                trace_context={"trace_id": ctx.trace_id},
                input=ctx.input,
            )
        ]
        if attrs:
            cms.append(self._propagate_cm(attrs))
        with self._guarded(*cms) as obs:
            yield obs

    @contextlib.contextmanager
    def node_span(
        self, name: str, *, input: Any = None, metadata: dict[str, Any] | None = None
    ) -> Iterator[Observation]:
        """LangGraph 节点 span（hypothesize / plan / tools / reevaluate / decide）。"""
        if _SUPPRESSED.get():
            yield _LfObservation(None)
            return
        kwargs: dict[str, Any] = {}
        if input is not None:
            kwargs["input"] = input
        if metadata:
            kwargs["metadata"] = metadata
        with self._guarded(self._start(as_type="span", name=name, **kwargs)) as obs:
            yield obs

    @contextlib.contextmanager
    def llm_generation(
        self,
        *,
        name: str,
        model: str,
        input: Any,
        model_parameters: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Observation]:
        """一次**真实** ``backend.complete()`` 的 generation（含重试的每一次）。"""
        if _SUPPRESSED.get():
            yield _LfObservation(None)
            return
        kwargs: dict[str, Any] = {"model": model, "input": input}
        if model_parameters:
            kwargs["model_parameters"] = model_parameters
        if metadata:
            kwargs["metadata"] = metadata
        with self._guarded(
            self._start(as_type="generation", name=name, **kwargs)
        ) as obs:
            yield obs

    @contextlib.contextmanager
    def tool_span(
        self,
        *,
        name: str,
        input: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Observation]:
        """一次工具调用（``await tool.call(...)``）。"""
        if _SUPPRESSED.get():
            yield _LfObservation(None)
            return
        kwargs: dict[str, Any] = {}
        if input is not None:
            kwargs["input"] = input
        if metadata:
            kwargs["metadata"] = metadata
        with self._guarded(self._start(as_type="tool", name=name, **kwargs)) as obs:
            yield obs

    def flush(self) -> None:
        """刷出缓冲（CLI 脚本/短生命周期进程退出前必须调用）；失败静默。"""
        try:
            self._client.flush()
        except Exception:  # noqa: BLE001
            return None
        return None


def build_langfuse_tracer(
    *, public_key: str, secret_key: str, host: str, sample: float = 1.0
) -> Tracer:
    """构造 Langfuse tracer；SDK 缺失或构造失败 → `NullTracer`（绝不抛）。"""
    try:
        from langfuse import Langfuse, propagate_attributes
    except Exception:  # noqa: BLE001 - optional 依赖未安装
        return NullTracer(reason="langfuse SDK not installed (uv sync --extra observability)")

    try:
        client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
    except Exception:  # noqa: BLE001 - 构造失败（配置非法等）
        return NullTracer(reason="Langfuse client init failed")

    return LangfuseTracer(client=client, propagate_attributes=propagate_attributes, sample=sample)
