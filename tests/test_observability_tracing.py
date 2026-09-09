"""`pra.observability` 适配层测试 —— Null Object / 确定性采样 / SDK 接线（无网络、无 SDK）。

覆盖（docs/09 §3 的"不变式"）：

1. 无凭据 → `NullTracer`，且**不 import** `langfuse`；
2. `PRA_LANGFUSE_ENABLED=0` → 即使有凭据也 no-op；
3. `NullTracer` 全部方法可用且无副作用（含异常路径）；
4. 采样判定确定性（同 key 同结果）与边界（0 / 1）；
5. `LangfuseTracer` 用**假 client** 验证：trace_context 传参、as_type、update/record_error、
   采样抑制、异常吞掉（观测失败不影响业务）；
6. 进程级单例 `get_tracer` / `set_tracer`。
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from pra.observability import tracing as T
from pra.observability.langfuse_backend import LangfuseTracer, build_langfuse_tracer


# --------------------------------------------------------------------------------------
# 1. 默认路径：无凭据 → NullTracer 且不拉起 SDK
# --------------------------------------------------------------------------------------


def test_no_credentials_returns_null_tracer_without_importing_sdk(monkeypatch) -> None:
    """无凭据 → NullTracer；且 `langfuse` 模块不被 import（默认路径零 SDK 依赖）。"""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("PRA_LANGFUSE_ENABLED", raising=False)
    monkeypatch.delitem(sys.modules, "langfuse", raising=False)

    tracer = T.make_tracer()

    assert isinstance(tracer, T.NullTracer)
    assert tracer.enabled is False
    assert "missing" in tracer.reason
    assert "langfuse" not in sys.modules  # 惰性 import：未启用不拉 SDK


def test_enabled_flag_off_disables_even_with_credentials(monkeypatch) -> None:
    """`PRA_LANGFUSE_ENABLED=0` → 即便凭据齐全也 no-op（一键关闭观测）。"""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-x")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-x")
    monkeypatch.setenv("PRA_LANGFUSE_ENABLED", "0")

    tracer = T.make_tracer()

    assert isinstance(tracer, T.NullTracer)
    assert "ENABLED" in tracer.reason


# --------------------------------------------------------------------------------------
# 2. NullTracer 行为：全 no-op、不抛、可用于四个埋点
# --------------------------------------------------------------------------------------


def test_null_tracer_is_noop_for_all_observation_types() -> None:
    """NullTracer 的四类埋点均可用且无副作用（业务侧无需分支）。"""
    tracer = T.NullTracer("test")
    ctx = T.TraceContext(
        trace_id="0af7651916cd43dd8448eb211c80319c",
        session_id="sess",
        metadata={"case_id": "C1"},
        tags=["env:local"],
        input={"case_id": "C1"},
    )

    with tracer.trace_root(ctx) as root:
        root.update(output={"decision": "PASS"})
        with tracer.node_span("hypothesize", input={"n": 1}) as ns:
            ns.update(output={"hypotheses": []})
            with tracer.llm_generation(
                name="llm.hypothesize", model="scripted", input=[{"role": "user", "content": "x"}]
            ) as gen:
                gen.update(output="{}", usage_details={"total": 0})
            with tracer.tool_span(name="ProductTool", input={"args": {}}) as ts:
                ts.update(output={"ok": True})
                ts.record_error(RuntimeError("boom"))  # no-op，不抛
    tracer.flush()


def test_null_observation_swallows_errors() -> None:
    """`NullTracer` 的 `record_error` 不抛（观测旁路）。"""
    tracer = T.NullTracer()
    with tracer.node_span("x") as obs:
        obs.record_error(ValueError("ignored"))


# --------------------------------------------------------------------------------------
# 3. 确定性采样
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("sample,expected", [(0.0, False), (-1.0, False), (1.0, True), (2.0, True)])
def test_should_sample_boundaries(sample: float, expected: bool) -> None:
    """采样边界：<=0 恒关、>=1 恒开。"""
    assert T.should_sample("trace-1", sample) is expected


def test_should_sample_is_deterministic() -> None:
    """同 key 同采样率恒同结果（评测可重放的前提）。"""
    keys = [f"trace-{i}" for i in range(200)]
    first = [T.should_sample(k, 0.3) for k in keys]
    second = [T.should_sample(k, 0.3) for k in keys]

    assert first == second
    # 0.3 采样率下比例应落在合理区间（避免"恒真/恒假"退化）
    ratio = sum(first) / len(first)
    assert 0.15 < ratio < 0.5, ratio


# --------------------------------------------------------------------------------------
# 4. LangfuseTracer（假 client，验证接线正确性）
# --------------------------------------------------------------------------------------


class _FakeObs:
    """假 observation：记录 update 调用。"""

    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)


class _FakeCM:
    """假上下文管理器（`start_as_current_observation` 的返回）。"""

    def __init__(self, obs: _FakeObs) -> None:
        self.obs = obs

    def __enter__(self) -> _FakeObs:
        return self.obs

    def __exit__(self, *exc: Any) -> bool:
        return False


class _FakePropagate:
    """假 `propagate_attributes`：记录属性并在退出时归还。"""

    def __init__(self, sink: list[dict[str, Any]]) -> None:
        self._sink = sink

    def __call__(self, **kwargs: Any) -> Any:
        self._sink.append(kwargs)

        class _CM:
            def __enter__(self) -> None:
                return None

            def __exit__(self, *exc: Any) -> bool:
                return False

        return _CM()


class _FakeClient:
    """假 Langfuse client：记录每次观测创建参数。"""

    def __init__(self, *, raise_on_start: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.observations: list[_FakeObs] = []
        self.propagated: list[dict[str, Any]] = []
        self.flushed = 0
        self._raise = raise_on_start

    def start_as_current_observation(self, **kwargs: Any) -> _FakeCM:
        if self._raise:
            raise RuntimeError("sdk boom")
        self.calls.append(kwargs)
        obs = _FakeObs()
        self.observations.append(obs)
        return _FakeCM(obs)

    def flush(self) -> None:
        self.flushed += 1


def _tracer(client: _FakeClient, sample: float = 1.0) -> LangfuseTracer:
    return LangfuseTracer(
        client=client, propagate_attributes=_FakePropagate(client.propagated), sample=sample
    )


def test_langfuse_tracer_maps_root_and_propagates_attributes() -> None:
    """root span：显式 trace_id 经 trace_context 传入 + trace 级属性下发。"""
    client = _FakeClient()
    ctx = T.TraceContext(
        trace_id="0af7651916cd43dd8448eb211c80319c",
        name="review",
        session_id="eval-run-1",
        version="baseline",
        metadata={"case_id": "C1", "source": "evaluation"},
        tags=["source:evaluation"],
        input={"case_id": "C1"},
    )

    with _tracer(client).trace_root(ctx) as root:
        root.update(output={"decision": "HUMAN_REVIEW"})

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["as_type"] == "span"
    assert call["name"] == "review"
    assert call["trace_context"] == {"trace_id": ctx.trace_id}
    assert call["input"] == {"case_id": "C1"}
    assert client.propagated == [
        {
            "session_id": "eval-run-1",
            "metadata": {"case_id": "C1", "source": "evaluation"},
            "tags": ["source:evaluation"],
            "version": "baseline",
        }
    ]
    assert client.observations[0].updates == [{"output": {"decision": "HUMAN_REVIEW"}}]


def test_langfuse_tracer_observation_types() -> None:
    """节点=span / LLM=generation（带 model+input）/ 工具=tool。"""
    client = _FakeClient()
    tracer = _tracer(client)
    with tracer.trace_root(T.TraceContext(trace_id="a" * 32)):
        with tracer.node_span("plan", input={"n": 1}) as ns:
            ns.update(output={"pending": []})
        with tracer.llm_generation(
            name="llm.plan",
            model="scripted",
            input=[{"role": "system", "content": "s"}],
            model_parameters={"temperature": 0.0},
            metadata={"node": "plan"},
        ) as gen:
            gen.update(output="{}", usage_details={"input": 0, "output": 0, "total": 0})
        with tracer.tool_span(name="ProductTool", input={"args": {"product_id": "P1"}}) as ts:
            ts.update(output={"ok": True})

    kinds = [c["as_type"] for c in client.calls]
    assert kinds == ["span", "span", "generation", "tool"]
    gen_call = client.calls[2]
    assert gen_call["model"] == "scripted"
    assert gen_call["input"] == [{"role": "system", "content": "s"}]
    assert gen_call["model_parameters"] == {"temperature": 0.0}
    assert client.observations[2].updates[0]["usage_details"] == {
        "input": 0,
        "output": 0,
        "total": 0,
    }


def test_langfuse_tracer_records_error_as_error_level() -> None:
    """`record_error` → level=ERROR + status_message（不改变业务异常传播）。"""
    client = _FakeClient()
    tracer = _tracer(client)
    with tracer.trace_root(T.TraceContext(trace_id="b" * 32)):
        with tracer.llm_generation(name="llm.decide", model="scripted", input="x") as gen:
            gen.record_error(TimeoutError("llm timeout"))

    assert client.observations[1].updates == [
        {"level": "ERROR", "status_message": "TimeoutError: llm timeout"}
    ]


def test_langfuse_tracer_sampling_suppresses_whole_subtree() -> None:
    """sample=0 → 整棵子树 no-op（连 client 都不调用），保证"不采样即零开销"。"""
    client = _FakeClient()
    tracer = _tracer(client, sample=0.0)

    with tracer.trace_root(T.TraceContext(trace_id="c" * 32)) as root:
        root.update(output="ignored")
        with tracer.node_span("plan") as ns:
            ns.update(output="ignored")
        with tracer.llm_generation(name="llm.plan", model="m", input="i") as gen:
            gen.update(output="ignored")

    assert client.calls == []
    assert client.propagated == []


def test_langfuse_tracer_never_raises_when_sdk_fails() -> None:
    """SDK 内部异常被吞掉：观测失败不得影响业务（四种埋点都要覆盖）。"""
    client = _FakeClient(raise_on_start=True)
    tracer = _tracer(client)

    with tracer.trace_root(T.TraceContext(trace_id="d" * 32)) as root:
        root.update(output="ignored")
        with tracer.node_span("plan") as ns:
            ns.update(output="ignored")
        with tracer.llm_generation(name="llm.plan", model="m", input="i") as gen:
            gen.record_error(RuntimeError("x"))
        with tracer.tool_span(name="ProductTool") as ts:
            ts.update(output="ignored")
    tracer.flush()

    assert client.flushed == 1


def test_build_langfuse_tracer_without_sdk_returns_null_tracer() -> None:
    """未安装 SDK（本项目默认）→ NullTracer，且给出安装提示。"""
    tracer = build_langfuse_tracer(
        public_key="pk", secret_key="sk", host="http://localhost:3000"
    )

    assert isinstance(tracer, T.NullTracer)
    assert "SDK" in tracer.reason or "not installed" in tracer.reason


# --------------------------------------------------------------------------------------
# 5. 进程级单例
# --------------------------------------------------------------------------------------


def test_get_tracer_caches_and_set_tracer_resets(monkeypatch) -> None:
    """`get_tracer` 懒装配并缓存；`set_tracer` 可注入/重置（测试与显式装配用）。"""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    T.set_tracer(None)

    first = T.get_tracer()
    assert first is T.get_tracer()  # 缓存同一实例
    assert isinstance(first, T.NullTracer)

    fake = T.NullTracer("injected")
    T.set_tracer(fake)
    assert T.get_tracer() is fake

    T.set_tracer(None)  # 复原，避免影响其他测试
