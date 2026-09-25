"""可观测性适配层的**不变式**测试 —— 只留两条。

观测是旁路，它的行为正确性不值得逐调用点铺开测试；这里只钉住两条「出错会伤到业务」的
契约（其余用例已于审计后删除）：

1. 无凭据 → ``NullTracer``，且**不 import** ``langfuse``（optional extra 未装 / CI 不装
   extra 时全量仍可跑）；
2. 观测上下文**绝不吞业务异常**（历史上踩过 ``generator didn't stop after throw()`` 把原始
   异常替换掉的缺陷）—— NullTracer 路径走不到这里，故必须单独守护。

不联网、不依赖 langfuse SDK：SDK 路径一律用假 client 验证。
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import pytest

from pra.observability import tracing as T
from pra.observability.langfuse_backend import LangfuseTracer

#: 子进程脚本：切断 `.env` 配置来源（开发机真配了 Langfuse 也不影响）后跑默认装配，
#: 回报「是否 NullTracer」与「进程里有没有 langfuse」。
_LAZY_IMPORT_CHILD = """
import json, sys
from pra.observability import tracing as T

T._env_or_settings = lambda *a, **k: None
tracer = T.make_tracer()
print("PRA_LAZY_IMPORT_JSON:" + json.dumps({
    "null": isinstance(tracer, T.NullTracer),
    "enabled": tracer.enabled,
    "reason": tracer.reason,
    "sdk_imported": "langfuse" in sys.modules,
}))
"""


def test_no_credentials_returns_null_tracer_without_importing_sdk() -> None:
    """无凭据 → NullTracer，且不 import `langfuse`（optional extra 未装也能跑全量）。

    用**子进程**：同进程里 `sys.modules` 可能已被别的用例污染，进程内查会变成永真/永假的
    空断言（同 `test_rag_default_path_no_extra.py` 的理由）；`del sys.modules[...]` 更不行 ——
    顶层 import 后删条目并不会卸载模块，反而把「惰性 import」这条断言变成永真。
    """
    proc = subprocess.run(
        [sys.executable, "-c", _LAZY_IMPORT_CHILD], capture_output=True, text=True, check=True
    )
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("PRA_LAZY_IMPORT_JSON:"))
    payload = json.loads(line.split(":", 1)[1])

    assert payload["null"] is True
    assert payload["enabled"] is False
    assert "missing" in payload["reason"]
    assert payload["sdk_imported"] is False


class _FakeObs:
    """假观测节点：只记账。"""

    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def record_error(self, exc: BaseException) -> None:
        return None


class _FakeCM:
    def __init__(self, obs: _FakeObs, *, raise_on_exit: bool = False) -> None:
        self._obs = obs
        self._raise_on_exit = raise_on_exit

    def __enter__(self) -> _FakeObs:
        return self._obs

    def __exit__(self, *exc: object) -> bool:
        if self._raise_on_exit:
            raise RuntimeError("sdk exit failed")
        return False


class _FakeClient:
    """假 SDK client：`raise_on_exit` 模拟 flush/export 在退出时失败。"""

    def __init__(self, *, raise_on_exit: bool = False) -> None:
        self._raise_on_exit = raise_on_exit

    def start_as_current_observation(self, **kwargs: Any) -> _FakeCM:
        return _FakeCM(_FakeObs(), raise_on_exit=self._raise_on_exit)

    def flush(self) -> None:
        return None


def _tracer(client: Any) -> LangfuseTracer:
    return LangfuseTracer(client=client, propagate_attributes=lambda **kw: _FakeCM(_FakeObs()), sample=1.0)


def test_observability_never_swallows_business_exception() -> None:
    """观测上下文必须让业务异常原样抛出（四类埋点 + SDK 退出失败都要覆盖）。

    反证：把 `langfuse_backend._guarded` 改回「捕获业务异常后 return」→ 本用例必红。
    """
    tracer = _tracer(_FakeClient())

    with pytest.raises(ValueError, match="boom"), tracer.trace_root(
        T.TraceContext(trace_id="a" * 32)
    ):
        raise ValueError("boom")
    with pytest.raises(KeyError), tracer.node_span("plan"):
        raise KeyError("x")
    with pytest.raises(TimeoutError), tracer.llm_generation(name="llm.plan", model="m", input="i"):
        raise TimeoutError("y")
    with pytest.raises(RuntimeError, match="tool failed"), tracer.tool_span(name="ProductTool"):
        raise RuntimeError("tool failed")

    # SDK 退出失败不得掩盖业务异常
    bad = _tracer(_FakeClient(raise_on_exit=True))
    with pytest.raises(ValueError, match="business"), bad.trace_root(
        T.TraceContext(trace_id="b" * 32)
    ):
        raise ValueError("business")
