"""可观测性适配层的**不变式**测试 —— 只留三条。

观测是旁路，它的行为正确性不值得逐调用点铺开测试；这里只钉住两条「出错会伤到业务或度量」
的契约（其余用例已于审计后删除）：

1. 无凭据 → ``NullTracer``，且**不 import** ``langfuse``（optional extra 未装 / CI 不装
   extra 时全量仍可跑）；
2. 观测上下文**绝不吞业务异常**（历史上踩过 ``generator didn't stop after throw()`` 把原始
   异常替换掉的缺陷）—— NullTracer 路径走不到这里，故必须单独守护；
3. 评测 ``trace_id`` 含 LLM 后端名 —— 否则同一 experiment 的不同后端臂落进同一条 trace，
   按 trace 汇总 token 会互相混入（度量失真）。

不联网、不依赖 langfuse SDK：SDK 路径一律用假 client 验证。
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import pytest

from pra.evaluation.dataset.loader import load_dataset
from pra.observability import tracing as T
from pra.observability.langfuse_backend import LangfuseTracer

REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_PATH_V2 = REPO_ROOT / "eval_data" / "v2" / "cases_v2.jsonl"
_EVAL_SCRIPT = REPO_ROOT / "scripts" / "run_evaluation.py"

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


def _load_eval_script():
    """按路径加载跑分脚本（它不是包）—— 取其中的评测 root trace 组装。"""
    spec = importlib.util.spec_from_file_location("run_evaluation_mod", _EVAL_SCRIPT)
    assert spec and spec.loader, f"无法定位脚本: {_EVAL_SCRIPT}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_eval_root_trace_id_is_scoped_by_llm_backend() -> None:
    """trace_id 必须含 LLM 后端名 —— 否则同进程的不同后端臂会落进同一条 trace。

    反证：把 `run_evaluation._root_trace_context` 的 uuid5 key 去掉 `:{backend_name}`
    → 本用例必红（两臂 trace 相同，按 trace 汇总 token 会混入另一臂的 generation）。
    """
    mod = _load_eval_script()
    case = load_dataset(_DATA_PATH_V2)[0]
    experiment = T.experiment_name()

    scripted = mod._root_trace_context(
        case, scheme="agent", backend_name="eval-scripted-reviewer"
    )
    real = mod._root_trace_context(
        case, scheme="agent", backend_name="litellm-deepseek/deepseek-chat"
    )

    assert scripted.trace_id != real.trace_id  # 两臂不混
    for backend, ctx in (
        ("eval-scripted-reviewer", scripted),
        ("litellm-deepseek/deepseek-chat", real),
    ):
        assert ctx.trace_id == uuid5(
            NAMESPACE_URL, f"{experiment}:{case.eval_case_id}:agent:{backend}"
        ).hex, "trace_id 须由 uuid5(experiment:case:scheme:后端名) 确定"
        assert ctx.metadata["llm_backend"] == backend
