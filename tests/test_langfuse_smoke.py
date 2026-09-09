"""tests/test_langfuse_smoke.py —— ``scripts/langfuse_smoke.py`` 的离线单元测试。

**全部不联网、不依赖 langfuse SDK**（当前主 venv 未装，属 optional extra）：

1. ``test_main_disabled_without_credentials``：无凭据 → ``main([])`` 返回 0，stdout 含
   ``tracing disabled`` 与 ``uv sync --extra observability`` 启用提示（CI 友好契约）；
2. ``test_main_rejects_bad_trace_id``：``--trace-id`` 非法 → 退出码 2（且不碰 tracer）；
3. ``test_trace_id_and_url_helpers``：``--trace-id`` 参数解析 + ``_trace_api_url`` /
   ``_ui_url`` 尾斜杠归一；
4. ``test_auth_header_is_basic_base64``：Basic Auth 头 = ``base64(public_key:secret_key)``；
5. ``test_api_get_builds_url_and_authorization``：假 ``urlopen`` 捕获 URL / 认证头 / 超时
   （用例内不发真实 HTTP）；
6. ``test_verify_trace_and_observation_tree``：断言清单在正常 payload 上全通过、在缺字段
   时能指出具体差异；observation 树按 ``parentObservationId`` 缩进。

脚本在 ``scripts/``（非包）→ 与 ``tests/test_evaluation_dataset_v2.py`` 同款 importlib
按路径加载，只取纯函数与 ``main``，不执行模块级副作用。
"""

from __future__ import annotations

import base64
import importlib.util
import json
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Self

import pytest

from pra.observability import tracing

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "langfuse_smoke.py"


def _load_smoke_module():
    """按路径加载自检脚本（scripts/ 非包；importlib 载入供离线测试）。"""
    spec = importlib.util.spec_from_file_location("langfuse_smoke_mod", SMOKE_SCRIPT)
    assert spec and spec.loader, f"无法定位自检脚本: {SMOKE_SCRIPT}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def smoke():
    """每个用例独立加载 + 前后重置进程级 tracer（避免被其他用例污染）。"""
    tracing.set_tracer(None)
    mod = _load_smoke_module()
    try:
        yield mod
    finally:
        tracing.set_tracer(None)


@pytest.fixture()
def no_credentials(monkeypatch: pytest.MonkeyPatch):
    """清空 Langfuse 凭据环境变量（无凭据路径）。"""
    for key in (
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
        "PRA_LANGFUSE_ENABLED",
        "PRA_LANGFUSE_SAMPLE",
        "PRA_LANGFUSE_EXPERIMENT",
        "PRA_LANGFUSE_SESSION",
    ):
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# 1) 无凭据 → 退出码 0 + 启用提示（不联网）
# ---------------------------------------------------------------------------


def test_main_disabled_without_credentials(smoke, no_credentials, capsys) -> None:
    rc = smoke.main([])
    out = capsys.readouterr().out
    assert rc == 0, "无凭据必须退出码 0（CI 友好）"
    assert "tracing disabled" in out
    assert "NullTracer:" in out
    assert "uv sync --extra observability" in out
    assert "LANGFUSE_PUBLIC_KEY" in out


def test_main_disabled_with_explicit_disable_flag(smoke, no_credentials, monkeypatch, capsys):
    """``PRA_LANGFUSE_ENABLED=0`` 同样是 NullTracer 分支（即便给了凭据）。"""
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-dummy")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-dummy")
    monkeypatch.setenv("PRA_LANGFUSE_ENABLED", "0")
    rc = smoke.main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "tracing disabled" in out
    assert "PRA_LANGFUSE_ENABLED=0" in out


# ---------------------------------------------------------------------------
# 2) 参数解析
# ---------------------------------------------------------------------------


def test_main_rejects_bad_trace_id(smoke, no_credentials, capsys) -> None:
    rc = smoke.main(["--trace-id", "NOT_A_HEX32"])
    out = capsys.readouterr().out
    assert rc == 2
    assert "invalid --trace-id" in out


def test_trace_id_and_url_helpers(smoke) -> None:
    parser = smoke._build_parser()
    args = parser.parse_args(["--trace-id", "a" * 32, "--timeout", "12", "--no-verify"])
    assert args.trace_id == "a" * 32
    assert args.timeout == 12.0
    assert args.no_verify is True
    assert args.host is None

    # 默认值
    defaults = parser.parse_args([])
    assert defaults.trace_id is None
    assert defaults.timeout == smoke.DEFAULT_TIMEOUT
    assert defaults.no_verify is False

    # 尾斜杠归一（避免 //api/... 双斜杠）
    assert (
        smoke._trace_api_url("http://localhost:3000/", "b" * 32)
        == f"http://localhost:3000/api/public/traces/{'b' * 32}"
    )
    assert smoke._ui_url("http://localhost:3000/", "c" * 32) == f"http://localhost:3000/trace/{'c' * 32}"


# ---------------------------------------------------------------------------
# 3) 认证头 + _api_get 的 URL / header 构造（假 urlopen，不发真实 HTTP）
# ---------------------------------------------------------------------------


def test_auth_header_is_basic_base64(smoke) -> None:
    header = smoke._auth_header("pk-lf-x", "sk-lf-y")
    expected = base64.b64encode(b"pk-lf-x:sk-lf-y").decode("ascii")
    assert header == f"Basic {expected}"
    assert header.startswith("Basic ")


class _FakeResponse:
    """``urlopen`` 替身：支持 ``with`` + ``.read()`` + ``.status``。"""

    def __init__(self, body: bytes, status: int = 200) -> None:
        self._body = body
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def test_api_get_builds_url_and_authorization(smoke, monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_urlopen(request, timeout=None):
        captured["url"] = request.full_url
        captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
        captured["timeout"] = timeout
        return _FakeResponse(b'{"id": "deadbeef"}')

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    status, payload, error = smoke._api_get(
        "http://localhost:3000/",
        "d" * 32,
        public_key="pk-lf-x",
        secret_key="sk-lf-y",
        timeout=3.0,
    )
    assert (status, payload, error) == (200, {"id": "deadbeef"}, None)
    assert captured["url"] == f"http://localhost:3000/api/public/traces/{'d' * 32}"
    assert captured["headers"]["authorization"] == smoke._auth_header("pk-lf-x", "sk-lf-y")
    assert captured["timeout"] == 3.0


def test_api_get_maps_http_error(smoke, monkeypatch) -> None:
    def _fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, None)  # type: ignore[arg-type]

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    status, payload, error = smoke._api_get(
        "http://localhost:3000", "e" * 32, public_key="pk", secret_key="sk"
    )
    assert status == 404
    assert payload is None
    assert error and "404" in error


# ---------------------------------------------------------------------------
# 4) 回读断言清单 + observation 树
# ---------------------------------------------------------------------------


def _sample_trace(trace_id: str) -> dict[str, Any]:
    """与脚本发出的合成 trace 同形状的回读 payload。"""
    return {
        "id": trace_id,
        "name": "smoke",
        "sessionId": "langfuse-smoke",
        "tags": ["env:local", "source:smoke"],
        "observations": [
            {"id": "o0", "parentObservationId": None, "name": "smoke", "type": "SPAN"},
            {"id": "o1", "parentObservationId": "o0", "name": "hypothesize", "type": "SPAN"},
            {
                "id": "o2",
                "parentObservationId": "o1",
                "name": "llm.hypothesize",
                "type": "GENERATION",
                "model": "scripted",
                "usageDetails": {"input": 12, "output": 34, "total": 46},
            },
            {"id": "o3", "parentObservationId": "o0", "name": "plan", "type": "SPAN"},
            {"id": "o4", "parentObservationId": "o0", "name": "tools", "type": "SPAN"},
            {"id": "o5", "parentObservationId": "o4", "name": "ProductTool", "type": "TOOL"},
        ],
    }


def test_verify_trace_passes_on_full_payload(smoke) -> None:
    trace_id = "f" * 32
    assert smoke._verify_trace(_sample_trace(trace_id), trace_id) == []


def test_verify_trace_reports_specific_differences(smoke) -> None:
    trace_id = "f" * 32
    broken = _sample_trace(trace_id)
    broken["sessionId"] = "wrong-session"
    broken["tags"] = ["env:local"]
    broken["observations"] = [obs for obs in broken["observations"] if obs["name"] != "plan"]
    broken["observations"][2]["usageDetails"] = {}
    broken["observations"][2]["model"] = ""
    problems = smoke._verify_trace(broken, trace_id)
    joined = "\n".join(problems)
    assert "sessionId mismatch" in joined
    assert "tags missing" in joined
    assert "missing observation 'plan'" in joined
    assert "usageDetails is empty" in joined
    assert "model is empty" in joined


def test_verify_trace_detects_id_and_type_mismatch(smoke) -> None:
    trace_id = "f" * 32
    broken = _sample_trace("0" * 32)
    broken["observations"][2]["type"] = "SPAN"
    broken["observations"][5]["type"] = "SPAN"
    problems = smoke._verify_trace(broken, trace_id)
    joined = "\n".join(problems)
    assert "trace id mismatch" in joined
    assert "want 'GENERATION'" in joined
    assert "want 'TOOL'" in joined


def test_observation_tree_is_indented_by_parent(smoke) -> None:
    lines = smoke._observation_tree(_sample_trace("f" * 32)["observations"])
    tree = "\n".join(lines)
    assert lines[0].startswith("- smoke [SPAN]")
    assert "  - hypothesize [SPAN]" in tree
    assert "    - llm.hypothesize [GENERATION]" in tree
    assert "  - tools [SPAN]" in tree
    assert "    - ProductTool [TOOL]" in tree
    # 层级：hypothesize / plan / tools 同深度（root 的直接子节点）
    assert [len(line) - len(line.lstrip()) for line in lines] == [0, 2, 4, 2, 2, 4]
    # usage 与 model 应打印出来（一眼看出 generation 的用量口径）
    gen_line = next(line for line in lines if "llm.hypothesize" in line)
    assert "model=scripted" in gen_line
    assert "usage=" in gen_line


def test_observation_tree_handles_orphans_and_empty(smoke) -> None:
    assert smoke._observation_tree([]) == []
    orphan = [{"id": "x", "parentObservationId": "missing", "name": "orphan", "type": "SPAN"}]
    assert smoke._observation_tree(orphan) == ["- orphan [SPAN]  id=x"]


# ---------------------------------------------------------------------------
# 5) main 正常路径（注入假 tracer + 假 urlopen；依旧不发真实 HTTP）
# ---------------------------------------------------------------------------


class _FakeObservation:
    """假观测节点：把 ``update`` 记录到 sink。"""

    def __init__(self, sink: list[tuple[str, dict[str, Any]]], name: str) -> None:
        self._sink = sink
        self._name = name

    def update(self, **kwargs: Any) -> None:
        self._sink.append((self._name, kwargs))

    def record_error(self, exc: BaseException) -> None:  # pragma: no cover - 未用到
        self._sink.append((self._name, {"error": repr(exc)}))


class _FakeTracer:
    """记录调用序列的假 tracer（enabled=True），用于覆盖 main 的已启用路径。"""

    enabled = True

    def __init__(self) -> None:
        self.spans: list[str] = []
        self.updates: list[tuple[str, dict[str, Any]]] = []
        self.flushes = 0

    @contextmanager
    def trace_root(self, ctx) -> Iterator[_FakeObservation]:
        self.spans.append(ctx.name)
        yield _FakeObservation(self.updates, ctx.name)

    @contextmanager
    def node_span(self, name: str, *, input=None, metadata=None) -> Iterator[_FakeObservation]:
        self.spans.append(name)
        yield _FakeObservation(self.updates, name)

    @contextmanager
    def llm_generation(
        self, *, name: str, model: str, input, model_parameters=None, metadata=None
    ) -> Iterator[_FakeObservation]:
        self.spans.append(name)
        yield _FakeObservation(self.updates, name)

    @contextmanager
    def tool_span(self, *, name: str, input=None, metadata=None) -> Iterator[_FakeObservation]:
        self.spans.append(name)
        yield _FakeObservation(self.updates, name)

    def flush(self) -> None:
        self.flushes += 1


@pytest.fixture()
def fake_enabled(monkeypatch: pytest.MonkeyPatch):
    """已启用路径：注入假 tracer + 假凭据（不装 SDK、不联网）。"""
    tracer = _FakeTracer()
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    tracing.set_tracer(tracer)
    return tracer


def test_main_no_verify_emits_all_four_observation_kinds(smoke, fake_enabled, capsys) -> None:
    rc = smoke.main(["--trace-id", "a" * 32, "--no-verify"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "--no-verify: skipped read-back verification" in out
    assert fake_enabled.flushes == 1
    # 四类观测 + 异步子 span 全部发出（root 在最前）
    assert fake_enabled.spans[0] == "smoke"
    for name in (
        "hypothesize",
        "llm.hypothesize",
        "tools",
        "ProductTool",
        "plan",
        smoke.ASYNC_CHILD_SPAN,
    ):
        assert name in fake_enabled.spans, f"缺少观测: {name}"
    # generation 的 update 字段（output / usage_details / metadata）
    gen_kwargs = next(kw for name, kw in fake_enabled.updates if name == "llm.hypothesize")
    assert gen_kwargs["output"] == "{}"
    assert gen_kwargs["usage_details"] == {"input": 12, "output": 34, "total": 46}
    assert gen_kwargs["metadata"] == {"latency_ms": 1}
    # tool 的 update 字段
    tool_kwargs = next(kw for name, kw in fake_enabled.updates if name == "ProductTool")
    assert tool_kwargs["output"] == {"evidence": ["PRODUCT_FACT P_88231"], "status": "ok"}
    # root 的终态 output
    root_kwargs = next(kw for name, kw in fake_enabled.updates if name == "smoke")
    assert root_kwargs["output"] == {"decision": "HUMAN_REVIEW"}


def test_main_prints_pass_with_fake_readback(smoke, fake_enabled, monkeypatch, capsys) -> None:
    trace_id = "f" * 32
    body = json.dumps(_sample_trace(trace_id)).encode()
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda request, timeout=None: _FakeResponse(body)
    )
    rc = smoke.main(["--trace-id", trace_id, "--timeout", "1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "SMOKE PASS" in out
    assert "observation tree:" in out
    assert f"/trace/{trace_id}" in out  # UI 链接
    assert "  - smoke [SPAN]" in out


def test_main_returns_one_when_assertion_fails(smoke, fake_enabled, monkeypatch, capsys) -> None:
    trace_id = "f" * 32
    payload = _sample_trace(trace_id)
    payload["observations"] = [obs for obs in payload["observations"] if obs["name"] != "plan"]
    body = json.dumps(payload).encode()
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda request, timeout=None: _FakeResponse(body)
    )
    rc = smoke.main(["--trace-id", trace_id, "--timeout", "1"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "SMOKE FAIL" in out
    assert "missing observation 'plan'" in out


def test_main_returns_one_when_trace_not_retrievable(smoke, fake_enabled, monkeypatch, capsys):
    def _boom(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    rc = smoke.main(["--trace-id", "f" * 32, "--timeout", "1"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "not retrievable" in out
    assert "URLError" in out
