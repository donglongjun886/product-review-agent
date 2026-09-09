"""tests/test_langfuse_smoke.py —— ``scripts/langfuse_smoke.py`` 的离线单元测试。

**全部不联网、不依赖 langfuse SDK**（当前主 venv 未装，属 optional extra）。
回读口径按 **Langfuse v4 观测列表**（``{"data": [...]}``）构造假 payload —— 服务端
``events_only`` 模式下 v3 的 ``/api/public/traces`` 已 404（实测，见脚本 docstring）：

1. ``test_main_disabled_without_credentials``：无凭据 → ``main([])`` 返回 0，stdout 含
   ``tracing disabled`` 与 ``uv sync --extra observability`` 启用提示（CI 友好契约）；
2. ``test_main_rejects_bad_trace_id``：``--trace-id`` 非法 → 退出码 2（且不碰 tracer）；
3. ``test_trace_id_and_url_helpers`` / ``test_observations_url_carries_fields``：
   ``--trace-id`` 参数解析 + v4 回读 URL（traceId / limit / **fields=** 齐全）；
4. ``test_auth_header_is_basic_base64``：Basic Auth 头 = ``base64(public_key:secret_key)``；
5. ``test_api_get_builds_url_and_authorization``：假 ``urlopen`` 捕获 URL / 认证头 / 超时
   （用例内不发真实 HTTP）；
6. ``test_verify_trace_*``：断言清单在正常 v4 payload 上全通过、在缺字段时能指出具体差异；
7. ``test_observation_tree_*``：树按 ``parentObservationId`` 缩进，**root 以
   ``isRootObservation`` 为准**（v4 的 root 带幽灵父 id）；
8. ``test_fetch_trace_*``：轮询语义（空 data 重试、始终为空 → ``None`` + notes）；
9. ``test_main_*``：main 的 PASS / FAIL / 不可达三条路径（注入假 tracer + 假 urlopen）。

脚本在 ``scripts/``（非包）→ 与 ``tests/test_evaluation_dataset_v2.py`` 同款 importlib
按路径加载，只取纯函数与 ``main``，不执行模块级副作用。
"""

from __future__ import annotations

import base64
import importlib.util
import json
import urllib.error
import urllib.parse
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
# 0) v4 假 payload（与实测形状一致）
# ---------------------------------------------------------------------------

#: 与真实 v4 响应一致的观测列表（root 的 parentObservationId 是**幽灵 id**）。
_GHOST_PARENT = "5f378a385f1a15a3"


def _observation(
    obs_id: str,
    name: str,
    obs_type: str,
    parent: str | None,
    *,
    trace_id: str,
    is_root: bool = False,
    model: str = "",
    usage: dict[str, int] | None = None,
) -> dict[str, Any]:
    """单条 v4 observation（非 generation 的 ``model``/``usageDetails`` 为空，实测）。"""
    return {
        "id": obs_id,
        "traceId": trace_id,
        "parentObservationId": parent,
        "isRootObservation": is_root,
        "type": obs_type,
        "name": name,
        "startTime": "2026-09-09T14:44:31.388Z",
        "endTime": "2026-09-09T14:44:31.388Z",
        "latency": 0,
        "level": "DEFAULT",
        "statusMessage": "",
        "environment": "default",
        "version": "smoke",
        "sessionId": "langfuse-smoke",
        "tags": ["env:local", "source:smoke"],
        "model": model,
        "usageDetails": usage if usage is not None else {},
    }


def _sample_observations(trace_id: str) -> list[dict[str, Any]]:
    """与脚本发出的合成 trace 同形状的 v4 回读观测列表（7 条，含异步子 span）。"""
    return [
        _observation("o0", "smoke", "SPAN", _GHOST_PARENT, trace_id=trace_id, is_root=True),
        _observation("o1", "hypothesize", "SPAN", "o0", trace_id=trace_id),
        _observation(
            "o2",
            "llm.hypothesize",
            "GENERATION",
            "o1",
            trace_id=trace_id,
            model="scripted",
            usage={"input": 12, "output": 34, "total": 46},
        ),
        _observation("o3", "plan", "SPAN", "o0", trace_id=trace_id),
        _observation("o4", "tools", "SPAN", "o0", trace_id=trace_id),
        _observation("o5", "ProductTool", "TOOL", "o4", trace_id=trace_id),
        _observation("o6", "plan.async_child", "SPAN", "o3", trace_id=trace_id),
    ]


def _sample_payload(trace_id: str) -> dict[str, Any]:
    """v4 响应体：``{"data": [...], "meta": {}}``（**不是** v3 的 ``{"observations": …}``）。"""
    return {"data": _sample_observations(trace_id), "meta": {}}


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
# 2) 参数解析 + v4 回读 URL
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

    # 尾斜杠归一（避免 //api/... 双斜杠）+ v4 端点
    url = smoke._observations_api_url("http://localhost:3000/", "b" * 32)
    assert url.startswith("http://localhost:3000/api/public/v2/observations?")
    assert "traceId=" + "b" * 32 in url
    assert "limit=50" in url
    assert smoke._ui_url("http://localhost:3000/", "c" * 32) == f"http://localhost:3000/trace/{'c' * 32}"


def test_observations_url_carries_fields(smoke) -> None:
    """**回归护栏**：回读 URL 必须带 ``fields=``，否则 v4 不返回 model/usageDetails。

    实测：不传 ``fields=`` 时服务端只回 ``core,basic`` → ``model`` / ``usageDetails`` /
    ``tags`` 字段**不存在**（不是 null），model/usage 断言会静默失败。
    """
    url = smoke._observations_api_url("http://localhost:3000", "d" * 32)
    query = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    assert query["fields"] == ["core,basic,model,usage,trace_context"]
    assert query["traceId"] == ["d" * 32]
    assert query["limit"] == ["50"]
    assert "model" in query["fields"][0] and "usage" in query["fields"][0]
    # 默认常量本身也要带上（防止有人把默认值改空）
    assert smoke.OBSERVATION_FIELDS == "core,basic,model,usage,trace_context"


def test_observations_from_payload_requires_v4_shape(smoke) -> None:
    """v3 的 ``{"observations": [...]}`` 形状必须被判为"没读到"，不得静默当空列表。"""
    trace_id = "f" * 32
    assert smoke._observations_from_payload(_sample_payload(trace_id)) == _sample_observations(
        trace_id
    )
    assert smoke._observations_from_payload({"observations": [{"id": "x"}]}) is None
    assert smoke._observations_from_payload({"data": []}) == []
    assert smoke._observations_from_payload({"data": "nope"}) is None
    assert smoke._observations_from_payload(None) is None
    # 非 dict 元素被剔除（外部 payload 不做信任假设）
    assert smoke._observations_from_payload({"data": [{"id": "a"}, "junk"]}) == [{"id": "a"}]


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
        return _FakeResponse(b'{"data": [], "meta": {}}')

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    status, payload, error = smoke._api_get(
        "http://localhost:3000/",
        "d" * 32,
        public_key="pk-lf-x",
        secret_key="sk-lf-y",
        timeout=3.0,
    )
    assert (status, payload, error) == (200, {"data": [], "meta": {}}, None)
    assert captured["url"] == smoke._observations_api_url("http://localhost:3000/", "d" * 32)
    assert "/api/public/v2/observations?" in captured["url"]
    assert f"traceId={'d' * 32}" in captured["url"]
    # 关键：请求 URL 必须带 fields=（否则 model/usage 静默缺失）
    assert "fields=core%2Cbasic%2Cmodel%2Cusage%2Ctrace_context" in captured["url"]
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
# 4) 回读轮询
# ---------------------------------------------------------------------------


def test_fetch_trace_polls_until_observations_visible(smoke, monkeypatch, capsys) -> None:
    """第一次空 ``data``（ingestion 未完成）→ 重试后拿到；轮询间隔缩短避免测试变慢。"""
    trace_id = "f" * 32
    bodies = [b'{"data": [], "meta": {}}', json.dumps(_sample_payload(trace_id)).encode()]
    calls: list[str] = []

    def _fake_urlopen(request, timeout=None):
        calls.append(request.full_url)
        return _FakeResponse(bodies[min(len(calls) - 1, len(bodies) - 1)])

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(smoke, "POLL_INTERVAL_S", 0.01)
    observations, notes = smoke._fetch_trace(
        "http://localhost:3000", trace_id, public_key="pk", secret_key="sk", timeout=10.0
    )
    assert observations == _sample_observations(trace_id)
    assert len(calls) >= 2, "空 data 必须重试"
    assert notes and "observations=0" in notes[0]
    assert "read back on attempt" in capsys.readouterr().out


def test_fetch_trace_returns_none_when_never_visible(smoke, monkeypatch, capsys) -> None:
    trace_id = "f" * 32

    def _fake_urlopen(request, timeout=None):
        return _FakeResponse(b'{"data": [], "meta": {}}')

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(smoke, "POLL_INTERVAL_S", 0.01)
    observations, notes = smoke._fetch_trace(
        "http://localhost:3000", trace_id, public_key="pk", secret_key="sk", timeout=0.05
    )
    assert observations is None
    assert notes, "每次尝试都要留说明（便于失败时定位）"
    assert "not visible yet" in capsys.readouterr().out


def test_fetch_trace_keeps_best_partial_payload(smoke, monkeypatch) -> None:
    """始终凑不满 7 条时返回"最全的一份"，让断言能打印具体差异而非笼统不可达。"""
    trace_id = "f" * 32
    partial = {"data": _sample_observations(trace_id)[:3], "meta": {}}

    def _fake_urlopen(request, timeout=None):
        return _FakeResponse(json.dumps(partial).encode())

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(smoke, "POLL_INTERVAL_S", 0.01)
    observations, _notes = smoke._fetch_trace(
        "http://localhost:3000", trace_id, public_key="pk", secret_key="sk", timeout=0.05
    )
    assert observations is not None and len(observations) == 3


# ---------------------------------------------------------------------------
# 5) 回读断言清单 + observation 树
# ---------------------------------------------------------------------------


def test_verify_trace_passes_on_full_payload(smoke) -> None:
    trace_id = "f" * 32
    assert smoke._verify_trace(_sample_observations(trace_id), trace_id) == []


def test_verify_trace_reports_specific_differences(smoke) -> None:
    trace_id = "f" * 32
    broken = _sample_observations(trace_id)
    broken = [obs for obs in broken if obs["name"] != "plan"]
    for obs in broken:
        if obs["name"] == "smoke":
            obs["sessionId"] = "wrong-session"
            obs["tags"] = ["env:local"]
            obs["version"] = "v0"
        if obs["name"] == "llm.hypothesize":
            obs["usageDetails"] = {}
            obs["model"] = ""
    problems = smoke._verify_trace(broken, trace_id)
    joined = "\n".join(problems)
    assert "observation count 6 < 7" in joined
    assert "sessionId = 'wrong-session'" in joined
    assert "tags missing 'source:smoke'" in joined
    assert "version = 'v0'" in joined
    assert "missing observation 'plan'" in joined
    assert "usageDetails missing/empty" in joined
    assert "model = ''" in joined


def test_verify_trace_detects_id_type_and_root_mismatch(smoke) -> None:
    trace_id = "f" * 32
    broken = _sample_observations("0" * 32)  # traceId 全错
    for obs in broken:
        if obs["name"] == "smoke":
            obs["isRootObservation"] = False
        if obs["name"] == "llm.hypothesize":
            obs["type"] = "SPAN"
        if obs["name"] == "ProductTool":
            obs["type"] = "SPAN"
    problems = smoke._verify_trace(broken, trace_id)
    joined = "\n".join(problems)
    assert "traceId mismatch" in joined
    assert f"want {trace_id!r}" in joined
    assert "isRootObservation = False, want True" in joined
    assert "want 'GENERATION'" in joined
    assert "want 'TOOL'" in joined


def test_verify_trace_detects_usage_key_and_parent_mismatch(smoke) -> None:
    trace_id = "f" * 32
    broken = _sample_observations(trace_id)
    for obs in broken:
        if obs["name"] == "llm.hypothesize":
            obs["usageDetails"] = {"input": 12, "output": 34, "total": 999}
            obs["parentObservationId"] = "o4"  # 挂错到 tools 下
        if obs["name"] == "plan.async_child":
            obs["parentObservationId"] = "o0"  # 应挂 plan
    joined = "\n".join(smoke._verify_trace(broken, trace_id))
    assert "usageDetails['total'] = 999, want 46" in joined
    assert "'llm.hypothesize' parent = 'o4', want 'hypothesize' id 'o1'" in joined
    assert "'plan.async_child' parent = 'o0', want 'plan' id 'o3'" in joined


def test_verify_trace_detects_root_parent_mismatch(smoke) -> None:
    trace_id = "f" * 32
    broken = _sample_observations(trace_id)
    for obs in broken:
        if obs["name"] == "tools":
            obs["parentObservationId"] = "o1"  # 应直接挂 root
    joined = "\n".join(smoke._verify_trace(broken, trace_id))
    assert "'tools' parent = 'o1', want root 'smoke' id 'o0'" in joined


def test_verify_trace_handles_non_list_payload(smoke) -> None:
    assert smoke._verify_trace({"observations": []}, "f" * 32) == [
        "observation payload is not a JSON list: dict"
    ]


def test_observation_tree_is_indented_by_parent(smoke) -> None:
    lines = smoke._observation_tree(_sample_observations("f" * 32))
    tree = "\n".join(lines)
    assert lines[0].startswith("- smoke [SPAN]")
    assert "  - hypothesize [SPAN]" in tree
    assert "    - llm.hypothesize [GENERATION]" in tree
    assert "  - tools [SPAN]" in tree
    assert "    - ProductTool [TOOL]" in tree
    assert "    - plan.async_child [SPAN]" in tree
    # 层级：hypothesize / plan / tools 同深度（root 的直接子节点）
    assert [len(line) - len(line.lstrip()) for line in lines] == [0, 2, 4, 2, 4, 2, 4]
    # usage 与 model 应打印出来（一眼看出 generation 的用量口径）
    gen_line = next(line for line in lines if "llm.hypothesize" in line)
    assert "model=scripted" in gen_line
    assert "usage=" in gen_line


def test_observation_tree_uses_is_root_observation_not_ghost_parent(smoke) -> None:
    """v4 的 root 带幽灵父 id：必须按 ``isRootObservation`` 放到顶层（实测行为）。"""
    observations = _sample_observations("f" * 32)
    # 让幽灵父 id 恰好等于一个真实观测 id —— 若按 parentObservationId 组装会挂错
    for obs in observations:
        if obs["name"] == "smoke":
            obs["parentObservationId"] = "o4"
    lines = smoke._observation_tree(observations)
    assert lines[0].startswith("- smoke [SPAN]")
    assert [len(line) - len(line.lstrip()) for line in lines] == [0, 2, 4, 2, 4, 2, 4]


def test_observation_tree_handles_orphans_and_empty(smoke) -> None:
    assert smoke._observation_tree([]) == []
    orphan = [{"id": "x", "parentObservationId": "missing", "name": "orphan", "type": "SPAN"}]
    assert smoke._observation_tree(orphan) == ["- orphan [SPAN]  id=x"]


# ---------------------------------------------------------------------------
# 6) main 正常路径（注入假 tracer + 假 urlopen；依旧不发真实 HTTP）
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
    body = json.dumps(_sample_payload(trace_id)).encode()
    seen: list[str] = []

    def _fake_urlopen(request, timeout=None):
        seen.append(request.full_url)
        return _FakeResponse(body)

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    rc = smoke.main(["--trace-id", trace_id, "--timeout", "1"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "SMOKE PASS" in out
    assert "observation tree:" in out
    assert f"/trace/{trace_id}" in out  # UI 链接
    assert "  - smoke [SPAN]" in out
    # main 实际发出的回读请求必须带 fields=（端到端护栏）
    assert seen, "应当发起回读请求"
    assert "/api/public/v2/observations?" in seen[0]
    assert "fields=core%2Cbasic%2Cmodel%2Cusage%2Ctrace_context" in seen[0]


def test_main_returns_one_when_assertion_fails(smoke, fake_enabled, monkeypatch, capsys) -> None:
    trace_id = "f" * 32
    payload = _sample_payload(trace_id)
    payload["data"] = [obs for obs in payload["data"] if obs["name"] != "plan"]
    body = json.dumps(payload).encode()
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda request, timeout=None: _FakeResponse(body)
    )
    rc = smoke.main(["--trace-id", trace_id, "--timeout", "1"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "SMOKE FAIL" in out
    assert "missing observation 'plan'" in out
    assert "observation count 6 < 7" in out


def test_main_returns_one_when_trace_not_retrievable(smoke, fake_enabled, monkeypatch, capsys):
    def _boom(request, timeout=None):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    rc = smoke.main(["--trace-id", "f" * 32, "--timeout", "1"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "not retrievable" in out
    assert "URLError" in out
    assert "/api/public/v2/observations" in out  # 失败提示里点明 v4 端点
