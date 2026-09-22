"""埋点测试：Node span、Root trace、``gate`` 子 span。

用**假 tracer** 经 ``pra.observability.tracing.set_tracer`` 注入 —— 不联网、不 import
``langfuse`` SDK、不依赖凭据；每个测试结束 ``set_tracer(None)`` 复原。
- ``run_review`` → **恰好 1 个 root**（HTTP 路径 trace_id = run_id 原样），入参只带轻量标识，
  root 结束写回终裁摘要；5 个 node span 都出现，``gate`` 是 ``decide`` 的**子** span。
- ``trace_id_from_run_id``：32-hex 原样 / 非 hex → uuid5（确定性）/ 空串·None 不抛；非 hex 时
  metadata.run_id 保留原始值。评测路径 trace_id = uuid5(experiment:case:agent:后端名)，
  **必须含 LLM 后端名**，否则同进程的不同后端臂会落进同一条 trace。
- **默认路径**决策与注入 tracer 时完全一致，且 ``langfuse`` 不进 ``sys.modules``。
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

import pytest
from helpers import make_case

from pra.api.service import run_review
from pra.domain.models import Decision
from pra.evaluation.dataset.loader import load_dataset
from pra.observability import flush_tracer as pkg_flush_tracer
from pra.observability import tracing as T

DATA_PATH_V2 = Path(__file__).resolve().parents[1] / "eval_data" / "v2" / "cases_v2.jsonl"
_EVAL_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_evaluation.py"


def _load_eval_script():
    """按路径加载跑分脚本（它不是包）—— 取其中的评测 root trace 组装。"""
    spec = importlib.util.spec_from_file_location("run_evaluation_mod", _EVAL_SCRIPT)
    assert spec and spec.loader, f"无法定位脚本: {_EVAL_SCRIPT}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_HEX32 = re.compile(r"\A[0-9a-f]{32}\Z")
_NODE_NAMES = ("hypothesize", "plan", "tools", "reevaluate", "decide")


# 假 tracer / 假 observation（记录 + 记录 span 嵌套父节点；不联网、不 import SDK）


class _FakeObs:
    """假 observation：分别记录 update / record_error 调用。"""

    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []
        self.errors: list[BaseException] = []

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def record_error(self, exc: BaseException) -> None:
        self.errors.append(exc)

    def last_output(self) -> Any:
        """最后一次 update 的 output（无则 None）。"""
        for update in reversed(self.updates):
            if "output" in update:
                return update["output"]
        return None


class _FakeCM:
    """假上下文管理器：进入时把 span 名压栈（供断言父子关系），退出时弹栈。"""

    def __init__(self, obs: _FakeObs, stack: list[str], name: str) -> None:
        self.obs = obs
        self._stack = stack
        self._name = name

    def __enter__(self) -> _FakeObs:
        self._stack.append(self._name)
        return self.obs

    def __exit__(self, *exc: object) -> bool:
        self._stack.pop()
        return False


class _FakeTracer:
    """假 tracer：记录 root / node span 的创建参数与嵌套关系。"""

    enabled = True

    def __init__(self) -> None:
        self.roots: list[T.TraceContext] = []
        self.root_obs: list[_FakeObs] = []
        self.node_spans: list[dict[str, Any]] = []
        self.llm_generations: list[dict[str, Any]] = []
        self.tool_spans: list[dict[str, Any]] = []
        self.flushes = 0
        self._stack: list[str] = []

    def trace_root(self, ctx: T.TraceContext) -> _FakeCM:
        self.roots.append(ctx)
        obs = _FakeObs()
        self.root_obs.append(obs)
        return _FakeCM(obs, self._stack, ctx.name)

    def node_span(
        self, name: str, *, input: Any = None, metadata: dict[str, Any] | None = None
    ) -> _FakeCM:
        parent = self._stack[-1] if self._stack else None
        obs = _FakeObs()
        self.node_spans.append(
            {"name": name, "input": input, "metadata": metadata, "parent": parent, "obs": obs}
        )
        return _FakeCM(obs, self._stack, name)

    def llm_generation(self, **kwargs: Any) -> _FakeCM:
        record = dict(kwargs)
        record["parent"] = self._stack[-1] if self._stack else None
        self.llm_generations.append(record)
        return _FakeCM(_FakeObs(), self._stack, kwargs.get("name", "llm"))

    def tool_span(self, **kwargs: Any) -> _FakeCM:
        record = dict(kwargs)
        record["parent"] = self._stack[-1] if self._stack else None
        self.tool_spans.append(record)
        return _FakeCM(_FakeObs(), self._stack, kwargs.get("name", "tool"))

    def flush(self) -> None:
        self.flushes += 1

    # -- 断言辅助 -------------------------------------------------------------------
    def spans_named(self, name: str) -> list[dict[str, Any]]:
        return [s for s in self.node_spans if s["name"] == name]


@pytest.fixture
def fake_tracer() -> Any:
    """注入假 tracer；测试结束复原（避免污染其他测试的默认 NullTracer 路径）。"""
    tracer = _FakeTracer()
    T.set_tracer(tracer)
    try:
        yield tracer
    finally:
        T.set_tracer(None)


def _v2_case():
    """v2 数据集第一条 case（评测路径用；确定性、无网络）。"""
    cases = load_dataset(DATA_PATH_V2)
    return cases[0]


# 1. 无 DB 执行入口 root trace（pra.api.service.run_review）


async def test_run_review_emits_exactly_one_root_trace_with_run_id_trace_id(fake_tracer) -> None:
    """跑一次 ``run_review`` → **恰好 1 个 root**，trace_id = run_id、metadata/tags 口径正确。"""
    case = make_case()
    run_id = uuid4().hex  # 32-hex run_id 形态（trace_id 原样 = run_id）

    result = await run_review(case, run_id=run_id)

    assert len(fake_tracer.roots) == 1
    ctx = fake_tracer.roots[0]
    assert ctx.name == "review"
    assert ctx.trace_id == run_id == T.trace_id_from_run_id(run_id)
    assert _HEX32.match(ctx.trace_id)
    # metadata 含 run_id 对齐键（原值，不是 trace_id）；入参轻量：只带 case 标识，不塞整个 case
    assert ctx.metadata["run_id"] == run_id
    # 本入口无落库、不走 HTTP：source 不得冒充 http（HTTP 路径是 persist_service）
    assert ctx.metadata["source"] == "local"
    assert ctx.tags == ["env:local", "source:local"]
    assert ctx.input == {"case_id": case.case_id}
    # root 结束写回终裁摘要
    assert fake_tracer.root_obs[0].last_output() == {
        "decision": result.review_decision.decision.value,
        "risk_level": result.review_decision.risk_level.value,
    }


async def test_node_spans_cover_all_five_nodes_and_gate_is_child_of_decide(fake_tracer) -> None:
    """5 个 node span 都出现、入参轻量；``gate`` 是 ``decide`` 的子 span（proposal → 终裁）。"""
    case = make_case()

    # 唯一 run_id → 全新 LangGraph 线程（避免与其它用例共用 checkpointer 线程导致
    # dedup 跳过 tools 访问；本用例要断言 5 个节点都出现）
    await run_review(case, run_id=f"RUN_NODES_{uuid4().hex[:8]}")

    names = [s["name"] for s in fake_tracer.node_spans]
    for expected in _NODE_NAMES:
        assert expected in names, f"缺少 node span: {expected}（实得 {names}）"
    assert names.count("hypothesize") == 1  # 入口节点只访问一次
    assert names.count("decide") == 1  # 唯一终态出口只访问一次

    # 入参摘要轻量：带 case 标识而非整个 state（逐键集合属实现细节，不锁定）
    for span in fake_tracer.node_spans:
        assert span["input"] is not None, span["name"]
        if span["name"] == "gate":
            continue
        assert span["input"]["case_id"] == case.case_id, span["name"]
    # 每个节点 span 都有 output 摘要（决定是否含 decision 标量）
    for span in fake_tracer.node_spans:
        assert span["obs"].last_output() is not None, span["name"]

    decide_span = fake_tracer.spans_named("decide")[0]
    assert decide_span["parent"] == "review"
    assert decide_span["obs"].last_output()["decision"] in {
        "PASS",
        "REJECT",
        "HUMAN_REVIEW",
    }

    # Gate 子 span：decide 内部调用 run_decision_overlay 的那一层（结构：proposal → 终裁）
    gate_spans = fake_tracer.spans_named("gate")
    assert len(gate_spans) == 1
    gate = gate_spans[0]
    assert gate["parent"] == "decide"  # 不是图节点，是 decide 的子观测
    gate_output = gate["obs"].last_output()
    assert gate_output is not None
    # gate 终裁与 decide 节点透出的 decision 一致（overlay 结果被采纳）
    assert gate_output["decision"] == decide_span["obs"].last_output()["decision"]


async def test_non_hex_run_id_gives_deterministic_32hex_trace_id(fake_tracer) -> None:
    """非 32-hex run_id → trace_id 是 32-hex 且**两次同值**（metadata.run_id 保留原值）。"""
    case = make_case()

    first = await run_review(case, run_id="RUN_CASE_1")
    second = await run_review(case, run_id="RUN_CASE_1")

    assert len(fake_tracer.roots) == 2
    ids = [ctx.trace_id for ctx in fake_tracer.roots]
    assert ids[0] == ids[1] == T.trace_id_from_run_id("RUN_CASE_1")
    assert _HEX32.match(ids[0])
    assert all(ctx.metadata["run_id"] == "RUN_CASE_1" for ctx in fake_tracer.roots)
    # 旁路：两次决策一致（包装不改判定）
    assert first.review_decision.decision == second.review_decision.decision


# 2. trace_id / experiment / session 纯函数


def test_trace_id_from_run_id_passthrough_and_uuid5_fallback() -> None:
    """32-hex 原样返回；非 hex → uuid5(NAMESPACE_URL, run_id)；空串/None 不抛。"""
    hex32 = "0af7651916cd43dd8448eb211c80319c"
    assert T.trace_id_from_run_id(hex32) == hex32  # 原样：与 MySQL run_id 硬对齐
    # 大写 hex 不是 W3C 形状 → uuid5（确定性，但不再等于 run_id）
    upper = hex32.upper()
    assert T.trace_id_from_run_id(upper) == uuid5(NAMESPACE_URL, upper).hex

    for raw in ("RUN_CASE_1", "eval-agent-EC_0123", "EC_0123"):
        expected = uuid5(NAMESPACE_URL, raw).hex
        assert T.trace_id_from_run_id(raw) == expected
        assert _HEX32.match(expected)
        assert T.trace_id_from_run_id(raw) == T.trace_id_from_run_id(raw)  # 确定性

    # 非法输入不抛（观测旁路）
    for raw in ("", None, 0):
        out = T.trace_id_from_run_id(raw)  # type: ignore[arg-type]
        assert _HEX32.match(out)
        assert out == uuid5(NAMESPACE_URL, str(raw or "")).hex


def test_experiment_name_and_session_id_env(monkeypatch) -> None:
    """``PRA_LANGFUSE_EXPERIMENT`` 缺省 baseline；``PRA_LANGFUSE_SESSION`` 缺省 None。"""
    monkeypatch.delenv("PRA_LANGFUSE_EXPERIMENT", raising=False)
    monkeypatch.delenv("PRA_LANGFUSE_SESSION", raising=False)
    assert T.experiment_name() == "baseline"
    assert T.session_id() is None

    monkeypatch.setenv("PRA_LANGFUSE_EXPERIMENT", "prompt-v2")
    monkeypatch.setenv("PRA_LANGFUSE_SESSION", "eval-run-2026-09-09")
    assert T.experiment_name() == "prompt-v2"
    assert T.session_id() == "eval-run-2026-09-09"

    monkeypatch.setenv("PRA_LANGFUSE_EXPERIMENT", "   ")
    monkeypatch.setenv("PRA_LANGFUSE_SESSION", "")
    assert T.experiment_name() == "baseline"  # 空白 → 缺省
    assert T.session_id() is None


def test_flush_tracer_delegates_and_swallows_errors() -> None:
    """``flush_tracer`` 委派 ``Tracer.flush``；flush 抛异常也不外泄（观测旁路）。"""
    tracer = _FakeTracer()
    T.set_tracer(tracer)
    try:
        T.flush_tracer()
        assert tracer.flushes == 1
    finally:
        T.set_tracer(None)

    class _Boom:
        enabled = True

        def flush(self) -> None:
            raise RuntimeError("flush boom")

    T.set_tracer(_Boom())  # type: ignore[arg-type]
    try:
        T.flush_tracer()  # 不抛
    finally:
        T.set_tracer(None)


def test_flush_tracer_is_noop_and_returns_none_with_null_tracer(monkeypatch) -> None:
    """S5 前置契约：默认（NullTracer）下 ``flush_tracer()`` 不抛、返回 None，且已包级导出。"""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("PRA_LANGFUSE_ENABLED", raising=False)
    T.set_tracer(None)
    try:
        assert isinstance(T.get_tracer(), T.NullTracer)  # 默认路径 = NullTracer
        assert T.flush_tracer() is None  # no-op：不抛、返回 None
    finally:
        T.set_tracer(None)

    # 冻结的导出契约：`from pra.observability import flush_tracer` 可用（S5b CLI 调用路径）
    assert pkg_flush_tracer is T.flush_tracer
    assert "flush_tracer" in T.__all__
    import pra.observability as pkg

    assert "flush_tracer" in pkg.__all__
    assert pkg_flush_tracer() is None


# 3. 落库路径 root trace（pra.infra.persist_service.run_and_persist，假 session 不连库）


class _FakeSession:
    """假 AsyncSession：只记录 add/execute/commit，不连任何数据库。"""

    def __init__(self) -> None:
        self.added: list[Any] = []
        self.commits = 0
        self.executed: list[Any] = []

    async def get(self, model: Any, pk: Any) -> None:
        return None  # 无既有行 → 走 INSERT 分支

    def add(self, row: Any) -> None:
        self.added.append(row)

    async def commit(self) -> None:
        self.commits += 1

    async def execute(self, stmt: Any) -> None:
        self.executed.append(stmt)


class _FakeSessionCM:
    """`async with sessionmaker() as session` 的假上下文。"""

    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    async def __aenter__(self) -> _FakeSession:
        return self._session

    async def __aexit__(self, *exc: object) -> bool:
        return False


async def test_run_and_persist_emits_root_and_nested_spans_without_db(
    fake_tracer, monkeypatch
) -> None:
    """落库路径（``astream``）同样 1 个 root + 节点 span + 嵌套子观测（假 session，不连库）。"""
    from pra.infra import persist_service as ps

    session = _FakeSession()
    monkeypatch.setattr(ps, "get_sessionmaker", lambda: (lambda: _FakeSessionCM(session)))
    case = make_case()
    run_id = uuid4().hex

    out = await ps.run_and_persist(case, run_id=run_id, triage_result="COMPLEX")

    # 业务行为不变：五表落库链路走完（假 session 记录 add/execute/commit）
    assert out["run_id"] == run_id
    assert out["decision"] is not None
    assert session.added and session.executed and session.commits > 0

    # root：恰好 1 个，trace_id = run_id（32-hex 原样）+ 含 run_id 对齐键 + 终裁 output
    assert len(fake_tracer.roots) == 1
    ctx = fake_tracer.roots[0]
    assert ctx.name == "review"
    assert ctx.trace_id == run_id == T.trace_id_from_run_id(run_id)
    assert ctx.metadata["run_id"] == run_id
    assert fake_tracer.root_obs[0].last_output() == {
        "decision": out["decision"].decision.value,
        "risk_level": out["decision"].risk_level.value,
    }

    # 5 个节点 span 齐全（嵌套子观测的父子挂点属实现细节，不逐项锁定）
    names = [s["name"] for s in fake_tracer.node_spans]
    for expected in _NODE_NAMES:
        assert expected in names, f"缺少 node span: {expected}（实得 {names}）"
    assert fake_tracer.tool_spans
    assert fake_tracer.llm_generations
    assert fake_tracer.flushes == 0  # 常驻/评测路径都不 per-request flush


# 4. 默认路径（无 tracer 注入）—— 行为完全一致 + 不拉起 SDK


async def test_default_path_behaves_identically_and_never_imports_sdk(monkeypatch) -> None:
    """不注入 tracer → 决策与注入 tracer 时完全一致；``langfuse`` 不在 ``sys.modules``。"""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("PRA_LANGFUSE_ENABLED", raising=False)
    case = make_case()

    # 注入假 tracer 的臂
    tracer = _FakeTracer()
    T.set_tracer(tracer)
    try:
        traced = await run_review(case, run_id="RUN_CASE_TRACED")
    finally:
        T.set_tracer(None)

    # 默认臂（NullTracer 懒装配）
    T.set_tracer(None)
    assert isinstance(T.get_tracer(), T.NullTracer)
    plain = await run_review(case, run_id="RUN_CASE_PLAIN")

    # 决策完全一致（埋点旁路，不改 AgentState / 路由 / 决策序列）
    assert traced.review_decision.decision == plain.review_decision.decision
    assert traced.review_decision.risk_level == plain.review_decision.risk_level
    assert traced.review_decision.overrides == plain.review_decision.overrides

    # 默认路径不拉起 SDK（惰性 import 契约）
    assert "langfuse" not in sys.modules

    T.set_tracer(None)


async def test_run_review_with_llm_pinned_to_stub_decision_is_unchanged(monkeypatch) -> None:
    """生产入口的 LLM 由 ``tests/conftest.py`` 钉回 scripted 桩：决策确定（实测 HUMAN_REVIEW/HIGH 档）。"""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    T.set_tracer(None)

    run_id = f"RUN_DEFAULT_{uuid4().hex[:8]}"
    result = await run_review(make_case(), run_id=run_id)

    assert result.run_id == run_id
    assert result.review_decision.decision == Decision.HUMAN_REVIEW
    assert "langfuse" not in sys.modules


# --------------------------------------------------------------------------------------
# 5. 评测 root trace：同 experiment + 同案 + 不同 LLM 后端 → 不同 trace
# --------------------------------------------------------------------------------------


def test_root_trace_id_is_scoped_by_llm_backend(monkeypatch) -> None:
    """trace_id 必须含 LLM 后端名 —— 否则同进程的不同后端臂会落进同一条 trace。

    同一 experiment 下不同 LLM 后端（对照臂 vs 真实臂）若共用旧公式
    ``uuid5(experiment:case:agent)``，两臂落进**同一条 trace**（每 trace 2 个 root、
    generation 交织，按 trace 汇总 token 会混入另一臂的 generation）。
    """
    monkeypatch.delenv("PRA_LANGFUSE_EXPERIMENT", raising=False)
    mod = _load_eval_script()

    case = _v2_case()
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
    # 除 trace_id / llm_backend 外，其余关联信息一致（同一批 case 可横向对比）
    assert {k: v for k, v in scripted.metadata.items() if k != "llm_backend"} == {
        k: v for k, v in real.metadata.items() if k != "llm_backend"
    }
