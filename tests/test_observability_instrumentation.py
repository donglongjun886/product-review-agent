"""S3 埋点测试 —— LLM generation（llm_shell）+ Tool span（tools_node）逐次记录。

用**假 tracer**（记录用的 `_FakeTracer` / `_FakeObs`，写法参考
`tests/test_observability_tracing.py`）经 `pra.observability.tracing.set_tracer`
注入，断言 docs/09 §3 的埋点位置与内容：

1. 一次成功的 ``call_structured_llm`` → **1 个 generation**（model / input / output /
   latency_ms）；
2. **schema 校验失败重试 → 2 个 generation**（本阶段最关键断言：埋点在内层
   ``backend.complete`` 外层，而不是 ``call_structured_llm`` 外壳）；
3. 后端抛异常 → 该 generation 收到 ``record_error``，且 ``LLMCallOutcome`` 与既有
   行为一致（attempts=2 / error 文本 / tokens=0）；
4. ``tools_node`` 成功调用 → 1 个 tool span（input=args、output 含 evidence、
   metadata.latency_ms 复用既有值）；
5. 工具异常（infra 重试后仍失败）→ tool span 收到 ``record_error``（原有 error record
   与 warn failure 逻辑不变）；
6. 默认路径（无 tracer 注入）→ ``NullTracer``，两条链路跑通且行为不变；
7. token usage 透出：``LLMResponse.usage`` / ``LLMCallOutcome.usage`` 默认 None、
   多次尝试按键累加、litellm 后端从假 usage 对象提取三键（**不联网**）、scripted 桩
   usage=None（**不伪造**）。

隔离：每个用到 tracer 的测试都经 ``_fake_tracer`` fixture 注入并在结束时
``set_tracer(None)`` 复原；LLM 后端注入由 tests/conftest.py 的 autouse fixture 还原。
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
from typing import Any

import pytest
from helpers import AlwaysRaiseBackend, SequenceBackend, ev, plan_conclude_json

from pra.agent.guardrails.llm_shell import (
    LLMBackendError,
    LLMCallOutcome,
    LLMResponse,
    call_structured_llm,
    set_llm_backend,
)
from pra.agent.guardrails.schemas import PlanOutput
from pra.agent.scripted_llm import ScriptedLLMBackend
from pra.agent.tools_node import make_tools_node
from pra.domain.models import Budget
from pra.observability import tracing as T
from pra.tools.base import ToolArgs, ToolResult

# litellm 1.100.0 被 import 时会尝试拉远程 model cost map（联网）—— 关掉它保证离线。
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

_MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "__STATE__ {}"}]
# 合法 JSON 但 schema 不满足（next_action 超词表）→ pydantic ValidationError
_SCHEMA_BAD = '{"next_action": "SOMETHING_ELSE", "tools": []}'


# --------------------------------------------------------------------------------------
# 假 tracer / 假 observation（只记录，不联网、不 import SDK）
# --------------------------------------------------------------------------------------


class _FakeObs:
    """假 observation：分别记录 update / record_error 调用。"""

    def __init__(self) -> None:
        self.updates: list[dict[str, Any]] = []
        self.errors: list[BaseException] = []

    def update(self, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def record_error(self, exc: BaseException) -> None:
        self.errors.append(exc)


class _FakeCM:
    """假上下文管理器（`llm_generation` / `tool_span` 的返回）。"""

    def __init__(self, obs: _FakeObs) -> None:
        self.obs = obs

    def __enter__(self) -> _FakeObs:
        return self.obs

    def __exit__(self, *exc: object) -> bool:
        return False


class _FakeTracer:
    """假 tracer：记录每次观测的创建参数 + 对应 observation。"""

    enabled = True

    def __init__(self) -> None:
        self.generations: list[dict[str, Any]] = []
        self.generation_obs: list[_FakeObs] = []
        self.tool_spans: list[dict[str, Any]] = []
        self.tool_obs: list[_FakeObs] = []

    def llm_generation(
        self,
        *,
        name: str,
        model: str,
        input: Any,
        model_parameters: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> _FakeCM:
        # input 快照：llm_shell 的 work_messages 是同一个 list 对象（重试时会被追加
        # 修正提示），这里存创建时刻的内容 —— 与"本次真实请求发出去的内容"一致。
        snapshot = list(input) if isinstance(input, list) else input
        self.generations.append(
            {
                "name": name,
                "model": model,
                "input": snapshot,
                "model_parameters": model_parameters,
                "metadata": metadata,
            }
        )
        obs = _FakeObs()
        self.generation_obs.append(obs)
        return _FakeCM(obs)

    def tool_span(
        self, *, name: str, input: Any = None, metadata: dict[str, Any] | None = None
    ) -> _FakeCM:
        self.tool_spans.append({"name": name, "input": input, "metadata": metadata})
        obs = _FakeObs()
        self.tool_obs.append(obs)
        return _FakeCM(obs)

    # Tracer 协议其余方法（本文件不消费；给最小实现即可）
    def trace_root(self, ctx: T.TraceContext) -> _FakeCM:
        return _FakeCM(_FakeObs())

    def node_span(
        self, name: str, *, input: Any = None, metadata: dict[str, Any] | None = None
    ) -> _FakeCM:
        return _FakeCM(_FakeObs())

    def flush(self) -> None:
        return None


@pytest.fixture
def fake_tracer() -> Any:
    """注入假 tracer；测试结束复原（避免污染其他测试的默认 NullTracer 路径）。"""
    tracer = _FakeTracer()
    T.set_tracer(tracer)
    try:
        yield tracer
    finally:
        T.set_tracer(None)


# --------------------------------------------------------------------------------------
# 假 Tool（协议：name/description/args_model/call；tools_node 另需 to_evidence）
# --------------------------------------------------------------------------------------


class _EchoArgs(ToolArgs):
    query: str = ""


class _EchoResult(ToolResult):
    """成功信封（ok=True 默认）。"""


class _EchoTool:
    """记录调用次数的测试工具：``fail_times`` 次内抛异常（infra 重试路径用）。"""

    name = "EchoTool"
    description = "测试用回声工具"
    args_model = _EchoArgs

    def __init__(self, *, fail_times: int = 0, ok: bool = True) -> None:
        self.calls = 0
        self.fail_times = fail_times
        self.ok = ok

    async def call(self, args: ToolArgs, ctx: Any) -> ToolResult:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("tool boom")
        if not self.ok:
            return _EchoResult(ok=False, error="业务失败（测试）")
        return _EchoResult(ok=True)

    def to_evidence(self, result: ToolResult) -> list:
        if not result.ok:
            return []
        return [
            ev("PRODUCT_FACT", source=self.name, value="brand=null", weight=0.6, ref_id="P_1")
        ]


def _tools_state(*, tool: str = "EchoTool", args: dict | None = None) -> dict:
    """tools_node 最小 state（case=None → backfill_extra 跳过 version_drift 回填）。"""
    return {
        "budget": Budget(),
        "case": None,
        "evidence": [],
        "tool_call_history": [],
        "pending_tool_calls": [
            {"tool": tool, "args": args if args is not None else {"query": "x"}, "priority": 1}
        ],
    }


_CONFIG = {"configurable": {"thread_id": "run-obs-test"}}


# --------------------------------------------------------------------------------------
# 1-3. LLM generation 埋点
# --------------------------------------------------------------------------------------


async def test_success_records_one_generation_with_model_input_output_latency(fake_tracer) -> None:
    """一次成功调用 → 1 个 generation：name=llm.{node}、model=后端自报、input/output/latency。"""
    backend = SequenceBackend(contents=[plan_conclude_json()], tokens=5)
    set_llm_backend(backend)

    outcome = await call_structured_llm(
        OutputModel=PlanOutput, node="plan", messages=[dict(m) for m in _MESSAGES]
    )

    # 业务行为不变
    assert outcome.model is not None and outcome.attempts == 1 and outcome.error is None
    # 观测：恰好 1 个 generation
    assert len(fake_tracer.generations) == 1
    gen = fake_tracer.generations[0]
    assert gen["name"] == "llm.plan"
    assert gen["model"] == "test-sequence"  # backend.name —— 不伪造模型名
    assert gen["input"] == backend.calls[0]  # 本次实际发给后端的 work_messages
    assert gen["input"] == _MESSAGES

    obs = fake_tracer.generation_obs[0]
    assert len(obs.updates) == 1 and obs.errors == []
    update = obs.updates[0]
    assert update["output"] == plan_conclude_json()
    assert update["metadata"]["attempt"] == 1
    assert update["metadata"]["truncated"] is False
    assert isinstance(update["metadata"]["latency_ms"], int)
    assert update["metadata"]["latency_ms"] >= 0
    # SequenceBackend 不带 usage → 不传 usage_details 键（不伪造 0）
    assert "usage_details" not in update


async def test_schema_retry_records_two_generations(fake_tracer) -> None:
    """**核心断言**：schema 校验失败重试 → 2 个 generation（埋点在内层真实调用外层）。

    第 1 次返回非法 JSON（后端成功、校验失败）→ llm_shell 追加修正提示 → 第 2 次成功。
    Langfuse 必须看到**两次真实 ``backend.complete()``**，而不是只记外壳的 1 次。
    """
    backend = SequenceBackend(contents=[_SCHEMA_BAD, plan_conclude_json()], tokens=5)
    set_llm_backend(backend)

    outcome = await call_structured_llm(
        OutputModel=PlanOutput, node="plan", messages=[dict(m) for m in _MESSAGES]
    )

    # 业务行为不变：重试 1 次后成功
    assert outcome.model is not None
    assert outcome.attempts == 2
    assert outcome.error is None
    assert outcome.tokens == 10
    assert len(backend.calls) == 2  # 两次真实后端调用

    # 观测：两次真实调用 = 两个 generation（关键断言）
    assert len(fake_tracer.generations) == 2
    assert [g["metadata"]["attempt"] for g in fake_tracer.generations] == [1, 2]
    assert [g["name"] for g in fake_tracer.generations] == ["llm.plan", "llm.plan"]
    assert [g["model"] for g in fake_tracer.generations] == ["test-sequence", "test-sequence"]

    obs_first, obs_second = fake_tracer.generation_obs
    # 第 1 次（非法输出）也如实记录：output = 后端原始返回文本
    assert obs_first.updates[0]["output"] == _SCHEMA_BAD
    assert obs_first.updates[0]["metadata"]["attempt"] == 1
    # 第 2 次（修正后）记录合法输出
    assert obs_second.updates[0]["output"] == plan_conclude_json()
    assert obs_second.updates[0]["metadata"]["attempt"] == 2
    # schema 校验失败不是 transport 失败 → 不该有 record_error
    assert obs_first.errors == [] and obs_second.errors == []

    # 第 2 个 generation 的 input 含修正提示（证明记的是内层每次真实调用）
    assert len(fake_tracer.generations[0]["input"]) == 2
    assert len(fake_tracer.generations[1]["input"]) == 3
    assert "请严格按 Schema 重新输出" in fake_tracer.generations[1]["input"][-1]["content"]


async def test_backend_exception_records_error_and_outcome_unchanged(fake_tracer) -> None:
    """后端抛异常 → 每次尝试的 generation 收到 record_error；既有失败语义不变。"""
    backend = AlwaysRaiseBackend()
    set_llm_backend(backend)

    outcome = await call_structured_llm(
        OutputModel=PlanOutput, node="plan", messages=[dict(m) for m in _MESSAGES]
    )

    # 既有行为（埋点前一致）：两次尝试、model=None、error=最后一次异常文本、tokens=0
    assert backend.calls == 2
    assert outcome.model is None
    assert outcome.attempts == 2
    assert outcome.error == "injected backend failure"
    assert outcome.tokens == 0

    # 观测：两次真实调用各一条 generation，均 record_error、均无 output
    assert len(fake_tracer.generations) == 2
    for obs in fake_tracer.generation_obs:
        assert len(obs.errors) == 1
        assert isinstance(obs.errors[0], LLMBackendError)
        assert obs.updates == []


async def test_transport_retry_then_success_records_two_generations(fake_tracer) -> None:
    """transport 类失败重试（第 1 次抛、第 2 次成功）→ 同样 2 个 generation。"""
    backend = SequenceBackend(contents=[None, plan_conclude_json()], tokens=7)
    set_llm_backend(backend)

    outcome = await call_structured_llm(
        OutputModel=PlanOutput, node="plan", messages=[dict(m) for m in _MESSAGES]
    )

    assert outcome.model is not None and outcome.attempts == 2 and outcome.tokens == 7
    assert len(fake_tracer.generations) == 2
    assert len(fake_tracer.generation_obs[0].errors) == 1
    assert fake_tracer.generation_obs[0].updates == []
    assert fake_tracer.generation_obs[1].updates[0]["output"] == plan_conclude_json()


# --------------------------------------------------------------------------------------
# 4-5. Tool span 埋点
# --------------------------------------------------------------------------------------


async def test_tools_node_success_records_one_tool_span(fake_tracer) -> None:
    """一次成功工具调用 → 1 个 tool span（input=args、output 含 evidence、latency 复用）。"""
    tool = _EchoTool()
    node = make_tools_node([tool])

    out = await node(_tools_state(), _CONFIG)

    # 业务行为不变
    record = out["tool_call_history"][0]
    assert record["status"] == "ok"
    assert record["evidence_added"] == ["PRODUCT_FACT brand=null"]
    assert out["failures"] == []

    assert len(fake_tracer.tool_spans) == 1
    span = fake_tracer.tool_spans[0]
    assert span["name"] == "EchoTool"
    assert span["input"] == {"query": "x"}  # args_raw
    assert span["metadata"] == {"seq": 1}

    obs = fake_tracer.tool_obs[0]
    assert obs.errors == []
    update = obs.updates[0]
    assert update["output"] == {"evidence": ["PRODUCT_FACT brand=null"], "status": "ok"}
    # latency_ms 复用既有值（不重测）
    assert update["metadata"]["latency_ms"] == record["latency_ms"]
    assert update["metadata"]["seq"] == record["seq"] == 1


async def test_tools_node_exception_records_error_on_tool_span(fake_tracer) -> None:
    """工具异常（infra 重试后仍失败）→ tool span record_error；error record/warn 不变。"""
    tool = _EchoTool(fail_times=99)  # 两次都抛
    node = make_tools_node([tool])

    out = await node(_tools_state(), _CONFIG)

    # 既有行为：infra 重试 1 次、error record + warn failure、预算计 1 次
    assert tool.calls == 2
    record = out["tool_call_history"][0]
    assert record["status"] == "error"
    assert "infra 重试 1 次后仍失败" in record["error"]
    assert "RuntimeError: tool boom" in record["error"]
    assert out["failures"][0]["severity"] == "warn"  # SEV_WARN 常量值
    assert out["budget"].tool_calls == 1

    # 观测：一次逻辑调用 = 1 个 tool span，收到 record_error
    assert len(fake_tracer.tool_spans) == 1
    obs = fake_tracer.tool_obs[0]
    assert len(obs.errors) == 1
    assert isinstance(obs.errors[0], RuntimeError)
    assert obs.updates == []


async def test_tools_node_business_failure_records_error_on_tool_span(fake_tracer) -> None:
    """业务失败（工具返回 ok=False）→ tool span record_error（reason 可读）。"""
    tool = _EchoTool(ok=False)
    node = make_tools_node([tool])

    out = await node(_tools_state(), _CONFIG)

    assert tool.calls == 1  # 业务失败不触发 infra 重试（行为不变）
    record = out["tool_call_history"][0]
    assert record["status"] == "error" and record["error"] == "业务失败（测试）"

    obs = fake_tracer.tool_obs[0]
    assert len(obs.errors) == 1
    assert "业务失败（测试）" in str(obs.errors[0])
    assert obs.updates == []


async def test_tools_node_multiple_calls_record_one_span_each(fake_tracer) -> None:
    """多次 pending 调用 → 每条一次 tool span（seq 与审计 record 对齐）。"""
    tool = _EchoTool()
    node = make_tools_node([tool])
    state = _tools_state()
    state["pending_tool_calls"] = [
        {"tool": "EchoTool", "args": {"query": "a"}, "priority": 1},
        {"tool": "EchoTool", "args": {"query": "b"}, "priority": 2},
    ]

    out = await node(state, _CONFIG)

    assert len(out["tool_call_history"]) == 2
    assert [s["input"] for s in fake_tracer.tool_spans] == [{"query": "a"}, {"query": "b"}]
    assert [s["metadata"]["seq"] for s in fake_tracer.tool_spans] == [1, 2]


# --------------------------------------------------------------------------------------
# 6. 默认路径（无 tracer 注入）—— NullTracer，行为不变
# --------------------------------------------------------------------------------------


async def test_default_path_uses_null_tracer_and_behaves_unchanged(monkeypatch) -> None:
    """无 tracer 注入 → get_tracer() 是 NullTracer；两条链路跑通且行为不变。"""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("PRA_LANGFUSE_ENABLED", raising=False)
    T.set_tracer(None)  # 默认路径
    try:
        assert isinstance(T.get_tracer(), T.NullTracer)

        # LLM 链路（默认 NullTracer + 假后端）
        backend = SequenceBackend(contents=[plan_conclude_json()], tokens=3)
        set_llm_backend(backend)
        outcome = await call_structured_llm(
            OutputModel=PlanOutput, node="plan", messages=[dict(m) for m in _MESSAGES]
        )
        assert outcome.model is not None and outcome.attempts == 1
        assert outcome.tokens == 3

        # 工具链路（默认 NullTracer）
        tool = _EchoTool()
        out = await make_tools_node([tool])(_tools_state(), _CONFIG)
        assert tool.calls == 1
        assert out["tool_call_history"][0]["status"] == "ok"
        assert out["evidence"][0].value == "brand=null"

        # 默认路径不拉起 SDK
        assert "langfuse" not in sys.modules
    finally:
        T.set_tracer(None)


# --------------------------------------------------------------------------------------
# 7. token usage 透出（向后兼容的可选字段）
# --------------------------------------------------------------------------------------


def test_usage_fields_default_to_none() -> None:
    """新增字段默认 None —— 不破坏任何既有构造点/测试。"""
    assert LLMResponse(content="{}", tokens=0).usage is None
    assert LLMResponse(content="{}", tokens=0, truncated=True).usage is None
    assert LLMCallOutcome(model=None, attempts=1, tokens=0, error="x").usage is None


class _UsageBackend:
    """按序返回 (content, usage) 的替身；usage 为 None 表示该次响应无 usage。"""

    name = "test-usage"

    def __init__(self, responses: list[tuple[str, dict | None]]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def complete(self, *, node: str, messages: list, json_schema: dict) -> LLMResponse:
        self.calls += 1
        content, usage = self._responses.pop(0)
        tokens = (usage or {}).get("total", 0)
        return LLMResponse(content=content, tokens=tokens, usage=usage)


async def test_outcome_usage_accumulated_across_attempts(fake_tracer) -> None:
    """多次尝试的 usage **按键累加**；generation 记每次的原始 usage。"""
    backend = _UsageBackend(
        [
            (_SCHEMA_BAD, {"input": 1, "output": 2, "total": 3}),
            (plan_conclude_json(), {"input": 4, "output": 5, "total": 9}),
        ]
    )
    set_llm_backend(backend)

    outcome = await call_structured_llm(
        OutputModel=PlanOutput, node="plan", messages=[dict(m) for m in _MESSAGES]
    )

    assert outcome.attempts == 2
    assert outcome.tokens == 12
    assert outcome.usage == {"input": 5, "output": 7, "total": 12}
    # 观测：每条 generation 记的是**本次** usage（不累加）
    assert fake_tracer.generation_obs[0].updates[0]["usage_details"] == {
        "input": 1,
        "output": 2,
        "total": 3,
    }
    assert fake_tracer.generation_obs[1].updates[0]["usage_details"] == {
        "input": 4,
        "output": 5,
        "total": 9,
    }


async def test_outcome_usage_none_when_all_responses_lack_usage() -> None:
    """全部响应无 usage → outcome.usage 保持 None（不伪造 0）。"""
    set_llm_backend(_UsageBackend([(plan_conclude_json(), None)]))

    outcome = await call_structured_llm(
        OutputModel=PlanOutput, node="plan", messages=[dict(m) for m in _MESSAGES]
    )

    assert outcome.model is not None
    assert outcome.usage is None
    assert outcome.tokens == 0


async def test_outcome_usage_partial_when_only_one_attempt_has_usage() -> None:
    """仅某次尝试有 usage → 累加该次（缺失的不补 0）。"""
    set_llm_backend(
        _UsageBackend([(_SCHEMA_BAD, None), (plan_conclude_json(), {"total": 8})])
    )

    outcome = await call_structured_llm(
        OutputModel=PlanOutput, node="plan", messages=[dict(m) for m in _MESSAGES]
    )

    assert outcome.model is not None and outcome.attempts == 2
    assert outcome.usage == {"total": 8}  # 缺失的第 1 次不补 0


def test_litellm_backend_extracts_usage_details(monkeypatch) -> None:
    """litellm 后端从假 ``resp.usage`` 提取三键（不联网）；缺键不放、无 usage → None。"""
    import litellm

    from pra.agent.litellm_backend import LiteLLMBackend

    backend = LiteLLMBackend(api_key="sk-test")
    messages = [{"role": "user", "content": "__STATE__ {}"}]

    def _fake_acompletion(usage: Any):
        async def fake(**kwargs):
            return _resp(usage)

        return fake

    def _resp(usage: Any) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content='{"a": 1}'), finish_reason="stop"
                )
            ],
            usage=usage,
        )

    cases = [
        # 三键齐全
        (
            types.SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14),
            {"input": 10, "output": 4, "total": 14},
            14,
        ),
        # 只有 total（旧口径兼容：usage 对象缺 prompt/completion）
        (types.SimpleNamespace(total_tokens=23), {"total": 23}, 23),
        # 缺 total：不填 0 冒充 → usage 只有 input/output、tokens=0
        (
            types.SimpleNamespace(prompt_tokens=2, completion_tokens=3),
            {"input": 2, "output": 3},
            0,
        ),
        # 无 usage 属性 → usage=None、tokens=0
        (None, None, 0),
    ]
    for usage_obj, expected_usage, expected_tokens in cases:
        monkeypatch.setattr(litellm, "acompletion", _fake_acompletion(usage_obj))
        resp = asyncio.run(
            backend.complete(node="hypothesize", messages=messages, json_schema={})
        )
        assert resp.usage == expected_usage, (usage_obj, resp.usage)
        assert resp.tokens == expected_tokens


async def test_scripted_backend_never_fabricates_usage() -> None:
    """确定性桩无真实 token：tokens=0、usage=None（绝不伪造）。"""
    resp = await ScriptedLLMBackend().complete(
        node="hypothesize", messages=list(_MESSAGES), json_schema={}
    )
    assert resp.tokens == 0
    assert resp.usage is None
