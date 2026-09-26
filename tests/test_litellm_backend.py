"""real LLM 路径的 mock 单测 —— LiteLLMBackend + llm_prompts 渲染层。"""

from __future__ import annotations

import os
import subprocess
import sys
import types
from typing import Any

import pytest
from helpers import plan_conclude_json
from pydantic import BaseModel, Field

from pra.agent.guardrails.llm_shell import (
    LLMBackendError,
    call_structured_llm,
)
from pra.agent.guardrails.schemas import PlanOutput
from pra.agent.litellm_backend import LiteLLMBackend
from pra.agent.llm_prompts import (
    SYSTEM_PROMPTS,
    build_system_prompt,
    build_user_prompt,
)

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

_PRODUCT_STATE = {
    "case": {
        "case_id": "CASE_MOCK_01",
        "merchant_id": "M_MOCK",
        "event_type": "NEW_LISTING",
        "product": {
            "product_id": "P_MOCK_99",
            "title": "复古跑鞋（高仿嫌疑样）",
            "description": "复刻经典款鞋型。",
            "category": "女鞋/运动鞋",
            "brand": None,
            "attributes": {"sole": "橡胶"},
            "sku_list": [
                {"sku_id": "S_1", "color": "米白", "size": "38", "price": 219.0},
            ],
            "images": [
                {
                    "url": "https://cdn.example.com/img1.jpg",
                    "source": "主图",
                    "ocr_text": "OCR：疑似品牌 LOGO 图案",
                },
            ],
            "listing_time": "2024-09-06T14:00:00",
        },
    },
}

_PLAN_STATE = {
    "hypotheses": [],
    "evidence": [],
    "case": {
        "case_id": "C_MOCK",
        "merchant_id": "M_MOCK",
        "event_type": "NEW_LISTING",
        "product": {
            "product_id": "P_1",
            "title": "t",
            "description": "d",
            "category": "c",
            "brand": None,
            "images": [],
            "sku_list": [],
            "listing_time": "2024-01-01T00:00:00",
        },
    },
}

_SCHEMA_BAD = '{"next_action": "SOMETHING_ELSE", "tools": [], "rationale": "x"}'

_SIMPLE_SCHEMA = {
    "title": "DecisionProposal",
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["PASS", "REJECT", "HUMAN_REVIEW"],
        },
        "risk_type": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["decision"],
}


def _make_fake_acompletion(contents: list, tokens: int = 7, finish_reason: str = "stop"):
    """返回可 monkeypatch 到 ``litellm.acompletion`` 的 async 替身，记录每次调用。

    ``contents`` 为逐个返回的 content 文本列表（耗尽后再被调用 → pytest.fail）；
    ``calls`` 记录每次的 kwargs/messages；``finish_reason`` 为每次响应的 finish_reason。
    """

    queue = list(contents)

    async def fake_acompletion(**kwargs):
        fake_acompletion.calls.append(
            {"messages": list(kwargs["messages"]), "kwargs": dict(kwargs)}
        )
        if not queue:
            pytest.fail(
                "litellm.acompletion 被多调用了一次 —— 有未被 mock 的真实调用泄漏？"
            )
        content = queue.pop(0)
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content=content),
                    finish_reason=finish_reason,
                )
            ],
            usage=types.SimpleNamespace(total_tokens=tokens),
        )

    fake_acompletion.calls = []
    return fake_acompletion


def _patch_acompletion(monkeypatch, contents: list, tokens: int = 7, finish_reason: str = "stop"):
    """import litellm 并把 acompletion monkeypatch 为按序返回的本地替身。"""
    import litellm

    fake = _make_fake_acompletion(contents, tokens=tokens, finish_reason=finish_reason)
    monkeypatch.setattr(litellm, "acompletion", fake)
    return fake


def _patch_acompletion_with_usage(monkeypatch, usage: Any):
    """按**任意** ``usage`` 对象 patch acompletion。"""
    import litellm

    calls: list = []

    async def fake(**kwargs):
        if calls:
            pytest.fail("litellm.acompletion 被多调用了一次 —— 有未被 mock 的真实调用泄漏？")
        calls.append(kwargs)
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content='{"a": 1}'), finish_reason="stop"
                )
            ],
            usage=usage,
        )

    monkeypatch.setattr(litellm, "acompletion", fake)
    return fake


def test_constructor_default_model_and_name_format():
    """默认 model="deepseek/deepseek-flash"；name = f"litellm-{model}"（构造无 key 不抛）。"""
    backend = LiteLLMBackend()
    assert backend.model == "deepseek/deepseek-flash"
    assert backend.name == "litellm-deepseek/deepseek-flash"
    alt = LiteLLMBackend(model="openai/gpt-4o-mini")
    assert alt.name == "litellm-openai/gpt-4o-mini"


def test_constructor_explicit_api_key_overrides_env(monkeypatch):
    """显式 api_key 优先于环境变量；base_url/max_tokens 一并固化。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
    backend = LiteLLMBackend(
        api_key="sk-explicit",
        base_url="https://gateway.example.com/v1",
        max_tokens=128,
        temperature=0.5,
    )
    assert backend._api_key == "sk-explicit"
    assert backend.base_url == "https://gateway.example.com/v1"
    assert backend.max_tokens == 128
    assert backend.temperature == 0.5


def test_constructor_api_key_from_env_when_not_passed(monkeypatch):
    """api_key=None 时读环境变量 DEEPSEEK_API_KEY 作为缺省来源。"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env")
    backend = LiteLLMBackend()
    assert backend._api_key == "sk-from-env"


async def test_no_api_key_constructor_ok_first_complete_raises(monkeypatch):
    """环境无 DEEPSEEK_API_KEY：构造不抛；首次 complete 抛 LLMBackendError 中文提示。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    backend = LiteLLMBackend()
    with pytest.raises(LLMBackendError) as ei:
        await backend.complete(node="hypothesize", state={}, json_schema={})
    message = str(ei.value)
    assert "DEEPSEEK_API_KEY" in message
    assert "未配置" in message


def test_tool_catalog_extracted_from_tool_objects():
    """tools 目录提取：name/description/args_schema；无 name 的坏工具被跳过。"""
    class _SearchArgs(BaseModel):
        query: str = Field(description="检索描述")
        top_k: int = Field(default=3, description="返回前 k 条")

    class _MerchantArgs(BaseModel):
        merchant_id: str = Field(description="商家 ID")

    class _FakeSearchTool:
        name = "PolicySearchTool"
        description = "政策条款检索取证工具"
        args_model = _SearchArgs

    class _FakeMerchantTool:
        name = "MerchantTool"
        description = "商家历史行为取证"
        args_model = _MerchantArgs

    class _BrokenTool:
        name = ""
        description = "broken"
        args_model = None

    backend = LiteLLMBackend(
        api_key="sk-test",
        tools=[_FakeSearchTool(), _FakeMerchantTool(), _BrokenTool()],
    )
    catalog = backend._tool_catalog
    assert [t["name"] for t in catalog] == ["PolicySearchTool", "MerchantTool"]
    search = catalog[0]
    assert search["description"] == "政策条款检索取证工具"
    assert "query" in search["args_schema"]["properties"]
    assert "top_k" in search["args_schema"]["properties"]
    assert search["args_schema"]["required"] == ["query"]
    assert LiteLLMBackend(api_key="sk-test", tools=None)._tool_catalog == []
    assert LiteLLMBackend(api_key="sk-test", tools=[])._tool_catalog == []


async def test_complete_success_content_tokens_and_kwargs(monkeypatch):
    """mock acompletion 成功路径：content/tokens 正确；kwargs 含 model/temperature/
    response_format/api_key/api_base/max_tokens；user 是渲染后的中文上下文。"""
    fake = _patch_acompletion(monkeypatch, contents=['{"foo": 1}'], tokens=23)
    backend = LiteLLMBackend(
        api_key="sk-test",
        base_url="https://gateway.example.com/v1",
        max_tokens=128,
        temperature=0.0,
    )
    resp = await backend.complete(
        node="hypothesize", state=_PRODUCT_STATE, json_schema=_SIMPLE_SCHEMA
    )
    assert resp.content == '{"foo": 1}'
    assert resp.tokens == 23

    assert len(fake.calls) == 1
    kwargs = fake.calls[0]["kwargs"]
    assert kwargs["model"] == "deepseek/deepseek-flash"
    assert kwargs["temperature"] == 0.0
    assert isinstance(kwargs["timeout"], (int, float)) and kwargs["timeout"] > 0
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["api_key"] == "sk-test"
    assert kwargs["api_base"] == "https://gateway.example.com/v1"
    assert kwargs["max_tokens"] == 128

    msgs = fake.calls[0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == build_system_prompt("hypothesize")
    user = msgs[1]["content"]
    assert "P_MOCK_99" in user
    assert "decision" in user and "HUMAN_REVIEW" in user


async def test_complete_usage_details_extraction_matrix(monkeypatch):
    """usage 拆分：三键齐全 / 只有 total / 缺 total / 无 usage 属性。"""
    cases = [
        (
            types.SimpleNamespace(prompt_tokens=10, completion_tokens=4, total_tokens=14),
            {"input": 10, "output": 4, "total": 14},
            14,
        ),
        (types.SimpleNamespace(total_tokens=23), {"total": 23}, 23),
        (types.SimpleNamespace(prompt_tokens=2, completion_tokens=3), {"input": 2, "output": 3}, 0),
        (None, None, 0),
    ]
    backend = LiteLLMBackend(api_key="sk-test")
    for usage_obj, expected_usage, expected_tokens in cases:
        _patch_acompletion_with_usage(monkeypatch, usage_obj)
        resp = await backend.complete(node="hypothesize", state={}, json_schema={})
        assert resp.usage == expected_usage, (usage_obj, resp.usage)
        assert resp.tokens == expected_tokens


async def test_thinking_config_kwargs_only_when_set(monkeypatch):
    """思考配置显式入口：为 None 不下发（用网关默认），非 None 才进 kwargs。"""
    fake = _patch_acompletion(monkeypatch, contents=['{"foo": 1}', '{"foo": 1}'], tokens=7)

    await LiteLLMBackend(api_key="sk-test").complete(
        node="hypothesize", state=_PRODUCT_STATE, json_schema=_SIMPLE_SCHEMA
    )
    default_kwargs = fake.calls[-1]["kwargs"]
    assert "extra_body" not in default_kwargs
    assert "reasoning_effort" not in default_kwargs

    await LiteLLMBackend(api_key="sk-test", thinking="disabled", reasoning_effort="low").complete(
        node="hypothesize", state=_PRODUCT_STATE, json_schema=_SIMPLE_SCHEMA
    )
    kwargs = fake.calls[-1]["kwargs"]
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert kwargs["reasoning_effort"] == "low"


async def test_complete_network_failure_raises_llm_backend_error(monkeypatch):
    """mock 网络失败：acompletion 抛任意异常 → complete 抛 LLMBackendError（非裸异常）。"""
    async def boom_acompletion(**kwargs):
        raise TimeoutError("connection reset by peer")

    import litellm

    monkeypatch.setattr(litellm, "acompletion", boom_acompletion)
    backend = LiteLLMBackend(api_key="sk-test")
    with pytest.raises(LLMBackendError) as ei:
        await backend.complete(
            node="hypothesize", state=_PRODUCT_STATE, json_schema={}
        )
    assert "调用失败" in str(ei.value)
    assert "connection reset by peer" in str(ei.value)


async def test_complete_cleans_model_content_three_states(monkeypatch):
    """模型文本清洗三态（经公开 ``complete`` 验证）。"""
    clean = '{"decision": "PASS", "confidence": 0.9}'
    fenced = '以下是结果：\n```json\n{"a": 1, "b": [1,2]}\n```\n以上仅供参考。'
    no_brace = "抱歉，我无法以 JSON 输出。"
    fake = _patch_acompletion(
        monkeypatch, contents=[clean, fenced, no_brace, None, ""]
    )
    backend = LiteLLMBackend(api_key="sk-test")
    call = dict(node="hypothesize", state=_PRODUCT_STATE, json_schema={})
    assert (await backend.complete(**call)).content == clean
    assert (await backend.complete(**call)).content == '{"a": 1, "b": [1,2]}'
    assert (await backend.complete(**call)).content == no_brace
    assert (await backend.complete(**call)).content == ""
    assert (await backend.complete(**call)).content == ""
    assert len(fake.calls) == 5


async def test_unknown_node_raises_without_calling_litellm(monkeypatch):
    """未知 node（"bogus"）→ LLMBackendError，且不触发 litellm 调用。"""

    async def should_not_be_called(**kwargs):
        pytest.fail("未知 node 不应触发 litellm 调用")

    import litellm

    monkeypatch.setattr(litellm, "acompletion", should_not_be_called)
    backend = LiteLLMBackend(api_key="sk-test")
    with pytest.raises(LLMBackendError) as ei:
        await backend.complete(node="bogus", state={}, json_schema={})
    assert "未知 node" in str(ei.value)
    assert "hypothesize" in str(ei.value)


async def test_call_structured_llm_schema_fail_then_success_full_chain(monkeypatch):
    """全链路重试闭环：第 1 次非法 JSON → 修正提示回喂 → 第 2 次合法 → 成功。"""
    fake = _patch_acompletion(monkeypatch, contents=[_SCHEMA_BAD, plan_conclude_json()])
    backend = LiteLLMBackend(api_key="sk-test")
    outcome = await call_structured_llm(
        OutputModel=PlanOutput, node="plan", state=_PLAN_STATE, llm=backend
    )
    assert outcome.model is not None
    assert isinstance(outcome.model, PlanOutput)
    assert outcome.model.next_action == "conclude"
    assert outcome.attempts == 2
    assert outcome.error is None
    assert outcome.tokens == 14
    assert len(fake.calls) == 2

    second_user = fake.calls[1]["messages"][1]["content"]
    assert "上一轮输出校验反馈" in second_user
    assert "重新输出" in second_user
    assert '"next_action": "SOMETHING_ELSE"' in second_user
    assert "validation error" in second_user.lower()
    first_user = fake.calls[0]["messages"][1]["content"]
    assert "上一轮输出校验反馈" not in first_user


async def test_call_structured_llm_backend_raise_then_success_full_chain(monkeypatch):
    """第 1 次 mock 网络失败（transport 类）→ 退避后原样重试 → 第 2 次成功恢复。"""
    import litellm

    call_count = {"n": 0}

    async def fake_acompletion(**kwargs):
        call_count["n"] += 1
        fake_acompletion.calls.append(list(kwargs["messages"]))
        if call_count["n"] == 1:
            raise TimeoutError("mock network timeout")
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content=plan_conclude_json()),
                    finish_reason="stop",
                )
            ],
            usage=types.SimpleNamespace(total_tokens=7),
        )

    fake_acompletion.calls = []
    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    backend = LiteLLMBackend(api_key="sk-test")
    outcome = await call_structured_llm(
        OutputModel=PlanOutput, node="plan", state=_PLAN_STATE, llm=backend
    )
    assert call_count["n"] == 2
    assert outcome.model is not None and outcome.model.next_action == "conclude"
    assert outcome.attempts == 2
    assert outcome.error is None
    assert outcome.tokens == 7

    first_user = fake_acompletion.calls[0][1]["content"]
    second_user = fake_acompletion.calls[1][1]["content"]
    assert second_user == first_user
    assert "上一轮输出校验反馈" not in second_user
    assert "输出不满足 JSON Schema" not in second_user


async def test_backend_complete_marks_truncated_on_finish_reason_length(monkeypatch):
    """``finish_reason=="length"`` → ``LLMResponse.truncated=True``。"""
    fake = _patch_acompletion(monkeypatch, contents=['{"partial": "json'], finish_reason="length")
    backend = LiteLLMBackend(api_key="sk-test")
    resp = await backend.complete(
        node="hypothesize", state=_PRODUCT_STATE, json_schema={}
    )
    assert resp.truncated is True
    assert len(fake.calls) == 1
    fake2 = _patch_acompletion(monkeypatch, contents=['{"ok": 1}'], finish_reason="stop")
    resp2 = await backend.complete(
        node="hypothesize", state=_PRODUCT_STATE, json_schema={}
    )
    assert resp2.truncated is False
    assert len(fake2.calls) == 1


async def test_call_structured_llm_truncated_invalid_output_not_retried(monkeypatch):
    """截断（finish_reason=length）且校验失败 → **不重试**（attempts=1 降级）。"""
    fake = _patch_acompletion(
        monkeypatch,
        contents=['{"next_action": "SOME_TRUNCATED'],
        finish_reason="length",
    )
    backend = LiteLLMBackend(api_key="sk-test")
    outcome = await call_structured_llm(
        OutputModel=PlanOutput, node="plan", state=_PLAN_STATE, llm=backend
    )
    assert outcome.model is None
    assert outcome.attempts == 1
    assert outcome.error and "截断" in outcome.error
    assert len(fake.calls) == 1


async def test_call_structured_llm_truncated_but_valid_content_succeeds(monkeypatch):
    """截断但内容恰好合法 → 照常成功（attempts=1，不浪费）。"""
    fake = _patch_acompletion(
        monkeypatch, contents=[plan_conclude_json()], finish_reason="length"
    )
    backend = LiteLLMBackend(api_key="sk-test")
    outcome = await call_structured_llm(
        OutputModel=PlanOutput, node="plan", state=_PLAN_STATE, llm=backend
    )
    assert outcome.model is not None and outcome.model.next_action == "conclude"
    assert outcome.attempts == 1
    assert outcome.error is None
    assert len(fake.calls) == 1


async def test_call_structured_llm_two_invalid_schema_failures(monkeypatch):
    """两次均 schema 校验失败 → model=None、attempts=2、error 非空、不抛异常。"""
    fake = _patch_acompletion(
        monkeypatch, contents=[_SCHEMA_BAD, _SCHEMA_BAD], tokens=5
    )
    backend = LiteLLMBackend(api_key="sk-test")
    outcome = await call_structured_llm(
        OutputModel=PlanOutput, node="plan", state=_PLAN_STATE, llm=backend
    )
    assert outcome.model is None
    assert outcome.attempts == 2
    assert outcome.error
    assert "validation error" in outcome.error.lower()
    assert outcome.tokens == 10
    assert len(fake.calls) == 2


def test_build_user_prompt_hypothesize_readable_and_no_raw_marker():
    """hypothesize user prompt：分节中文上下文含商品事实。"""
    text = build_user_prompt(
        node="hypothesize", state=_PRODUCT_STATE, json_schema=_SIMPLE_SCHEMA
    )
    assert "P_MOCK_99" in text
    assert "复古跑鞋（高仿嫌疑样）" in text
    assert "decision" in text and "HUMAN_REVIEW" in text


def test_build_user_prompt_decide_sections():
    """decide user prompt：假设仪表盘/政策先例/预算分节；证据引用串保真。"""
    state = {
        "hypotheses": [
            {
                "id": "H1",
                "statement": "商家存在系统性规避行为",
                "status": "SUPPORTED",
                "prior": 0.8,
                "posterior": 0.91,
                "evidence_for": ["MERCHANT_HISTORY 5 removals / 3 title-relisting"],
                "evidence_against": [],
            },
            {
                "id": "H2",
                "statement": "普通设计非品牌款",
                "status": "REFUTED",
                "prior": 0.3,
                "posterior": 0.05,
                "evidence_for": [],
                "evidence_against": [],
            },
        ],
        "evidence": [
            {
                "type": "MERCHANT_HISTORY",
                "weight": 0.85,
                "source": "MerchantTool",
                "ref_id": "M_5512",
                "value": "5 removals / 3 title-relisting, credit=38",
                "extra": {"removals": 5, "title": 3},
            },
            {
                "type": "POLICY_REF",
                "weight": 0.9,
                "source": "PolicySearchTool",
                "ref_id": "POLICY_3.2_v2_c1",
                "value": "POLICY_3.2 v2：外观高度模仿知名品牌",
                "extra": {"policy_id": "POLICY_3.2"},
            },
        ],
        "degraded": False,
        "failures": [
            {"step_type": "tools", "severity": "WARN", "reason": "某工具超时"},
        ],
        "budget": {
            "llm_calls": 5,
            "tool_calls": 2,
            "tokens": 1000,
            "limits": {
                "max_llm_calls": 10,
                "max_tool_calls": 20,
            },
        },
    }
    text = build_user_prompt(node="decide", state=state, json_schema=_SIMPLE_SCHEMA)
    assert "H1" in text and "SUPPORTED" in text
    assert "商家存在系统性规避行为" in text
    assert "POLICY_REF" in text
    assert "degraded" in text
    assert "1000" in text
    assert "LLM≤10" in text and "工具≤20" in text


def test_build_user_prompt_plan_tool_catalog_and_feedback():
    """plan user prompt：可用工具目录（入参必填要点）+ 修正反馈回喂分节；空目录兜底。"""
    args_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索描述"},
            "top_k": {"type": "integer", "description": "返回条数"},
        },
        "required": ["query"],
    }
    state = {
        "hypotheses": [],
        "evidence": [
            {
                "type": "PRODUCT_FACT",
                "weight": 0.6,
                "source": "ProductTool",
                "ref_id": "P_1",
                "value": "brand=null, version=3",
                "extra": {"a": 1},
            }
        ],
    }
    catalog = [
        {"name": "PolicySearchTool", "description": "政策条款检索", "args_schema": args_schema}
    ]
    text = build_user_prompt(
        node="plan",
        state=state,
        json_schema=_SIMPLE_SCHEMA,
        tool_catalog=catalog,
        feedbacks=["JSON 校验失败：decision 字段缺失，请补全后重新输出"],
    )
    assert "PolicySearchTool" in text
    assert "query" in text and "top_k" in text
    assert "必填" in text and "可选" in text
    assert "decision 字段缺失" in text
    assert "上一轮输出校验反馈" in text
    empty = build_user_prompt(node="plan", state={}, json_schema={})
    assert "conclude" in empty
    assert "无可用取证工具" in empty


def test_build_user_prompt_empty_state_all_nodes_robust():
    """空 state / 缺键：四个节点都不崩、返回非空文本。"""
    for node in SYSTEM_PROMPTS:
        text = build_user_prompt(node=node, state={}, json_schema={})
        assert isinstance(text, str) and text
        assert "## 输出格式要求" in text
    text = build_user_prompt(
        node="decide", state={"hypotheses": "not-a-list"}, json_schema={}
    )
    assert isinstance(text, str) and text


def test_build_user_prompt_schema_enum_and_required_fields():
    """输出格式要点来自 json_schema：schema 的标题/字段/枚举值/必填可选信息均渲染。"""
    text = build_user_prompt(node="decide", state={}, json_schema=_SIMPLE_SCHEMA)
    assert "DecisionProposal" in text
    assert "decision" in text
    assert "PASS" in text and "REJECT" in text and "HUMAN_REVIEW" in text
    assert "risk_type" in text
    assert "必填" in text and "可选" in text


def test_import_backend_does_not_pull_litellm():
    """import pra.agent.litellm_backend 不 import litellm（延迟 import）。"""
    repo_src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
    )
    code = (
        "import sys\n"
        f"sys.path.insert(0, {repo_src!r})\n"
        "import pra.agent.litellm_backend  # noqa: F401\n"
        "assert 'litellm' not in sys.modules, 'litellm_backend import 拉起了 litellm'\n"
    )
    env = dict(os.environ)
    env.pop("DEEPSEEK_API_KEY", None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
