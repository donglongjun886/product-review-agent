"""Phase 3（real LLM）mock 单测 —— LiteLLMBackend + llm_prompts 渲染层。

无网络、无 API key、不依赖 .env：所有 litellm 调用都被 monkeypatch 到
``litellm.acompletion`` 的本地替身；真实调用留 scripts/run_evaluation_real.py 人工实测。

覆盖分组：
- A. LiteLLMBackend 直接测：构造（model/name/api_key 来源）/ tools 目录提取 /
  mock acompletion 成功路径（content/tokens/kwargs）/ mock 网络失败 → LLMBackendError /
  _clean_json_text 三态 / 未知 node 不触发调用 / call_structured_llm 全链路重试闭环；
- B. llm_prompts 渲染纯测：四个节点 system prompt 关键词 / user prompt 人读上下文
  （无裸 __STATE__ 泄漏）/ 空 state 防御 / json_schema 的 enum/必填要点；
- C. 确定性回归守护：import pra.agent.litellm_backend 不拉起 litellm（延迟 import 是
  "测试无网络"的前提 —— litellm 被 import 时会尝试拉远程 cost map）。

实现约定（与 src 一致）：顶部 ``from __future__ import annotations``；import 一律
``pra.*``；中文注释；asyncio_mode="auto"（pytest-asyncio），``async def test_*`` 直接
写即可。backend 注入的恢复：本仓库 conftest.py 的 autouse fixture
``_reset_llm_backend`` 已对 ``llm_shell._backend`` / ``_default_backend`` 前后快照还原，
本文件仍对 set_llm_backend 的测试加 try/finally 双保险。

网络自检说明（本文件如何保证不烧 key / 不走真实调用）：
- 不 mock litellm 模块整体 import：litellm_backend 是**延迟 import** litellm，测试只需
  monkeypatch ``litellm.acompletion`` 属性 —— 在 import litellm（进程内首次）前设置
  ``LITELLM_LOCAL_MODEL_COST_MAP=True`` 关闭 litellm 启动时拉取远程 cost map 的联网；
- 每个触发 complete 的测试都先 monkeypatch acompletion（耗尽即 pytest.fail 哨兵，防
  "多出的真实调用"）；无 key / 未知 node 路径在 litellm import 之前就抛 LLMBackendError。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types

import pytest
from helpers import plan_conclude_json  # 既有测试替身 JSON（tests/helpers.py）
from pydantic import BaseModel, Field

from pra.agent.guardrails.llm_shell import (
    LLMBackendError,
    call_structured_llm,
    set_llm_backend,
)
from pra.agent.guardrails.schemas import PlanOutput
from pra.agent.litellm_backend import LiteLLMBackend
from pra.agent.llm_prompts import (
    SYSTEM_PROMPTS,
    build_system_prompt,
    build_user_prompt,
)

# litellm 被 import 时默认会尝试拉远程 model cost map（联网、慢）；关掉它保证测试进程
# 全程离线 —— 本文件首个 import litellm 发生在测试函数内（延迟 import，同 src 惯例）。
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

# ---------------------------------------------------------------------------
# 测试数据（手写 dict 状态 —— 与 nodes/*.py _build_messages 的 __STATE__ JSON 同构，
# 但不用 domain 对象：LiteLLMBackend 只解析 __STATE__ JSON，喂 dict 即可）
# ---------------------------------------------------------------------------

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
    "screening_signals": [
        {"name": "KEYWORD", "result": "HIT", "score": 0.9},
    ],
}

_HYPOTHESIZE_MESSAGES = [
    {"role": "system", "content": "（节点 system，LiteLLMBackend 渲染时会被替换）"},
    {
        "role": "user",
        "content": "__STATE__ " + json.dumps(_PRODUCT_STATE, ensure_ascii=False),
    },
]

# plan 最小 state：无假设/无证据 → plan 渲染走"无可用取证工具 → conclude"兜底分支。
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

_PLAN_MESSAGES = [
    {"role": "system", "content": "plan sys"},
    {"role": "user", "content": "__STATE__ " + json.dumps(_PLAN_STATE, ensure_ascii=False)},
]

# 合法 JSON 但不符合 PlanOutput schema（next_action 超词表）→ pydantic ValidationError。
_SCHEMA_BAD = '{"next_action": "SOMETHING_ELSE", "tools": [], "rationale": "x"}'

# 简单 OutputModel JSON Schema（decide 侧；enum + required 供 schema 要点断言）。
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

# ---------------------------------------------------------------------------
# litellm.acompletion 替身（本文件所有真实 litellm 调用都被它替换）
# ---------------------------------------------------------------------------


def _make_fake_acompletion(contents: list, tokens: int = 7, finish_reason: str = "stop"):
    """返回可 monkeypatch 到 ``litellm.acompletion`` 的 async 替身，记录每次调用。

    ``contents`` 为逐个返回的 content 文本列表；耗尽后再被调用 → pytest.fail
    （哨兵：说明存在未 mock 的真实调用泄漏）。记录 ``calls`` = 每次的 kwargs/messages。
    ``finish_reason`` 为每次响应的 finish_reason（"stop"/"length"；P2-15 截断测试用）。
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
    import litellm  # 延迟 import：本文件顶部不 import，避免收集期拉起 litellm

    fake = _make_fake_acompletion(contents, tokens=tokens, finish_reason=finish_reason)
    monkeypatch.setattr(litellm, "acompletion", fake)
    return fake


# ---------------------------------------------------------------------------
# A. LiteLLMBackend 直接测
# ---------------------------------------------------------------------------


def test_constructor_default_model_and_name_format():
    """默认 model="deepseek/deepseek-chat"；name = f"litellm-{model}"（构造无 key 不抛）。"""
    backend = LiteLLMBackend()
    assert backend.model == "deepseek/deepseek-chat"
    assert backend.name == "litellm-deepseek/deepseek-chat"
    # 自定义 model 时 name 跟着变（Protocol.name 只做审计/展示标识）
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
    assert backend._api_key == "sk-explicit"  # 入参覆盖环境变量
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
    backend = LiteLLMBackend()  # 无 key 构造不抛（便于无 key 环境 import/渲染调试）
    with pytest.raises(LLMBackendError) as ei:
        await backend.complete(node="hypothesize", messages=[], json_schema={})
    message = str(ei.value)
    assert "DEEPSEEK_API_KEY" in message  # 提示告诉用户该设哪个环境变量
    assert "未配置" in message


def test_tool_catalog_extracted_from_tool_objects():
    """tools 目录提取：name/description/args_schema；无 name 的坏工具被跳过。"""

    class _ImgArgs(BaseModel):
        image_url: str = Field(description="待比对的图片 URL")
        top_k: int = Field(default=3, description="返回前 k 张相似图")

    class _MerchantArgs(BaseModel):
        merchant_id: str = Field(description="商家 ID")

    class _FakeImageTool:  # 协议只需 name/description/args_model（结构类型）
        name = "ImageAnalysisTool"
        description = "图片外观相似度比对取证工具"
        args_model = _ImgArgs

    class _FakeMerchantTool:
        name = "MerchantTool"
        description = "商家历史行为取证"
        args_model = _MerchantArgs

    class _BrokenTool:  # 协议缺 name → 构造时跳过，不影响其余工具
        name = ""
        description = "broken"
        args_model = None

    backend = LiteLLMBackend(
        api_key="sk-test",
        tools=[_FakeImageTool(), _FakeMerchantTool(), _BrokenTool()],
    )
    catalog = backend._tool_catalog
    assert [t["name"] for t in catalog] == ["ImageAnalysisTool", "MerchantTool"]
    img = catalog[0]
    assert img["description"] == "图片外观相似度比对取证工具"
    # args_schema 由 pydantic model_json_schema() 生成：含入参字段名与必填信息
    assert "image_url" in img["args_schema"]["properties"]
    assert "top_k" in img["args_schema"]["properties"]
    assert img["args_schema"]["required"] == ["image_url"]
    # None/空 tools → 空目录（plan 渲染走"无可用工具"分支）
    assert LiteLLMBackend(api_key="sk-test", tools=None)._tool_catalog == []
    assert LiteLLMBackend(api_key="sk-test", tools=[])._tool_catalog == []


async def test_complete_success_content_tokens_and_kwargs(monkeypatch):
    """mock acompletion 成功路径：content/tokens 正确；kwargs 含 model/temperature/
    response_format/api_key/api_base/max_tokens；user 是渲染后的中文上下文（无 __STATE__）。"""
    fake = _patch_acompletion(monkeypatch, contents=['{"foo": 1}'], tokens=23)
    backend = LiteLLMBackend(
        api_key="sk-test",
        base_url="https://gateway.example.com/v1",
        max_tokens=128,
        temperature=0.0,
    )
    resp = await backend.complete(
        node="hypothesize", messages=_HYPOTHESIZE_MESSAGES, json_schema=_SIMPLE_SCHEMA
    )
    # content == 模型返回的干净 JSON 文本；tokens == usage.total_tokens
    assert resp.content == '{"foo": 1}'
    assert resp.tokens == 23

    assert len(fake.calls) == 1  # 恰好一次调用（无额外泄漏）
    kwargs = fake.calls[0]["kwargs"]
    assert kwargs["model"] == "deepseek/deepseek-chat"
    assert kwargs["temperature"] == 0.0
    assert kwargs["timeout"] == 60.0
    assert kwargs["response_format"] == {"type": "json_object"}  # 强制 JSON 对象输出
    assert kwargs["api_key"] == "sk-test"  # 显式 key 透传给 litellm
    assert kwargs["api_base"] == "https://gateway.example.com/v1"
    assert kwargs["max_tokens"] == 128

    # 发给模型的 messages：system=完整约束中文指令；user=渲染上下文
    msgs = fake.calls[0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "风险假设生成器" in msgs[0]["content"]  # SYSTEM_PROMPTS["hypothesize"] 开头
    user = msgs[1]["content"]
    assert "商品 ID：P_MOCK_99" in user  # __STATE__ 里的商品事实被渲染出来
    assert "KEYWORD" in user  # 机审信号入上下文
    assert "## 输出格式要求" in user  # Schema 要点附加在 user 尾部
    assert "__STATE__" not in user  # 不发裸 __STATE__ 标记给真实模型


async def test_complete_network_failure_raises_llm_backend_error(monkeypatch):
    """mock 网络失败：acompletion 抛任意异常 → complete 抛 LLMBackendError（非裸异常）。"""

    async def boom_acompletion(**kwargs):
        raise TimeoutError("connection reset by peer")

    import litellm

    monkeypatch.setattr(litellm, "acompletion", boom_acompletion)
    backend = LiteLLMBackend(api_key="sk-test")
    with pytest.raises(LLMBackendError) as ei:
        await backend.complete(
            node="hypothesize", messages=_HYPOTHESIZE_MESSAGES, json_schema={}
        )
    assert "调用失败" in str(ei.value)
    assert "connection reset by peer" in str(ei.value)


def test_clean_json_text_three_states():
    """_clean_json_text 三态：干净 JSON 原样 / 围栏+杂文本截首{到末} / 无 { 原样返回。"""
    backend = LiteLLMBackend(api_key="sk-test")  # 静态方法，实例/类调用皆可
    # 1) 干净 JSON 原样返回
    clean = '{"decision": "PASS", "confidence": 0.9}'
    assert backend._clean_json_text(clean) == clean
    # 2) ```json 围栏 + 前后杂文本 → 截取首个 { 到末个 }
    fenced = '以下是结果：\n```json\n{"a": 1, "b": [1,2]}\n```\n以上仅供参考。'
    assert backend._clean_json_text(fenced) == '{"a": 1, "b": [1,2]}'
    # 3) 无 { → 原样返回（交由 llm_shell model_validate_json 强校验 + 回喂重试）
    no_brace = "抱歉，我无法以 JSON 输出。"
    assert backend._clean_json_text(no_brace) == no_brace
    # 防御：None/空串不抛
    assert backend._clean_json_text(None) == ""
    assert backend._clean_json_text("") == ""


async def test_unknown_node_raises_without_calling_litellm(monkeypatch):
    """未知 node（"bogus"）→ LLMBackendError，且不触发 litellm 调用。"""

    async def should_not_be_called(**kwargs):
        pytest.fail("未知 node 不应触发 litellm 调用")

    import litellm

    monkeypatch.setattr(litellm, "acompletion", should_not_be_called)
    backend = LiteLLMBackend(api_key="sk-test")  # 有 key，确保失败源于未知 node
    with pytest.raises(LLMBackendError) as ei:
        await backend.complete(node="bogus", messages=[], json_schema={})
    assert "未知 node" in str(ei.value)
    assert "hypothesize" in str(ei.value)  # 提示里带合法 node 词表


async def test_call_structured_llm_schema_fail_then_success_full_chain(monkeypatch):
    """全链路重试闭环：第 1 次非法 JSON → 修正提示回喂 → 第 2 次合法 → 成功。

    验证 LiteLLMBackend（真实渲染路径）+ llm_shell 的"校验失败重试 1 次"协作：
    修正提示被 _collect_feedbacks 收进 user 尾部回喂模型，不破坏重试。
    P2-15：回喂内容必须含**第 1 次非法输出原文**（模型第 2 次能"看到"自己上一版
    输出去修正），不只是 ValidationError 文本。
    """
    fake = _patch_acompletion(monkeypatch, contents=[_SCHEMA_BAD, plan_conclude_json()])
    backend = LiteLLMBackend(api_key="sk-test")
    set_llm_backend(backend)
    try:
        outcome = await call_structured_llm(
            OutputModel=PlanOutput, node="plan", messages=_PLAN_MESSAGES
        )
    finally:
        set_llm_backend(None)  # 恢复默认桩（conftest autouse 亦有兜底）
    assert outcome.model is not None
    assert isinstance(outcome.model, PlanOutput)
    assert outcome.model.next_action == "conclude"
    assert outcome.attempts == 2
    assert outcome.error is None
    assert outcome.tokens == 14  # 两次响应 tokens 累计（每次 7）
    assert len(fake.calls) == 2  # 恰好两次（重试 1 次）

    # 第 2 次发给模型的 user 里带"上一轮输出校验反馈"分节 —— 修正提示被回喂（非空转）
    second_user = fake.calls[1]["messages"][1]["content"]
    assert "上一轮输出校验反馈" in second_user
    assert "重新输出" in second_user
    # P2-15：回喂含第 1 次非法输出原文（next_action 超词表的 _SCHEMA_BAD 全文）
    assert '"next_action": "SOMETHING_ELSE"' in second_user
    assert "validation error" in second_user.lower()  # 校验错误文本也一并回喂
    # 第 1 次 user 无反馈分节（没有可回喂的上一轮错误）
    first_user = fake.calls[0]["messages"][1]["content"]
    assert "上一轮输出校验反馈" not in first_user


async def test_call_structured_llm_backend_raise_then_success_full_chain(monkeypatch):
    """第 1 次 mock 网络失败（transport 类）→ 退避后原样重试 → 第 2 次成功恢复。

    P2-15：transport 失败没有可"修正"的输出 —— 第 2 次请求**不追加 schema 修正
    文案**（不再误导模型"你的输出不满足 Schema"，实为网络超时），user 与第 1 次
    完全一致（纯重试）。
    """
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
    set_llm_backend(backend)
    try:
        outcome = await call_structured_llm(
            OutputModel=PlanOutput, node="plan", messages=_PLAN_MESSAGES
        )
    finally:
        set_llm_backend(None)
    assert call_count["n"] == 2  # 失败后重试了 1 次
    assert outcome.model is not None and outcome.model.next_action == "conclude"
    assert outcome.attempts == 2
    assert outcome.error is None
    assert outcome.tokens == 7  # 第 1 次后端异常不计 tokens

    # P2-15：transport 重试不加 schema 修正文案 —— 两次请求的 user 内容一致（纯重试）
    first_user = fake_acompletion.calls[0][1]["content"]
    second_user = fake_acompletion.calls[1][1]["content"]
    assert second_user == first_user
    assert "上一轮输出校验反馈" not in second_user
    assert "输出不满足 JSON Schema" not in second_user


async def test_backend_complete_marks_truncated_on_finish_reason_length(monkeypatch):
    """P2-15：finish_reason=="length" → ``LLMResponse.truncated=True``（供 shell 分类）。"""
    fake = _patch_acompletion(monkeypatch, contents=['{"partial": "json'], finish_reason="length")
    backend = LiteLLMBackend(api_key="sk-test")
    resp = await backend.complete(
        node="hypothesize", messages=_HYPOTHESIZE_MESSAGES, json_schema={}
    )
    assert resp.truncated is True
    assert len(fake.calls) == 1
    # 对照：finish_reason="stop"（默认）→ truncated=False
    fake2 = _patch_acompletion(monkeypatch, contents=['{"ok": 1}'], finish_reason="stop")
    resp2 = await backend.complete(
        node="hypothesize", messages=_HYPOTHESIZE_MESSAGES, json_schema={}
    )
    assert resp2.truncated is False
    assert len(fake2.calls) == 1


async def test_call_structured_llm_truncated_invalid_output_not_retried(monkeypatch):
    """P2-15：截断（finish_reason=length）且校验失败 → **不重试**（attempts=1 降级）。

    只 mock 一次响应：若 shell 仍按"普通 schema 校验失败"重试 → 哨兵 pytest.fail
    拦截第二次 acompletion（省一次大概率无效的全量调用）。
    """
    fake = _patch_acompletion(
        monkeypatch,
        contents=['{"next_action": "SOME_TRUNCATED'],  # 截断 + 非法 JSON
        finish_reason="length",
    )
    backend = LiteLLMBackend(api_key="sk-test")
    set_llm_backend(backend)
    try:
        outcome = await call_structured_llm(
            OutputModel=PlanOutput, node="plan", messages=_PLAN_MESSAGES
        )
    finally:
        set_llm_backend(None)
    assert outcome.model is None
    assert outcome.attempts == 1  # 截断按 transport 类：不重试
    assert outcome.error and "截断" in outcome.error
    assert len(fake.calls) == 1  # 恰好一次（没有白烧第 2 次）


async def test_call_structured_llm_truncated_but_valid_content_succeeds(monkeypatch):
    """P2-15：截断但内容恰好合法 → 照常成功（attempts=1，不浪费）。"""
    fake = _patch_acompletion(
        monkeypatch, contents=[plan_conclude_json()], finish_reason="length"
    )
    backend = LiteLLMBackend(api_key="sk-test")
    set_llm_backend(backend)
    try:
        outcome = await call_structured_llm(
            OutputModel=PlanOutput, node="plan", messages=_PLAN_MESSAGES
        )
    finally:
        set_llm_backend(None)
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
    set_llm_backend(backend)
    try:
        outcome = await call_structured_llm(
            OutputModel=PlanOutput, node="plan", messages=_PLAN_MESSAGES
        )
    finally:
        set_llm_backend(None)
    assert outcome.model is None
    assert outcome.attempts == 2
    assert outcome.error  # 最后一次 ValidationError 文本（供节点写 failure.reason）
    assert "validation error" in outcome.error.lower()
    assert outcome.tokens == 10  # transport 成功两次都记 token；校验失败在 llm_shell
    assert len(fake.calls) == 2


# ---------------------------------------------------------------------------
# B. llm_prompts 渲染纯测
# ---------------------------------------------------------------------------


def test_build_system_prompt_keywords_for_all_four_nodes():
    """四个节点的 system prompt 均非空且含关键约束关键词（稳定子串）。"""
    expected_keywords = {
        "hypothesize": ["低风险", "先验"],  # 低风险/正常假设 + prior 语义
        "plan": ["next_action", "新证据", "conclude"],
        "reevaluate": ["UNRESOLVED", "SUPPORTED"],  # 证据不足 ≠ 证伪
        "decide": ["HUMAN_REVIEW", "PASS", "REJECT"],
    }
    assert set(SYSTEM_PROMPTS) == set(expected_keywords)
    for node, keywords in expected_keywords.items():
        prompt = build_system_prompt(node)
        assert prompt and len(prompt) > 100
        for kw in keywords:
            assert kw in prompt, f"node={node} 的 system prompt 缺关键词 {kw!r}"
    # 未知 node → ValueError（调用方先查词表，complete 前置校验同口径）
    with pytest.raises(ValueError):
        build_system_prompt("bogus")


def test_build_user_prompt_hypothesize_readable_and_no_raw_marker():
    """hypothesize user prompt：分节中文上下文含商品事实/图片/信号值；无裸 __STATE__。"""
    text = build_user_prompt(
        node="hypothesize", state=_PRODUCT_STATE, json_schema=_SIMPLE_SCHEMA
    )
    assert "## 一、案件与商品事实" in text
    assert "商品 ID：P_MOCK_99" in text  # 字段值渲染而非裸 JSON
    assert "复古跑鞋（高仿嫌疑样）" in text
    assert "## 二、商品图片" in text and "图1" in text
    assert "## 三、机审信号" in text and "KEYWORD" in text
    assert "## 输出格式要求" in text
    assert "__STATE__" not in text  # 不把 __STATE__ 标记泄漏给真实模型


def test_build_user_prompt_decide_sections():
    """decide user prompt：假设仪表盘/政策先例/预算分节；证据引用串保真。"""
    state = {
        "hypotheses": [
            {
                "id": "H1",
                "statement": "外观高度模仿某品牌经典款",
                "status": "SUPPORTED",
                "prior": 0.8,
                "posterior": 0.91,
                "evidence_for": ["IMAGE_SIMILARITY sim=0.91, match=某品牌"],
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
                "type": "IMAGE_SIMILARITY",
                "weight": 0.91,
                "source": "ImageAnalysisTool",
                "ref_id": "https://cdn/img1.jpg",
                "value": "sim=0.91, match=某品牌经典款",
                "extra": {},
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
                "max_tokens": 5000,
                "max_latency_ms": 60000,
            },
        },
    }
    text = build_user_prompt(node="decide", state=state, json_schema=_SIMPLE_SCHEMA)
    assert "## 一、假设仪表盘" in text
    assert "H1 | status=SUPPORTED" in text
    assert "外观高度模仿某品牌经典款" in text  # hypothesis.statement 值渲染
    assert "## 二、可引用政策与先例" in text
    assert "type=POLICY_REF" in text  # 证据 type 保真（REJECT 引用来源）
    assert "## 四、运行状态" in text and "degraded=False" in text
    assert "## 五、预算摘要" in text and "LLM 调用 5 次" in text
    assert "__STATE__" not in text


def test_build_user_prompt_plan_tool_catalog_and_feedback():
    """plan user prompt：可用工具目录（入参必填要点）+ 修正反馈回喂分节；空目录兜底。"""
    args_schema = {
        "type": "object",
        "properties": {
            "image_url": {"type": "string", "description": "图片 URL"},
            "top_k": {"type": "integer", "description": "返回条数"},
        },
        "required": ["image_url"],
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
        "investigation_queue": [{"q": "是否仿牌？", "priority": 1, "status": "OPEN"}],
    }
    catalog = [
        {"name": "ImageAnalysisTool", "description": "图片外观相似度比对", "args_schema": args_schema}
    ]
    text = build_user_prompt(
        node="plan",
        state=state,
        json_schema=_SIMPLE_SCHEMA,
        tool_catalog=catalog,
        feedbacks=["JSON 校验失败：decision 字段缺失，请补全后重新输出"],
    )
    assert "## 六、可用取证工具目录" in text
    assert "ImageAnalysisTool" in text
    assert "image_url（必填）" in text  # Schema required → 必填标记
    assert "top_k（可选）" in text
    assert "上一轮输出校验反馈" in text  # llm_shell 修正提示被回喂到 user 尾部
    assert "decision 字段缺失" in text
    assert "__STATE__" not in text
    # 空目录 → plan 上下文提示"无可用工具 → conclude"
    empty = build_user_prompt(node="plan", state={}, json_schema={})
    assert "无可用取证工具" in empty and "conclude" in empty


def test_build_user_prompt_empty_state_all_nodes_robust():
    """空 state / 缺键：四个节点都不崩、返回非空文本（防御式降级口径）。"""
    for node in SYSTEM_PROMPTS:
        text = build_user_prompt(node=node, state={}, json_schema={})
        assert isinstance(text, str) and text
        assert "## 输出格式要求" in text  # Schema 要点段始终在
    # 畸形 state（hypotheses 不是列表）也不崩
    text = build_user_prompt(
        node="decide", state={"hypotheses": "not-a-list"}, json_schema={}
    )
    assert "（无假设）" in text


def test_build_user_prompt_schema_enum_and_required_fields():
    """输出格式要点来自 json_schema：含枚举可取值与必填字段名标记。"""
    text = build_user_prompt(node="decide", state={}, json_schema=_SIMPLE_SCHEMA)
    assert "顶层对象：DecisionProposal" in text
    assert "decision（必填）" in text
    assert "字符串枚举，只能取：PASS | REJECT | HUMAN_REVIEW" in text
    assert "risk_type（可选）" in text


# ---------------------------------------------------------------------------
# C. 确定性回归守护（无网络 / 无 key）
# ---------------------------------------------------------------------------


def test_import_backend_does_not_pull_litellm():
    """import pra.agent.litellm_backend 不 import litellm（延迟 import 契约守护）。

    本仓库 .venv 里 litellm 1.100.0 被 import 时会尝试拉远程 model cost map（联网）：
    若有人把 litellm_backend 的 ``import litellm`` 提到模块顶层，会让每个测试收集期都
    产生网络请求 —— 本测试用**全新子进程**验证模块 import 不拉起 litellm
    （子进程隔离保证与 pytest 进程内是否已 import litellm 无关）。
    """
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
    env.pop("DEEPSEEK_API_KEY", None)  # 子进程也不带 key（防任何隐式联网读 key）
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,  # 失败由下方 returncode 断言给出可读 stderr
    )
    assert result.returncode == 0, result.stderr
