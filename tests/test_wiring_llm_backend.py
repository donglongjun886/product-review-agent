"""S1 契约：生产 LLM 装配（``pra.wiring.build_llm_backend`` + ``get_production_graph``）。

政策：生产入口必须注入**真实** LLM 后端（``LiteLLMBackend``）；缺 ``DEEPSEEK_API_KEY`` 显式抛
``RuntimeError``，绝不静默回落 scripted 桩。本文件锁死四条：

1. 缺 key（None / 全空白）→ ``RuntimeError``，消息含 ``DEEPSEEK_API_KEY``；
2. 模型名 / base_url / 凭据逐项来自 ``Settings``（字段名与 env 键映射）；
3. ``tools`` 真的传进后端工具目录 —— plan 节点 prompt 靠它渲染，漏传会让真模型看到
   "无可用工具" 而规划环节静默废掉；
4. 生产图把同一份后端与同一份工具列表注入 ``build_agent_graph``（删 ``llm=`` / ``tools=``
   立刻变红）。

**为什么不用 ``wiring.build_llm_backend`` 而是导入期捕获的真函数**：``tests/conftest.py`` 的
autouse fixture ``_production_entry_uses_scripted_llm`` 会把 ``pra.wiring.build_llm_backend`` 钉成
返回 ``ScriptedLLMBackend()`` 的桩（生产图用例需要无凭据可跑）。fixture 只在用例执行前后生效，
**模块导入（collection 期）早于它**，故此处捕获到的是真实现 ``REAL_BUILD_LLM_BACKEND``；用例里
再 monkeypatch 回真身（第 4 条反过来钉成自己的假函数）。

零网络 / 零真钱 / 不依赖本机真实 key：只构造后端（``LiteLLMBackend.__init__`` 不触发网络，
litellm 是延迟 import），``Settings`` 一律钉成 ``_env_file=None`` 或 ``tmp_path`` 造的临时 env
文件，绝不读仓库根 ``.env``；断言只在假 key 上做等值比较。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers import make_case

from pra import wiring
from pra.agent.guardrails.llm_shell import LLMResponse
from pra.agent.litellm_backend import LiteLLMBackend
from pra.agent.scripted_llm import ScriptedLLMBackend
from pra.api.service import run_review
from pra.infra import db as infra_db
from pra.infra.db import Settings as RealSettings
from pra.tools import build_tools

# 导入期捕获（早于任何 fixture 执行）→ 真实现，而非 conftest 钉的桩（见模块 docstring）。
from pra.wiring import build_llm_backend as REAL_BUILD_LLM_BACKEND

# 假凭据：只用于断言"值来自 Settings"，不是真实 key，也不会发请求。
_FAKE_KEY = "sk-test-not-real"
_MODEL_FROM_ENV = "deepseek/deepseek-reasoner"  # 刻意≠LiteLLMBackend 默认值，防"碰巧相等"
_BASE_URL_FROM_ENV = "https://example.invalid"


def _write_env_file(tmp_path: Path, text: str) -> str:
    """在 ``tmp_path`` 造临时 env 文件并返回路径（不碰仓库根 .env）。"""
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return str(path)


def _restore_real_build_llm_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """把 ``pra.wiring.build_llm_backend`` 还原成导入期捕获的真实现（绕过 conftest 的桩）。"""
    assert REAL_BUILD_LLM_BACKEND.__module__ == "pra.wiring", (
        "导入期捕获到的不是 pra.wiring 的模块级函数 —— conftest 的 pin 早于本模块导入，"
        "绕过方案失效，需改用 wiring.__dict__ 原始对象或 inspect.unwrap"
    )
    monkeypatch.setattr(wiring, "build_llm_backend", REAL_BUILD_LLM_BACKEND)


def _pin_settings(monkeypatch: pytest.MonkeyPatch, factory) -> None:
    """把 ``Settings`` 的查找点钉成 ``factory``（``build_llm_backend`` 的配置唯一来源）。

    当前唯一查找点是 ``wiring`` 的模块全局属性（``from pra.infra.db import Settings``）；同时钉
    ``pra.infra.db`` 的模块属性属防御：实现若改成函数内惰性 import，本用例仍读到钉死的配置，
    否则会悄悄去读本机真实 ``.env`` 而失去确定性。
    """
    monkeypatch.setattr(wiring, "Settings", factory, raising=False)
    monkeypatch.setattr(infra_db, "Settings", factory, raising=False)


def test_missing_api_key_raises_runtime_error(monkeypatch):
    """契约 1：key 缺失或全空白 → ``RuntimeError``（消息含 DEEPSEEK_API_KEY），不回落桩。"""
    _restore_real_build_llm_backend(monkeypatch)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    _pin_settings(monkeypatch, lambda: RealSettings(_env_file=None))

    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        REAL_BUILD_LLM_BACKEND()

    # 全空白同样视为未配置（不得 strip 后当成有效凭据去调真网关）。
    monkeypatch.setenv("DEEPSEEK_API_KEY", "   ")
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        REAL_BUILD_LLM_BACKEND()


def test_backend_model_and_credentials_come_from_settings(monkeypatch, tmp_path):
    """契约 2：后端类型与 model / base_url / api_key 逐项来自 ``Settings``（只构造，不调 API）。"""
    _restore_real_build_llm_backend(monkeypatch)
    # 真实环境变量优先级高于 env_file：先清空，保证取值唯一来自下面这个临时文件。
    for key in ("DEEPSEEK_API_KEY", "DEEPSEEK_MODEL", "DEEPSEEK_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    env_file = _write_env_file(
        tmp_path,
        f"DEEPSEEK_API_KEY={_FAKE_KEY}\n"
        f"DEEPSEEK_MODEL={_MODEL_FROM_ENV}\n"
        f"DEEPSEEK_BASE_URL={_BASE_URL_FROM_ENV}\n",
    )
    _pin_settings(monkeypatch, lambda: RealSettings(_env_file=env_file))

    backend = REAL_BUILD_LLM_BACKEND()

    assert isinstance(backend, LiteLLMBackend)
    assert backend.model == _MODEL_FROM_ENV
    assert backend.base_url == _BASE_URL_FROM_ENV
    assert backend._api_key == _FAKE_KEY


def test_tools_reach_backend_tool_catalog(monkeypatch, tmp_path):
    """契约 3：``tools=`` 真的进后端工具目录（plan 渲染依赖），逐条对应且非空。"""
    _restore_real_build_llm_backend(monkeypatch)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    env_file = _write_env_file(tmp_path, f"DEEPSEEK_API_KEY={_FAKE_KEY}\n")
    _pin_settings(monkeypatch, lambda: RealSettings(_env_file=env_file))

    tools = build_tools()
    backend = REAL_BUILD_LLM_BACKEND(tools=tools)

    assert isinstance(backend, LiteLLMBackend)
    assert len(tools) > 0
    # 漏传 tools → 空目录（"无可用工具"），本断言立刻变红。
    catalog = backend._tool_catalog
    assert len(catalog) == len(tools)
    assert [entry["name"] for entry in catalog] == [tool.name for tool in tools]
    for entry in catalog:
        assert set(entry) == {"name", "description", "args_schema"}
        assert entry["description"]  # 空说明 = 真模型看不到工具用途
        assert isinstance(entry["args_schema"], dict)


def test_production_graph_injects_backend_and_same_tools(monkeypatch):
    """契约 4：生产图注入工厂返回的后端 + **同一份**工具列表（删任一转发参数即变红）。"""
    sentinel_backend = object()
    sentinel_graph = object()
    factory_calls: dict[str, object] = {}
    graph_kwargs: dict[str, object] = {}

    def fake_build_llm_backend(*, tools=None):
        factory_calls["tools"] = tools
        return sentinel_backend

    def fake_build_agent_graph(*args, **kwargs):
        graph_kwargs.update(kwargs)
        return sentinel_graph

    monkeypatch.setattr(wiring, "build_llm_backend", fake_build_llm_backend)
    monkeypatch.setattr(wiring, "build_agent_graph", fake_build_agent_graph)
    monkeypatch.setattr(wiring, "_graph", None)

    graph = wiring.get_production_graph()

    assert graph is sentinel_graph
    assert "llm" in graph_kwargs, "生产装配未把 llm= 交给 build_agent_graph（缺它会直接 TypeError）"
    assert graph_kwargs["llm"] is sentinel_backend
    assert "tools" in graph_kwargs, "生产装配未把 tools= 交给 build_agent_graph"
    factory_tools = factory_calls.get("tools")
    assert isinstance(factory_tools, list) and factory_tools, "LLM 工厂未收到非空工具列表"
    assert graph_kwargs["tools"] is factory_tools


async def test_production_graph_runs_on_injected_backend(monkeypatch):
    """行为守护：生产图**运行时**真的经注入后端调 LLM —— 删 ``llm=`` 或节点读不到注入后端即变红。

    与上一条的区别：上一条只锁 ``build_agent_graph`` 收到的 kwargs。本用例真的跑一遍生产图
    （InMemory 工具世界 + 记录调用的桩后端），断言注入的后端被调用过。
    """
    nodes: list[str] = []

    class _RecordingStub(ScriptedLLMBackend):
        async def complete(self, *, node: str, messages: list, json_schema: dict) -> LLMResponse:
            nodes.append(node)
            return await super().complete(
                node=node, messages=messages, json_schema=json_schema
            )

    recorder = _RecordingStub()
    monkeypatch.setattr(wiring, "build_llm_backend", lambda *, tools=None: recorder)
    monkeypatch.setattr(wiring, "_graph", None)

    result = await run_review(make_case(case_id="CASE_S1_BEHAVIOR"))

    assert result.review_decision is not None
    assert nodes, "生产图运行时没有调用注入的后端 —— 装配丢了 llm=，或节点读的不是注入的后端"
