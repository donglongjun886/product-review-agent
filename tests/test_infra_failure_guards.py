"""B 段失败路径守护四条（**净新增**：旧 ``tools_node`` 失败分支此前零覆盖）。

守护 1：工具 ``call`` 抛 infra 异常 → 恰好重试 1 次后**裸抛**，且不写 ``severity=warn`` 的
failure。它防的回归 = 旧行为吞掉异常、转 warn 并 ``continue``（或重试次数变成 0 / 多次）。

守护 2：业务失败（``ok=False``）→ 仍写一条 warn failure、记 error record、记账并继续本 visit。
它防的回归 = 把业务失败也当 infra 异常上抛，或不再记账 / 不再清空待办。本守护是**防回归**，
不证明新行为（新旧实现在本分支上一致）。

守护 3：``run_and_persist`` 遇图执行异常 → 先落 ``decision=HUMAN_REVIEW`` 的 review_result 与
一条 ``step_type="infra_error"`` 的 review_trace 行（承载 R6 归因码），再原样上抛，run 保持
RUNNING。它防的回归 = 旧行为只有裸抛：库中既无显式终态也无归因，「悬空 run」无法解释。
"""

from __future__ import annotations

import json
import socket
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import pytest
from helpers import ev, make_case
from sqlalchemy import text

from pra.agent import tools_node as tools_node_mod
from pra.agent.guardrails.errors import SEV_WARN
from pra.agent.tools_node import make_tools_node
from pra.domain.models import Budget, Evidence
from pra.infra import persist_service
from pra.infra.db import Settings, get_sessionmaker
from pra.infra.persist_service import run_and_persist
from pra.tools.base import ToolArgs, ToolContext, ToolResult


class _EmptyArgs(ToolArgs):
    """空入参模型 —— tools_node 用 ``model_validate({})`` 校验 plan 给的 args，恒通过。"""


class _InfraFailureTool:
    """工具替身：``call`` 恒抛 RuntimeError（模拟工具 infra 故障）。"""

    name = "InfraFailureTool"
    description = "注入 infra 异常的测试替身工具"
    args_model = _EmptyArgs

    def __init__(self) -> None:
        self.calls = 0  # call 被调用次数（断言重试恰好 1 次）

    async def call(self, args: ToolArgs, ctx: ToolContext) -> ToolResult:
        """抛 RuntimeError；``args`` / ``ctx`` 由 tools_node 传入。"""
        self.calls += 1
        raise RuntimeError("injected tool infra failure")

    def to_evidence(self, result: ToolResult) -> list[Evidence]:
        """恒抛路径不会走到成功分支；返回空证据列表以满足 Tool 协议。"""
        return []


class _BusinessFailureTool:
    """工具替身：``call`` 正常返回 ``ok=False`` 信封（模拟业务失败）。"""

    name = "BusinessFailureTool"
    description = "注入业务失败的测试替身工具"
    args_model = _EmptyArgs

    def __init__(self) -> None:
        self.calls = 0  # call 被调用次数（业务失败不应重试）

    async def call(self, args: ToolArgs, ctx: ToolContext) -> ToolResult:
        """返回业务失败信封（``error`` 为人读原因）。"""
        self.calls += 1
        return ToolResult(ok=False, error="商品不存在")

    def to_evidence(self, result: ToolResult) -> list[Evidence]:
        """业务失败分支不调用本方法；返回空证据列表以满足 Tool 协议。"""
        return []


def _tools_state(tool_name: str) -> dict:
    """tools_node 的最小 state：单条待办工具调用 + 空工作区。"""
    return {
        "case": make_case(),
        "hypotheses": [],
        "evidence": [],
        "pending_tool_calls": [{"tool": tool_name, "args": {}, "priority": 1}],
        "budget": Budget(),
        "degraded": False,
        "failures": [],
    }


async def test_tool_infra_error_retries_once_then_raises(monkeypatch):
    """守护 1：infra 异常 → 重试 1 次后裸抛，且 make_failure 零调用（不写 warn failure）。"""
    tool = _InfraFailureTool()
    node = make_tools_node([tool])
    failure_calls: list[dict] = []

    def _spy_make_failure(**kwargs: Any) -> dict:
        """记录一次 make_failure 调用并返回其入参（spy）。"""
        failure_calls.append(kwargs)
        return dict(kwargs)

    monkeypatch.setattr(tools_node_mod, "make_failure", _spy_make_failure)

    with pytest.raises(RuntimeError, match="injected tool infra failure"):
        await node(_tools_state(tool.name), {})

    assert tool.calls == 2, "首次异常 + 重试 1 次 = 恰好 2 次调用（非 1 次也非 3 次）"
    assert failure_calls == [], "infra 异常路径不得把异常降级成 warn failure"


async def test_tool_business_failure_still_warns_and_continues():
    """守护 2：业务失败 → 不抛异常，写 1 条 warn failure + error record，记账并清空待办。"""
    tool = _BusinessFailureTool()
    node = make_tools_node([tool])

    out = await node(_tools_state(tool.name), {})

    assert tool.calls == 1, "业务失败不重试（重试只针对 infra 异常）"
    failures = out["failures"]
    assert len(failures) == 1, "业务失败恰好产生 1 条 failure"
    assert failures[0]["severity"] == SEV_WARN
    assert "商品不存在" in failures[0]["reason"]
    records = out["tool_call_history"]
    assert len(records) == 1, "业务失败恰好产生 1 条 audit record"
    assert records[0]["status"] == "error"
    assert "商品不存在" in records[0]["error"]
    assert out["budget"].tool_calls == 1, "业务失败仍计 1 次 tool_calls"
    assert out["pending_tool_calls"] == [], "本 visit 待办消费完（空）"


def _mysql_reachable() -> bool:
    """按默认配置路径解析 DSN 并做 1s socket 探测（不连库、不建 engine）。

    ``Settings()`` 抛错时不吞异常 —— 配置层坏掉就该让本文件变红（收集期报错），
    而不是伪装成"环境没 DB"跳过。
    """
    parsed = urlparse(Settings().database_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 3306
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


async def _cleanup(case_id: str, run_id: str) -> None:
    """按 case_id/run_id 清理本次写入的行（先子后主；可重复运行、不污染开发库）。"""
    sm = get_sessionmaker()
    async with sm() as s:
        await s.execute(text("delete from review_trace where run_id = :r"), {"r": run_id})
        await s.execute(text("delete from review_evidence where run_id = :r"), {"r": run_id})
        await s.execute(text("delete from review_result where case_id = :c"), {"c": case_id})
        await s.execute(text("delete from review_run where case_id = :c"), {"c": case_id})
        await s.execute(text("delete from review_case where case_id = :c"), {"c": case_id})
        await s.commit()


class _FakeFailingGraph:
    """假图替身：``astream`` 先产出一条 plan update，再抛 RuntimeError。"""

    async def astream(
        self, state: dict, config: dict, *, stream_mode: str
    ) -> AsyncIterator[dict]:
        """按 ``stream_mode="updates"`` 产出一条 update 后抛图执行异常。"""
        yield {"plan": {"pending_tool_calls": []}}
        raise RuntimeError("injected graph failure")

    async def aget_state(self, config: dict) -> Any:
        """不该被调用（``astream`` 已抛出）；被调用即测试前提失效。"""
        raise AssertionError("aget_state 不应被调用：astream 已抛异常")


@pytest.mark.skipif(
    not _mysql_reachable(), reason="MySQL 不可达（未起 mysql-dev 容器）→ 跳过真库终态守护"
)
async def test_run_and_persist_graph_failure_writes_terminal_state(monkeypatch):
    """守护 3：图异常 → review_result 显式 HUMAN_REVIEW + infra_error trace，异常仍穿透。"""
    tag = uuid4().hex[:8]
    case_id = f"PYTEST_INFRA_GUARD_{tag}"
    run_id = f"PYTEST_INFRA_GUARD_RUN_{tag}"
    monkeypatch.setattr(persist_service, "get_production_graph", lambda: _FakeFailingGraph())
    try:
        with pytest.raises(RuntimeError, match="injected graph failure"):
            await run_and_persist(make_case(case_id=case_id), run_id=run_id)

        sm = get_sessionmaker()
        async with sm() as s:
            decisions = (
                await s.execute(
                    text("select decision from review_result where case_id = :c"),
                    {"c": case_id},
                )
            ).fetchall()
            assert [row[0] for row in decisions] == ["HUMAN_REVIEW"], "异常须落显式终态"

            traces = (
                await s.execute(
                    text(
                        "select output_json from review_trace "
                        "where run_id = :r and step_type = 'infra_error'"
                    ),
                    {"r": run_id},
                )
            ).fetchall()
            assert len(traces) == 1, "该 run 恰好 1 条 infra_error trace"
            out_text = json.dumps(traces[0][0], ensure_ascii=False, default=str)
            assert "R6_INFRA_UNAVAILABLE" in out_text, "R6 归因码须可查（review_result 无该列）"
            assert "RuntimeError" in out_text, "trace 须含异常类型短名"

            status = (
                await s.execute(
                    text("select status from review_run where run_id = :r"), {"r": run_id}
                )
            ).scalar()
            assert status == "RUNNING", "悬空 run 语义：异常路径不改 run 状态"
    finally:
        await _cleanup(case_id, run_id)


class _FakeTwoVisitFailingGraph:
    """假图替身：两轮 tools visit 各带 1 条证据，随后抛图执行异常。"""

    async def astream(
        self, state: dict, config: dict, *, stream_mode: str
    ) -> AsyncIterator[dict]:
        """按 ``stream_mode="updates"`` 产出两轮 tools delta（各带 1 条证据）后抛异常。"""
        yield {"tools": {"evidence": [ev("PRODUCT_FACT", ref_id="P_88231")]}}
        yield {"tools": {"evidence": [ev("MERCHANT_HISTORY", ref_id="M_5512")]}}
        raise RuntimeError("injected graph failure after two visits")

    async def aget_state(self, config: dict) -> Any:
        """不该被调用（``astream`` 已抛出）；被调用即测试前提失效。"""
        raise AssertionError("aget_state 不应被调用：astream 已抛异常")


@pytest.mark.skipif(
    not _mysql_reachable(), reason="MySQL 不可达（未起 mysql-dev 容器）→ 跳过真库终态守护"
)
async def test_run_and_persist_failure_keeps_evidence_of_all_visits(monkeypatch):
    """守护 4：异常前每一轮 tools visit 的证据都要落 review_evidence（跨 visit 累积）。"""
    tag = uuid4().hex[:8]
    case_id = f"PYTEST_INFRA_EVIDENCE_{tag}"
    run_id = f"PYTEST_INFRA_EVIDENCE_RUN_{tag}"
    monkeypatch.setattr(
        persist_service, "get_production_graph", lambda: _FakeTwoVisitFailingGraph()
    )
    try:
        with pytest.raises(RuntimeError, match="injected graph failure after two visits"):
            await run_and_persist(make_case(case_id=case_id), run_id=run_id)

        sm = get_sessionmaker()
        async with sm() as s:
            refs = (
                await s.execute(
                    text(
                        "select ref_id from review_evidence where run_id = :r order by ref_id"
                    ),
                    {"r": run_id},
                )
            ).scalars().all()
            assert refs == ["M_5512", "P_88231"], (
                f"两轮 visit 的证据都要落库（覆盖式累积会只剩最后一轮）: {refs}"
            )
    finally:
        await _cleanup(case_id, run_id)
