"""真库落库冒烟（infra 持久化链路）：走默认配置路径，MySQL 不可达时跳过。

既有单测都不碰真库（摘要函数单测、落库层 mock、``process_review`` mock），于是
「.env 含 DEEPSEEK_* → Settings(extra="forbid") 抛错 → process_review 两分支全挂
→ HTTP 恒 500」在全绿假象下无人发现。

因此本测试不得注入 ``_env_file=None``、不得 monkeypatch ``db._settings``，必须经
``get_sessionmaker()`` 默认路径；覆盖 COMPLEX（brand 空缺 → R-301 → Agent 图）与
PASS（规则直判）各一发，断言五表落库行数与关键列，``finally`` 按 case_id 清理。
MySQL 未起时跳过，配置层缺陷另有 ``test_infra_settings.py`` 纯单测兜底（恒跑）。
"""

from __future__ import annotations

import socket
from urllib.parse import urlparse
from uuid import uuid4

import pytest
from helpers import make_case
from sqlalchemy import text

from pra.domain.models import ProductReviewCase
from pra.infra.db import Settings, get_sessionmaker
from pra.infra.persist_service import process_review


def _mysql_reachable() -> bool:
    """按 **默认配置路径** 解析 DSN 并做 1s socket 探测（不连库、不建 engine）。

    ``Settings()`` 抛错时**不吞异常**：配置层坏掉就该让本文件变红（收集期报错），
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


pytestmark = pytest.mark.skipif(
    not _mysql_reachable(), reason="MySQL 不可达（未起 mysql-dev 容器）→ 跳过真库冒烟"
)


def _case(case_id: str, *, brand: str | None) -> ProductReviewCase:
    """冒烟案件：复用 tests/helpers.make_case（默认标题/描述干净，只有 brand 决定路径）。

    """
    return make_case(case_id=case_id, brand=brand)


async def _cleanup(case_ids: tuple[str, str]) -> None:
    """按 case_id 清理本测试写入的五表行（先子表后主表；可重复运行、不污染开发库）。"""
    sm = get_sessionmaker()
    async with sm() as s:
        await s.execute(
            text(
                "delete from review_trace where run_id in "
                "(select run_id from review_run where case_id in (:a, :b))"
            ),
            {"a": case_ids[0], "b": case_ids[1]},
        )
        await s.execute(
            text(
                "delete from review_evidence where run_id in "
                "(select run_id from review_run where case_id in (:a, :b))"
            ),
            {"a": case_ids[0], "b": case_ids[1]},
        )
        await s.execute(
            text("delete from review_result where case_id in (:a, :b)"),
            {"a": case_ids[0], "b": case_ids[1]},
        )
        await s.execute(
            text("delete from review_run where case_id in (:a, :b)"),
            {"a": case_ids[0], "b": case_ids[1]},
        )
        await s.execute(
            text("delete from review_case where case_id in (:a, :b)"),
            {"a": case_ids[0], "b": case_ids[1]},
        )
        await s.commit()


async def test_persist_smoke_two_branches():
    tag = uuid4().hex[:8]
    cx_id, pass_id = f"PYTEST_SMOKE_CMPLX_{tag}", f"PYTEST_SMOKE_PASS_{tag}"
    ids = (cx_id, pass_id)
    try:
        out_cx = await process_review(_case(cx_id, brand=None))
        out_pass = await process_review(_case(pass_id, brand="UNIQLO"))

        assert out_cx["verdict"] == "COMPLEX"
        assert out_cx["decision"] is not None
        assert out_pass["verdict"] == "PASS"
        assert out_pass["decision"].decision.value == "PASS"

        sm = get_sessionmaker()
        async with sm() as s:
            n_case = (
                await s.execute(
                    text("select count(*) from review_case where case_id in (:a, :b)"),
                    {"a": cx_id, "b": pass_id},
                )
            ).scalar()
            assert n_case == 2, "两案 case 行均应落库"

            runs = (
                await s.execute(
                    text(
                        "select trigger_type, status from review_run "
                        "where case_id in (:a, :b)"
                    ),
                    {"a": cx_id, "b": pass_id},
                )
            ).fetchall()
            assert len(runs) == 2, "两案 run 行均应落库"
            assert {r[0] for r in runs} == {"INITIAL", "SCREENING_DIRECT"}
            assert all(r[1] == "DECIDED" for r in runs), "两 run 均应收尾 DECIDED"

            results = (
                await s.execute(
                    text(
                        "select case_id, decision from review_result "
                        "where case_id in (:a, :b)"
                    ),
                    {"a": cx_id, "b": pass_id},
                )
            ).fetchall()
            dec = {r[0]: r[1] for r in results}
            assert dec[cx_id] == out_cx["decision"].decision.value
            assert dec[pass_id] == "PASS"

            n_trace_cx = (
                await s.execute(
                    text("select count(*) from review_trace where run_id = :r"),
                    {"r": out_cx["run_id"]},
                )
            ).scalar()
            n_trace_pass = (
                await s.execute(
                    text("select count(*) from review_trace where run_id = :r"),
                    {"r": out_pass["run_id"]},
                )
            ).scalar()
            assert n_trace_cx >= 1, "Agent 路径应有 trace 行"
            assert n_trace_pass == 0, "规则直判路径无 trace 行（设计语义）"
    finally:
        await _cleanup(ids)
