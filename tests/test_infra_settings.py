"""infra 配置层（pra/infra/db.py Settings）单测 —— **不连库、不联网**。

回归护栏（2026-09-09 实测缺陷）：`.env` 同时承载 DB 配置与 real LLM 评测凭据
（``DEEPSEEK_API_KEY`` / ``DEEPSEEK_BASE_URL``），而 pydantic-settings 默认
``extra="forbid"`` —— 若 Settings 未声明这两个字段，``Settings()`` 直接抛
ValidationError，沿 ``get_sessionmaker → process_review`` 一路炸到 HTTP 500
（真库路径全线不可用，而既有测试因不碰真库仍全绿）。

本文件固化两件事：
1. 含 DEEPSEEK_* 的 .env 不再让 Settings 报错（修复点）；
2. ``extra="forbid"`` **不得被放宽** —— 未知键仍须报错（防 DB 键名拼错静默回落
   开发 DSN 连错库，见 db.py docstring）。

语法/import 约定：顶部 ``from __future__ import annotations``；import 一律 pra.*。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from pra.infra.db import Settings

_DEFAULT_DSN = "mysql+aiomysql://root:root@127.0.0.1:3306/product_review"


def _env_file(tmp_path: Path, text: str) -> str:
    """把给定内容写成临时 .env 并返回路径（Settings 经 ``_env_file`` 显式注入）。"""
    p = tmp_path / ".env"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_settings_accepts_deepseek_keys(tmp_path):
    """含 DB + DEEPSEEK_* 三类键的 .env → 不抛异常，且逐字段读对（修复点）。"""
    env = _env_file(
        tmp_path,
        "DATABASE_URL=mysql+aiomysql://u:p@127.0.0.1:3306/other_db\n"
        "DEEPSEEK_API_KEY=sk-placeholder\n"
        "DEEPSEEK_BASE_URL=https://example.invalid\n",
    )
    s = Settings(_env_file=env)
    assert s.database_url == "mysql+aiomysql://u:p@127.0.0.1:3306/other_db"
    assert s.deepseek_api_key == "sk-placeholder"
    assert s.deepseek_base_url == "https://example.invalid"


def test_settings_deepseek_keys_optional(tmp_path):
    """只给 DATABASE_URL → 两个可选字段为 None（不跑 real 评测时无需这两项）。"""
    env = _env_file(tmp_path, "DATABASE_URL=mysql+aiomysql://u:p@127.0.0.1:3306/x\n")
    s = Settings(_env_file=env)
    assert s.deepseek_api_key is None
    assert s.deepseek_base_url is None


def test_settings_accepts_langfuse_keys(tmp_path):
    """含 Langfuse 观测配置的 .env → 不抛异常，且逐字段读对（同一 extra 坑的护栏）。

    `.env` 现已是「DB + real 评测 + 可观测性」三类配置的单一载体：不声明这些键，
    `Settings()` 会因 `extra="forbid"` 整体抛 ValidationError（docs/09 §8）。
    """
    env = _env_file(
        tmp_path,
        "DATABASE_URL=mysql+aiomysql://u:p@127.0.0.1:3306/x\n"
        "LANGFUSE_PUBLIC_KEY=pk-lf-placeholder\n"
        "LANGFUSE_SECRET_KEY=sk-lf-placeholder\n"
        "LANGFUSE_HOST=http://localhost:3000\n"
        "PRA_LANGFUSE_ENABLED=1\n"
        "PRA_LANGFUSE_EXPERIMENT=baseline\n"
        "PRA_LANGFUSE_SAMPLE=0.5\n"
        "PRA_LANGFUSE_SESSION=eval-run-1\n",
    )
    s = Settings(_env_file=env)
    assert s.langfuse_public_key == "pk-lf-placeholder"
    assert s.langfuse_secret_key == "sk-lf-placeholder"
    assert s.langfuse_host == "http://localhost:3000"
    assert s.pra_langfuse_enabled == "1"
    assert s.pra_langfuse_experiment == "baseline"
    assert s.pra_langfuse_sample == "0.5"  # str 声明：infra 不做类型转换（避免副作用）
    assert s.pra_langfuse_session == "eval-run-1"


def test_settings_langfuse_keys_optional(tmp_path):
    """不配可观测性 → 七个字段全 None（默认路径 no-op，CI 无需任何 key）。"""
    env = _env_file(tmp_path, "DATABASE_URL=mysql+aiomysql://u:p@127.0.0.1:3306/x\n")
    s = Settings(_env_file=env)
    assert s.langfuse_public_key is None
    assert s.langfuse_secret_key is None
    assert s.langfuse_host is None
    assert s.pra_langfuse_enabled is None
    assert s.pra_langfuse_experiment is None
    assert s.pra_langfuse_sample is None
    assert s.pra_langfuse_session is None


def test_settings_defaults_without_env_file(monkeypatch):
    """``_env_file=None``（完全不读 .env）→ 字段默认值（本地开发 DSN + 可选字段 None）。"""
    for key in (
        "DATABASE_URL",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST",
        "PRA_LANGFUSE_ENABLED",
        "PRA_LANGFUSE_EXPERIMENT",
        "PRA_LANGFUSE_SAMPLE",
        "PRA_LANGFUSE_SESSION",
    ):
        monkeypatch.delenv(key, raising=False)
    s = Settings(_env_file=None)
    assert s.database_url == _DEFAULT_DSN
    assert s.deepseek_api_key is None
    assert s.deepseek_base_url is None
    assert s.langfuse_public_key is None
    assert s.langfuse_secret_key is None


def test_settings_rejects_unknown_key(tmp_path):
    """``extra="forbid"`` 护栏：未知键仍须报错（勿为方便改成 ignore）。"""
    env = _env_file(
        tmp_path,
        "DATABASE_URL=mysql+aiomysql://u:p@127.0.0.1:3306/x\nUNKNOWN_KEY=oops\n",
    )
    with pytest.raises(ValidationError) as ei:
        Settings(_env_file=env)
    assert "unknown_key" in str(ei.value)
