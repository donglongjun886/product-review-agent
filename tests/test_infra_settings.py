"""infra 配置层（pra/infra/db.py Settings）单测：**不连库、不联网**。

`.env` 同时承载 DB 配置与 real LLM 凭据（``DEEPSEEK_*``），而 pydantic-settings
``extra="forbid"`` 会在字段未声明时直接抛 ValidationError，沿
``get_sessionmaker → process_review`` 炸到 HTTP 500（不碰真库的测试仍全绿，故需本文件）。

固化两件事：① 含 DEEPSEEK_* / LANGFUSE_* 的 .env 不再报错（修复点）；
② ``extra="forbid"`` **不得被放宽** —— 未知键仍须报错，防 DB 键名拼错静默回落
开发 DSN 连错库。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from pra.infra.db import Settings

_DEFAULT_DSN = "mysql+aiomysql://root:root@127.0.0.1:3306/product_review"


def _env_file(tmp_path: Path, text: str) -> str:
    p = tmp_path / ".env"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_settings_accepts_deepseek_keys(tmp_path):
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
    env = _env_file(tmp_path, "DATABASE_URL=mysql+aiomysql://u:p@127.0.0.1:3306/x\n")
    s = Settings(_env_file=env)
    assert s.deepseek_api_key is None
    assert s.deepseek_base_url is None


def test_settings_accepts_langfuse_keys(tmp_path):
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
