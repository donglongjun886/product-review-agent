"""infra 数据库接入 —— 配置 + async engine / session 工厂。"""

from __future__ import annotations

from functools import cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

__all__ = ["Settings", "get_engine", "get_sessionmaker"]


def _repo_root_env_file() -> str:
    """仓库根 ``.env`` 的绝对路径。

    从本文件逐级上溯，取第一个含 ``pyproject.toml`` 的目录为仓库根；找不到（脱离仓库安装）
    退回相对 cwd 的 ``".env"``。
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return str(parent / ".env")
    return ".env"


# 仓库根 .env 的绝对路径（模块加载时定位一次）。
_ENV_FILE = _repo_root_env_file()


class Settings(BaseSettings):
    """应用配置（pydantic-settings）：仓库根 ``.env`` 可覆盖全部字段。"""

    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_file_encoding="utf-8")

    database_url: str = "mysql+aiomysql://root:root@127.0.0.1:3306/product_review"

    # 生产 LLM 网关的模型名（provider 前缀写法，如 deepseek/deepseek-flash）。
    deepseek_model: str = "deepseek/deepseek-flash"

    # 真实 LLM 网关凭据。
    deepseek_api_key: str | None = None
    deepseek_base_url: str | None = None

    # 可选：Langfuse 可观测性。
    langfuse_public_key: str | None = None  # LANGFUSE_PUBLIC_KEY（缺失 = 观测关闭）
    langfuse_secret_key: str | None = None  # LANGFUSE_SECRET_KEY（**勿提交**）
    langfuse_host: str | None = None  # LANGFUSE_HOST，缺省 http://localhost:3000


@cache
def get_engine() -> AsyncEngine:
    """返回进程内单例 async engine（``create_async_engine``，MySQL/aiomysql）。

    首次调用才实例化 ``Settings()``（读 .env / 环境变量）。
    """
    return create_async_engine(
        Settings().database_url,
        pool_pre_ping=True,
        echo=False,
        poolclass=NullPool,
    )


@cache
def get_sessionmaker() -> async_sessionmaker:
    """返回进程内单例 async sessionmaker（``expire_on_commit=False``）。

    用法：``sm = get_sessionmaker(); async with sm() as session: ...``（工厂需**调用**才生成
    AsyncSession；退出时自动 close/回滚未提交事务）。
    """
    return async_sessionmaker(get_engine(), expire_on_commit=False)
