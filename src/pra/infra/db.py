"""infra 数据库接入 —— 配置 + async engine / session 工厂。

只负责连接工厂（engine / sessionmaker 的懒加载单例），不持有业务落库逻辑（在
``pra.infra.persist_service``）；ORM 映射在 ``pra.infra.rdb_models``，本模块不 import 它，
保持「工厂-模型」解耦。

仅 MySQL（dialect ``aiomysql``，异步驱动）：``Settings.database_url`` 默认
``mysql+aiomysql://root:root@127.0.0.1:3306/product_review`` —— **仅供本地开发**（root/root
明文只指向本地容器 mysql-dev），**勿用于生产**；其他环境一律经仓库根 ``.env`` 的
``DATABASE_URL`` 覆盖。env_file 是仓库根 .env 的绝对路径（``_ENV_FILE``），从任何 cwd 启动都
读同一文件，不会静默漏读并回落开发 DSN。

时间口径：MySQL ``DATETIME(fsp=3)`` 存 naive；业务层统一写 naive UTC
（``persist_service._utcnow``），读取后一律按 UTC 解释 —— 勿混入本地时区 naive。
``pool_pre_ping=True`` 防 wait_timeout 断连后拿到死连接；``echo=False`` 不打印 SQL。

并发：engine/sessionmaker 的「先查缓存再创建」之间无 ``await`` 切换点，同一事件循环内原子，
无需加锁。**事件循环绑定**：async engine 连接池绑定创建它的 loop，跨 loop 复用会抛
``RuntimeError: attached to a different loop``（典型：先 ``asyncio.run`` 直调，再用 TestClient
的不同 anyio loop 打 HTTP），故默认 ``poolclass=NullPool``（无池复用，代价是每 session 一次
TCP 握手）。生产 MQ worker 是单事件循环常驻进程，届时去掉 ``NullPool`` 改回默认池。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

__all__ = ["Settings", "get_engine", "get_sessionmaker"]


def _repo_root_env_file() -> str:
    """仓库根 ``.env`` 的绝对路径 —— env_file 不能按 cwd 相对解析（会静默漏读并回落默认开发
    DSN，连错库无报错）。

    从本文件逐级上溯，取第一个含 ``pyproject.toml`` 的目录为仓库根；找不到（脱离仓库安装）
    退回相对 cwd 的 ``".env"``。
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").is_file():
            return str(parent / ".env")
    return ".env"


# 仓库根 .env 的绝对路径（模块加载时定位一次；Settings 实例化直接使用）。
_ENV_FILE = _repo_root_env_file()


class Settings(BaseSettings):
    """应用配置（pydantic-settings）：仓库根 ``.env`` 可覆盖全部字段。

    env_file 为 ``_ENV_FILE``（绝对路径）—— 从任何 cwd 启动都读同一文件，路径不存在时静默
    忽略。真实环境变量优先级高于 .env，字段名大小写不敏感映射环境键。``extra`` 保持
    pydantic-settings 默认的 **forbid**：未知键一律报错，不静默忽略（防拼错键名连错库）。
    """

    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_file_encoding="utf-8")

    # 仅本地开发的默认值（root/root@127.0.0.1:3306 的 Docker mysql-dev 容器）；
    # 生产/测试必须经 .env 的 DATABASE_URL 覆盖。
    database_url: str = "mysql+aiomysql://root:root@127.0.0.1:3306/product_review"

    # 可选：真实 LLM 评测脚本的网关凭据。.env 同时承载 DB 配置与评测凭据，而 extra="forbid"
    # 会让未声明的键把 Settings() 整体炸成 ValidationError，沿 get_sessionmaker →
    # process_review 一路到 HTTP 500。infra 自身不消费这两个值（默认 None）。
    deepseek_api_key: str | None = None
    deepseek_base_url: str | None = None

    # 可选：Langfuse 可观测性。与 DEEPSEEK_* 同理 —— 不声明同样会让 Settings() 整体抛
    # ValidationError。**infra 不消费**这些值，真正读取方是 ``pra.observability.tracing``。
    langfuse_public_key: str | None = None  # LANGFUSE_PUBLIC_KEY（缺失 → NullTracer）
    langfuse_secret_key: str | None = None  # LANGFUSE_SECRET_KEY（**勿提交**）
    langfuse_host: str | None = None  # LANGFUSE_HOST，缺省 http://localhost:3000
    pra_langfuse_enabled: str | None = None  # PRA_LANGFUSE_ENABLED（0/false/no/off 关闭）
    pra_langfuse_experiment: str | None = None  # PRA_LANGFUSE_EXPERIMENT（缺省 baseline）
    pra_langfuse_sample: str | None = None  # PRA_LANGFUSE_SAMPLE（缺省 1.0；str 避免类型转换副作用）
    pra_langfuse_session: str | None = None  # PRA_LANGFUSE_SESSION（一次 evaluation run 的分组 id）


# 模块级懒加载缓存（进程内单例）：首次 get_engine/get_sessionmaker 时创建，之后复用同一连接池。
_settings: Optional[Settings] = None
_engine: Optional[AsyncEngine] = None
_sessionmaker: Optional[async_sessionmaker] = None


def _get_settings() -> Settings:
    """懒加载 Settings 单例（读 .env / 环境变量的时刻 = 首次调用 engine 工厂）。"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def get_engine() -> AsyncEngine:
    """懒加载返回全局 async engine 单例（``create_async_engine``，MySQL/aiomysql）。"""
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            _get_settings().database_url,
            pool_pre_ping=True,
            echo=False,
            # MVP/开发/测试：无池连接（async 池绑定事件循环，跨 loop 会 RuntimeError）；
            # 生产 MQ worker（单事件循环常驻）去掉本行改回默认池。
            poolclass=NullPool,
        )
    return _engine


def get_sessionmaker() -> async_sessionmaker:
    """懒加载返回全局 async sessionmaker 单例（``async_sessionmaker``）。

    ``expire_on_commit=False``：commit 后 ORM 实例属性不失效 —— 落库编排在多个 commit 里程碑
    之间持有 case/run 行引用继续改状态，避免 commit 后属性访问触发额外 refresh。用法：
    ``sm = get_sessionmaker(); async with sm() as session: ...``（工厂需**调用**才生成
    AsyncSession；退出时自动 close/回滚未提交事务）。
    """
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _sessionmaker
