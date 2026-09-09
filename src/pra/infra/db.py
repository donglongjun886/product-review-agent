"""infra 数据库接入 —— 配置 + async engine / session 工厂（总链路 ⑪「持久化」底座）。

职责与边界：
- 只负责**连接工厂**（engine / sessionmaker 的懒加载单例），不持有任何业务落库逻辑
  （落库编排在 ``pra.infra.persist_service``）；ORM 映射在 ``pra.infra.rdb_models``
  （本模块不 import 它，保持"工厂-模型"解耦，避免初始化顺序耦合）。
- **仅 MySQL**（dialect ``aiomysql``，异步驱动）：``Settings.database_url`` 默认开发
  DSN ``mysql+aiomysql://root:root@127.0.0.1:3306/product_review`` —— 该默认值**仅供
  本地开发**（root/root 明文仅指向本地容器 mysql-dev）；生产/其他环境一律通过仓库根
  ``.env`` 的 ``DATABASE_URL`` 覆盖（env_file=仓库根 .env 的**绝对路径**，见
  ``_ENV_FILE``/Settings —— 从任何 cwd 启动都读同一 .env，不会因 cwd 不同而静默
  漏读并回落开发 DSN），.env 已被仓库 .gitignore 忽略；可提交的模板见仓库根
  ``.env.example``。``.env`` 亦可同时含 real LLM 评测凭据（``DEEPSEEK_API_KEY`` /
  ``DEEPSEEK_BASE_URL``）—— Settings 已将其声明为可选字段，两类配置共存于同一
  .env（见 Settings 字段注释；不声明会因 extra="forbid" 让配置层整体抛错）。
- 时间口径：MySQL ``DATETIME(fsp=3)`` 存 naive（无时区）时间；业务层统一写
  **naive UTC**（``datetime.now(timezone.utc).replace(tzinfo=None)``，见
  ``pra.infra.persist_service._utcnow``），读取后一律按 UTC 解释 —— 单一约定，
  勿混入本地时区 naive。
- ``pool_pre_ping=True``：取连接前探测，防 MySQL 侧 wait_timeout 断连后拿到死连接；
  ``echo=False``：不打印 SQL（排查时可临时改 True）。

并发说明：engine/sessionmaker 的"先查缓存再创建"之间无 ``await`` 切换点（同一事件
循环内原子），无需加锁 —— 与 ``pra.api.service.get_graph`` 的懒加载同款论证；未来若
引入异步初始化（如连接探测/池预热）需改 asyncio 单飞（once）模式。
**事件循环绑定**：async engine 的连接池绑定创建它的事件循环，跨 loop 复用会抛
``RuntimeError: attached to a different loop``（典型：同一进程先 ``asyncio.run``
直调、再用 fastapi TestClient 的不同 anyio loop 打 HTTP）。因此 MVP 默认
``poolclass=NullPool``（每次取用新建连接、无池复用）—— 单机/开发/测试最稳，代价是
每 session 一次 TCP 握手；生产 MQ worker 是**单事件循环**常驻进程，届时改回默认池
（去掉 ``poolclass=NullPool`` 即可）以复用连接，见 docstring 演进指引。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

__all__ = ["Settings", "get_engine", "get_sessionmaker"]


def _repo_root_env_file() -> str:
    """仓库根 ``.env`` 的绝对路径 —— env_file 不能按 cwd 相对解析（静默漏读回落
    默认开发 DSN，连错库无报错）。

    从本文件（``src/pra/infra/db.py``）逐级上溯，取第一个含 ``pyproject.toml`` 的
    目录视为仓库根（.env / .env.example 均在其下）；找不到（如脱离仓库安装）退回
    相对 cwd 的 ``".env"``（与历史行为一致）。
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

    ``model_config = SettingsConfigDict(env_file=_ENV_FILE, ...)``：env_file 为本
    文件上溯定位的**仓库根 .env 绝对路径** —— 从任何 cwd 启动都读同一文件；该路径
    不存在时 pydantic-settings 静默忽略（与旧行为一致）。真实环境变量优先级高于
    .env。字段名大小写不敏感映射环境键（``DATABASE_URL`` ↔ ``database_url``）。

    字段：``database_url``（DSN）+ ``deepseek_api_key`` / ``deepseek_base_url``
    （可选，real LLM 评测凭据）+ ``langfuse_*`` / ``pra_langfuse_*``（可选，可观测性
    接入凭据与开关 —— 声明为字段的理由见下方字段注释）。``extra`` 保持
    pydantic-settings 默认的 **forbid**：未知键一律报错，不静默忽略。
    """

    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_file_encoding="utf-8")

    # 仅本地开发的默认值（root/root@127.0.0.1:3306 的 Docker mysql-dev 容器）；
    # 生产/测试必须经 .env 的 DATABASE_URL 覆盖 —— 见模块 docstring。
    database_url: str = "mysql+aiomysql://root:root@127.0.0.1:3306/product_review"

    # 可选：真实 LLM 评测脚本（scripts/run_evaluation_real.py）的网关凭据。
    # 为什么声明为字段而不是让 Settings 忽略未知键：pydantic-settings 默认
    # extra="forbid"，而 .env 同时承载 DB 配置与这两项评测凭据 —— 不声明就会让
    # Settings() 直接抛 ValidationError，沿 get_sessionmaker → process_review 一路
    # 炸到 HTTP 500（2026-09-09 实测缺陷）。声明为可选字段后，`.env` 单文件承载
    # 两类配置，且 extra="forbid" 的严格性不变（DB 键名拼错仍报错，防静默回落
    # 开发 DSN 连错库）。infra 自身不消费这两个值（默认 None，无副作用）。
    deepseek_api_key: str | None = None
    deepseek_base_url: str | None = None

    # 可选：Langfuse 可观测性（docs/09-langfuse-observability.md）。
    # 与 DEEPSEEK_* 同理 —— .env 现已是「DB + real 评测 + 可观测性」三类配置的
    # 单一载体，不声明这些键同样会让 Settings() 整体抛 ValidationError（同一坑，
    # commit 6cf763e）。**infra 不消费**这些值：真正的读取方是
    # ``pra.observability.tracing``（直接读环境变量，见该模块 make_tracer）。
    # 这里声明只为让 .env 能安全承载它们，且保持 extra="forbid" 不变。
    langfuse_public_key: str | None = None  # LANGFUSE_PUBLIC_KEY（缺失 → NullTracer）
    langfuse_secret_key: str | None = None  # LANGFUSE_SECRET_KEY（**勿提交**）
    langfuse_host: str | None = None  # LANGFUSE_HOST，缺省 http://localhost:3000
    pra_langfuse_enabled: str | None = None  # PRA_LANGFUSE_ENABLED（0/false/no/off 关闭）
    pra_langfuse_experiment: str | None = None  # PRA_LANGFUSE_EXPERIMENT（缺省 baseline）
    pra_langfuse_sample: str | None = None  # PRA_LANGFUSE_SAMPLE（缺省 1.0；str 避免类型转换副作用）
    pra_langfuse_session: str | None = None  # PRA_LANGFUSE_SESSION（一次 evaluation run 的分组 id）


# 模块级懒加载缓存（进程内单例）：首次 get_engine/get_sessionmaker 时创建，之后复用
# 同一连接池（未来 MQ worker 多协程消费共享同一池）。Settings 在首次取用时实例化，
# 使 .env 的读取延迟到 engine 真正创建前一刻。
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
    """懒加载返回全局 async engine 单例（``create_async_engine``，MySQL/aiomysql）。

    :return: SQLAlchemy ``AsyncEngine``（aiomysql 方言）。

    注：仅 MySQL（dialect aiomysql）；开发 DSN 默认值见 ``Settings.database_url``，
    生产经 .env 覆盖。任务书的 ``pool_pre_sing=True`` 系笔误 —— SQLAlchemy 参数名为
    ``pool_pre_ping``（断连探测），此处用后者。
    """
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            _get_settings().database_url,
            pool_pre_ping=True,
            echo=False,
            # MVP/开发/测试：无池连接（async 池绑定事件循环，跨 loop 会 RuntimeError）；
            # 生产 MQ worker（单事件循环常驻）去掉本行改回默认池 —— 见模块 docstring。
            poolclass=NullPool,
        )
    return _engine


def get_sessionmaker() -> async_sessionmaker:
    """懒加载返回全局 async sessionmaker 单例（``async_sessionmaker``）。

    ``expire_on_commit=False``：commit 后 ORM 实例属性不失效 —— 落库编排在多个
    commit 里程碑之间持有 case/run 行引用继续改状态（见 persist_service 流程），
    避免 commit 后属性访问触发额外 refresh 查询。
    用法：``sm = get_sessionmaker(); async with sm() as session: ...``
    （``async_sessionmaker`` 是工厂，需**调用**生成 AsyncSession 才是 async 上下文
    管理器；退出时自动 close/回滚未提交事务）。
    """
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _sessionmaker
