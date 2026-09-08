"""infra 数据库接入 —— 配置 + async engine / session 工厂（总链路 ⑪「持久化」底座）。

职责与边界：
- 只负责**连接工厂**（engine / sessionmaker 的懒加载单例），不持有任何业务落库逻辑
  （落库编排在 ``pra.infra.persist_service``）；ORM 映射在 ``pra.infra.rdb_models``
  （本模块不 import 它，保持"工厂-模型"解耦，避免初始化顺序耦合）。
- **仅 MySQL**（dialect ``aiomysql``，异步驱动）：``Settings.database_url`` 默认开发
  DSN ``mysql+aiomysql://root:root@127.0.0.1:3306/product_review`` —— 该默认值**仅供
  本地开发**（root/root 明文仅指向本地容器 mysql-dev）；生产/其他环境一律通过
  ``.env`` 的 ``DATABASE_URL`` 覆盖（``SettingsConfigDict(env_file=".env")`` 允许），
  .env 已被仓库 .gitignore 忽略；可提交的模板见仓库根 ``.env.example``。
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

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

__all__ = ["Settings", "get_engine", "get_sessionmaker"]


class Settings(BaseSettings):
    """应用配置（pydantic-settings）：``.env`` 可覆盖全部字段。

    ``model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")``：
    实例化时读取进程 cwd 的 ``.env``（不存在则忽略），真实环境变量优先级高于 .env。
    字段名大小写不敏感映射环境键（``DATABASE_URL`` ↔ ``database_url``）。
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # 仅本地开发的默认值（root/root@127.0.0.1:3306 的 Docker mysql-dev 容器）；
    # 生产/测试必须经 .env 的 DATABASE_URL 覆盖 —— 见模块 docstring。
    database_url: str = "mysql+aiomysql://root:root@127.0.0.1:3306/product_review"


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
