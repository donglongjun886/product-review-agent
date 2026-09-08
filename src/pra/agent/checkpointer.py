"""Checkpointer 工厂 —— LangGraph 线程状态持久化（MVP：InMemorySaver；业务真相在 MySQL）。

**线程 checkpoint 与业务 MySQL 表分离（选型 A，docs/04-graph-design.md §7.2）**：
LangGraph 的线程 Checkpointer 只保存"线程中间状态"（``AgentState`` 每步快照），
服务于断点续跑 / eval 重放 / super-step 恢复；**业务真相**（agent_step / evidence /
decision 等）由 worker 层显式落 MySQL，**不是本模块职责**（04 §7.4 口径）——
本模块不写任何业务表，只提供线程状态 (de)serialize 与保存介质工厂。

MVP 只落 **InMemorySaver 工厂**（进程内内存，联调/单测/走查脚本用）：
- ``make_serde``：``JsonPlusSerializer`` + msgpack **白名单**（
  ``ALLOWED_MSG_PACK_MODULES`` = ``pra.domain.models`` 全量 14 个 domain 模型）；
  白名单外的任意类禁止按 msgpack 扩展类型还原 → 受信反序列化边界（防注入）；
  有 ``to_json``/``model_dump`` 能力的白名单对象经字段级路径读写。
- ``make_memory_checkpointer``：``InMemorySaver(serde=make_serde())``。

未来持久化位（按选型 A 演进，**勿在 MVP 提前实现**，需求到时再开）：
- ``SqliteSaver`` / ``AsyncSqliteSaver``（``langgraph.checkpoint.sqlite``）：
  单机本地落盘 / 轻量异步场景；
- ``PostgresSaver`` / ``AsyncPostgresSaver``（``langgraph.checkpoint.postgres``）：
  多副本共享的线程状态存储；
- **自研 MySQL Checkpointer**（SQLAlchemy async / aiomysql，与本仓库 migrations
  对齐）：生产选型 —— 与业务表同库、按 thread_id(=run_id) 恢复案件调查进度、
  预算中断续跑，并配合 Redis 幂等去重。
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# msgpack 扩展类型白名单：允许按类型完整 (de)serialize 的 domain 模型（全量 14 个，
# 与 src/pra/domain/models.py 类名逐字对齐；graph-mvp-contracts §7 定稿清单）。
# 元素一律 ("pra.domain.models", "<类名>")：只放 AgentState 直接承载的模型，
# 其嵌套子模型（SkuInfo/ProductImage 等已列出者）随父模型字段级递归处理。
ALLOWED_MSG_PACK_MODULES: list[tuple[str, str]] = [
    ("pra.domain.models", "Budget"),
    ("pra.domain.models", "BudgetLimits"),
    ("pra.domain.models", "Evidence"),
    ("pra.domain.models", "Hypothesis"),
    ("pra.domain.models", "HypothesisStatus"),
    ("pra.domain.models", "Decision"),
    ("pra.domain.models", "RiskLevel"),
    ("pra.domain.models", "RiskType"),
    ("pra.domain.models", "ProductReviewCase"),
    ("pra.domain.models", "ProductInfo"),
    ("pra.domain.models", "ProductImage"),
    ("pra.domain.models", "SkuInfo"),
    ("pra.domain.models", "ScreeningSignal"),
    ("pra.domain.models", "ReviewDecision"),
]


def make_serde() -> JsonPlusSerializer:
    """构造线程状态 (de)serializer：JsonPlusSerializer + msgpack 白名单。

    允许 (de)serialize 的类限定为 ``ALLOWED_MSG_PACK_MODULES``（pra.domain.models
    全量 14 个 domain 模型）；白名单外对象走 jsonplus 的 pickle 回退或报错，
    保证线程 checkpoint 落库/恢复的类型安全与受信边界。
    """
    return JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSG_PACK_MODULES)


def make_memory_checkpointer() -> InMemorySaver:
    """MVP 内存 Checkpointer 工厂（进程内有效；联调/单测/走查用）。

    ``InMemorySaver(serde=...)``：线程中间状态（AgentState）逐节点落内存，
    供断点续跑与 eval 重放；业务真相仍由 worker 显式落 MySQL（选型 A，见模块 docstring）。
    Sqlite/Postgres/自研 MySQL saver 留未来注释位（模块 docstring），MVP 不实现。
    """
    return InMemorySaver(serde=make_serde())


__all__ = [
    "ALLOWED_MSG_PACK_MODULES",
    "make_memory_checkpointer",
    "make_serde",
]
