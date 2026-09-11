"""Checkpointer 工厂：LangGraph 线程状态持久化，MVP 是 InMemorySaver。

线程 checkpoint 只存 ``AgentState`` 每步快照，服务于断点续跑 / eval 重放 / super-step
恢复；业务真相（review_trace / review_evidence / review_result 等）由 worker 层显式落
MySQL，不是本模块职责 —— 本模块不写业务表，只提供状态 (de)serialize 与介质工厂。

未实现、需求到时再开的持久化位：``SqliteSaver``（单机落盘）、``PostgresSaver``（多副本
共享）、自研 MySQL Checkpointer（与业务表同库、按 thread_id(=run_id) 续跑 + Redis 去重）。
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# msgpack 扩展类型白名单：允许完整 (de)serialize 的 domain 模型（全量 14 个，与
# src/pra/domain/models.py 类名逐字对齐），元素一律 ("pra.domain.models", "<类名>")；
# 只放 AgentState 直接承载的模型，嵌套子模型随父模型字段级递归处理。
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
    """构造线程状态 (de)serializer：白名单外对象走 pickle 回退或报错（受信反序列化边界）。"""
    return JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSG_PACK_MODULES)


def make_memory_checkpointer() -> InMemorySaver:
    """MVP 内存 Checkpointer 工厂（进程内有效；联调/单测/走查用）。"""
    return InMemorySaver(serde=make_serde())


__all__ = [
    "ALLOWED_MSG_PACK_MODULES",
    "make_memory_checkpointer",
    "make_serde",
]
