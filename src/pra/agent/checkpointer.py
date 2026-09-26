"""Checkpointer 工厂：LangGraph 线程状态快照的 (de)serialize 与介质工厂（进程内 ``InMemorySaver``）。"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# msgpack 扩展类型白名单：元素为 ("pra.domain.models", "<类名>")。
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
    ("pra.domain.models", "ReviewDecision"),
]


def make_serde() -> JsonPlusSerializer:
    """构造线程状态 (de)serializer。"""
    return JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_MSG_PACK_MODULES)


def make_memory_checkpointer() -> InMemorySaver:
    """进程内 Checkpointer 工厂（``InMemorySaver``）。"""
    return InMemorySaver(serde=make_serde())


__all__ = [
    "ALLOWED_MSG_PACK_MODULES",
    "make_memory_checkpointer",
    "make_serde",
]
