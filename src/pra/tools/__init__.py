# 4 个 Tool + 统一 Tool 接口。
from __future__ import annotations

from pra.rag.dto import CaseIndex, PolicyIndex
from pra.rag.embedding import BGE_MODEL, production_embedder
from pra.rag.factory import build_case_index, build_policy_index
from pra.rag.lazy_index import LazyCaseIndex, LazyPolicyIndex

from .base import Tool, ToolArgs, ToolContext, ToolResult
from .case_search.tool import CaseSearchTool
from .merchant.mysql_repo import MySQLMerchantRepository
from .merchant.tool import MerchantTool
from .policy_search.tool import PolicySearchTool
from .product.mysql_repo import MySQLProductRepository
from .product.tool import ProductTool

__all__ = [
    "BGE_MODEL",
    "Tool",
    "ToolArgs",
    "ToolContext",
    "ToolResult",
    "build_production_tools",
    "production_embedder",
]


def build_production_tools() -> list[Tool]:
    """生产与评测入口的工具世界：商品事实与商家行为读真库，案例与政策读真实 RAG。"""
    return [
        ProductTool(repo=MySQLProductRepository()),
        MerchantTool(repo=MySQLMerchantRepository()),
        CaseSearchTool(index=LazyCaseIndex(_build_production_case_index)),
        PolicySearchTool(index=LazyPolicyIndex(_build_production_policy_index)),
    ]


def _build_production_case_index() -> CaseIndex:
    """构建生产 CaseSearch 索引（首次检索时调用）。"""
    return build_case_index(
        embedding_model=production_embedder(),
    )


def _build_production_policy_index() -> PolicyIndex:
    """构建生产 PolicySearch 索引（首次检索时调用）。"""
    return build_policy_index(
        embedding_model=production_embedder(),
    )
