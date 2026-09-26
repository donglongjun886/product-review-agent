# 4 个 Tool + 统一 Tool 接口。
# tools/base.py 是契约层（统一接口与参数契约）；4 个具体 Tool 在各子包实现
# （product / merchant / case_search / policy_search），每个子包一个 tool.py。
# 数据源一律在**构造时显式注入**（依赖倒置：真实 MySQL / 向量库实现同一 Repository/Index
# 接口后注入即可，工具本体零改动），装配后不再改动工具列表。
# 生产与评测的唯一事实来源 = 真 MySQL + 真 RAG，即 ``build_production_tools()``；
# 测试世界由测试侧自行装配。
from __future__ import annotations

from pra.rag.embedding import BGE_MODEL, production_embedder
from pra.rag.factory import build_case_index, build_policy_index
from pra.rag.lazy_index import LazyCaseIndex, LazyPolicyIndex

from .base import Tool, ToolArgs, ToolContext, ToolResult
from .case_search.tool import CaseIndex, CaseSearchTool
from .merchant.mysql_repo import MySQLMerchantRepository
from .merchant.tool import MerchantTool
from .policy_search.tool import PolicyIndex, PolicySearchTool
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
    """生产与评测入口的工具世界：商品事实与商家行为读真库，案例与政策读真实 RAG。

    ``ProductTool`` → ``MySQLProductRepository``、``MerchantTool`` → ``MySQLMerchantRepository``、
    ``CaseSearchTool`` / ``PolicySearchTool`` → 真实 RAG 索引（llama-index 官方 FastEmbed 编码器
    + hybrid 检索，经 ``Lazy*Index`` **惰性构建**：装配期零 IO，首次检索才建库连服务端）。

    真库表 / Chroma 服务端 / BGE 模型不可用时**装配本身不抛错**：两个 repo 构造期不建 engine，
    RAG 索引构造期不 import 后端、不加载模型；首次检索失败时异常经工具上抛（落库编排层记
    ``R6_INFRA_UNAVAILABLE`` 终态），不静默降级成「无结果」。
    """
    return [
        ProductTool(repo=MySQLProductRepository()),
        MerchantTool(repo=MySQLMerchantRepository()),
        CaseSearchTool(index=LazyCaseIndex(_build_production_case_index)),
        PolicySearchTool(index=LazyPolicyIndex(_build_production_policy_index)),
    ]


# 生产 RAG 检索口径 = hybrid（BM25 + Vector + RRF）；索引层没有模式开关。
# **不在这里给编码器兜底 mock** —— 真模型失败要显式报错，
# 缺 ``--extra rag`` / 服务端不可达 / 模型未缓存都会在首次检索时抛出带指引的错误。


def _build_production_case_index() -> CaseIndex:
    """构建生产 CaseSearch 索引（首次检索时调用；失败原样上抛，不缓存失败）。"""
    return build_case_index(
        embedding_model=production_embedder(),
    )


def _build_production_policy_index() -> PolicyIndex:
    """构建生产 PolicySearch 索引（首次检索时调用；失败原样上抛，不缓存失败）。"""
    return build_policy_index(
        embedding_model=production_embedder(),
    )
