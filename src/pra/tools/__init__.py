# 4 个 Tool + 统一 Tool 接口。
# tools/base.py 是契约层（统一接口与参数契约）；4 个具体 Tool 在各子包实现
# （product / merchant / case_search / policy_search），每个子包
# 一个 tool.py，构造时默认注入各自的 InMemory 数据源（依赖倒置：真实 MySQL /
# 向量库实现同一 Repository/Index 接口后注入即可，工具本体零改动）。
# 两条装配路径：``build_tools()`` = 默认 InMemory 世界（CI / 单测）；
# ``build_production_tools()`` = 生产与评测入口（商品与商家读 MySQL、案例与政策读真实 RAG）。
# 数据源一律**构造时注入**（``build_tools(..., case_index=...)``），装配后不再改动工具列表。
from __future__ import annotations

from pra.rag.embedding import BGE_MODEL, production_embedder
from pra.rag.factory import build_case_index, build_policy_index
from pra.rag.lazy_index import LazyCaseIndex, LazyPolicyIndex

from .base import Tool, ToolArgs, ToolContext, ToolResult
from .case_search.tool import CaseIndex, CaseSearchTool
from .merchant.mysql_repo import MySQLMerchantRepository
from .merchant.tool import MerchantRepository, MerchantTool
from .policy_search.tool import PolicyIndex, PolicySearchTool
from .product.mysql_repo import MySQLProductRepository
from .product.tool import ProductRepository, ProductTool

__all__ = [
    "BGE_MODEL",
    "Tool",
    "ToolArgs",
    "ToolContext",
    "ToolResult",
    "build_production_tools",
    "build_tools",
    "production_embedder",
]


def build_tools(
    *,
    product_repo: ProductRepository | None = None,
    merchant_repo: MerchantRepository | None = None,
    case_index: CaseIndex | None = None,
    policy_index: PolicyIndex | None = None,
) -> list[Tool]:
    """组装并返回 4 个调查工具（数据源全部在**构造时**注入，默认 InMemory）。

    :param product_repo: ProductTool 的数据源；默认 **None → InMemory**（CI 不连库、评测可
        重放）。要读真库须**显式**传入 ``pra.tools.product.mysql_repo.MySQLProductRepository()``
        —— 连库与否由此入参单点决定（repo 构造期不建 engine，首次查询才连库）。生产/HTTP 装配
        即 ``build_production_tools()``（调用方持有 repo 生命周期）。
    :param merchant_repo: MerchantTool 的数据源，语义同 ``product_repo``（默认 InMemory，
        显式传 ``pra.tools.merchant.mysql_repo.MySQLMerchantRepository()`` 才读真库）。
    :param case_index: CaseSearchTool 的检索索引；默认 None → ``InMemoryCaseIndex`` 种子世界。
    :param policy_index: PolicySearchTool 的检索索引；默认 None → ``InMemoryPolicyIndex``。

    每个工具类可用作结构性 ``Tool``，由 tools_node 按名调度；替换数据源只需换构造入参。
    """
    tools: list[Tool] = [
        ProductTool(repo=product_repo),
        MerchantTool(repo=merchant_repo),
        CaseSearchTool(index=case_index),
        PolicySearchTool(index=policy_index),
    ]
    return tools


def build_production_tools() -> list[Tool]:
    """生产/HTTP 入口的工具世界：商品事实与商家行为读真库，案例与政策读真实 RAG。

    相对 ``build_tools()`` 的差别（共 4 个工具的数据源）：
    ``ProductTool`` → ``MySQLProductRepository``、``MerchantTool`` → ``MySQLMerchantRepository``、
    ``CaseSearchTool`` / ``PolicySearchTool`` → 真实 RAG 索引（llama-index 官方 FastEmbed 编码器
    + hybrid 检索，经 ``Lazy*Index`` **惰性构建**：装配期零 IO，首次检索才建库连服务端）。

    **默认装配路径（``build_tools()``）仍是 InMemory** —— 单测与 CI 不连库/不连 Chroma、评测
    可重放；只有生产入口（HTTP 路由 / 落库编排）走本函数。仓库测试有 autouse fixture 把本函数
    钉回 ``build_tools()``（见 ``tests/conftest.py``）。

    真库表 / Chroma 服务端 / BGE 模型不可用时**装配本身不抛错**：两个 repo 构造期不建 engine，
    RAG 索引构造期不 import 后端、不加载模型；失败由工具层记 error record（warn failure），
    不静默降级成「无结果」。
    """
    return build_tools(
        product_repo=MySQLProductRepository(),
        merchant_repo=MySQLMerchantRepository(),
        case_index=LazyCaseIndex(_build_production_case_index),
        policy_index=LazyPolicyIndex(_build_production_policy_index),
    )


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
