# 6 个 Tool + ToolRegistry + 统一 Tool 接口。
# tools/base.py 是契约层（统一接口与注册骨架）；6 个具体 Tool 在各子包实现
# （product / image_analysis / ocr / merchant / case_search / policy_search），每个子包
# 一个 tool.py，构造时默认注入各自的 InMemory/Mock 数据源（依赖倒置：真实 MySQL /
# 向量库 / OCR 服务实现同一 Repository/Provider/Index 接口后注入即可，工具本体零改动）。
# 两条装配路径：``build_tools()`` = 默认 InMemory/Mock 世界（评测/CI/可重放）；
# ``build_production_tools()`` = 生产/HTTP 入口（商品与商家读 MySQL、案例与政策读真实 RAG）。
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from .base import Tool, ToolArgs, ToolContext, ToolRegistry, ToolResult

if TYPE_CHECKING:  # 仅注解用：默认装配路径不 import 这些工具子包
    from pra.rag.embedder import BgeEmbedder
    from pra.rag.retrieval import RetrievalMode

    from .case_search.tool import CaseIndex
    from .merchant.tool import MerchantRepository
    from .policy_search.tool import PolicyIndex
    from .product.tool import ProductRepository

__all__ = [
    "Tool",
    "ToolArgs",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "build_production_tools",
    "build_tools",
]


def build_tools(
    data_source: Literal["memory", "rag"] = "memory",
    rag_backend: Literal["local", "qdrant", "chroma"] = "local",
    rag_embedder: Any | None = None,
    *,
    rag_backend_options: dict[str, Any] | None = None,
    product_repo: ProductRepository | None = None,
    merchant_repo: MerchantRepository | None = None,
    vision_measurement_available: bool = True,
) -> list[Tool]:
    """组装并返回 6 个调查工具（默认注入 InMemory/Mock 数据源）。

    :param data_source: ``"memory"``（默认）= 6 工具 InMemory 种子世界；``"rag"`` =
        CaseSearchTool / PolicySearchTool 注入真实 RAG 索引，其余 4 工具仍为 InMemory 事实世界。
        RAG 索引经 ``pra.rag.factory`` **延迟 import**（默认 memory 路径零额外 import）。
    :param rag_backend: 仅 ``data_source="rag"`` 生效 —— ``"local"``（默认）/ ``"qdrant"`` /
        ``"chroma"``；后两者经 factory **延迟 import**，缺依赖时抛 ``RuntimeError``。
    :param rag_embedder: RAG 检索 embedder（默认 None → factory 缺省 ``MockHashEmbedder``）。
    :param rag_backend_options: 后端专属装配参数透传字典；键名与 ``pra.rag.factory`` 参数
        **逐字对应**，未给键走 factory 缺省。
    :param product_repo: ProductTool 的数据源；默认 **None → InMemory**（CI 不连库、评测可
        重放）。要读真库须**显式**传入 ``pra.tools.product.mysql_repo.MySQLProductRepository()``
        —— 该模块自身不在本函数里 import，连库与否由此入参单点决定。生产/HTTP 装配即
        ``build_production_tools()``（调用方持有 repo 生命周期）。
    :param merchant_repo: MerchantTool 的数据源，语义同 ``product_repo``（默认 InMemory，
        显式传 ``pra.tools.merchant.mysql_repo.MySQLMerchantRepository()`` 才读真库）。

    每个工具类可用作结构性 ``Tool``，经 ``ToolRegistry.register`` 注册后由 tools_node 调度；
    替换真实数据源只需换构造入参，本函数保持不变。
    """
    # 延迟 import：避免 pra.tools 包导入期拉起全部子包（防循环/省启动）。
    from .case_search.tool import CaseSearchTool
    from .image_analysis.tool import ImageAnalysisTool
    from .merchant.tool import MerchantTool
    from .ocr.tool import OCRTool
    from .policy_search.tool import PolicySearchTool
    from .product.tool import ProductTool

    tools: list[Tool] = [
        ProductTool(repo=product_repo),
        ImageAnalysisTool(measurement_available=vision_measurement_available),
        OCRTool(),
        MerchantTool(repo=merchant_repo),
        CaseSearchTool(),
        PolicySearchTool(),
    ]
    if data_source == "rag":
        # 真实 RAG 索引替换两个"知识库检索"工具的数据源（其余 4 工具不受影响）。
        # 延迟 import：pra.rag 只有在显式选择 rag 数据源时才被拉起（防循环/省启动）。
        from pra.rag.factory import build_case_index, build_policy_index

        options = dict(rag_backend_options or {})
        tools[4] = CaseSearchTool(
            index=build_case_index(backend=rag_backend, embedder=rag_embedder, **options)
        )
        tools[5] = PolicySearchTool(
            index=build_policy_index(backend=rag_backend, embedder=rag_embedder, **options)
        )
    return tools


def build_production_tools() -> list[Tool]:
    """生产/HTTP 入口的工具世界：商品事实与商家行为读真库，案例与政策读真实 RAG。

    相对 ``build_tools()`` 的差别（共 3 个工具的数据源）：
    ``ProductTool`` → ``MySQLProductRepository``、``MerchantTool`` → ``MySQLMerchantRepository``、
    ``CaseSearchTool`` / ``PolicySearchTool`` → 真实 RAG 索引（``rag_backend="chroma"`` +
    ``BgeEmbedder`` + hybrid 检索，经 ``Lazy*Index`` **惰性构建**：装配期零 import/零 IO，
    首次检索才建库连服务端）。其余 2 个（image_analysis / ocr）仍是 Mock 桩。

    **默认装配路径（``build_tools()`` 与 ``build_agent_graph()`` 缺省）仍是 InMemory** ——
    单测与 CI 不连库/不连 Chroma、评测可重放；只有生产入口（HTTP 路由 / 落库编排）走本函数。
    仓库测试有 autouse fixture 把本函数钉回 ``build_tools()``（见 ``tests/conftest.py``）。

    三条链路都延迟到首次工具调用：两个 repo 构造期不建 engine；RAG 索引构造期不 import
    chromadb/llama_index、不加载模型。故真库表 / Chroma 服务端 / BGE 模型不可用时**装配本身
    也不抛错**，失败由工具层记 error record（warn failure），不静默降级成「无结果」。
    """
    # 延迟 import：默认装配路径不拉 pra.infra / pra.rag
    from pra.rag.lazy_index import LazyCaseIndex, LazyPolicyIndex

    from .case_search.tool import CaseSearchTool
    from .merchant.mysql_repo import MySQLMerchantRepository
    from .policy_search.tool import PolicySearchTool
    from .product.mysql_repo import MySQLProductRepository

    tools = build_tools(
        product_repo=MySQLProductRepository(),
        merchant_repo=MySQLMerchantRepository(),
        # 生产视觉链路仍是**冻结的 Mock 桩**（真实商品图永远空命中）⇒ 声明"外观维度不可测"。
        # 若不声明，桩的"零命中"会被 gate 当成"测过且阴性"，把"测不出"误判成"证明无风险"。
        vision_measurement_available=False,
    )
    tools[4] = CaseSearchTool(index=LazyCaseIndex(_build_production_case_index))
    tools[5] = PolicySearchTool(index=LazyPolicyIndex(_build_production_policy_index))
    return tools


# 生产 RAG 检索口径：真实后端（ChromaDB + LlamaIndex 装配 + BGE + BM25(jieba) + RRF）与
# hybrid 三路融合。**不在这里给 embedder 兜底 mock** —— 真模型失败要显式报错（见 BgeEmbedder），
# 缺 ``--extra rag`` / 服务端不可达 / 模型未缓存都会在首次检索时抛出带指引的错误。
_PRODUCTION_RAG_BACKEND: Literal["chroma"] = "chroma"
_PRODUCTION_RAG_MODE: RetrievalMode = "hybrid"


def _production_rag_embedder() -> BgeEmbedder:
    """生产检索用真语义 embedder；**请求期绝不下载模型**（缺依赖/未缓存即快速报错）。

    服务器请求线程里下载 ~90MB 模型会把一次审核拖成分钟级并可能被墙挂死。首次部署须预热一次
    （设 ``HF_ENDPOINT`` 下载到 ``PRA_EMBED_CACHE_DIR`` 指向的目录），否则这里抛错 → 工具层记
    warn failure，检索降级但不阻塞审核，也**不静默回退 MockHashEmbedder**。
    """
    from pra.rag.embedder import BgeEmbedder

    if not BgeEmbedder.available():
        raise RuntimeError(
            "生产 RAG 需要 rag extra：`uv sync --extra rag --extra observability`"
        )
    embedder = BgeEmbedder()
    if not embedder.model_ready():
        raise RuntimeError(
            "生产 RAG 的 BGE 模型未缓存（生产路径不在请求期下载模型）。预热一次：设 "
            "`HF_ENDPOINT=https://hf-mirror.com` 并让 `BgeEmbedder().embed('预热')` 跑通，"
            "或用 `PRA_EMBED_CACHE_DIR` 指向已缓存的模型目录。"
        )
    return embedder


def _build_production_case_index() -> CaseIndex:
    """构建生产 CaseSearch 索引（首次检索时调用；失败原样上抛，不缓存失败）。"""
    from pra.rag.factory import build_case_index

    return build_case_index(
        backend=_PRODUCTION_RAG_BACKEND,
        mode=_PRODUCTION_RAG_MODE,
        embedder=_production_rag_embedder(),
    )


def _build_production_policy_index() -> PolicyIndex:
    """构建生产 PolicySearch 索引（首次检索时调用；失败原样上抛，不缓存失败）。"""
    from pra.rag.factory import build_policy_index

    return build_policy_index(
        backend=_PRODUCTION_RAG_BACKEND,
        mode=_PRODUCTION_RAG_MODE,
        embedder=_production_rag_embedder(),
    )
