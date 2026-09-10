# 6 个 Tool + ToolRegistry + 统一 Tool 接口
# 契约层（tools/base.py）：统一接口与注册骨架，不接具体实现。
# 6 个具体 Tool 在各子包实现（product / image_analysis / ocr / merchant /
# case_search / policy_search），每个子包一个 tool.py，构造时默认注入各自的
# InMemory/Mock 数据源（依赖倒置：真实 MySQL/向量库/OCR 服务在 infra 阶段接入，
# 实现同一 Repository/Provider/Index 接口后注入即可，工具本体零改动）。
from __future__ import annotations

from typing import Any, Literal

from .base import Tool, ToolArgs, ToolContext, ToolRegistry, ToolResult

__all__ = ["Tool", "ToolArgs", "ToolContext", "ToolRegistry", "ToolResult", "build_tools"]


def build_tools(
    data_source: Literal["memory", "rag"] = "memory",
    rag_backend: Literal["local", "qdrant", "chroma"] = "local",
    rag_embedder: Any | None = None,
    *,
    rag_backend_options: dict[str, Any] | None = None,
) -> list[Tool]:
    """组装并返回 6 个调查工具（默认注入 InMemory/Mock 数据源，开箱可测）。

    :param data_source: 数据源开关（rag-implementation-plan.md R-3 拍板）——
        ``"memory"``（默认）= 现有 6 工具 InMemory 种子世界（**逐字节不变**，回归
        不破坏）；``"rag"`` = CaseSearchTool / PolicySearchTool 注入真实 RAG 索引
        （Policy KB / Case KB，确定性 mock embedding + BM25 + 余弦，三模式可切），
        其余 4 工具（product/image/ocr/merchant）仍为 InMemory 事实世界。
        RAG 索引经 ``pra.rag.factory`` **延迟 import**（防 pra.tools 导入期拉起
        pra.rag → 循环依赖风险；默认 memory 路径零额外 import）。
    :param rag_backend: RAG 检索后端开关（docs/06 §2.3 / docs/10 §2 扩展，仅
        ``data_source="rag"`` 生效）——``"local"``（默认）= 既有
        RagPolicyIndex/RagCaseIndex（内存 numpy 余弦，**行为不变**）；``"qdrant"`` =
        QdrantPolicyIndex/QdrantCaseIndex；``"chroma"`` = ChromaPolicyIndex/
        ChromaCaseIndex（ChromaDB + LlamaIndex：VectorRetriever / BM25Retriever(jieba) /
        QueryFusionRetriever-RRF）。后两者均经 factory **延迟 import**，缺依赖时构造抛
        ``RuntimeError`` 提示 ``uv sync --extra rag``；无对应依赖的环境默认路径不受影响。
    :param rag_embedder: RAG 检索 embedder（默认 None → factory 内部缺省
        ``MockHashEmbedder``）——chroma / qdrant 后端也可离线单测（mock embedder）；
        Phase 2 换真实本地模型（如 BGE）时由此注入。
    :param rag_backend_options: 后端专属装配参数的透传字典（默认 None = 不传）。
        用于 chroma 后端注入 ``chroma_client`` / ``chroma_ephemeral`` / ``chroma_host`` /
        ``chroma_port``（测试用 ``EphemeralClient`` 离线跑）或 qdrant 后端注入
        ``qdrant_client`` / ``location``；键名与 ``pra.rag.factory`` 构造参数**逐字对应**，
        未给键一律走 factory 自身缺省（默认路径行为不变）。

    每个工具类可用作结构性 ``Tool``（name/description/async call），后续
    tools_node 阶段经 ``ToolRegistry.register`` 注册。真实数据源替换示例：
    ``ProductTool(repo=MySQLProductRepository(session))`` —— 由 infra 阶段接线，
    本函数保持不变即可（或届时改为按配置注入）。
    """
    # 延迟 import：避免 pra.tools 包导入期拉起全部子包（防循环/省启动）。
    from .case_search.tool import CaseSearchTool
    from .image_analysis.tool import ImageAnalysisTool
    from .merchant.tool import MerchantTool
    from .ocr.tool import OCRTool
    from .policy_search.tool import PolicySearchTool
    from .product.tool import ProductTool

    tools: list[Tool] = [
        ProductTool(),
        ImageAnalysisTool(),
        OCRTool(),
        MerchantTool(),
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
