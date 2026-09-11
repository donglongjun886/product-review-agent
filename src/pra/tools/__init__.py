# 6 个 Tool + ToolRegistry + 统一 Tool 接口。
# tools/base.py 是契约层（统一接口与注册骨架）；6 个具体 Tool 在各子包实现
# （product / image_analysis / ocr / merchant / case_search / policy_search），每个子包
# 一个 tool.py，构造时默认注入各自的 InMemory/Mock 数据源（依赖倒置：真实 MySQL /
# 向量库 / OCR 服务实现同一 Repository/Provider/Index 接口后注入即可，工具本体零改动）。
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
    """组装并返回 6 个调查工具（默认注入 InMemory/Mock 数据源）。

    :param data_source: ``"memory"``（默认）= 6 工具 InMemory 种子世界；``"rag"`` =
        CaseSearchTool / PolicySearchTool 注入真实 RAG 索引，其余 4 工具仍为 InMemory 事实世界。
        RAG 索引经 ``pra.rag.factory`` **延迟 import**（默认 memory 路径零额外 import）。
    :param rag_backend: 仅 ``data_source="rag"`` 生效 —— ``"local"``（默认）/ ``"qdrant"`` /
        ``"chroma"``；后两者经 factory **延迟 import**，缺依赖时抛 ``RuntimeError``。
    :param rag_embedder: RAG 检索 embedder（默认 None → factory 缺省 ``MockHashEmbedder``）。
    :param rag_backend_options: 后端专属装配参数透传字典；键名与 ``pra.rag.factory`` 参数
        **逐字对应**，未给键走 factory 缺省。

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
