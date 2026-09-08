# 6 个 Tool + ToolRegistry + 统一 Tool 接口
# 契约层（tools/base.py）：统一接口与注册骨架，不接具体实现。
# 6 个具体 Tool 在各子包实现（product / image_analysis / ocr / merchant /
# case_search / policy_search），每个子包一个 tool.py，构造时默认注入各自的
# InMemory/Mock 数据源（依赖倒置：真实 MySQL/向量库/OCR 服务在 infra 阶段接入，
# 实现同一 Repository/Provider/Index 接口后注入即可，工具本体零改动）。
from .base import Tool, ToolArgs, ToolContext, ToolRegistry, ToolResult

__all__ = ["Tool", "ToolArgs", "ToolContext", "ToolRegistry", "ToolResult", "build_tools"]


def build_tools() -> list[Tool]:
    """组装并返回 6 个调查工具（各自带 InMemory/Mock 默认数据源，开箱可测）。

    每个工具类可用作结构性 ``Tool``（name/description/async call），后续
    tools_node 阶段经 ``ToolRegistry.register`` 注册。真实数据源替换示例：
    ``ProductTool(repo=MySQLProductRepository(session))`` —— 由 infra 阶段接线，
    本函数保持不变即可（或届时改为按配置注入）。
    """
    # 延迟 import：避免 pra.tools 包导入期拉起全部子包（防循环/省启动）。
    from .product.tool import ProductTool
    from .image_analysis.tool import ImageAnalysisTool
    from .ocr.tool import OCRTool
    from .merchant.tool import MerchantTool
    from .case_search.tool import CaseSearchTool
    from .policy_search.tool import PolicySearchTool

    return [
        ProductTool(),
        ImageAnalysisTool(),
        OCRTool(),
        MerchantTool(),
        CaseSearchTool(),
        PolicySearchTool(),
    ]
