"""OCRTool —— 交叉验证工具（docs/01-agent-loop.md §5.3 /《00》§5）。

回答的业务问题：图片里到底写了什么 —— 用于"标题/描述"与"图片实际内容"的交叉
验证（如 真丝 vs 100% Polyester），发现字段冲突（《00》§5.1）。

分层（依赖倒置）：

- ``OcrProvider``（Protocol）：窄接口 —— 单张图片（URL 或 base64 data-url）
  识别为文字 + 坐标块。返回 None = 确定性"无法识别该图片"（工具转 ok=False）。
- ``MockOcrProvider``：**Mock 默认实现**（显式标注，仅供开发/测试/演示），按
  image_url 种子返回识别文本。真实实现 = OCR 服务（确定性，§5.3），待 infra
  阶段接入。

本工具不含业务判定：是否与商品描述冲突（§5.3 的 ``extra.conflict_hint``，
T-12 待定）属于确定性字段冲突检测器（guardrails/二期），不在工具内做；
OCR 只返回识别事实。
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from pydantic import BaseModel, Field

from ...domain.models import Evidence
from ..base import ToolArgs, ToolContext, ToolResult

# ---- §5.7 受控证据类型 & §5.3 默认证据强度 ----
OCR_TEXT_TYPE = "OCR_TEXT"
OCR_TEXT_WEIGHT = 0.5  # §5.3 默认权重；暂定默认，待 T-5 拍板后可调
OCR_VALUE_MAX_CHARS = 500  # §5.3：value 内嵌 full_text 截断上限（完整文本进结构化负载/审计）


# ---------------------------------------------------------------------------
# 数据访问契约（依赖倒置）
# ---------------------------------------------------------------------------


class OcrBlockBBox(BaseModel):
    """文字块坐标框（§5.3 blocks[].bbox{x,y,w,h}，像素）。"""

    x: int
    y: int
    w: int
    h: int


class OcrBlock(BaseModel):
    """单个文字块（§5.3 blocks[] 元素）。"""

    text: str
    bbox: OcrBlockBBox
    lang: str = Field(default="", description="语言，如 zh / en")
    confidence: float = Field(ge=0.0, le=1.0)


class OcrText(BaseModel):
    """OCR 识别结果（§5.3 result.data：full_text + blocks）。"""

    full_text: str = ""
    blocks: list[OcrBlock] = Field(default_factory=list)


class OcrProvider(Protocol):
    """OCR 服务窄接口。

    返回 None 表示该图片无法识别（确定性无结果 → 工具转 ok=False）；识别到但无
    文字是合法结果（full_text 为空串）。真实服务错误由 infra 层处理。
    """

    async def recognize(self, image: str) -> OcrText | None: ...


_DEFAULT_OCR: Mapping[str, dict[str, Any]] = {
    "https://cdn.example.com/products/P_88231/img1.jpg": {
        "full_text": "FABRIC: 100% Polyester\nCARE: Machine wash cold\nSIZE: 36-40",
        "blocks": [
            {"text": "FABRIC: 100% Polyester", "bbox": {"x": 10, "y": 20, "w": 200, "h": 24}, "lang": "en", "confidence": 0.97},
            {"text": "SIZE: 36-40", "bbox": {"x": 10, "y": 60, "w": 120, "h": 22}, "lang": "en", "confidence": 0.95},
        ],
    },
}


class MockOcrProvider:
    """OcrProvider 的 Mock 默认实现（显式标注，仅供开发/测试/演示）。

    按 ``image``（URL 或 data-url 前缀）精确匹配种子；未命中返回 None
    （模拟"OCR 无法识别该图片"）。
    """

    def __init__(self, data: Mapping[str, dict[str, Any]] | None = None) -> None:
        self._data: dict[str, dict[str, Any]] = dict(data or _DEFAULT_OCR)

    async def recognize(self, image: str) -> OcrText | None:
        row = self._data.get(image)
        if row is None:
            return None
        return OcrText(
            full_text=row.get("full_text", ""),
            blocks=[OcrBlock.model_validate(b) for b in row.get("blocks", [])],
        )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class OcrArgs(ToolArgs):
    """OCRTool 入参（§5.3 args Schema）。"""

    image: str = Field(description="图片 URL 或 base64 data-url")


class OcrResult(ToolResult):
    """OCRTool 出参信封 + 负载（§5.3 result.data 原样平铺）。

    ``ok=False``（无法识别）时 ``full_text`` 为空串、``blocks`` 为空。
    """

    full_text: str = Field(default="", description="识别全文")
    blocks: list[OcrBlock] = Field(default_factory=list, description="带坐标/置信度的文字块")


class OCRTool:
    """识别图片中的文字内容（含坐标与置信度），用于标题/描述与图片实际内容的交叉验证。"""

    name = "OCRTool"
    description = "识别图片中的文字内容（含坐标与置信度），用于标题/描述与图片实际内容的交叉验证"

    def __init__(self, provider: OcrProvider | None = None) -> None:
        self._provider: OcrProvider = provider or MockOcrProvider()

    async def call(self, args: OcrArgs, ctx: ToolContext) -> OcrResult:
        ocr = await self._provider.recognize(args.image)
        if ocr is None:
            return OcrResult(ok=False, error=f"OCR 无法识别该图片: {args.image}")
        return OcrResult(full_text=ocr.full_text, blocks=ocr.blocks)

    def to_evidence(self, result: OcrResult) -> list[Evidence]:
        """结果 → Evidence（§5.3 → Evidence 列）：1 条聚合 OCR_TEXT。

        value 内嵌 full_text（截断 ≤500 字，OCR_VALUE_MAX_CHARS）；blocks 完整
        留在 Result 负载与 tool_call_history（审计），证据链只取人读摘要。
        冲突关键词判定（conflict_hint，T-12）属确定性检测器，不在本工具。
        """
        if not result.ok:
            return []
        truncated = result.full_text[:OCR_VALUE_MAX_CHARS]
        return [
            Evidence(
                type=OCR_TEXT_TYPE,
                source=self.name,
                value=truncated,
                weight=OCR_TEXT_WEIGHT,
                ref_id=None,
            )
        ]
