"""ImageAnalysisTool —— 多模态核心工具。

回答的业务问题：商品外观是否与某品牌/违禁视觉高度相似 —— 外观相似是本案最大、规则无法覆盖
的证据缺口。

分层（依赖倒置）：``ImageAnalysisProvider`` 是窄接口（对单张图片做「图片向量库召回 Top-K +
Logo 检测」，返回结构化条目）；``MockImageAnalysisProvider`` 是**Mock 默认实现**（仅供开发/
测试/演示），按种子数据返回相似度 Top-K 与 Logo；真实实现 = 图片向量库召回 + 视觉 LLM 复核。

**不在本工具内做阈值裁决**：call() 原样返回每张图的 top_similar / logos（含低相似度命中）。
产证据下限 ``EVIDENCE_MIN_SIM=0.70``、矛盾「高相似」分界 ``EVIDENCE_STRONG=0.85`` 属于确定性
证据质量过滤与矛盾检测，留给下游 tools_node/guardrails 层。
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from pydantic import BaseModel, Field

from ...domain.measurement import (
    DIM_IMAGE_APPEARANCE,
    VERDICT_NEGATIVE,
    VERDICT_POSITIVE,
    make_measurement,
)
from ...domain.models import Evidence
from ..base import ToolArgs, ToolContext, ToolResult

# ---- 受控证据类型 ----
IMAGE_SIMILARITY_TYPE = "IMAGE_SIMILARITY"
IMAGE_LOGO_TYPE = "IMAGE_LOGO"

# 产 IMAGE_SIMILARITY 证据的相似度下限 = 0.70；矛盾启发式「高相似」分界 = 0.85
# （``SIM_HIGH_CONTRADICT`` 与 ``EVIDENCE_STRONG`` 同值同义，下游按各自命名引用）。
# 本工具不据此过滤（只返回原始相似度，阈值裁决在确定性层）。
EVIDENCE_MIN_SIM = 0.70
EVIDENCE_STRONG = 0.85


# ---------------------------------------------------------------------------
# 数据访问契约（依赖倒置）
# ---------------------------------------------------------------------------


class BrandMatch(BaseModel):
    """单条品牌款相似命中。

    ``brand_ref`` 指向图片品牌向量库条目的引用（保留可追溯引用）。
    """

    brand_ref: str = Field(description="命中的品牌款引用/名称，如 某品牌经典鞋款")
    similarity: float = Field(ge=0.0, le=1.0, description="外观相似度 0~1")


class LogoHit(BaseModel):

    brand: str = Field(description="识别到的品牌")
    confidence: float = Field(ge=0.0, le=1.0, description="检测置信度 0~1")


class ImageAnalysisItem(BaseModel):

    image_url: str
    top_similar: list[BrandMatch] = Field(default_factory=list, description="按相似度降序")
    logos: list[LogoHit] = Field(default_factory=list)
    visual_risk: str = Field(default="", description="视觉风险描述（人读）")


class ImageAnalysisProvider(Protocol):
    """图片分析服务窄接口（图片向量检索 + 视觉模型复核）。

    未知/无入库图片可返回空条目（等价「没查到」），不应抛异常；真实实现的基础设施级错误由
    infra/ToolNode 层重试与降级，工具不吞异常。
    """

    async def analyze(self, image_url: str, top_k: int, detect_logo: bool) -> ImageAnalysisItem: ...


_DEFAULT_MATCHES: Mapping[str, dict[str, Any]] = {
    "https://cdn.example.com/products/P_88231/img1.jpg": {
        "top_similar": [{"brand_ref": "某品牌经典鞋款", "similarity": 0.91}],
        "logos": [],
        "visual_risk": "外观与某品牌经典复古跑鞋高度相似",
    },
    "https://cdn.example.com/products/P_88231/img2.jpg": {
        "top_similar": [{"brand_ref": "某品牌条纹运动鞋", "similarity": 0.42}],
        "logos": [{"brand": "某品牌", "confidence": 0.93}],
        "visual_risk": "检测到疑似品牌 Logo",
    },
}


class MockImageAnalysisProvider:
    """ImageAnalysisProvider 的 Mock 默认实现（仅供开发/测试/演示）。

    种子数据按 image_url 返回命中；``top_k`` 生效（截断 top_similar）、``detect_logo=False``
    时清空 logos；未知图片返回空条目。
    """

    def __init__(self, data: Mapping[str, dict[str, Any]] | None = None) -> None:
        self._data: dict[str, dict[str, Any]] = dict(data or _DEFAULT_MATCHES)

    async def analyze(self, image_url: str, top_k: int, detect_logo: bool) -> ImageAnalysisItem:
        row = self._data.get(image_url)
        if row is None:
            return ImageAnalysisItem(image_url=image_url)
        similar = row.get("top_similar", [])[:top_k]
        logos = row.get("logos", []) if detect_logo else []
        return ImageAnalysisItem(
            image_url=image_url,
            top_similar=[BrandMatch.model_validate(m) for m in similar],
            logos=[LogoHit.model_validate(l) for l in logos],
            visual_risk=row.get("visual_risk", ""),
        )


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class ImageAnalysisArgs(ToolArgs):

    image_urls: list[str] = Field(..., min_length=1, max_length=5, description="待分析图片 URL 列表（1..5 张）")
    top_k: int = Field(default=5, ge=1, le=10, description="每图返回相似 Top-K（默认 5）")
    detect_logo: bool = Field(default=True, description="是否做 Logo 检测（默认 true）")


class ImageAnalysisResult(ToolResult):
    """ImageAnalysisTool 出参信封 + 负载。

    ``items`` 与入参 ``image_urls`` 一一对应（顺序一致）；图片无命中时该项为空
    top_similar/logos —— 原始结果，不做阈值过滤。
    """

    items: list[ImageAnalysisItem] = Field(default_factory=list)


class ImageAnalysisTool:

    name = "ImageAnalysisTool"
    description = "分析商品图片外观是否与知名品牌款/违禁视觉高度相似；返回相似度 Top-K、Logo 检测、视觉风险描述"
    args_model = ImageAnalysisArgs
    measured_dimensions: frozenset[str] = frozenset({DIM_IMAGE_APPEARANCE})
    # 本部署是否真的能产出外观测量：评测/演示世界用**种子数据源**（Mock 即数据源）→ True；
    # 生产装配是**冻结的 Mock 桩**（真实商品图永远空命中）→ build_production_tools() 传 False
    # ⇒ gate 侧判 UNMEASURABLE（不得把"桩测不出"当成"测过且阴性"）。
    measurement_available: bool = True

    def __init__(
        self,
        provider: ImageAnalysisProvider | None = None,
        *,
        measurement_available: bool = True,
    ) -> None:
        self._provider: ImageAnalysisProvider = provider or MockImageAnalysisProvider()
        self.measurement_available = measurement_available

    async def call(self, args: ImageAnalysisArgs, ctx: ToolContext) -> ImageAnalysisResult:
        items = [
            await self._provider.analyze(url, top_k=args.top_k, detect_logo=args.detect_logo)
            for url in args.image_urls
        ]
        return ImageAnalysisResult(items=items)

    def to_evidence(self, result: ImageAnalysisResult) -> list[Evidence]:
        """结果 → Evidence：命中 → 阳性类型；每张图**必产 1 条** ``MEASUREMENT``。

        weight 分别取 ``similarity`` / ``confidence``。**不套 EVIDENCE_MIN_SIM 下限**，下游按
        上述常量做证据质量过滤。``ref_id=item.image_url``（源图片为稳定业务
        标识，去重 key 以它为准 —— 同一品牌多图命中不会因 ref=None 互相吞并）；
        ``extra.similarity`` 等派生数值由 tools_node 的 backfill_extra 回填，本工具不写。

        ``MEASUREMENT`` 的 verdict 相对 ``EVIDENCE_STRONG`` **硬阈值**判定：
        有 ``similarity >= EVIDENCE_STRONG`` 或任一 Logo → ``POSITIVE``；否则 ``NEGATIVE``。
        故**弱相似（0.70~0.85）算 NEGATIVE**（未达处置阈值）—— 它仍作为 ``IMAGE_SIMILARITY``
        证据留在链里供 LLM 参考，但不构成"风险阳性"，也不会阻塞 PASS（其应然处置是
        "需与商品事实交叉"，由 required set 的 listing_registry 覆盖承担）。
        """
        evidences: list[Evidence] = []
        for item in result.items:
            strong_hit = False
            for m in item.top_similar:
                if m.similarity >= EVIDENCE_STRONG:
                    strong_hit = True
                evidences.append(
                    Evidence(
                        type=IMAGE_SIMILARITY_TYPE,
                        source=self.name,
                        value=f"similarity={m.similarity:.2f}, match={m.brand_ref}",
                        weight=m.similarity,
                        ref_id=item.image_url,
                    )
                )
            for logo in item.logos:
                strong_hit = True
                evidences.append(
                    Evidence(
                        type=IMAGE_LOGO_TYPE,
                        source=self.name,
                        value=f"logo={logo.brand}, conf={logo.confidence:.2f}",
                        weight=logo.confidence,
                        ref_id=item.image_url,
                    )
                )
            top = max((m.similarity for m in item.top_similar), default=0.0)
            evidences.append(
                make_measurement(
                    dimension=DIM_IMAGE_APPEARANCE,
                    source=self.name,
                    source_ref=item.image_url,
                    verdict=VERDICT_POSITIVE if strong_hit else VERDICT_NEGATIVE,
                    # 测量可信度按"该图确实被分析过"给固定高值；风险强度由阳性证据自身承载。
                    weight=EVIDENCE_STRONG,
                    value=(
                        f"外观比对完成：最高相似度 {top:.2f}、Logo 命中 {len(item.logos)} 个"
                        f"（处置阈值 {EVIDENCE_STRONG}）"
                    ),
                )
            )
        return evidences
