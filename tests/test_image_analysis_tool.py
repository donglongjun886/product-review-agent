"""ImageAnalysisTool.to_evidence 的「测量能力开关」单测（直接调工具，不经图）。

锁住的事实侧契约：``measurement_available=False``（本部署外观测不出）时**不产 ``MEASUREMENT``**
—— "未测量"不得被写成"测过且阴性"；原始命中证据（``IMAGE_SIMILARITY`` / ``IMAGE_LOGO``）照常
输出。``measurement_available=True`` 时每张图产 1 条 ``MEASUREMENT``。
"""

from __future__ import annotations

from pra.domain.measurement import (
    DIM_IMAGE_APPEARANCE,
    MEASUREMENT_TYPE,
    measurement_dimension,
)
from pra.domain.models import Budget, Evidence
from pra.tools.base import ToolContext
from pra.tools.image_analysis.tool import (
    IMAGE_LOGO_TYPE,
    IMAGE_SIMILARITY_TYPE,
    ImageAnalysisArgs,
    ImageAnalysisTool,
)

# _DEFAULT_MATCHES 中的两张种子图：img1 = 0.91 强相似；img2 = 0.42 弱相似 + Logo 0.93。
_IMG_STRONG = "https://cdn.example.com/products/P_88231/img1.jpg"
_IMG_LOGO = "https://cdn.example.com/products/P_88231/img2.jpg"


def _ctx() -> ToolContext:
    return ToolContext(run_id="R_TEST", case_id="C_TEST", budget=Budget())


async def _evidence(measurement_available: bool) -> list[Evidence]:
    """跑一遍工具（Mock 默认 provider）并返回 to_evidence 结果。"""
    tool = ImageAnalysisTool(measurement_available=measurement_available)
    args = ImageAnalysisArgs(image_urls=[_IMG_STRONG, _IMG_LOGO])
    result = await tool.call(args, _ctx())
    return tool.to_evidence(result)


async def test_unavailable_measurement_emits_no_measurement_evidence():
    """不可测（measurement_available=False）：零 ``MEASUREMENT``，原始命中证据照常产出。"""
    evidences = await _evidence(measurement_available=False)

    assert [e for e in evidences if e.type == MEASUREMENT_TYPE] == []

    sims = [e for e in evidences if e.type == IMAGE_SIMILARITY_TYPE]
    assert {e.ref_id for e in sims} == {_IMG_STRONG, _IMG_LOGO}
    assert max(e.weight for e in sims) == 0.91

    logos = [e for e in evidences if e.type == IMAGE_LOGO_TYPE]
    assert [e.ref_id for e in logos] == [_IMG_LOGO]
    assert logos[0].weight == 0.93


async def test_available_measurement_emits_one_measurement_per_image():
    """可测（measurement_available=True）：每张图 1 条 ``MEASUREMENT``（维度 image_appearance）。"""
    evidences = await _evidence(measurement_available=True)

    measurements = [e for e in evidences if e.type == MEASUREMENT_TYPE]
    assert {measurement_dimension(e) for e in measurements} == {DIM_IMAGE_APPEARANCE}
    assert {e.ref_id for e in measurements} == {
        f"{DIM_IMAGE_APPEARANCE}:{_IMG_STRONG}",
        f"{DIM_IMAGE_APPEARANCE}:{_IMG_LOGO}",
    }
    # 开关只影响 MEASUREMENT：原始 IMAGE_SIMILARITY / IMAGE_LOGO 证据与关闭时同构。
    assert {e.type for e in evidences} == {
        IMAGE_SIMILARITY_TYPE,
        IMAGE_LOGO_TYPE,
        MEASUREMENT_TYPE,
    }
