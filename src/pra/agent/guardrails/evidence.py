"""证据质量过滤（tools_node 的确定性 Python 步骤，不调 LLM）。

``quality_filter``：丢弃 ``IMAGE_SIMILARITY`` 且 ``weight < EVIDENCE_MIN_SIM``
（0.70）的弱相似命中，其余类型全部保留（含低相似度的 ``IMAGE_LOGO`` —— logo 是
「检出即事实」，留给业务层判断）。工具层 ``to_evidence`` 只产原始事实、不设下限，
阈值裁决在本层。

证据的结构化派生数值（``similarity`` / ``removals`` / ``policy_id`` …）由各工具在
``to_evidence`` 中从自身的结构化出参直接写入 ``Evidence.extra`` —— 本层不反向解析
人读 ``value``。
"""

from __future__ import annotations

from pra.domain.measurement import EVIDENCE_MIN_SIM
from pra.domain.models import Evidence
from pra.tools.image_analysis.tool import IMAGE_SIMILARITY_TYPE

__all__ = ["quality_filter"]


def quality_filter(raw: list[Evidence], *, evid_min_sim: float = EVIDENCE_MIN_SIM) -> list[Evidence]:
    """丢弃弱相似命中（``weight < evid_min_sim``）；返回新 list、元素复用原引用，入参空 → ``[]``。"""
    kept: list[Evidence] = []
    for e in raw or []:
        if e.type == IMAGE_SIMILARITY_TYPE and e.weight < evid_min_sim:
            continue  # 弱相似命中 —— 噪声，不进证据链
        kept.append(e)
    return kept
