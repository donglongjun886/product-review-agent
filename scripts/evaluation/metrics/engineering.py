"""工程指标：调用次数 / token / 延迟的分布（均值 + P50 + P95）。"""

from __future__ import annotations

import math

from evaluation.record import EvalRecord
from pydantic import BaseModel, Field

__all__ = [
    "DistributionMetrics",
    "EngineeringEvaluator",
    "EngineeringMetrics",
    "percentile",
]


def percentile(values: list[float], p: float) -> float | None:
    """nearest-rank 分位数（``values`` 可为无序；空列表 → None）。"""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, min(len(ordered), math.ceil(p / 100 * len(ordered))))
    return ordered[rank - 1]


class DistributionMetrics(BaseModel):
    """一组数值的 count / mean / p50 / p95（空集 → 除 count 外全 None）。"""

    count: int = 0
    mean: float | None = None
    p50: float | None = None
    p95: float | None = None

    @staticmethod
    def of(values: list[float]) -> DistributionMetrics:
        if not values:
            return DistributionMetrics(count=0)
        return DistributionMetrics(
            count=len(values),
            mean=round(sum(values) / len(values), 2),
            p50=percentile(values, 50),
            p95=percentile(values, 95),
        )


class EngineeringMetrics(BaseModel):
    """一次运行某方案的工程分布（成本 + 可选延迟）。"""

    total_records: int = 0
    llm_calls: DistributionMetrics = Field(default_factory=DistributionMetrics)
    tool_calls: DistributionMetrics = Field(default_factory=DistributionMetrics)
    tokens: DistributionMetrics = Field(default_factory=DistributionMetrics)
    latency_ms: DistributionMetrics | None = Field(
        default=None, description="跑分脚本进程内墙钟（不落 record）；未传恒 None"
    )


class EngineeringEvaluator:
    @staticmethod
    def evaluate(
        records: list[EvalRecord], *, latency_ms: list[float] | None = None
    ) -> EngineeringMetrics:
        llm = [float(r.cost.get("llm_calls") or 0) for r in records]
        tools = [float(r.cost.get("tool_calls") or 0) for r in records]
        tokens = [float(r.cost.get("tokens") or 0) for r in records]
        return EngineeringMetrics(
            total_records=len(records),
            llm_calls=DistributionMetrics.of(llm),
            tool_calls=DistributionMetrics.of(tools),
            tokens=DistributionMetrics.of(tokens),
            latency_ms=DistributionMetrics.of(list(latency_ms)) if latency_ms else None,
        )
