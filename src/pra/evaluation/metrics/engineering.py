"""工程指标：调用次数 / token / 延迟的分布（均值 + P50 + P95），三方案同口径。

只吃 ``EvalRecord.cost``（``{llm_calls, tool_calls, tokens}``，确定性字段）。**墙钟延迟
不在 EvalRecord 里**（评测要逐字节可重放）—— 只有 real 臂在进程内计时后经 ``latency_ms``
入参传进来，报告另行标注"进程内墙钟、不落 record"。

scripted 路径下 ``tokens=0`` 是真实情况（确定性桩不烧 token），报告如实显示 0，**不伪造、
不用估算值替代**；分位数只有在 real 臂才有多样性。分位数为 nearest-rank（排序后取第
``ceil(p/100·n)`` 个），无插值、无随机，同输入必同输出。
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

from pra.evaluation.harness.base import EvalRecord

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
    """一组数值的 count / mean / p50 / p95 / max（空集 → 除 count 外全 None）。"""

    count: int = 0
    mean: float | None = None
    p50: float | None = None
    p95: float | None = None
    max: float | None = None

    @staticmethod
    def of(values: list[float]) -> DistributionMetrics:
        if not values:
            return DistributionMetrics(count=0)
        return DistributionMetrics(
            count=len(values),
            mean=round(sum(values) / len(values), 2),
            p50=percentile(values, 50),
            p95=percentile(values, 95),
            max=max(values),
        )

    def metric_row(self) -> dict:
        def fmt(value: float | None) -> str:
            return "-" if value is None else f"{value:g}"

        return {
            "n": self.count,
            "mean": fmt(self.mean),
            "p50": fmt(self.p50),
            "p95": fmt(self.p95),
            "max": fmt(self.max),
        }


class EngineeringMetrics(BaseModel):
    """一次运行某方案的工程分布（成本 + 可选延迟）。"""

    total_records: int = 0
    llm_calls: DistributionMetrics = Field(default_factory=DistributionMetrics)
    tool_calls: DistributionMetrics = Field(default_factory=DistributionMetrics)
    tokens: DistributionMetrics = Field(default_factory=DistributionMetrics)
    latency_ms: DistributionMetrics | None = Field(
        default=None, description="仅 real 臂传入（进程内墙钟，不落 record）；scripted 恒 None"
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
