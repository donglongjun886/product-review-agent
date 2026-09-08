"""EvaluationRunner —— Phase 1 三方案对比最小闭环编排（evaluation/runner.py）。

流程（docs/00-system-design.md §11.4 / 02-evaluation.md §8 M2）::

    load 数据集（JSONL → EvalCase[]）
        ↓
    （可选 smoke 子集：确定性取前 N 条）
        ↓
    对每个 SchemeRunner（rule / single_call_llm / agent）跑同一数据集
        ↓
    统一 EvalRecord（DecisionEvaluator 只吃它）
        ↓
    总体 + 按 scene 分层指标（Accuracy/Precision/Recall/FPR/FNR + HRR/Automation）
        ↓
    report.py 打印 Console Report

确定性约定（可重放断言的基础）：
- 数据集行序即遍历序；三方案**顺序串行**执行（不并发），避免共享态/调度抖动；
- Agent 每 case 独立 build 图 + 唯一 thread_id；进程级 LLM 后端每次运行后恢复默认；
- EvalRecord 不含墙钟字段（latency_ms 不进 cost）→ 同数据重跑产出可逐字节比对。

不落 DB：全部在内存完成（评测以内存 EvalRecord 为主，快、不污染业务表 ——
docs/02-evaluation.md §3.6"DB 落库非必需"）。
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from pra.evaluation.dataset.loader import load_dataset, scene_stats, smoke_subset
from pra.evaluation.dataset.schema import EvalCase
from pra.evaluation.harness.agent_scheme import AgentScheme
from pra.evaluation.harness.base import EvalContext, EvalRecord, SchemeRunner
from pra.evaluation.harness.rule_scheme import RuleBaseline
from pra.evaluation.harness.single_call_scheme import SingleCallScheme
from pra.evaluation.metrics.business import DecisionEvaluator, DecisionMetrics

__all__ = [
    "ALL_SCHEMES",
    "EvaluationResult",
    "EvaluationRunner",
    "expected_index",
]

ALL_SCHEMES: tuple[str, ...] = ("rule", "single_call_llm", "agent")


def expected_index(cases: list[EvalCase]) -> dict[str, dict]:
    """由数据集构造 expected 索引：``{eval_case_id: {"decision": ..., "scene": ...}}``。"""
    return {
        c.eval_case_id: {"decision": c.expected.decision, "scene": c.scene}
        for c in cases
    }


class EvaluationResult(BaseModel):
    """一次评测运行的全部产物（报告 / 测试断言共同消费）。"""

    data_path: str | None = Field(default=None, description="数据集路径（None=外部注入 cases）")
    smoke: bool = Field(default=False)
    total_cases: int = Field(default=0)
    scene_stats: dict = Field(default_factory=dict)
    expected: dict[str, dict] = Field(default_factory=dict)  # {eval_case_id: {decision, scene}}
    records: dict[str, list[EvalRecord]] = Field(default_factory=dict)  # scheme → records
    overall: dict[str, DecisionMetrics] = Field(default_factory=dict)  # scheme → metrics
    grouped: dict[str, dict[str, DecisionMetrics]] = Field(default_factory=dict)  # scheme → scene → metrics
    cost_summary: dict[str, dict] = Field(default_factory=dict)  # scheme → {llm_calls, tool_calls, tokens} 均值

    @property
    def all_records(self) -> list[EvalRecord]:
        """全部 records 拍平（保序：按 scheme 声明序 × 数据行序）。"""
        out: list[EvalRecord] = []
        for name in ALL_SCHEMES:
            out.extend(self.records.get(name) or [])
        return out


class EvaluationRunner:
    """三方案对比运行器（Phase 1：不落 DB、确定性、CI 可跑）。"""

    def __init__(
        self,
        *,
        data_path: str | Path | None = None,
        ctx: EvalContext | None = None,
    ) -> None:
        self.data_path = Path(data_path) if data_path is not None else None
        self.ctx = ctx or EvalContext()

    def build_schemes(self, include: tuple[str, ...] | list[str] | None = None) -> dict[str, SchemeRunner]:
        """按名装配 SchemeRunner（rule / single_call_llm / agent；None = 全三方案）。"""
        wanted = tuple(include) if include is not None else ALL_SCHEMES
        factories = {
            "rule": RuleBaseline,
            "single_call_llm": SingleCallScheme,
            "agent": AgentScheme,
        }
        unknown = [n for n in wanted if n not in factories]
        if unknown:
            raise ValueError(f"未知 scheme: {unknown}（可选: {ALL_SCHEMES}）")
        return {name: factories[name]() for name in wanted}

    async def run(
        self,
        *,
        cases: list[EvalCase] | None = None,
        include: tuple[str, ...] | list[str] | None = None,
        smoke: bool = False,
        smoke_limit: int = 10,
    ) -> EvaluationResult:
        """跑一次评测：load → 三 scheme → metrics（总体 + 按 scene 分层）。

        :param cases: 外部注入数据集（测试/多数据复用）；None → 从 self.data_path 加载。
        :param include: scheme 名子集；None → 全三方案。
        :param smoke: True → 取 smoke_limit 条确定性冒烟子集。
        :param smoke_limit: smoke 子集上限（默认 10，Phase 1 ≤10 要求）。
        """
        if cases is None:
            if self.data_path is None:
                raise ValueError("EvaluationRunner 未配置 data_path 且未注入 cases")
            cases = load_dataset(self.data_path)
        if smoke:
            cases = smoke_subset(cases, smoke_limit)
        if not cases:
            raise ValueError("评测运行无有效 case（数据集为空）")

        schemes = self.build_schemes(include)
        exp = expected_index(cases)
        records: dict[str, list[EvalRecord]] = {}
        overall: dict[str, DecisionMetrics] = {}
        grouped: dict[str, dict[str, DecisionMetrics]] = {}
        cost_summary: dict[str, dict] = {}

        for name, scheme in schemes.items():
            scheme_records: list[EvalRecord] = []
            for case in cases:  # 顺序串行（确定性；数据集行序即遍历序）
                scheme_records.append(await scheme.run(case, self.ctx))
            records[name] = scheme_records
            overall[name] = DecisionEvaluator.evaluate(scheme_records, exp)
            grouped[name] = DecisionEvaluator.evaluate_grouped(scheme_records, exp)
            cost_summary[name] = _cost_summary(scheme_records)

        return EvaluationResult(
            data_path=str(self.data_path) if self.data_path is not None else None,
            smoke=smoke,
            total_cases=len(cases),
            scene_stats=scene_stats(cases),
            expected=exp,
            records=records,
            overall=overall,
            grouped=grouped,
            cost_summary=cost_summary,
        )


def _cost_summary(records: list[EvalRecord]) -> dict:
    """成本均值摘要（确定性；无 case 时全 0）。"""
    n = len(records)
    if n == 0:
        return {"llm_calls": 0.0, "tool_calls": 0.0, "tokens": 0.0}
    return {
        "llm_calls": round(sum(r.cost.get("llm_calls") or 0 for r in records) / n, 2),
        "tool_calls": round(sum(r.cost.get("tool_calls") or 0 for r in records) / n, 2),
        "tokens": round(sum(r.cost.get("tokens") or 0 for r in records) / n, 2),
    }
