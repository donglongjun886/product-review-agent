"""ThresholdSweepRunner —— Evidence 阈值 sweep（evaluation/sweep.py，docs/02-evaluation.md §5）。

P-5 已拍板：第一轮**只 sweep Evidence 阈值、单参数**，不做多参数联合 Grid Search：

- ``EVIDENCE_MIN_SIM`` 与 ``EVIDENCE_STRONG`` 各在网格 ``{0.60..0.90, 步 0.05}`` 上
  **单变量扫**：每次只动一个常量、另一个取当前默认（0.70 / 0.85）固定；
- ``CONFIDENCE_ABSTAIN_THRESHOLD`` **固定 0.7，本轮不参与扫描**（加注：联合校准待
  主扫结果定效果来源后再做，见模块 docstring 注记与 docs §5.1）。

实现约束（docs §5.1"只动配置不动判定逻辑"）：sweep 只换 ``EvalContext`` 的
``evidence_thresholds`` 重跑，不触碰任何判定代码。**实际生效范围（如实声明）**：
评测侧相似度分档读取路径 —— agent_scheme 的确定性审查员模型（EvalScriptedLLMBackend）
按 ctx 阈值做 (a) 相似度证据视图下限过滤（min_sim）与 (b) 强相似分档（strong）；
真实图 tools_node 的 ``quality_filter`` / gate overlay 常量属 pra.agent 业务层（0.70 /
0.85 模块常量），本轮不随 ctx 变 —— 因此 agent 的 sweep 效应是"评测审查员读到的证据
视图"随阈值变化；**rule 与 single_call_llm 的决策路径不读取图片相似度**（screening 规则
为词表命中；single-call mock 为表面文本）→ 这两行指标随阈值恒定，属预期，报告注明。

**数据带局限（实证注记，不改数据）**：当前种子世界可观测的相似度权重只落两簇 ——
{0.72, 0.73} 与 {0.90+}，(0.73, 0.90) 之间无 case。因此档间曲线平**不代表生产阈值
不敏感**，只说明该数据带无样本可判别；校准结论受数据带限制，不能外推成"生产阈值不
敏感"或据此回写生产常量（tools_node quality_filter / gate overlay 0.70/0.85 不随 ctx
变，见上）。被扫参数当前只作用于评测审查员读证据视图；Q4（注入下沉 vs 文档收窄）待
拍板，本模块只陈述现状与边界，不做该方向决策。

产出：每档阈值的七项指标（Accuracy / Precision / Recall / FPR / FNR /
human_review_rate / automation_coverage —— docs §5.2）每 scheme 一组；
另给四曲线要点（FPR / Recall / HRR / Acc vs 阈值）。可导出 CSV（``to_csv_lines``）。

全程确定性：无真 LLM / 网络 / 随机；case 行序即遍历序、每档 ctx 独立重跑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pra.evaluation.dataset.loader import load_dataset
from pra.evaluation.dataset.schema import EvalCase
from pra.evaluation.harness.base import EvalContext
from pra.evaluation.metrics.abstention import AbstentionEvaluator, AbstentionMetrics
from pra.evaluation.metrics.business import DecisionMetrics
from pra.evaluation.runner import ALL_SCHEMES, EvaluationRunner, expected_index

__all__ = [
    "DEFAULT_THRESHOLDS",
    "EVIDENCE_GRID",
    "SWEEP_VARIABLES",
    "SweepPoint",
    "SweepResult",
    "ThresholdSweepRunner",
    "to_csv_lines",
]

# sweep 网格（P-5：0.60/0.65/0.70/0.75/0.80/0.85/0.90）
EVIDENCE_GRID: tuple[float, ...] = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)

# 变量名（权威）→ EvalContext.evidence_thresholds 的键
SWEEP_VARIABLES: dict[str, str] = {
    "EVIDENCE_MIN_SIM": "min_sim",
    "EVIDENCE_STRONG": "strong",
}

# 当前默认（另一常量扫时固定取此值；与 tools/image_analysis 常量同值）
DEFAULT_THRESHOLDS: dict[str, float] = {"min_sim": 0.70, "strong": 0.85}

# 观察指标（docs §5.2 七项）—— 与 DecisionMetrics / 报告字段一一对应
SEVEN_METRICS: tuple[str, ...] = (
    "accuracy",
    "precision",
    "recall",
    "fpr",
    "fnr",
    "human_review_rate",
    "automation_coverage",
)

# §4.4/§5.2 规范名 → DecisionMetrics 字段名（Phase 1 业务模块字段是 human_rate/automation，
# 规范指标名是 human_review_rate/automation_coverage —— 只差别名，全代码以规范名为准）
METRIC_FIELD_ALIASES: dict[str, str] = {
    "human_review_rate": "human_rate",
    "automation_coverage": "automation",
}


@dataclass
class SweepPoint:
    """单个 (变量, 档值) 的评测结果。"""

    variable: str  # "EVIDENCE_MIN_SIM" | "EVIDENCE_STRONG"
    value: float
    thresholds: dict  # 实际生效的 {"min_sim": float, "strong": float}
    metrics_by_scheme: dict[str, DecisionMetrics] = field(default_factory=dict)
    abstention_by_scheme: dict[str, AbstentionMetrics] = field(default_factory=dict)

    def value_for(self, scheme: str, metric: str) -> float | None:
        """取该档某 scheme 的某指标值（指标用规范名；自动映射 DecisionMetrics 字段别名）。"""
        m = self.metrics_by_scheme.get(scheme)
        if m is None:
            return None
        field = METRIC_FIELD_ALIASES.get(metric, metric)
        return getattr(m, field, None)


@dataclass
class SweepResult:
    """一次 sweep 的全部档点 + 注记（报告/CSV 消费）。"""

    data_path: str | None = None
    total_cases: int = 0
    points: list[SweepPoint] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def points_of(self, variable: str) -> list[SweepPoint]:
        return [p for p in self.points if p.variable == variable]

    def grid_of(self, variable: str) -> list[float]:
        return [p.value for p in self.points_of(variable)]


def _ctx_with_thresholds(base: EvalContext, variable: str, value: float) -> EvalContext:
    """在 ``base`` 上覆盖单变量档值，另一常量取默认 —— sweep 只动 EvalContext。"""
    key = SWEEP_VARIABLES[variable]
    overrides = dict(DEFAULT_THRESHOLDS)
    overrides[key] = value
    return base.model_copy(update={"evidence_thresholds": overrides})


class ThresholdSweepRunner:
    """Threshold sweep 驱动（docs §5）：每档阈值换 EvalContext 重跑同一数据集。"""

    def __init__(
        self,
        *,
        data_path: str | None = None,
        ctx: EvalContext | None = None,
    ) -> None:
        self.data_path = data_path
        self.ctx = ctx or EvalContext()

    async def run(
        self,
        *,
        cases: list[EvalCase] | None = None,
        variables: tuple[str, ...] | list[str] = tuple(SWEEP_VARIABLES),
        schemes: tuple[str, ...] | list[str] = tuple(ALL_SCHEMES),
    ) -> SweepResult:
        """扫指定变量（默认两个 Evidence 常量都扫）在网格上的全部档点。

        每档点跑 EvaluationRunner（同一数据集、顺序串行、确定性），产出
        DecisionMetrics（七项观察）与 AbstentionMetrics（Phase 2 语义可用时）。
        """
        if cases is None:
            if self.data_path is None:
                raise ValueError("ThresholdSweepRunner 未配置 data_path 且未注入 cases")
            cases = load_dataset(self.data_path)
        if not cases:
            raise ValueError("Sweep 无有效 case（数据集为空）")
        unknown = [v for v in variables if v not in SWEEP_VARIABLES]
        if unknown:
            raise ValueError(f"未知 sweep 变量: {unknown}（可选: {tuple(SWEEP_VARIABLES)}）")
        wanted = tuple(schemes)
        exp = expected_index(cases)

        result = SweepResult(data_path=self.data_path, total_cases=len(cases))
        for variable in variables:
            for value in EVIDENCE_GRID:
                ctx_i = _ctx_with_thresholds(self.ctx, variable, value)
                runner = EvaluationRunner(data_path=self.data_path, ctx=ctx_i)
                ev_result = await runner.run(cases=cases, include=wanted)
                point = SweepPoint(
                    variable=variable,
                    value=value,
                    thresholds={
                        k: round(v, 4) for k, v in ctx_i.resolve_evidence_thresholds().items()
                    },
                )
                for scheme in wanted:
                    records = ev_result.records.get(scheme) or []
                    point.metrics_by_scheme[scheme] = ev_result.overall[scheme]
                    point.abstention_by_scheme[scheme] = AbstentionEvaluator.evaluate(records, exp)
                result.points.append(point)

        result.notes = [
            "CONFIDENCE_ABSTAIN_THRESHOLD 固定 0.7，本轮不参与扫描（联合校准待主扫结果定效果来源后再说）。",
            "sweep 只动 EvalContext.evidence_thresholds（配置注入）；判定逻辑零改动。",
            (
                "生效范围: agent（评测审查员的相似度证据视图 min_sim 过滤 + strong 分档）；"
                "rule / single_call_llm 决策路径不读图片相似度 → 行恒定（预期）。"
            ),
            (
                "真实图 tools_node quality_filter(0.70) / gate overlay(0.85) 为 pra.agent 模块常量，"
                "本轮不随 sweep —— agent 档间差异源于评测侧证据视图（报告口径见模块 docstring）。"
            ),
            (
                "数据带局限（实证注记）: 本种子世界相似度权重仅两簇 {0.72,0.73} 与 {0.90+}，"
                "(0.73,0.90) 无 case —— 曲线平不代表生产阈值不敏感，校准结论受数据带限制；"
                "被扫参数只作用于评测审查员读证据视图（生产常量不随 ctx 变，不写回；"
                "Q4 注入下沉 vs 文档收窄待拍板）。"
            ),
        ]
        return result


# ---------------------------------------------------------------------------
# 输出：CSV 行 + 四曲线要点（确定性纯文本）
# ---------------------------------------------------------------------------


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.3f}"


def to_csv_lines(result: SweepResult, *, schemes=None) -> list[str]:
    """sweep 结果 → CSV 行（表头 + 每 (变量, 档, scheme) 一行，七项观察指标）。

    docs §5.2 七项：Accuracy/Precision/Recall/FPR/FNR/human_review_rate/automation_coverage
    （automation 用 §4.4 命名 automation_coverage）。
    """
    wanted = schemes or ALL_SCHEMES
    header = [
        "variable",
        "value",
        "min_sim",
        "strong",
        "scheme",
        "total",
        "accuracy",
        "precision",
        "recall",
        "fpr",
        "fnr",
        "human_review_rate",
        "automation_coverage",
    ]
    lines = [",".join(header)]
    for p in result.points:
        for scheme in wanted:
            m = p.metrics_by_scheme.get(scheme)
            if m is None:
                continue
            row = [
                p.variable,
                f"{p.value:.2f}",
                f"{p.thresholds.get('min_sim'):.2f}",
                f"{p.thresholds.get('strong'):.2f}",
                scheme,
                str(m.total),
                _fmt(m.accuracy),
                _fmt(m.precision),
                _fmt(m.recall),
                _fmt(m.fpr),
                _fmt(m.fnr),
                _fmt(m.human_rate),
                _fmt(m.automation),
            ]
            lines.append(",".join(row))
    return lines


def curve_highlights(result: SweepResult, *, schemes=None) -> list[str]:
    """四曲线要点（FPR / Recall / HRR / Acc vs 阈值）的确定性文本摘要。

    对每个 (scheme, 观察指标) 输出：各档值 → 指标值串 + 值域 + 发生变化的档值
    （无变化 → 注明恒定 —— 防把"阈值不敏感"误读为"曲线很平是 bug"）。
    """
    wanted = schemes or ALL_SCHEMES
    lines: list[str] = []
    for variable in SWEEP_VARIABLES:
        points = result.points_of(variable)
        if not points:
            continue
        lines.append(f"[{variable}] 网格: {[f'{p.value:.2f}' for p in points]}")
        for scheme in wanted:
            for metric in ("fpr", "recall", "human_review_rate", "accuracy"):
                values = [p.value_for(scheme, metric) for p in points]
                rendered = " ".join(_fmt(v) for v in values)
                non_none = [v for v in values if v is not None]
                span = (f"{min(non_none):.3f}..{max(non_none):.3f}") if non_none else "-"
                changed = [
                    f"{points[i].value:.2f}"
                    for i in range(1, len(points))
                    if values[i] != values[i - 1]
                ]
                note = (
                    f"（档间变化于: {', '.join(changed)}）" if changed else "（恒定为当前默认档行为）"
                )
                lines.append(f"  {scheme:<16} {metric:<18} {rendered}   值域 {span} {note}")
    return lines


def write_csv(result: SweepResult, path: str | Path, *, schemes=None) -> None:
    """把 sweep 结果写入 CSV 文件（脚本 CLI 用）。"""
    p = Path(path)
    p.write_text("\n".join(to_csv_lines(result, schemes=schemes)) + "\n", encoding="utf-8")
