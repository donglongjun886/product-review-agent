"""消融评测：方案级（2a/2b/2c）与组件级（full / −rag / −merchant / −case / −image）。

两类消融跑**同一 eval_dataset**：差异唯一归因于装配，判定逻辑与数据集不动。

方案级 —— Single-call 的上下文 vs Agent 的主动调查：
- ``2a`` = SingleCallScheme 现行为（Raw Input，无任何预塞知识）；
- ``2b`` = SingleCallScheme + RAG-in-prompt（``extra_context`` 预塞政策/先例**文本
  摘要**；仍不给工具）。**公平性**：预塞的只是"基础输入外的事实文本"（本模块
  ``build_rag_context`` 由 EVAL_PRECEDENTS / EVAL_POLICY_CLAUSES 生成），不含 expected
  答案；
- ``2c`` = AgentScheme 现行为（多步主动调查）。

**差异口径诚实化（实证注记）**：分解 2b 的"更多文本"增益、再把 2c−2b 归因"主动
调查增量"，依赖 2b 腿可激活；实测 v1/v2 全量 2a≡2b（0 差异）—— R-2b 升级路径在本
评测世界的种子摘要下永远不命中（见 ``run()`` 注记），故"更多文本"增益**不可测**；
2c−2b 因而 = 2c−2a = **Single-call→Agent 整体差异**（工具 + 多步 + mock 变体一并
替换），**不可分解归因于"主动调查"**。报告差异行据此命名（2c vs 2b 不称"主动调查
增量"）。

组件级 —— 同图结构逐组件去掉：实现为**装配层裁剪** ``AgentScheme(allowed_tools=…)``
—— 图工具注册与该方案的 plan tool schema 都只给 ``全工具 − 被裁组件``，判定逻辑不变。
**0 决策变化 ≠ 组件无用**：证据可能被另一组件**同案冗余替代** —— REJECT Gate 的
citable 只要 POLICY_REF 或 CASE_PRECEDENT 之一，本评测世界 REJECT 案政策齐备 →
CaseSearch 证据被 PolicySearch 兜底（实测 −CaseTool 零变化）；报告在 0 变化组件旁
标注该读法。

**InMemory 种子局限标注**：种子里查不到某工具的证据时，−该工具必然无差异、结论失效
—— 本模块统计每个组件的"证据覆盖"（哪些 case 的 ``expected.expected_tools`` 含该工具），
报告如实标注空消融风险。

确定性：全程无真 LLM / 网络 / 随机；EvalRecord 不含墙钟字段 → 同数据重跑结果一致；
全部在内存完成（不落 DB）。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from pra.evaluation.dataset.loader import load_dataset
from pra.evaluation.dataset.schema import EvalCase
from pra.evaluation.harness.agent_scheme import (
    EVAL_POLICY_CLAUSES,
    EVAL_PRECEDENTS,
    AgentScheme,
)
from pra.evaluation.harness.base import EvalContext, EvalRecord, SchemeRunner
from pra.evaluation.harness.single_call_scheme import SingleCallScheme
from pra.evaluation.metrics.abstention import AbstentionEvaluator, AbstentionMetrics
from pra.evaluation.metrics.business import DecisionEvaluator, DecisionMetrics
from pra.evaluation.runner import expected_index

__all__ = [
    "ALL_TOOL_NAMES",
    "COMPONENT_ABLATIONS",
    "SCHEME_LEVEL_VARIANTS",
    "AblationResult",
    "AblationRunner",
    "build_rag_context",
    "decision_sequence",
    "expected_tool_coverage",
    "render_ablation_report",
]

# 变体定义（命名权威；字符串即 CLI/报告标识）

SCHEME_LEVEL_VARIANTS: tuple[str, ...] = ("2a", "2b", "2c")

# eval 世界 5 个 InMemory 工具（产品/图片/商家/先例/政策）
ALL_TOOL_NAMES: tuple[str, ...] = (
    "ProductTool",
    "ImageAnalysisTool",
    "MerchantTool",
    "CaseSearchTool",
    "PolicySearchTool",
)

# 组件级消融：变体名 → 被裁掉的工具集合（装配层裁剪 = 全工具 − 被裁）
COMPONENT_ABLATIONS: dict[str, frozenset[str]] = {
    "full": frozenset(),                      # Full Agent（基线）
    "-rag": frozenset({"CaseSearchTool", "PolicySearchTool"}),
    "-merchant": frozenset({"MerchantTool"}),
    "-case": frozenset({"CaseSearchTool"}),
    "-image": frozenset({"ImageAnalysisTool"}),
}

# 人类可读的组件名（报告对照表用）
COMPONENT_LABELS: dict[str, str] = {
    "full": "Full Agent（基线）",
    "-rag": "−RAG（无 CaseSearch+PolicySearch）",
    "-merchant": "−MerchantTool",
    "-case": "−CaseTool",
    "-image": "−ImageTool",
}


# RAG-in-prompt 上下文（2b 的预塞文本；来自评测世界静态事实，不含 expected 答案）


def _fold(text: str) -> str:
    return (text or "").lower().replace("：", ":")


def build_rag_context(case: EvalCase) -> list[str]:
    """构造 2b 的预塞文本：该案**类目**下可查的判例/政策 digest（确定性静态文本）。

    只取评测世界里**已裁决**的先例（EVAL_PRECEDENTS 含 decision）作 ``判例:`` 行
    （格式 ``判例: <id> 类目[<类目>] <summary> → <决策>``，由 single_call_scheme 的
    R-2b 消费）；政策条款（EVAL_POLICY_CLAUSES）摘要行仅作参考材料一并预塞 ——
    R-2b 只认自动拒绝判例，政策现行口径为"转人工"，不会错误触发升级。
    类目匹配不上时不预塞该行（确定性检索近似：只给相关材料）。
    """
    product = case.input.product
    category = (product.category or "").strip()
    lines: list[str] = []
    if not category:
        return lines
    c_fold = _fold(category)
    for p in EVAL_PRECEDENTS:
        if _fold(str(p.get("category") or "")) != c_fold:
            continue
        decision = str(p.get("decision") or "")
        summary = str(p.get("summary") or "")
        lines.append(
            f"判例: {p.get('case_id')} 类目[{category}] {summary} → {decision}"
        )
    for c in EVAL_POLICY_CLAUSES:
        if _fold(str(c.get("category") or "")) != c_fold:
            continue
        title = str(c.get("title") or "")
        text = str(c.get("text") or "")
        lines.append(f"政策: {c.get('policy_id')} 类目[{category}] {title}：{text}")
    return lines


# 小工具（确定性）


def decision_sequence(records: list[EvalRecord]) -> list[str]:
    """records（已按 case 行序）的 decision 序列（regression/消融差异对比用）。"""
    return [r.decision for r in records]


def expected_tool_coverage(cases: list[EvalCase]) -> dict[str, list[str]]:
    """每个工具的**评测世界证据覆盖**：expected.expected_tools 含该工具的 case id 列表。

    用于组件级消融的"空消融风险"标注：某工具 coverage 为空 → 评测世界里没有 case
    标注需要它 → 该工具无差异不代表组件不必要。
    """
    coverage: dict[str, list[str]] = {name: [] for name in ALL_TOOL_NAMES}
    for c in cases:
        tools = {str(t) for t in (c.expected.expected_tools or [])}
        for name in ALL_TOOL_NAMES:
            if name in tools:
                coverage[name].append(c.eval_case_id)
    for name, covered_ids in coverage.items():
        covered_ids.sort()
    return coverage


def _decision_diff(
    full: list[EvalRecord], variant: list[EvalRecord]
) -> tuple[list[str], list[str]]:
    """variant vs full 的决策差异 case id 与决策对（按行序对齐；长度不一致 → 报错）。"""
    if len(full) != len(variant):
        raise ValueError("同数据集上两变体 records 长度不一致，无法对齐比较")
    changed: list[str] = []
    pairs: list[str] = []
    for f, v in zip(full, variant):
        if f.eval_case_id != v.eval_case_id:
            raise ValueError(f"records 顺序不一致: {f.eval_case_id} vs {v.eval_case_id}")
        if f.decision != v.decision:
            changed.append(f.eval_case_id)
            pairs.append(f"{f.eval_case_id}: {f.decision}→{v.decision}")
    return changed, pairs


# 变体运行与结果容器


@dataclass
class VariantOutcome:
    """一个变体在同一数据集上的产出（供并排报告 / 差异计算）。"""

    name: str
    label: str
    records: list[EvalRecord]
    overall: DecisionMetrics
    abstention: AbstentionMetrics

    @property
    def decisions(self) -> list[str]:
        return decision_sequence(self.records)


@dataclass
class AblationResult:
    """一次消融运行的全部产物（方案级 + 组件级可各自独立开启）。"""

    data_path: str | None = None
    total_cases: int = 0
    scheme_outcomes: dict[str, VariantOutcome] = field(default_factory=dict)  # 2a/2b/2c
    component_outcomes: dict[str, VariantOutcome] = field(default_factory=dict)  # full/-rag/…
    tool_coverage: dict[str, list[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def all_outcomes(self) -> dict[str, VariantOutcome]:
        out = dict(self.scheme_outcomes)
        out.update(self.component_outcomes)
        return out


async def _run_records(
    scheme: SchemeRunner, cases: list[EvalCase], ctx: EvalContext
) -> list[EvalRecord]:
    """顺序串行跑单 scheme（确定性：case 行序即遍历序）。"""
    return [await scheme.run(c, ctx) for c in cases]


class AblationRunner:
    """Ablation runner：方案级（2a/2b/2c）+ 组件级（full/−rag/…）消融。

    每变体产出统一 EvalRecord，再走同一 metrics（DecisionEvaluator +
    AbstentionEvaluator）。
    """

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
        scheme_level: bool = True,
        component_level: bool = True,
    ) -> AblationResult:
        if cases is None:
            if self.data_path is None:
                raise ValueError("AblationRunner 未配置 data_path 且未注入 cases")
            cases = load_dataset(self.data_path)
        if not cases:
            raise ValueError("Ablation 无有效 case（数据集为空）")
        exp = expected_index(cases)
        result = AblationResult(
            data_path=self.data_path,
            total_cases=len(cases),
            tool_coverage=expected_tool_coverage(cases),
        )

        if scheme_level:
            # 2a：Raw Input（现行为）
            rec_2a = await _run_records(SingleCallScheme(), cases, self.ctx)
            # 2b：RAG-in-prompt —— 每 case 预塞其类目静态 digest（确定性）
            rec_2b: list[EvalRecord] = []
            for case in cases:
                ctx_text = build_rag_context(case)
                scheme_2b = SingleCallScheme(
                    llm_fn=None,
                    extra_context=ctx_text if ctx_text else None,
                )
                rec_2b.append(await scheme_2b.run(case, self.ctx))
            # 2c：Multi-step Agent（主动调查，现行为）
            rec_2c = await _run_records(AgentScheme(), cases, self.ctx)
            for name, records in (("2a", rec_2a), ("2b", rec_2b), ("2c", rec_2c)):
                result.scheme_outcomes[name] = self._outcome(name, records, exp)
            result.notes.append(
                "2b 预塞文本 = 类目静态判例/政策 digest（build_rag_context，无 expected 答案）；"
                "R-2b 只在'弱 REJECT 候选 + 自动拒绝判例命中表面词'时升级 —— 现行评测世界政策"
                "口径为转人工、先例摘要不含表面规避词 → 2b 腿 inert（P2-12 实证）：v1/v2 "
                "全量实测 2a≡2b（0 差异），'更多文本'增益在本数据上不可测；2c−2b 因而 = "
                "2c−2a = Single-call→Agent 整体差异（工具+多步+ContextAware mock 一并替换），"
                "不可分解归因于'主动调查' —— 本报告差异行按此口径命名。"
            )

        if component_level:
            for name, removed in COMPONENT_ABLATIONS.items():
                allowed = (
                    None
                    if not removed
                    else set(ALL_TOOL_NAMES) - removed
                )
                records = await _run_records(AgentScheme(allowed_tools=allowed), cases, self.ctx)
                result.component_outcomes[name] = self._outcome(name, records, exp)
                result.notes.append(
                    f"{COMPONENT_LABELS[name]} 装配裁剪: allowed_tools={sorted(allowed) if allowed else '全工具'}"
                )

        # 空消融风险标注（工具覆盖）
        variant_of = {
            "ProductTool": None,
            "ImageAnalysisTool": "-image",
            "MerchantTool": "-merchant",
            "CaseSearchTool": "-case",
            "PolicySearchTool": "-rag",
        }
        for name in ALL_TOOL_NAMES:
            covered = result.tool_coverage.get(name) or []
            if not covered:
                variant_name = variant_of.get(name)
                suffix = (
                    f"变体 {variant_name} 无差异时不能判定该组件不必要"
                    if variant_name
                    else "（无单独裁剪变体）"
                )
                result.notes.append(
                    f"空消融风险: 评测世界里无 case 的 expected_tools 含 {name} —— 需该工具的案缺失，"
                    f"{suffix}"
                )
        return result

    @staticmethod
    def _outcome(name: str, records: list[EvalRecord], exp: Mapping) -> VariantOutcome:
        label = {
            "2a": "Single-call + Raw Input",
            "2b": "Single-call + RAG-in-prompt",
            "2c": "Multi-step Agent（主动调查）",
        }.get(name, COMPONENT_LABELS.get(name, name))
        return VariantOutcome(
            name=name,
            label=label,
            records=records,
            overall=DecisionEvaluator.evaluate(records, exp),
            abstention=AbstentionEvaluator.evaluate(records, exp),
        )


# 报告渲染（纯文本，stdout；无文件 IO）


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.3f}"


def _metric_cells(m: DecisionMetrics) -> list[str]:
    return [_fmt(m.accuracy), _fmt(m.precision), _fmt(m.recall), _fmt(m.fpr), _fmt(m.fnr)]


def _abstention_cells(a: AbstentionMetrics) -> list[str]:
    return [
        _fmt(a.human_review_rate),
        _fmt(a.automation_coverage),
        _fmt(a.abstention_rate),
        _fmt(a.abstention_recall),
        _fmt(a.wrong_auto_decision_rate),
    ]


def render_ablation_report(result: AblationResult) -> str:
    """渲染 Ablation Console Report（纯文本）。"""
    out: list[str] = []
    add = out.append
    add("=" * 108)
    add("商品审核 Agent · Evaluation Phase 2 Ablation Report")
    add("=" * 108)
    add(f"数据集: {result.data_path or '(外部注入 cases)'}   案数: {result.total_cases}")
    add("口径: acc/prec/recall/fpr/fnr = 二分类（自动终裁子集，真值仅 PASS/REJECT 计入）；")
    add("      hrr/automation/abstention_rate/abstention_recall/wrong_auto = abstention 五指标")
    add("      （'-' = 分母 0，如 Phase 1 无 SHOULD_ABSTAIN 真值 → abstention_recall '-'）")

    def _section(title: str, order: tuple, table: dict[str, VariantOutcome]) -> None:
        add("")
        add("-" * 108)
        add(title)
        add("-" * 108)
        header = (
            f"{'variant':<34} acc    prec   recall  fpr    fnr    | hrr    autom  abst_r "
            f"abst_rl w_auto"
        )
        add(header)
        for name in order:
            v = table.get(name)
            if v is None:
                continue
            cells = _metric_cells(v.overall) + _abstention_cells(v.abstention)
            add(f"{v.name + ' ' + v.label:<34} " + " ".join(f"{c:>6}" for c in cells))

    if result.scheme_outcomes:
        _section(
            "方案级消融（同一数据集；差异行口径：2b vs 2a 观察'更多文本'增益（2b 腿 inert 时"
            "不可测）、2c vs 2b = Single-call→Agent 整体差异，非'主动调查'增量 —— 见注记）",
            SCHEME_LEVEL_VARIANTS,
            result.scheme_outcomes,
        )
        add("")
        for label, a_name, b_name in (
            ("2b vs 2a（'更多文本'增益观察；2b 腿 inert → 差异不可测，见注记）", "2a", "2b"),
            ("2c vs 2b（Single-call→Agent 整体差异；非'主动调查'增量，见注记）", "2b", "2c"),
        ):
            if a_name not in result.scheme_outcomes or b_name not in result.scheme_outcomes:
                continue
            _changed, pairs = _decision_diff(
                result.scheme_outcomes[a_name].records,
                result.scheme_outcomes[b_name].records,
            )
            add(f"[{label}] 决策变化 case 数 = {len(_changed)}")
            for p in pairs:
                add(f"    {p}")
            if not _changed:
                add("    （无决策变化 —— 见 notes 中的机制/覆盖说明）")

    if result.component_outcomes:
        _section(
            "组件级消融（Full Agent 基线；逐组件装配层裁剪）",
            tuple(COMPONENT_ABLATIONS.keys()),
            result.component_outcomes,
        )
        full = result.component_outcomes.get("full")
        if full is not None:
            add("")
            add("[相对 Full Agent 的决策变化]")
            for name in COMPONENT_ABLATIONS:
                if name == "full":
                    continue
                variant = result.component_outcomes.get(name)
                if variant is None:
                    continue
                _changed, pairs = _decision_diff(full.records, variant.records)
                if _changed:
                    add(f"    {COMPONENT_LABELS[name]}: {len(_changed)} 个 case 决策改变")
                    for p in pairs:
                        add(f"        {p}")
                else:
                    # P2-13 防误读：0 决策变化 ≠ 组件无用 —— 证据可能被另一组件同案冗余替代
                    #（REJECT Gate 的 citable 只要 POLICY_REF 或 CASE_PRECEDENT 之一；
                    # 本世界 REJECT 案政策齐备 → CASE_PRECEDENT 被 POLICY_REF 兜底）。
                    add(
                        f"    {COMPONENT_LABELS[name]}: 0 个 case 决策改变 —— 勿读成组件无用："
                        "该组件证据可能被另一组件同案冗余替代（CASE_PRECEDENT 被 POLICY_REF "
                        "兜底，REJECT Gate 只要二者之一）；须与上方工具覆盖并读。"
                    )

    add("")
    add("=" * 108)
    add("工具证据覆盖（组件必要性判定的前置标注）:")
    for name in ALL_TOOL_NAMES:
        covered = result.tool_coverage.get(name) or []
        add(f"  · {name}: {len(covered)} 个 case 的 expected_tools 含该工具"
            + (f"（{', '.join(covered[:8])}{' …' if len(covered) > 8 else ''}）" if covered else "（无覆盖）"))
    add("")
    add("注记:")
    for n in result.notes:
        add(f"  · {n}")
    add("结论边界: 工具为 InMemory 种子 + LLM 为桩 → 覆盖有限可能低估 Agent 上限；")
    add("         组件级结论须与工具覆盖并读（覆盖为空的组件不做'不必要'判定）。")
    add("=" * 108)
    return "\n".join(out)


def print_ablation_report(result: AblationResult) -> None:
    """打印 Ablation Report 到 stdout。"""
    print(render_ablation_report(result))
