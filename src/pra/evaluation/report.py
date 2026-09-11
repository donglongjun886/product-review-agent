"""Console Report：总体 + 按 scene 分层的纯文本报告。

只做格式化（无文件 IO；``print_report`` 打印 stdout）。口径说明随报告输出：
数据集分布（实际分布由 loader 统计，防 manifest 漂移）；工具数据源 = InMemory 种子、
LLM = 确定性桩；输出空间对齐：Rule COMPLEX→HUMAN_REVIEW、Single-call conf<门槛的
REJECT→HUMAN。

**分母口径随数据版本分叉（透明化，不改计算）**：v2（真值含 HUMAN_REVIEW /
SHOULD_ABSTAIN）时决策指标行取二值真值（PASS+REJECT）分母，abstention 五指标行取全量
分母，头行同步列出三值分布与两套分母；v1（无 abstain 标签）两套分母重合，abstention 区
不渲染。Accuracy：pred HUMAN_REVIEW 记为判错（入分母不入 (TP+TN) 分子）；
Precision/Recall/FPR/FNR 只在自动判出子集上计算。

v1 词表现状：``BLACKLISTED_BRANDS`` 为空 → Rule baseline 无自动 REJECT（品牌词/规避词/
空缺一律 COMPLEX），如实呈现，非缺陷。结论边界块含「标注-审查员同口径耦合」声明与 real
实测对照（同口径耦合高估一致性、工具覆盖有限低估真实上限）。
"""

from __future__ import annotations

from pra.evaluation.metrics.abstention import abstain_subset_of
from pra.evaluation.metrics.business import DecisionMetrics
from pra.evaluation.runner import ALL_SCHEMES, EvaluationResult

__all__ = ["print_report", "render_report"]

_SCENES = ("normal", "violation", "boundary", "multi-signal", "evasion")
_DECISIONS = ("PASS", "REJECT", "HUMAN_REVIEW")


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _row(cells: list) -> str:
    """单行格式化：数字列等宽；None → '-'。"""
    return "  ".join(_fmt(c) for c in cells)


def _overall_row(scheme: str, m: DecisionMetrics, cost: dict) -> str:
    return "  ".join(
        [
            f"{scheme:<16}",
            _fmt(m.accuracy),
            _fmt(m.precision),
            _fmt(m.recall),
            _fmt(m.fpr),
            _fmt(m.fnr),
            _fmt(m.human_rate),
            _fmt(m.automation),
            f"{m.tp}/{m.fp}/{m.tn}/{m.fn}",
            f"llm={cost.get('llm_calls')} tool={cost.get('tool_calls')} tok={cost.get('tokens')}",
        ]
    )


def _scene_row(scheme: str, m: DecisionMetrics) -> str:
    return "  ".join(
        [
            f"{scheme:<16}",
            _fmt(m.accuracy),
            _fmt(m.precision),
            _fmt(m.recall),
            _fmt(m.fpr),
            _fmt(m.fnr),
            _fmt(m.human_rate),
            _fmt(m.automation),
            f"{m.tp}/{m.fp}/{m.tn}/{m.fn}",
        ]
    )


def _scene_truth_parts(entry: dict) -> str:
    """单 scene 三值真值分布的人读片段（仅列非零桶；P/R/H=真值 PASS/REJECT/HUMAN_REVIEW）。"""
    total = int(entry.get("total", 0))
    buckets = []
    for key, tag in (("PASS", "P"), ("REJECT", "R"), ("HUMAN_REVIEW", "H")):
        n = int(entry.get(key, 0))
        if n:
            buckets.append(f"{tag}{n}")
    return f"{total}" + (f"({'/'.join(buckets)})" if buckets else "")


def render_report(result: EvaluationResult) -> str:
    """渲染整份 Console Report（纯文本；行内口径说明见模块 docstring）。"""
    out: list[str] = []
    add = out.append

    add("=" * 100)
    add("商品审核 Agent · Evaluation 三方案对比 Console Report")
    add("=" * 100)
    src = result.data_path or "(外部注入 cases)"
    add(f"数据集: {src}" + ("   [smoke 冒烟子集]" if result.smoke else ""))
    stats = result.scene_stats
    by_scene = stats.get("by_scene", {})
    total = int(stats.get("total", 0))
    pass_n = sum(int(s.get("PASS", 0)) for s in by_scene.values() if isinstance(s, dict))
    reject_n = sum(int(s.get("REJECT", 0)) for s in by_scene.values() if isinstance(s, dict))
    human_n = sum(int(s.get("HUMAN_REVIEW", 0)) for s in by_scene.values() if isinstance(s, dict))
    binary = pass_n + reject_n

    # 三值真值分布 + 两套分母（P1-2/P2-4：46 条 SHOULD/HUMAN 真值不再隐身）
    if result.has_should_abstain:
        auto_n = sum(1 for v in result.expected.values() if abstain_subset_of(v) == "AUTO_DECIDABLE")
        should_n = sum(1 for v in result.expected.values() if abstain_subset_of(v) == "SHOULD_ABSTAIN")
        add(f"真值口径: Phase 2 三值（含 HUMAN_REVIEW/SHOULD_ABSTAIN 真值）→ 决策指标分母=二值真值 "
            f"{binary}，abstention 五指标分母=全量 {total}")
        add(f"真值案: 共 {total} 条（PASS={pass_n} / REJECT={reject_n} / HUMAN_REVIEW={human_n}）")
        add(f"分母注记: 决策指标行（acc/prec/recall/fpr/fnr/hrr/auto）只计二值真值 {binary} 案 "
            f"（AUTO_DECIDABLE={auto_n}；HUMAN 真值不计入其分母）；")
        add(f"          abstention 五指标行分母 = 全量 {total}（AUTO_DECIDABLE={auto_n} / "
            f"SHOULD_ABSTAIN={should_n}，human_review_rate 为全量分母，见下节）")
    else:
        add(f"真值口径: Phase 1 二值（v1 无 abstain 标签/HUMAN 真值）→ 决策指标分母=全量 {total}；"
            f"abstention 五指标不适用")
        add(f"真值案: 共 {total} 条（PASS={pass_n} / REJECT={reject_n} / HUMAN_REVIEW={human_n}）")
    dist = " | ".join(
        f"{scene}={int(by_scene.get(scene, {}).get('total', 0))}" for scene in _SCENES
    )
    add(f"scene 分布(全量): {dist}")
    if human_n:
        add("scene 真值三值(P/R/H，仅列非零): " + " | ".join(
            f"{scene}={_scene_truth_parts(by_scene.get(scene, {}) or {})}" for scene in _SCENES
        ))

    add("-" * 100)
    add("结论边界 / 口径注记:")
    add("  · 工具数据源: InMemory 种子世界 v1（与 eval_data/v1 同一份；仅 Agent 经工具取证可见）")
    add("  · LLM: 确定性桩（rule=无 / single_call=single-call-mock-v1 / agent=eval-scripted-reviewer）")
    add("  · 输出空间对齐: Rule COMPLEX→HUMAN_REVIEW（评测语义：不可自动判）；Single-call confidence<0.7 的 REJECT→HUMAN")
    add("  · Accuracy=(TP+TN)/真值总数，预测 HUMAN_REVIEW 计为未命中真值(判错，入分母不入分子)；")
    add("    Precision/Recall/FPR/FNR 只在自动判出(pred∈{PASS,REJECT})子集上计算")
    add("  · REJECT 为正类: Recall=TP/(TP+FN) 违规召回 / FPR=FP/(FP+TN) 误杀红线 / FNR=FN/(TP+FN) 漏放")
    add("  · HRR=转人工率 / auto=自动化率；reject_unhandled=该 REJECT 却转人工占比（保守度观测）")
    add("  · screening BLACKLISTED_BRANDS 为空 → Rule 无自动 REJECT（品牌词/规避词/空缺一律 COMPLEX→HUMAN）")
    if not result.has_should_abstain:
        add("  · abstention: v1 无 abstain 标签 → Phase 1 兼容口径（全案等价 AUTO_DECIDABLE，"
            "五指标区不渲染；如需五指标请用 v2 数据集）")
    add("  · 结论边界（双向，勿单向解读）:")
    add("    - 低估侧: 工具 = InMemory 种子 + LLM = 桩（覆盖有限，缺真实先例/完整规避史）→ 可能低估 Agent 真实上限")
    add("    - 高估侧: 本集真值由生成器按「与审查员同源 EVAL_* 世界 + 同语义规则」程序化标注（单标注者、"
        "SHOULD 无负例）")
    add("      → scripted 高分含「标注-审查员同口径」耦合，主要衡量实现一致性而非调查能力；")
    add("      不可外推为真实 LLM 能力（理由见上一行同口径耦合）")
    add("    - real 对照（v1 35 案 real 单次抽样）: acc 0.200 / human_review_rate 0.771，")
    add("      27/35 转人工由确定性 Gate 归因（R3_BUDGET_EXHAUSTED×19 / R3_HYPOTHESES_INDISTINGUISHABLE×7）")
    add("      —— 该抽样早于 Gate 语义重构，其中归因码已移除（仅作历史对照）")
    add("      —— 与 scripted 高分方向相反（同口径耦合只会高估一致性，real 未调优首跑则大幅保守转人工）；"
        "该抽样仅验证链路，非模型固定水平")

    add("-" * 100)
    add("总体指标   acc   prec  recall  fpr   fnr   hrr   auto    TP/FP/TN/FN   成本均值(llm/tool/tok)")
    for scheme in ALL_SCHEMES:
        m = result.overall.get(scheme)
        if m is None:
            continue
        add(_overall_row(scheme, m, result.cost_summary.get(scheme, {})))
    if result.has_should_abstain and result.abstention:
        _render_abstention_section(add, result)
    if result.agent_metrics is not None:
        _render_agent_metrics_section(add, result)
    if result.engineering:
        _render_engineering_section(add, result)

    add("-" * 100)
    add("按 scene 分层  acc   prec  recall  fpr   fnr   hrr   auto    TP/FP/TN/FN")
    scene_n = {s: int(by_scene.get(s, {}).get("total", 0)) for s in _SCENES}
    # P2-4：分层分母 = 该 scene 二值真值案数；scene 含 HUMAN 真值时标注（仅标注，不改计算）
    parts = []
    for s in _SCENES:
        entry = by_scene.get(s, {}) or {}
        tot = int(entry.get("total", 0))
        bin_n = int(entry.get("PASS", 0)) + int(entry.get("REJECT", 0))
        parts.append(f"{s}={tot}" if bin_n == tot else f"{s}={tot}(二值{bin_n})")
    add("  " + "  ".join(parts))
    if human_n:
        add("  （注: 分层行分母=该 scene 二值真值数 PASS+REJECT；HUMAN 真值案不计入下列 acc/prec/recall 数字）")
    for scene in _SCENES:
        if scene_n[scene] == 0:
            continue
        add(f"[{scene}]")
        for scheme in ALL_SCHEMES:
            m = (result.grouped.get(scheme) or {}).get(scene)
            if m is not None:
                add(_scene_row(scheme, m))

    add("-" * 100)
    add("决策分布审计（pred→truth 列计数；truth 仅 PASS/REJECT（二值真值），HUMAN 真值案不计入本矩阵）")
    add("  scheme           pred         →truth PASS →truth REJECT")
    for scheme in ALL_SCHEMES:
        matrix = _decision_matrix(result, scheme)
        for pred in _DECISIONS:
            cells = matrix[pred]
            add(
                "  " + "  ".join(
                    [f"{scheme if pred == 'PASS' else '':<16}", f"{pred:<13}",
                     str(cells["PASS"]), str(cells["REJECT"])]
                )
            )
    add("=" * 100)
    return "\n".join(out)


def _render_abstention_section(add, result: EvaluationResult) -> None:
    """abstention 五指标渲染区（仅数据集含 SHOULD_ABSTAIN 真值时）。

    分母 = 全量：``human_review_rate`` = pred HUMAN / 全部；``automation_coverage`` =
    1 − human_review_rate；``abstention_rate`` = AUTO_DECIDABLE 案中 pred HUMAN
    （过度保守）；``abstention_recall`` = SHOULD_ABSTAIN 案中 pred HUMAN（越高越克制）；
    ``wrong_auto_decision_rate`` = AUTO_DECIDABLE 自动终裁中的错误占比。
    与决策指标行的区别：决策行的 hrr/auto 以二值真值为分母（见上节分母注记）。
    """
    auto_n = sum(1 for v in result.expected.values() if abstain_subset_of(v) == "AUTO_DECIDABLE")
    should_n = sum(1 for v in result.expected.values() if abstain_subset_of(v) == "SHOULD_ABSTAIN")
    binary_n = sum(
        int(s.get("PASS", 0)) + int(s.get("REJECT", 0))
        for s in result.scene_stats.get("by_scene", {}).values()
        if isinstance(s, dict)
    )
    add("-" * 100)
    add("abstention 五指标（分母=全量: AUTO_DECIDABLE=%d / SHOULD_ABSTAIN=%d）" % (auto_n, should_n))
    add("  行含义: hrr/autom=human_review_rate/automation_coverage(全量分母) | "
        "abst_r=abstention_rate(AUTO 中过度转人工) / abst_rl=abstention_recall(SHOULD 正确转人工) / "
        "w_auto=wrong_auto_decision_rate(AUTO 自动终裁错误率)")
    add("  scheme           hrr    autom  abst_r abst_rl w_auto   计数(AUTO/SHOULD；pred H/A)")
    for scheme in ALL_SCHEMES:
        a = result.abstention.get(scheme)
        if a is None:
            continue
        add(
            "  ".join(
                [
                    f"{scheme:<16}",
                    _fmt(a.human_review_rate),
                    _fmt(a.automation_coverage),
                    _fmt(a.abstention_rate),
                    _fmt(a.abstention_recall),
                    _fmt(a.wrong_auto_decision_rate),
                    f"{a.auto_decidable_total}/{a.should_abstain_total}; {a.human_pred_total}/{a.auto_pred_total}",
                ]
            )
        )
    add("  （注: 决策指标行 hrr/auto 分母=二值真值 %d，与本区 hrr/autom 全量分母不同，勿混读）"
        % binary_n)


def _render_agent_metrics_section(add, result: EvaluationResult) -> None:
    """Agent 级指标渲染区（工具选择 / 证据充分性 / 推理正确性 / 边际增益）。

    全部为**只读统计**：不改判定、不进任何既有指标分母。空真值案（未标注工具期望 /
    干净案无期望证据 / 无期望 risk_type）一律不进分母，被排除的案数与无法映射的标签数
    在行内显式给出（不静默丢弃）。推理正确性为**自动代理**（risk_type + risk_level），
    不含人工复核理由。
    """
    am = result.agent_metrics
    if am is None:  # 调用方已判空；本函数只做渲染
        return
    ts, es, rc, mg = am.tool_selection, am.evidence_sufficiency, am.reasoning_correctness, am.marginal_gain
    add("-" * 100)
    add("Agent 级指标（仅 agent；空真值案不进分母，被排除的案数行内显式给出）")
    add("  工具选择: 覆盖口径 expected_tools ⊆ actual_tools（不把「调用少」混进准确性）")
    add(
        f"  tool_selection_accuracy = {_fmt(ts.tool_selection_accuracy)}   "
        f"（{ts.covered_cases}/{ts.cases_with_expectation} 案；"
        f"期望外调用案 {ts.redundant_cases}，期望外工具数 {ts.redundant_tool_calls}/{ts.actual_tool_calls}）"
    )
    add(
        f"  redundant_tool_rate      = {_fmt(ts.redundant_tool_rate)}   "
        "（有期望外调用的案占比；工具名去重口径）"
    )
    add("  证据充分性: 两栏分开（类型覆盖率 micro + REJECT 依据前置），不合成一个数")
    add(
        f"  evidence_type_coverage   = {_fmt(es.evidence_type_coverage)}   "
        f"（{es.covered_expected_types}/{es.total_expected_types} 期望类型被命中；"
        f"全命中案 {es.fully_covered_cases}/{es.cases_with_expected_evidence}）"
    )
    add(
        f"  unmapped 期望标签 {es.unmapped_label_instances} 个实例、"
        f"仅含未映射标签被剔除的案 {es.cases_excluded_unmapped_only}（缺口显式化，不硬猜映射）"
    )
    add(
        f"  reject_evidence_gate_pass_rate = {_fmt(es.reject_evidence_gate_pass_rate)}   "
        f"（预测 REJECT {es.reject_cases_pred_reject} 案中 {es.reject_cases_with_citable} 案"
        "有可引用依据 = Gate 前置近似）"
    )
    add("  推理正确性（自动代理，不含人工复核理由）:")
    add(
        f"  risk_type_coverage       = {_fmt(rc.risk_type_coverage)}   "
        f"（{rc.risk_type_covered_cases}/{rc.cases_with_expected_risk_type} 案 expected.risk_type ⊆ 输出）"
    )
    add(
        f"  risk_level_agreement     = {_fmt(rc.risk_level_agreement)}   "
        f"（{rc.risk_level_agreement_cases}/{rc.cases_with_expected_risk_level} 案 risk_level 与真值一致）"
    )
    add("  边际证据增益（无加权合成；只计 status==ok 的调用）:")
    add(
        f"  evidence_gain_rate       = {_fmt(mg.evidence_gain_rate)}   "
        f"（{mg.calls_with_new_evidence}/{mg.ok_tool_calls} 调用带来新增证据；无增益调用 {mg.no_gain_calls}）"
    )
    add(
        f"  decision_changed_rate    = {_fmt(mg.decision_changed_rate)}   "
        f"（{mg.calls_decision_changed}/{mg.ok_tool_calls} 调用触发 Gate 判定翻转；"
        f"新增证据引用合计 {mg.evidence_added_total}）"
    )


def _render_engineering_section(add, result: EvaluationResult) -> None:
    """工程指标渲染区：各方案 调用次数 / token / 延迟 的均值与 P50/P95。

    scripted 路径 ``tokens=0`` 是真实情况（桩不烧 token），如实显示；墙钟延迟**不落
    EvalRecord**，只有 real 臂在进程内计时后传入（本节在主报告里通常为 "-"）。
    """
    add("-" * 100)
    add("工程指标（分布；脚本路径 token 恒 0 为真实值，不伪造）")
    add("  scheme           llm mean/p50/p95    tool mean/p50/p95    tok mean/p50/p95    latency(p50/p95)")
    for scheme in ALL_SCHEMES:
        e = result.engineering.get(scheme)
        if e is None:
            continue

        def _triple(d):
            return f"{_fmt(d.mean)}/{_fmt(d.p50)}/{_fmt(d.p95)}"

        lat = "-" if e.latency_ms is None else f"{_fmt(e.latency_ms.p50)}/{_fmt(e.latency_ms.p95)}"
        add(
            "  ".join(
                [
                    f"{scheme:<16}",
                    f"{_triple(e.llm_calls):<17}",
                    f"{_triple(e.tool_calls):<20}",
                    f"{_triple(e.tokens):<19}",
                    lat,
                ]
            )
        )
    add("  （注: P50/P95 为 nearest-rank；延迟为 real 臂进程内墙钟，scripted 无此项）")


def _decision_matrix(result: EvaluationResult, scheme: str) -> dict:
    """pred(行) × truth(列 PASS/REJECT) 计数（由 records × expected 直接重建，确定性）。"""
    matrix = {pred: {"PASS": 0, "REJECT": 0} for pred in _DECISIONS}
    expected = result.expected
    for rec in result.records.get(scheme) or []:
        truth = (expected.get(rec.eval_case_id) or {}).get("decision")
        if truth not in ("PASS", "REJECT"):
            continue
        matrix[rec.decision][truth] += 1
    return matrix


def print_report(result: EvaluationResult) -> None:
    """打印 Console Report 到 stdout。"""
    print(render_report(result))
