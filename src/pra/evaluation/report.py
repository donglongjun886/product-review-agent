"""Console Report 渲染：两臂业务指标 / 工程分布 / Agent 级指标 / overrides 归因 / 逐案明细。

``render_report(payload, extra, *, out_path=None)`` 返回整份纯文本报告，``print_report`` 打印到
stdout。``payload`` = ``run_evaluation_real.py --out`` 落盘的 JSON（数据集路径 / 模型 / count /
逐案 real 记录 / overrides 汇总）；``extra`` = 渲染中间物（``rows`` 逐案对比行、``by_scene``
一致计数、两臂 ``*_metrics`` / ``*_engineering`` / ``*_agent_metrics`` / ``*_overrides``、
``scene_stats``）；``out_path`` 非 None 时末行提示 JSON 落盘路径。

两臂为 rule（``RuleBaseline`` 确定性初筛）与 real（真实 LLM，非确定性、不可重放）。业务指标
同时用两套分母（AUTO_DECIDABLE 与全量），表下注记逐项写清，勿混读。
"""

from __future__ import annotations

from pra.agent.guardrails.gate import (
    R3_BUDGET_EXHAUSTED,
    R5_DEGRADED_OR_FAILED_STEP,
)
from pra.evaluation.harness.agent_scheme import EVAL_WORLD_LABEL
from pra.evaluation.metrics.business import DecisionMetrics
from pra.evaluation.metrics.engineering import DistributionMetrics, EngineeringMetrics

__all__ = ["print_report", "render_report"]

SCENES = ("normal", "violation", "boundary", "multi-signal", "evasion")


def _fmt(v: float | None) -> str:
    """比率单元格：None（分母为 0，未定义）→ ``-``，否则 3 位小数。"""
    return "-" if v is None else f"{v:.3f}"


def _fmt0(v: float | None) -> str:
    """分布单元格：None → ``-``，否则紧凑数值。"""
    return "-" if v is None else f"{v:g}"


def _triple(d: DistributionMetrics) -> str:
    """分布三元组 ``mean/p50/p95``。"""
    return f"{_fmt0(d.mean)}/{_fmt0(d.p50)}/{_fmt0(d.p95)}"


def _table(headers: list[str], rows: list[list[str]], *, indent: str = "  ") -> list[str]:
    """渲染左对齐文本表：列宽 = 该列最大单元格宽度，按传入顺序输出表头与数据行。"""
    widths = [
        max([len(head), *(len(row[i]) for row in rows)]) for i, head in enumerate(headers)
    ]
    lines = [indent + "  ".join(head.ljust(w) for head, w in zip(headers, widths))]
    lines.extend(indent + "  ".join(cell.ljust(w) for cell, w in zip(row, widths)) for row in rows)
    return lines


def _metrics_row(label: str, m: DecisionMetrics, eng: EngineeringMetrics) -> list[str]:
    """业务指标一行的单元格：三项 AUTO_DECIDABLE 分母比率 + 真值 REJECT 分母比率 + 全量分母比率
    + ``TP/FP/TN/FN`` + 成本均值（取 ``EngineeringMetrics.*.mean``）。"""
    return [
        label,
        _fmt(m.accuracy),
        _fmt(m.precision),
        _fmt(m.recall),
        _fmt(m.wrong_auto_decision_rate),
        _fmt(m.reject_unhandled),
        _fmt(m.human_review_rate),
        _fmt(m.automation_coverage),
        f"{m.tp}/{m.fp}/{m.tn}/{m.fn}",
        f"llm={_fmt0(eng.llm_calls.mean)} tool={_fmt0(eng.tool_calls.mean)} tok={_fmt0(eng.tokens.mean)}",
    ]


def _risk_cell(row: dict) -> str:
    """逐案 real 风险摘要 ``level/types/conf``（缺值 → ``-``）。"""
    types = ",".join(row["real_risk_type"]) or "-"
    conf = "-" if row["real_decision_confidence"] is None else f"{row['real_decision_confidence']:.2f}"
    return f"{row['real_risk_level'] or '-'}/{types}/conf={conf}"


def _agent_cell(triple: tuple) -> str:
    """Agent 指标单元格 ``(值, 分子, 分母)``；值 None → 未定义显示 ``-``（不填 0 冒充）。"""
    value, numer, denom = triple
    body = "-" if value is None else f"{value:.3f}"
    return f"{body}({numer}/{denom})"


def _agent_metrics_lines(extra: dict) -> list[str]:
    """Agent 级指标区（两臂并排 · 只读统计；空真值案不进分母，分子/分母行内给出）。"""
    ts_rule = extra["rule_agent_metrics"].tool_selection
    rc_rule = extra["rule_agent_metrics"].reasoning_correctness
    mg_rule = extra["rule_agent_metrics"].marginal_gain
    ts_real = extra["real_agent_metrics"].tool_selection
    rc_real = extra["real_agent_metrics"].reasoning_correctness
    mg_real = extra["real_agent_metrics"].marginal_gain
    rows = [
        [
            "tool_selection_accuracy",
            _agent_cell((ts_rule.tool_selection_accuracy, ts_rule.covered_cases, ts_rule.cases_with_expectation)),
            _agent_cell((ts_real.tool_selection_accuracy, ts_real.covered_cases, ts_real.cases_with_expectation)),
        ],
        [
            "redundant_tool_rate",
            _agent_cell((ts_rule.redundant_tool_rate, ts_rule.redundant_cases, ts_rule.cases_with_expectation)),
            _agent_cell((ts_real.redundant_tool_rate, ts_real.redundant_cases, ts_real.cases_with_expectation)),
        ],
        [
            "risk_type_coverage",
            _agent_cell((rc_rule.risk_type_coverage, rc_rule.risk_type_covered_cases, rc_rule.cases_with_expected_risk_type)),
            _agent_cell((rc_real.risk_type_coverage, rc_real.risk_type_covered_cases, rc_real.cases_with_expected_risk_type)),
        ],
        [
            "risk_level_agreement",
            _agent_cell((rc_rule.risk_level_agreement, rc_rule.risk_level_agreement_cases, rc_rule.cases_with_expected_risk_level)),
            _agent_cell((rc_real.risk_level_agreement, rc_real.risk_level_agreement_cases, rc_real.cases_with_expected_risk_level)),
        ],
        [
            "evidence_gain_rate",
            _agent_cell((mg_rule.evidence_gain_rate, mg_rule.calls_with_new_evidence, mg_rule.ok_tool_calls)),
            _agent_cell((mg_real.evidence_gain_rate, mg_real.calls_with_new_evidence, mg_real.ok_tool_calls)),
        ],
        [
            "decision_changed_rate",
            _agent_cell((mg_rule.decision_changed_rate, mg_rule.calls_decision_changed, mg_rule.ok_tool_calls)),
            _agent_cell((mg_real.decision_changed_rate, mg_real.calls_decision_changed, mg_real.ok_tool_calls)),
        ],
    ]
    return [
        "Agent 级指标（两臂同口径 · 只读统计；空真值案不进分母，分子/分母行内给出）:",
        *_table(["metric", "rule(num/den)", "real(num/den)"], rows),
    ]


def _overrides_cell(row: dict) -> str:
    """逐案 real overrides 缩写（R3=预算截胡 / R5=LLM 步降级兜底）。"""
    ovs = row.get("real_overrides") or []
    if not ovs:
        return "-"
    short = {R3_BUDGET_EXHAUSTED: "R3", R5_DEGRADED_OR_FAILED_STEP: "R5"}
    return ",".join(short.get(code, code) for code in ovs)


def _overrides_line(ov: dict, total: int) -> str:
    """一行 overrides 汇总：带码案数 / R5 降级 / R3 截胡（含先撞维度）/ 混合 / 其它码。"""
    parts = [
        f"带 overrides {ov['cases_with_any']}/{total} 案",
        f"R5 降级 {ov[R5_DEGRADED_OR_FAILED_STEP]} 案",
        f"R3 预算截胡 {ov[R3_BUDGET_EXHAUSTED]} 案",
        f"R3+R5 混合 {ov['r3_r5_mixed']} 案",
    ]
    if ov.get("budget_hit_dims"):
        parts.append("R3 先撞限 " + ",".join(f"{k}={v}" for k, v in ov["budget_hit_dims"].items()))
    if ov.get("other_codes"):
        parts.append("其它码 " + ",".join(f"{k}={v}" for k, v in ov["other_codes"].items()))
    return " ｜ ".join(parts)


def render_report(payload: dict, extra: dict, *, out_path: str | None = None) -> str:
    """渲染整份 Console Report（纯文本；real 臂非确定性如实标注）。"""
    out: list[str] = []
    add = out.append

    stats = extra["scene_stats"]
    by_scene = stats.get("by_scene", {})
    scene_n = {scene: int(by_scene.get(scene, {}).get("total", 0)) for scene in SCENES}
    truth_n = {
        label: sum(int(by_scene.get(scene, {}).get(label, 0)) for scene in SCENES)
        for label in ("PASS", "REJECT", "HUMAN_REVIEW")
    }
    rule_metrics: DecisionMetrics = extra["rule_metrics"]
    total, agree = payload["count"], payload["agree"]
    diff_n = total - agree
    truth_human = rule_metrics.total - rule_metrics.auto_decidable_total

    add("=" * 100)
    add("商品审核 Agent · Evaluation 正式跑分：rule（确定性初筛）vs real（真实 LLM）")
    add("=" * 100)
    add(f"数据集: {payload['data']}（{payload['count']} 条）| real 模型: {payload['model']}")
    add("scene 分布: " + " | ".join(f"{scene}={scene_n[scene]}" for scene in SCENES))
    add("真值分布: " + " / ".join(f"{label}={n}" for label, n in truth_n.items()))

    add("-" * 100)
    add("结论边界 / 口径注记:")
    add("  · rule = RuleBaseline（pra.screening 确定性三分流；零 LLM、零工具，COMPLEX → HUMAN_REVIEW）")
    add(f"  · real = {payload['model']}（真实 LLM —— 非确定性、不可重放、需 API key 与费用；")
    add("    本报告 real 数字 = 单次运行抽样，不代表模型固定水平）")
    add(f"  · 工具数据源: {EVAL_WORLD_LABEL}（世界固定为 Eval World；两臂同一数据 → LLM 是唯一变量）")
    conc = payload.get("real_concurrency") or 1
    add(
        "  · real 臂调度: "
        + (
            "并发 " + str(conc) + "（只改调度；用例间天然隔离 —— 每案独立建图，LLM 后端随图"
            "显式注入，不跨案共享）"
            if conc > 1
            else "逐案串行"
        )
    )
    add("  · 一致性口径: rule.decision == real.decision 判为一致（risk/evidence 差异不参与）")
    add("  · 指标口径（DecisionEvaluator）: accuracy=(TP+TN)/AUTO_DECIDABLE 案数，pred HUMAN_REVIEW 计为判错")
    add("    （入分母不入分子）；precision/wrong_auto_decision_rate 与 TP/FP/TN/FN 只在 AUTO_DECIDABLE 案上计算")
    add("  · REJECT 为正类: recall=TP/真值 REJECT 数（转人工算未拦下）/ wrong_auto_decision_rate=(FP+FN)/(TP+FP+TN+FN) /")
    add("    reject_unhandled = 真值 REJECT 中 pred HUMAN_REVIEW 占比；三者恒等: recall + reject_unhandled + 漏放率 = 1")
    add("  · cost.tokens 口径 = usage.total_tokens：input+output 合计、含 provider 缓存命中 token；schema 校验")
    add("    失败的尝试也全额累计。tokens 为观测字段，**不参与** R3 预算截胡判定（R3 只看 llm_calls /")
    add("    tool_calls）；EvalRecord.detail.budget_hit_dim 记录哪一维先撞限")
    if truth_human:
        add(
            f"  · 真值含 HUMAN_REVIEW 的案 {truth_human} 条（SHOULD_ABSTAIN）：只进全量分母，"
            "不进 AUTO_DECIDABLE 分母 —— 如实呈现，不硬算"
        )

    add("-" * 100)
    share = f"{agree / total:.1%}" if total else "-"
    add(f"rule vs real 决策一致性: 一致 {agree}/{total}（{share}）· 差异 {diff_n} 条")
    scene_parts = []
    for scene in SCENES:
        counter = extra["by_scene"].get(scene)
        if counter and counter["total"]:
            scene_parts.append(f"{scene}={counter['agree']}/{counter['total']}")
    add("按 scene 一致数: " + (" | ".join(scene_parts) if scene_parts else "-"))

    add("-" * 100)
    disagree = payload.get("disagree") or []
    if not disagree:
        add("差异 case 列表: （无 —— 两臂逐案裁决完全一致）")
    else:
        add(f"差异 case 列表（共 {len(disagree)} 条 · 各行含 real risk 摘要 + overrides）:")
        for row in disagree:
            ovr = _overrides_cell(row)
            ovr_note = f" | real ovr: {ovr}" if ovr != "-" else ""
            add(
                f"  · {row['eval_case_id']} [{row['scene']}] truth={row['truth']} | "
                f"rule {row['rule_decision']} → real {row['real_decision']} （{_risk_cell(row)}）{ovr_note}"
            )

    add("-" * 100)
    add("逐案对比明细:")
    out.extend(
        _table(
            ["case", "scene", "truth", "rule", "real", "agree", "real risk/type/conf", "real ovr"],
            [
                [
                    row["eval_case_id"],
                    row["scene"],
                    row["truth"],
                    row["rule_decision"],
                    row["real_decision"],
                    "是" if row["agree"] else "否",
                    _risk_cell(row),
                    _overrides_cell(row),
                ]
                for row in extra["rows"]
            ],
        )
    )

    add("-" * 100)
    add("业务指标（两臂同口径 · DecisionEvaluator）:")
    out.extend(
        _table(
            [
                "arm",
                "accuracy",
                "precision",
                "recall",
                "wrong_auto_decision_rate",
                "reject_unhandled",
                "human_review_rate",
                "automation_coverage",
                "TP/FP/TN/FN",
                "成本均值(llm/tool/tok)",
            ],
            [
                _metrics_row("rule", extra["rule_metrics"], extra["rule_engineering"]),
                _metrics_row("real", extra["real_metrics"], extra["real_engineering"]),
            ],
        )
    )
    add(
        f"  分母注记: accuracy/precision/wrong_auto_decision_rate 与 TP/FP/TN/FN 的分母 = "
        f"AUTO_DECIDABLE 案 {rule_metrics.auto_decidable_total}（真值 PASS/REJECT）；"
        f"recall/reject_unhandled 分母 = 真值 REJECT 案 {rule_metrics.reject_truth}；"
        f"human_review_rate/automation_coverage 分母 = 全量 {rule_metrics.total} —— 三组分母不同，勿混读"
    )

    add("-" * 100)
    add("工程指标（分布：均值/P50/P95；rule 无 LLM/工具调用 → 恒 0；latency_ms 仅 real 臂进程内墙钟、不落 record）:")
    out.extend(
        _table(
            ["arm", "llm_calls", "tool_calls", "tokens", "latency_ms"],
            [
                [label, _triple(eng.llm_calls), _triple(eng.tool_calls), _triple(eng.tokens), _triple(eng.latency_ms) if eng.latency_ms else "-"]
                for label, eng in (("rule", extra["rule_engineering"]), ("real", extra["real_engineering"]))
            ],
        )
    )

    add("-" * 100)
    out.extend(_agent_metrics_lines(extra))

    add("-" * 100)
    add("overrides 汇总（审计：R5=LLM 步降级兜底转 HUMAN、R3=预算截胡）:")
    add(f"  · real: {_overrides_line(extra['real_overrides'], total)}")
    add(f"  · rule: {_overrides_line(extra['rule_overrides'], total)}")
    real_ov = extra["real_overrides"]
    if real_ov[R5_DEGRADED_OR_FAILED_STEP]:
        add(
            f"    ⚠ real 有 {real_ov[R5_DEGRADED_OR_FAILED_STEP]} 案触发 R5 降级 —— 这些案的 real 裁决"
            "来自降级兜底（HUMAN），**不是模型行为**；解读全卷差异/指标须扣除"
        )
    if total and real_ov[R5_DEGRADED_OR_FAILED_STEP] == total:
        add(
            "    ⚠⚠ real 全部案均 R5 降级：本卷 real 结果 = 100% 链路降级（典型原因：无 key/网关/"
            "base-url 配置问题或逐节点连续失败）—— 请勿把本卷 HUMAN 当模型结论"
        )

    add("-" * 100)
    out_note = f" | JSON 已写入: {out_path}" if out_path else " | 未写文件（--out 可落盘）"
    add(f"[OK] 跑分完成: {total} 条 · 一致 {agree} · 差异 {diff_n}{out_note}")
    add(f"[NOTE] {payload['note']} —— real 侧输出不可用于逐字节回归比对")
    add("=" * 100)
    return "\n".join(out)


def print_report(payload: dict, extra: dict, *, out_path: str | None = None) -> None:
    """打印 Console Report 到 stdout。"""
    print(render_report(payload, extra, out_path=out_path))
