"""Console Report（evaluation/report.py）—— 总体 + 按 scene 分层的纯文本报告。

只做格式化（无文件 IO；``print_report`` 打印 stdout，供 scripts/run_evaluation.py
与 CI 日志使用）。口径说明随报告输出（docs/02-evaluation.md §8 M4"附结论边界标注"
的轻量版），见 render_report 内的"口径注记"块与模块 docstring：

- 数据集版本/分布（实际分布由 loader 统计，防 manifest 漂移）；
- 工具数据源 = InMemory 种子世界；LLM = 确定性桩（边界标注，见 agent_scheme）；
- 三分类输出空间对齐映射：Rule COMPLEX→HUMAN_REVIEW；Single-call conf<门槛的
  REJECT→HUMAN（见 single_call_scheme / metrics.business 各自注明）；
- Accuracy 口径：预测 HUMAN_REVIEW 计为"未命中业务真值"（判错：入分母不入
  (TP+TN) 分子）；Precision/Recall/FPR/FNR 只在自动判出子集上计算；
- v1 screening 词表现状：BLACKLISTED_BRANDS 为空 → Rule baseline 无自动 REJECT
  （品牌词/规避词/空缺一律 COMPLEX → 评测映射 HUMAN_REVIEW）—— 如实呈现，非缺陷。
"""

from __future__ import annotations

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


def render_report(result: EvaluationResult) -> str:
    """渲染整份 Console Report（纯文本；行内口径说明见模块 docstring）。"""
    out: list[str] = []
    add = out.append

    add("=" * 100)
    add("商品审核 Agent · Evaluation Phase 1 三方案对比 Console Report")
    add("=" * 100)
    src = result.data_path or "(外部注入 cases)"
    add(f"数据集: {src}" + ("   [smoke 冒烟子集]" if result.smoke else ""))
    stats = result.scene_stats
    by_scene = stats.get("by_scene", {})
    pass_n = sum(int(s.get("PASS", 0)) for s in by_scene.values() if isinstance(s, dict))
    reject_n = sum(int(s.get("REJECT", 0)) for s in by_scene.values() if isinstance(s, dict))
    add(f"真值案: 共 {int(stats.get('total', 0))} 条（PASS={pass_n} / REJECT={reject_n}）")
    dist = " | ".join(
        f"{scene}={int(by_scene.get(scene, {}).get('total', 0))}" for scene in _SCENES
    )
    add(f"scene 分布: {dist}")

    add("-" * 100)
    add("结论边界 / 口径注记:")
    add("  · 工具数据源: InMemory 种子世界 v1（与 eval_data/v1 同一份；仅 Agent 经工具取证可见）")
    add("  · LLM: 确定性桩（rule=无 / single_call=single-call-mock-v1 / agent=eval-scripted-reviewer）")
    add("  · 输出空间对齐: Rule COMPLEX→HUMAN_REVIEW（评测语义：不可自动判）；Single-call confidence<0.7 的 REJECT→HUMAN")
    add("  · Accuracy=(TP+TN)/真值总数，预测 HUMAN_REVIEW 计为未命中真值(判错，入分母不入分子)；")
    add("    Precision/Recall/FPR/FNR 只在自动判出(pred∈{PASS,REJECT})子集上计算")
    add("  · REJECT 为正类: Recall=TP/(TP+FN) 违规召回 / FPR=FP/(FP+TN) 误杀红线 / FNR=FN/(TP+FN) 漏放")
    add("  · HRR=转人工率 / auto=自动化率；reject_unhandled=该 REJECT 却转人工占比（保守度观测）")
    add("  · v1 BLACKLISTED_BRANDS 为空 → Rule 无自动 REJECT（品牌词/规避词/空缺一律 COMPLEX→HUMAN）")
    add("  · 结论边界: 工具为 InMemory 种子 + LLM 为桩 → 低估 Agent 上限；real 模式 / 全量指标留 Phase 2")

    add("-" * 100)
    add("总体指标   acc   prec  recall  fpr   fnr   hrr   auto    TP/FP/TN/FN   成本均值(llm/tool/tok)")
    for scheme in ALL_SCHEMES:
        m = result.overall.get(scheme)
        if m is None:
            continue
        add(_overall_row(scheme, m, result.cost_summary.get(scheme, {})))

    add("-" * 100)
    add("按 scene 分层  acc   prec  recall  fpr   fnr   hrr   auto    TP/FP/TN/FN")
    scene_n = {s: int(by_scene.get(s, {}).get("total", 0)) for s in _SCENES}
    add("  " + "  ".join(f"{s}={scene_n[s]}" for s in _SCENES))
    for scene in _SCENES:
        if scene_n[scene] == 0:
            continue
        add(f"[{scene}]")
        for scheme in ALL_SCHEMES:
            m = (result.grouped.get(scheme) or {}).get(scene)
            if m is not None:
                add(_scene_row(scheme, m))

    add("-" * 100)
    add("决策分布审计（每行: scheme  pred→truth 列计数；truth 仅 PASS/REJECT）")
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
