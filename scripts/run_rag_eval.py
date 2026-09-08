"""run_rag_eval.py —— RAG 世界评测入口（M4：InMemory vs RAG + BM25/Vector/Hybrid 三路）。

用法::

    uv run python scripts/run_rag_eval.py                       # 全量 v1（35 条）agent 方案
    uv run python scripts/run_rag_eval.py --smoke --smoke-limit 10
    uv run python scripts/run_rag_eval.py --modes bm25 hybrid   # 只跑指定 RAG 模式
    uv run python scripts/run_rag_eval.py --data eval_data/v2/cases_v2.jsonl

语义（rag-implementation-plan.md R-4 / R-6 / M4）：
- 评测**默认仍 InMemory**（回归不破坏）；本脚本是 RAG 单独模式入口 —— 对同一
  eval_data 分别以 ``tool_world="eval"``（InMemory）与 ``tool_world="rag"`` +
  mode ∈ {bm25, vector, hybrid} 跑 ``agent`` 方案，输出：
  1. 各世界/模式的决策指标（Accuracy/Precision/Recall/HMR 等，DecisionEvaluator
     口径）与决策序列 digest；
  2. InMemory vs 各 RAG 模式的**逐案决策差异**（差异 count + 前 N 条 transition），
     结论边界收窄说明（真实政策/先例检索下 Agent 表现如何）；
  3. 三检索模式并排（**不预设 Hybrid 优于单路 —— 如实呈现**）；
  4. 运行时证据级隔离抽查：RAG 世界实际引用的先例 ref_id 全部为 ``RAG_CASE_*``
     （无一引用 eval GT / InMemory 种子先例）。
全链路确定性：无真 LLM / 无网络 / 顺序串行；同数据重跑逐字节可重放。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys

from pra.evaluation.harness.base import EvalContext
from pra.evaluation.runner import EvaluationRunner

MODES = ("bm25", "vector", "hybrid")
DEFAULT_DATA = "eval_data/v1/cases_v1.jsonl"
DIFF_HEAD = 12  # 差异明细打印条数上限


def _decision_digest(records) -> str:
    seq = [r.decision for r in records]
    return hashlib.sha256(
        "|".join(seq).encode("utf-8")
    ).hexdigest()[:16]


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.3f}"


def _metrics_row(label: str, result, records) -> str:
    m = result.overall["agent"]
    cost = result.cost_summary.get("agent") or {}
    return (
        f"{label:<16} acc={_fmt(m.accuracy)} prec={_fmt(m.precision)} "
        f"recall={_fmt(m.recall)} fpr={_fmt(m.fpr)} fnr={_fmt(m.fnr)} "
        f"hmr={_fmt(m.human_rate)} auto={_fmt(m.automation)} "
        f"tool={cost.get('tool_calls', 0):.2f} digest={_decision_digest(records)}"
    )


def _transition_label(r) -> str:
    extra = ""
    if r.detail.get("overrides"):
        extra = f"[{','.join(r.detail['overrides'])}]"
    return f"{r.decision}{extra}"


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    modes = list(args.modes)
    configs: list[tuple[str, dict]] = [("InMemory(eval)", {})]
    configs += [(f"RAG-{mode}", {"tool_world": "rag", "rag_mode": mode}) for mode in modes]

    print("=" * 100)
    print("商品审核 Agent · RAG 世界 vs InMemory（agent 方案 · 决策序列 digest + 指标）")
    print("=" * 100)
    print(f"数据集: {args.data}（smoke={args.smoke}）| 世界: eval(InMemory) + rag×{len(modes)} 模式")

    results: dict[str, tuple] = {}
    for label, overrides in configs:
        ctx = EvalContext(**overrides)
        runner = EvaluationRunner(data_path=args.data, ctx=ctx)
        result = await runner.run(include=("agent",), smoke=args.smoke, smoke_limit=args.smoke_limit)
        records = result.records["agent"]
        results[label] = (result, records, ctx)
        print(_metrics_row(label, result, records))

    # --- InMemory vs RAG 逐案差异 + 三路对比 --------------------------------
    mem_result, mem_records, _ = results["InMemory(eval)"]
    mem_by_id = {r.eval_case_id: r for r in mem_records}
    exp = mem_result.expected

    print("\n" + "-" * 100)
    print("逐案决策差异（相对 InMemory(eval)）")
    for label in [f"RAG-{m}" for m in modes]:
        result, records, _ctx = results[label]
        diffs = []
        for r in records:
            base = mem_by_id[r.eval_case_id]
            if base.decision != r.decision:
                scene = exp.get(r.eval_case_id, {}).get("scene", "?")
                diffs.append((r.eval_case_id, scene, base.decision, r.decision))
        same = len(records) - len(diffs)
        print(f"\n  {label}：与 InMemory 决策一致 {same}/{len(records)}，差异 {len(diffs)} 条")
        for cid, scene, base, cur in diffs[:DIFF_HEAD]:
            print(f"    · {cid} [{scene}] InMemory {base} → {label} {cur}")
        if len(diffs) > DIFF_HEAD:
            print(f"    … 其余 {len(diffs) - DIFF_HEAD} 条（见上方指标 digest）")
        if not diffs:
            print("    （本数据集上无决策差异 —— 证据内容/引用来源差异见下方隔离抽查）")

    # --- RAG 模式间三路对比（R-6：如实报告，不预设 Hybrid 最优） ------------
    print("\n" + "-" * 100)
    print("三检索模式对比（agent 决策序列两两比对）")
    mode_results = {f"RAG-{m}": results[f"RAG-{m}"][1] for m in modes}
    for i, m1 in enumerate(modes):
        for m2 in modes[i + 1:]:
            r1, r2 = mode_results[f"RAG-{m1}"], mode_results[f"RAG-{m2}"]
            d = sum(1 for a, b in zip(r1, r2) if a.decision != b.decision)
            print(f"  RAG-{m1:<7} vs RAG-{m2:<7}: {d}/{len(r1)} 条决策不同")

    # --- 证据来源差异摘要（决策相同 ≠ 证据相同：InMemory vs RAG 引用集合） ---
    print("\n" + "-" * 100)
    print("证据来源差异摘要（决策一致时仍看引用来源/集合）")
    print(
        f"  {'world':<16} {'REJECT':>6} {'w/policy':>8} {'w/case':>7} "
        f"{'distinct_case':>13} {'distinct_policy':>15}  样例 case refs"
    )
    for label in ["InMemory(eval)"] + [f"RAG-{m}" for m in modes]:
        result, records, _ = results[label]
        rejects = [r for r in records if r.decision == "REJECT"]
        pol_refs = {
            ev["ref_id"]
            for r in records
            for ev in r.evidence
            if ev.get("type") == "POLICY_REF" and ev.get("ref_id")
        }
        case_refs = {
            ev["ref_id"]
            for r in records
            for ev in r.evidence
            if ev.get("type") == "CASE_PRECEDENT" and ev.get("ref_id")
        }
        w_pol = sum(1 for r in rejects if any(
            ev.get("type") == "POLICY_REF" for ev in r.evidence))
        w_case = sum(1 for r in rejects if any(
            ev.get("type") == "CASE_PRECEDENT" for ev in r.evidence))
        sample = ", ".join(sorted(case_refs)[:4]) or "-"
        print(
            f"  {label:<16} {len(rejects):>6} {w_pol:>8} {w_case:>7} "
            f"{len(case_refs):>13} {len(pol_refs):>15}  {sample}"
        )

    # --- 证据级隔离抽查（R-4 运行期） --------------------------------------
    print("\n" + "-" * 100)
    print("运行时证据级隔离抽查（RAG 世界引用 ref_id 前缀）")
    for label in [f"RAG-{m}" for m in modes]:
        result, records, _ = results[label]
        refs = set()
        for r in records:
            for ev in r.evidence:
                if ev.get("type") in ("CASE_PRECEDENT", "POLICY_REF") and ev.get("ref_id"):
                    refs.add(ev["ref_id"])
        case_refs = sorted(x for x in refs if x.startswith("RAG_CASE_"))
        other_case_refs = sorted(x for x in refs if not x.startswith("RAG_CASE_") and not x.startswith("POLICY_"))
        bad = [x for x in other_case_refs if not x.startswith("POLICY_")]
        verdict = "PASS（RAG 世界仅引用 RAG_CASE_*/KB POLICY_*，无 eval GT / InMemory 先例）" if not bad else f"FAIL: {bad}"
        print(f"  {label}: CASE 引用 {len(case_refs)} 个（样例 {case_refs[:5]}…）| {verdict}")

    print("\n" + "=" * 100)
    print("[OK] RAG 世界评测完成（确定性；RAG 接入后的结论边界说明见上方差异与指标）")
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG 世界评测：InMemory vs RAG + 三模式对比")
    parser.add_argument("--data", default=DEFAULT_DATA, help=f"评测集 JSONL（默认 {DEFAULT_DATA}）")
    parser.add_argument("--modes", nargs="*", default=list(MODES), choices=list(MODES),
                        help="要跑的 RAG 检索模式（默认全三路）")
    parser.add_argument("--smoke", action="store_true", help="冒烟：只跑前 N 条（确定性）")
    parser.add_argument("--smoke-limit", type=int, default=10, help="smoke 上限（默认 10）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_main(argv))
    except Exception as exc:
        print(f"[FAIL] RAG 世界评测失败: {exc!r}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
