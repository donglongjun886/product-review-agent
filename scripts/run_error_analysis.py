"""错误归因分析：复用三方案评测结果，回答「Agent 修好了什么、还错在哪」。

输入与 ``run_evaluation.py`` 同一份 harness / 数据集（默认 v2）与同一套真值口径，
因此本脚本的业务数字必须与 Console Report 逐项一致（不一致即 harness 漂移，属缺陷）。

产出四块：三分类混淆矩阵（pred × truth，含 HUMAN 真值）、自动案错误构成、方案间决策迁移
（Rule / Single → Agent 的修好 / 退步 / 同错）、Agent 剩余失败清单。
只读统计：不改判定、不改 metrics 口径、不写 DB；``--json`` 可落一份机器可读产物。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from pra.evaluation.runner import ALL_SCHEMES, EvaluationRunner

#: 三分类标签（与 domain DecisionLabel 同集合；真值含 HUMAN_REVIEW = SHOULD_ABSTAIN）
LABELS: tuple[str, ...] = ("PASS", "REJECT", "HUMAN_REVIEW")
#: 二值真值（PASS/REJECT）—— 业务指标分母
AUTO_TRUTH: frozenset[str] = frozenset({"PASS", "REJECT"})
DEFAULT_DATA = "eval_data/v2/cases_v2.jsonl"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="三方案错误归因分析（确定性、只读、不落 DB）")
    parser.add_argument("--data", default=DEFAULT_DATA, help=f"评测集 JSONL（默认 {DEFAULT_DATA}）")
    parser.add_argument("--json", default=None, help="把全部统计写入该 JSON 路径（默认只打印）")
    return parser.parse_args(argv)


def _fill(preds: dict[str, str], truth: dict[str, str]) -> dict[str, dict[str, int]]:
    """pred（行）× truth（列）三分类计数矩阵。"""
    matrix = {p: {t: 0 for t in LABELS} for p in LABELS}
    for case_id, t in truth.items():
        matrix[preds[case_id]][t] += 1
    return matrix


def _auto_errors(preds: dict[str, str], truth: dict[str, str]) -> dict:
    """二值真值案上的错误构成（严格口径：pred HUMAN 也算错）。"""
    wrong = {cid: (truth[cid], preds[cid]) for cid in truth if truth[cid] in AUTO_TRUTH and preds[cid] != truth[cid]}
    return {
        "auto_truth_total": sum(1 for t in truth.values() if t in AUTO_TRUTH),
        "wrong_total": len(wrong),
        "wrong_abstained": sum(1 for t, p in wrong.values() if p == "HUMAN_REVIEW"),  # 过度转人工
        "wrong_flipped": sum(1 for t, p in wrong.values() if p in AUTO_TRUTH and p != t),  # 判反
        "wrong_missed_violation": sum(1 for t, p in wrong.values() if t == "REJECT" and p == "PASS"),  # 漏放
        "wrong_false_kill": sum(1 for t, p in wrong.values() if t == "PASS" and p == "REJECT"),  # 误杀
    }


def _transition(
    auto_map: dict[str, str], agent_map: dict[str, str], truth: dict[str, str]
) -> dict[str, int]:
    """二值真值案上 baseline → agent 的决策迁移四格 + 我方归因细分。"""
    out = {"both_right": 0, "fixed_by_agent": 0, "broke_by_agent": 0, "both_wrong": 0}
    detail = {"fixed_from_abstain": 0, "fixed_from_flip": 0, "broke_to_abstain": 0, "broke_to_flip": 0}
    for cid, t in truth.items():
        if t not in AUTO_TRUTH:
            continue
        base_right = auto_map[cid] == t
        agent_right = agent_map[cid] == t
        if base_right and agent_right:
            out["both_right"] += 1
        elif not base_right and agent_right:
            out["fixed_by_agent"] += 1
            detail["fixed_from_abstain" if auto_map[cid] == "HUMAN_REVIEW" else "fixed_from_flip"] += 1
        elif base_right and not agent_right:
            out["broke_by_agent"] += 1
            detail["broke_to_abstain" if agent_map[cid] == "HUMAN_REVIEW" else "broke_to_flip"] += 1
        else:
            out["both_wrong"] += 1
    out.update(detail)
    return out


def _agent_failures(agent_map: dict[str, str], truth: dict[str, str], scenes: dict[str, str]) -> list[dict]:
    """Agent 未命中真值的案件清单（严格口径，含 SHOULD_ABSTAIN 真值案）。"""
    rows: list[dict] = []
    for cid, t in truth.items():
        p = agent_map[cid]
        if p == t:
            continue
        if t in AUTO_TRUTH and p == "HUMAN_REVIEW":
            reason = "过度转人工"
        elif t in AUTO_TRUTH and p in AUTO_TRUTH:
            reason = "漏放" if (t == "REJECT" and p == "PASS") else "误杀"
        elif t == "HUMAN_REVIEW":
            reason = "该转人工却自动终裁"
        else:
            reason = "其他"
        rows.append({"case_id": cid, "scene": scenes.get(cid), "truth": t, "pred": p, "reason": reason})
    return rows


def _matrix_lines(name: str, matrix: dict[str, dict[str, int]]) -> list[str]:
    lines = [f"[{name}]   pred \\ truth" + "".join(f"{t:>14}" for t in LABELS)]
    for p in LABELS:
        lines.append(f"  {p:<14}" + "".join(f"{matrix[p][t]:>14}" for t in LABELS))
    return lines


async def _main() -> int:
    args = _parse_args()
    result = await EvaluationRunner(data_path=args.data).run(include=list(ALL_SCHEMES))
    truth = {cid: meta["decision"] for cid, meta in result.expected.items()}
    scenes = {cid: meta.get("scene") for cid, meta in result.expected.items()}
    preds = {name: {r.eval_case_id: r.decision for r in recs} for name, recs in result.records.items()}

    matrices = {}
    for name in ALL_SCHEMES:
        matrices[name] = _fill(preds[name], truth)

    print("=" * 96)
    print("三方案错误归因分析（数据源与 run_evaluation.py 同一份 harness，确定性）")
    print(f"数据集: {args.data}    案件数: {result.total_cases}")
    print("真值分布: " + " / ".join(f"{t}={sum(1 for v in truth.values() if v == t)}" for t in LABELS))
    print("=" * 96)

    print("\n[1] 三分类混淆矩阵（行 = pred，列 = truth；对角线即命中）")
    for name in ALL_SCHEMES:
        for line in _matrix_lines(name, matrices[name]):
            print(line)
        hit = sum(matrices[name][t][t] for t in LABELS)
        print(f"  → 命中 {hit}/{result.total_cases}；错列明细见上\n")

    print("[2] 二值真值案（PASS/REJECT）错误构成（严格口径：pred HUMAN_REVIEW 计错）")
    errors = {}
    for name in ALL_SCHEMES:
        e = _auto_errors(preds[name], truth)
        errors[name] = e
        print(
            f"  {name:<16} 错 {e['wrong_total']:>3}/{e['auto_truth_total']}"
            f" = 过度转人工 {e['wrong_abstained']:>3} + 判反 {e['wrong_flipped']:>3}"
            f"（漏放 {e['wrong_missed_violation']} / 误杀 {e['wrong_false_kill']}）"
        )

    print("\n[3] 决策迁移：baseline → agent（仅二值真值案）")
    transitions = {}
    for base in ("rule", "single_call_llm"):
        t = _transition(preds[base], preds["agent"], truth)
        transitions[base] = t
        print(
            f"  {base:<16} 修好 {t['fixed_by_agent']:>3}（其中从转人工修回 {t['fixed_from_abstain']}）"
            f" | 退步 {t['broke_by_agent']}"
            f" | 同错 {t['both_wrong']:>3} | 同对 {t['both_right']:>3}"
        )

    print("\n[4] Agent 剩余失败清单（严格口径）")
    failures = _agent_failures(preds["agent"], truth, scenes)
    by_reason: dict[str, int] = {}
    by_scene: dict[str, int] = {}
    for row in failures:
        by_reason[row["reason"]] = by_reason.get(row["reason"], 0) + 1
        by_scene[str(row["scene"])] = by_scene.get(str(row["scene"]), 0) + 1
    print(f"  合计 {len(failures)}/{result.total_cases}："
          + " / ".join(f"{k}={v}" for k, v in sorted(by_reason.items())))
    print("  按 scene：" + " / ".join(f"{k}={v}" for k, v in sorted(by_scene.items())))
    for row in failures[:20]:
        print(f"    {row['case_id']}  scene={row['scene']:<12} truth={row['truth']:<12} pred={row['pred']:<12} {row['reason']}")
    if len(failures) > 20:
        print(f"    …（其余 {len(failures) - 20} 条见 --json 产物）")

    payload = {
        "data": args.data,
        "total_cases": result.total_cases,
        "truth_distribution": {t: sum(1 for v in truth.values() if v == t) for t in LABELS},
        "confusion": matrices,
        "auto_errors": errors,
        "transitions_to_agent": transitions,
        "agent_failures": failures,
    }
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\n[JSON] 已写入 {path}")
    print("\n[OK] 错误归因完成（只读统计，未改判定 / 未改 metrics 口径）")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
