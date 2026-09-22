"""错误归因分析：只读某一次 ``run_evaluation_real.py --out`` 产物，回答「Agent 修好了什么、还错在哪」。

输入 = 真实跑分 JSON（``--from``）+ 该产物 ``data`` 指向的数据集真值。本脚本**不重跑 Agent、不调用
LLM、不使用任何桩** —— 分析对象就是那一次产物（real 臂非确定性、不可重放），不是新一次评测。

产出四块：三分类混淆矩阵（rule / real 两臂，含 HUMAN 真值列）、二值真值案错误构成（严格口径：
pred HUMAN_REVIEW 计错）、决策迁移 rule → real（修好 / 退步 / 同错 / 同对 + 从转人工修回 /
从判反修回）、real 臂剩余失败清单。``--json PATH`` 另落一份机器可读产物；只读统计，不改判定、
不写 DB。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pra.evaluation.dataset.loader import load_dataset

#: 两臂名（与 run_evaluation_real.py 的 payload 键一致）
ARMS: tuple[str, ...] = ("rule", "real")
#: 三分类标签（真值含 HUMAN_REVIEW = SHOULD_ABSTAIN）
LABELS: tuple[str, ...] = ("PASS", "REJECT", "HUMAN_REVIEW")
#: 二值真值（PASS/REJECT）—— 业务指标分母
AUTO_TRUTH: frozenset[str] = frozenset({"PASS", "REJECT"})


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="错误归因分析：只读 run_evaluation_real.py --out 产物（不重跑 Agent、不调用 LLM）"
    )
    parser.add_argument(
        "--from",
        dest="from_path",
        required=True,
        help="真实跑分产物 JSON（run_evaluation_real.py --out 的落盘文件）",
    )
    parser.add_argument("--json", default=None, help="把全部统计写入该 JSON 路径（默认只打印）")
    return parser.parse_args(argv)


def _load_payload(path: Path) -> dict:
    """读取 ``--from`` 产物 JSON；文件缺失或不是合法 JSON 即报错。"""
    if not path.is_file():
        raise ValueError(f"--from 产物不存在: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"--from 产物不是合法 JSON: {path}: {exc}") from exc


def _fill(preds: dict[str, str], truth: dict[str, str]) -> dict[str, dict[str, int]]:
    """pred（行）× truth（列）三分类计数矩阵。"""
    matrix = {p: {t: 0 for t in LABELS} for p in LABELS}
    for case_id, t in truth.items():
        matrix[preds[case_id]][t] += 1
    return matrix


def _auto_errors(preds: dict[str, str], truth: dict[str, str]) -> dict:
    """二值真值案上的错误构成（严格口径：pred HUMAN_REVIEW 计错）。"""
    wrong = {
        cid: (truth[cid], preds[cid])
        for cid in truth
        if truth[cid] in AUTO_TRUTH and preds[cid] != truth[cid]
    }
    return {
        "auto_truth_total": sum(1 for t in truth.values() if t in AUTO_TRUTH),
        "wrong_total": len(wrong),
        "wrong_abstained": sum(1 for t, p in wrong.values() if p == "HUMAN_REVIEW"),  # 过度转人工
        "wrong_flipped": sum(1 for t, p in wrong.values() if p in AUTO_TRUTH and p != t),  # 判反
        "wrong_missed_violation": sum(
            1 for t, p in wrong.values() if t == "REJECT" and p == "PASS"
        ),  # 漏放
        "wrong_false_kill": sum(1 for t, p in wrong.values() if t == "PASS" and p == "REJECT"),  # 误杀
    }


def _transition(
    rule_map: dict[str, str], real_map: dict[str, str], truth: dict[str, str]
) -> dict[str, int]:
    """二值真值案上 rule → real 的决策迁移四格 + 我方归因细分。"""
    out = {"both_right": 0, "fixed_by_agent": 0, "broke_by_agent": 0, "both_wrong": 0}
    detail = {
        "fixed_from_abstain": 0,
        "fixed_from_flip": 0,
        "broke_to_abstain": 0,
        "broke_to_flip": 0,
    }
    for cid, t in truth.items():
        if t not in AUTO_TRUTH:
            continue
        rule_right = rule_map[cid] == t
        real_right = real_map[cid] == t
        if rule_right and real_right:
            out["both_right"] += 1
        elif not rule_right and real_right:
            out["fixed_by_agent"] += 1
            detail["fixed_from_abstain" if rule_map[cid] == "HUMAN_REVIEW" else "fixed_from_flip"] += 1
        elif rule_right and not real_right:
            out["broke_by_agent"] += 1
            detail["broke_to_abstain" if real_map[cid] == "HUMAN_REVIEW" else "broke_to_flip"] += 1
        else:
            out["both_wrong"] += 1
    out.update(detail)
    return out


def _real_failures(
    real_map: dict[str, str], truth: dict[str, str], scenes: dict[str, str]
) -> list[dict]:
    """real 臂未命中真值的案件清单（严格口径，含 SHOULD_ABSTAIN 真值案）。"""
    rows: list[dict] = []
    for cid, t in truth.items():
        p = real_map[cid]
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
        rows.append(
            {"case_id": cid, "scene": scenes.get(cid), "truth": t, "pred": p, "reason": reason}
        )
    return rows


def _matrix_lines(name: str, matrix: dict[str, dict[str, int]]) -> list[str]:
    lines = [f"[{name}]   pred \\ truth" + "".join(f"{t:>14}" for t in LABELS)]
    for p in LABELS:
        lines.append(f"  {p:<14}" + "".join(f"{matrix[p][t]:>14}" for t in LABELS))
    return lines


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = _load_payload(Path(args.from_path))
    data_path = payload.get("data")
    if not data_path:
        raise ValueError("产物缺少 data 字段（数据集路径）—— 无法取真值，拒绝在无真值下出数")
    cases = load_dataset(data_path)
    rule_map = {d["eval_case_id"]: d["decision"] for d in payload.get("rule_decisions") or []}
    real_map = {r["eval_case_id"]: r["decision"] for r in payload.get("real_records") or []}
    # 数据契约防静默失配：产物案必须都能在本数据集里找到真值，且两臂覆盖同一批案
    unknown = (set(rule_map) | set(real_map)) - {c.eval_case_id for c in cases}
    if unknown:
        raise ValueError(
            f"产物含 {len(unknown)} 个不在数据集 {data_path} 中的案（数据契约不兼容，拒绝静默混用）: "
            f"{sorted(unknown)}"
        )
    if set(rule_map) != set(real_map):
        raise ValueError(
            "产物两臂覆盖的案不一致（rule - real = "
            f"{sorted(set(rule_map) - set(real_map))}；real - rule = "
            f"{sorted(set(real_map) - set(rule_map))}）"
        )

    covered = [c for c in cases if c.eval_case_id in real_map]
    truth = {c.eval_case_id: c.expected.decision for c in covered}
    scenes = {c.eval_case_id: c.scene for c in covered}
    total = len(covered)
    preds = {"rule": rule_map, "real": real_map}
    truth_dist = {t: sum(1 for v in truth.values() if v == t) for t in LABELS}

    print("=" * 96)
    print("错误归因分析（对象 = 某一次真实跑分产物：非确定性、不可重放）")
    print("只读该产物的 JSON 与数据集真值：不重跑 Agent、不调用 LLM —— 不是新一次评测")
    print(f"产物: {args.from_path}    数据集: {data_path}（覆盖 {total}/{len(cases)} 案）")
    print(f"模型: {payload.get('model')}    产物时间: {payload.get('timestamp_utc')}")
    print("真值分布: " + " / ".join(f"{t}={n}" for t, n in truth_dist.items()))
    print("=" * 96)

    matrices = {name: _fill(preds[name], truth) for name in ARMS}
    print("\n[1] 三分类混淆矩阵（行 = pred，列 = truth；对角线即命中）")
    for name in ARMS:
        for line in _matrix_lines(name, matrices[name]):
            print(line)
        hit = sum(matrices[name][t][t] for t in LABELS)
        print(f"  → 命中 {hit}/{total}；错列明细见上\n")

    print("[2] 二值真值案（PASS/REJECT）错误构成（严格口径：pred HUMAN_REVIEW 计错）")
    errors = {}
    for name in ARMS:
        e = _auto_errors(preds[name], truth)
        errors[name] = e
        print(
            f"  {name:<6} 错 {e['wrong_total']:>3}/{e['auto_truth_total']}"
            f" = 过度转人工 {e['wrong_abstained']:>3} + 判反 {e['wrong_flipped']:>3}"
            f"（漏放 {e['wrong_missed_violation']} / 误杀 {e['wrong_false_kill']}）"
        )

    print("\n[3] 决策迁移：rule → real（仅二值真值案）")
    transition = _transition(preds["rule"], preds["real"], truth)
    print(
        f"  rule → real  修好 {transition['fixed_by_agent']:>3}"
        f"（从转人工修回 {transition['fixed_from_abstain']} / 从判反修回 {transition['fixed_from_flip']}）"
        f" | 退步 {transition['broke_by_agent']}"
        f"（转人工 {transition['broke_to_abstain']} / 判反 {transition['broke_to_flip']}）"
        f" | 同错 {transition['both_wrong']:>3} | 同对 {transition['both_right']:>3}"
    )

    print("\n[4] real 臂剩余失败清单（严格口径，含 SHOULD_ABSTAIN 真值案）")
    failures = _real_failures(preds["real"], truth, scenes)
    by_reason: dict[str, int] = {}
    by_scene: dict[str, int] = {}
    for row in failures:
        by_reason[row["reason"]] = by_reason.get(row["reason"], 0) + 1
        by_scene[str(row["scene"])] = by_scene.get(str(row["scene"]), 0) + 1
    print(
        f"  合计 {len(failures)}/{total}："
        + " / ".join(f"{k}={v}" for k, v in sorted(by_reason.items()))
    )
    print("  按 scene：" + " / ".join(f"{k}={v}" for k, v in sorted(by_scene.items())))
    for row in failures[:20]:
        print(
            f"    {row['case_id']}  scene={row['scene']:<12} truth={row['truth']:<12} "
            f"pred={row['pred']:<12} {row['reason']}"
        )
    if len(failures) > 20:
        print(f"    …（其余 {len(failures) - 20} 条见 --json 产物）")

    out = {
        "from": str(args.from_path),
        "data": data_path,
        "model": payload.get("model"),
        "timestamp_utc": payload.get("timestamp_utc"),
        "total_cases": total,
        "truth_distribution": truth_dist,
        "confusion": matrices,
        "auto_errors": errors,
        "transition_rule_to_real": transition,
        "real_failures": failures,
        "note": "只读某一次真实跑分产物；real 臂非确定性、不可重放",
    }
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\n[JSON] 已写入 {path}")
    print("\n[OK] 错误归因完成（只读产物与真值；未重跑 Agent / 未改判定）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
