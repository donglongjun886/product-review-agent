"""比对盲标审计答案与冻结真值，产出审计报告。

输入默认是填好的 ``eval_data/v3/audit/blind_30.xlsx``（也接受同结构的 CSV），
真值来自 ``eval_data/v3/truth_v3.jsonl``。只比对判定 / 风险类型 / 依据条款；
``expected_tools`` 与 ``expected_evidence`` 未在标注表收集，不参与比对。

运行（xlsx 输入需要 openpyxl，不在项目依赖里）：

    uv run --no-project --with openpyxl python scripts/check_blind_audit.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sample_blind_audit import STRATA, feature_stratum, truth_key

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = REPO_ROOT / "eval_data" / "v3" / "audit"
DEFAULT_INPUT = AUDIT_DIR / "blind_30.xlsx"
TRUTH_PATH = REPO_ROOT / "eval_data" / "v3" / "truth_v3.jsonl"
RULEBOOK_PATH = REPO_ROOT / "scripts" / "evaluation" / "labeling" / "rulebook.json"
RESULT_JSON = "audit_result.json"
RESULT_MD = "audit_result.md"

_EMPTY_VALUES = {"", "无"}
_XLSX_COLUMNS = {
    "eval_case_id": "案件ID",
    "decision": "判定",
    "risk_type": "风险类型",
    "policies": "依据条款",
    "reason": "理由",
}
_CSV_COLUMNS = {
    "eval_case_id": "eval_case_id",
    "decision": "human_decision",
    "risk_type": "human_risk_type",
    "policies": "human_policy_ref",
    "reason": "human_reason",
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _split(value: Any) -> list[str]:
    """把「无」「A+B」「A」统一成列表。"""
    if value is None:
        return []
    text = str(value).strip()
    if text in _EMPTY_VALUES:
        return []
    return [item.strip() for item in text.split("+") if item.strip()]


def _load_xlsx(path: Path) -> list[dict[str, Any]]:
    from openpyxl import load_workbook

    workbook = load_workbook(path, data_only=True)
    sheet = workbook["标注"]
    headers = {
        sheet.cell(row=1, column=column).value: column
        for column in range(1, sheet.max_column + 1)
    }
    missing = [name for name in _XLSX_COLUMNS.values() if name not in headers]
    if missing:
        raise SystemExit(f"xlsx 缺少列：{missing}")

    answers: list[dict[str, Any]] = []
    for row in range(2, sheet.max_row + 1):
        eval_case_id = sheet.cell(
            row=row, column=headers[_XLSX_COLUMNS["eval_case_id"]]
        ).value
        if not eval_case_id:
            continue
        answers.append(
            {
                key: sheet.cell(row=row, column=headers[column]).value
                for key, column in _XLSX_COLUMNS.items()
            }
        )
    return answers


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    missing = [name for name in _CSV_COLUMNS.values() if name not in rows[0]]
    if missing:
        raise SystemExit(f"CSV 缺少列：{missing}")
    return [{key: row[column] for key, column in _CSV_COLUMNS.items()} for row in rows]


def load_answers(path: Path) -> list[dict[str, Any]]:
    """按后缀读取标注答案（xlsx 或 csv）。"""
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return _load_xlsx(path)
    if suffix == ".csv":
        return _load_csv(path)
    raise SystemExit(f"不支持的输入格式：{path}")


def upper_bound(errors: int, total: int, alpha: float = 0.05) -> float:
    """Clopper-Pearson 单侧上界：解 P(X ≤ errors; p) = alpha。"""
    if errors >= total:
        return 1.0

    def cdf(p: float) -> float:
        return sum(
            math.comb(total, index) * p**index * (1 - p) ** (total - index)
            for index in range(errors + 1)
        )

    low, high = 0.0, 1.0
    for _ in range(80):
        middle = (low + high) / 2
        if cdf(middle) > alpha:
            low = middle
        else:
            high = middle
    return high


def _normalize(answer: dict[str, Any]) -> tuple[str, set[str], set[str]]:
    decision = str(answer["decision"] or "").strip()
    if not decision:
        raise SystemExit(f"案件 {answer['eval_case_id']} 未填判定")
    return decision, set(_split(answer["risk_type"])), set(_split(answer["policies"]))


def _expected(expected: dict[str, Any]) -> tuple[str, set[str], set[str]]:
    return (
        str(expected["decision"]),
        set(expected["risk_type"]),
        set(expected["expected_policies"]),
    )


def compare(
    answers: list[dict[str, Any]],
    truths: list[dict[str, Any]],
    rulebook: dict[str, Any],
) -> dict[str, Any]:
    truth_by_id = {truth["eval_case_id"]: truth for truth in truths}
    unknown = [
        answer["eval_case_id"]
        for answer in answers
        if answer["eval_case_id"] not in truth_by_id
    ]
    if unknown:
        raise SystemExit(f"真值里找不到这些案件：{unknown}")

    agreements = Counter()
    by_stratum: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    disagreements: list[dict[str, Any]] = []
    truth_keys: set[tuple[Any, ...]] = set()

    for answer in answers:
        truth = truth_by_id[answer["eval_case_id"]]
        manifest = truth["manifest"]
        expected = truth["expected"]
        stratum = feature_stratum(manifest, rulebook)
        truth_keys.add(truth_key(expected))

        human = _normalize(answer)
        target = _expected(expected)
        delta = {
            "decision": human[0] == target[0],
            "risk_type": human[1] == target[1],
            "policies": human[2] == target[2],
        }
        agrees = all(delta.values())
        for name, hit in delta.items():
            agreements[name] += int(hit)
        agreements["all"] += int(agrees)

        by_stratum[stratum][0] += 1
        by_stratum[stratum][1] += int(agrees)
        if not agrees:
            disagreements.append(
                {
                    "eval_case_id": answer["eval_case_id"],
                    "stratum": stratum,
                    "human": {
                        "decision": human[0],
                        "risk_type": sorted(human[1]),
                        "policies": sorted(human[2]),
                    },
                    "truth": {
                        "decision": target[0],
                        "risk_type": sorted(target[1]),
                        "policies": sorted(target[2]),
                    },
                    "reason": str(answer["reason"] or "").strip(),
                    "fields": [name for name, hit in delta.items() if not hit],
                }
            )

    total = len(answers)
    errors = total - agreements["all"]
    return {
        "sample_size": total,
        "agreement": {
            "decision": agreements["decision"],
            "risk_type": agreements["risk_type"],
            "policies": agreements["policies"],
            "all": agreements["all"],
        },
        "errors": errors,
        "error_rate": errors / total if total else 0.0,
        "error_rate_upper_95": upper_bound(errors, total),
        "by_stratum": [
            {
                "stratum": name,
                "n": by_stratum[name][0],
                "agree": by_stratum[name][1],
            }
            for name in STRATA
            if name in by_stratum
        ],
        "truth_key_groups": len(truth_keys),
        "disagreements": disagreements,
        "note": "只比对判定/风险类型/依据条款；expected_tools/evidence 未收集",
    }


def _render_markdown(report: dict[str, Any]) -> str:
    agreement = report["agreement"]
    total = report["sample_size"]
    lines = [
        "# 盲标审计结果 · v3",
        "",
        (
            f"样本 {total} 条：**{agreement['all']}/{total} 全维度一致**"
            f"（判定 {agreement['decision']}/{total}、风险类型 "
            f"{agreement['risk_type']}/{total}、依据条款 "
            f"{agreement['policies']}/{total}）。"
        ),
        "",
        (
            f"标签错误率点估计 {report['error_rate']:.1%}，"
            f"**95% 上界 {report['error_rate_upper_95']:.1%}**"
            f"（Clopper-Pearson 单侧）。"
        ),
        "",
        "## 分层一致率",
        "",
        "| 特征向量 | 条数 | 一致 |",
        "|---|---|---|",
    ]
    for item in report["by_stratum"]:
        lines.append(
            f"| {item['stratum']} | {item['n']} | {item['agree']}/{item['n']} |"
        )
    lines += ["", f"真值输出分组覆盖：{report['truth_key_groups']} 组。", ""]

    lines += ["## 分歧清单", ""]
    if report["disagreements"]:
        for item in report["disagreements"]:
            lines += [
                f"### {item['eval_case_id']}（{item['stratum']}）",
                "",
                f"- 分歧字段：{', '.join(item['fields'])}",
                f"- 标注：{item['human']}",
                f"- 真值：{item['truth']}",
                f"- 标注理由：{item['reason']}",
                "",
            ]
    else:
        lines += ["无。", ""]

    lines += [
        "## 口径与局限",
        "",
        (
            "- 只比对判定 / 风险类型 / 依据条款；`expected_tools` / "
            "`expected_evidence` 未在标注表收集，不参与比对。"
        ),
        (
            "- 标注者与 labeler 共用同一份冻结规则书："
            "测的是“实现与规则书是否一致”，测不了“规则书本身是否合理”。"
        ),
        f"- n={total} 只能给错误率的量级上界，不能支撑分层准确率或显著性结论。",
        "- 与 320 条主指标分开报，不混算；也不得用这 30 条回头调阈值或改生成器。",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="比对盲标审计答案与真值")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=AUDIT_DIR)
    args = parser.parse_args(argv)

    answers = load_answers(args.input)
    truths = _read_jsonl(TRUTH_PATH)
    rulebook = json.loads(RULEBOOK_PATH.read_text(encoding="utf-8"))
    report = compare(answers, truths, rulebook)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    (args.out_dir / RESULT_JSON).write_text(payload, encoding="utf-8")
    (args.out_dir / RESULT_MD).write_text(_render_markdown(report), encoding="utf-8")

    print(
        f"全维度一致 {report['agreement']['all']}/{report['sample_size']}；"
        f"错误率上界 {report['error_rate_upper_95']:.1%}"
    )
    print(f"输出：{args.out_dir / RESULT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
