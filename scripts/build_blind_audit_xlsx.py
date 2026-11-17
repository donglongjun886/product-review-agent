"""把盲标审计 CSV 渲染成带下拉选项的 xlsx（判定 / 风险类型 / 依据条款可直接勾选）。

以 ``eval_data/v3/audit/blind_30.csv`` 为唯一数据源；信息列原样搬运，
只在右侧追加四个空白的标注列，其中前三个带数据验证下拉。

运行（openpyxl 不在项目依赖里，按需临时装）：

    uv run --no-project --with openpyxl python scripts/build_blind_audit_xlsx.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = REPO_ROOT / "eval_data" / "v3" / "audit"
CSV_PATH = AUDIT_DIR / "blind_30.csv"
RULEBOOK_PATH = REPO_ROOT / "scripts" / "evaluation" / "labeling" / "rulebook.json"
XLSX_PATH = AUDIT_DIR / "blind_30.xlsx"

DECISION_OPTIONS = ("PASS", "REJECT", "HUMAN_REVIEW")
RISK_OPTIONS = (
    "无",
    "POTENTIAL_IP_RISK",
    "EVASION_PATTERN",
    "POTENTIAL_IP_RISK+EVASION_PATTERN",
)
POLICY_OPTIONS = (
    "无",
    "POLICY_1.1",
    "POLICY_1.2",
    "POLICY_4.5",
    "POLICY_1.2+POLICY_4.5",
)

_INFO_COLUMNS: tuple[tuple[str, str, int], ...] = (
    ("编号", "audit_id", 8),
    ("案件ID", "eval_case_id", 16),
    ("标题", "title", 28),
    ("描述", "description", 34),
    ("类目", "category", 14),
    ("品牌", "brand", 12),
    ("价格", "price", 8),
    ("上架时间", "listing_time", 18),
    ("事件", "event_type", 12),
    ("图片数", "image_count", 8),
    ("图片OCR", "image_ocr", 20),
    ("商家", "merchant_id", 13),
    ("历史移除", "merchant_removals", 9),
    ("改标题重上架", "merchant_title_relisting_count", 13),
    ("相似商品", "merchant_similar_product_count", 11),
    ("信用分", "merchant_credit_score", 8),
    ("注册品牌", "registry_brand", 12),
    ("注册类目", "registry_category", 14),
    ("注册版本", "registry_version", 9),
    ("注册状态", "registry_status", 10),
)
_ANSWER_COLUMNS: tuple[tuple[str, str, int], ...] = (
    ("判定", "human_decision", 14),
    ("风险类型", "human_risk_type", 30),
    ("依据条款", "human_policy_ref", 24),
    ("理由", "human_reason", 46),
)

_HEADER_FILL = PatternFill("solid", fgColor="1F4E79")
_ANSWER_HEADER_FILL = PatternFill("solid", fgColor="385723")
_ANSWER_FILL = PatternFill("solid", fgColor="FFF2CC")
_HEADER_FONT = Font(bold=True, color="FFFFFF")
_WRAP = Alignment(vertical="top", wrap_text=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _cell_value(value: str) -> str | int | float:
    """能转数字的写成数字，其余保持文本（价格/计数用数字便于排序）。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return int(number) if number.is_integer() else number


def _add_dropdown(
    ws, column: str, last_row: int, options: tuple[str, ...], prompt: str
) -> None:
    validation = DataValidation(
        type="list",
        formula1='"' + ",".join(options) + '"',
        allow_blank=True,
    )
    validation.promptTitle = "选择一项"
    validation.prompt = prompt
    validation.showInputMessage = True
    validation.showErrorMessage = True
    validation.error = "请从下拉选项中选择；需要其他组合时写在「理由」列。"
    ws.add_data_validation(validation)
    validation.add(f"{column}2:{column}{last_row}")


def _style_header(ws, headers: list[str], answer_start: int) -> None:
    for index, header in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=index, value=header)
        cell.font = _HEADER_FONT
        cell.fill = _ANSWER_HEADER_FILL if index >= answer_start else _HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 24


def _build_sheet(ws, rows: list[dict[str, str]]) -> None:
    info_headers = [header for header, _, _ in _INFO_COLUMNS]
    answer_headers = [header for header, _, _ in _ANSWER_COLUMNS]
    headers = info_headers + answer_headers
    _style_header(ws, headers, answer_start=len(info_headers) + 1)

    for offset, row in enumerate(rows, start=2):
        for index, (_, key, _) in enumerate(_INFO_COLUMNS, start=1):
            cell = ws.cell(row=offset, column=index, value=_cell_value(row[key]))
            cell.alignment = _WRAP
        for index, (_, key, _) in enumerate(
            _ANSWER_COLUMNS, start=len(info_headers) + 1
        ):
            if key in {"human_decision", "human_risk_type", "human_policy_ref"}:
                cell = ws.cell(row=offset, column=index, value=None)
            else:
                cell = ws.cell(row=offset, column=index, value="")
            cell.fill = _ANSWER_FILL
            cell.alignment = _WRAP

    for index, (_, _, width) in enumerate(_INFO_COLUMNS + _ANSWER_COLUMNS, start=1):
        ws.column_dimensions[ws.cell(row=1, column=index).column_letter].width = width
    ws.freeze_panes = "D2"
    ws.auto_filter.ref = (
        f"A1:{ws.cell(row=1, column=len(headers)).column_letter}{len(rows) + 1}"
    )

    last_row = len(rows) + 1
    decision_col = ws.cell(row=1, column=len(info_headers) + 1).column_letter
    risk_col = ws.cell(row=1, column=len(info_headers) + 2).column_letter
    policy_col = ws.cell(row=1, column=len(info_headers) + 3).column_letter
    _add_dropdown(
        ws, decision_col, last_row, DECISION_OPTIONS, "PASS / REJECT / HUMAN_REVIEW"
    )
    _add_dropdown(ws, risk_col, last_row, RISK_OPTIONS, "可多选时写成 A+B 形式")
    _add_dropdown(ws, policy_col, last_row, POLICY_OPTIONS, "可多选时写成 A+B 形式")


def _build_guide_sheet(ws) -> None:
    lines = [
        "盲标审计怎么填",
        (
            "1. 只依据「标注」表的事实 + 规则书判定；不要打开 cases_v3.jsonl / "
            "truth_v3.jsonl / result_latest.json。"
        ),
        "2. 「判定」下拉三选一：PASS / REJECT / HUMAN_REVIEW。",
        "3. 「风险类型」「依据条款」按需下拉；需要其他组合时写在「理由」列。",
        "4. REJECT 必须能引用条款；理由写一句话即可。",
        "",
        "报告口径",
        "30 条全对：写「抽检 0 错，95% 上界约 9.5%」，不要写「准确率 100%」。",
        "出现分歧：写清是规则书问题还是标注问题，并单独列出分歧条目。",
        "这 30 条只用于审计标签，不得拿来调阈值或改生成器。",
    ]
    for index, text in enumerate(lines, start=1):
        cell = ws.cell(row=index, column=1, value=text)
        if index == 1:
            cell.font = Font(bold=True, size=14)
    ws.column_dimensions["A"].width = 96


def _build_rulebook_sheet(ws, rulebook: dict[str, Any]) -> None:
    rows: list[tuple[str, str]] = [
        ("规则书版本", str(rulebook["rulebook_version"])),
        ("黑名单品牌", "、".join(str(b) for b in rulebook["blacklisted_brands"])),
        ("规避词", "、".join(str(t) for t in rulebook["evasion_terms"])),
        ("品牌词", "、".join(str(t) for t in rulebook["brand_terms"])),
        (
            "商家画像脏阈值",
            f"移除数或改标题重上架数 ≥ {rulebook['merchant_dirty_min']}",
        ),
        (
            "判定口径",
            "PASS = 放行；REJECT = 自动拒绝（须可引用依据）；HUMAN_REVIEW = 转人工",
        ),
    ]
    for mapping, refs in rulebook["policy_refs"].items():
        rows.append((f"条款映射 · {mapping}", " + ".join(str(ref) for ref in refs)))

    cell = ws.cell(row=1, column=1, value="规则书")
    cell.font = Font(bold=True, size=14)
    for index, (label, value) in enumerate(rows, start=3):
        ws.cell(row=index, column=1, value=label).font = Font(bold=True)
        ws.cell(row=index, column=2, value=value)
    ws.column_dimensions["A"].width = 24
    ws.column_dimensions["B"].width = 80
    ws.cell(
        row=len(rows) + 4,
        column=1,
        value="以 scripts/evaluation/labeling/rulebook.json 为准",
    )


def build_workbook(rows: list[dict[str, str]], rulebook: dict[str, Any]) -> Workbook:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "标注"
    _build_sheet(sheet, rows)

    guide = workbook.create_sheet("说明")
    _build_guide_sheet(guide)

    rules = workbook.create_sheet("规则书")
    _build_rulebook_sheet(rules, rulebook)
    return workbook


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="渲染盲标审计 xlsx")
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--out", type=Path, default=XLSX_PATH)
    args = parser.parse_args(argv)

    rows = _read_csv(args.csv)
    if not rows:
        raise SystemExit(f"CSV 为空：{args.csv}")
    rulebook = json.loads(RULEBOOK_PATH.read_text(encoding="utf-8"))

    workbook = build_workbook(rows, rulebook)
    workbook.save(args.out)
    print(f"已生成 {args.out}（{len(rows)} 行，3 张表）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
