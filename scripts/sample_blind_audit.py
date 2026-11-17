"""从 v3 合成评测集盲抽 30 条人工审计样本（不含构造参数与真值）。

产物（默认 ``eval_data/v3/audit/``）：

- ``blind_30.csv``：待填标注表（快照 + 工具事实 + 人工判定空列）；
- ``blind_30.md``：人读标注单；
- ``blind_30.sampling.json``：抽样口径与覆盖自检（仅聚合，不含逐案真值映射）。

分层口径 = 10 个可观测特征向量（见 :func:`feature_stratum`），每层固定 3 条；
层内按构造模式轮转、模式内优先抽边界（见 ``_BOUNDARY_RANK``），
使 30 条覆盖全部构造模式而不是堆在某一种模板上。

标注者只打开 csv / md；不要打开 ``cases_v3.jsonl``、``truth_v3.jsonl``、
``result_latest.json`` —— 后三者含构造参数与真值，看了就不叫盲标。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = REPO_ROOT / "eval_data" / "v3" / "cases_v3.jsonl"
DEFAULT_TRUTH = REPO_ROOT / "eval_data" / "v3" / "truth_v3.jsonl"
DEFAULT_RULEBOOK = REPO_ROOT / "scripts" / "evaluation" / "labeling" / "rulebook.json"
DEFAULT_OUT_DIR = REPO_ROOT / "eval_data" / "v3" / "audit"
DEFAULT_SEED = 20261116

# 画像“脏”的阈值：移除数或改标题重上架数达到 3 才计入风险信号；
# 生成器给 clean/边界案埋的 0~2 是噪声，不应把真值推向弃权。
MATERIAL_DIRT = 3

# 十个可观测特征向量：clean 沉淀在最后，便于人工核对零散信号。
STRATA: tuple[str, ...] = (
    "blacklist+evasion",
    "blacklist",
    "evasion+dirty",
    "evasion clean",
    "brand+dirty",
    "brand only",
    "gap",
    "mismatch",
    "dirty only",
    "clean",
)
PER_STRATUM = 3

# 同层内优先抽这些构造模式：0 = 阈值/双信号边界，3 = 最普通的干净模板。
_BOUNDARY_RANK = {
    "merchant_count_two": 0,
    "merchant_count_three": 0,
    "blacklist_plus_evasion": 0,
    "hidden_evasion_in_description": 1,
    "brand_word_plus_dirty_merchant": 1,
    "strong_evasion": 1,
    "evasion_word_clean_merchant": 1,
    "brand_word_only": 2,
    "evasion_word_dirty_merchant": 2,
    "snapshot_gap": 2,
    "registry_mismatch": 2,
    "merchant_dirty_only": 2,
    "blacklist_brand": 2,
    "clean": 3,
}

_OUTPUT_STEM = "blind_30"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _read_exclude(path: Path | None) -> frozenset[str]:
    if path is None:
        return frozenset()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            ids.add(text)
    return frozenset(ids)


def _term_sets(rulebook: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    blacklist = {str(item) for item in rulebook["blacklisted_brands"]}
    evasion = {str(item) for item in rulebook["evasion_terms"]}
    brand = {str(item) for item in rulebook["brand_terms"]}
    return blacklist, evasion, brand


def feature_stratum(manifest: dict[str, Any], rulebook: dict[str, Any]) -> str:
    """按可观测事实把案件归入 10 个特征向量之一（顺序即判定优先级）。"""
    blacklist, evasion, brand = _term_sets(rulebook)
    hit_terms = [str(hit["term"]) for hit in manifest["text_hits"]]
    has_evasion = any(term in evasion for term in hit_terms)
    has_brand = not has_evasion and any(term in brand for term in hit_terms)
    brand_blacklisted = (
        manifest["snapshot_brand"] is not None
        and manifest["snapshot_brand"] in blacklist
    )
    merchant = manifest["merchant"]
    dirty = (
        merchant["removals"] >= MATERIAL_DIRT
        or merchant["title_relisting_count"] >= MATERIAL_DIRT
    )
    if brand_blacklisted and has_evasion:
        return "blacklist+evasion"
    if brand_blacklisted:
        return "blacklist"
    if has_evasion and dirty:
        return "evasion+dirty"
    if has_evasion:
        return "evasion clean"
    if has_brand and dirty:
        return "brand+dirty"
    if has_brand:
        return "brand only"
    if manifest["snapshot_gaps"]:
        return "gap"
    if manifest["registry_mismatch"]:
        return "mismatch"
    if dirty:
        return "dirty only"
    return "clean"


def truth_key(expected: dict[str, Any]) -> tuple[Any, ...]:
    """真值输出分组：决策 × 弃权标签 × 风险类型 × 条款 × 工具。"""
    return (
        expected["decision"],
        expected["abstain_label"],
        tuple(sorted(expected["risk_type"])),
        tuple(sorted(expected["expected_policies"])),
        tuple(sorted(expected["expected_tools"])),
    )


def _boundary_rank(pattern: str) -> int:
    if pattern not in _BOUNDARY_RANK:
        raise ValueError(f"未知构造模式 {pattern!r}：请先更新 _BOUNDARY_RANK")
    return _BOUNDARY_RANK[pattern]


def _pick_from_pool(
    pool: list[dict[str, Any]], size: int, rng: random.Random
) -> list[dict[str, Any]]:
    """层内按构造模式轮转抽取：先保证模式多样性，再按边界优先级补位。"""
    by_pattern: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in sorted(pool, key=lambda item: item["eval_case_id"]):
        by_pattern[case["construction"]["pattern"]].append(case)
    for group in by_pattern.values():
        rng.shuffle(group)

    patterns = sorted(by_pattern, key=lambda name: (_boundary_rank(name), name))
    picked: list[dict[str, Any]] = []
    while len(picked) < size:
        advanced = False
        for name in patterns:
            if len(picked) >= size:
                break
            group = by_pattern[name]
            if group:
                picked.append(group.pop())
                advanced = True
        if not advanced:
            break
    return picked


def _audit_row(
    index: int, case: dict[str, Any], truth: dict[str, Any]
) -> dict[str, Any]:
    product = case["input"]["product"]
    manifest = truth["manifest"]
    merchant = manifest["merchant"]
    registry = manifest["registry"]
    ocr_texts = [
        f"{image.get('source') or '图'}:{image['ocr_text']}"
        for image in product.get("images", [])
        if image.get("ocr_text")
    ]
    prices = [float(sku["price"]) for sku in product.get("sku_list", [])]
    return {
        "audit_id": f"A{index:02d}",
        "eval_case_id": case["eval_case_id"],
        "case_id": case["input"]["case_id"],
        "product_id": product["product_id"],
        "event_type": case["input"]["event_type"],
        "title": product["title"],
        "description": product["description"],
        "category": product["category"],
        "brand": product["brand"],
        "attributes": json.dumps(
            product.get("attributes", {}), ensure_ascii=False, sort_keys=True
        ),
        "price": min(prices) if prices else "",
        "listing_time": product["listing_time"],
        "image_count": len(product.get("images", [])),
        "image_ocr": " | ".join(ocr_texts),
        "merchant_id": case["input"]["merchant_id"],
        "merchant_removals": merchant["removals"],
        "merchant_title_relisting_count": merchant["title_relisting_count"],
        "merchant_similar_product_count": merchant["similar_product_count"],
        "merchant_credit_score": merchant["credit_score"],
        "registry_brand": registry["brand"] or "",
        "registry_category": registry["category"],
        "registry_version": registry["version"],
        "registry_status": registry["status"],
        "human_decision": "",
        "human_risk_type": "",
        "human_policy_ref": "",
        "human_reason": "",
    }


def select_rows(
    cases: list[dict[str, Any]],
    truths: list[dict[str, Any]],
    rulebook: dict[str, Any],
    *,
    seed: int,
    exclude: frozenset[str] = frozenset(),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """每层抽 ``PER_STRATUM`` 条，层内优先边界模式；返回 (选中案件, 标注行)。"""
    truth_by_id = {truth["eval_case_id"]: truth for truth in truths}
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        case_id = case["eval_case_id"]
        if case_id in exclude or case_id not in truth_by_id:
            continue
        manifest = truth_by_id[case_id]["manifest"]
        pools[feature_stratum(manifest, rulebook)].append(case)

    short = [name for name in STRATA if len(pools[name]) < PER_STRATUM]
    if short:
        raise ValueError(f"分层样本不足（每层需 {PER_STRATUM} 条，排除后）：{short}")

    rng = random.Random(seed)
    chosen: list[dict[str, Any]] = []
    for name in STRATA:
        chosen.extend(_pick_from_pool(pools[name], PER_STRATUM, rng))

    rows = [
        _audit_row(index + 1, case, truth_by_id[case["eval_case_id"]])
        for index, case in enumerate(chosen)
    ]
    return chosen, rows


def build_report(
    chosen: list[dict[str, Any]],
    truth_by_id: dict[str, dict[str, Any]],
    rulebook: dict[str, Any],
    *,
    seed: int,
    excluded_count: int,
) -> dict[str, Any]:
    strata = Counter(
        feature_stratum(truth_by_id[case["eval_case_id"]]["manifest"], rulebook)
        for case in chosen
    )
    key_counts = Counter(
        truth_key(truth_by_id[case["eval_case_id"]]["expected"]) for case in chosen
    )
    pattern_counts = Counter(case["construction"]["pattern"] for case in chosen)
    return {
        "dataset": "v3",
        "seed": seed,
        "sample_size": len(chosen),
        "per_stratum": PER_STRATUM,
        "strata": {name: strata[name] for name in STRATA},
        "pattern_coverage": len(pattern_counts),
        "patterns": dict(sorted(pattern_counts.items())),
        "truth_key_groups": len(key_counts),
        "truth_key_min_per_group": min(key_counts.values()),
        "truth_key_definition": "decision + abstain_label + risk_type + policies + tools",
        "excluded_by_user": excluded_count,
        "source": {
            "cases": DEFAULT_CASES.relative_to(REPO_ROOT).as_posix(),
            "truth": DEFAULT_TRUTH.relative_to(REPO_ROOT).as_posix(),
            "rulebook": DEFAULT_RULEBOOK.relative_to(REPO_ROOT).as_posix(),
        },
        "note": "仅聚合统计，不含逐案真值映射",
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# 盲标审计单 · v3（30 条）",
        "",
        "> 只依据本单事实与冻结规则书判定。不要打开 `cases_v3.jsonl`、",
        "> `truth_v3.jsonl`、`result_latest.json` —— 后三者含构造参数与真值。",
        "",
        "判定口径：`PASS` / `REJECT` / `HUMAN_REVIEW`；`REJECT` 须写可引用条款。",
        "",
    ]
    for row in rows:
        lines += [
            f"## {row['audit_id']} · {row['case_id']}",
            "",
            f"- 标题：{row['title']}",
            f"- 描述：{row['description']}",
            (
                f"- 类目 / 品牌：{row['category'] or '（空）'} / "
                f"{row['brand'] or '（空）'}"
            ),
            f"- 属性：{row['attributes']}",
            (
                f"- 价格：{row['price']}；上架时间：{row['listing_time']}；"
                f"事件：{row['event_type']}"
            ),
            f"- 图片 {row['image_count']} 张；OCR：{row['image_ocr'] or '（无）'}",
            (
                f"- 商家：{row['merchant_id']}；历史移除 "
                f"{row['merchant_removals']}；改标题重上架 "
                f"{row['merchant_title_relisting_count']}；相似商品 "
                f"{row['merchant_similar_product_count']}；信用 "
                f"{row['merchant_credit_score']}"
            ),
            (
                f"- 注册信息：品牌 {row['registry_brand'] or '（空）'}；类目 "
                f"{row['registry_category']}；版本 {row['registry_version']}；"
                f"状态 {row['registry_status']}"
            ),
            "",
            "**人工判定（PASS / REJECT / HUMAN_REVIEW）**：",
            "",
            "**风险类型**：",
            "",
            "**依据条款**：",
            "",
            "**理由**：",
            "",
            "---",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    rows: list[dict[str, Any]], report: dict[str, Any], out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, out_dir / f"{_OUTPUT_STEM}.csv")
    _write_markdown(rows, out_dir / f"{_OUTPUT_STEM}.md")
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    (out_dir / f"{_OUTPUT_STEM}.sampling.json").write_text(payload, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="盲抽 v3 人工审计样本")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--exclude-file", type=Path, default=None)
    args = parser.parse_args(argv)

    cases = _read_jsonl(DEFAULT_CASES)
    truths = _read_jsonl(DEFAULT_TRUTH)
    rulebook = json.loads(DEFAULT_RULEBOOK.read_text(encoding="utf-8"))
    exclude = _read_exclude(args.exclude_file)

    chosen, rows = select_rows(cases, truths, rulebook, seed=args.seed, exclude=exclude)
    truth_by_id = {truth["eval_case_id"]: truth for truth in truths}
    report = build_report(
        chosen,
        truth_by_id,
        rulebook,
        seed=args.seed,
        excluded_count=len(exclude),
    )
    write_outputs(rows, report, args.out_dir)

    print(f"抽样 {len(rows)} 条 = {len(STRATA)} 层 × {PER_STRATUM}")
    print(
        f"构造模式覆盖 {report['pattern_coverage']} 种；"
        f"真值分组覆盖 {report['truth_key_groups']} 组，"
        f"每组最少 {report['truth_key_min_per_group']} 条"
    )
    print(f"输出目录：{args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
