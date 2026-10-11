"""确定性生成 v3 评测数据集（构造即标签）并自检。

一次运行产出 eval_data/v3/ 四件产物：cases_v3.jsonl / truth_v3.jsonl / seed_v3.sql /
manifest.json。退出码 0 = 生成成功且自检通过，非零 = 自检失败。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from evaluation.generator.builder import (
    DEFAULT_SEED,
    POLICIES_PATH,
    TOTAL,
    BuiltDataset,
    build_dataset,
)
from evaluation.labeling.labeler import label_case, load_rulebook, scan_text_hits

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "eval_data" / "v3"

_SCENE_PCT = {"normal": 0.20, "violation": 0.20, "boundary": 0.30,
              "multi-signal": 0.20, "evasion": 0.10}
_URL_LEAKED_TOKENS = ("viol", "clean", "bound", "logo", "brand", "risk", "reject", "pass", "human")


def self_check(built: BuiltDataset) -> list[str]:
    """数据集自检：规模分布 / 同输入同真值 / 文本自证 / 政策可得 / URL 中性。"""
    errors: list[str] = []
    rulebook = load_rulebook()
    cases, truths = built.cases, built.truths

    if len(cases) != TOTAL:
        errors.append(f"案数 {len(cases)} != {TOTAL}")
    by_scene = Counter(c.scene for c in cases)
    for scene, pct in _SCENE_PCT.items():
        if abs(by_scene[scene] - pct * TOTAL) > 0.05 * TOTAL:
            errors.append(f"scene={scene} 占比 {by_scene[scene]}/{TOTAL} 超出 ±5pp")

    effective_policies = {
        p["policy_id"]
        for p in json.loads(POLICIES_PATH.read_text(encoding="utf-8"))["policies"]
        if p["status"] == "EFFECTIVE"
    }
    for case, truth in zip(cases, truths, strict=True):
        cid = case.eval_case_id
        if cid != truth.eval_case_id:
            errors.append(f"{cid} 案件与真值行错位（truth={truth.eval_case_id}）")
            continue
        # 同输入必得同真值：对埋点清单二次打标必须复现已落真值。
        if label_case(truth.manifest, rulebook) != truth.expected:
            errors.append(f"{cid} 二次打标与已落真值不一致")
        # 文本自证：词命中重扫与埋点清单一致。
        scanned = scan_text_hits(case.input.product.title, case.input.product.description, rulebook)
        if scanned != truth.manifest.text_hits:
            errors.append(f"{cid} 文本重扫与埋点清单不一致")
        if truth.manifest.snapshot_brand != case.input.product.brand:
            errors.append(f"{cid} 快照品牌与埋点清单不一致")
        # 政策引用可得性：语料固定，应引条款必须生效可得。
        missing = set(truth.expected.expected_policies) - effective_policies
        if missing:
            errors.append(f"{cid} 应引政策不可得或已失效: {sorted(missing)}")
        for img in case.input.product.images:
            name = img.url.rsplit("/", 1)[-1].lower()
            segment = img.url.rsplit("/", 2)[-2].lower()
            if not segment.startswith(("asset-", "p_")) or any(
                tok in segment or tok in name for tok in _URL_LEAKED_TOKENS
            ):
                errors.append(f"{cid} 图片 URL 含类别语义: {img.url}")
    return errors


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v3 评测数据集确定性生成 + 自检")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="生成种子（默认固定值）")
    parser.add_argument("--no-write", action="store_true", help="只跑校验/自检，不写文件")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    built = build_dataset(args.seed)
    errors = self_check(built)

    decisions = Counter(t.expected.decision for t in built.truths)
    scenes = Counter(c.scene for c in built.cases)
    print("v3 评测数据集自检")
    print(f"  规模      : {len(built.cases)} 案（seed={args.seed}）")
    print(f"  场景分布  : {dict(scenes)}")
    print(f"  真值分布  : {dict(decisions)}")
    if errors:
        for err in errors[:20]:
            print(f"  自检 FAIL : {err}", file=sys.stderr)
        print(f"  自检      : FAIL（共 {len(errors)} 处）", file=sys.stderr)
    else:
        print("  自检      : PASS（分布 / 二次打标 / 文本自证 / 政策可得 / URL 中性）")

    if args.no_write:
        return 0 if not errors else 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, content in built.artifacts.items():
        path = OUT_DIR / name
        path.write_text(content, encoding="utf-8")
        print(f"  已写入    : {path}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
