"""v2 数据集（320 案）数据契约：只锁数据 / schema 契约，不跑评估器。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from pra.domain.models import ProductReviewCase
from pra.evaluation.dataset.loader import load_dataset
from pra.evaluation.dataset.schema import EvalCase

REPO_ROOT = Path(__file__).resolve().parents[1]
V2_PATH = REPO_ROOT / "eval_data" / "v2" / "cases_v2.jsonl"
V2_MANIFEST = REPO_ROOT / "eval_data" / "v2" / "manifest.json"

_SCENE_PCT = {"normal": 0.20, "violation": 0.20, "boundary": 0.30,
              "multi-signal": 0.20, "evasion": 0.10}
_ABSTAIN_FOCUS_SCENES = {"boundary", "evasion", "multi-signal"}
_ABSTAIN_BUCKETS = ("AUTO_DECIDABLE", "SHOULD_ABSTAIN", "UNLABELED")


def _abstain_buckets(cases: list[EvalCase]) -> dict:
    """就地按 ``expected.abstain_label`` 统计三桶计数与各 scene 计数。

    ``abstain_label is None`` 记入 ``UNLABELED``。

    :return: ``{"counts": {桶: n}, "by_scene": {scene: {桶: n}}}``
    """
    counts: dict[str, int] = dict.fromkeys(_ABSTAIN_BUCKETS, 0)
    by_scene: dict[str, dict[str, int]] = {
        s: dict.fromkeys(_ABSTAIN_BUCKETS, 0) for s in _SCENE_PCT
    }
    for c in cases:
        label = c.expected.abstain_label
        bucket = label if label is not None else "UNLABELED"
        counts[bucket] += 1
        if c.scene in by_scene:
            by_scene[c.scene][bucket] += 1
    return {"counts": counts, "by_scene": by_scene}


def test_v2_dataset_scale_and_scene_distribution() -> None:
    cases = load_dataset(V2_PATH)
    assert len(cases) >= 300, "Phase 2 正式集至少 300 条"
    total = len(cases)
    by_scene = Counter(c.scene for c in cases)
    for scene, pct in _SCENE_PCT.items():
        n = by_scene[scene]
        expect_n = pct * total
        assert abs(n - expect_n) <= 0.05 * total, (
            f"scene={scene} 占比 {n}/{total} 超出 ±5pp（期望 {pct:.0%}）"
        )
    manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["total"] == total
    assert Path(manifest["file"]).name == "cases_v2.jsonl"


def test_v2_truth_abstain_consistency() -> None:
    cases = load_dataset(V2_PATH)
    for c in cases:
        exp = c.expected
        if exp.decision == "HUMAN_REVIEW":
            assert exp.abstain_label == "SHOULD_ABSTAIN", c.eval_case_id
        else:
            assert exp.abstain_label == "AUTO_DECIDABLE", c.eval_case_id

    total = len(cases)
    buckets = _abstain_buckets(cases)
    assert buckets["counts"]["UNLABELED"] == 0, "v2 正式集不允许未标注 abstain_label 的行"
    auto_share = buckets["counts"]["AUTO_DECIDABLE"] / total
    abstain_share = buckets["counts"]["SHOULD_ABSTAIN"] / total
    assert 0.80 <= auto_share <= 0.92
    assert 0.08 <= abstain_share <= 0.18
    assert 30 <= buckets["counts"]["SHOULD_ABSTAIN"] <= 50
    focus = sum(buckets["by_scene"][s]["SHOULD_ABSTAIN"] for s in _ABSTAIN_FOCUS_SCENES)
    assert focus == buckets["counts"]["SHOULD_ABSTAIN"], (
        "SHOULD_ABSTAIN 应集中在 boundary/evasion/multi-signal"
    )
    for s in _ABSTAIN_FOCUS_SCENES:
        assert buckets["by_scene"][s]["SHOULD_ABSTAIN"] >= 1, f"scene={s} 缺少 SHOULD_ABSTAIN 案"


def test_v2_rows_are_valid_product_review_cases() -> None:
    cases = load_dataset(V2_PATH)
    for c in cases:
        assert c.schema_version == 2, f"{c.eval_case_id} schema_version 应为 2"
        parsed = ProductReviewCase.model_validate(c.input.model_dump())
        assert parsed.product.title and parsed.merchant_id
        assert c.lineage is not None and c.lineage.seed_case_id
        seed = c.lineage.seed_case_id
        assert seed.startswith(("EC_", "SEED_V2_")), (
            f"{c.eval_case_id} lineage.seed_case_id 非法: {seed}"
        )
        if c.expected.decision == "REJECT":
            assert c.expected.risk_level == "HIGH"
        for img in c.input.product.images:
            assert img.url.startswith("https://cdn.example.com/")


def test_eval_image_urls_carry_no_class_semantics() -> None:
    """评测输入 URL 不得含类别语义。"""
    leaked = ("viol", "clean", "bound", "logo", "brand", "risk", "reject", "pass", "human")
    for c in load_dataset(V2_PATH):
        for img in c.input.product.images:
            name = img.url.rsplit("/", 1)[-1].lower()
            segment = img.url.rsplit("/", 2)[-2].lower()
            assert segment.startswith(("asset-", "p_")), (
                f"{c.eval_case_id} 的图片 url 资源段非中性: {img.url}"
            )
            for token in leaked:
                assert token not in segment and token not in name, (
                    f"{c.eval_case_id} 的图片 url 含类别语义 {token!r}: {img.url}"
                )


def test_legacy_rows_without_abstain_label(tmp_path: Path) -> None:
    """老格式 JSONL 照常读入：缺 abstain_label → None，语义等价 AUTO_DECIDABLE。"""
    src = load_dataset(V2_PATH)[0]
    row = src.model_dump(mode="json")
    row["schema_version"] = 1
    row["lineage"] = None
    row["expected"].pop("abstain_label", None)
    row["expected"]["decision"] = "PASS"
    path = tmp_path / "legacy.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    cases = load_dataset(path)
    assert len(cases) == 1
    c = cases[0]
    assert c.expected.abstain_label is None, "老数据缺 abstain_label 应读成 None"
    assert c.expected.decision in {"PASS", "REJECT"}
    assert c.lineage is None, "老数据无 lineage"
    buckets = _abstain_buckets(cases)
    assert buckets["counts"]["UNLABELED"] == 1
    auto_equivalent = buckets["counts"]["AUTO_DECIDABLE"] + buckets["counts"]["UNLABELED"]
    assert auto_equivalent == 1
    EvalCase.model_validate(c.model_dump())
