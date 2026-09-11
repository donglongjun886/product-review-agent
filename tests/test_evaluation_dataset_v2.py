"""v2 数据集（320 案）完整性：只锁数据 / schema / 生成器契约，不跑评估器。

1. 规模 ≥300、五类 scene 分布容差 ±5pp、manifest 与实际 JSONL 分布一致；
2. ``abstain_label`` 与 ``decision`` 100% 一致；AUTO 为主、SHOULD_ABSTAIN 集中在
   boundary/evasion/multi-signal；
3. 每条 input 经 ProductReviewCase 强解析、schema_version=2、id 与 v1 不重叠、
   lineage 完整、REJECT 案有可引用政策依据；
4. 同 seed 生成两遍逐字节一致；
5. v1 老 JSONL（无 abstain_label）照常读入且 None 等价 AUTO_DECIDABLE。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from pra.domain.models import ProductReviewCase
from pra.evaluation.dataset.loader import abstain_stats, load_dataset, scene_stats
from pra.evaluation.dataset.schema import EvalCase

REPO_ROOT = Path(__file__).resolve().parents[1]
V1_PATH = REPO_ROOT / "eval_data" / "v1" / "cases_v1.jsonl"
V2_PATH = REPO_ROOT / "eval_data" / "v2" / "cases_v2.jsonl"
V2_MANIFEST = REPO_ROOT / "eval_data" / "v2" / "manifest.json"
GEN_SCRIPT = REPO_ROOT / "scripts" / "eval_dataset_gen.py"

_SCENE_PCT = {"normal": 0.20, "violation": 0.20, "boundary": 0.30,
              "multi-signal": 0.20, "evasion": 0.10}
_ABSTAIN_FOCUS_SCENES = {"boundary", "evasion", "multi-signal"}


def _load_gen_module():
    spec = importlib.util.spec_from_file_location("eval_dataset_gen_mod", GEN_SCRIPT)
    assert spec and spec.loader, f"无法定位生成器脚本: {GEN_SCRIPT}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- 1) 规模 + 五类分布 + manifest 一致性


def test_v2_dataset_scale_and_scene_distribution() -> None:
    cases = load_dataset(V2_PATH)
    assert len(cases) >= 300, "Phase 2 正式集至少 300 条"
    total = len(cases)
    ss = scene_stats(cases)
    for scene, pct in _SCENE_PCT.items():
        n = ss["by_scene"][scene]["total"]
        expect_n = pct * total
        # 容差 ±5 个百分点
        assert abs(n - expect_n) <= 0.05 * total, (
            f"scene={scene} 占比 {n}/{total} 超出 ±5pp（期望 {pct:.0%}）"
        )
    manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["total"] == total
    assert manifest["scene_distribution"] == {
        s: ss["by_scene"][s]["total"] for s in _SCENE_PCT
    }


# --- 2) 真值 × abstention 一致性 + 分布口径


def test_v2_truth_abstain_consistency() -> None:
    cases = load_dataset(V2_PATH)
    for c in cases:
        exp = c.expected
        if exp.decision == "HUMAN_REVIEW":
            assert exp.abstain_label == "SHOULD_ABSTAIN", c.eval_case_id
        else:
            assert exp.abstain_label == "AUTO_DECIDABLE", c.eval_case_id
    aa = abstain_stats(cases)
    assert aa["LEGACY_UNLABELED"] == 0, "v2 正式集不允许未标注 abstain_label 的行"
    # AUTO 为主、SHOULD_ABSTAIN 占一成上下（30-50 条）
    assert 0.80 <= aa["auto_share"] <= 0.92
    assert 0.08 <= aa["abstain_share"] <= 0.18
    assert 30 <= aa["SHOULD_ABSTAIN"] <= 50
    # SHOULD_ABSTAIN 全部集中在 boundary/evasion/multi-signal
    focus = sum(aa["by_scene"][s]["SHOULD_ABSTAIN"] for s in _ABSTAIN_FOCUS_SCENES)
    assert focus == aa["SHOULD_ABSTAIN"], "SHOULD_ABSTAIN 应集中在 boundary/evasion/multi-signal"
    # 每类 SHOULD_ABSTAIN 出现的 scene 都有 ≥1 条（abstention 指标可 scene 分层）
    for s in _ABSTAIN_FOCUS_SCENES:
        assert aa["by_scene"][s]["SHOULD_ABSTAIN"] >= 1, f"scene={s} 缺少 SHOULD_ABSTAIN 案"


# --- 3) 行结构：ProductReviewCase 可解析 / schema_version / id 隔离 / lineage / 政策依据


def test_v2_rows_are_valid_product_review_cases() -> None:
    cases = load_dataset(V2_PATH)
    v1_ids = {c.eval_case_id for c in load_dataset(V1_PATH)}
    for c in cases:
        assert c.schema_version == 2, f"{c.eval_case_id} schema_version 应为 2"
        assert c.eval_case_id not in v1_ids, "v2 与 v1 的 eval_case_id 不得重叠"
        parsed = ProductReviewCase.model_validate(c.input.model_dump())
        assert parsed.product.title and parsed.merchant_id
        # lineage seed 为 v1 老案或 SEED_V2_* 语义种子
        assert c.lineage is not None and c.lineage.seed_case_id and c.lineage.mutation
        seed = c.lineage.seed_case_id
        assert seed.startswith(("EC_", "SEED_V2_")), (
            f"{c.eval_case_id} lineage.seed_case_id 非法: {seed}"
        )
        if c.expected.decision == "REJECT":
            assert c.expected.applicable_policy, (
                f"REJECT 案 {c.eval_case_id} 缺可引用政策依据（REJECT Gate 前置）"
            )
            assert c.expected.risk_level == "HIGH"
        for img in c.input.product.images:
            assert img.url.startswith("https://cdn.example.com/")


def test_v2_families_cover_all_intended_shapes() -> None:
    cases = load_dataset(V2_PATH)
    fams = {c.annotation["family"] for c in cases if c.annotation}
    expected_fams = {
        "clean_own", "text_evasion", "ocr_brand_dirty", "blackbrand_field",
        "brand_missing_verify", "cat_missing_verify", "weak_sim_own", "neutral_clean",
        "styleword_own", "adapter_brandword", "brand_missing_unverifiable",
        "weak_sim_noinfo", "ssim_dirty", "wsim_dirty", "own_adversarial",
        "logo_dirty", "pair_hide", "dirty_brand_missing_cleanimg",
        "ssim_cleanmerchant_bm", "multi_weak_abstain",
    }
    assert expected_fams <= fams, f"缺 family: {sorted(expected_fams - fams)}"


# --- 4) 生成器确定性（同 seed 两遍 → 逐字节一致）


def test_v2_generator_determinism(tmp_path: Path) -> None:
    gen = _load_gen_module()
    rows_a = gen.generate(count=60, seed=7)
    rows_b = gen.generate(count=60, seed=7)
    assert rows_a == rows_b, "同 seed 生成两遍必须逐字节一致（确定性可重放）"
    for row in rows_a:
        EvalCase.model_validate(row)
    out_a = tmp_path / "a.jsonl"
    out_b = tmp_path / "b.jsonl"
    out_a.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows_a), encoding="utf-8")
    out_b.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows_b), encoding="utf-8")
    assert out_a.read_bytes() == out_b.read_bytes()


def test_v2_generator_quota_matches_schedule(tmp_path: Path) -> None:
    gen = _load_gen_module()
    rows = gen.generate(count=320, seed=42)
    assert len(rows) == 320
    from collections import Counter
    scenes = Counter(r["scene"] for r in rows)
    assert scenes == {"normal": 64, "violation": 64, "boundary": 96,
                      "multi-signal": 64, "evasion": 32}


# --- 5) Phase 1 老数据向后兼容（schema v2 校验器不破坏 v1）


def test_v1_backward_compat() -> None:
    cases = load_dataset(V1_PATH)  # 无 abstain_label 字段的老 JSONL
    assert len(cases) == 35
    for c in cases:
        assert c.expected.abstain_label is None, "v1 老数据 abstain_label 应为 None"
        assert c.expected.decision in {"PASS", "REJECT"}
        assert c.lineage is None, "v1 老数据无 lineage 字段"
    aa = abstain_stats(cases)
    assert aa["LEGACY_UNLABELED"] == 35  # None 等价 AUTO_DECIDABLE
    assert aa["auto_decidable_equivalent"] == 35
    for c in cases[:3]:
        EvalCase.model_validate(c.model_dump())
