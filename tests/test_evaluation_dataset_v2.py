"""v2 数据集（320 案）数据契约：只锁数据 / schema / 生成器契约，不跑评估器。

1. 规模 ≥300、五类 scene 分布容差 ±5pp；manifest 只校验最小契约
   （``schema_version`` / ``total`` / ``file``）；
2. ``abstain_label`` 与 ``decision`` 100% 一致；AUTO 为主、SHOULD_ABSTAIN 约占一成且集中在
   boundary/evasion/multi-signal（按 ``expected.abstain_label`` 就地统计）；
3. 每条 input 经 ProductReviewCase 强解析、``schema_version=2``、
   ``lineage.seed_case_id`` 为合法种子、REJECT 案 ``risk_level=HIGH``、图片 url 中性无类别语义；
4. 同 seed 生成两遍逐字节一致；
5. 老格式 JSONL（无 abstain_label / 无 lineage）照常读入且 None 等价 AUTO_DECIDABLE。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from pra.domain.models import ProductReviewCase
from pra.evaluation.dataset.loader import load_dataset, scene_stats
from pra.evaluation.dataset.schema import EvalCase

REPO_ROOT = Path(__file__).resolve().parents[1]
V2_PATH = REPO_ROOT / "eval_data" / "v2" / "cases_v2.jsonl"
V2_MANIFEST = REPO_ROOT / "eval_data" / "v2" / "manifest.json"
GEN_SCRIPT = REPO_ROOT / "scripts" / "eval_dataset_gen.py"

_SCENE_PCT = {"normal": 0.20, "violation": 0.20, "boundary": 0.30,
              "multi-signal": 0.20, "evasion": 0.10}
_ABSTAIN_FOCUS_SCENES = {"boundary", "evasion", "multi-signal"}
# abstention 三桶：两个标签 + 未标注（v1 老数据 / v2 缺字段）
_ABSTAIN_BUCKETS = ("AUTO_DECIDABLE", "SHOULD_ABSTAIN", "UNLABELED")


def _load_gen_module():
    spec = importlib.util.spec_from_file_location("eval_dataset_gen_mod", GEN_SCRIPT)
    assert spec and spec.loader, f"无法定位生成器脚本: {GEN_SCRIPT}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _abstain_buckets(cases: list[EvalCase]) -> dict:
    """就地按 ``expected.abstain_label`` 统计三桶计数与各 scene 计数。

    ``abstain_label is None`` 记入 ``UNLABELED``（v1 老数据，语义等价 AUTO_DECIDABLE）。

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


# --- 1) 规模 + 五类分布 + manifest 最小契约


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
    assert Path(manifest["file"]).name == "cases_v2.jsonl"


# --- 2) 真值 × abstention 一致性 + 分布口径


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
    # AUTO 为主、SHOULD_ABSTAIN 占一成上下（30-50 条）
    auto_share = buckets["counts"]["AUTO_DECIDABLE"] / total
    abstain_share = buckets["counts"]["SHOULD_ABSTAIN"] / total
    assert 0.80 <= auto_share <= 0.92
    assert 0.08 <= abstain_share <= 0.18
    assert 30 <= buckets["counts"]["SHOULD_ABSTAIN"] <= 50
    # SHOULD_ABSTAIN 全部集中在 boundary/evasion/multi-signal
    focus = sum(buckets["by_scene"][s]["SHOULD_ABSTAIN"] for s in _ABSTAIN_FOCUS_SCENES)
    assert focus == buckets["counts"]["SHOULD_ABSTAIN"], (
        "SHOULD_ABSTAIN 应集中在 boundary/evasion/multi-signal"
    )
    # 每类 SHOULD_ABSTAIN 出现的 scene 都有 ≥1 条（两套分母在各 scene 都可分层）
    for s in _ABSTAIN_FOCUS_SCENES:
        assert buckets["by_scene"][s]["SHOULD_ABSTAIN"] >= 1, f"scene={s} 缺少 SHOULD_ABSTAIN 案"


# --- 3) 行结构：ProductReviewCase 可解析 / schema_version / id 隔离 / lineage / 案例标签


def test_v2_rows_are_valid_product_review_cases() -> None:
    cases = load_dataset(V2_PATH)
    for c in cases:
        assert c.schema_version == 2, f"{c.eval_case_id} schema_version 应为 2"
        parsed = ProductReviewCase.model_validate(c.input.model_dump())
        assert parsed.product.title and parsed.merchant_id
        # lineage seed 为语义种子（SEED_V2_*）或模板案标识（EC_*）
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
    """评测输入 URL 不得含类别语义（== 把 GT 直接喂给模型）。

    v1/v2 的图片 url 一律为中性资源 ID（``…/eval/asset-NNNN/img1.jpg``）：真实 LLM 会连
    url 字面一起看到，``viol_*`` / ``clean_*`` / ``bound_*`` / ``logo_*`` 这类路径段等同于
    泄漏真值类目。本守护锁死「中性」这条不变量，防再手写回语义 url。
    """
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


# --- 5) 老格式（无 abstain_label / 无 lineage）向后兼容


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
    assert buckets["counts"]["UNLABELED"] == 1  # None 等价 AUTO_DECIDABLE
    auto_equivalent = buckets["counts"]["AUTO_DECIDABLE"] + buckets["counts"]["UNLABELED"]
    assert auto_equivalent == 1
    EvalCase.model_validate(c.model_dump())
