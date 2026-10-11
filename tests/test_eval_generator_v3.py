"""生成器 seam：v3 数据集契约 —— schema / 规模分布 / 字节级复现 / 世界一致性 / 词表 / 隔离。"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from evaluation.dataset.loader import load_cases_v3, load_truth_v3
from evaluation.generator.builder import DEFAULT_SEED, build_artifacts
from evaluation.labeling.labeler import load_rulebook, scan_text_hits
from helpers import load_script

REPO_ROOT = Path(__file__).resolve().parents[1]
V3_DIR = REPO_ROOT / "eval_data" / "v3"
CASES_PATH = V3_DIR / "cases_v3.jsonl"
TRUTH_PATH = V3_DIR / "truth_v3.jsonl"
SEED_SQL_PATH = V3_DIR / "seed_v3.sql"
MANIFEST_PATH = V3_DIR / "manifest.json"

_SCENE_PCT = {"normal": 0.20, "violation": 0.20, "boundary": 0.30,
              "multi-signal": 0.20, "evasion": 0.10}
_TOOL_VOCAB = {"ProductTool", "MerchantTool", "CaseSearchTool", "PolicySearchTool"}

_PRODUCT_ROW = re.compile(
    r"^\s*\('(?P<pid>P_V3_\d+)', '(?P<cat>[^']*)', (?P<brand>NULL|'(?:[^']|'')*'), "
    r"(?P<ver>\d+), '(?P<status>[^']+)'\)(?P<tail>,| AS new)?$"
)
_MERCHANT_ROW = re.compile(
    r"^\s*\('(?P<mid>M_V3_\d+)', (?P<sim>\d+), (?P<rem>\d+), (?P<rel>\d+), (?P<credit>\d+)\)(?P<tail>,| AS new)?$"
)


def _pairs():
    """案件 × 真值按 eval_case_id 一一配对（行序一致）。"""
    cases = load_cases_v3(CASES_PATH)
    truths = load_truth_v3(TRUTH_PATH)
    assert [c.eval_case_id for c in cases] == [t.eval_case_id for t in truths]
    return list(zip(cases, truths, strict=True))


def _seed_rows() -> tuple[dict, dict]:
    """解析 seed_v3.sql：返回 (product 行表, merchant 行表)，键为主键。"""
    products: dict[str, dict] = {}
    merchants: dict[str, dict] = {}
    for line in SEED_SQL_PATH.read_text(encoding="utf-8").splitlines():
        m = _PRODUCT_ROW.match(line)
        if m:
            brand = m.group("brand")
            products[m.group("pid")] = {
                "category": m.group("cat"),
                "brand": None if brand == "NULL" else brand[1:-1].replace("''", "'"),
                "version": int(m.group("ver")),
                "status": m.group("status"),
            }
            continue
        m = _MERCHANT_ROW.match(line)
        if m:
            merchants[m.group("mid")] = {
                "similar_product_count": int(m.group("sim")),
                "removals": int(m.group("rem")),
                "title_relisting_count": int(m.group("rel")),
                "credit_score": int(m.group("credit")),
            }
    return products, merchants


def test_v3_dataset_scale_scene_distribution_and_manifest() -> None:
    """320 案、schema_version=3、五类场景占比 20/20/30/20/10（±5pp）、dataset manifest 口径一致。"""
    cases = load_cases_v3(CASES_PATH)
    assert len(cases) == 320
    for i, c in enumerate(cases, start=1):
        assert c.schema_version == 3
        assert c.eval_case_id == f"EC_V3_{i:04d}"
    total = len(cases)
    by_scene = Counter(c.scene for c in cases)
    for scene, pct in _SCENE_PCT.items():
        assert abs(by_scene[scene] - pct * total) <= 0.05 * total, (
            f"scene={scene} 占比 {by_scene[scene]}/{total} 超出 ±5pp"
        )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 3
    assert manifest["dataset_version"] == "v3"
    assert manifest["seed"] == DEFAULT_SEED
    assert manifest["total"] == total
    assert manifest["scene_distribution"] == dict(by_scene)
    rulebook = load_rulebook()
    assert manifest["rulebook_version"] == rulebook["rulebook_version"]


def test_v3_cases_rows_carry_no_truth_or_manifest() -> None:
    """案件文件只含输入与构造记录：埋点清单与真值只在真值侧。"""
    for lineno, line in enumerate(CASES_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        row = json.loads(line)
        assert set(row) == {"eval_case_id", "schema_version", "scene", "construction", "input"}, (
            f"cases_v3.jsonl 第 {lineno} 行混入真值侧字段: {sorted(row)}"
        )
    for line in TRUTH_PATH.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert set(row) == {"eval_case_id", "manifest", "expected"}


def test_v3_byte_level_reproducible() -> None:
    """同 seed 两次构建字节级一致，且与已提交产物逐字节相同。"""
    first = build_artifacts(DEFAULT_SEED)
    second = build_artifacts(DEFAULT_SEED)
    assert first == second
    assert set(first) == {"cases_v3.jsonl", "truth_v3.jsonl", "seed_v3.sql", "manifest.json"}
    for name, content in first.items():
        assert (V3_DIR / name).read_text(encoding="utf-8") == content, (
            f"已提交产物 {name} 与当前生成器输出不一致 —— 需重跑 scripts/build_eval_dataset.py"
        )


def test_v3_world_consistency_with_seed_sql() -> None:
    """世界一致性：埋点清单声称的商家数值/在库事实与种子 SQL 行完全一致。"""
    products, merchants = _seed_rows()
    assert len(products) == 320 and len(merchants) == 320
    for case, truth in _pairs():
        pid, mid = case.input.product.product_id, case.input.merchant_id
        assert pid == truth.manifest.registry.product_id, case.eval_case_id
        assert mid == truth.manifest.merchant.merchant_id, case.eval_case_id

        row = products[pid]
        snap = case.input.product
        assert row["version"] == snap.version == truth.manifest.registry.version
        assert row["status"] == truth.manifest.registry.status
        if truth.manifest.registry_mismatch:
            assert row["brand"] != snap.brand, f"{case.eval_case_id} 声称在库不一致但种子行相同"
        else:
            assert row["brand"] == snap.brand, case.eval_case_id
        if "category" in truth.manifest.snapshot_gaps:
            assert not snap.category and row["category"], case.eval_case_id
        else:
            assert row["category"] == snap.category, case.eval_case_id

        m = truth.manifest.merchant
        assert merchants[mid] == {
            "similar_product_count": m.similar_product_count,
            "removals": m.removals,
            "title_relisting_count": m.title_relisting_count,
            "credit_score": m.credit_score,
        }, case.eval_case_id


def test_v3_text_hits_are_self_certifying() -> None:
    """词命中快照自证：按规则书重扫标题/描述，命中集合与埋点清单一致；不多不少。"""
    rulebook = load_rulebook()
    for case, truth in _pairs():
        snap = case.input.product
        scanned = scan_text_hits(snap.title, snap.description, rulebook)
        assert scanned == truth.manifest.text_hits, case.eval_case_id
        assert truth.manifest.snapshot_brand == snap.brand, case.eval_case_id
        gaps = set(truth.manifest.snapshot_gaps)
        assert ("brand" in gaps) == (snap.brand is None or not snap.brand.strip())
        assert ("category" in gaps) == (not snap.category.strip())


def test_v3_image_urls_carry_no_class_semantics() -> None:
    """评测输入 URL 不得含类别语义。"""
    leaked = ("viol", "clean", "bound", "logo", "brand", "risk", "reject", "pass", "human")
    for c in load_cases_v3(CASES_PATH):
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


def test_v3_truth_vocab_and_policy_availability() -> None:
    """风险类型只含 2 值；期望工具只含生产四工具；应引政策在 RAG 政策库中可得且生效。"""
    corpus = json.loads(
        (REPO_ROOT / "src" / "pra" / "rag" / "corpus" / "policies.json").read_text(encoding="utf-8")
    )
    effective = {p["policy_id"] for p in corpus["policies"] if p["status"] == "EFFECTIVE"}
    for _, truth in _pairs():
        exp = truth.expected
        assert set(exp.risk_type) <= {"POTENTIAL_IP_RISK", "EVASION_PATTERN"}
        assert set(exp.expected_tools) <= _TOOL_VOCAB
        assert set(exp.expected_policies) <= effective, truth.eval_case_id


def test_v3_scene_decision_semantics() -> None:
    """场景→真值的语义映射：normal 全 PASS、violation 全 REJECT、evasion 无 PASS。"""
    scene_decisions: dict[str, set] = {}
    for case, truth in _pairs():
        scene_decisions.setdefault(case.scene, set()).add(truth.expected.decision)
    assert scene_decisions["normal"] == {"PASS"}
    assert scene_decisions["violation"] == {"REJECT"}
    assert "PASS" not in scene_decisions["evasion"]
    assert scene_decisions["boundary"] == {"PASS", "HUMAN_REVIEW"}
    assert scene_decisions["multi-signal"] == {"REJECT", "HUMAN_REVIEW"}


def test_v3_hard_cases_present_by_construction() -> None:
    """难例是构造维度：强规避 / 弱证据 / 计数临界 / 在库不一致各至少 1 案。"""
    pairs = _pairs()
    rulebook = load_rulebook()
    threshold = rulebook["merchant_dirty_min"]

    def dirty(t) -> bool:
        m = t.manifest.merchant
        return m.removals >= threshold or m.title_relisting_count >= threshold

    strong = [t for _, t in pairs if len(t.manifest.text_hits) >= 2 and dirty(t)
              and t.expected.decision == "REJECT"]
    weak = [t for _, t in pairs
            if len(t.manifest.text_hits) == 1 and not dirty(t)
            and t.expected.decision == "HUMAN_REVIEW"]
    below = [t for _, t in pairs
             if max(t.manifest.merchant.removals, t.manifest.merchant.title_relisting_count) == threshold - 1]
    at = [t for _, t in pairs
          if max(t.manifest.merchant.removals, t.manifest.merchant.title_relisting_count) == threshold
          and not t.manifest.text_hits]
    mismatch = [t for _, t in pairs if t.manifest.registry_mismatch]
    assert strong and weak and below and at and mismatch
    assert all(t.expected.decision == "PASS" for t in below)
    assert all(t.expected.decision == "HUMAN_REVIEW" for t in at)


def test_rag_corpus_isolation_covers_v3() -> None:
    """隔离 seam：build_rag_corpus 自检把 v3 案件纳入取数纪律（退出码 0）。"""
    module = load_script(REPO_ROOT / "scripts" / "build_rag_corpus.py")
    assert str(V3_DIR / "cases_v3.jsonl") in {str(p) for p in module.EVAL_FILES}
    assert module.main(["--no-write"]) == 0
