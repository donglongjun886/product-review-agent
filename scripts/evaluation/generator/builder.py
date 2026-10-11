"""v3 评测数据集生成器（构造即标签）：确定性合成案件 + 埋点清单 + 种子 SQL。

埋点清单（``CaseManifestV3``）交独立打标函数按冻结规则书推导真值；本模块只在
「文本自证」处调用 labeling 的扫描函数做构造期校验（模板污染会显式失败）。
同 seed → 输出字节级一致，由 ``tests/test_eval_generator_v3.py`` 守护。
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from evaluation.dataset.schema import (
    CaseManifestV3,
    EvalCaseV3,
    EvalConstructionV3,
    EvalTruthV3,
    MerchantFactsV3,
    RegistryFactsV3,
    SceneName,
)
from evaluation.labeling.labeler import (
    RULEBOOK_PATH,
    label_case,
    load_rulebook,
    scan_text_hits,
)

from pra.domain.models import ProductImage, ProductInfo, ProductReviewCase, SkuInfo

DEFAULT_SEED = 20261012
DATASET_VERSION = "v3"
TOTAL = 320

REPO_ROOT = Path(__file__).resolve().parents[3]
POLICIES_PATH = REPO_ROOT / "src" / "pra" / "rag" / "corpus" / "policies.json"

# (场景, 构造模式, 案数)：占比 20/20/30/20/10。
_PATTERN_PLAN: tuple[tuple[SceneName, str, int], ...] = (
    ("normal", "clean", 64),
    ("violation", "blacklist_brand", 16),
    ("violation", "evasion_word_clean_merchant", 24),
    ("violation", "evasion_word_dirty_merchant", 24),
    ("boundary", "brand_word_only", 24),
    ("boundary", "merchant_count_two", 24),
    ("boundary", "merchant_count_three", 24),
    ("boundary", "snapshot_gap", 12),
    ("boundary", "registry_mismatch", 12),
    ("multi-signal", "strong_evasion", 22),
    ("multi-signal", "brand_word_plus_dirty_merchant", 22),
    ("multi-signal", "blacklist_plus_evasion", 20),
    ("evasion", "merchant_dirty_only", 16),
    ("evasion", "hidden_evasion_in_description", 16),
)

_CATS = ("女鞋/运动鞋", "箱包/女包", "服装/卫衣")
_GOODS = {
    "女鞋/运动鞋": ("复古跑鞋", "板鞋", "休闲鞋", "缓震跑步鞋"),
    "箱包/女包": ("托特包", "双肩包", "斜挎小方包", "通勤手提包"),
    "服装/卫衣": ("宽松卫衣", "连帽卫衣", "印花外套", "基础款卫衣"),
}
_OWN_BRANDS = ("潮动", "云步", "简行", "山丘", "山野")

_TITLE_SUFFIX = ("轻便透气", "日常百搭", "经典版型", "新款上市")
_DESC_POOL = (
    "织物面料，做工精细，日常通勤穿着。",
    "重磅面料，宽松版型，日常休闲穿着。",
    "材质结实耐用，多隔层设计，通勤出行适用。",
)

# 埋词模板：{goods} 为中性商品名；模板本身不含其它词表词（构造期校验守住）。
_EVASION_TITLE = {
    "高仿": "高仿{goods}",
    "复刻": "复刻版{goods}",
    "1:1": "{goods} 1:1 精制",
    "原单": "原单{goods} 尾货",
    "同款": "明星同款{goods}",
}
_EVASION_DESC = {
    "高仿": "高仿做工，细节到位，支持对比。",
    "复刻": "复刻经典鞋型，还原度高。",
    "1:1": "按原版 1:1 打造，包装齐全。",
    "原单": "原单渠道货源，数量有限。",
    "同款": "与柜台同款出货，品质保证。",
}
_BRAND_TITLE = {
    "NIKE": "NIKE 风格{goods}",
    "ADIDAS": "{goods} ADIDAS 元素",
    "GUCCI": "GUCCI 风{goods}",
    "LOUIS VUITTON": "LOUIS VUITTON 风格{goods}",
    "LV": "{goods} LV 老花款式",
}

_ASSET_BASE = 2000
# 快照时间口径同 DB DATETIME（naive），与生产 ProductInfo.listing_time 一致。
_BASE_TIME = datetime.fromisoformat("2024-09-01 09:00:00")


@dataclass(frozen=True)
class BuiltDataset:
    """一次构建的全部产物：案件、真值与四个落盘文件内容。"""

    cases: list[EvalCaseV3]
    truths: list[EvalTruthV3]
    artifacts: dict[str, str]


def _merchant(mid: str, kind: str, ordinal: int, rng: random.Random) -> MerchantFactsV3:
    """按构造档位产出商家画像事实（clean/two/three/dirty/strong_dirty）。"""
    if kind == "clean":
        removals, title = rng.randint(0, 1), rng.randint(0, 1)
        similar, credit = rng.randint(0, 4), rng.randint(85, 98)
    elif kind == "two":
        if ordinal % 2 == 0:
            removals, title = 2, rng.randint(0, 1)
        else:
            removals, title = rng.randint(0, 1), 2
        similar, credit = rng.randint(0, 4), rng.randint(80, 95)
    elif kind == "three":
        if ordinal % 2 == 0:
            removals, title = 3, 0
        else:
            removals, title = 0, 3
        similar, credit = rng.randint(1, 8), rng.randint(55, 80)
    elif kind == "strong_dirty":
        removals, title = rng.randint(5, 9), rng.randint(3, 6)
        similar, credit = rng.randint(10, 40), rng.randint(20, 50)
    else:  # dirty
        if ordinal % 2 == 0:
            removals, title = rng.randint(3, 8), rng.randint(0, 2)
        else:
            removals, title = rng.randint(0, 2), rng.randint(3, 6)
        similar, credit = rng.randint(5, 30), rng.randint(30, 60)
    return MerchantFactsV3(
        merchant_id=mid, removals=removals, title_relisting_count=title,
        similar_product_count=similar, credit_score=credit,
    )


def _neutral_text(goods: str, brand: str | None, rng: random.Random) -> tuple[str, str]:
    """中性标题/描述（保证不含任何词表词）。"""
    title = f"{brand or ''}{goods} {rng.choice(_TITLE_SUFFIX)}"
    return title, rng.choice(_DESC_POOL)


def _plant(
    title: str, description: str, goods: str, planted: list[tuple[str, str]],
) -> tuple[str, str]:
    """把埋点词按模板写入标题/描述（同字段多词拼接，不互相覆盖）。"""
    desc_parts: list[str] = []
    for field, term in planted:
        if field == "title":
            template = _EVASION_TITLE.get(term) or _BRAND_TITLE.get(term)
            title = template.format(goods=goods) if template else title
        else:
            template = _EVASION_DESC.get(term) or f"设计参考 {term} 经典款。"
            desc_parts.append(template.format(goods=goods))
    if desc_parts:
        description = "\n".join(desc_parts)
    return title, description


def _build_one(
    seq: int, scene: SceneName, pattern: str, ordinal: int,
    rng: random.Random, rulebook: dict,
) -> tuple[EvalCaseV3, CaseManifestV3]:
    """构造单案：返回案件（含构造记录）+ 埋点清单。

    :param seq: 全数据集 1 起始序号（决定 ID 与图片资源号）。
    :param ordinal: 模式内序号（0 起始，用于确定性轮换）。
    """
    eval_case_id = f"EC_V3_{seq:04d}"
    product_id = f"P_V3_{seq:04d}"
    merchant_id = f"M_V3_{seq:04d}"
    category = _CATS[seq % len(_CATS)]
    goods = rng.choice(_GOODS[category])
    own_brand = rng.choice(_OWN_BRANDS)
    version = rng.randint(1, 3)

    evasion_terms = [str(t) for t in rulebook["evasion_terms"]]
    brand_terms = [str(t) for t in rulebook["brand_terms"]]

    brand: str | None = own_brand
    registry_brand: str | None = own_brand
    gaps: list[Literal["brand", "category"]] = []
    mismatch = False
    merchant_kind = "clean"
    planted: list[tuple[str, str]] = []
    event_type = "NEW_LISTING"

    if pattern == "clean":
        pass
    elif pattern == "blacklist_brand":
        brand = registry_brand = rng.choice([str(b) for b in rulebook["blacklisted_brands"]])
    elif pattern == "evasion_word_clean_merchant":
        term = evasion_terms[ordinal % len(evasion_terms)]
        planted = [("title", term)] if ordinal % 2 == 0 else [("description", term)]
    elif pattern == "evasion_word_dirty_merchant":
        planted = [("title", evasion_terms[ordinal % len(evasion_terms)])]
        merchant_kind = "dirty"
    elif pattern == "brand_word_only":
        planted = [("title", brand_terms[ordinal % len(brand_terms)])]
    elif pattern == "merchant_count_two":
        merchant_kind = "two"
    elif pattern == "merchant_count_three":
        merchant_kind = "three"
    elif pattern == "snapshot_gap":
        if ordinal % 3 == 2:
            gaps = ["category"]
        else:
            gaps = ["brand"]
            brand = registry_brand = None
    elif pattern == "registry_mismatch":
        mismatch = True
        registry_brand = None
    elif pattern == "strong_evasion":
        k = 2 + ordinal % 2
        terms = [evasion_terms[(ordinal + j) % len(evasion_terms)] for j in range(k)]
        planted = [("title", terms[0])] + [("description", t) for t in terms[1:]]
        merchant_kind = "strong_dirty"
    elif pattern == "brand_word_plus_dirty_merchant":
        planted = [("title", brand_terms[ordinal % len(brand_terms)])]
        merchant_kind = "dirty"
    elif pattern == "blacklist_plus_evasion":
        brand = registry_brand = rng.choice([str(b) for b in rulebook["blacklisted_brands"]])
        planted = [("title", evasion_terms[ordinal % len(evasion_terms)])]
    elif pattern == "merchant_dirty_only":
        merchant_kind = "dirty"
        event_type = "UPDATE_TITLE"
    elif pattern == "hidden_evasion_in_description":
        planted = [("description", evasion_terms[ordinal % len(evasion_terms)])]
    else:
        raise ValueError(f"未知构造模式: {pattern}")

    snapshot_category = "" if "category" in gaps else category
    title, description = _neutral_text(goods, brand, rng)
    title, description = _plant(title, description, goods, planted)

    # 文本自证：重扫结果必须与植入清单一致（模板污染显式失败），并以其为准入清单。
    scanned = scan_text_hits(title, description, rulebook)
    if {(h.field, h.term) for h in scanned} != set(planted):
        raise RuntimeError(
            f"{eval_case_id} 文本自证失败: 植入 {sorted(planted)} 与重扫 "
            f"{sorted((h.field, h.term) for h in scanned)} 不一致"
        )

    merchant = _merchant(merchant_id, merchant_kind, ordinal, rng)
    registry = RegistryFactsV3(
        product_id=product_id, category=category, brand=registry_brand,
        version=version, status="ON_SALE",
    )
    product = ProductInfo(
        product_id=product_id, title=title, description=description,
        category=snapshot_category, brand=brand,
        attributes={"材质": "织物", "适用人群": "通用"},
        sku_list=[SkuInfo(sku_id="S_1", color="默认", size="均码",
                          price=float(rng.randint(59, 599)))],
        images=[ProductImage(
            url=f"https://cdn.example.com/eval/asset-{_ASSET_BASE + seq}/img1.jpg",
            source="主图",
        )],
        listing_time=_BASE_TIME + timedelta(minutes=7 * seq),
        version=version,
    )
    case = EvalCaseV3(
        eval_case_id=eval_case_id,
        scene=scene,
        construction=EvalConstructionV3(
            pattern=pattern,
            params={
                "goods": goods,
                "brand": brand or "",
                "planted": ",".join(f"{f}:{t}" for f, t in planted),
                "removals": merchant.removals,
                "title_relisting_count": merchant.title_relisting_count,
                "registry_mismatch": mismatch,
            },
        ),
        input=ProductReviewCase(
            case_id=f"CASE_EC_V3_{seq:04d}", product=product,
            merchant_id=merchant_id, event_type=event_type,
        ),
    )
    manifest = CaseManifestV3(
        text_hits=scanned, snapshot_brand=brand,
        snapshot_gaps=gaps,
        registry_mismatch=mismatch, merchant=merchant, registry=registry,
    )
    return case, manifest


def _sql_str(value: str) -> str:
    """SQL 字符串字面量（单引号转义）。"""
    return "'" + value.replace("'", "''") + "'"


def _render_seed_sql(truths: list[EvalTruthV3]) -> str:
    """把全部案件引用的 product / merchant 行渲染成幂等种子 SQL。"""
    lines = [
        "-- ============================================================================",
        "-- eval_data/v3 评测世界种子（scripts/build_eval_dataset.py 生成；重跑即覆盖）",
        "-- 口径与幂等约定同 migrations/003、004：SET NAMES utf8mb4 +",
        "-- INSERT ... AS new ON DUPLICATE KEY UPDATE；brand 真空缺为 SQL NULL。",
        "-- 表结构由 migrations/003、004 建立；本文件只插评测专用行（P_V3_* / M_V3_*）。",
        "-- ============================================================================",
        "",
        "SET NAMES utf8mb4;",
        "",
        "INSERT INTO product",
        "  (product_id, category, brand, version, status)",
        "VALUES",
    ]
    product_rows = []
    for t in truths:
        r = t.manifest.registry
        brand = "NULL" if r.brand is None else _sql_str(r.brand)
        product_rows.append(
            f"  ({_sql_str(r.product_id)}, {_sql_str(r.category)}, {brand}, {r.version}, {_sql_str(r.status)})"
        )
    lines.append(",\n".join(product_rows) + " AS new")
    lines += [
        "ON DUPLICATE KEY UPDATE",
        "  category = new.category,",
        "  brand = new.brand,",
        "  version = new.version,",
        "  status = new.status;",
        "",
        "INSERT INTO merchant",
        "  (merchant_id, similar_product_count, removals, title_relisting_count, credit_score)",
        "VALUES",
    ]
    merchant_rows = []
    for t in truths:
        m = t.manifest.merchant
        merchant_rows.append(
            f"  ({_sql_str(m.merchant_id)}, {m.similar_product_count}, {m.removals}, "
            f"{m.title_relisting_count}, {m.credit_score})"
        )
    lines.append(",\n".join(merchant_rows) + " AS new")
    lines += [
        "ON DUPLICATE KEY UPDATE",
        "  similar_product_count = new.similar_product_count,",
        "  removals = new.removals,",
        "  title_relisting_count = new.title_relisting_count,",
        "  credit_score = new.credit_score;",
        "",
    ]
    return "\n".join(lines)


def build_dataset(seed: int = DEFAULT_SEED) -> BuiltDataset:
    """构建完整 v3 数据集（确定性：同 seed 同输出）。"""
    rng = random.Random(seed)
    rulebook = load_rulebook()
    cases: list[EvalCaseV3] = []
    truths: list[EvalTruthV3] = []
    seq = 0
    for scene, pattern, count in _PATTERN_PLAN:
        for ordinal in range(count):
            seq += 1
            case, manifest = _build_one(seq, scene, pattern, ordinal, rng, rulebook)
            truths.append(EvalTruthV3(
                eval_case_id=case.eval_case_id,
                manifest=manifest,
                expected=label_case(manifest, rulebook),
            ))
            cases.append(case)
    if len(cases) != TOTAL:
        raise RuntimeError(f"构造模式案数合计 {len(cases)} != {TOTAL}")

    cases_jsonl = "".join(
        json.dumps(c.model_dump(mode="json"), ensure_ascii=False) + "\n" for c in cases
    )
    truth_jsonl = "".join(
        json.dumps(t.model_dump(mode="json"), ensure_ascii=False) + "\n" for t in truths
    )
    seed_sql = _render_seed_sql(truths)
    manifest_json = json.dumps(
        {
            "schema_version": 3,
            "dataset_version": DATASET_VERSION,
            "seed": seed,
            "total": len(cases),
            "rulebook_version": rulebook["rulebook_version"],
            "rulebook_sha256": hashlib.sha256(RULEBOOK_PATH.read_bytes()).hexdigest(),
            "scene_distribution": dict(Counter(c.scene for c in cases)),
            "decision_distribution": dict(Counter(t.expected.decision for t in truths)),
            "files": {
                "cases": "cases_v3.jsonl",
                "truth": "truth_v3.jsonl",
                "seed_sql": "seed_v3.sql",
            },
            "generated_by": "scripts/build_eval_dataset.py",
        },
        ensure_ascii=False, indent=2,
    ) + "\n"
    return BuiltDataset(
        cases=cases,
        truths=truths,
        artifacts={
            "cases_v3.jsonl": cases_jsonl,
            "truth_v3.jsonl": truth_jsonl,
            "seed_v3.sql": seed_sql,
            "manifest.json": manifest_json,
        },
    )


def build_artifacts(seed: int = DEFAULT_SEED) -> dict[str, str]:
    """只取四个落盘文件的内容（文件名 → 文本）。"""
    return build_dataset(seed).artifacts
