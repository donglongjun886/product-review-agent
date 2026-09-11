"""Phase 2 正式集（v2）确定性变异生成器。

以 eval_data/v1 的 35 条手工种子 + 少量新增语义种子为模板做确定性字段变异（固定随机种子），
产出 ``eval_data/v2/cases_v2.jsonl`` + ``manifest.json``；命令
``uv run python scripts/eval_dataset_gen.py --out eval_data/v2/ --count 320 --seed 42``。
不联网、不调 LLM；同 (count, seed) → 产物逐字节一致（可重放，进 Regression）。

变异维度（**变异后真值必须仍自洽** —— expected 的 decision/abstain_label 按该案在
Rule / Single-call / Agent 下的真实语义重标，非机械复制模板真值）：
- 标题词：换品牌 / 换品类核心词 / 加规避词 / 加品牌词 / 加风格词，严格受控词表（避免干净案
  变异出意外词命中，也避免风险案变异成表面干净）；
- 图片相似度：世界种子只有 强 0.85+ / 弱 0.70~0.85 / 干净 / Logo 四档，按 URL 选择；
- 商家历史：removals 0~7（M_3307/M_9904 干净、M_6602 中性 1、M_5512/M_8801 脏）；
- OCR 文本：含 / 不含品牌词（仅对 Single-call/Agent 可见的表面信号）；
- category / brand：明确 / 空缺 / 黑名单字段 / 高危或普通类目。

真值标注规则：明确违规（文本明示 高仿/复刻/1:1/同款/原单、brand 命中知名黑名单、强相似且
无授权证据）→ REJECT + AUTO_DECIDABLE；明确正常（brand 明确、无风险词、无规避、无强相似）
→ PASS + AUTO_DECIDABLE；边界（单弱信号 0.7~0.8 且无其它可查信息、brand/category 空缺、
文本含品牌词但可能是适配/风格描述）→ HUMAN_REVIEW + SHOULD_ABSTAIN；特例（有意构造，
annotation.notes 注明）是**需调查才能判的 AUTO_DECIDABLE 案** —— Rule/Single 因缺工具会
COMPLEX/HUMAN（brand 空缺、仅 OCR 品牌词、需商家历史交叉），但 Agent 用评测世界工具
（EVAL_* 种子）能查到证据判对。

目标分布（容差 ±5pp）：normal 20% / violation 20% / boundary 30% / multi-signal 20% /
evasion 10%；AUTO_DECIDABLE 为主（~85-90%），SHOULD_ABSTAIN ~10-15%，集中在
boundary/evasion/multi-signal 以保证 abstention 指标有统计意义。

世界事实锚点与 ``pra.evaluation.harness.agent_scheme`` 的 EVAL_* 种子同一份（生成器 import
该常量集），保证每条 input 的 product_id / merchant_id / image URL / category 在 Agent 评测
世界里可查到与标注一致的事实；真 RAG/商家库接入需扩 EVAL_* 种子并重新生成（manifest 记录
世界标签与生成命令，防口径漂移）。每条 JSONL 行 = EvalCase schema（schema_version=2，含
lineage 溯源与 annotation.family 标签）。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from pra.evaluation.dataset.loader import abstain_stats, scene_stats

# 世界事实单一来源：与 B 面 agent_scheme（Agent 评测世界）同一份 EVAL_* 种子。
# import 失败 → 生成器显式失败，拒绝「手抄世界」漂移。
from pra.evaluation.harness.agent_scheme import (
    EVAL_CATEGORIES,
    EVAL_IMAGE_MATCHES,
    EVAL_MERCHANTS,
    EVAL_POLICY_CLAUSES,
    EVAL_PRECEDENTS,
    EVAL_PRODUCTS,
    EVAL_WORLD_LABEL,
)

# ---------------------------------------------------------------------------
# 世界派生常量（只读自 EVAL_* 种子，不手抄第二份）
# ---------------------------------------------------------------------------

_SCENES = ("normal", "violation", "boundary", "multi-signal", "evasion")
_TOOLS_ALL = ["ImageAnalysisTool", "ProductTool", "MerchantTool", "CaseSearchTool", "PolicySearchTool"]
_TOOLS_ABSTAIN = ["ImageAnalysisTool", "ProductTool", "MerchantTool"]
_MERCHANT_TIER = {
    mid: ("dirty" if int(row["removals"]) >= 3 or int(row["title_relisting_count"]) >= 3
          else "clean" if int(row["removals"]) == 0 else "neutral")
    for mid, row in EVAL_MERCHANTS.items()
}
_MERCHANT_REMOVALS = {mid: int(row["removals"]) for mid, row in EVAL_MERCHANTS.items()}

# 图片世界语义（url → 档位）：clean=无命中；weak=0.70~0.85；strong>=0.85；logo。
_IMG_STRONG: dict[str, list[str]] = {c: [] for c in EVAL_CATEGORIES}
_IMG_WEAK: dict[str, list[str]] = {c: [] for c in EVAL_CATEGORIES}
_IMG_CLEAN: dict[str, list[str]] = {c: [] for c in EVAL_CATEGORIES}
_IMG_LOGO: list[str] = []
for url, row in EVAL_IMAGE_MATCHES.items():
    sims = [m.get("similarity", 0.0) for m in row.get("top_similar", [])]
    if row.get("logos"):
        _IMG_LOGO.append(url)
    elif sims and max(sims) >= 0.85:
        for cat, keys in (("女鞋/运动鞋", ("shoe", "P_88231")), ("箱包/女包", ("bag",)),
                          ("服装/卫衣", ("hoodie",))):
            if any(k in url for k in keys):
                _IMG_STRONG[cat].append(url)
                break
    elif sims:
        for cat, keys in (("女鞋/运动鞋", ("bound_shoe",)), ("箱包/女包", ("bound_bag",))):
            if any(k in url for k in keys):
                _IMG_WEAK[cat].append(url)
                break
    else:
        for cat, keys in (("女鞋/运动鞋", ("clean_shoe",)), ("箱包/女包", ("clean_bag",)),
                          ("服装/卫衣", ("clean_hoodie",))):
            if any(k in url for k in keys):
                _IMG_CLEAN[cat].append(url)
                break

_PID_INFO = {pid: row for pid, row in EVAL_PRODUCTS.items()}
_OWN_ANCHORS: dict[str, list[dict]] = {}   # cat -> [{pid, mid, brand}]
_DIRTY_ANCHORS: dict[str, list[dict]] = {}  # cat -> [{pid, mid, brand=None}]
for cat in EVAL_CATEGORIES:
    own, dirty = [], []
    for pid, row in _PID_INFO.items():
        if row["category"] != cat:
            continue
        mid = row["merchant_id"]
        if row["brand"]:
            own.append({"pid": pid, "mid": mid, "brand": row["brand"]})
        else:
            dirty.append({"pid": pid, "mid": mid, "brand": None})
    _OWN_ANCHORS[cat] = own
    _DIRTY_ANCHORS[cat] = dirty
# 潮动 = 唯一「自有品牌但脏商家」的在库锚点（对抗旗舰 EC_0402 同源，只能进对抗家族）
_ADV_ANCHOR = next(a for a in _OWN_ANCHORS["女鞋/运动鞋"] if a["brand"] == "潮动")
# 中性商家(M_6602)在库自有品牌锚点（山野卫衣）
_NEUTRAL_ANCHOR = next(
    a for a in _OWN_ANCHORS["服装/卫衣"] if _MERCHANT_TIER[a["mid"]] == "neutral"
)
# PASS/干净上下文家族只能用「在库自有品牌 + 商家干净/中性」的锚点：脏商家锚点只进对抗
# 家族，否则「干净自有品牌」案会因脏商家被 Agent 判 HUMAN/REJECT，与真值冲突
_CLEAN_OWN_ANCHORS: dict[str, list[dict]] = {
    cat: [dict(a) for a in anchors if _MERCHANT_TIER[a["mid"]] in ("clean", "neutral")]
    for cat, anchors in _OWN_ANCHORS.items()
}

_POLICY_BY_CAT = {row["category"]: row["policy_id"] for row in EVAL_POLICY_CLAUSES}

# ---------------------------------------------------------------------------
# 受控词表（与 screening terms / agent 表面信号同语义；此处只管生成文本不出界）
# ---------------------------------------------------------------------------
_BRAND_WORDS = frozenset({"NIKE", "ADIDAS", "GUCCI", "LV", "LOUIS VUITTON"})
_EVASION_WORDS = frozenset({"同款", "复刻", "高仿", "1:1", "原单"})
_STYLE_WORDS = frozenset({"复古", "经典", "潮流", "ins风", "韩版"})
# 干净标题核心词（不得含风格词/品牌词/规避词 —— 干净案的表面必须真的干净）
_TITLE_CORES = {
    "女鞋/运动鞋": [
        "轻弹缓震跑步鞋", "百搭小白鞋", "轻便缓震跑步鞋", "软底通勤小白鞋",
        "网面透气跑步鞋", "百搭休闲板鞋", "轻量日常跑鞋", "简约运动休闲鞋",
    ],
    "箱包/女包": [
        "极简通勤托特包", "大容量帆布托特包", "百搭单肩手提包", "通勤大容量托特包",
        "简约帆布购物袋", "轻便大容量手提包",
    ],
    "服装/卫衣": [
        "基础款纯色卫衣", "重磅纯棉圆领卫衣", "宽松落肩卫衣", "简约加绒卫衣",
        "纯色休闲连帽卫衣", "重磅宽松卫衣",
    ],
}
_CLEAN_SUFFIX = ["女款", "男款", "女士", "男女通用", "日常通勤", ""]
_EVENT_POOL = ["NEW_LISTING", "UPDATE_TITLE", "UPDATE_IMAGE", "UPDATE_PRICE"]
# 明示规避短语（真值 REJECT 家族用；含 EVASION_WORDS 词）
_EVASION_PHRASES = ["1:1 复刻", "高仿", "原单 1:1", "同款", "高仿 1:1", "原单", "复刻"]
# 品牌词但可能是适配/风格描述（SHOULD_ABSTAIN 家族；只有品牌词、无规避词）
_ADAPTER_TITLES = {
    "女鞋/运动鞋": ["适配 NIKE 鞋垫的轻弹跑步鞋", "可替换 NIKE 跑鞋鞋带的云步小白鞋",
                   "GUCCI 风格复古跑鞋"],
    "箱包/女包": ["GUCCI 风格大容量托特包", "适配 LV 老花元素的通勤手提包"],
    "服装/卫衣": ["NIKE 联名风格宽松卫衣", "GUCCI 风格印花卫衣"],
}
# 风格词标题核心（boundary styleword 观察案；含且仅含风格词，无品牌/规避词）
_STYLE_CORES = {
    "女鞋/运动鞋": ["复古缓震跑步鞋", "经典百搭小白鞋", "潮流轻便跑步鞋"],
    "箱包/女包": ["复古极简托特包", "经典大容量帆布包"],
    "服装/卫衣": ["复古重磅卫衣", "经典宽松卫衣"],
}
_GEN_DESC = {
    "女鞋/运动鞋": "轻量舒适，日常跑步通勤皆宜。",
    "箱包/女包": "简约大容量设计，日常通勤百搭。",
    "服装/卫衣": "重磅面料，宽松版型，日常休闲穿着。",
}
_OCR_BRAND_WORDS = ["GUCCI", "NIKE", "LOUIS VUITTON", "ADIDAS"]
_BLACK_BRAND_BY_CAT = {
    "女鞋/运动鞋": ["NIKE", "ADIDAS"],
    "箱包/女包": ["GUCCI", "LV"],
    "服装/卫衣": ["GUCCI", "NIKE"],
}

# 溯源种子：v1 老案按（family, cat）映射；无直接 v1 模板的用 SEED_V2_ 语义种子标记。
_V1_SEED: dict[str, dict[str, list[str]]] = {
    "clean_own": {"女鞋/运动鞋": ["EC_0001", "EC_0002", "EC_0005", "EC_0007"],
                  "箱包/女包": ["EC_0003", "EC_0006"],
                  "服装/卫衣": ["EC_0004", "EC_0008"]},
    "text_evasion": {"女鞋/运动鞋": ["EC_0101", "EC_0106"],
                     "箱包/女包": ["EC_0102", "EC_0105"],
                     "服装/卫衣": ["EC_0103"]},
    "ocr_brand_dirty": {"箱包/女包": ["EC_0102"]},
    "blackbrand_field": {"箱包/女包": ["EC_0102"]},
    "brand_missing_verify": {"女鞋/运动鞋": ["EC_0201"], "箱包/女包": ["EC_0207"]},
    "cat_missing_verify": {"女鞋/运动鞋": ["EC_0204"]},
    "weak_sim_own": {"女鞋/运动鞋": ["EC_0202"], "箱包/女包": ["EC_0205"]},
    "neutral_clean": {"服装/卫衣": ["EC_0203"]},
    "styleword_own": {"女鞋/运动鞋": ["EC_0401"]},
    "ssim_dirty": {"女鞋/运动鞋": ["EC_0301", "EC_0401", "EC_0407"], "箱包/女包": ["EC_0302"],
                   "服装/卫衣": ["EC_0304"]},
    "wsim_dirty": {"箱包/女包": ["EC_0303", "EC_0105"]},
    "own_adversarial": {"女鞋/运动鞋": ["EC_0402"]},
    "logo_dirty": {"箱包/女包": ["EC_0404"]},
    "pair_hide": {"女鞋/运动鞋": ["EC_0403", "EC_0407"], "箱包/女包": ["EC_0406"],
                  "服装/卫衣": ["EC_0405"]},
}
_SEED_V2 = {
    "adapter_brandword": "SEED_V2_adapter_brandword",
    "brand_missing_unverifiable": "SEED_V2_brand_missing_unverifiable",
    "weak_sim_noinfo": "SEED_V2_weak_sim_noinfo",
    "ssim_cleanmerchant_bm": "SEED_V2_ssim_cleanmerchant_bm",
    "multi_weak_abstain": "SEED_V2_multi_weak_abstain",
    "dirty_brand_missing_cleanimg": "SEED_V2_dirty_brand_missing_cleanimg",
    "styleword_own": "SEED_V2_styleword_own",
}

# ---------------------------------------------------------------------------
# 变体装配小工具
# ---------------------------------------------------------------------------


def _pick(rng: random.Random, pool: list) -> object:
    """确定性取池内随机元素（同 seed 同序列）。"""
    return pool[rng.randrange(len(pool))]


def _anchor(rng: random.Random, cat: str, pool: list[dict]) -> dict:
    return dict(_pick(rng, pool))


def _clean_text(text: str, *, allow_style: bool = False) -> str:
    """干净文本断言：规避词/品牌词一律不允许；风格词默认不允许（allow_style 例外）。"""
    forbidden = _EVASION_WORDS | _BRAND_WORDS
    if not allow_style:
        forbidden = forbidden | _STYLE_WORDS
    found = sorted(w for w in forbidden if w in text)
    if found:
        raise ValueError(f"干净文本命中禁用词 {found}: {text!r}")
    return text


def _title_with_brand(brand: str, core: str, rng: random.Random) -> str:
    """自有品牌标题：品牌前缀 + 干净核心词 + 可选后缀。"""
    suffix = _pick(rng, _CLEAN_SUFFIX)
    t = f"{brand}{core}"
    return f"{t} {suffix}".strip() if suffix else t


def _plain_title(cat: str, rng: random.Random) -> str:
    """无品牌前缀的干净标题（brand 空缺案的表面不得暗示品牌）。"""
    core = _clean_text(_pick(rng, _TITLE_CORES[cat]))
    suffix = _pick(rng, _CLEAN_SUFFIX)
    return f"{core} {suffix}".strip() if suffix else core


def _images(*items: tuple[str, str | None]) -> list[dict]:
    """[(url, ocr|None), ...] → ProductImage JSON 列表（source 依次 主图/附图1/…）。"""
    return [
        {"url": url, "ocr_text": ocr, "source": src}
        for (url, ocr), src in zip(items, ["主图", "附图1", "附图2", "附图3"])
    ]


def _merchant_evidence(mid: str) -> str:
    return f"merchant_history={_MERCHANT_REMOVALS[mid]}_removals"


def _listing(seq: int) -> str:
    day = 1 + (seq % 28)
    return f"2024-09-{day:02d} 10:00:00"


# ---------------------------------------------------------------------------
# Builder 语义口径 —— 每个 family 一个函数：产出 Row（input 字段 + 真值标签）。
# 签名统一 (rng, seq, scene)；只在单个 scene 用的 builder 忽略 scene 参数。
# ---------------------------------------------------------------------------


def _row_base(*, seq, pid, mid, cat, brand, title, desc, images, event,
              scene, family, decision, abstain, src, hard, reason, note,
              risk_level, risk_type, evidence, tools, policy, seed_id, mutation) -> dict:
    """公共装配：input JSON + expected + annotation + lineage。"""
    input_json = {
        "case_id": f"CASE_EC_V2_{seq:04d}",
        "product": {
            "product_id": pid,
            "title": title,
            "description": desc,
            "category": cat,
            "brand": brand,
            "attributes": {"材质": "织物", "适用人群": "通用"},
            "sku_list": [{"sku_id": "S_1", "color": "默认", "size": "均码", "price": 129.0}],
            "images": images,
            "listing_time": _listing(seq),
            "version": 1 + (seq % 3),
        },
        "merchant_id": mid,
        "event_type": event,
        "screening_signals": [],
    }
    expected = {
        "decision": decision,
        "abstain_label": abstain,
        "risk_level": risk_level,
        "risk_type": list(risk_type),
        "evidence": list(evidence),
        "expected_tools": list(tools),
        "applicable_policy": list(policy),
    }
    annotation = {"labelers": ["eval-phase2"], "agreed": True, "notes": note, "family": family}
    if hard and reason:
        annotation["hard_reason"] = list(reason)
    return {
        "eval_case_id": f"EC_V2_{seq:04d}",
        "schema_version": 2,
        "scene": scene,
        "source_type": src,
        "lineage": {"seed_case_id": seed_id, "mutation": mutation},
        "hard_case": hard,
        "input": input_json,
        "expected": expected,
        "annotation": annotation,
    }


# ---- normal ----------------------------------------------------------------


def b_clean_own(rng: random.Random, seq: int, scene: str) -> dict:
    cat = _pick(rng, list(_OWN_ANCHORS))
    a = _anchor(rng, cat, _CLEAN_OWN_ANCHORS[cat])
    core = _clean_text(_pick(rng, _TITLE_CORES[cat]))
    title = _title_with_brand(a["brand"], core, rng)
    img = _pick(rng, _IMG_CLEAN[cat])
    event = _pick(rng, _EVENT_POOL)
    return _row_base(
        seq=seq, pid=a["pid"], mid=a["mid"], cat=cat, brand=a["brand"],
        title=title, desc=_GEN_DESC[cat], images=_images((img, None)), event=event,
        scene="normal", family="clean_own", decision="PASS", abstain="AUTO_DECIDABLE",
        src="VARIANT", hard=False, reason=[], note=(
            f"干净自有品牌（{a['brand']}，在库可查；{a['mid']} {_MERCHANT_TIER[a['mid']]}"
            "商家；图无命中；文本无风险词）；Rule/Single/Agent 三方案一致 PASS。"),
        risk_level="NONE", risk_type=[], evidence=[], tools=[], policy=[],
        seed_id=_pick(rng, _V1_SEED["clean_own"][cat]),
        mutation="title词/事件变异（在库自有品牌 anchor 不变）；真值保持 PASS/AUTO",
    )


# ---- violation（全部 REJECT / AUTO）------------------------------------------


def v_text_evasion(rng: random.Random, seq: int, scene: str) -> dict:
    cat = _pick(rng, list(_DIRTY_ANCHORS))
    a = _anchor(rng, cat, _DIRTY_ANCHORS[cat])
    core = _clean_text(_pick(rng, _TITLE_CORES[cat]))
    phrase = _pick(rng, _EVASION_PHRASES)
    title = f"{phrase} {core}"
    desc = f"{_GEN_DESC[cat]}（{phrase}风格）。"
    strong = bool(_IMG_STRONG[cat]) and rng.random() < 0.8
    img = (_pick(rng, _IMG_STRONG[cat]) if strong
           else _pick(rng, _IMG_WEAK[cat] or _IMG_STRONG[cat]))
    ev = ["text_evasion_word", _merchant_evidence(a["mid"])]
    if strong:
        ev.append("image_similarity>=0.85")
    risk_types = ["POTENTIAL_IP_RISK"]
    if _MERCHANT_TIER[a["mid"]] == "dirty":
        risk_types.append("EVASION_PATTERN")
    return _row_base(
        seq=seq, pid=a["pid"], mid=a["mid"], cat=cat, brand=None,
        title=title, desc=desc, images=_images((img, None)), event="NEW_LISTING",
        scene="violation", family="text_evasion", decision="REJECT",
        abstain="AUTO_DECIDABLE", src="VARIANT", hard=True,
        reason=["rule_cannot_judge"],
        note=(f"文本明示规避词（{phrase}）+ brand 空缺 + 脏商家 {a['mid']} → REJECT。"
              "Rule: R-302/R-301 COMPLEX→HUMAN；Single: 文本自证 REJECT；Agent: REJECT。"),
        risk_level="HIGH", risk_type=risk_types, evidence=ev,
        tools=_TOOLS_ALL, policy=[_POLICY_BY_CAT[cat]],
        seed_id=_pick(rng, _V1_SEED["text_evasion"][cat]),
        mutation=f"标题加规避词 {phrase!r}（anchor/商家/图档位变异）；真值 REJECT/AUTO",
    )


def v_ocr_brand_dirty(rng: random.Random, seq: int, scene: str) -> dict:
    cat = _pick(rng, list(_DIRTY_ANCHORS))
    a = _anchor(rng, cat, _DIRTY_ANCHORS[cat])
    title = _plain_title(cat, rng)
    img_pool = _IMG_STRONG[cat] or _IMG_LOGO
    img = _pick(rng, img_pool)
    ocr = _pick(rng, _OCR_BRAND_WORDS)
    ev = [f"image_ocr_brand_word_{ocr}", _merchant_evidence(a["mid"])]
    if img in _IMG_LOGO:
        ev.append("image_logo_detected")
    else:
        ev.append("image_similarity>=0.85")
    return _row_base(
        seq=seq, pid=a["pid"], mid=a["mid"], cat=cat, brand=None,
        title=title, desc=_GEN_DESC[cat], images=_images((img, ocr)),
        event="NEW_LISTING", scene="violation", family="ocr_brand_dirty",
        decision="REJECT", abstain="AUTO_DECIDABLE", src="VARIANT", hard=True,
        reason=["agent_can_discover"],
        note=(f"标题干净但图片 OCR 含品牌词 {ocr} + 强视觉 + 脏商家 {a['mid']} —— OCR "
              "为表面信号、Rule/Single 看不到图证据 → HUMAN；Agent 图×商家交叉 REJECT "
              "（需调查才能判的 AUTO 案）。"),
        risk_level="HIGH", risk_type=["POTENTIAL_IP_RISK", "EVASION_PATTERN"], evidence=ev,
        tools=_TOOLS_ALL, policy=[_POLICY_BY_CAT[cat]],
        seed_id=_pick(rng, _V1_SEED["ocr_brand_dirty"].get(cat, ["EC_0102"])),
        mutation=f"图片 OCR 加品牌词 {ocr}（标题保持干净）；真值 REJECT/AUTO（需调查）",
    )


def v_blackbrand_field(rng: random.Random, seq: int, scene: str) -> dict:
    cat = _pick(rng, list(_DIRTY_ANCHORS))
    a = _anchor(rng, cat, _DIRTY_ANCHORS[cat])
    title = _plain_title(cat, rng)
    brand = _pick(rng, _BLACK_BRAND_BY_CAT[cat])
    img = _pick(rng, _IMG_STRONG[cat])
    return _row_base(
        seq=seq, pid=a["pid"], mid=a["mid"], cat=cat, brand=brand,
        title=title, desc=_GEN_DESC[cat], images=_images((img, None)),
        event="NEW_LISTING", scene="violation", family="blackbrand_field",
        decision="REJECT", abstain="AUTO_DECIDABLE", src="SYNTHETIC", hard=True,
        reason=["rule_cannot_judge"],
        note=(f"product.brand={brand}（黑名单 R-101 语义）+ 脏商家 + 强相似图；文本干净 "
              "→ 当前 Rule（BLACKLISTED_BRANDS=∅）直漏 PASS、Single 漏放 PASS（缺陷观测）；"
              "Agent 图×商家 REJECT；黑名单词表注入后 Rule 应 R-101 直判 REJECT。"),
        risk_level="HIGH", risk_type=["POTENTIAL_IP_RISK", "EVASION_PATTERN"],
        evidence=[_merchant_evidence(a["mid"]), "image_similarity>=0.85"],
        tools=_TOOLS_ALL, policy=[_POLICY_BY_CAT[cat]],
        seed_id=_pick(rng, _V1_SEED["blackbrand_field"].get(cat, ["EC_0102"])),
        mutation=f"case.brand→{brand}（黑名单字段语义）；真值 REJECT/AUTO",
    )


# ---- boundary AUTO PASS（需调查核验 / 弱信号 / 风格词观测）---------------------


def b_brand_missing_verify(rng: random.Random, seq: int, scene: str) -> dict:
    cat = _pick(rng, list(_OWN_ANCHORS))
    a = _anchor(rng, cat, _CLEAN_OWN_ANCHORS[cat])
    title = _plain_title(cat, rng)
    img = _pick(rng, _IMG_CLEAN[cat])
    return _row_base(
        seq=seq, pid=a["pid"], mid=a["mid"], cat=cat, brand=None,
        title=title, desc=_GEN_DESC[cat], images=_images((img, None)),
        event="NEW_LISTING", scene="boundary", family="brand_missing_verify",
        decision="PASS", abstain="AUTO_DECIDABLE", src="VARIANT", hard=True,
        reason=["agent_can_discover"],
        note=("brand 空缺（Rule R-301/Single 均→HUMAN）但在库自有品牌可查 + "
              f"{_MERCHANT_TIER[a['mid']]}商家 → Agent 核验后 PASS —— 需调查才能判的 "
              "AUTO 案（评测核心观察对象）。"),
        risk_level="NONE", risk_type=[], evidence=[], tools=[], policy=[],
        seed_id=_pick(rng, _V1_SEED["brand_missing_verify"].get(cat, ["EC_0201"])),
        mutation="case.brand→None（在库 brand 可查）；Rule/Single COMPLEX→HUMAN，Agent 核验 PASS —— 需调查 AUTO 案",
    )


def b_cat_missing_verify(rng: random.Random, seq: int, scene: str) -> dict:
    cat = _pick(rng, list(_OWN_ANCHORS))
    a = _anchor(rng, cat, _CLEAN_OWN_ANCHORS[cat])
    core = _clean_text(_pick(rng, _TITLE_CORES[cat]))
    title = _title_with_brand(a["brand"], core, rng)
    img = _pick(rng, _IMG_CLEAN[cat])
    return _row_base(
        seq=seq, pid=a["pid"], mid=a["mid"], cat="", brand=a["brand"],
        title=title, desc=_GEN_DESC[cat], images=_images((img, None)),
        event="NEW_LISTING", scene="boundary", family="cat_missing_verify",
        decision="PASS", abstain="AUTO_DECIDABLE", src="VARIANT", hard=True,
        reason=["agent_can_discover"],
        note=("category 空缺（Rule R-301/Single 均→HUMAN）但在库可核验 → Agent 核验 "
              "商品/商家后 PASS —— 需调查才能判的 AUTO 案。"),
        risk_level="NONE", risk_type=[], evidence=[], tools=[], policy=[],
        seed_id=_pick(rng, _V1_SEED["cat_missing_verify"].get(cat, ["EC_0204"])),
        mutation="case.category→空串（在库可查）；Rule/Single→HUMAN，Agent 核验 PASS —— 需调查 AUTO 案",
    )


def b_weak_sim_own(rng: random.Random, seq: int, scene: str) -> dict:
    cat = _pick(rng, ["女鞋/运动鞋", "箱包/女包"])  # 弱相似图只存在于鞋/包
    a = _anchor(rng, cat, _CLEAN_OWN_ANCHORS[cat])
    core = _clean_text(_pick(rng, _TITLE_CORES[cat]))
    title = _title_with_brand(a["brand"], core, rng)
    img = _pick(rng, _IMG_WEAK[cat])
    return _row_base(
        seq=seq, pid=a["pid"], mid=a["mid"], cat=cat, brand=a["brand"],
        title=title, desc=_GEN_DESC[cat], images=_images((img, None)),
        event="NEW_LISTING", scene="boundary", family="weak_sim_own",
        decision="PASS", abstain="AUTO_DECIDABLE", src="VARIANT", hard=False,
        reason=[], note=("单弱信号（相似 0.70~0.85）+ 自有品牌 + 干净/中性商家：有其它 "
                         "信息可证正常 → PASS（三方案一致）。"),
        risk_level="LOW", risk_type=[], evidence=["image_similarity 0.70~0.85"], tools=[],
        policy=[], seed_id=_pick(rng, _V1_SEED["weak_sim_own"][cat]),
        mutation="图片→弱相似(0.70~0.85)；自有品牌+干净商家 → PASS/AUTO（全部方案可判）",
    )


def b_neutral_clean(rng: random.Random, seq: int, scene: str) -> dict:
    a = dict(_NEUTRAL_ANCHOR)
    core = _clean_text(_pick(rng, _TITLE_CORES["服装/卫衣"]))
    title = _title_with_brand(a["brand"], core, rng)
    img = _pick(rng, _IMG_CLEAN["服装/卫衣"])
    return _row_base(
        seq=seq, pid=a["pid"], mid=a["mid"], cat="服装/卫衣", brand=a["brand"],
        title=title, desc=_GEN_DESC["服装/卫衣"], images=_images((img, None)),
        event="NEW_LISTING", scene="boundary", family="neutral_clean",
        decision="PASS", abstain="AUTO_DECIDABLE", src="VARIANT", hard=False,
        reason=[], note=("中性商家（1 次下架 < 系统性阈值）+ 自有品牌干净案 → PASS "
                         "（Agent：中性历史证伪'系统性'主张）。"),
        risk_level="NONE", risk_type=[], evidence=[], tools=[], policy=[],
        seed_id=_pick(rng, _V1_SEED["neutral_clean"]["服装/卫衣"]),
        mutation="anchor=M_6602 中性商家自有品牌（山野卫衣）；真值保持 PASS/AUTO",
    )


def b_styleword_own(rng: random.Random, seq: int, scene: str) -> dict:
    cat = _pick(rng, list(_STYLE_CORES))
    a = _anchor(rng, cat, _CLEAN_OWN_ANCHORS[cat])
    style_core = _pick(rng, _STYLE_CORES[cat])
    title = f"{a['brand']}{style_core}"
    img = _pick(rng, _IMG_CLEAN[cat])
    return _row_base(
        seq=seq, pid=a["pid"], mid=a["mid"], cat=cat, brand=a["brand"],
        title=title, desc=_GEN_DESC[cat], images=_images((img, None)),
        event="NEW_LISTING", scene="boundary", family="styleword_own",
        decision="PASS", abstain="AUTO_DECIDABLE", src="SYNTHETIC", hard=False,
        reason=[], note=("风格词标题 + 自有品牌 + 干净图：真值 PASS（在库可证伪、风格词非"
                         "违规证据）；Rule/Single 直判 PASS，Agent 因视觉未确证保守 HUMAN "
                         "—— AUTO 案上过度 abstention 观测案。"),
        risk_level="NONE", risk_type=[], evidence=[], tools=[], policy=[],
        seed_id=_SEED_V2["styleword_own"],
        mutation="标题加风格词（自有品牌/干净图不变）；真值 PASS/AUTO（Agent 过度 abstention 观测）",
    )


# ---- boundary SHOULD_ABSTAIN --------------------------------------------------


def b_adapter_brandword(rng: random.Random, seq: int, scene: str) -> dict:
    cat = _pick(rng, list(_ADAPTER_TITLES))
    a = _anchor(rng, cat, _CLEAN_OWN_ANCHORS[cat])
    title = _pick(rng, _ADAPTER_TITLES[cat])
    img = _pick(rng, _IMG_CLEAN[cat])
    word = next(w for w in _BRAND_WORDS if w in title)
    return _row_base(
        seq=seq, pid=a["pid"], mid=a["mid"], cat=cat, brand=a["brand"],
        title=title, desc=_GEN_DESC[cat], images=_images((img, None)),
        event="NEW_LISTING", scene="boundary", family="adapter_brandword",
        decision="HUMAN_REVIEW", abstain="SHOULD_ABSTAIN", src="SYNTHETIC",
        hard=False, reason=[], note=(f"文本含品牌词 {word} 但可能是适配/风格描述（无规避"
                                     "词、无授权信息可查）→ Rule R-102 / Single / Agent "
                                     "均无法核验授权与真伪 → SHOULD_ABSTAIN。"),
        risk_level="MEDIUM", risk_type=[], evidence=[f"text_brand_word_{word}(未核验授权)"],
        tools=_TOOLS_ABSTAIN, policy=[], seed_id=_SEED_V2["adapter_brandword"],
        mutation="标题加品牌词（适配/风格语境、无规避词）；真值 HUMAN/SHOULD_ABSTAIN",
    )


def b_brand_missing_unverifiable(rng: random.Random, seq: int, scene: str) -> dict:
    cat = _pick(rng, list(_TITLE_CORES))
    mid = _pick(rng, ["M_3307", "M_9904", "M_6602"])
    title = _plain_title(cat, rng)
    img = _pick(rng, _IMG_CLEAN[cat])
    pid = f"P_V2_UNVER_{1000 + seq}"
    return _row_base(
        seq=seq, pid=pid, mid=mid, cat=cat, brand=None,
        title=title, desc=_GEN_DESC[cat], images=_images((img, None)),
        event="NEW_LISTING", scene="boundary", family="brand_missing_unverifiable",
        decision="HUMAN_REVIEW", abstain="SHOULD_ABSTAIN", src="SYNTHETIC",
        hard=False, reason=[], note=("brand 空缺且商品不在库可查（虚构 pid）、商家干净/中性"
                                     "、图无命中 → 疑似规避但无确证 → Rule R-301 / Single "
                                     "/ Agent 一致转人工，不硬判 PASS。"),
        risk_level="LOW", risk_type=[], evidence=["brand_missing", _merchant_evidence(mid)],
        tools=_TOOLS_ABSTAIN, policy=[], seed_id=_SEED_V2["brand_missing_unverifiable"],
        mutation="case.brand→None + pid 虚构（在库不可查）；真值 HUMAN/SHOULD_ABSTAIN",
    )


def b_weak_sim_noinfo(rng: random.Random, seq: int, scene: str) -> dict:
    cat = _pick(rng, ["女鞋/运动鞋", "箱包/女包"])
    mid = _pick(rng, ["M_3307", "M_9904", "M_6602"])
    title = _plain_title(cat, rng)
    img = _pick(rng, _IMG_WEAK[cat])
    pid = f"P_V2_WEAK_{1000 + seq}"
    return _row_base(
        seq=seq, pid=pid, mid=mid, cat=cat, brand=None,
        title=title, desc=_GEN_DESC[cat], images=_images((img, None)),
        event="NEW_LISTING", scene="boundary", family="weak_sim_noinfo",
        decision="HUMAN_REVIEW", abstain="SHOULD_ABSTAIN", src="SYNTHETIC",
        hard=False, reason=[], note=("单弱信号（相似 0.70~0.85）且无其它可查信息（brand "
                                     "空缺、虚构 pid、商家干净/中性）→ Rule/Agent 无法确定"
                                     "是否模仿 → SHOULD_ABSTAIN。"),
        risk_level="LOW", risk_type=[], evidence=["image_similarity 0.70~0.85", "brand_missing"],
        tools=_TOOLS_ABSTAIN, policy=[], seed_id=_SEED_V2["weak_sim_noinfo"],
        mutation="单弱信号弱相似 + brand 空缺 + 无在库/商家信息；真值 HUMAN/SHOULD_ABSTAIN",
    )


# ---- multi-signal / evasion（REJECT AUTO 需调查 / SHOULD_ABSTAIN）----------------


def m_ssim_dirty(rng: random.Random, seq: int, scene: str) -> dict:
    cat = _pick(rng, list(_DIRTY_ANCHORS))
    a = _anchor(rng, cat, _DIRTY_ANCHORS[cat])
    core = _clean_text(_pick(rng, _TITLE_CORES[cat]))
    img = _pick(rng, _IMG_STRONG[cat])
    if scene == "evasion":
        title = f"{_pick(rng, ['复古', '潮流'])}{core}"
        event = _pick(rng, ["UPDATE_IMAGE", "UPDATE_TITLE", "NEW_LISTING"])
    else:
        title = f"{core} {_pick(rng, _CLEAN_SUFFIX)}".strip()
        event = _pick(rng, ["NEW_LISTING", "UPDATE_TITLE"])
    return _row_base(
        seq=seq, pid=a["pid"], mid=a["mid"], cat=cat, brand=None,
        title=title, desc=_GEN_DESC[cat], images=_images((img, None)), event=event,
        scene=scene, family="ssim_dirty", decision="REJECT", abstain="AUTO_DECIDABLE",
        src="VARIANT", hard=True, reason=["agent_can_discover"],
        note=(f"强相似图(>=0.85)+脏商家 {a['mid']}，无文本明示 → 需图×商家交叉；Rule/"
              "Single 无图证据 → HUMAN，Agent REJECT"
              + ("（风格词包装 + 改图/改标题事件掩盖的对抗形态）。" if scene == "evasion" else "。")),
        risk_level="HIGH", risk_type=["POTENTIAL_IP_RISK", "EVASION_PATTERN"],
        evidence=[_merchant_evidence(a["mid"]), "image_similarity>=0.85"],
        tools=_TOOLS_ALL, policy=[_POLICY_BY_CAT[cat]],
        seed_id=_pick(rng, _V1_SEED["ssim_dirty"][cat]),
        mutation="强相似(>=0.85)+脏商家（无文本明示）；真值 REJECT/AUTO（需调查）",
    )


def m_wsim_dirty(rng: random.Random, seq: int, scene: str) -> dict:
    cat = _pick(rng, ["女鞋/运动鞋", "箱包/女包"])
    a = _anchor(rng, cat, _DIRTY_ANCHORS[cat])
    title = _plain_title(cat, rng)
    img = _pick(rng, _IMG_WEAK[cat])
    return _row_base(
        seq=seq, pid=a["pid"], mid=a["mid"], cat=cat, brand=None,
        title=title, desc=_GEN_DESC[cat], images=_images((img, None)),
        event="NEW_LISTING", scene="multi-signal", family="wsim_dirty",
        decision="REJECT", abstain="AUTO_DECIDABLE", src="VARIANT", hard=True,
        reason=["agent_can_discover"],
        note=(f"弱相似(0.70~0.85)+脏商家 {a['mid']}：单弱信号不足、商家系统性历史交叉后 "
              "Agent REJECT；Rule/Single 看不到 → HUMAN（需调查才能判的 AUTO 案）。"),
        risk_level="HIGH", risk_type=["POTENTIAL_IP_RISK", "EVASION_PATTERN"],
        evidence=[_merchant_evidence(a["mid"]), "image_similarity 0.70~0.85"],
        tools=_TOOLS_ALL, policy=[_POLICY_BY_CAT[cat]],
        seed_id=_pick(rng, _V1_SEED["wsim_dirty"].get(cat, ["EC_0303"])),
        mutation="弱相似(0.70~0.85)+脏商家交叉；真值 REJECT/AUTO（需调查）",
    )


def m_own_adversarial(rng: random.Random, seq: int, scene: str) -> dict:
    a = dict(_ADV_ANCHOR)
    cat = "女鞋/运动鞋"
    core = _clean_text(_pick(rng, _TITLE_CORES[cat]))
    title = _title_with_brand(a["brand"], core, rng)
    img = _pick(rng, _IMG_STRONG[cat])
    return _row_base(
        seq=seq, pid=a["pid"], mid=a["mid"], cat=cat, brand=a["brand"],
        title=title, desc=_GEN_DESC[cat], images=_images((img, None)),
        event="NEW_LISTING", scene=scene, family="own_adversarial",
        decision="REJECT", abstain="AUTO_DECIDABLE", src="SYNTHETIC", hard=True,
        reason=["agent_can_discover"],
        note=("自有品牌(潮动)表面干净 + 强相似图 + 脏商家 M_5512 → Rule 直漏 PASS / "
              "Single 漏放 PASS（缺陷观测），Agent 图×商家 REJECT（对抗旗舰形态）。"),
        risk_level="HIGH", risk_type=["POTENTIAL_IP_RISK", "EVASION_PATTERN"],
        evidence=[_merchant_evidence(a["mid"]), "image_similarity>=0.85"],
        tools=_TOOLS_ALL, policy=[_POLICY_BY_CAT[cat]],
        seed_id=_pick(rng, _V1_SEED["own_adversarial"]["女鞋/运动鞋"]),
        mutation="自有品牌 anchor(潮动)+强相似+脏商家；真值 REJECT/AUTO",
    )


def m_logo_dirty(rng: random.Random, seq: int, scene: str) -> dict:
    a = _anchor(rng, "箱包/女包", _DIRTY_ANCHORS["箱包/女包"])
    title = _plain_title("箱包/女包", rng)
    img = _pick(rng, _IMG_LOGO)
    return _row_base(
        seq=seq, pid=a["pid"], mid=a["mid"], cat="箱包/女包", brand=None,
        title=title, desc=_GEN_DESC["箱包/女包"], images=_images((img, None)),
        event="NEW_LISTING", scene="multi-signal", family="logo_dirty",
        decision="REJECT", abstain="AUTO_DECIDABLE", src="VARIANT", hard=True,
        reason=["agent_can_discover"],
        note=(f"图片 Logo 检测命中(GUCCI 0.90) + 脏商家 {a['mid']}，无文本品牌词 → 仅工具"
              "可得 → Agent REJECT；Rule/Single HUMAN（需调查才能判的 AUTO 案）。"),
        risk_level="HIGH", risk_type=["POTENTIAL_IP_RISK", "EVASION_PATTERN"],
        evidence=[_merchant_evidence(a["mid"]), "image_logo_detected"],
        tools=_TOOLS_ALL, policy=[_POLICY_BY_CAT["箱包/女包"]],
        seed_id=_pick(rng, _V1_SEED["logo_dirty"]["箱包/女包"]),
        mutation="图片→Logo 命中(GUCCI 0.90)+脏商家；真值 REJECT/AUTO（需调查）",
    )


def m_pair_hide(rng: random.Random, seq: int, scene: str) -> dict:
    cat = _pick(rng, list(_DIRTY_ANCHORS))
    a = _anchor(rng, cat, _DIRTY_ANCHORS[cat])
    title = _plain_title(cat, rng)
    main = _pick(rng, _IMG_CLEAN[cat])
    alt = _pick(rng, _IMG_STRONG[cat])
    return _row_base(
        seq=seq, pid=a["pid"], mid=a["mid"], cat=cat, brand=None,
        title=title, desc=_GEN_DESC[cat], images=_images((main, None), (alt, None)),
        event="UPDATE_IMAGE", scene=scene, family="pair_hide",
        decision="REJECT", abstain="AUTO_DECIDABLE", src="VARIANT", hard=True,
        reason=["agent_can_discover"],
        note=(f"双图：主图干净 + 附图强相似 + 脏商家 {a['mid']}（改图事件规避抽查）→ "
              "Agent 对全图取 max 相似 REJECT；Rule/Single 无图比对能力 → HUMAN。"),
        risk_level="HIGH", risk_type=["POTENTIAL_IP_RISK", "EVASION_PATTERN"],
        evidence=[_merchant_evidence(a["mid"]), "image_similarity>=0.85"],
        tools=_TOOLS_ALL, policy=[_POLICY_BY_CAT[cat]],
        seed_id=_pick(rng, _V1_SEED["pair_hide"][cat]),
        mutation="主图干净+附图强相似（改图事件）；真值 REJECT/AUTO（需调查）",
    )


def e_dirty_brand_missing_cleanimg(rng: random.Random, seq: int, scene: str) -> dict:
    cat = _pick(rng, list(_DIRTY_ANCHORS))
    a = _anchor(rng, cat, _DIRTY_ANCHORS[cat])
    title = _plain_title(cat, rng)
    img = _pick(rng, _IMG_CLEAN[cat])
    event = _pick(rng, ["UPDATE_IMAGE", "UPDATE_TITLE", "NEW_LISTING"])
    return _row_base(
        seq=seq, pid=a["pid"], mid=a["mid"], cat=cat, brand=None,
        title=title, desc=_GEN_DESC[cat], images=_images((img, None)), event=event,
        scene="evasion", family="dirty_brand_missing_cleanimg",
        decision="HUMAN_REVIEW", abstain="SHOULD_ABSTAIN", src="SYNTHETIC",
        hard=False, reason=[], note=(f"疑似规避（brand 空缺+脏商家 {a['mid']} + {event}）"
                                     f"但图/文本无确证（查不实）→ Rule R-301 / Single / "
                                     "Agent 均克制转人工，不硬判。"),
        risk_level="MEDIUM", risk_type=[],
        evidence=["brand_missing", _merchant_evidence(a["mid"])],
        tools=_TOOLS_ABSTAIN, policy=[], seed_id=_SEED_V2["dirty_brand_missing_cleanimg"],
        mutation="brand 空缺+脏商家+规避事件但图/文本无确证；真值 HUMAN/SHOULD_ABSTAIN",
    )


def e_ssim_cleanmerchant_bm(rng: random.Random, seq: int, scene: str) -> dict:
    cat = _pick(rng, list(_DIRTY_ANCHORS))
    mid = _pick(rng, ["M_3307", "M_9904"])
    title = _plain_title(cat, rng)
    img = _pick(rng, _IMG_STRONG[cat])
    pid = f"P_V2_SSIMCLEAN_{1000 + seq}"
    return _row_base(
        seq=seq, pid=pid, mid=mid, cat=cat, brand=None,
        title=title, desc=_GEN_DESC[cat], images=_images((img, None)),
        event="NEW_LISTING", scene="evasion", family="ssim_cleanmerchant_bm",
        decision="HUMAN_REVIEW", abstain="SHOULD_ABSTAIN", src="SYNTHETIC",
        hard=False, reason=[], note=("强相似图 + 干净商家 + brand 空缺（虚构 pid 在库查无）"
                                     "：视觉强但无商家/在库佐证 → Rule R-301 / Single / "
                                     "Agent 一致克制转人工（无授权证据时外观高度模仿需人工审核）。"),
        risk_level="MEDIUM", risk_type=[],
        evidence=["image_similarity>=0.85", "brand_missing", _merchant_evidence(mid)],
        tools=_TOOLS_ABSTAIN, policy=[], seed_id=_SEED_V2["ssim_cleanmerchant_bm"],
        mutation="强相似+干净商家+brand 空缺（虚构 pid）；真值 HUMAN/SHOULD_ABSTAIN",
    )


def m_multi_weak_abstain(rng: random.Random, seq: int, scene: str) -> dict:
    cat = _pick(rng, ["女鞋/运动鞋", "箱包/女包"])
    mid = _pick(rng, ["M_6602", "M_3307"])
    title = _plain_title(cat, rng)
    img = _pick(rng, _IMG_WEAK[cat])
    pid = f"P_V2_MULTIWEAK_{1000 + seq}"
    return _row_base(
        seq=seq, pid=pid, mid=mid, cat=cat, brand=None,
        title=title, desc=_GEN_DESC[cat], images=_images((img, None)),
        event="NEW_LISTING", scene="multi-signal", family="multi_weak_abstain",
        decision="HUMAN_REVIEW", abstain="SHOULD_ABSTAIN", src="SYNTHETIC",
        hard=False, reason=[], note=("多弱信号（弱相似 + 中性/干净商家 + brand 空缺虚构 "
                                     "pid）交叉后仍无定论 → Rule R-301 / Single / Agent "
                                     "一致克制转人工。"),
        risk_level="LOW", risk_type=[],
        evidence=["image_similarity 0.70~0.85", "brand_missing", _merchant_evidence(mid)],
        tools=_TOOLS_ABSTAIN, policy=[], seed_id=_SEED_V2["multi_weak_abstain"],
        mutation="多弱信号（弱相似+中性商家+brand 空缺）交叉无定论；真值 HUMAN/SHOULD_ABSTAIN",
    )


# 每个 scene 的 (builder, 权重)。权重即"320 条设计下的目标条数"（各列和=该 scene 配额）；
# 其它 count 时用最大余数法按比例分配。abstain builder 的条数合计必须 ≤ 该 scene 配额。
_BUILDERS_BY_SCENE: dict[str, list[tuple[str, int]]] = {
    "normal": [("clean_own", 64)],
    "violation": [("text_evasion", 38), ("ocr_brand_dirty", 14), ("blackbrand_field", 12)],
    "boundary": [
        ("brand_missing_verify", 22), ("cat_missing_verify", 8), ("weak_sim_own", 24),
        ("neutral_clean", 6), ("styleword_own", 10),
        ("adapter_brandword", 10), ("brand_missing_unverifiable", 8), ("weak_sim_noinfo", 8),
    ],
    "multi-signal": [
        ("ssim_dirty", 24), ("wsim_dirty", 14), ("own_adversarial", 6),
        ("logo_dirty", 6), ("pair_hide", 6), ("multi_weak_abstain", 8),
    ],
    "evasion": [
        ("ssim_dirty", 12), ("own_adversarial", 4), ("pair_hide", 4),
        ("dirty_brand_missing_cleanimg", 8), ("ssim_cleanmerchant_bm", 4),
    ],
}
# abstain builder（decision=HUMAN_REVIEW / abstain_label=SHOULD_ABSTAIN）
_ABSTAIN_BUILDERS = {
    "adapter_brandword", "brand_missing_unverifiable", "weak_sim_noinfo",
    "multi_weak_abstain", "dirty_brand_missing_cleanimg", "ssim_cleanmerchant_bm",
}
_BUILD_FN = {
    "clean_own": b_clean_own,
    "text_evasion": v_text_evasion,
    "ocr_brand_dirty": v_ocr_brand_dirty,
    "blackbrand_field": v_blackbrand_field,
    "brand_missing_verify": b_brand_missing_verify,
    "cat_missing_verify": b_cat_missing_verify,
    "weak_sim_own": b_weak_sim_own,
    "neutral_clean": b_neutral_clean,
    "styleword_own": b_styleword_own,
    "adapter_brandword": b_adapter_brandword,
    "brand_missing_unverifiable": b_brand_missing_unverifiable,
    "weak_sim_noinfo": b_weak_sim_noinfo,
    "ssim_dirty": m_ssim_dirty,
    "wsim_dirty": m_wsim_dirty,
    "own_adversarial": m_own_adversarial,
    "logo_dirty": m_logo_dirty,
    "pair_hide": m_pair_hide,
    "dirty_brand_missing_cleanimg": e_dirty_brand_missing_cleanimg,
    "ssim_cleanmerchant_bm": e_ssim_cleanmerchant_bm,
    "multi_weak_abstain": m_multi_weak_abstain,
}


def _largest_remainder(total: int, weights: list[int]) -> list[int]:
    """整数配额分配（确定性；余数按小数部分降序，同余按序取）。"""
    if total <= 0:
        return [0] * len(weights)
    wsum = sum(weights)
    raw = [total * w / wsum for w in weights]
    base = [int(x) for x in raw]
    rem = total - sum(base)
    order = sorted(range(len(weights)), key=lambda i: (raw[i] - base[i], -i), reverse=True)
    for i in order[:rem]:
        base[i] += 1
    return base


def _scene_quota(total: int) -> dict[str, int]:
    """按占比切 scene 配额（evasion 吸收取整尾差，总数恒等于 total）。"""
    n_norm = round(total * 0.20)
    n_viol = round(total * 0.20)
    n_bdy = round(total * 0.30)
    n_msi = round(total * 0.20)
    n_eva = total - n_norm - n_viol - n_bdy - n_msi
    return {"normal": n_norm, "violation": n_viol, "boundary": n_bdy,
            "multi-signal": n_msi, "evasion": n_eva}


def generate(count: int, seed: int) -> list[dict]:
    """确定性生成 count 条 v2 case（scene 配额 + abstain 配额 + builder 权重分配）。

    返回行序 = 按 scene 轮转交错（normal→violation→boundary→multi-signal→evasion
    循环取值），保证文件前缀即五类混合（smoke 子集取前 N 条也有 scene 覆盖）。
    """
    if count < 60:
        raise ValueError(f"count 过小（{count}），无法满足五类分布容差与 SHOULD_ABSTAIN 统计意义")
    rng = random.Random(seed)
    scene_q = _scene_quota(count)

    per_scene_builders: dict[str, list[tuple[str, int]]] = {}
    for scene, builders in _BUILDERS_BY_SCENE.items():
        total_n = scene_q[scene]
        weights = [w for _, w in builders]
        kinds = [k for k, _ in builders]
        allocated = dict(zip(kinds, _largest_remainder(total_n, weights)))
        # 非 320 条时，abstain builder 的条数可能偏大 → 削到每 scene 预留至少 1 条 AUTO
        abstain_n = sum(c for k, c in allocated.items() if k in _ABSTAIN_BUILDERS)
        if abstain_n > 0 and abstain_n > total_n - 1:
            deficit = abstain_n - (total_n - 1)
            for k in kinds:
                if k in _ABSTAIN_BUILDERS and deficit > 0 and allocated[k] > 0:
                    cut = min(allocated[k], deficit)
                    allocated[k] -= cut
                    deficit -= cut
        per_scene_builders[scene] = [(k, allocated[k]) for k in kinds if allocated[k] > 0]

    # 逐 scene 生成（seq 全局递增 → pid/时间戳全局唯一）
    scene_rows: dict[str, list[dict]] = {s: [] for s in _SCENES}
    seq = 1
    for scene in _SCENES:
        for kind, n in per_scene_builders[scene]:
            fn = _BUILD_FN[kind]
            for _ in range(n):
                try:
                    row = fn(rng, seq, scene)
                except ValueError as exc:  # 文本越界等生成失败 → 显式报错（确定性 bug 不静默）
                    raise ValueError(
                        f"builder={kind} scene={scene} seq={seq} 生成失败: {exc}"
                    ) from exc
                if row["expected"]["decision"] == "HUMAN_REVIEW":
                    assert row["expected"]["abstain_label"] == "SHOULD_ABSTAIN", kind
                else:
                    assert row["expected"]["abstain_label"] == "AUTO_DECIDABLE", kind
                scene_rows[scene].append(row)
                seq += 1
        if len(scene_rows[scene]) != scene_q[scene]:
            raise AssertionError(
                f"scene={scene} 生成 {len(scene_rows[scene])} 条 ≠ 配额 {scene_q[scene]}"
            )

    # scene 轮转交错（保证 smoke 前缀有 scene 覆盖）
    final: list[dict] = []
    max_len = max(len(v) for v in scene_rows.values())
    for i in range(max_len):
        for s in _SCENES:
            if i < len(scene_rows[s]):
                final.append(scene_rows[s][i])
    for i, row in enumerate(final, start=1):  # eval_case_id 按最终文件序重排
        row["eval_case_id"] = f"EC_V2_{i:04d}"
        row["input"]["case_id"] = f"CASE_EC_V2_{i:04d}"
    return final


def _write_dataset(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cases_v2.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )


def _build_manifest(rows: list[dict], out_dir: Path, count: int, seed: int) -> dict:
    """manifest：分布统计（loader 同口径取数）+ 口径快照 + 生成命令与 seed。"""
    from pra.evaluation.dataset.schema import EvalCase

    cases = [EvalCase.model_validate(r) for r in rows]  # loader 同款强校验（schema v2）
    ss = scene_stats(cases)
    aa = abstain_stats(cases)
    dec_dist: dict = {}
    for c in cases:
        dec_dist[c.expected.decision] = dec_dist.get(c.expected.decision, 0) + 1
    hard_n = sum(1 for c in cases if c.hard_case)
    src_dist: dict = {}
    for c in cases:
        src_dist[c.source_type] = src_dist.get(c.source_type, 0) + 1
    return {
        "schema_version": 2,
        "dataset_version": "v2",
        "file": "cases_v2.jsonl",
        "generator": "scripts/eval_dataset_gen.py",
        "generator_command": (
            f"uv run python scripts/eval_dataset_gen.py --out {out_dir} --count {count} --seed {seed}"
        ),
        "seed": seed,
        "total": len(cases),
        "scene_distribution": {s: ss["by_scene"][s]["total"] for s in _SCENES},
        "expected_decision_distribution": dec_dist,
        "abstain_distribution": {
            "AUTO_DECIDABLE": aa["AUTO_DECIDABLE"],
            "SHOULD_ABSTAIN": aa["SHOULD_ABSTAIN"],
            "LEGACY_UNLABELED": aa["LEGACY_UNLABELED"],
            "auto_decidable_equivalent": aa["auto_decidable_equivalent"],
            "auto_share": aa["auto_share"],
            "abstain_share": aa["abstain_share"],
        },
        "abstain_by_scene": {
            s: {k: aa["by_scene"][s][k] for k in ("total", "AUTO_DECIDABLE", "SHOULD_ABSTAIN")}
            for s in _SCENES
        },
        "hard_case": {"count": hard_n, "share": round(hard_n / len(cases), 2)},
        "source_type_distribution": src_dist,
        "world": {
            "label": EVAL_WORLD_LABEL,
            "products": len(EVAL_PRODUCTS),
            "merchants": len(EVAL_MERCHANTS),
            "image_urls": len(EVAL_IMAGE_MATCHES),
            "precedents": len(EVAL_PRECEDENTS),
            "policy_clauses": len(EVAL_POLICY_CLAUSES),
        },
        "threshold_snapshot": {
            "EVIDENCE_MIN_SIM": 0.7,
            "EVIDENCE_STRONG": 0.85,
            "CONFIDENCE_ABSTAIN_THRESHOLD": 0.7,
        },
        "annotation_notes": (
            "Phase 2 正式集：以 v1 35 条为模板的程序化确定性变异（seed 固定、可重放）；"
            "真值三值语义与 abstention 口径见 scripts/eval_dataset_gen.py 模块 docstring："
            "明确违规→REJECT/AUTO，明确正常→PASS/AUTO，单弱信号/brand·类目"
            "空缺/品牌词可能为适配描述→HUMAN_REVIEW/SHOULD_ABSTAIN；'需调查才能判'的 AUTO "
            "案（Rule/Single 会 COMPLEX/HUMAN、Agent 经 EVAL_* 世界工具可判对）在 "
            "annotation.notes 注明设计意图；标注阈值口径与 EvalContext 运行时一致"
            "（EVIDENCE_MIN_SIM=0.7 / EVIDENCE_STRONG=0.85 / CONFIDENCE_ABSTAIN_THRESHOLD=0.7）。"
        ),
        "lineage_notes": "每条 lineage 记录 seed_case_id（v1 老案 EC_xxxx 或 SEED_V2_* 语义种子）与 mutation 摘要。",
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 2 eval_data v2 确定性变异生成器")
    parser.add_argument("--out", default="eval_data/v2", help="输出目录（含 manifest.json）")
    parser.add_argument("--count", type=int, default=320, help="目标条数（默认 320）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    args = parser.parse_args(argv)

    rows = generate(count=args.count, seed=args.seed)
    out_dir = Path(args.out)
    _write_dataset(rows, out_dir)
    manifest = _build_manifest(rows, out_dir, args.count, args.seed)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"写入 {out_dir / 'cases_v2.jsonl'}：{len(rows)} 条（seed={args.seed}）")
    print(json.dumps({
        "scene": manifest["scene_distribution"],
        "decision": manifest["expected_decision_distribution"],
        "abstain": {k: manifest["abstain_distribution"][k]
                    for k in ("AUTO_DECIDABLE", "SHOULD_ABSTAIN", "auto_share", "abstain_share")},
        "hard": manifest["hard_case"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
