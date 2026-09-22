"""Phase 2 正式集（v2）确定性变异生成器。

以旧手工种子案（EC_*，35 条，已随 eval_data/v1 退役） + 少量新增语义种子为模板做确定性字段变异（固定随机种子），
产出 ``eval_data/v2/cases_v2.jsonl`` + ``manifest.json``；命令
``uv run python scripts/eval_dataset_gen.py --out eval_data/v2/ --count 320 --seed 42``。
不联网、不调 LLM；同 (count, seed) → 产物逐字节一致（可重放，测试守住）。

变异维度（**变异后真值必须仍自洽** —— expected 的 decision/abstain_label 按该案在
Rule / Agent 下的真实语义重标，非机械复制模板真值）：
- 标题词：换品牌 / 换品类核心词 / 加规避词 / 加品牌词 / 加风格词，严格受控词表（避免干净案
  变异出意外词命中，也避免风险案变异成表面干净）；
- 图片相似度：世界种子只有 强 0.85+ / 弱 0.70~0.85 / 干净 / Logo 四档，按 URL 选择；
- 商家历史：removals 0~7（M_3307/M_9904 干净、M_6602 中性 1、M_5512/M_8801 脏）；
- OCR 文本：含 / 不含品牌词（表面信号）；
- category / brand：明确 / 空缺 / 黑名单字段 / 高危或普通类目。

真值标注规则：明确违规（文本明示 高仿/复刻/1:1/同款/原单、brand 命中知名黑名单、强相似且
无授权证据）→ REJECT + AUTO_DECIDABLE；明确正常（brand 明确、无风险词、无规避、无强相似）
→ PASS + AUTO_DECIDABLE；边界（单弱信号 0.7~0.8 且无其它可查信息、brand/category 空缺、
文本含品牌词但可能是适配/风格描述）→ HUMAN_REVIEW + SHOULD_ABSTAIN；特例（有意构造）是
**需调查才能判的 AUTO_DECIDABLE 案** —— Rule 因缺工具会 COMPLEX/HUMAN（brand 空缺、
仅 OCR 品牌词、需商家历史交叉），但 Agent 用评测世界工具（EVAL_* 种子）能查到证据判对。

目标分布（容差 ±5pp）：normal 20% / violation 20% / boundary 30% / multi-signal 20% /
evasion 10%；AUTO_DECIDABLE 为主（~85-90%），SHOULD_ABSTAIN ~10-15%，集中在
boundary/evasion/multi-signal，使 SHOULD_ABSTAIN 在各 scene 都有分布（全量与 AUTO_DECIDABLE
两套分母可分层对比）。

去重与 ``expected_tools`` 语义：**可见输入不出现重复案** —— 标题词池相对 (类目, 品牌, 商家,
事件, 图片) 组合偏小，两次抽样可能命中同一组合（只差自增的 version / listing_time），这类行
由 ``_dedupe_visible_rows`` 换用未占用的干净后缀，避免同一条案在指标里重复计权。
``expected_tools`` **空列表 = 未标注该案的调查工具期望**（干净案、两臂一致 PASS；不计入
Tool Selection 指标分母），**不得解读为「应调用 0 个工具」**；「需调查才能判」的 AUTO 案必须
给非空期望（brand / category 空缺核验 = ProductTool + MerchantTool）。

世界事实锚点与 ``pra.evaluation.harness.agent_scheme`` 的 EVAL_* 种子同一份（生成器 import
该常量集），保证每条 input 的 product_id / merchant_id / image URL / category 在 Agent 评测
世界里可查到与标注一致的事实；真 RAG/商家库接入需扩 EVAL_* 种子并重新生成（manifest 记录
生成命令与 seed，防口径漂移）。每条 JSONL 行 = EvalCase schema（schema_version=2，含 lineage
溯源锚点 seed_case_id）。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from pra.evaluation.dataset.schema import EvalCase

# 世界事实单一来源：与 B 面 agent_scheme（Agent 评测世界）同一份 EVAL_* 种子。
# import 失败 → 生成器显式失败，拒绝「手抄世界」漂移。
from pra.evaluation.harness.agent_scheme import (
    EVAL_CATEGORIES,
    EVAL_IMAGE_MATCHES,
    EVAL_MERCHANTS,
    EVAL_PRODUCTS,
)

# 品牌词表单一来源：blackbrand_field 家族的 brand 字段必须取真实黑名单成员（命中 R-101 直判
# REJECT），且生成器的干净文本守卫必须取同一份品牌词/规避词（命中 R-102 → COMPLEX），均不得
# 手抄第二份 —— 否则词表语义漂移且逐字节锁抓不到语义错误。
from pra.screening.rule_engine.terms import (
    BLACKLISTED_BRANDS,
    BRAND_TERMS,
    EVASION_TERMS,
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
# 档位随图片语义固定（url 是中性资源 ID，不含档位语义），故按资源 ID 显式登记 ——
# 评测输入 URL 一旦暴露 clean/viol/bound 字样就等于把真值直接喂给模型。
_IMG_BUCKET: dict[str, str] = {
    "P_88231": "strong:女鞋/运动鞋",
    "asset-1001": "strong:女鞋/运动鞋",
    "asset-1002": "strong:箱包/女包",
    "asset-1003": "strong:服装/卫衣",
    "asset-1004": "weak:女鞋/运动鞋",
    "asset-1005": "weak:箱包/女包",
    "asset-1006": "logo",
    "asset-1007": "clean:女鞋/运动鞋",
    "asset-1008": "clean:女鞋/运动鞋",
    "asset-1009": "clean:箱包/女包",
    "asset-1010": "clean:服装/卫衣",
}
_IMG_STRONG: dict[str, list[str]] = {c: [] for c in EVAL_CATEGORIES}
_IMG_WEAK: dict[str, list[str]] = {c: [] for c in EVAL_CATEGORIES}
_IMG_CLEAN: dict[str, list[str]] = {c: [] for c in EVAL_CATEGORIES}
_IMG_LOGO: list[str] = []
for url in EVAL_IMAGE_MATCHES:
    bucket = _IMG_BUCKET.get(url.rsplit("/", 2)[-2])
    if bucket is None:
        raise ValueError(f"图片世界种子有未登记档位的资源: {url}")
    kind, _, cat = bucket.partition(":")
    if kind == "logo":
        _IMG_LOGO.append(url)
    elif kind == "strong":
        _IMG_STRONG[cat].append(url)
    elif kind == "weak":
        _IMG_WEAK[cat].append(url)
    elif kind == "clean":
        _IMG_CLEAN[cat].append(url)
    else:
        raise ValueError(f"未知图片档位 {bucket!r}（资源 {url}）")

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

# ---------------------------------------------------------------------------
# 受控词表（与 screening terms / agent 表面信号同语义；此处只管生成文本不出界）
# ---------------------------------------------------------------------------
# 干净文本守卫用的品牌词/规避词直接绑定 terms 单一来源（本地名保留以兼容既有用法）：在
# terms 里新增一个词即自动进入守卫，不会出现「生成器仍按旧词表放行」的静默失配。
_BRAND_WORDS: frozenset[str] = BRAND_TERMS
_EVASION_WORDS: frozenset[str] = EVASION_TERMS
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
# blackbrand_field 家族的 brand 字段候选 = **terms.BLACKLISTED_BRANDS 成员**（命中 R-101 直判
# REJECT 的黑名单语义）；与 BRAND_TERMS（命中 R-102 → COMPLEX 交 Agent 调查）严格区分，两者
# 不得混用。sorted 固定顺序 —— set 迭代序不稳定，直接迭代会破坏「同 (count, seed) 逐字节一致」。
_BLACK_BRANDS_ORDERED = sorted(BLACKLISTED_BRANDS)
_BLACK_BRAND_BY_CAT = {cat: _BLACK_BRANDS_ORDERED for cat in EVAL_CATEGORIES}

# 溯源种子：v1 老案按（形态, cat）映射；无直接 v1 模板的用 SEED_V2_ 语义种子标记。
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
# Builder 语义口径 —— 每个形态一个函数：产出 Row（input 字段 + 真值标签）。
# 签名统一 (rng, seq, scene)；只在单个 scene 用的 builder 忽略 scene 参数。
# ---------------------------------------------------------------------------


def _row_base(*, seq, pid, mid, cat, brand, title, desc, images, event,
              scene, decision, abstain,
              risk_level, risk_type, evidence, tools, seed_id) -> dict:
    """公共装配：input JSON + expected + lineage。"""
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
    }
    expected = {
        "decision": decision,
        "abstain_label": abstain,
        "risk_level": risk_level,
        "risk_type": list(risk_type),
        "evidence": list(evidence),
        "expected_tools": list(tools),
    }
    return {
        "eval_case_id": f"EC_V2_{seq:04d}",
        "schema_version": 2,
        "scene": scene,
        "lineage": {"seed_case_id": seed_id},
        "input": input_json,
        "expected": expected,
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
        scene="normal", decision="PASS", abstain="AUTO_DECIDABLE",
        risk_level="NONE", risk_type=[], evidence=[], tools=[],
        seed_id=_pick(rng, _V1_SEED["clean_own"][cat]),
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
        scene="violation", decision="REJECT",
        abstain="AUTO_DECIDABLE",
        risk_level="HIGH", risk_type=risk_types, evidence=ev,
        tools=_TOOLS_ALL,
        seed_id=_pick(rng, _V1_SEED["text_evasion"][cat]),
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
        event="NEW_LISTING", scene="violation",
        decision="REJECT", abstain="AUTO_DECIDABLE",
        risk_level="HIGH", risk_type=["POTENTIAL_IP_RISK", "EVASION_PATTERN"], evidence=ev,
        tools=_TOOLS_ALL,
        seed_id=_pick(rng, _V1_SEED["ocr_brand_dirty"].get(cat, ["EC_0102"])),
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
        event="NEW_LISTING", scene="violation",
        decision="REJECT", abstain="AUTO_DECIDABLE",
        risk_level="HIGH", risk_type=["POTENTIAL_IP_RISK", "EVASION_PATTERN"],
        evidence=[_merchant_evidence(a["mid"]), "image_similarity>=0.85"],
        tools=_TOOLS_ALL,
        seed_id=_pick(rng, _V1_SEED["blackbrand_field"].get(cat, ["EC_0102"])),
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
        event="NEW_LISTING", scene="boundary",
        decision="PASS", abstain="AUTO_DECIDABLE",
        risk_level="NONE", risk_type=[], evidence=[], tools=["ProductTool", "MerchantTool"],
        seed_id=_pick(rng, _V1_SEED["brand_missing_verify"].get(cat, ["EC_0201"])),
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
        event="NEW_LISTING", scene="boundary",
        decision="PASS", abstain="AUTO_DECIDABLE",
        risk_level="NONE", risk_type=[], evidence=[], tools=["ProductTool", "MerchantTool"],
        seed_id=_pick(rng, _V1_SEED["cat_missing_verify"].get(cat, ["EC_0204"])),
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
        event="NEW_LISTING", scene="boundary",
        decision="PASS", abstain="AUTO_DECIDABLE",
        risk_level="LOW", risk_type=[], evidence=["image_similarity 0.70~0.85"], tools=[],
        seed_id=_pick(rng, _V1_SEED["weak_sim_own"][cat]),
    )


def b_neutral_clean(rng: random.Random, seq: int, scene: str) -> dict:
    a = dict(_NEUTRAL_ANCHOR)
    core = _clean_text(_pick(rng, _TITLE_CORES["服装/卫衣"]))
    title = _title_with_brand(a["brand"], core, rng)
    img = _pick(rng, _IMG_CLEAN["服装/卫衣"])
    return _row_base(
        seq=seq, pid=a["pid"], mid=a["mid"], cat="服装/卫衣", brand=a["brand"],
        title=title, desc=_GEN_DESC["服装/卫衣"], images=_images((img, None)),
        event="NEW_LISTING", scene="boundary",
        decision="PASS", abstain="AUTO_DECIDABLE",
        risk_level="NONE", risk_type=[], evidence=[], tools=[],
        seed_id=_pick(rng, _V1_SEED["neutral_clean"]["服装/卫衣"]),
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
        event="NEW_LISTING", scene="boundary",
        decision="PASS", abstain="AUTO_DECIDABLE",
        risk_level="NONE", risk_type=[], evidence=[], tools=[],
        seed_id=_SEED_V2["styleword_own"],
    )


# ---- boundary SHOULD_ABSTAIN --------------------------------------------------


def b_adapter_brandword(rng: random.Random, seq: int, scene: str) -> dict:
    cat = _pick(rng, list(_ADAPTER_TITLES))
    a = _anchor(rng, cat, _CLEAN_OWN_ANCHORS[cat])
    title = _pick(rng, _ADAPTER_TITLES[cat])
    img = _pick(rng, _IMG_CLEAN[cat])
    word = min(w for w in _BRAND_WORDS if w in title)
    return _row_base(
        seq=seq, pid=a["pid"], mid=a["mid"], cat=cat, brand=a["brand"],
        title=title, desc=_GEN_DESC[cat], images=_images((img, None)),
        event="NEW_LISTING", scene="boundary",
        decision="HUMAN_REVIEW", abstain="SHOULD_ABSTAIN",
        risk_level="MEDIUM", risk_type=[], evidence=[f"text_brand_word_{word}(未核验授权)"],
        tools=_TOOLS_ABSTAIN, seed_id=_SEED_V2["adapter_brandword"],
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
        event="NEW_LISTING", scene="boundary",
        decision="HUMAN_REVIEW", abstain="SHOULD_ABSTAIN",
        risk_level="LOW", risk_type=[], evidence=["brand_missing", _merchant_evidence(mid)],
        tools=_TOOLS_ABSTAIN, seed_id=_SEED_V2["brand_missing_unverifiable"],
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
        event="NEW_LISTING", scene="boundary",
        decision="HUMAN_REVIEW", abstain="SHOULD_ABSTAIN",
        risk_level="LOW", risk_type=[], evidence=["image_similarity 0.70~0.85", "brand_missing"],
        tools=_TOOLS_ABSTAIN, seed_id=_SEED_V2["weak_sim_noinfo"],
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
        scene=scene, decision="REJECT", abstain="AUTO_DECIDABLE",
        risk_level="HIGH", risk_type=["POTENTIAL_IP_RISK", "EVASION_PATTERN"],
        evidence=[_merchant_evidence(a["mid"]), "image_similarity>=0.85"],
        tools=_TOOLS_ALL,
        seed_id=_pick(rng, _V1_SEED["ssim_dirty"][cat]),
    )


def m_wsim_dirty(rng: random.Random, seq: int, scene: str) -> dict:
    cat = _pick(rng, ["女鞋/运动鞋", "箱包/女包"])
    a = _anchor(rng, cat, _DIRTY_ANCHORS[cat])
    title = _plain_title(cat, rng)
    img = _pick(rng, _IMG_WEAK[cat])
    return _row_base(
        seq=seq, pid=a["pid"], mid=a["mid"], cat=cat, brand=None,
        title=title, desc=_GEN_DESC[cat], images=_images((img, None)),
        event="NEW_LISTING", scene="multi-signal",
        decision="REJECT", abstain="AUTO_DECIDABLE",
        risk_level="HIGH", risk_type=["POTENTIAL_IP_RISK", "EVASION_PATTERN"],
        evidence=[_merchant_evidence(a["mid"]), "image_similarity 0.70~0.85"],
        tools=_TOOLS_ALL,
        seed_id=_pick(rng, _V1_SEED["wsim_dirty"].get(cat, ["EC_0303"])),
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
        event="NEW_LISTING", scene=scene,
        decision="REJECT", abstain="AUTO_DECIDABLE",
        risk_level="HIGH", risk_type=["POTENTIAL_IP_RISK", "EVASION_PATTERN"],
        evidence=[_merchant_evidence(a["mid"]), "image_similarity>=0.85"],
        tools=_TOOLS_ALL,
        seed_id=_pick(rng, _V1_SEED["own_adversarial"]["女鞋/运动鞋"]),
    )


def m_logo_dirty(rng: random.Random, seq: int, scene: str) -> dict:
    a = _anchor(rng, "箱包/女包", _DIRTY_ANCHORS["箱包/女包"])
    title = _plain_title("箱包/女包", rng)
    img = _pick(rng, _IMG_LOGO)
    return _row_base(
        seq=seq, pid=a["pid"], mid=a["mid"], cat="箱包/女包", brand=None,
        title=title, desc=_GEN_DESC["箱包/女包"], images=_images((img, None)),
        event="NEW_LISTING", scene="multi-signal",
        decision="REJECT", abstain="AUTO_DECIDABLE",
        risk_level="HIGH", risk_type=["POTENTIAL_IP_RISK", "EVASION_PATTERN"],
        evidence=[_merchant_evidence(a["mid"]), "image_logo_detected"],
        tools=_TOOLS_ALL,
        seed_id=_pick(rng, _V1_SEED["logo_dirty"]["箱包/女包"]),
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
        event="UPDATE_IMAGE", scene=scene,
        decision="REJECT", abstain="AUTO_DECIDABLE",
        risk_level="HIGH", risk_type=["POTENTIAL_IP_RISK", "EVASION_PATTERN"],
        evidence=[_merchant_evidence(a["mid"]), "image_similarity>=0.85"],
        tools=_TOOLS_ALL,
        seed_id=_pick(rng, _V1_SEED["pair_hide"][cat]),
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
        scene="evasion",
        decision="HUMAN_REVIEW", abstain="SHOULD_ABSTAIN",
        risk_level="MEDIUM", risk_type=[],
        evidence=["brand_missing", _merchant_evidence(a["mid"])],
        tools=_TOOLS_ABSTAIN, seed_id=_SEED_V2["dirty_brand_missing_cleanimg"],
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
        event="NEW_LISTING", scene="evasion",
        decision="HUMAN_REVIEW", abstain="SHOULD_ABSTAIN",
        risk_level="MEDIUM", risk_type=[],
        evidence=["image_similarity>=0.85", "brand_missing", _merchant_evidence(mid)],
        tools=_TOOLS_ABSTAIN, seed_id=_SEED_V2["ssim_cleanmerchant_bm"],
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
        event="NEW_LISTING", scene="multi-signal",
        decision="HUMAN_REVIEW", abstain="SHOULD_ABSTAIN",
        risk_level="LOW", risk_type=[],
        evidence=["image_similarity 0.70~0.85", "brand_missing", _merchant_evidence(mid)],
        tools=_TOOLS_ABSTAIN, seed_id=_SEED_V2["multi_weak_abstain"],
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


def _visible_key(row: dict) -> str:
    """**内容**指纹（标题/描述/类目/品牌/属性/图片/商家/事件）—— 去重判定用。

    刻意不含 product_id / version / listing_time：前者是标识、后两者由 seq 自增，都不构成
    内容差异；只差它们的多行在指标里就是同一条案被重复计权（表观多样性虚高）。
    """
    p = row["input"]["product"]
    return json.dumps(
        {
            "t": p.get("title"),
            "d": p.get("description"),
            "c": p.get("category"),
            "b": p.get("brand"),
            "a": p.get("attributes"),
            "img": p.get("images"),
            "m": row["input"].get("merchant_id"),
            "e": row["input"].get("event_type"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _dedupe_visible_rows(rows: list[dict]) -> int:
    """改写重复行的标题后缀以消除「可见内容相同」的案（原地）；返回改写条数。

    只动后缀、不动核心词与品牌前缀规则 → 「干净标题」前提与该案真值不变。候选（各干净后缀）
    全部被占用时显式报错 —— 确定性 bug 不静默降级。
    """
    seen: set[str] = set()
    rewritten = 0
    for row in rows:
        if _visible_key(row) not in seen:
            seen.add(_visible_key(row))
            continue
        title = row["input"]["product"]["title"]
        stem = title
        for suffix in _CLEAN_SUFFIX:
            if suffix and title.endswith(" " + suffix):
                stem = title[: -(len(suffix) + 1)]
                break
        for cand in (f"{stem} {s}" for s in _CLEAN_SUFFIX if s):
            row["input"]["product"]["title"] = cand
            cand_key = _visible_key(row)
            if cand_key not in seen:
                seen.add(cand_key)
                break
        else:
            raise ValueError(f"可见内容去重失败：{row.get('eval_case_id')} 候选后缀已全部占用")
        rewritten += 1
    return rewritten


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
    _dedupe_visible_rows(final)  # 只差 version/listing_time 的重复行换干净后缀
    return final


def _write_dataset(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cases_v2.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )


def _build_manifest(rows: list[dict], out_dir: Path, count: int, seed: int) -> dict:
    """manifest 最小契约：数据文件标识 + 生成命令与 seed + 规模。"""
    cases = [EvalCase.model_validate(r) for r in rows]  # loader 同款强校验（schema v2）
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
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
