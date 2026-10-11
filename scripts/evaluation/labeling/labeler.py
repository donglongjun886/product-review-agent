"""独立打标函数：埋点清单 + 冻结规则书 → 真值（纯函数，无 IO 副作用）。

规则书冻结为同目录 ``rulebook.json`` 数据文件（词表/阈值自带副本，不随生产代码漂移）；
本包不 import 生成器与生产代码，由 ``tests/test_eval_labeling_independence.py`` 守住。

判定优先级（镜像生产决策语义）：
黑名单品牌 → 规避词（商家脏从重 / 干净普通） → 快照缺口与在库不一致 →
品牌词（叠加商家脏）→ 商家脏 → 干净放行。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from evaluation.dataset.schema import CaseManifestV3, EvalExpectedV3, TextHitV3

RULEBOOK_PATH = Path(__file__).with_name("rulebook.json")

__all__ = ["RULEBOOK_PATH", "label_case", "load_rulebook", "scan_text_hits"]

_ASCII_LATIN = re.compile(r"[a-z ]+\Z")


def load_rulebook(path: Path | None = None) -> dict:
    """读取冻结规则书数据文件。

    :param path: 缺省用包内 ``rulebook.json``。
    :return: 规则书 dict（词表 / 阈值 / 政策引用映射）。
    """
    return json.loads((path or RULEBOOK_PATH).read_text(encoding="utf-8"))


def _fold(text: str) -> str:
    """匹配归一化：转小写 + 全角冒号→半角。"""
    return (text or "").lower().replace("：", ":")


def _term_hit(folded_hay: str, term: str) -> bool:
    """单词命中判定：纯拉丁词按词边界，其余按子串。"""
    folded = _fold(term)
    if _ASCII_LATIN.fullmatch(folded):
        return re.search(rf"(?<![a-z0-9_]){re.escape(folded)}(?![a-z0-9_])", folded_hay) is not None
    return folded in folded_hay


def scan_text_hits(title: str, description: str, rulebook: dict) -> list[TextHitV3]:
    """按规则书词表扫描标题/描述，返回全部词表命中（快照自证用）。

    :return: 按（字段声明序、词表声明序）稳定排序的命中清单。
    """
    hits: list[TextHitV3] = []
    terms = list(rulebook["brand_terms"]) + list(rulebook["evasion_terms"])
    for field, text in (("title", title), ("description", description)):
        folded = _fold(text)
        for term in terms:
            if _term_hit(folded, term):
                hits.append(TextHitV3(field=field, term=term))
    return hits


def _merchant_dirty(manifest: CaseManifestV3, rulebook: dict) -> bool:
    """商家脏：removals 或 title_relisting_count 达到规则书阈值。"""
    threshold = int(rulebook["merchant_dirty_min"])
    m = manifest.merchant
    return m.removals >= threshold or m.title_relisting_count >= threshold


def _hit_kinds(manifest: CaseManifestV3, rulebook: dict) -> tuple[bool, bool]:
    """文本命中归类：返回 (是否命中规避词, 是否命中品牌词)。"""
    evasion = {str(t) for t in rulebook["evasion_terms"]}
    brand = {str(t) for t in rulebook["brand_terms"]}
    has_evasion = any(h.term in evasion for h in manifest.text_hits)
    has_brand = any(h.term in brand for h in manifest.text_hits)
    return has_evasion, has_brand


def label_case(manifest: CaseManifestV3, rulebook: dict) -> EvalExpectedV3:
    """按冻结规则书从埋点清单推导真值（同输入必得同真值）。

    :param manifest: 埋点清单（生成器产出的原始事实，所有判定由此推导）。
    :param rulebook: ``load_rulebook`` 返回的规则书 dict。
    :return: 真值（decision / abstain_label / risk_level / risk_type / 期望工具 / 应引政策）。
    """
    policies = rulebook["policy_refs"]
    blacklist = {str(b) for b in rulebook["blacklisted_brands"]}
    has_evasion, has_brand_word = _hit_kinds(manifest, rulebook)
    dirty = _merchant_dirty(manifest, rulebook)
    gap = bool(manifest.snapshot_gaps) or manifest.registry_mismatch

    if manifest.snapshot_brand is not None and manifest.snapshot_brand in blacklist:
        # R1 硬规则：黑名单品牌强制拒绝，规则层可判、无需工具。
        return EvalExpectedV3(
            decision="REJECT", abstain_label="AUTO_DECIDABLE", risk_level="HIGH",
            risk_type=["POTENTIAL_IP_RISK"], expected_tools=[],
            expected_policies=list(policies["blacklist_brand"]),
        )

    if has_evasion:
        # R-302 文本自证 + 可引用依据 → REJECT；商家脏叠加从重。
        if dirty:
            return EvalExpectedV3(
                decision="REJECT", abstain_label="AUTO_DECIDABLE", risk_level="HIGH",
                risk_type=["POTENTIAL_IP_RISK", "EVASION_PATTERN"],
                expected_tools=["MerchantTool", "PolicySearchTool"],
                expected_policies=list(policies["evasion_word_aggravated"]),
            )
        return EvalExpectedV3(
            decision="REJECT", abstain_label="AUTO_DECIDABLE", risk_level="HIGH",
            risk_type=["POTENTIAL_IP_RISK"], expected_tools=["PolicySearchTool"],
            expected_policies=list(policies["evasion_word"]),
        )

    if gap:
        # 测量缺口 / 在库事实冲突：禁止推测补齐，转人工。
        tools = ["ProductTool"] if manifest.registry_mismatch and not manifest.snapshot_gaps else ["MerchantTool", "ProductTool"]
        return EvalExpectedV3(
            decision="HUMAN_REVIEW", abstain_label="SHOULD_ABSTAIN", risk_level="MEDIUM",
            risk_type=[], expected_tools=tools, expected_policies=[],
        )

    if has_brand_word or dirty:
        # R-102 只挡 PASS 撑不起 REJECT；商家历史画像不得单独构成自动拒绝依据。
        risk: list[str] = []
        if has_brand_word:
            risk.append("POTENTIAL_IP_RISK")
        if dirty:
            risk.append("EVASION_PATTERN")
        return EvalExpectedV3(
            decision="HUMAN_REVIEW", abstain_label="SHOULD_ABSTAIN", risk_level="MEDIUM",
            risk_type=risk, expected_tools=["MerchantTool", "ProductTool"], expected_policies=[],
        )

    # 干净案：必需测量维度（在库事实 / 商家画像）靠工具覆盖后放行。
    return EvalExpectedV3(
        decision="PASS", abstain_label="AUTO_DECIDABLE", risk_level="NONE",
        risk_type=[], expected_tools=["MerchantTool", "ProductTool"], expected_policies=[],
    )
