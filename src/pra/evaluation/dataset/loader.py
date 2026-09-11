"""评测集加载器（dataset/loader.py）—— JSONL 读取 + 分层统计 + smoke 子集。

职责（Phase 1 最小闭环口径；Phase 2 增 abstention 统计）：

- ``load_dataset(path)``：逐行读取 JSONL，每条经 ``EvalCase.model_validate_json``
  强校验（input 复用 ProductReviewCase 契约 → 任何 domain 未声明键在加载期即报错；
  expected 的 abstain_label⇔decision 一致性由 schema 校验器保证）；
  **向后兼容**：Phase 1 老 JSONL（无 abstain_label 字段、decision∈{PASS,REJECT}）
  读入后 abstain_label=None（等价 AUTO_DECIDABLE，docs/02 §4.4），照常加载；
  解析失败抛出带行号的 ``ValueError``（评测集损坏不该被静默跳过）。
- ``scene_stats(cases)``：按 scene 分层的计数统计（含 expected.decision 分布；
  Phase 2 起含 HUMAN_REVIEW 真值计数），供 manifest 生成 / 报告声明分布。
- ``abstain_stats(cases)``：abstention 语义标签（AUTO_DECIDABLE / SHOULD_ABSTAIN /
  老数据 None）计数 —— manifest 与 AbstentionEvaluator 前置口径的单一取数点。
- ``smoke_subset(cases, limit)``：确定性取前 ``limit`` 条做冒烟（不随机 ——
  Phase 1 要求全程确定性；数据集文件行序即稳定序）。
- ``manifest 解析``：load_manifest(dir) 读取 manifest.json（版本/分布/阈值口径
  快照/生成命令等元数据）；loader 只负责读与回传，不校验与 JSONL 强一致
  （防"文档口径漂移 vs 数据实际分布"由评审复核，报告里会打印实际分布）。

错误语义：行号从 1 起，异常信息含行号与 eval_case_id（若可解析）。
"""

from __future__ import annotations

import json
from pathlib import Path

from pra.evaluation.dataset.schema import AbstainLabel, EvalCase

__all__ = [
    "abstain_stats",
    "load_dataset",
    "load_manifest",
    "scene_stats",
    "smoke_subset",
]

_SCENES = ("normal", "violation", "boundary", "multi-signal", "evasion")
_ABSTAIN_LABELS: tuple[AbstainLabel, ...] = ("AUTO_DECIDABLE", "SHOULD_ABSTAIN")


def load_dataset(path: str | Path) -> list[EvalCase]:
    """读取 JSONL 评测集并强校验；返回按文件行序的 EvalCase 列表（稳定序）。

    :raises ValueError: 文件缺失 / 空文件 / 某行 JSON 损坏或 schema 校验失败（带行号）。
    """
    p = Path(path)
    if not p.exists():
        raise ValueError(f"评测集文件不存在: {p}")
    cases: list[EvalCase] = []
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:  # 容忍空行（编辑器尾行）
                continue
            try:
                case = EvalCase.model_validate_json(stripped)
            except Exception as exc:  # ValueError(JSON) / ValidationError(schema)
                raise ValueError(
                    f"评测集第 {lineno} 行解析失败（含行号定位）: {exc}"
                ) from exc
            cases.append(case)
    if not cases:
        raise ValueError(f"评测集为空（无有效 case 行）: {p}")
    return cases


def scene_stats(cases: list[EvalCase]) -> dict:
    """按 scene × expected.decision 分层的计数统计（manifest / 报告用）。

    返回 ``{"total": N, "by_scene": {scene: {total, PASS, REJECT, HUMAN_REVIEW,
    share}} }``；share 保留 2 位小数（确定性四舍五入，纯展示）。
    Phase 1 数据无 HUMAN_REVIEW 真值 → 该键恒 0（对 report/runner 纯增量，不破坏
    旧口径）；Phase 2 数据含 SHOULD_ABSTAIN（decision=HUMAN_REVIEW）案时如实计数。
    """
    total = len(cases)
    by_scene: dict = {}
    for scene in _SCENES:
        rows = [c for c in cases if c.scene == scene]
        pass_n = sum(1 for c in rows if c.expected.decision == "PASS")
        reject_n = sum(1 for c in rows if c.expected.decision == "REJECT")
        human_n = sum(1 for c in rows if c.expected.decision == "HUMAN_REVIEW")
        share = round(len(rows) / total, 2) if total else 0.0
        by_scene[scene] = {
            "total": len(rows),
            "PASS": pass_n,
            "REJECT": reject_n,
            "HUMAN_REVIEW": human_n,
            "share": share,
        }
    by_scene["_unknown_scene"] = {
        "total": sum(1 for c in cases if c.scene not in _SCENES)
    }
    return {"total": total, "by_scene": by_scene}


def abstain_stats(cases: list[EvalCase]) -> dict:
    """abstention 语义标签计数（docs/02 §4.4；manifest / 测试 / B 面指标共用口径）。

    返回::

        {
          "total": N,
          "AUTO_DECIDABLE": n,        # 显式标 AUTO_DECIDABLE（decision∈{PASS,REJECT}）
          "SHOULD_ABSTAIN": n,        # decision==HUMAN_REVIEW 案
          "LEGACY_UNLABELED": n,      # abstain_label=None（Phase 1 老数据；等价 AUTO_DECIDABLE）
          "auto_decidable_equivalent": n,  # AUTO_DECIDABLE + LEGACY_UNLABELED
          "auto_share": float,        # auto_decidable_equivalent / total（2 位小数）
          "abstain_share": float,     # SHOULD_ABSTAIN / total（2 位小数）
          "by_scene": {scene: {"total", "AUTO_DECIDABLE", "SHOULD_ABSTAIN", "LEGACY_UNLABELED"}},
        }

    None（老数据缺失字段）在语义上等价 AUTO_DECIDABLE，但单独计数以便区分
    Phase 1 老数据与 Phase 2 显式标注。
    """
    total = len(cases)
    counts = {label: 0 for label in _ABSTAIN_LABELS}
    legacy = 0
    for c in cases:
        label = c.expected.abstain_label
        if label is None:
            legacy += 1
        else:
            counts[label] += 1
    auto_equiv = counts["AUTO_DECIDABLE"] + legacy
    by_scene: dict = {}
    for scene in _SCENES:
        rows = [c for c in cases if c.scene == scene]
        sc: dict = {"total": len(rows), "AUTO_DECIDABLE": 0, "SHOULD_ABSTAIN": 0, "LEGACY_UNLABELED": 0}
        for c in rows:
            label = c.expected.abstain_label
            key = "LEGACY_UNLABELED" if label is None else str(label)
            sc[key] += 1
        by_scene[scene] = sc
    by_scene["_unknown_scene"] = {
        "total": sum(1 for c in cases if c.scene not in _SCENES)
    }
    return {
        "total": total,
        "AUTO_DECIDABLE": counts["AUTO_DECIDABLE"],
        "SHOULD_ABSTAIN": counts["SHOULD_ABSTAIN"],
        "LEGACY_UNLABELED": legacy,
        "auto_decidable_equivalent": auto_equiv,
        "auto_share": round(auto_equiv / total, 2) if total else 0.0,
        "abstain_share": round(counts["SHOULD_ABSTAIN"] / total, 2) if total else 0.0,
        "by_scene": by_scene,
    }


def smoke_subset(cases: list[EvalCase], limit: int = 10) -> list[EvalCase]:
    """确定性冒烟子集：取文件前 ``limit`` 条（数据行序稳定，不随机）。"""
    if limit <= 0:
        return []
    return list(cases[:limit])


def load_manifest(data_dir: str | Path) -> dict:
    """读取 ``manifest.json``（版本/分布/口径快照等元数据）；缺失返回 {}。

    Phase 1 只读不校验：manifest 与 JSONL 的一致性由报告侧打印实际分布供人核对。
    """
    p = Path(data_dir) / "manifest.json"
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as fh:
        try:
            obj = json.load(fh)
        except (ValueError, TypeError):
            return {}
    return obj if isinstance(obj, dict) else {}
