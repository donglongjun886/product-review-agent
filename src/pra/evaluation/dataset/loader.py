"""评测集加载器：JSONL 读取 + 分层统计 + smoke 子集。

- ``load_dataset(path)``：逐行读 JSONL，每条经 ``EvalCase.model_validate_json`` 强校验
  （任何 domain 未声明键在加载期即报错；abstain_label⇔decision 一致性由 schema 校验器
  保证）。向后兼容：v1 老 JSONL（无 abstain_label、decision ∈ {PASS, REJECT}）读入后
  abstain_label=None（等价 AUTO_DECIDABLE）。解析失败抛带行号的 ``ValueError`` ——
  评测集损坏不该被静默跳过。
- ``scene_stats(cases)``：按 scene 分层的计数（含 expected.decision 分布），供报告声明分布。
- ``smoke_subset(cases, limit)``：确定性取前 ``limit`` 条（不随机；文件行序即稳定序）。

行号从 1 起，异常信息含行号与 eval_case_id（若可解析）。
"""

from __future__ import annotations

from pathlib import Path

from pra.evaluation.dataset.schema import EvalCase

__all__ = [
    "load_dataset",
    "scene_stats",
    "smoke_subset",
]

_SCENES = ("normal", "violation", "boundary", "multi-signal", "evasion")


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
    """按 scene × expected.decision 分层的计数统计（报告用）。

    返回 ``{"total": N, "by_scene": {scene: {total, PASS, REJECT, HUMAN_REVIEW,
    share}} }``；share 保留 2 位小数（纯展示）。v1 数据无 HUMAN_REVIEW 真值 →
    该键恒 0（对 report/runner 纯增量，不破坏旧口径）。
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


def smoke_subset(cases: list[EvalCase], limit: int = 10) -> list[EvalCase]:
    """确定性冒烟子集：取文件前 ``limit`` 条（数据行序稳定，不随机）。"""
    if limit <= 0:
        return []
    return list(cases[:limit])
