"""确定性重放回归：三方案决策序列 vs 基线快照。

目的：任何改动（screening 修正 / RAG / LLM 接入）若改变三方案在 v1 集上的决策就报错，
防静默行为漂移。机制：对 eval_data/v1 跑 rule / single_call_llm / agent（确定性脚本
模式），把每条 EvalRecord 的 decision 序列做规范化序列化 + sha256 digest，连同
eval_case_id 序与各 scheme 决策列表存基线快照 JSON（首次运行生成，之后比对）；
重跑重算 digest 与序列比对 → ``REGRESSION PASS / FAIL``（退出码 0/1；报告含差异
scheme 与首个差异 case）。

快照另含确定性元数据（format_version / data_hint / 总案数）供人读，比对只依据
``digest`` 与各 scheme 决策序列。

**digest 边界（如实声明）**：``canonical_digest`` 只覆盖 **case_id 序 + 决策串**，
**不锁 case 内容**（标题/证据文本等输入不在 payload 内）—— 本回归是**决策漂移守护**，
不是数据守护：手改 case 输入而决策不变时 digest 不报。要锁 case 内容需另加内容级
digest（v2 内容锁目前缺失，正是该边界另一侧的缺口）。

本模块不写 eval_data 之外的任何东西；默认基线路径由 scripts/run_regression.py 声明
（可由 --baseline 覆盖；测试一律用 tmp_path，不污染评测数据目录）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pra.evaluation.dataset.loader import load_dataset
from pra.evaluation.dataset.schema import EvalCase
from pra.evaluation.harness.base import EvalRecord, SchemeRunner
from pra.evaluation.runner import ALL_SCHEMES

__all__ = [
    "FORMAT_VERSION",
    "RegressionReport",
    "canonical_digest",
    "compare_snapshots",
    "compute_current_snapshot",
    "run_regression",
    "snapshot_from_records",
    "write_baseline",
]

FORMAT_VERSION = 1


def _canonical_json(obj) -> str:
    """规范化 JSON 序列化（键排序 + ensure_ascii=False）—— digest 的确定性输入。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def canonical_digest(per_case_ids: list[str], decisions: dict[str, list[str]]) -> str:
    """(case 序, scheme→决策序列) → sha256（确定性 digest）。

    payload 只含 case_id 序 + 决策串，不含任何 case 内容 → digest 是决策漂移守护
    而非数据守护（手改 case 输入、决策不变时不报）。
    """
    payload = _canonical_json({"per_case_ids": per_case_ids, "decisions": decisions})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def snapshot_from_records(
    cases: list[EvalCase],
    records_by_scheme: dict[str, list[EvalRecord]],
    *,
    data_hint: str | None = None,
    scheme_order: tuple[str, ...] = tuple(ALL_SCHEMES),
) -> dict:
    """由数据集 + 各 scheme records 构造基线快照 dict（确定性；可 JSON 落盘）。

    每份 records 必须按 case 行序与 cases 对齐（runner 保证）；决策序列 = 每 case 的
    ``record.decision`` 逐条转录。
    """
    per_case_ids = [c.eval_case_id for c in cases]
    decisions: dict[str, list[str]] = {}
    for scheme in scheme_order:
        records = records_by_scheme.get(scheme) or []
        if len(records) != len(cases):
            raise ValueError(
                f"scheme={scheme} records 数 {len(records)} != case 数 {len(cases)}"
            )
        seq: list[str] = []
        for case, rec in zip(cases, records):
            if rec.eval_case_id != case.eval_case_id:
                raise ValueError(
                    f"scheme={scheme} records 与 case 行序不一致: "
                    f"{rec.eval_case_id} != {case.eval_case_id}"
                )
            seq.append(rec.decision)
        decisions[scheme] = seq
    return {
        "format_version": FORMAT_VERSION,
        "data_hint": data_hint,
        "total_cases": len(cases),
        "scheme_order": list(scheme_order),
        "per_case_ids": per_case_ids,
        "decisions": decisions,
        "digest": canonical_digest(per_case_ids, decisions),
    }


def _baseline_validate(baseline: dict) -> None:
    """基线快照结构校验（防把无关 JSON 当基线比对出错）。"""
    if not isinstance(baseline, dict) or baseline.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"基线快照格式不兼容（期望 format_version={FORMAT_VERSION}）: {baseline!r:.120}"
        )


async def compute_current_snapshot(
    data_path: str | Path,
    *,
    schemes: tuple[str, ...] | list[str] = tuple(ALL_SCHEMES),
    ctx=None,
) -> dict:
    """跑指定数据集的指定方案（rule/single_call_llm/agent），返回当前快照 dict。

    快照确定性：case 行序即遍历序、方案顺序串行、EvalRecord 无墙钟字段 →
    同数据重跑 digest 逐字节一致。由 scripts/run_regression.py 落盘，
    也是"真实跑两次 → PASS"单测的复用入口。
    """
    from pra.evaluation.harness.agent_scheme import AgentScheme
    from pra.evaluation.harness.base import EvalContext
    from pra.evaluation.harness.rule_scheme import RuleBaseline
    from pra.evaluation.harness.single_call_scheme import SingleCallScheme

    ctx = ctx or EvalContext()
    cases = load_dataset(data_path)
    factories: dict[str, type[SchemeRunner]] = {
        "rule": RuleBaseline,
        "single_call_llm": SingleCallScheme,
        "agent": AgentScheme,
    }
    records_by_scheme: dict[str, list[EvalRecord]] = {}
    for name in schemes:
        if name not in factories:
            raise ValueError(f"未知 scheme: {name}（可选: {ALL_SCHEMES}）")
        scheme = factories[name]()
        records_by_scheme[name] = [await scheme.run(c, ctx) for c in cases]
    return snapshot_from_records(
        cases,
        records_by_scheme,
        data_hint=str(data_path),
        scheme_order=tuple(schemes),
    )


def write_baseline(snapshot: dict, path: str | Path) -> None:
    """把快照确定性 JSON 落盘为基线文件（父目录自动创建）。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def compare_snapshots(current: dict, baseline: dict) -> RegressionReport:
    """当前快照 vs 基线：逐 scheme 决策序列 + digest 比对 → 报告（PASS/FAIL）。

    比对口径 = **当前运行声明的 scheme 集合**（current["scheme_order"]）：基线里同名的
    方案逐案比对；当前比基线多出的方案 → FAIL 提示重录；基线比当前多的方案不参与
    （子集比对 —— 少跑某方案时不会误报）。digest 亦按该公共口径重算（基线存储的 digest
    按其录制时的方案集，直接比会误报）。
    """
    _baseline_validate(baseline)
    cur_ids = list(current.get("per_case_ids") or [])
    base_ids = list(baseline.get("per_case_ids") or [])
    cur_dec = current.get("decisions") or {}
    base_dec = baseline.get("decisions") or {}
    scheme_order = tuple(current.get("scheme_order") or ALL_SCHEMES)

    # 当前多出的方案不在基线 → 无法比对（FAIL，提示重录）
    missing_in_baseline = [s for s in scheme_order if s not in base_dec]
    mismatches: dict[str, list[tuple[int, str, str, str]]] = {}
    for scheme in scheme_order:
        if scheme not in base_dec:
            continue  # 已计入 missing_in_baseline，不进逐案差异
        a = list(base_dec.get(scheme) or [])
        b = list(cur_dec.get(scheme) or [])
        if len(a) != len(b):
            mismatches[scheme] = [(-1, "-", f"len={len(a)}", f"len={len(b)}")]
            continue
        diffs: list[tuple[int, str, str, str]] = []
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                cid = cur_ids[i] if i < len(cur_ids) else f"#{i}"
                diffs.append((i, cid, x, y))
        if diffs:
            mismatches[scheme] = diffs

    ids_changed = cur_ids != base_ids
    # digest 按公共（当前）scheme 口径重算：两边都裁到 scheme_order 的决策序列
    cur_sub = {s: cur_dec[s] for s in scheme_order if s in cur_dec}
    base_sub = {s: base_dec[s] for s in scheme_order if s in base_dec}
    digest_match = (
        not missing_in_baseline
        and canonical_digest(cur_ids, cur_sub)
        == canonical_digest(base_ids, base_sub)
    )
    ok = digest_match and not ids_changed and not mismatches and not missing_in_baseline

    summary: list[str] = []
    if not ok:
        if ids_changed:
            summary.append(
                f"数据集 case 序/集合与基线不一致（基线 {len(base_ids)} 条 vs 当前 {len(cur_ids)} 条）"
            )
        if missing_in_baseline:
            summary.append(
                f"当前运行的 scheme {missing_in_baseline} 不在基线快照里（基线录制于 "
                f"{list(base_dec)}）—— 请用 --record 重录基线（有意扩展回归范围时）。"
            )
        if digest_match is False and not mismatches and not missing_in_baseline:
            summary.append("digest 不一致但逐 scheme 序列未见差异 —— 检查序列化口径")
        for scheme, diffs in sorted(mismatches.items()):
            head = ", ".join(
                f"[{i}] {cid}: {base}→{cur}" for i, cid, base, cur in diffs[:3]
            )
            summary.append(f"scheme={scheme}: {len(diffs)} 处决策差异（前 3: {head}）")
    return RegressionReport(
        ok=ok,
        digest_match=digest_match,
        ids_changed=ids_changed,
        mismatches=mismatches,
        summary=summary,
        baseline_cases=len(base_ids),
        current_cases=len(cur_ids),
    )


class RegressionReport:
    """回归比对结果（PASS/FAIL + 差异明细；供 CLI 打印与测试断言）。"""

    def __init__(
        self,
        *,
        ok: bool,
        digest_match: bool,
        ids_changed: bool,
        mismatches: dict,
        summary: list[str],
        baseline_cases: int,
        current_cases: int,
    ) -> None:
        self.ok = ok
        self.digest_match = digest_match
        self.ids_changed = ids_changed
        self.mismatches = mismatches
        self.summary = summary
        self.baseline_cases = baseline_cases
        self.current_cases = current_cases

    @property
    def status(self) -> str:
        return "PASS" if self.ok else "FAIL"


async def run_regression(
    data_path: str | Path,
    baseline_path: str | Path,
    *,
    schemes: tuple[str, ...] | list[str] = tuple(ALL_SCHEMES),
    ctx=None,
) -> RegressionReport:
    """跑当前数据集的指定方案并与基线比对（基线须已存在，格式不兼容即报错）。

    :raises ValueError: 基线缺失 / 快照格式不兼容。
    """
    baseline_file = Path(baseline_path)
    if not baseline_file.exists():
        raise ValueError(
            f"基线快照不存在: {baseline_file}（首次运行请先 scripts/run_regression.py --record）"
        )
    try:
        baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ValueError(f"基线快照读取失败: {baseline_file}: {exc}") from exc
    _baseline_validate(baseline)

    current = await compute_current_snapshot(data_path, schemes=schemes, ctx=ctx)
    return compare_snapshots(current, baseline)
