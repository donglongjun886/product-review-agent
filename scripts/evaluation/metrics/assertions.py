"""过程断言（docs/02 §6）：不依赖标签的过程合规检查，失败清单进报告。

五条断言：
- REJECT_WITHOUT_CITATION：自动拒绝（REJECT）但 policy 为空，或证据链无可引用依据
  （无带 ref_id 的证据、也无规则命中的 rule_id）—— 自动拒绝必须同时具备可引用依据。
  只评 agent 臂：rule 臂的可引用依据是规则命中本身，single 臂无证据通道是基线设计；
- GATE_NOT_TRIGGERED：真值侧可推导的硬规则情形（manifest.snapshot_brand 命中冻结规则书的
  品牌黑名单），agent 臂 overrides 应含 R1_HARD_RULE，未含即判错
  （只查 agent 臂：rule/single 无 overrides 通道）；
- BUDGET_EXCEEDED：record.cost 超过本次运行的预算上限（三臂恒为生产默认档，由调用方传入）；
- CITATION_UNRESOLVABLE：record.policy 的条款 id（policy_id 或 clause_id）在政策库
  policies.json 找不回，或 CASE_PRECEDENT 证据的 ref_id 在先例库 cases.json 找不回；
- SILENT_DEGRADATION：auto 决策（PASS/REJECT）但 record detail 带 error，
  或 agent 臂 PASS 但证据链为空 —— 禁止空结果冒充正常。
"""

from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

from evaluation.dataset.schema import EvalTruthV3
from evaluation.record import EvalRecord
from pydantic import BaseModel, Field

__all__ = [
    "BUDGET_EXCEEDED",
    "CITATION_UNRESOLVABLE",
    "GATE_NOT_TRIGGERED",
    "REJECT_WITHOUT_CITATION",
    "SILENT_DEGRADATION",
    "AssertionFailure",
    "CitationIndex",
    "check_process_assertions",
    "load_citation_index",
]

REJECT_WITHOUT_CITATION = "REJECT_WITHOUT_CITATION"
GATE_NOT_TRIGGERED = "GATE_NOT_TRIGGERED"
BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
CITATION_UNRESOLVABLE = "CITATION_UNRESOLVABLE"
SILENT_DEGRADATION = "SILENT_DEGRADATION"

_R1_HARD_RULE = "R1_HARD_RULE"
_CASE_PRECEDENT = "CASE_PRECEDENT"
_AUTO_DECISIONS = frozenset({"PASS", "REJECT"})


class AssertionFailure(BaseModel):
    """一条过程断言失败：哪一案、哪一臂、哪条断言、人读细节。"""

    eval_case_id: str
    scheme: str
    code: str = Field(description="断言码（REJECT_WITHOUT_CITATION / GATE_NOT_TRIGGERED / ...）")
    detail: str


@dataclass(frozen=True)
class CitationIndex:
    """引用可解析索引：政策条款 id 集合（policy_id + clause_id）与先例 case_id 集合。"""

    policy_refs: frozenset[str]
    precedent_ids: frozenset[str]


def load_citation_index(corpus_dir: Path) -> CitationIndex:
    """从 RAG 语料目录直接读 JSON 构造引用索引（确定性，不走 RAG 检索）。

    :raises ValueError: 语料文件缺失或损坏（显式失败，不静默降级为空索引）。
    """
    policies_path = corpus_dir / "policies.json"
    cases_path = corpus_dir / "cases.json"
    for path in (policies_path, cases_path):
        if not path.is_file():
            raise ValueError(f"引用解析语料缺失: {path}")
    try:
        policies = json.loads(policies_path.read_text(encoding="utf-8"))["policies"]
        cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise ValueError(f"引用解析语料损坏: {exc!r}") from exc
    policy_refs = {
        str(clause[key])
        for clause in policies
        for key in ("policy_id", "clause_id")
        if clause.get(key)
    }
    precedent_ids = {str(c["case_id"]) for c in cases if c.get("case_id")}
    return CitationIndex(policy_refs=frozenset(policy_refs), precedent_ids=frozenset(precedent_ids))


def _has_citable_evidence(record: EvalRecord) -> bool:
    """证据链是否存在可引用依据：带 ref_id 的证据，或规则命中的 rule_id。"""
    for item in record.evidence:
        if not isinstance(item, dict):
            continue
        if item.get("ref_id"):
            return True
        extra = item.get("extra")
        if isinstance(extra, dict) and extra.get("rule_id"):
            return True
    return False


def check_process_assertions(
    records: list[EvalRecord],
    truth: list[EvalTruthV3],
    *,
    scheme: str,
    citations: CitationIndex,
    blacklisted_brands: Collection[str],
    max_llm_calls: int,
    max_tool_calls: int,
) -> list[AssertionFailure]:
    """对一臂记录跑五条过程断言，返回失败清单（按记录顺序）。

    :param records: 一臂的评测记录；
    :param truth: v3 真值行（GATE_NOT_TRIGGERED 需要 manifest.snapshot_brand）；
    :param scheme: 本批记录所属臂名（断言的臂范围以此为准，与记录 scheme 字段一致）；
    :param citations: 引用可解析索引（见 ``load_citation_index``）；
    :param blacklisted_brands: 冻结规则书的品牌黑名单（真值口径单一来源）；
    :param max_llm_calls / max_tool_calls: 本次运行的预算上限。
    """
    manifests = {t.eval_case_id: t.manifest for t in truth}
    blacklist = set(blacklisted_brands)
    failures: list[AssertionFailure] = []
    for rec in records:
        cid = rec.eval_case_id

        if scheme == "agent" and rec.decision == "REJECT" and (
            not rec.policy or not _has_citable_evidence(rec)
        ):
            failures.append(
                AssertionFailure(
                    eval_case_id=cid,
                    scheme=scheme,
                    code=REJECT_WITHOUT_CITATION,
                    detail=(
                        f"自动拒绝缺可引用依据（policy={list(rec.policy)}，"
                        f"证据链可引用={_has_citable_evidence(rec)}）"
                    ),
                )
            )

        manifest = manifests.get(cid)
        if (
            scheme == "agent"
            and manifest is not None
            and manifest.snapshot_brand is not None
            and manifest.snapshot_brand in blacklist
            and _R1_HARD_RULE not in list((rec.detail or {}).get("overrides") or [])
        ):
            failures.append(
                AssertionFailure(
                    eval_case_id=cid,
                    scheme=scheme,
                    code=GATE_NOT_TRIGGERED,
                    detail=(
                        f"快照品牌 {manifest.snapshot_brand!r} 命中黑名单，"
                        "overrides 未含 R1_HARD_RULE（硬规则应触发未触发）"
                    ),
                )
            )

        cost = rec.cost or {}
        llm_calls = int(cost.get("llm_calls") or 0)
        tool_calls = int(cost.get("tool_calls") or 0)
        if llm_calls > max_llm_calls or tool_calls > max_tool_calls:
            failures.append(
                AssertionFailure(
                    eval_case_id=cid,
                    scheme=scheme,
                    code=BUDGET_EXCEEDED,
                    detail=(
                        f"成本超预算上限（llm_calls={llm_calls}/{max_llm_calls}，"
                        f"tool_calls={tool_calls}/{max_tool_calls}）"
                    ),
                )
            )

        unresolvable = [p for p in rec.policy if p not in citations.policy_refs]
        for item in rec.evidence:
            if not isinstance(item, dict):
                continue
            ref = item.get("ref_id")
            if item.get("type") == _CASE_PRECEDENT and ref and ref not in citations.precedent_ids:
                unresolvable.append(f"先例 {ref}")
        if unresolvable:
            failures.append(
                AssertionFailure(
                    eval_case_id=cid,
                    scheme=scheme,
                    code=CITATION_UNRESOLVABLE,
                    detail=f"引用在语料库找不回: {unresolvable}",
                )
            )

        error = (rec.detail or {}).get("error")
        if rec.decision in _AUTO_DECISIONS and error:
            failures.append(
                AssertionFailure(
                    eval_case_id=cid,
                    scheme=scheme,
                    code=SILENT_DEGRADATION,
                    detail=f"自动决策（{rec.decision}）但 detail 带 error: {str(error)[:60]}",
                )
            )
        elif scheme == "agent" and rec.decision == "PASS" and not rec.evidence:
            failures.append(
                AssertionFailure(
                    eval_case_id=cid,
                    scheme=scheme,
                    code=SILENT_DEGRADATION,
                    detail="agent 臂 PASS 但证据链为空（空结果冒充正常）",
                )
            )
    return failures
