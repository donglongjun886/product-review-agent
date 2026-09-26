"""落库编排 —— 接入 → 落库闭环的 worker 侧主入口。

把一次审核判定活动的业务真相落 MySQL 五表（review_case / review_run / review_trace /
review_evidence / review_result）。两类判定活动：Agent 调查（``run_and_persist``）与规则直判
（``run_screening_direct``）；``process_review`` 是 POST /api/v1/reviews 主入口（受理即分流）。
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from sqlalchemy.dialects.mysql import insert as mysql_insert

from pra.agent.guardrails.gate import R6_INFRA_UNAVAILABLE
from pra.agent.state import build_initial_state, merge_evidence
from pra.domain.models import (
    Budget,
    Decision,
    Evidence,
    ProductReviewCase,
    ReviewDecision,
    RiskLevel,
)
from pra.infra.db import get_sessionmaker
from pra.infra.rdb_models import (
    ReviewCaseORM,
    ReviewEvidenceORM,
    ReviewResultORM,
    ReviewRunORM,
    ReviewTraceORM,
)
from pra.screening.engine import TriageResult, rule_evidence, triage
from pra.wiring import get_production_graph, trace_callbacks

logger = logging.getLogger(__name__)

__all__ = ["process_review", "run_and_persist", "run_screening_direct"]

# 单行 JSON 上限。
_JSON_CAP = 64 * 1024

# 规则直判 run 的 trigger_type。
_TRIGGER_SCREENING_DIRECT = "SCREENING_DIRECT"

# 节点名 → review_trace.step_type 词汇表。
_NODE_STEP_TYPE = {
    "hypothesize": "HYPOTHESIZE",
    "plan": "PLAN",
    "reevaluate": "REEVALUATE",
    "decide": "DECIDE",
}


def _utcnow() -> datetime:
    """当前 naive UTC 时间。"""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _enum_value(v: Any) -> Any:
    """Enum → .value（普通值原样返回）。"""
    return v.value if isinstance(v, Enum) else v


def _token_count(budget: Any) -> int:
    """从 budget 快照取 tokens（兼容 Pydantic Budget / JSON dict 两种形态）。"""
    if budget is None:
        return 0
    tokens = budget.tokens if not isinstance(budget, dict) else budget.get("tokens")
    return int(tokens or 0)


def _json_cap(obj: Any, cap: int = _JSON_CAP) -> Any:
    """把 JSON 可序列化对象装箱成 DB JSON 列可存形态；超长时退化为截断占位。"""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    try:
        text = json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as exc:
        return {"_json_error": str(exc), "_type": type(obj).__name__}
    if len(text) <= cap:
        return obj
    return {
        "_truncated": True,
        "_type": type(obj).__name__,
        "_length": len(text),
        "_preview": text[:cap],
    }


def _hypothesis_summary(h: Any) -> dict:
    """Hypothesis 摘要（id/statement/status/prior/posterior；JSON 标量化）。"""
    return {
        "id": h.id,
        "statement": h.statement,
        "status": _enum_value(h.status),
        "prior": h.prior,
        "posterior": h.posterior,
    }


def _node_output_summary(node_name: str, update: dict) -> dict:
    """按节点构造 trace 输出摘要，含 budget 快照。"""
    budget = update.get("budget")
    budget_snap = None
    if budget is not None:
        budget_snap = {
            "llm_calls": budget.llm_calls if not isinstance(budget, dict) else budget.get("llm_calls"),
            "tool_calls": budget.tool_calls if not isinstance(budget, dict) else budget.get("tool_calls"),
            "tokens": _token_count(budget),
        }

    if node_name == "hypothesize":
        hypos = list(update.get("hypotheses") or [])
        return {
            "node": node_name,
            "degraded": bool(update.get("degraded")),
            "hypotheses_count": len(hypos),
            "hypotheses": [_hypothesis_summary(h) for h in hypos],
            "budget": budget_snap,
        }
    if node_name == "plan":
        pending = list(update.get("pending_tool_calls") or [])
        skipped = list(update.get("tool_call_history") or [])  # dedup 跳过审计
        return {
            "node": node_name,
            "degraded": bool(update.get("degraded")),
            "pending_tool_calls_count": len(pending),
            "pending_tool_calls": [
                {"tool": c.get("tool"), "priority": c.get("priority"), "reason": c.get("reason")}
                for c in pending
            ],
            "dedup_skipped_count": len(skipped),
            "budget": budget_snap,
        }
    if node_name == "reevaluate":
        hypos = list(update.get("hypotheses") or [])
        status_counts: dict[str, int] = {}
        for h in hypos:
            st = str(_enum_value(h.status))
            status_counts[st] = status_counts.get(st, 0) + 1
        return {
            "node": node_name,
            "degraded": bool(update.get("degraded")),
            "hypotheses_count": len(hypos),
            "hypotheses": [_hypothesis_summary(h) for h in hypos],
            "status_counts": status_counts,
            "budget": budget_snap,
        }
    if node_name == "decide":
        decision = update.get("decision")
        if decision is None:
            return {"node": node_name, "degraded": bool(update.get("degraded")), "decision": None}
        return {
            "node": node_name,
            "decision": decision.decision.value,
            "risk_level": decision.risk_level.value,
            "risk_type": [_enum_value(t) for t in decision.risk_type],
            "decision_confidence": decision.decision_confidence,
            "policy": list(decision.policy),
            "overrides": list(decision.overrides),
            "evidence_count": len(decision.evidence),
            "hypothesis_trace_count": len(decision.hypothesis_trace),
            "budget": budget_snap,
        }
    return {
        "node": node_name,
        "update_keys": sorted(update.keys()),
        "budget": budget_snap,
    }


def _node_input_summary(node_name: str, update: dict, case_id: str) -> dict:
    """LLM 行 input_json —— 该步的轻量状态摘要。"""
    return {
        "node": node_name,
        "case_id": case_id,
        "hypotheses_count": len(list(update.get("hypotheses") or [])),
        "pending_tool_calls_count": len(list(update.get("pending_tool_calls") or [])),
        "budget_tokens": _token_count(update.get("budget")),
    }


def _evidence_row(
    session: Any,
    run_id: str,
    ev: Evidence,
    created_at: datetime | None = None,
) -> None:
    """把一条 Evidence 映射为 review_evidence ORM 行并 add（不 commit）。"""
    session.add(
        ReviewEvidenceORM(
            run_id=run_id,
            type=ev.type,
            source_tool=ev.source,
            value=ev.value,
            weight=ev.weight,
            ref_id=ev.ref_id,
            extra_json=_json_cap(ev.extra) if ev.extra else None,
            created_at=created_at or _utcnow(),
        )
    )


def _result_payload(
    decision: ReviewDecision, case_id: str, source_run_id: str
) -> dict:
    """review_result 行 payload。"""
    return {
        "case_id": case_id,
        "source_run_id": source_run_id,
        "decision": decision.decision.value,
        "risk_level": decision.risk_level.value,
        "risk_type_json": [_enum_value(t) for t in decision.risk_type],
        "decision_confidence": decision.decision_confidence,
        "policy_refs_json": list(decision.policy),
        "created_at": _utcnow(),
        "updated_at": _utcnow(),
    }


async def _upsert_result(session: Any, payload: dict) -> None:
    """review_result upsert（last-writer-wins）。"""
    stmt = (
        mysql_insert(ReviewResultORM)
        .values(**payload)
        .on_duplicate_key_update(
            source_run_id=payload["source_run_id"],
            decision=payload["decision"],
            risk_level=payload["risk_level"],
            risk_type_json=payload["risk_type_json"],
            decision_confidence=payload["decision_confidence"],
            policy_refs_json=payload["policy_refs_json"],
            updated_at=payload["updated_at"],
        )
    )
    await session.execute(stmt)


async def run_and_persist(
    case: ProductReviewCase,
    *,
    run_id: str | None = None,
    trigger_type: str = "INITIAL",
    triage_result: str | None = None,
    extra_evidence: list[Evidence] | None = None,
) -> dict:
    """执行一次完整复杂风险调查（Agent 调查路径）并把业务真相落 MySQL 五表。

    :param case: 审核案件；case_json 落 ``case.model_dump(mode="json")`` 全量快照。
    :param run_id: 本次运行 ID（= LangGraph thread_id）；None → ``uuid4().hex``。
    :param trigger_type: run 目的（INITIAL/RE_REVIEW/...）。
    :param triage_result: 非 None 时写 review_case.triage_result（创建与复用行均写）。
    :param extra_evidence: RULE_HIT Evidence 列表（type=RULE_HIT /
        source_tool=ScreeningRuleEngine / weight=1.0 / extra_json={"rule_id"}）；默认不写。
    :return: ``{"case_id", "run_id", "decision", "counts": {"trace", "evidence"}}`` ——
        decision 为图终态 ReviewDecision（供 API 包装返回），counts 为本 run 落库行数。
    """
    resolved_run_id = run_id or uuid4().hex
    case_id = case.case_id
    now = _utcnow()

    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        case_row = await session.get(ReviewCaseORM, case_id)
        if case_row is None:
            case_row = ReviewCaseORM(
                case_id=case_id,
                product_id=case.product.product_id,
                merchant_id=case.merchant_id,
                event_type=case.event_type,
                status="INVESTIGATING",
                version=case.product.version,
                triage_result=triage_result,
                case_json=case.model_dump(mode="json"),
                created_at=now,
                updated_at=now,
            )
            session.add(case_row)
        else:
            case_row.status = "INVESTIGATING"
            if triage_result is not None:
                case_row.triage_result = triage_result
            case_row.updated_at = now

        run_row = ReviewRunORM(
            run_id=resolved_run_id,
            case_id=case_id,
            status="RUNNING",
            trigger_type=trigger_type,
            started_at=now,
            ended_at=None,
        )
        session.add(run_row)
        await session.commit()

        extra_rows = 0
        for ev in extra_evidence or []:
            _evidence_row(session, resolved_run_id, ev)
            extra_rows += 1
        if extra_rows:
            await session.commit()

        app = get_production_graph()
        config = {
            "configurable": {"thread_id": resolved_run_id},
            "callbacks": trace_callbacks(),
        }
        seq = 0
        trace_rows = 0
        prev_budget_tokens = 0
        wall_prev = time.perf_counter()
        # 异常前已观测到的证据（异常终态用）。
        latest_evidence: list[Evidence] = []
        try:
            async for chunk in app.astream(
                build_initial_state(case), config, stream_mode="updates"
            ):
                for node_name, update in chunk.items():
                    wall_now = time.perf_counter()
                    node_latency_ms = max(int((wall_now - wall_prev) * 1000), 0)
                    wall_prev = wall_now

                    budget = update.get("budget")
                    if budget is None:
                        step_tokens = 0
                    else:
                        new_tokens = _token_count(budget)
                        step_tokens = max(new_tokens - prev_budget_tokens, 0)
                        prev_budget_tokens = new_tokens

                    raw_evidence = update.get("evidence")
                    if raw_evidence:
                        latest_evidence = merge_evidence(
                            latest_evidence,
                            [e for e in raw_evidence if isinstance(e, Evidence)],
                        )

                    if node_name == "tools":
                        records = list(update.get("tool_call_history") or [])
                        for rec in records:
                            seq += 1
                            session.add(
                                ReviewTraceORM(
                                    run_id=resolved_run_id,
                                    seq=seq,
                                    step_type="TOOL_CALL",
                                    tool_name=rec.get("tool"),
                                    input_json=_json_cap(rec.get("args")) if rec.get("args") else None,
                                    output_json=_json_cap(rec),
                                    tokens=int(rec.get("tokens") or 0),
                                    latency_ms=int(rec.get("latency_ms") or 0),
                                    created_at=_utcnow(),
                                )
                            )
                            trace_rows += 1
                    else:
                        seq += 1
                        session.add(
                            ReviewTraceORM(
                                run_id=resolved_run_id,
                                seq=seq,
                                step_type=_NODE_STEP_TYPE.get(node_name, node_name.upper()),
                                tool_name=None,
                                input_json=_json_cap(
                                    _node_input_summary(node_name, update, case_id)
                                ),
                                output_json=_json_cap(
                                    _node_output_summary(node_name, update)
                                ),
                                tokens=step_tokens,
                                latency_ms=node_latency_ms,
                                created_at=_utcnow(),
                            )
                        )
                        trace_rows += 1
                    await session.commit()

            logger.info(
                "run_and_persist 图执行完成 case_id=%s run_id=%s trace_rows=%d",
                case_id, resolved_run_id, trace_rows,
            )

            snapshot = await app.aget_state(config)
            try:
                final_state = snapshot["values"]  # type: ignore[index]
            except TypeError:
                final_state = snapshot.values  # type: ignore[attr-defined]

            decision = final_state.get("decision")

            if not isinstance(decision, ReviewDecision):
                raise RuntimeError(
                    f"调查图执行完成但终态缺少 decision（run_id={resolved_run_id}, "
                    f"case_id={case_id}）—— 违反 'decide 为图唯一终态出口' 契约"
                )
        except Exception as exc:
            error_decision = ReviewDecision(
                decision=Decision.HUMAN_REVIEW,
                risk_level=RiskLevel.NONE,
                risk_type=[],
                decision_confidence=1.0,
                evidence=latest_evidence,
            )
            await _upsert_result(
                session, _result_payload(error_decision, case_id, resolved_run_id)
            )
            await session.commit()

            for ev in error_decision.evidence:
                _evidence_row(session, resolved_run_id, ev)

            seq += 1
            session.add(
                ReviewTraceORM(
                    run_id=resolved_run_id,
                    seq=seq,
                    step_type="infra_error",
                    tool_name=None,
                    input_json=None,
                    output_json=_json_cap(
                        {
                            "error": f"{type(exc).__name__}: {exc}",
                            "overrides": [R6_INFRA_UNAVAILABLE],
                            "evidence_count": len(error_decision.evidence),
                        }
                    ),
                    tokens=0,
                    latency_ms=0,
                    created_at=_utcnow(),
                )
            )
            await session.commit()
            raise

        extra_keys = {
            (e.type, e.source, e.value) for e in (extra_evidence or [])
        }
        overlap = [
            ev
            for ev in decision.evidence
            if (ev.type, ev.source, ev.value) in extra_keys
        ]
        if overlap:
            raise RuntimeError(
                f"decision.evidence 与分流 extra_evidence 重叠 {len(overlap)} 条"
                f"（run_id={resolved_run_id}）—— 分流命中不应进入 AgentState 证据链"
            )

        evidence_rows = extra_rows
        for ev in decision.evidence:
            _evidence_row(session, resolved_run_id, ev)
            evidence_rows += 1

        await _upsert_result(
            session, _result_payload(decision, case_id, resolved_run_id)
        )

        end_now = _utcnow()
        run_row.status = "DECIDED"
        run_row.ended_at = end_now
        case_row.status = "DECIDED"
        case_row.updated_at = end_now
        await session.commit()

    return {
        "case_id": case_id,
        "run_id": resolved_run_id,
        "decision": decision,
        "counts": {"trace": trace_rows, "evidence": evidence_rows},
    }


def _direct_decision(verdict: str, evidence: list[Evidence]) -> ReviewDecision:
    """构造规则直判的同构 ReviewDecision（确定性直判：confidence=1.0）。

    PASS → decision=PASS / risk_level=NONE；REJECT → decision=REJECT / risk_level=HIGH。
    """
    return ReviewDecision(
        decision=Decision.PASS if verdict == "PASS" else Decision.REJECT,
        risk_level=RiskLevel.NONE if verdict == "PASS" else RiskLevel.HIGH,
        risk_type=[],
        decision_confidence=1.0,
        evidence=list(evidence),
        policy=[],
        hypothesis_trace=[],
        budget_used=Budget(),
        overrides=[],
    )


async def run_screening_direct(
    case: ProductReviewCase,
    triage: TriageResult,
    *,
    run_id: str | None = None,
) -> dict:
    """Screening 规则直判（PASS/REJECT 确定性直接终裁）落库 —— Agent 路径之外的第二类判定活动。

    :param case: 审核案件（case_id 即主键；case_json 全量快照）。
    :param triage: ``triage`` 产物，verdict ∈ {PASS, REJECT}；传 COMPLEX 抛 ValueError。
    :param run_id: 本次判定运行 ID；None → ``uuid4().hex``。
    :return: ``{"case_id", "run_id", "verdict", "decision", "counts"}`` —— counts =
        {"evidence"} RULE_HIT 行数（PASS 零命中则 0）。
    """
    if triage.verdict not in ("PASS", "REJECT"):
        raise ValueError(
            f"run_screening_direct 只接受 PASS/REJECT 直判，收到 verdict="
            f"{triage.verdict!r} —— COMPLEX 应走 run_and_persist（Agent 调查路径）"
        )
    resolved_run_id = run_id or uuid4().hex
    case_id = case.case_id
    now = _utcnow()

    sessionmaker = get_sessionmaker()
    async with sessionmaker() as session:
        case_row = await session.get(ReviewCaseORM, case_id)
        if case_row is None:
            case_row = ReviewCaseORM(
                case_id=case_id,
                product_id=case.product.product_id,
                merchant_id=case.merchant_id,
                event_type=case.event_type,
                status="DECIDED",
                version=case.product.version,
                triage_result=triage.verdict,
                case_json=case.model_dump(mode="json"),
                created_at=now,
                updated_at=now,
            )
            session.add(case_row)
        else:
            case_row.status = "DECIDED"
            case_row.triage_result = triage.verdict
            case_row.updated_at = now

        run_row = ReviewRunORM(
            run_id=resolved_run_id,
            case_id=case_id,
            status="DECIDED",
            trigger_type=_TRIGGER_SCREENING_DIRECT,
            started_at=now,
            ended_at=now,
        )
        session.add(run_row)

        evidence_rows = 0
        evidence_list: list[Evidence] = []
        for hit in triage.hits:
            ev = rule_evidence(hit)
            evidence_list.append(ev)
            _evidence_row(session, resolved_run_id, ev)
            evidence_rows += 1

        decision = _direct_decision(triage.verdict, evidence_list)
        await _upsert_result(
            session, _result_payload(decision, case_id, resolved_run_id)
        )

        await session.commit()

    logger.info(
        "run_screening_direct 直判落库 case_id=%s run_id=%s verdict=%s evidence=%d",
        case_id, resolved_run_id, triage.verdict, evidence_rows,
    )
    return {
        "case_id": case_id,
        "run_id": resolved_run_id,
        "verdict": triage.verdict,
        "decision": decision,
        "counts": {"evidence": evidence_rows},
    }


async def process_review(
    case: ProductReviewCase,
    *,
    run_id: str | None = None,
) -> dict:
    """POST /api/v1/reviews 主入口 —— 受理即分流（Screening 三分流先行再分支）。

    :param run_id: 运行 ID；None → 各分支自动 ``uuid4().hex``。
    :return: ``{"case_id", "run_id", "verdict", "decision", "counts"}`` —— verdict ∈
        {PASS, REJECT, COMPLEX}；decision 供 API 包装返回；counts 为 COMPLEX: {trace,
        evidence}、直判: {evidence}。
    """
    t = triage(case)
    if t.verdict == "COMPLEX":
        summary = await run_and_persist(
            case,
            run_id=run_id,
            triage_result="COMPLEX",
            extra_evidence=[rule_evidence(hit) for hit in t.hits],
        )
        return {
            "case_id": summary["case_id"],
            "run_id": summary["run_id"],
            "verdict": t.verdict,
            "decision": summary["decision"],
            "counts": dict(summary.get("counts") or {}),
        }

    summary = await run_screening_direct(case, t, run_id=run_id)
    return {
        "case_id": summary["case_id"],
        "run_id": summary["run_id"],
        "verdict": t.verdict,
        "decision": summary["decision"],
        "counts": dict(summary.get("counts") or {}),
    }
