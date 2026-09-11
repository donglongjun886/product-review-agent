"""落库编排 —— 接入 → 落库闭环的 worker 侧主入口。

把一次审核判定活动的业务真相落 MySQL 五表（review_case / review_run / review_trace /
review_evidence / review_result）。两类判定活动：Agent 调查（``run_and_persist``，LangGraph
子图执行，trigger_type=INITIAL/RE_REVIEW，逐节点落 review_trace）与规则直判
（``run_screening_direct``，Screening PASS/REJECT 确定性终裁，trigger_type=
"SCREENING_DIRECT"，无 trace 行、started_at≈ended_at、status 直接 DECIDED）。两类都写
review_result（source_run_id 指向被采纳的 run），命中证据挂 review_evidence ——
``result → run → evidence`` 审计链对两类裁决统一成立。``process_review`` 是 POST
/api/v1/reviews 主入口（受理即分流）：COMPLEX → ``run_and_persist``
（triage_result='COMPLEX'，RULE_HIT 证据随 agent run 落库，与直判路径审计对称）；PASS/REJECT
→ ``run_screening_direct``。

不变量与口径：
- 幂等：``run_id``（缺省 uuid4().hex）是 review_run 主键，worker 可用业务键派生确定性 run_id；
  case 按 case_id 复用，**不覆盖 case_json**（快照=首投内容，防上游漂移）。
- 断点：run 行先于 stream 落库（RUNNING），每步 trace 即写即 commit —— 中途崩溃时已落 trace
  可查、run 停留 RUNNING 可续跑；直判路径无 stream，单 commit。
- 观测列：review_trace 每行 = 一步（LLM 节点行 / 工具 TOOL_CALL 行，seq 连续唯一）。tokens
  由相邻 budget 快照差分（首步用其值；无 budget 键的短路步记 0 且**不推进基线**，防 0 重置
  致后续全量重复计），工具行取 record 自带 tokens；latency_ms 工具行走 record、LLM 行走节点
  到达墙钟差（近似）。
- 时间：所有时间列写 naive UTC（``_utcnow``），MySQL DATETIME(3) 存 naive。
- 错误：图执行/落库异常向上抛出（HTTP 层转 500），已 commit 的 run/trace 行保留 —— 不吞异常、
  不做部分回滚（crash 可查优先于 all-or-nothing）。
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

from pra.agent.checkpointer import make_memory_checkpointer
from pra.agent.state import build_initial_state
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
from pra.observability.tracing import (
    TraceContext,
    experiment_name,
    get_tracer,
    trace_id_from_run_id,
)
from pra.screening.engine import TriageResult, rule_evidence, triage

logger = logging.getLogger(__name__)

__all__ = ["process_review", "run_and_persist", "run_screening_direct"]

# 单行 JSON 上限（防工具/LLM 巨行撑爆 review_trace 列的体量与可读性）。
_JSON_CAP = 64 * 1024

# 规则直判 run 的 trigger_type。
_TRIGGER_SCREENING_DIRECT = "SCREENING_DIRECT"

# 节点名 → review_trace.step_type 词汇表。tools 节点不落 "TOOLS" 行 —— 其 update 内每条
# tool_call_history record 各落一行 TOOL_CALL（执行审计粒度）。
_NODE_STEP_TYPE = {
    "hypothesize": "HYPOTHESIZE",
    "plan": "PLAN",
    "reevaluate": "REEVALUATE",
    "decide": "DECIDE",
    "tools": "TOOL_CALL",  # 仅供识别；实际每行取 TOOL_CALL（见 run_and_persist tools 特判）
}

# 模块级懒加载缓存（与 pra.api.service.get_graph 同款原子性论证：无 await 切换点）。
_compiled_graph: Any = None  # CompiledStateGraph（延迟 import 防 pra.infra 依赖链循环）


def _get_graph() -> Any:
    """懒加载返回编译图单例（默认 6 InMemory Tools + scripted LLM 桩，无 API key）。

    每次执行经唯一 ``thread_id=run_id`` 隔离线程状态（InMemorySaver）；真实 LLM / MySQL
    Checkpointer 接入时改这里即可。
    """
    global _compiled_graph
    if _compiled_graph is None:
        from pra.agent.graph import build_agent_graph  # 延迟 import：编译代价大、仅首次

        _compiled_graph = build_agent_graph(checkpointer=make_memory_checkpointer())
    return _compiled_graph


def _utcnow() -> datetime:
    """当前 naive UTC 时间（DB DATETIME 口径：去掉 tzinfo）。"""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _enum_value(v: Any) -> Any:
    """Enum → .value（普通值原样返回）；供 domain 模型字段摘要转 JSON 标量。"""
    return v.value if isinstance(v, Enum) else v


def _token_count(budget: Any) -> int:
    """从 budget 快照取 tokens（兼容 Pydantic Budget / JSON dict 两种形态）。"""
    if budget is None:
        return 0
    tokens = budget.tokens if not isinstance(budget, dict) else budget.get("tokens")
    return int(tokens or 0)


def _json_cap(obj: Any, cap: int = _JSON_CAP) -> Any:
    """把 JSON 可序列化对象装箱成 DB JSON 列可存形态；超长时退化为截断占位。

    正常路径 obj 为纯 dict/list/标量，tools 的 audit record 整存；意外超限存
    {"_truncated": True, "_preview": ...} 而非让 DB 行失控。
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    try:
        text = json.dumps(obj, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as exc:  # 理论上不可达（摘要均为标量/容器）
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
    """按节点构造 trace 输出摘要（小体积、可读、可统计），含 budget 快照供 step 级预算审计。"""
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
        queue = list(update.get("investigation_queue") or [])
        return {
            "node": node_name,
            "degraded": bool(update.get("degraded")),
            "hypotheses_count": len(hypos),
            "hypotheses": [_hypothesis_summary(h) for h in hypos],
            "investigation_queue_count": len(queue),
            "investigation_queue": [
                {"q": q.get("q"), "priority": q.get("priority"), "status": q.get("status")}
                for q in queue
            ],
            "budget": budget_snap,
        }
    if node_name == "plan":
        pending = list(update.get("pending_tool_calls") or [])
        skipped = list(update.get("tool_call_history") or [])  # dedup 跳过审计（并入本行）
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
        queue = list(update.get("investigation_queue") or [])
        return {
            "node": node_name,
            "degraded": bool(update.get("degraded")),
            "hypotheses_count": len(hypos),
            "hypotheses": [_hypothesis_summary(h) for h in hypos],
            "status_counts": status_counts,
            "queue_done_count": sum(1 for q in queue if q.get("status") == "DONE"),
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
    # 未知节点（不应出现）：兜底保留 update 键名与摘要，防静默丢步。
    return {
        "node": node_name,
        "update_keys": sorted(update.keys()),
        "budget": budget_snap,
    }


def _node_input_summary(node_name: str, update: dict, case_id: str) -> dict:
    """LLM 行 input_json —— 该步的轻量状态摘要（替代恒 ``{"case_id": ...}`` 占位）。

    ``stream_mode="updates"`` 下只有节点写入的 state 片段，故只取片段内可见的关键计数 +
    budget 占用，不引入 state 全量；输出侧明细由 output_json 承载。
    """
    return {
        "node": node_name,
        "case_id": case_id,
        "hypotheses_count": len(list(update.get("hypotheses") or [])),
        "investigation_queue_count": len(list(update.get("investigation_queue") or [])),
        "pending_tool_calls_count": len(list(update.get("pending_tool_calls") or [])),
        "budget_tokens": _token_count(update.get("budget")),
    }


def _evidence_row(
    session: Any,
    run_id: str,
    ev: Evidence,
    created_at: datetime | None = None,
) -> None:
    """把一条 Evidence 映射为 review_evidence ORM 行并 add（不 commit）。

    三处落库（extra_evidence / decision.evidence / 直判 hits）共用同一映射，ORM 加列只改这里。
    """
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
    """review_result 行 payload（图终态与直判共用）；decision_json = ReviewDecision 全量快照，列清单单点维护防漂移。"""
    return {
        "case_id": case_id,
        "source_run_id": source_run_id,
        "decision": decision.decision.value,
        "risk_level": decision.risk_level.value,
        "risk_type_json": [_enum_value(t) for t in decision.risk_type],
        "decision_confidence": decision.decision_confidence,
        "policy_refs_json": list(decision.policy),
        "decision_json": decision.model_dump(mode="json"),
        "created_at": _utcnow(),
        "updated_at": _utcnow(),
    }


async def _upsert_result(session: Any, payload: dict) -> None:
    """review_result upsert（last-writer-wins）；``created_at`` 只进 .values() —— 首裁时间不被覆盖。"""
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
            decision_json=payload["decision_json"],
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

    ``extra_evidence``（分流命中的 RULE_HIT）在 run 行建立后以本 run 的 run_id 落
    review_evidence —— 回答「哪条规则把 case 送进 Agent」，与直判路径的证据链同构。

    :param case: 审核案件；case_json 落 ``case.model_dump(mode="json")`` 全量快照。
    :param run_id: 本次运行 ID（= LangGraph thread_id）；None → ``uuid4().hex``。
    :param trigger_type: run 目的（INITIAL/RE_REVIEW/...）。
    :param triage_result: 非 None 时写 review_case.triage_result（创建与复用行均写）。
    :param extra_evidence: RULE_HIT Evidence 列表（type=RULE_HIT /
        source_tool=ScreeningRuleEngine / weight=1.0 / extra_json={"rule_id"}）；默认不写。
    :return: ``{"case_id", "run_id", "decision", "counts": {"trace", "evidence"}}`` ——
        decision 为图终态 ReviewDecision（供 API 包装返回），counts 为本 run 落库行数。

    里程碑（单 session 多 commit，保证断点可查）：case 行复用/新建 → run 行 INSERT
    （RUNNING）后 commit → extra_evidence 落库 → ``astream(stream_mode="updates")`` 每步落
    review_trace 并 commit（tools 节点每条 record 一行 TOOL_CALL，其余节点一行、step_type=
    节点名大写）→ ``aget_state`` 终态 decision 落 review_result + decision.evidence 全量落
    review_evidence → run/case 收尾 DECIDED。
    """
    resolved_run_id = run_id or uuid4().hex
    case_id = case.case_id
    now = _utcnow()

    sessionmaker = get_sessionmaker()  # async_sessionmaker 需调用生成 AsyncSession 才是 async CM
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
            case_row.status = "INVESTIGATING"  # 复用：刷新为调查中（不覆盖 case_json）
            if triage_result is not None:  # 分流结果随本次判定刷新
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

        # 分流命中证据（RULE_HIT）挂本 run：run 行建立后即落库并 commit —— 与 run 行同批
        # 里程碑，run 存在即可查到「为何进 Agent」。
        extra_rows = 0
        for ev in extra_evidence or []:
            _evidence_row(session, resolved_run_id, ev)
            extra_rows += 1
        if extra_rows:
            await session.commit()

        app = _get_graph()
        config = {"configurable": {"thread_id": resolved_run_id}}
        seq = 0
        trace_rows = 0
        prev_budget_tokens = 0  # 初始 state Budget().tokens == 0
        wall_prev = time.perf_counter()  # 节点到达墙钟基线（LLM 行 latency 近似）
        # Root trace：trace_id = run_id 映射 → Langfuse trace 与 MySQL review_run.run_id 硬
        # 对齐；root 覆盖整个 astream 直到终态可读。常驻服务不 per-request flush。
        root_ctx = TraceContext(
            trace_id=trace_id_from_run_id(resolved_run_id),
            name="review",
            session_id=None,
            version=experiment_name(),
            metadata={
                "case_id": case_id,
                "run_id": resolved_run_id,
                "event_type": case.event_type,
                "source": "http",
            },
            tags=["env:local", "source:http"],
            input={"case_id": case_id},
        )
        with get_tracer().trace_root(root_ctx) as root:
            async for chunk in app.astream(
                build_initial_state(case), config, stream_mode="updates"
            ):
                for node_name, update in chunk.items():
                    wall_now = time.perf_counter()
                    node_latency_ms = max(int((wall_now - wall_prev) * 1000), 0)
                    wall_prev = wall_now

                    budget = update.get("budget")
                    if budget is None:
                        # 无 budget 键（plan/reevaluate 短路）：tokens 记 0，但**不推进差分
                        # 基线** —— 否则基线被重置为 0，后续节点会把全量当差值重复计。
                        step_tokens = 0
                    else:
                        new_tokens = _token_count(budget)
                        step_tokens = max(new_tokens - prev_budget_tokens, 0)  # 差分：首步用其值
                        prev_budget_tokens = new_tokens  # 只对带 budget 的 update 更新基线

                    if node_name == "tools":
                        # tools 节点：每条 audit record 一行 TOOL_CALL（含边际增益 4 字段）；
                        # 无 record 的访问（预算截断空批等）不落行、seq 不推进。
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
                    await session.commit()  # 每 update 后 commit：中途崩溃可查已落 trace

            logger.info(
                "run_and_persist 图执行完成 case_id=%s run_id=%s trace_rows=%d",
                case_id, resolved_run_id, trace_rows,
            )

            snapshot = await app.aget_state(config)
            try:  # langgraph 1.2.11 StateSnapshot 为 NamedTuple（不可下标）
                final_state = snapshot["values"]  # type: ignore[index]
            except TypeError:
                final_state = snapshot.values  # type: ignore[attr-defined]

            decision = final_state.get("decision")
            if isinstance(decision, ReviewDecision):
                root.update(
                    output={
                        "decision": decision.decision.value,
                        "risk_level": decision.risk_level.value,
                    }
                )

        if not isinstance(decision, ReviewDecision):
            raise RuntimeError(
                f"调查图执行完成但终态缺少 decision（run_id={resolved_run_id}, "
                f"case_id={case_id}）—— 违反 'decide 为图唯一终态出口' 契约"
            )

        # 不变量：分流命中证据经上文已落本 run，且从不进 AgentState（build_initial_state
        # evidence=[]）—— decision.evidence 不应含同源行，否则 RULE_HIT 重复落库、
        # counts.evidence 双计。断言把「两集合不相交」变成可执行契约。
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

        evidence_rows = extra_rows  # 基数含前置 extra_evidence 分流命中行
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
    risk_type/policy/hypothesis_trace/overrides 一律空；evidence=命中规则证据全量。
    """
    return ReviewDecision(
        decision=Decision.PASS if verdict == "PASS" else Decision.REJECT,
        risk_level=RiskLevel.NONE if verdict == "PASS" else RiskLevel.HIGH,
        risk_type=[],
        decision_confidence=1.0,  # 确定性直判：置信 1.0（非模型概率语义）
        evidence=list(evidence),
        policy=[],
        hypothesis_trace=[],
        budget_used=Budget(),  # 直判无调查预算：默认空快照（含限额）
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
    :param triage: ``triage`` 产物，verdict ∈ {PASS, REJECT}；传 COMPLEX 抛 ValueError，
        防直判误入（COMPLEX 应走 ``run_and_persist``）。
    :param run_id: 本次判定运行 ID；None → ``uuid4().hex``。
    :return: ``{"case_id", "run_id", "verdict", "decision", "counts"}`` —— counts =
        {"evidence"} RULE_HIT 行数（PASS 零命中则 0）。

    单 session 单 commit（无 stream 断点语义）：case 行建/复用（status=DECIDED、
    triage_result=verdict）→ run 行（status=DECIDED、trigger_type=SCREENING_DIRECT、
    started_at≈ended_at、**无 review_trace 行**）→ hits 逐条写 review_evidence
    （type=RULE_HIT、source_tool=ScreeningRuleEngine、value=f"{rule_id} {name}: {detail}"、
    weight=1.0、extra_json={"rule_id"}）→ review_result upsert（risk_type_json=[]、
    decision_confidence=1.0、policy_refs_json=[]、source_run_id=本 run）→ commit。
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
            # 复用不覆盖 case_json
            case_row.status = "DECIDED"
            case_row.triage_result = triage.verdict
            case_row.updated_at = now

        run_row = ReviewRunORM(
            run_id=resolved_run_id,
            case_id=case_id,
            status="DECIDED",
            trigger_type=_TRIGGER_SCREENING_DIRECT,
            started_at=now,
            ended_at=now,  # started ≈ ended：直判无持续调查时段
        )
        session.add(run_row)

        evidence_rows = 0
        evidence_list: list[Evidence] = []
        for hit in triage.hits:
            ev = rule_evidence(case, hit)
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
        "decision": decision,  # 直判同构 ReviewDecision（process_review 直接取用）
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

    三分流不是建议：COMPLEX → ``run_and_persist``（triage_result='COMPLEX'）；PASS/REJECT
    → ``run_screening_direct``（确定性终裁）。
    """
    t = triage(case)
    if t.verdict == "COMPLEX":
        summary = await run_and_persist(
            case,
            run_id=run_id,
            triage_result="COMPLEX",
            # t.hits → RULE_HIT evidence：与直判路径同构，run_id 归属由 run_and_persist 落库。
            extra_evidence=[rule_evidence(case, hit) for hit in t.hits],
        )
        return {
            "case_id": summary["case_id"],
            "run_id": summary["run_id"],
            "verdict": t.verdict,
            "decision": summary["decision"],  # 图终态 ReviewDecision（如 HUMAN_REVIEW）
            "counts": dict(summary.get("counts") or {}),
        }

    summary = await run_screening_direct(case, t, run_id=run_id)
    return {
        "case_id": summary["case_id"],
        "run_id": summary["run_id"],
        "verdict": t.verdict,
        "decision": summary["decision"],  # 直判同构裁决（run_screening_direct 构造一次）
        "counts": dict(summary.get("counts") or {}),
    }
