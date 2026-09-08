"""落库编排（持久化服务）—— 「接入 → 落库闭环」worker 侧主入口（总链路 ② case 落库 + ⑪ 持久化）。

职责：把一次复杂风险调查（LangGraph 子图执行）的**业务真相**显式落 MySQL 五表
（review_case / review_run / review_trace / review_evidence / review_result）。
与线程 Checkpointer 的分工（选型 A，docs/04-graph-design.md §7.2/§7.4）：线程状态
checkpoint（InMemorySaver）只服务断点续跑/eval 重放；本模块负责把 agent_step
（review_trace）、evidence（review_evidence）、最终裁决（review_result）与运行/案件
状态机（review_run / review_case）落业务表 —— 业务真相在 MySQL，不在内存。

``run_and_persist`` 是**未来 MQ worker 复用的主入口**（总链路 A·1 演进：
HTTP 路由 → service.run_review；MQ/worker 化 → 消费 product_review_request 后调本函数）。
演进指引（对后续 worker 实施者）：
1. 幂等键策略：本函数用 ``run_id``（缺省 uuid4().hex）作 review_run 主键 —— worker
   若需消费幂等，可用消息/业务键派生确定性 run_id（如 f"{case_id}:{attempt}"），
   run_id PK 天然去重；case 维度幂等键（product_id+version）按 DDL 注记留待 API/
   worker 请求语义定，本函数按 case_id 复用 review_case 行（不重复建 case）。
2. 断点语义：run 行先于 stream 落库（status=RUNNING）；stream 每步 trace 即写即
   commit —— 中途崩溃（进程/DB/图异常）时已落 trace 可查、run 停留 RUNNING 可续跑；
   后续接自研 MySQL Checkpointer + Redis 幂等去重时可在此补"续跑探测"。
3. 状态机：run/case 收尾置 DECIDED；review_result 用 ``INSERT ... ON DUPLICATE KEY
   UPDATE`` 覆盖该 case 最新裁决（last-writer-wins；历史裁决/多 run 对比留待回流需求）。
4. 时间口径：所有时间列写 **naive UTC**（``_utcnow``，见 infra/db.py docstring），
   MySQL DATETIME(3) 存 naive；读取方按 UTC 解释 —— 勿混入本地时区。
5. 观测：review_trace 每行 = 一步（LLM 节点行 / 工具 TOOL_CALL 行，seq 连续唯一）；
   tokens 由相邻 budget 快照差分（首步用其值）、工具行取 record 自带 tokens；
   latency_ms 工具行走 record、LLM 行用节点到达墙钟差（近似）。真实 LLM/耗时接入后
   这些列即自洽，无需改表结构。

错误语义：图执行/落库中途异常向上抛出（HTTP 层转 500；未来 worker 转失败重试/死信）。
已 commit 的 run/trace 行保留（RUNNING/INVESTIGATING），供观测与续跑 —— 不在本函数
内吞异常或做部分回滚（crash 可查优先于 all-or-nothing）。
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
from pra.domain.models import Evidence, ProductReviewCase, ReviewDecision
from pra.infra.db import get_sessionmaker
from pra.infra.rdb_models import (
    ReviewCaseORM,
    ReviewEvidenceORM,
    ReviewResultORM,
    ReviewRunORM,
    ReviewTraceORM,
)

logger = logging.getLogger(__name__)

__all__ = ["run_and_persist"]

# 单行 JSON 上限（防工具/LLM 巨行撑爆 review_trace.output_json / input_json 可读性）。
_JSON_CAP = 64 * 1024

# 节点名 → review_trace.step_type 词汇表（DDL 注释：HYPOTHESIZE/PLAN/TOOL_CALL/
# REEVALUATE/DECIDE）。图内 tools 节点不落 "TOOLS" 行 —— 其 update 内每条
# tool_call_history record 各落一行 step_type=TOOL_CALL（执行审计粒度，含边际增益）。
_NODE_STEP_TYPE = {
    "hypothesize": "HYPOTHESIZE",
    "plan": "PLAN",
    "reevaluate": "REEVALUATE",
    "decide": "DECIDE",
    "tools": "TOOL_CALL",  # 仅供识别；实际每行取 TOOL_CALL（见 _write_trace 分发）
}


# ---------------------------------------------------------------------------
# 图单例（模块级懒加载，与 pra.api.service.get_graph 同款论证，见其并发说明）
# ---------------------------------------------------------------------------
_compiled_graph: Any = None  # CompiledStateGraph（延迟 import 防 pra.infra 依赖链循环）


def _get_graph() -> Any:
    """懒加载返回编译图单例（默认 6 InMemory Tools + scripted LLM 桩，无 API key）。

    复用同一编译图实例（省编译）；每次执行经唯一 ``thread_id=run_id`` 隔离线程状态
    （InMemorySaver）。真实 LLM / MySQL Checkpointer 演进时改这里即可，调用方零改动。
    """
    global _compiled_graph
    if _compiled_graph is None:
        from pra.agent.graph import build_agent_graph  # 延迟 import：编译代价大、仅首次

        _compiled_graph = build_agent_graph(checkpointer=make_memory_checkpointer())
    return _compiled_graph


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """当前 naive UTC 时间（DB DATETIME 口径：去掉 tzinfo，存 naive UTC，见 db.py）。"""
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

    正常路径 obj 为纯 dict/list/标量（已由各摘要构造保证可序列化）；tools 的 audit
    record 直接整存（含边际增益 4 字段，O-10）。若意外超限（防未来字段膨胀成巨行），
    存 {"_truncated": True, "_preview": ...} 而非让 DB 行失控。
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


# ---------------------------------------------------------------------------
# 每步 trace 摘要构造（output_json 内容；LLM 行为紧凑摘要，不做全量巨行）
# ---------------------------------------------------------------------------


def _hypothesis_summary(h: Any) -> dict:
    """Hypothesis 摘要（id/statement/status/prior/posterior；JSON 标量化）。"""
    return {
        "id": h.id,
        "statement": h.statement,
        "status": _enum_value(h.status),
        "prior": h.prior,
        "posterior": h.posterior,
    }


def _node_output_summary(node_name: str, update: dict, case_id: str) -> dict:
    """按节点构造 trace 输出摘要（保持小体积、可读、可统计）。

    摘要含 budget 快照（llm_calls/tool_calls/tokens）供 step 级预算审计 —— 该摘要为
    每步 output_json 的正文（tools 节点不在此构造，其 output_json = 单条 record）。
    """
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


# ---------------------------------------------------------------------------
# 主入口：执行并落库
# ---------------------------------------------------------------------------


async def run_and_persist(
    case: ProductReviewCase,
    *,
    run_id: str | None = None,
    trigger_type: str = "INITIAL",
) -> dict:
    """执行一次完整复杂风险调查并把业务真相落 MySQL 五表 —— **MQ worker 复用主入口**。

    :param case: 审核案件（domain 输入 DTO）。case_id 即 review_case 主键；
        case_json 落 ``case.model_dump(mode="json")`` 全量快照。
    :param run_id: 本次运行 ID（= LangGraph thread_id，O-6）。None → 自动
        ``uuid4().hex``；worker 幂等场景可传确定性 run_id（见模块 docstring 演进指引）。
    :param trigger_type: run 目的（INITIAL/RE_REVIEW/...），落 review_run.trigger_type。
    :return: 摘要 dict：::

            {
                "case_id": str,
                "run_id": str,
                "decision": ReviewDecision,          # 图终态裁决对象（供 API 包装返回）
                "counts": {"trace": int, "evidence": int},  # 本 run 落库行数
            }

    流程（单 session 多 commit，里程碑保证断点可查）：
    a. case 行：按 case_id SELECT；不存在 INSERT（status=INVESTIGATING、case_json=全量
       快照、product_id/merchant_id/event_type/version 照填）；已存在则**复用不覆盖
       case_json**（快照=首投内容，防上游漂移），仅刷新 status=INVESTIGATING。
    b. run 行：INSERT（RUNNING + trigger_type + started_at）后 commit —— run 先落库，
       保证 stream 途中崩溃可查到 run 与已落 trace。
    c. ``astream(..., stream_mode="updates")`` 逐步执行：每节点按 seq（1 起连续自增）
       落 review_trace；tools 节点内 update["tool_call_history"] 每条 record 各一行
       step_type=TOOL_CALL（tool_name=record["tool"]、output_json=record 本身含边际
       增益 4 字段、input_json=args）；其余节点一行 step_type=节点名大写、output_json=
       紧凑摘要。tokens：相邻 budget 快照差分（首步用其值）；latency_ms：工具行走
       record、LLM 行用节点到达墙钟差（近似）。每个 update 后 commit。
    d. stream 结束后 ``aget_state(config)`` 取终态；st["decision"] 落 review_result
       （INSERT ... ON DUPLICATE KEY UPDATE 覆盖该 case 最新裁决）；decision.evidence
       全量落 review_evidence（挂本 run_id，多 run 证据隔离）。
    e. 收尾：run.status=DECIDED + ended_at、case.status=DECIDED + updated_at；commit。
    """
    resolved_run_id = run_id or uuid4().hex
    case_id = case.case_id
    now = _utcnow()

    sessionmaker = get_sessionmaker()  # async_sessionmaker 需调用生成 AsyncSession 才是 async CM
    async with sessionmaker() as session:
        # ---- a. case 行（复用不重建；复用时不覆盖 case_json）----
        case_row = await session.get(ReviewCaseORM, case_id)
        if case_row is None:
            case_row = ReviewCaseORM(
                case_id=case_id,
                product_id=case.product.product_id,
                merchant_id=case.merchant_id,
                event_type=case.event_type,
                status="INVESTIGATING",
                version=case.product.version,
                case_json=case.model_dump(mode="json"),
                created_at=now,
                updated_at=now,
            )
            session.add(case_row)
        else:
            case_row.status = "INVESTIGATING"  # 复用：刷新为调查中（不覆盖 case_json）
            case_row.updated_at = now

        # ---- b. run 行先落库并 commit（断点可查的锚点）----
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

        # ---- c. stream 逐步执行并落 review_trace ----
        app = _get_graph()
        config = {"configurable": {"thread_id": resolved_run_id}}
        seq = 0
        trace_rows = 0
        prev_budget_tokens = 0  # 初始 state Budget().tokens == 0
        wall_prev = time.perf_counter()  # 节点到达墙钟基线（LLM 行 latency 近似）
        async for chunk in app.astream(
            build_initial_state(case), config, stream_mode="updates"
        ):
            for node_name, update in chunk.items():
                wall_now = time.perf_counter()
                node_latency_ms = max(int((wall_now - wall_prev) * 1000), 0)
                wall_prev = wall_now

                budget = update.get("budget")
                new_tokens = _token_count(budget)
                step_tokens = max(new_tokens - prev_budget_tokens, 0)  # 首步用其值
                prev_budget_tokens = new_tokens

                if node_name == "tools":
                    # tools 节点：每条 audit record 一行 TOOL_CALL（含边际增益 4 字段）
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
                    # 无 record 的 tools 访问（预算截断空批等）：不落行，seq 不推进
                else:
                    seq += 1
                    session.add(
                        ReviewTraceORM(
                            run_id=resolved_run_id,
                            seq=seq,
                            step_type=_NODE_STEP_TYPE.get(node_name, node_name.upper()),
                            tool_name=None,
                            input_json={"case_id": case_id},
                            output_json=_json_cap(
                                _node_output_summary(node_name, update, case_id)
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

        # ---- d. 终态：review_result upsert + review_evidence 落库 ----
        snapshot = await app.aget_state(config)
        try:  # langgraph 1.2.11 StateSnapshot 为 NamedTuple（不可下标）
            final_state = snapshot["values"]  # type: ignore[index]
        except TypeError:
            final_state = snapshot.values  # type: ignore[attr-defined]

        decision: ReviewDecision | None = final_state.get("decision")
        if decision is None:
            raise RuntimeError(
                f"调查图执行完成但终态缺少 decision（run_id={resolved_run_id}, "
                f"case_id={case_id}）—— 违反 'decide 为图唯一终态出口' 契约"
            )

        # evidence 全量（state 全量与 decision.evidence 同型，以裁决链为准）
        evidence_rows = 0
        for ev in decision.evidence:  # type: Evidence
            session.add(
                ReviewEvidenceORM(
                    run_id=resolved_run_id,
                    type=ev.type,
                    source_tool=ev.source,
                    value=ev.value,
                    weight=ev.weight,
                    ref_id=ev.ref_id,
                    extra_json=_json_cap(ev.extra) if ev.extra else None,
                    created_at=_utcnow(),
                )
            )
            evidence_rows += 1

        result_payload = {
            "case_id": case_id,
            "source_run_id": resolved_run_id,
            "decision": decision.decision.value,
            "risk_level": decision.risk_level.value,
            "risk_type_json": [_enum_value(t) for t in decision.risk_type],
            "decision_confidence": decision.decision_confidence,
            "policy_refs_json": list(decision.policy),
            "decision_json": decision.model_dump(mode="json"),  # ReviewDecision 全量快照
            "created_at": _utcnow(),
            "updated_at": _utcnow(),
        }
        stmt = (
            mysql_insert(ReviewResultORM)
            .values(**result_payload)
            .on_duplicate_key_update(
                source_run_id=result_payload["source_run_id"],
                decision=result_payload["decision"],
                risk_level=result_payload["risk_level"],
                risk_type_json=result_payload["risk_type_json"],
                decision_confidence=result_payload["decision_confidence"],
                policy_refs_json=result_payload["policy_refs_json"],
                decision_json=result_payload["decision_json"],
                updated_at=result_payload["updated_at"],
            )
        )
        await session.execute(stmt)

        # ---- e. 收尾状态机：run/case 置 DECIDED ----
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
