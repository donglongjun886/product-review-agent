"""落库编排（持久化服务）—— 「接入 → 落库闭环」worker 侧主入口（总链路 ② case 落库 + ⑪ 持久化）。

职责：把一次**审核判定活动**的**业务真相**显式落 MySQL 五表
（review_case / review_run / review_trace / review_evidence / review_result）。
判定活动有两种执行方式（拍板 D：review_run 语义重定义为"一次审核判定活动"）：
- **Agent 调查**：``run_and_persist`` —— LangGraph 子图执行（trigger_type=
  INITIAL/RE_REVIEW），逐节点落 review_trace；
- **规则直判**：``run_screening_direct`` —— Screening 三分流 PASS/REJECT 确定性
  直接终裁（trigger_type="SCREENING_DIRECT"），无 trace 行、started_at≈ended_at、
  status 直接 DECIDED；规则命中证据（RULE_HIT）挂直判 run 的 review_evidence，
  终裁写 review_result（source_run_id=直判 run）—— 保证 ``result → run → evidence``
  审计链对两类裁决统一成立。

``process_review`` 是 **POST /api/v1/reviews 主入口（受理即分流）**：对 case 先做
Screening 三分流（``pra.screening.engine.triage``）—— COMPLEX → 走 run_and_persist
（Agent 调查，case.triage_result='COMPLEX'；触发进 Agent 的规则命中 RULE_HIT 证据随
agent run 挂 review_evidence —— 审计对称）；PASS/REJECT → 走 run_screening_direct
（规则直判终裁）。``run_and_persist`` 亦为**未来 MQ worker 复用的 Agent 调查主入口**
（总链路 A·1 演进：HTTP 路由 → process_review；MQ/worker 化 → 消费
product_review_request 后对投递 case 先 triage 再按 verdict 分支调本模块）。
演进指引（对后续 worker 实施者）：
1. 幂等键策略：本模块用 ``run_id``（缺省 uuid4().hex）作 review_run 主键 —— worker
   若需消费幂等，可用消息/业务键派生确定性 run_id（如 f"{case_id}:{attempt}"），
   run_id PK 天然去重；case 维度幂等键（product_id+version）按 DDL 注记留待 API/
   worker 请求语义定，本模块按 case_id 复用 review_case 行（不重复建 case）。
2. 断点语义（Agent 路径）：run 行先于 stream 落库（status=RUNNING）；stream 每步
   trace 即写即 commit —— 中途崩溃（进程/DB/图异常）时已落 trace 可查、run 停留
   RUNNING 可续跑；后续接自研 MySQL Checkpointer + Redis 幂等去重时可在此补"续跑
   探测"。规则直判路径无 stream，单 commit 原子落库即可。
3. 状态机：run/case 收尾置 DECIDED（直判 run 建行即 DECIDED）；review_result 用
   ``INSERT ... ON DUPLICATE KEY UPDATE`` 覆盖该 case 最新裁决（last-writer-wins；
   历史裁决/多 run 对比留待回流需求）。
4. 时间口径：所有时间列写 **naive UTC**（``_utcnow``，见 infra/db.py docstring），
   MySQL DATETIME(3) 存 naive；读取方按 UTC 解释 —— 勿混入本地时区。
5. 观测：review_trace 每行 = 一步（LLM 节点行 / 工具 TOOL_CALL 行，seq 连续唯一）；
   tokens 由相邻 budget 快照差分（首步用其值；节点 update 无 budget 键的短路步记 0
   且**不推进基线**，防 0 重置致后续全量重复计）、工具行取 record 自带 tokens；
   latency_ms 工具行走 record、LLM 行用节点到达墙钟差（近似）。真实 LLM/耗时接入后
   这些列即自洽，无需改表结构。

错误语义：图执行/落库中途异常向上抛出（HTTP 层转 500；未来 worker 转失败重试/死信）。
已 commit 的 run/trace 行保留（RUNNING/INVESTIGATING），供观测与续跑 —— 不在本模块
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

# 单行 JSON 上限（防工具/LLM 巨行撑爆 review_trace.output_json / input_json 可读性）。
_JSON_CAP = 64 * 1024

# 规则直判 run 的 trigger_type（拍板 D：规则直判 = review_run 的一种执行方式）。
_TRIGGER_SCREENING_DIRECT = "SCREENING_DIRECT"

# 节点名 → review_trace.step_type 词汇表（DDL 注释：HYPOTHESIZE/PLAN/TOOL_CALL/
# REEVALUATE/DECIDE）。图内 tools 节点不落 "TOOLS" 行 —— 其 update 内每条
# tool_call_history record 各落一行 step_type=TOOL_CALL（执行审计粒度，含边际增益）。
_NODE_STEP_TYPE = {
    "hypothesize": "HYPOTHESIZE",
    "plan": "PLAN",
    "reevaluate": "REEVALUATE",
    "decide": "DECIDE",
    "tools": "TOOL_CALL",  # 仅供识别；实际每行取 TOOL_CALL（见 run_and_persist c 段 tools 特判分发）
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


def _node_output_summary(node_name: str, update: dict) -> dict:
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


def _node_input_summary(node_name: str, update: dict, case_id: str) -> dict:
    """LLM 行 input_json —— 该步的**轻量状态摘要**（MVP 口径，对齐 DDL 001 注释
    "LLM 步=状态摘要"；替代恒 ``{"case_id": ...}`` 占位，输入侧才有审计价值）。

    ``stream_mode="updates"`` 下本处只有节点写入的 state 片段（update dict；覆盖写
    channel 的节点其结果≈该步处理后的状态），故摘要只取片段内**可见的关键计数 +
    budget 占用**（假设 / 调查队列 / 待执行工具数 / tokens 占用），不引入 state
    全量、不为摘要加新逻辑。输出侧明细（假设列表/裁决等）由 output_json 承载；
    真实完整 prompt 输入审计随真实 LLM 接入后补。
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
    """把一条 Evidence 映射为 review_evidence ORM 行并 add（不 commit；commit 节奏由调用方控制）。

    三处落库（extra_evidence / decision.evidence / 直判 hits）共用同一映射 ——
    ORM 加列只改这里，杜绝逐字段复制漂移（persist review 建议 1）。
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
    """review_result 行 12 键 payload（Agent 图终态与直判 _direct_decision 共用）。

    decision_json = ReviewDecision 全量快照（同构列）；列清单单点维护 ——
    原两处逐字重复的 upsert 链是 DDL 演进最高危漂移点（persist review 建议 2）。
    """
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
    """review_result upsert（last-writer-wins 覆盖该 case 最新裁决，语义同原内联实现）。

    ``created_at`` 只进 .values() 不进 update 子句 —— 首裁时间不被覆盖。
    """
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


# ---------------------------------------------------------------------------
# 主入口：执行并落库
# ---------------------------------------------------------------------------


async def run_and_persist(
    case: ProductReviewCase,
    *,
    run_id: str | None = None,
    trigger_type: str = "INITIAL",
    triage_result: str | None = None,
    extra_evidence: list[Evidence] | None = None,
) -> dict:
    """执行一次完整复杂风险调查（**Agent 调查路径**）并把业务真相落 MySQL 五表。

    ``process_review`` 分流后 verdict=COMPLEX 时调用本函数（triage_result="COMPLEX"）；
    本函数保持 Agent 调查路径语义不变（图执行 + trace/evidence/result 落库），并新增：
    - case 行写 ``triage_result``（拍板：COMPLEX 时记录 —— 回答"case 为什么进 Agent"）；
    - ``extra_evidence``（分流命中 RULE_HIT）在 run 行建立后即以本 run 的 run_id 落
      review_evidence —— 回答"**哪条规则**把 case 送进 Agent"，与直判路径的
      RULE_HIT 证据链同构（审计对称，见 Fix 4）。

    :param case: 审核案件（domain 输入 DTO）。case_id 即 review_case 主键；
        case_json 落 ``case.model_dump(mode="json")`` 全量快照。
    :param run_id: 本次运行 ID（= LangGraph thread_id，O-6）。None → 自动
        ``uuid4().hex``；worker 幂等场景可传确定性 run_id（见模块 docstring 演进指引）。
    :param trigger_type: run 目的（INITIAL/RE_REVIEW/...），落 review_run.trigger_type。
    :param triage_result: Screening 分流结果（COMPLEX/PASS/REJECT），非 None 时写
        review_case.triage_result（创建与复用 case 行均写）。
    :param extra_evidence: 前置分流命中证据（RULE_HIT Evidence 列表，由
        ``process_review`` 按 ``triage().hits`` 经 ``rule_evidence`` 构造，与直判路径
        同构）；run 行创建后、以本 run 的 run_id 落 review_evidence（type=RULE_HIT /
        source_tool=ScreeningRuleEngine / weight=1.0 / extra_json={"rule_id"}）。
        None 或空列表 = 不写。默认 None。
    :return: 摘要 dict：::

            {
                "case_id": str,
                "run_id": str,
                "decision": ReviewDecision,          # 图终态裁决对象（供 API 包装返回）
                "counts": {"trace": int, "evidence": int},  # 本 run 落库行数（含 extra_evidence）
            }

    流程（单 session 多 commit，里程碑保证断点可查）：
    a. case 行：按 case_id SELECT；不存在 INSERT（status=INVESTIGATING、case_json=全量
       快照、product_id/merchant_id/event_type/version 照填，triage_result=参数值）；
       已存在则**复用不覆盖 case_json**（快照=首投内容，防上游漂移），仅刷新
       status=INVESTIGATING（与 triage_result，参数非 None 时）。
    b. run 行：INSERT（RUNNING + trigger_type + started_at）后 commit —— run 先落库，
       保证 stream 途中崩溃可查到 run 与已落 trace。
    b'. extra_evidence：run 行 commit 后即以本 run 的 run_id 逐条落 review_evidence
       （RULE_HIT 分流命中证据），并 commit —— "为何进 Agent" 与 run 行同批里程碑可查。
    c. ``astream(..., stream_mode="updates")`` 逐步执行：每节点按 seq（1 起连续自增）
       落 review_trace；tools 节点内 update["tool_call_history"] 每条 record 各一行
       step_type=TOOL_CALL（tool_name=record["tool"]、output_json=record 本身含边际
       增益 4 字段、input_json=args）；其余节点一行 step_type=节点名大写、
       input_json=轻量状态摘要（``_node_input_summary``：update 可见计数 + budget
       占用）、output_json=紧凑摘要。tokens：相邻 budget 快照差分（首步用其值；
       update 无 budget 键的短路步记 0 且不推进基线）；latency_ms：工具行走 record、
       LLM 行用节点到达墙钟差（近似）。每个 update 后 commit。
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

        # ---- b'. 分流命中证据（RULE_HIT）挂本 run：run 行建立后即落库并 commit ----
        # 审计对称（Fix 4）：COMPLEX 进 Agent 前由哪条规则命中（t.hits）也留档 ——
        # 与直判路径逐条挂 RULE_HIT 到 review_evidence 同构；run_id 归属 = 本 run
        # （resolved_run_id，含 process_review 未传 run_id 时内部生成的情形）。
        extra_rows = 0
        for ev in extra_evidence or []:
            _evidence_row(session, resolved_run_id, ev)
            extra_rows += 1
        if extra_rows:
            await session.commit()  # 与 run 行同批里程碑：run 存在即可查到命中证据

        # ---- c. stream 逐步执行并落 review_trace ----
        app = _get_graph()
        config = {"configurable": {"thread_id": resolved_run_id}}
        seq = 0
        trace_rows = 0
        prev_budget_tokens = 0  # 初始 state Budget().tokens == 0
        wall_prev = time.perf_counter()  # 节点到达墙钟基线（LLM 行 latency 近似）
        # Root trace（docs/09 §4.1 落点 2）：trace_id = run_id 映射（32-hex 原样，
        # 否则确定性 uuid5）→ Langfuse trace 与 MySQL review_run.run_id 硬对齐。
        # root 覆盖整个 astream 直到终态可读（output 在终态写回）；常驻服务不
        # per-request flush（缓冲由 SDK 后台批量上报）。
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
                        # update 无 budget 键（plan/reevaluate 短路只返回 pending_tool_calls/
                        # {} 等）：tokens 记 0，但**不推进差分基线** —— 否则基线被重置为 0，
                        # 后续带真实 budget 的节点会把全量当差值重复计（差分口径失真）。
                        step_tokens = 0
                    else:
                        new_tokens = _token_count(budget)
                        step_tokens = max(new_tokens - prev_budget_tokens, 0)  # 差分：首步用其值
                        prev_budget_tokens = new_tokens  # 只对带 budget 的 update 更新基线

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

            # ---- d. 终态：review_result upsert + review_evidence 落库 ----
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

        # 不变量（persist review P2-2 收口）：分流命中证据（extra_evidence）经 b'
        # 已落本 run，且从不进 AgentState（build_initial_state evidence=[]）——
        # decision.evidence 不应含同源行，否则 RULE_HIT 重复落库、counts.evidence
        # 双计。显式断言把"两集合不相交"变成可执行契约（未来若图内引入 Screening
        # 来源证据会在此炸响，而不是静默重复落库）。
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

        # evidence 全量（state 全量与 decision.evidence 同型，以裁决链为准；
        # 基数含前置 extra_evidence 分流命中行）
        evidence_rows = extra_rows
        for ev in decision.evidence:
            _evidence_row(session, resolved_run_id, ev)
            evidence_rows += 1

        await _upsert_result(
            session, _result_payload(decision, case_id, resolved_run_id)
        )

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


# ---------------------------------------------------------------------------
# Screening 规则直判（trigger_type=SCREENING_DIRECT）与 POST 受理入口
# ---------------------------------------------------------------------------


def _direct_decision(verdict: str, evidence: list[Evidence]) -> ReviewDecision:
    """构造规则直判的同构 ReviewDecision（确定性直判：confidence=1.0）。

    PASS → decision=PASS / risk_level=NONE；REJECT → decision=REJECT / risk_level=HIGH。
    risk_type/policy/hypothesis_trace/overrides 一律空（v1 直判不产出政策引用与假设
    轨迹）；evidence=命中规则证据（RULE_HIT）全量。与 review_result 的
    decision_json 快照同构（decision_json=本对象 model_dump(mode="json")）。
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
    """Screening 规则直判（PASS/REJECT 确定性直接终裁）落库 —— **Agent 路径之外
    的第二类判定活动**（trigger_type=SCREENING_DIRECT）。

    :param case: 审核案件（同 run_and_persist：case_id 即主键；case_json 全量快照）。
    :param triage: ``pra.screening.engine.triage`` 产物，verdict ∈ {PASS, REJECT}
        （调用方须先分流确认非 COMPLEX —— COMPLEX 应走 run_and_persist Agent 路径；
        传 COMPLEX 在此抛 ValueError，防直判误入）。
    :param run_id: 本次判定运行 ID；None → ``uuid4().hex``。
    :return: 摘要 dict：::

            {
                "case_id": str,
                "run_id": str,
                "verdict": "PASS" | "REJECT",
                "decision": ReviewDecision,    # 直判同构裁决（供 API 包装返回）
                "counts": {"evidence": int},   # RULE_HIT 行数（PASS 零命中则 0）
            }

    落库（单 session 单 commit，无 stream 断点语义）：
    a. case 行：不存在 INSERT（status=DECIDED —— 直判无调查中间态、triage_result=
       verdict、case_json 全量快照）；已存在则复用（status=DECIDED、triage_result=
       verdict 刷新，不覆盖 case_json）。
    b. run 行：INSERT（status=DECIDED、trigger_type=SCREENING_DIRECT、started_at≈
       ended_at=now）—— **无 review_trace 行**。
    c. hits 逐条写 review_evidence（type=RULE_HIT、source_tool=ScreeningRuleEngine、
       value=f"{rule_id} {name}: {detail}"、weight=1.0、extra_json={"rule_id"}）。
    d. review_result upsert（decision=PASS/REJECT、risk_level=NONE/HIGH、
       risk_type_json=[]、decision_confidence=1.0、policy_refs_json=[]、
       decision_json=同构 ReviewDecision 快照、source_run_id=本 run）。
    e. commit。返回 case/run/verdict/counts。
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
        # ---- a. case 行（直判建行即终裁；复用不覆盖 case_json）----
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

        # ---- b. run 行（status 直接 DECIDED、无 trace）----
        run_row = ReviewRunORM(
            run_id=resolved_run_id,
            case_id=case_id,
            status="DECIDED",
            trigger_type=_TRIGGER_SCREENING_DIRECT,
            started_at=now,
            ended_at=now,  # started ≈ ended：直判无持续调查时段
        )
        session.add(run_row)

        # ---- c. hits → review_evidence（PASS 零命中则 0 行）----
        evidence_rows = 0
        evidence_list: list[Evidence] = []
        for hit in triage.hits:
            ev = rule_evidence(case, hit)
            evidence_list.append(ev)
            _evidence_row(session, resolved_run_id, ev)
            evidence_rows += 1

        # ---- d. review_result（source_run_id=直判 run；decision_json 同构快照）----
        decision = _direct_decision(triage.verdict, evidence_list)
        await _upsert_result(
            session, _result_payload(decision, case_id, resolved_run_id)
        )

        # ---- e. commit ----
        await session.commit()

    logger.info(
        "run_screening_direct 直判落库 case_id=%s run_id=%s verdict=%s evidence=%d",
        case_id, resolved_run_id, triage.verdict, evidence_rows,
    )
    return {
        "case_id": case_id,
        "run_id": resolved_run_id,
        "verdict": triage.verdict,
        "decision": decision,  # 直判同构 ReviewDecision（process_review 直接取用，P3-8）
        "counts": {"evidence": evidence_rows},
    }


async def process_review(
    case: ProductReviewCase,
    *,
    run_id: str | None = None,
) -> dict:
    """POST /api/v1/reviews 主入口 —— **受理即分流**（Screening 三分流先行再分支）。

    :param case: 审核案件（domain 输入 DTO）。
    :param run_id: 运行 ID；None → 各分支自动 ``uuid4().hex``。
    :return: 归一化摘要 dict：::

            {
                "case_id": str,
                "run_id": str,
                "verdict": "PASS" | "REJECT" | "COMPLEX",
                "decision": ReviewDecision,   # 图终态或直判同构裁决（供 API 包装返回）
                "counts": {...},              # COMPLEX: {trace, evidence}；直判: {evidence}
            }

    分流语义（拍板：三分流不是分流建议，PASS/REJECT 即终裁）：
    - ``triage(case).verdict == COMPLEX`` → ``run_and_persist(case, run_id=...,
      triage_result="COMPLEX", extra_evidence=t.hits 的 RULE_HIT evidence)`` —— Agent
      调查（case.triage_result='COMPLEX'、run trigger_type=INITIAL/RE_REVIEW；触发进
      Agent 的规则命中证据随 agent run 挂 review_evidence，审计对称，见 Fix 4）；
    - 否则（PASS/REJECT）→ ``run_screening_direct(case, t, run_id=...)`` —— 确定性
      规则直判终裁（无 trace、status 直接 DECIDED、result.source_run_id=直判 run）。
    """
    t = triage(case)
    if t.verdict == "COMPLEX":
        summary = await run_and_persist(
            case,
            run_id=run_id,
            triage_result="COMPLEX",
            # t.hits → RULE_HIT evidence：与直判路径（run_screening_direct 逐条挂
            # review_evidence）同构，run_id 归属由 run_and_persist 内部解析后落库。
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
