"""AgentState 类型定义（TypedDict + reducer）—— 复杂风险调查 Agent 的状态契约。

AgentState 即 LangGraph 的 State（§3.1 设计要点 2 / §4.2 草图）：以 TypedDict 承载
调查主线的**显式、可序列化、可持久化、可恢复**记忆 —— 而不是藏在 LLM 上下文里。
LangGraph 的**线程 State Checkpointer**（MVP 用 InMemorySaver；Sqlite/Postgres/自研
见 docs/04-graph-design.md §7 选型 A）每步保存线程中间状态用于断点续跑/eval 重放；
业务真相（review_trace / review_evidence / review_result 等）由 worker 层显式落 MySQL（非 checkpointer
职责，04 §7.4 口径）。**reducer 已按 docs/04-graph-design.md §2.3 接线**（evidence
去重合并 / tool_call_history、failures append），初始状态由 ``build_initial_state``
构造 —— 每次 invoke 一律 ``await app.ainvoke(build_initial_state(case), {...})`` 保证
每个 channel 首读有值、reducer 首写安全（04 §2.3 invoke 约定）。

案件身份（run_id / case_id）**不放进本 State**：接入层映射为 LangGraph 的
**thread 维度（thread_id = run_id，O-6 拍板）**，由 Checkpointer 与调用方携带；
节点/工具需 run_id 时从节点第二参 ``config["configurable"]["thread_id"]`` 读取。

字段即 §4.1 Loop 每一环的产物：
- ``case``:            输入事实快照（起点，调查过程中不变；§2.1 ProductReviewCase）
- ``hypotheses``:      假设验证器状态（prior→posterior→status，§3.1 要点 1）
- ``evidence``:        结论依据（去重、合并后的证据链；与 tool_call_history 分离，§3.1 要点 4）
- ``investigation_queue``: 待验证问题队列（按优先级排序）
- ``tool_call_history``:   调用审计（谁、何时、调了哪个工具、tokens/latency + 边际增益 4 字段）
- ``budget``:          成本/延迟硬字段 —— 条件边路由函数每轮进入节点前检查，超限转人工（§3.1 要点 3 / §8.1）
- ``decision``:        收敛后写入的 ReviewDecision（调查中为 None）
- 图内通道（T-9 拍板保留，docs/03-decisions.md §2.7；01 §2.1 标〔细化新增〕）：
  ``pending_tool_calls`` 承载 plan→tools 的"本轮待执行工具调用"传递（覆盖写）；
  ``degraded`` 承载"上一 LLM 步 schema 校验失败"的降级信号（覆盖写）；
  ``failures`` 为步骤失败审计（append reducer；元素含 tool?/severity，O-3 拍板，见
  03-decisions §2.7 注记）——三者是让第 3~7 章契约可落地的图内部通道，不进最终决策
  输出；``run_id/case_id`` 按代码方案迁出为 thread 维度。

reducer 语义（docs/04-graph-design.md §2.3 速查表，代码即定义）：
- ``evidence`` = ``Annotated[list[Evidence], merge_evidence]``：自定义去重合并
  （key 规则 O-1：ref_id 优先稳定业务标识，无则回退 value；防同 (type, source)
  的多条无 ref 证据互相吞并），写入方 tools_node 只返回**新增**证据；
- ``tool_call_history`` / ``failures`` = ``Annotated[list[dict], add]``：append；
- 其余字段（hypotheses / investigation_queue / budget / decision / degraded /
  pending_tool_calls / case）= 覆盖写 —— 单路径线性链上每步只有一个合法写方，
  reducer 只服务于"追加/去重幂等"两类需求（checkpointer 断点重放、super-step
  重跑时重复执行不产生重复证据/记录）。
"""

from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict

from pra.domain import (
    Budget,
    Evidence,
    Hypothesis,
    ProductReviewCase,
    ReviewDecision,
)


def _evidence_key(e: Evidence) -> tuple:
    """证据去重指纹（O-1 已拍板，docs/04-graph-design.md §2.3）：
    ref_id 优先稳定业务标识（image_url / product_id / merchant_id / case_id /
    clause_id）；ref_id 为 None 时回退 value —— 防同 (type, source) 的多条无 ref
    证据（如多图多品牌命中）互相吞并。"""
    return (e.type, e.source, e.ref_id if e.ref_id is not None else e.value)


def merge_evidence(left: list[Evidence], right: list[Evidence]) -> list[Evidence]:
    """evidence channel reducer：按 _evidence_key 去重合并（左=当前 state，右=节点新增）。

    语义：已存在同 key → 丢弃新增（证据一旦收集不可篡改，重放/重试幂等）；新 key → append。
    left 可能为空/None（首写或空输入），按空列表处理。
    """
    seen: dict[tuple, Evidence] = {_evidence_key(e): e for e in left or []}
    out: list[Evidence] = list(left or [])
    for e in right or []:
        if _evidence_key(e) not in seen:
            seen[_evidence_key(e)] = e
            out.append(e)
    return out


class AgentState(TypedDict, total=False):
    """LangGraph State 的状态契约（§4.2 Python 草图，字段类型提升为 domain 模型）。

    值均为可 JSON 序列化的 domain 模型 / 原始 dict，供 Checkpointer 落库与 eval 重放。
    ``total=False``：所有 channel 可选，首读取值需依赖 ``build_initial_state`` 全量
    初始化（invoke 约定）；reducer channel（evidence 等）缺省时自动从空值开始累积。
    """

    case: ProductReviewCase  # 输入商品事实快照（调查起点，不改写）
    hypotheses: list[Hypothesis]  # 风险假设（可解释的 prior→posterior→status 演变）
    evidence: Annotated[list[Evidence], merge_evidence]  # 已收集证据（结论依据；自定义去重合并 reducer）
    investigation_queue: list[dict]  # 待验证问题，如 {"q", "priority", "status"}
    tool_call_history: Annotated[list[dict], add]  # 调用审计（append；含边际增益 4 字段，04 §4）
    budget: Budget  # 已用 + 限额（§8.1）；条件边路由的确定性检查对象（覆盖写，整对象）
    decision: ReviewDecision | None  # 收敛后的裁决；调查中为 None

    # ---- 图内通道（T-9 拍板保留）----
    pending_tool_calls: list[dict]  # plan 写、tools 消费后置 []；覆盖写；元素 {tool, args, reason, priority}
    degraded: bool  # 上一 LLM 步 schema 校验失败降级标记；覆盖写（True 后不再调 LLM，透传 decide）
    failures: Annotated[list[dict], add]  # 步骤失败审计 {step_type, tool?, severity, reason, ts}（O-3）；append reducer


def build_initial_state(case: ProductReviewCase) -> AgentState:
    """每次 invoke 的完整初始输入（docs/04-graph-design.md §2.3）。

    保证所有 channel 有值（首节点读不炸、reducer 首写安全）；``budget`` 默认
    10/15/40000/30000（BudgetLimits 默认，T-7 拍板 —— 接线时勿再覆盖回 8/12）。
    """
    return AgentState(
        case=case,
        hypotheses=[],
        evidence=[],
        investigation_queue=[],
        tool_call_history=[],
        budget=Budget(),
        decision=None,
        pending_tool_calls=[],
        degraded=False,
        failures=[],
    )
