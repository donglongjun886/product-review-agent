"""graph 走查脚本：端到端驱动 LangGraph 并断言终态。

流程 hypothesize → plan → tools → reevaluate ×N → decide，逐步打印节点关键内容；
流结束后读终态、跑全量断言，末尾打印决策摘要块。

要点：
- 直接可执行：``uv run python scripts/demo_walkthrough.py``（默认 6 tools + scripted
  LLM 桩，无 API key）。
- ``build_agent_graph(checkpointer=make_memory_checkpointer())``；每次 stream 传
  ``build_initial_state(case)``；checkpointer 场景下终态用
  ``(await app.aget_state(config))["values"]`` 读取。兼容注记：仓库锁定
  langgraph 1.2.11，其 ``StateSnapshot`` 是 **NamedTuple**（``snap["values"]`` 抛
  TypeError），故先按字面写法、失败后回退 ``snapshot.values`` 属性。
- 期望结局：HUMAN_REVIEW / HIGH / [POTENTIAL_IP_RISK, EVASION_PATTERN] / dc>=0.7；
  evidence 覆盖 IMAGE_SIMILARITY(similarity=0.91) + MERCHANT_HISTORY + CASE_PRECEDENT
  + POLICY_REF；llm_calls=8、tool_calls=5（限值 10/15）；overrides=[]；decide 是唯一出口。
- ``pra.agent.graph`` 延迟到 ``_build_app()`` 内 import，未落盘时给出明确报错。
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from pra.agent.checkpointer import make_memory_checkpointer
from pra.agent.state import build_initial_state
from pra.domain.models import (
    Decision,
    HypothesisStatus,
    ProductImage,
    ProductInfo,
    ProductReviewCase,
    RiskLevel,
    RiskType,
    ScreeningSignal,
    SkuInfo,
)

# 断言清单里的受控常量（与 pra.domain.models / 工具证据类型对齐）
_IMG_URL = "https://cdn.example.com/products/P_88231/img1.jpg"
_REQUIRED_EVIDENCE_TYPES = frozenset(
    {"IMAGE_SIMILARITY", "MERCHANT_HISTORY", "CASE_PRECEDENT", "POLICY_REF"}
)
_EXPECTED_LLM_CALLS = 8
_EXPECTED_TOOL_CALLS = 5


# case 构造（P_88231 / M_5512 / NEW_LISTING）


def build_demo_case() -> ProductReviewCase:
    """构造走查输入 case（brand=None、无品牌词、img1、version=3）。"""
    product = ProductInfo(
        product_id="P_88231",
        title="新款厚底复古跑鞋 女士百搭运动鞋",
        description="复古厚底设计，舒适百搭，适合日常通勤与运动。",  # 无品牌词
        category="女鞋/运动鞋",
        brand=None,  # 品牌真空缺 —— 规避品牌识别调查的起点信号
        sku_list=[SkuInfo(sku_id="S_1", color="米白", size="38", price=219.0)],
        images=[ProductImage(url=_IMG_URL, source="主图")],
        listing_time=datetime(2024, 9, 6, 14, 0, 0),  # naive datetime（DB DATETIME 口径）
        version=3,
    )
    signals = [
        ScreeningSignal(name="KEYWORD", result="PASS", score=0.80),
        ScreeningSignal(name="LOGO_DETECT", result="PASS", score=0.20),
        ScreeningSignal(name="CATEGORY_RULE", result="PASS", score=0.95),
        ScreeningSignal(name="DUPLICATE_CHECK", result="PASS", score=0.10),
    ]
    return ProductReviewCase(
        case_id="CASE_20240907_001",
        product=product,
        merchant_id="M_5512",
        event_type="NEW_LISTING",
        screening_signals=signals,
    )


# 每步节点关键内容打印（按序打印节点名 + 关键内容）


def _print_hypothesize(update: dict) -> list:
    """hypothesize 步：假设数与 prior 摘要。返回本次假设列表（供 reevaluate 比对）。"""
    hypos = list(update.get("hypotheses") or [])
    queue = list(update.get("investigation_queue") or [])
    print(f"[hypothesize] 生成 {len(hypos)} 条假设、{len(queue)} 个调查问题")
    for h in hypos:
        prior = "-" if h.prior is None else f"{h.prior:.2f}"
        print(f"    {h.id} [{h.status.value}, prior={prior}] {h.statement}")
    return hypos


def _print_plan(update: dict) -> None:
    """plan 步：planned 工具名（及 dedup 跳过数）。"""
    pending = list(update.get("pending_tool_calls") or [])
    skipped = [r for r in (update.get("tool_call_history") or []) if r.get("status") == "skipped"]
    if not pending:
        print("[plan] 无可计划工具调用（conclude / 空计划）")
    else:
        print(f"[plan] 计划 {len(pending)} 个工具调用:")
        for c in pending:
            reason = c.get("reason") or ""
            print(f"    - {c.get('tool')} (priority={c.get('priority')}) {reason}")
    if skipped:
        print(f"    (dedup 跳过 {len(skipped)} 条重复调用)")


def _print_tools(update: dict) -> None:
    """tools 步：执行的 tool 与 evidence 增量 + 边际增益 4 字段。"""
    records = list(update.get("tool_call_history") or [])
    added = list(update.get("evidence") or [])
    print(f"[tools] 执行 {len(records)} 个调用，本轮新增 {len(added)} 条证据")
    for r in records:
        tool = r.get("tool")
        status = r.get("status")
        if status == "ok":
            added_refs = r.get("evidence_added") or []
            print(
                f"    seq={r.get('seq')} {tool} ok | 边际增益: "
                f"before_confidence={r.get('before_confidence')} "
                f"after_confidence={r.get('after_confidence')} "
                f"evidence_added={len(added_refs)} 条 "
                f"decision_changed={r.get('decision_changed')} "
                f"| latency_ms={r.get('latency_ms')}"
            )
            for ref in added_refs:
                print(f"        + {ref}")
        else:
            print(
                f"    seq={r.get('seq')} {tool} {status}: {r.get('error') or r.get('reason')}"
            )


def _print_reevaluate(update: dict, snapshot: list) -> list:
    """reevaluate 步：假设状态变化（对照上次快照打印 prior→posterior/status delta）。

    snapshot 为上一次看到的全量假设列表；返回本次的全量假设列表作为新快照。
    """
    hypos = list(update.get("hypotheses") or [])
    before = {h.id: (h.status.value, h.posterior) for h in snapshot}
    lines = []
    for h in hypos:
        prev = before.get(h.id)
        if prev is None:
            lines.append(f"    + {h.id}（新增, PENDING）: {h.statement}")
            continue
        st0, po0 = prev
        if st0 != h.status.value or po0 != h.posterior:
            post0 = "-" if po0 is None else f"{po0:.2f}"
            post1 = "-" if h.posterior is None else f"{h.posterior:.2f}"
            lines.append(
                f"    {h.id}: status {st0} -> {h.status.value} | "
                f"posterior {post0} -> {post1}"
            )
    if lines:
        print(f"[reevaluate] {len(lines)} 条假设状态变化:")
        print("\n".join(lines))
    else:
        print("[reevaluate] 假设状态无变化")
    queue = list(update.get("investigation_queue") or [])
    if queue:
        done = [q for q in queue if q.get("status") == "DONE"]
        print(f"    调查队列: {len(done)}/{len(queue)} DONE")
    return hypos


def _print_decide(update: dict) -> None:
    """decide 步：最终 ReviewDecision 摘要。"""
    decision = update.get("decision")
    if decision is None:
        print("[decide] 本步未产出 decision（update 键: " + str(sorted(update.keys())) + ")")
        return
    print(
        f"[decide] 最终裁决: decision={decision.decision.value} "
        f"risk_level={decision.risk_level.value} "
        f"risk_type={[t.value for t in decision.risk_type]} "
        f"decision_confidence={decision.decision_confidence} "
        f"overrides={decision.overrides} "
        f"evidence={len(decision.evidence)} 条 policy={decision.policy}"
    )


def _print_node(node_name: str, update: dict, snapshot: list) -> list:
    """按节点名分发打印，返回更新后的假设快照（仅 reevaluate/hypothesize 改动）。"""
    if node_name == "hypothesize":
        return _print_hypothesize(update)
    if node_name == "plan":
        _print_plan(update)
    elif node_name == "tools":
        _print_tools(update)
    elif node_name == "reevaluate":
        return _print_reevaluate(update, snapshot)
    elif node_name == "decide":
        _print_decide(update)
    else:
        print(f"[{node_name}] （未识别的节点更新键: {sorted(update.keys())}）")
    return snapshot


# 断言辅助（assert + 清晰消息；全部通过打印 "ALL CHECKS PASSED"）

def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _assert_all(final_state: dict, node_seq: list) -> None:
    """断言清单 1~8（失败即抛，消息带实际值）。"""
    decision = final_state.get("decision")
    _check(
        decision is not None,
        "断言1 失败: 终态 decision 为 None —— 图未产出最终裁决",
    )

    # 1) 三分类
    _check(
        decision.decision == Decision.HUMAN_REVIEW,
        f"断言1 失败: decision={decision.decision.value!r}，期望 {Decision.HUMAN_REVIEW.value!r}",
    )
    # 2) 风险等级
    _check(
        decision.risk_level == RiskLevel.HIGH,
        f"断言2 失败: risk_level={decision.risk_level.value!r}，期望 {RiskLevel.HIGH.value!r}",
    )
    # 3) 风险类型覆盖
    expected_rt = {RiskType.POTENTIAL_IP_RISK, RiskType.EVASION_PATTERN}
    got_rt = set(decision.risk_type)
    _check(
        expected_rt <= got_rt,
        "断言3 失败: risk_type={} 未覆盖期望 {}（实际缺失 {}）".format(
            [t.value for t in decision.risk_type],
            [t.value for t in expected_rt],
            [t.value for t in expected_rt - got_rt],
        ),
    )
    # 4) 置信度门槛
    _check(
        decision.decision_confidence >= 0.7,
        f"断言4 失败: decision_confidence={decision.decision_confidence} < 0.7",
    )
    # 5) evidence 覆盖 + IMAGE_SIMILARITY similarity≈0.91
    got_types = {e.type for e in decision.evidence}
    missing = _REQUIRED_EVIDENCE_TYPES - got_types
    _check(
        not missing,
        f"断言5 失败: evidence 类型 {sorted(got_types)} 缺少 {sorted(missing)}",
    )
    sims = [
        float(e.extra["similarity"])
        for e in decision.evidence
        if e.type == "IMAGE_SIMILARITY" and "similarity" in e.extra
    ]
    _check(
        any(abs(v - 0.91) < 1e-6 or v >= 0.9 for v in sims),
        f"断言5 失败: 无 IMAGE_SIMILARITY extra.similarity≈0.91 的证据（实际 {sims}）",
    )
    # 6) overlay 未改判
    _check(
        decision.overrides == [],
        f"断言6 失败: overrides={decision.overrides!r}，期望 []",
    )
    # 7) 预算计数（先打印 utilization，再断言）
    budget = final_state.get("budget")
    _check(budget is not None, "断言7 失败: 终态 budget 缺失")
    limits = budget.limits
    print("[assert-7] budget utilization vs limits:")
    print(
        f"    llm_calls={budget.llm_calls}/{limits.max_llm_calls} | "
        f"tool_calls={budget.tool_calls}/{limits.max_tool_calls} | "
        f"tokens={budget.tokens}/{limits.max_tokens} | "
        f"latency_ms={budget.latency_ms}/{limits.max_latency_ms}"
    )
    _check(
        budget.llm_calls == _EXPECTED_LLM_CALLS and budget.tool_calls == _EXPECTED_TOOL_CALLS,
        "断言7 失败: 预算计数不符 —— llm_calls={}（期望 {}）、tool_calls={}（期望 {}）。"
        "若实现偏差致计数不同（非本脚本问题），协调者请据此排查 scripted 分支/记账/路由。".format(
            budget.llm_calls, _EXPECTED_LLM_CALLS, budget.tool_calls, _EXPECTED_TOOL_CALLS
        ),
    )
    # 8) 终止性：hypothesize 恰 1 次；decide 恰 1 次且为最后一个节点；无未收敛假设
    n_hyp = node_seq.count("hypothesize")
    n_dec = node_seq.count("decide")
    _check(
        n_hyp == 1,
        f"断言8 失败: hypothesize 出现 {n_hyp} 次（期望恰 1 次）",
    )
    _check(
        n_dec == 1 and node_seq[-1] == "decide",
        f"断言8 失败: decide 出现 {n_dec} 次且须为最后节点（实际序列尾部: {node_seq[-4:]}）",
    )
    open_hyp = [
        h.id
        for h in (final_state.get("hypotheses") or [])
        if h.status in (HypothesisStatus.PENDING, HypothesisStatus.UNRESOLVED)
    ]
    _check(
        not open_hyp,
        f"断言8 失败: 终态存在未收敛假设 {open_hyp}（PENDING/UNRESOLVED 残留）",
    )


# 末尾摘要块


def _print_summary(final_state: dict) -> None:
    decision = final_state.get("decision")
    budget = final_state.get("budget")
    history = final_state.get("tool_call_history") or []
    print("\n===== 摘要 =====")
    print(f"decision:            {decision.decision.value}")
    print(f"risk_level:          {decision.risk_level.value}")
    print(f"risk_type:           {[t.value for t in decision.risk_type]}")
    print(f"decision_confidence: {decision.decision_confidence:.2f}")
    print(f"policy:              {decision.policy}")
    print(f"overrides:           {decision.overrides}")
    print("evidence:")
    for i, e in enumerate(decision.evidence, 1):
        print(f"  [{i}] type={e.type} weight={e.weight} ref_id={e.ref_id} extra={e.extra}")
        print(f"       value={e.value}")
    print("hypothesis_trace:")
    for h in decision.hypothesis_trace:
        prior = "-" if h.prior is None else f"{h.prior:.2f}"
        posterior = "-" if h.posterior is None else f"{h.posterior:.2f}"
        print(
            f"  {h.id} {h.statement} | prior={prior} -> posterior={posterior} "
            f"| {h.status.value}"
        )
    print("budget:")
    print(
        f"  llm_calls={budget.llm_calls} tool_calls={budget.tool_calls} "
        f"tokens={budget.tokens} latency_ms={budget.latency_ms}"
    )
    print(f"tool_call_history:   {len(history)} 条")


# 主流程


def _build_app():
    """构造编译图；graph.py 未落盘时给出明确报错。"""
    try:
        from pra.agent.graph import build_agent_graph  # 延迟 import：防未落盘/循环
    except Exception as exc:  # pragma: no cover - 仅 graph.py 未就绪时触发
        raise RuntimeError(
            "无法 import pra.agent.graph.build_agent_graph —— graph.py 尚未落盘或未实现 "
            f"（原错误: {exc!r}）。demo 依赖批 2 graph.py 装配，待协调者验收时执行。"
        ) from exc
    return build_agent_graph(checkpointer=make_memory_checkpointer())


async def main() -> None:
    """构造 case → astream 按序打印 → 终态断言 → 摘要块。"""
    case = build_demo_case()
    print("=" * 76)
    print(
        f"Demo 走查: case={case.case_id} | product={case.product.product_id} "
        f"| merchant={case.merchant_id} | event={case.event_type}"
    )
    print(
        f"  标题: {case.product.title} | 类目: {case.product.category} "
        f"| brand={case.product.brand} | version={case.product.version}"
    )
    print(
        f"  图片: {[i.url for i in case.product.images]} "
        f"| sku: {[s.sku_id for s in case.product.sku_list]}"
    )
    print(f"  机审信号: {[(s.name, s.result) for s in case.screening_signals]}")
    print("=" * 76)

    app = _build_app()
    config = {"configurable": {"thread_id": f"RUN_{case.case_id}"}}

    print("\n----- 节点走查（stream_mode='updates'）-----")
    node_seq: list = []
    hypo_snapshot: list = []
    async for chunk in app.astream(build_initial_state(case), config, stream_mode="updates"):
        for node_name, update in chunk.items():
            node_seq.append(node_name)
            hypo_snapshot = _print_node(node_name, update, hypo_snapshot)
    print(f"节点序列: {' -> '.join(node_seq)}")

    # checkpointer 场景：流结束后取终态。langgraph 1.2.11 的 StateSnapshot 是
    # NamedTuple（不可下标），失败后退化为 .values 属性读取。
    snapshot = await app.aget_state(config)
    try:
        final_state = snapshot["values"]  # type: ignore[index]
    except TypeError:
        final_state = snapshot.values  # type: ignore[attr-defined]

    print("\n----- 断言 -----")
    _assert_all(final_state, node_seq)
    print("ALL CHECKS PASSED")

    _print_summary(final_state)


if __name__ == "__main__":
    asyncio.run(main())
