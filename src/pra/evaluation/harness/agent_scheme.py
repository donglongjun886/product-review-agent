"""System 3：完整调查 Agent（走真实图 + 固定 Eval World + 调用方注入的 LLM）。

``AgentScheme.run`` 每 case 独立 ``build_agent_graph`` + ``build_initial_state`` 后
``ainvoke``，终态 ``ReviewDecision`` 经 ``_transcribe`` 转成 ``EvalRecord``；不落 DB。
预算恒为**生产默认档**（LLM_CALLS=10 / TOOL_CALLS=15）—— 评测不覆盖 Guardrail 档位，
预算是否够用本身就是被测行为，超限由生产 Gate 收口。

工具世界固定为 ``make_eval_world_tools()``（与 ``eval_data/v2`` 同一份 InMemory 种子事实）；
``llm`` 由调用方必填注入并直接交给 ``build_agent_graph``，须实现
``pra.agent.guardrails.llm_shell.LLMBackend`` Protocol（``name`` 属性 +
``async complete(*, node, state, json_schema, feedback=None)``）。

**结论边界（报告必标注 ``EVAL_WORLD_LABEL``）**：真实 LLM 非确定、不可重放（同数据重跑
结果可不同）且需 API key；工具世界是种子而非生产 KB —— 种子里查不到的先例/规避史会低估
Agent 上限。
"""

from __future__ import annotations

from typing import Any
from uuid import NAMESPACE_URL, uuid5

from langgraph.graph.state import CompiledStateGraph

from pra.agent.graph import build_agent_graph
from pra.agent.guardrails.budget import (
    budget_exceeded,  # 预算超限维度判定（P2-16 记录侧复用，单一实现）
)
from pra.agent.state import build_initial_state
from pra.domain.models import ReviewDecision
from pra.evaluation.dataset.schema import EvalCase
from pra.evaluation.harness.base import EvalRecord, SchemeRunner
from pra.observability.tracing import (
    TraceContext,
    experiment_name,
    get_tracer,
    session_id,
)

__all__ = [
    "EVAL_CATEGORIES",
    "EVAL_IMAGE_MATCHES",
    "EVAL_MERCHANTS",
    "EVAL_POLICY_CLAUSES",
    "EVAL_PRECEDENTS",
    "EVAL_PRODUCTS",
    "EVAL_WORLD_LABEL",
    "AgentScheme",
    "make_eval_world_tools",
]

# 评测种子世界（与 eval_data/v2 同一份"事实知识"；默认演示种子同源扩展）
# 本世界知识只能经 Agent 的 5 个 InMemory 工具获得 —— 多信号/对抗类案需要调查才能发现。

EVAL_CATEGORIES: tuple[str, ...] = ("女鞋/运动鞋", "箱包/女包", "服装/卫衣")

# 图片外观分析种子（url → top_similar / logos / visual_risk）。
EVAL_IMAGE_MATCHES: dict[str, dict[str, Any]] = {
    # 强相似（>=0.85）：外观模仿确证
    "https://cdn.example.com/products/P_88231/img1.jpg": {
        "top_similar": [{"brand_ref": "某品牌经典鞋款", "similarity": 0.91}],
        "logos": [],
        "visual_risk": "外观与某品牌经典复古跑鞋高度相似",
    },
    "https://cdn.example.com/eval/asset-1001/img1.jpg": {
        "top_similar": [{"brand_ref": "某品牌经典鞋款", "similarity": 0.93}],
        "logos": [],
        "visual_risk": "外观与某品牌经典鞋款高度相似",
    },
    "https://cdn.example.com/eval/asset-1002/img1.jpg": {
        "top_similar": [{"brand_ref": "某品牌托特包", "similarity": 0.95}],
        "logos": [],
        "visual_risk": "外观与某品牌托特包高度相似",
    },
    "https://cdn.example.com/eval/asset-1003/img1.jpg": {
        "top_similar": [{"brand_ref": "某品牌卫衣", "similarity": 0.90}],
        "logos": [],
        "visual_risk": "图案/版型与某品牌卫衣高度相似",
    },
    # 弱相似（0.70~0.85 单信号）：需与商家/在库事实交叉才可判
    "https://cdn.example.com/eval/asset-1004/img1.jpg": {
        "top_similar": [{"brand_ref": "某品牌经典鞋款", "similarity": 0.72}],
        "logos": [],
        "visual_risk": "外观与某品牌经典鞋款存在一定相似",
    },
    "https://cdn.example.com/eval/asset-1005/img1.jpg": {
        "top_similar": [{"brand_ref": "某品牌托特包", "similarity": 0.73}],
        "logos": [],
        "visual_risk": "外观与某品牌托特包存在一定相似",
    },
    # Logo 检测命中（仅工具可得；与文本解耦）
    "https://cdn.example.com/eval/asset-1006/img1.jpg": {
        "top_similar": [],
        "logos": [{"brand": "GUCCI", "confidence": 0.90}],
        "visual_risk": "检测到疑似品牌 Logo",
    },
    # 干净图（无命中 → 无外观证据；"没查到"按低先验 UNRESOLVED 处理）
    "https://cdn.example.com/eval/asset-1007/img1.jpg": {},
    "https://cdn.example.com/eval/asset-1008/img1.jpg": {},
    "https://cdn.example.com/eval/asset-1009/img1.jpg": {},
    "https://cdn.example.com/eval/asset-1010/img1.jpg": {},
}

# 商家行为画像种子（removals/title_relisting_count >= 3 = 系统性信号）
EVAL_MERCHANTS: dict[str, dict[str, Any]] = {
    "M_5512": {  # 脏：5 removals / 3 改标题重上架（默认演示商家，同源）
        "merchant_id": "M_5512",
        "similar_product_count": 23,
        "removals": 5,
        "title_relisting_count": 3,
        "credit_score": 62,
    },
    "M_8801": {  # 脏：7 removals / 4 改标题
        "merchant_id": "M_8801",
        "similar_product_count": 31,
        "removals": 7,
        "title_relisting_count": 4,
        "credit_score": 38,
    },
    "M_3307": {  # 干净
        "merchant_id": "M_3307",
        "similar_product_count": 0,
        "removals": 0,
        "title_relisting_count": 0,
        "credit_score": 96,
    },
    "M_9904": {  # 干净
        "merchant_id": "M_9904",
        "similar_product_count": 0,
        "removals": 0,
        "title_relisting_count": 0,
        "credit_score": 92,
    },
    "M_6602": {  # 中性：1 removals（低于系统性阈值 3）
        "merchant_id": "M_6602",
        "similar_product_count": 1,
        "removals": 1,
        "title_relisting_count": 0,
        "credit_score": 85,
    },
}

# 商品在库事实种子（ProductTool 用）。``merchant_id`` 只服务 ``eval_dataset_gen.py`` 的
# 「商品 → 商家」配对，不属 ``ProductSnapshot``（工具模型读出时忽略该键）。
EVAL_PRODUCTS: dict[str, dict[str, Any]] = {
    "P_88231": {  # 默认演示商品（在库 brand=None；强相似图 0.91）
        "product_id": "P_88231",
        "merchant_id": "M_5512",
        "category": "女鞋/运动鞋",
        "brand": None,
        "version": 3,
        "status": "ON_SALE",
    },
    "P_77310": {
        "product_id": "P_77310",
        "merchant_id": "M_5512",
        "category": "女鞋/运动鞋",
        "brand": None,
        "version": 1,
        "status": "ON_SALE",
    },
    "P_55208": {  # 自有品牌（潮动）但外观 0.93 强相似 + 脏商家 —— 对抗案核心
        "product_id": "P_55208",
        "merchant_id": "M_5512",
        "category": "女鞋/运动鞋",
        "brand": "潮动",
        "version": 2,
        "status": "ON_SALE",
    },
    "P_31240": {
        "product_id": "P_31240",
        "merchant_id": "M_5512",
        "category": "箱包/女包",
        "brand": None,
        "version": 1,
        "status": "ON_SALE",
    },
    "P_66820": {
        "product_id": "P_66820",
        "merchant_id": "M_8801",
        "category": "箱包/女包",
        "brand": None,
        "version": 2,
        "status": "ON_SALE",
    },
    "P_66900": {
        "product_id": "P_66900",
        "merchant_id": "M_8801",
        "category": "箱包/女包",
        "brand": None,
        "version": 1,
        "status": "ON_SALE",
    },
    "P_90771": {
        "product_id": "P_90771",
        "merchant_id": "M_8801",
        "category": "服装/卫衣",
        "brand": None,
        "version": 1,
        "status": "ON_SALE",
    },
    "P_55190": {  # 干净自有品牌（云步），在库 brand 可查
        "product_id": "P_55190",
        "merchant_id": "M_3307",
        "category": "女鞋/运动鞋",
        "brand": "云步",
        "version": 2,
        "status": "ON_SALE",
    },
    "P_44702": {
        "product_id": "P_44702",
        "merchant_id": "M_3307",
        "category": "女鞋/运动鞋",
        "brand": "云步",
        "version": 1,
        "status": "ON_SALE",
    },
    "P_61040": {
        "product_id": "P_61040",
        "merchant_id": "M_3307",
        "category": "箱包/女包",
        "brand": "简行",
        "version": 1,
        "status": "ON_SALE",
    },
    "P_23411": {
        "product_id": "P_23411",
        "merchant_id": "M_9904",
        "category": "服装/卫衣",
        "brand": "山丘",
        "version": 1,
        "status": "ON_SALE",
    },
    "P_44120": {  # 中性商家（M_6602）的干净自有品牌商品
        "product_id": "P_44120",
        "merchant_id": "M_6602",
        "category": "服装/卫衣",
        "brand": "山野",
        "version": 1,
        "status": "ON_SALE",
    },
}

# 先例库种子（CaseSearchTool：category 精确 + risk_type 交叠 + retrieval_score 排序）
EVAL_PRECEDENTS: list[dict[str, Any]] = [
    {
        "case_id": "CASE_1832",
        "retrieval_score": 0.86,
        "decision": "REJECT",
        "risk_level": "HIGH",
        "risk_type": ["POTENTIAL_IP_RISK"],
        "summary": "无品牌标识 + 外观高度模仿知名品牌复古跑鞋 + 商家多次改标题重上架",
        "key_evidence": ["image_similarity>=0.85", "merchant_history>=5_removals"],
        "policy_refs": ["POLICY_3.2"],
        "category": "女鞋/运动鞋",
    },
    {
        "case_id": "CASE_0911",
        "retrieval_score": 0.31,
        "decision": "PASS",
        "risk_level": "NONE",
        "risk_type": [],
        "summary": "普通休闲鞋，无品牌标识且无相似外观",
        "key_evidence": [],
        "policy_refs": [],
        "category": "女鞋/运动鞋",
    },
    {
        "case_id": "CASE_2033",
        "retrieval_score": 0.88,
        "decision": "REJECT",
        "risk_level": "HIGH",
        "risk_type": ["POTENTIAL_IP_RISK"],
        "summary": "无品牌标识 + 外观高度模仿知名品牌手袋 + 商家多次改标题重上架",
        "key_evidence": ["image_similarity>=0.85", "merchant_history>=5_removals"],
        "policy_refs": ["POLICY_4.1"],
        "category": "箱包/女包",
    },
    {
        "case_id": "CASE_2120",
        "retrieval_score": 0.84,
        "decision": "REJECT",
        "risk_level": "HIGH",
        "risk_type": ["POTENTIAL_IP_RISK"],
        "summary": "卫衣图案/版型高度模仿知名潮牌 + 商家系统性重上架",
        "key_evidence": ["image_similarity>=0.85", "merchant_history>=5_removals"],
        "policy_refs": ["POLICY_5.2"],
        "category": "服装/卫衣",
    },
]

# 政策条款种子（PolicySearchTool：effective_only + category/risk_type 过滤）
EVAL_POLICY_CLAUSES: list[dict[str, Any]] = [
    {
        "policy_id": "POLICY_3.2",
        "version": 2,
        "clause_id": "POLICY_3.2_v2_c1",
        "title": "外观高度模仿知名品牌",
        "text": "商品外观高度模仿知名品牌设计且无品牌授权，判定为高风险，转人工审核处理",
        "category": "女鞋/运动鞋",
        "risk_type": ["POTENTIAL_IP_RISK"],
        "status": "EFFECTIVE",
        "effective_date": "2024-03-01",
    },
    {
        "policy_id": "POLICY_4.1",
        "version": 1,
        "clause_id": "POLICY_4.1_v1_c1",
        "title": "箱包外观高度模仿知名品牌",
        "text": "箱包类商品外观高度模仿知名品牌设计且无授权，判定为高风险，转人工审核处理",
        "category": "箱包/女包",
        "risk_type": ["POTENTIAL_IP_RISK"],
        "status": "EFFECTIVE",
        "effective_date": "2024-05-01",
    },
    {
        "policy_id": "POLICY_5.2",
        "version": 1,
        "clause_id": "POLICY_5.2_v1_c1",
        "title": "服装外观/图案模仿知名品牌",
        "text": "服装外观或图案高度模仿知名品牌且无授权，判定为高风险，转人工审核处理",
        "category": "服装/卫衣",
        "risk_type": ["POTENTIAL_IP_RISK"],
        "status": "EFFECTIVE",
        "effective_date": "2024-06-01",
    },
]

# 评测世界标识（报告"结论边界"标注用）
EVAL_WORLD_LABEL = "InMemory 种子世界 v1（含 P_88231/M_5512 演示种子扩展）"


def make_eval_world_tools():
    """构造 Eval World 的 5 个 InMemory 工具（Product / Image / Merchant / CaseSearch / PolicySearch）。

    比 ``pra.tools.build_tools`` 少一个 ``OCRTool``：评测集把 OCR 文本当基础输入
    （``input.product.images[].ocr_text``）直接消费。检索索引固定为本模块 ``EVAL_*`` 种子 ——
    与 ``eval_data/v2`` 同一份事实。

    :return: 5 个工具实例，顺序固定。
    """
    # 延迟 import：避免 evaluation 包导入期拉起全部工具子包（防环/省启动）
    from pra.tools.base import Tool
    from pra.tools.case_search.tool import CaseSearchTool, InMemoryCaseIndex
    from pra.tools.image_analysis.tool import (
        ImageAnalysisTool,
        MockImageAnalysisProvider,
    )
    from pra.tools.merchant.tool import InMemoryMerchantRepository, MerchantTool
    from pra.tools.policy_search.tool import InMemoryPolicyIndex, PolicySearchTool
    from pra.tools.product.tool import InMemoryProductRepository, ProductTool

    tools: list[Tool] = [
        ProductTool(repo=InMemoryProductRepository(EVAL_PRODUCTS)),
        ImageAnalysisTool(provider=MockImageAnalysisProvider(EVAL_IMAGE_MATCHES)),
        MerchantTool(repo=InMemoryMerchantRepository(EVAL_MERCHANTS)),
        CaseSearchTool(index=InMemoryCaseIndex(EVAL_PRECEDENTS)),
        PolicySearchTool(index=InMemoryPolicyIndex(EVAL_POLICY_CLAUSES)),
    ]
    return tools


# SchemeRunner：走真实图（build_agent_graph + 固定 Eval World + 注入的 LLM 后端）


def _root_trace_context(
    case: EvalCase, initial_state: dict, *, backend_name: str = "unknown"
) -> TraceContext:
    """按确定性规则包装评测路径的 root trace（每案一条）。

    ``trace_id = uuid5(NAMESPACE_URL, f"{experiment}:{eval_case_id}:agent:{backend_name}")``
    —— 同 experiment + 同案 + 同后端重跑落同一条 trace；后端名进 trace_id 使不同 LLM 后端
    各自成 trace，按 trace 汇总 token 时不互相混入。

    ``session_id`` 取 ``PRA_LANGFUSE_SESSION``（一次评测 run 一个值）；``version`` =
    experiment 名。只读：不写 state、不参与任何判定。

    :param case: 评测案（读 ``input.case_id`` / ``eval_case_id`` / ``scene``）。
    :param initial_state: 图初始状态（读 ``budget.limits`` 记入 metadata）。
    :param backend_name: LLM 后端名，进 trace_id 与 ``metadata.llm_backend``。
    """
    experiment = experiment_name()
    metadata: dict[str, Any] = {
        "case_id": case.input.case_id,
        "eval_case_id": case.eval_case_id,
        "scene": case.scene,
        "scheme": "agent",
        "experiment": experiment,
        "llm_backend": backend_name,
        "tool_world": "eval",
        "source": "evaluation",
    }
    limits = getattr(initial_state.get("budget"), "limits", None)
    if limits is not None and hasattr(limits, "model_dump"):
        metadata["budget_limits"] = limits.model_dump()
    tags = [
        "env:local",
        "scheme:agent",
        f"experiment:{experiment}",
        "source:evaluation",
        "tool_world:eval",
    ]
    return TraceContext(
        trace_id=uuid5(
            NAMESPACE_URL, f"{experiment}:{case.eval_case_id}:agent:{backend_name}"
        ).hex,
        name="review",
        session_id=session_id(),
        version=experiment,
        metadata=metadata,
        tags=tags,
        input={"case_id": case.input.case_id, "eval_case_id": case.eval_case_id},
    )


class AgentScheme(SchemeRunner):
    """System 3 —— 完整调查 Agent（真实图 + 固定 Eval World + 注入的 LLM 后端）。

    每 case 独立 build + compile 一个图（不落 checkpoint），每次 ``ainvoke`` 都由
    ``build_initial_state`` 起算 → 天然隔离；LLM 后端每案经 ``build_agent_graph(llm=...)``
    显式注入，不碰任何进程级全局、无需收尾复位。

    预算恒为**生产默认**（LLM_CALLS=10 / TOOL_CALLS=15）—— 评测不覆盖 Guardrail 档位：
    预算是否够用本身就是被测行为，超限 → HUMAN_REVIEW 由生产 Gate 收口。

    :param llm: LLM 后端（必填），交给 ``build_agent_graph``；须实现
        ``pra.agent.guardrails.llm_shell.LLMBackend`` Protocol。
    """

    name = "agent"

    def __init__(self, *, llm: object) -> None:
        self._llm: object = llm

    async def run(self, case: EvalCase) -> EvalRecord:
        tools = make_eval_world_tools()
        graph: CompiledStateGraph = build_agent_graph(
            tools=tools,
            llm=self._llm,
        )
        initial_state = build_initial_state(case.input)
        # Root trace：每案一条，trace_id 确定性 uuid5（含 LLM 后端名）；不 per-case flush，
        # 由评测入口整轮结束后 ``tracing.flush_tracer()`` 统一刷出。
        root_ctx = _root_trace_context(
            case, initial_state, backend_name=getattr(self._llm, "name", "unknown")
        )
        with get_tracer().trace_root(root_ctx) as root:
            # 不挂 checkpointer：图无恢复/续跑需求，终态由 ainvoke 直接返回。thread_id 仍传 ——
            # tools_node 从 ``configurable.thread_id`` 读 run_id 落审计。
            final_state = await graph.ainvoke(
                initial_state,
                {"configurable": {"thread_id": f"eval-agent-{case.eval_case_id}"}},
            )
            _decision = final_state.get("decision")
            if _decision is not None:
                root.update(
                    output={
                        "decision": _decision.decision.value,
                        "risk_level": _decision.risk_level.value,
                    }
                )

        decision: ReviewDecision | None = final_state.get("decision")
        if decision is None:
            raise RuntimeError(
                f"Agent 图执行完成但终态缺少 decision（eval_case_id={case.eval_case_id}）"
            )
        return self._transcribe(case, final_state, decision)

    # 终态 → EvalRecord 转录（只读 ReviewDecision / state 摘要）

    @staticmethod
    def _transcribe(case: EvalCase, final_state: dict, decision: ReviewDecision) -> EvalRecord:
        history = final_state.get("tool_call_history") or []
        ok_calls = [r for r in history if isinstance(r, dict) and r.get("status") == "ok"]
        tool_names = list(dict.fromkeys(str(r.get("tool")) for r in ok_calls))  # 去重保序

        evidence = [
            {
                "type": e.type,
                "source": e.source,
                "value": e.value,
                "weight": e.weight,
                "ref_id": e.ref_id,
                "extra": dict(e.extra or {}),
            }
            for e in decision.evidence
        ]
        budget = decision.budget_used
        trace = [
            {
                "id": h.id,
                "status": h.status.value,
                "prior": h.prior,
                "posterior": h.posterior,
                "statement": h.statement,
            }
            for h in decision.hypothesis_trace
        ]
        return EvalRecord(
            eval_case_id=case.eval_case_id,
            scheme="agent",
            decision=decision.decision.value,
            risk_level=decision.risk_level.value,
            risk_type=[t.value for t in decision.risk_type],
            decision_confidence=decision.decision_confidence,
            evidence=evidence,
            policy=list(decision.policy),
            tool_calls_actual=tool_names,
            cost={
                "llm_calls": budget.llm_calls,
                "tool_calls": budget.tool_calls,
                "tokens": budget.tokens,
            },
            detail={
                "overrides": list(decision.overrides),
                # P2-16：R3 命中时附"哪一维先撞限"（LLM_CALLS/TOOL_CALLS）供真实跑分
                # 归因 —— 直接复用 ``budget_exceeded``（同阈值同顺序的单一实现），
                # 不改 R3_BUDGET_EXHAUSTED 码字面/语义。
                "budget_hit_dim": budget_exceeded(budget),
                "hypothesis_trace": trace,
                "tool_history_count": len(history),
                # 边际增益审计字段透传（tools_node 的记录原样带出，指标层只读）：
                # 只保留统计需要的四项，避免把整段 args/result 复制进 record。
                "tool_history": [
                    {
                        "tool": h.get("tool"),
                        "status": h.get("status"),
                        "evidence_added": list(h.get("evidence_added") or []),
                        "decision_changed": bool(h.get("decision_changed")),
                    }
                    for h in history
                    if isinstance(h, dict)
                ],
            },
        )
