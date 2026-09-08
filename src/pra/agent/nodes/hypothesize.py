"""hypothesize 节点 —— 初始风险假设生成（LLM 语义步；确定性护栏在 tools_node/guardrails）。

职责（docs/04-graph-design.md §2.2 /《01》§3.1；graph-mvp-contracts §3 G5）：
把输入 ``ProductReviewCase``（商品快照 + 商家 + 机审信号）转化为**初始待验证假设集**
（每条带 ``prior`` 先验 = 未经调查的怀疑度 0..1，不归一化、不要求总和为 1）与
**调查问题队列**，交给 plan → tools → reevaluate 循环逐条验证。

- 本节点是图的**入口首节点**：无 degraded 短路（前面没有 LLM 步可失败）；仅防御性
  检查 ``budget_exceeded``（checkpoint 续跑等异常态）→ 止损降级，不调 LLM。
- 假设的**内容与 prior** 来自 LLM（结构化 ``HypothesizeOutput``，由 llm_shell 强
  校验：失败重试 1 次仍失败 → 本节点返回降级结果并置 ``degraded=True``，§2.1）。
- 其余为节点内**确定性 apply**：``id`` 按序编号（H1..Hn）、``status=PENDING``、
  ``posterior=None``、``evidence_for/against=[]``、queue 元素补 ``status:"OPEN"``。
- ``rationale``（LLM 的一句话说明）在 AgentState 中**无 channel**，apply 时直接丢弃
  （如需审计，后续并入 tool_call_history 类审计通道 —— MVP 不落）。
- ``_build_messages`` MVP 只做"首条 user 消息以 ``__STATE__ {json}`` 注入 state 子集
  （case 全量 + screening_signals；domain 对象经 ``model_dump(mode="json")`` 转成
  JSON 形状；scripted_llm 桩与真实后端据此决策）"；真实 litellm 的完整 prompt
  （低风险假设要求、prior 语义等系统指令展开）在接入真实 LLM 时补充。

分工声明：本节点只做"生成 + 初始化"（LLM 语义步）；预算/降级判定、failures 审计、
确定性工具执行分别在 guardrails/{budget,errors} 与 tools_node —— 不在本节点重复实现。
"""

from __future__ import annotations

import json

from pra.agent.guardrails.budget import budget_exceeded, bump_llm_usage
from pra.agent.guardrails.errors import SEV_CRITICAL, STEP_HYPOTHESIZE, make_failure
from pra.agent.guardrails.llm_shell import call_structured_llm
from pra.agent.guardrails.schemas import HypothesizeOutput
from pra.domain.models import Hypothesis, HypothesisStatus

__all__ = ["hypothesize_node"]

# LLM 步降级 failure 文案（契约 §2.1 允许用默认文案；文件规格给定字面值，故不拼
# outcome.error —— 保持失败路径 reason 稳定、可断言）。
_DEGRADE_REASON = "schema 校验重试仍失败"

_SYSTEM_PROMPT = (
    "你是电商上架审核的「初始风险假设生成器」。你只负责根据案件事实生成待验证的"
    "风险假设与调查问题队列，绝不据此下最终结论（终判由收敛后的 decide 完成）。\n"
    "要求：\n"
    "1. 每条假设一句话可验证，聚焦可取证的风险维度（外观/品牌/商家行为/字段冲突等）；\n"
    "2. 必须包含至少 1 条低风险/正常假设（避免只报风险、预设违规）；\n"
    "3. prior = 未经调查的先验怀疑度（0..1），不要求归一化、不要求总和为 1；\n"
    "4. 调查问题 1~8 条，priority 1 最优先。\n"
    "只输出符合 HypothesizeOutput JSON Schema 的 JSON。"
)


def _build_messages(state: dict) -> list[dict]:
    """组装 LLM 消息：system=角色/约束；首条 user 以 "__STATE__ " 开头携带 state 子集。

    state 子集 = case 全量 + screening_signals（domain 对象已 ``model_dump(mode="json")``
    转成可 json 序列化形状，保证 ``json.dumps`` 直接可用）。MVP 简短注入即可，
    完整 prompt（含低风险假设/prior 语义展开）后续补。
    """
    case = state["case"]
    payload = {
        "case": case.model_dump(mode="json"),
        "screening_signals": [
            s.model_dump(mode="json") for s in (case.screening_signals or [])
        ],
    }
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "__STATE__ " + json.dumps(payload, ensure_ascii=False),
        },
    ]


def _apply_hypotheses(out: HypothesizeOutput) -> list[Hypothesis]:
    """LLM 提案 → Hypothesis 列表（确定性 apply）。

    按序编号 H1..Hn；``status=PENDING``、``posterior=None``、
    ``evidence_for/against=[]``（用重建而非引用 LLM 对象）。``evidence_hint`` 与
    ``rationale`` 不落 Hypothesis（无对应字段），直接丢弃。
    """
    return [
        Hypothesis(
            id=f"H{idx + 1}",
            statement=proposal.statement,
            prior=proposal.prior,
            posterior=None,
            status=HypothesisStatus.PENDING,
            evidence_for=[],
            evidence_against=[],
        )
        for idx, proposal in enumerate(out.hypotheses)
    ]


def _apply_queue(out: HypothesizeOutput) -> list[dict]:
    """LLM 调查问题 → ``AgentState.investigation_queue`` 元素 {q, priority, status}。"""
    return [
        {"q": item.q, "priority": item.priority, "status": "OPEN"}
        for item in out.investigation_queue
    ]


async def hypothesize_node(state: dict, config) -> dict:
    """hypothesize 图节点（入口首节点）：生成初始假设集 + 调查队列。

    返回 dict 只含 AgentState channel 键：hypotheses / investigation_queue /
    degraded / failures / budget（其余为覆盖写；failures 走 append reducer）。
    """
    # 入口防御：预算已超限（checkpoint 续跑等异常态）→ 不调 LLM，止损降级转人工。
    if budget_exceeded(state["budget"]) is not None:
        return {
            "hypotheses": [],
            "investigation_queue": [],
            "degraded": True,
            "failures": [
                make_failure(
                    step_type=STEP_HYPOTHESIZE,
                    severity=SEV_CRITICAL,
                    reason="预算已超限，hypothesize 无法生成假设（止损转人工）",
                )
            ],
            "budget": state["budget"],
        }

    outcome = await call_structured_llm(
        OutputModel=HypothesizeOutput,
        node="hypothesize",
        messages=_build_messages(state),
    )
    # LLM 记账：按实际尝试次数 bump（成功 1 次 / 重试后成功 2 次 / 两次失败仍 2 次）
    budget = bump_llm_usage(
        state["budget"], llm_calls=outcome.attempts, tokens=outcome.tokens
    )
    if outcome.model is None:
        return {
            "hypotheses": [],
            "investigation_queue": [],
            "degraded": True,
            "failures": [
                make_failure(
                    step_type=STEP_HYPOTHESIZE,
                    severity=SEV_CRITICAL,
                    reason=_DEGRADE_REASON,
                )
            ],
            "budget": budget,
        }
    return {
        "hypotheses": _apply_hypotheses(outcome.model),
        "investigation_queue": _apply_queue(outcome.model),
        "degraded": False,
        "budget": budget,
    }
