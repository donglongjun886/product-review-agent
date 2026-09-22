"""hypothesize 节点 —— 初始风险假设生成（LLM 语义步）。

把输入 ``ProductReviewCase``（商品快照 + 商家 + 机审信号）转化为**初始待验证假设集**
（每条带 ``prior`` 先验 = 未经调查的怀疑度 0..1，不归一化、不要求和为 1），
交给 plan → tools → reevaluate 循环逐条验证。

- 本节点是图的**入口首节点**：无 degraded 短路；仅防御性检查 ``budget_exceeded``
  （checkpoint 续跑等异常态）→ 止损降级，不调 LLM。
- 假设内容与 prior 来自 LLM（``HypothesizeOutput`` 由 llm_shell 强校验：失败重试 1 次
  仍失败 → 返回降级结果并置 ``degraded=True``）。
- 其余为确定性 apply：``id`` 按序编号 H1..Hn、``status=PENDING``、``posterior=None``、
  ``evidence_for/against=[]``。LLM 的 ``rationale`` 在 AgentState 中无 channel，apply
  时直接丢弃。
- ``_state_payload`` 以结构化 dict 注入 state 子集（case 全量 + screening_signals，
  domain 对象经 ``model_dump(mode="json")`` 转 JSON 形状）—— 后端按 ``state=`` 接收，
  不再有 ``__STATE__`` 文本协议。
"""

from __future__ import annotations

from pra.agent.guardrails.budget import budget_exceeded, bump_llm_usage
from pra.agent.guardrails.errors import SEV_CRITICAL, STEP_HYPOTHESIZE, make_failure
from pra.agent.guardrails.llm_shell import LLMBackend, call_structured_llm
from pra.agent.guardrails.schemas import HypothesizeOutput
from pra.domain.models import Hypothesis, HypothesisStatus

__all__ = ["hypothesize_node"]

# LLM 步降级 failure 文案：取固定字面值、不拼 outcome.error（reason 稳定、可断言）。
_DEGRADE_REASON = "schema 校验重试仍失败"


def _state_payload(state: dict) -> dict:
    """LLM 入参 state 子集：case 全量 + screening_signals（domain 对象已
    ``model_dump(mode="json")``）。"""
    case = state["case"]
    return {
        "case": case.model_dump(mode="json"),
        "screening_signals": [
            s.model_dump(mode="json") for s in (case.screening_signals or [])
        ],
    }


def _apply_hypotheses(out: HypothesizeOutput) -> list[Hypothesis]:
    """LLM 提案 → Hypothesis 列表（确定性 apply）。

    按序编号 H1..Hn；``status=PENDING``、``posterior=None``、``evidence_for/against=[]``
    （重建而非引用 LLM 对象）；``evidence_hint`` / ``rationale`` 无对应字段，直接丢弃。
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


async def hypothesize_node(state: dict, config, *, llm: LLMBackend) -> dict:
    """hypothesize 图节点（入口首节点）：生成初始假设集。

    ``llm`` 由 ``build_agent_graph`` 装配期显式注入（本节点不持有/不查找任何默认后端）。

    返回 hypotheses / degraded / failures / budget（failures 只含本次新增）。
    """
    # 入口防御：预算已超限（checkpoint 续跑等异常态）→ 不调 LLM，降级转人工。
    if budget_exceeded(state["budget"]) is not None:
        return {
            "hypotheses": [],
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
        state=_state_payload(state),
        llm=llm,
    )
    # LLM 记账：按实际尝试次数 bump（成功 1 次 / 重试后成功 2 次 / 两次失败仍 2 次）
    budget = bump_llm_usage(
        state["budget"], llm_calls=outcome.attempts, tokens=outcome.tokens
    )
    if outcome.model is None:
        return {
            "hypotheses": [],
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
        "degraded": False,
        "budget": budget,
    }
