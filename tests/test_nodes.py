"""图节点（pra/agent/nodes/*）失败路径 / 短路 / apply 单测。

覆盖：plan/reevaluate/hypothesize 入口短路；hypothesize/reevaluate apply 的语义（按 id 命中、
``model_copy`` 不就地改、新假设续号）与 LLM 失败降级（degraded、critical
failure、attempts 记账）；decide 预算超限不调 LLM（R3 归因）、LLM 失败 → R5 落 overrides
且返回 degraded=False、degraded 透传短路，以及成功采纳 REJECT 提案；四节点的
``_state_payload`` 结构化 state 直传（键集合 + 全量 case）。

decide 采用「注入坏后端 + 最小确定性 state」，不 stub overlay。后端一律经节点关键字参数
``llm=`` 显式注入（无进程级全局）。
"""

from __future__ import annotations

from pra.agent.guardrails.errors import SEV_CRITICAL, STEP_DECIDE, STEP_HYPOTHESIZE
from pra.agent.nodes import decide as decide_mod
from pra.agent.nodes import hypothesize as hypothesize_mod
from pra.agent.nodes import plan as plan_mod
from pra.agent.nodes.decide import decide_node
from pra.agent.nodes.hypothesize import hypothesize_node
from pra.agent.nodes.plan import plan_node
from pra.agent.nodes import reevaluate as reevaluate_mod
from pra.agent.nodes.reevaluate import reevaluate_node
from pra.agent.guardrails.schemas import (
    HypothesizeOutput,
    HypothesisProposal,
    HypothesisUpdate,
    ReevaluateOutput,
)
from pra.domain.models import Budget, Decision, HypothesisStatus, RiskLevel, RiskType
from helpers import (
    AlwaysRaiseBackend,
    NodePayloadBackend,
    SequenceBackend,
    budget_exhausted_state,
    dc_anchor_state,
    decide_reject_json,
    hypothesize_json,
    hp,
    make_case,
    reevaluate_json,
)

_CONFIG: dict = {}


def _raise_backend() -> AlwaysRaiseBackend:
    """构造恒抛异常的后端替身并返回它，供断言调用次数。"""
    return AlwaysRaiseBackend()


# T1：四节点结构化 state 直传（无 __STATE__ 文本协议、无 case 重复裁剪）


def _payload_state() -> dict:
    """四节点共用的富 state（节点侧真实 domain 对象，非 __STATE__ JSON 形状）。"""
    return {
        "case": make_case(brand=None),
        "hypotheses": [hp("H1", prior=0.5)],
        "evidence": [],
        "pending_tool_calls": [{"tool": "ImageAnalysisTool", "priority": 1, "reason": "r"}],
        "measurement_capabilities": {"image_appearance": True},
        "budget": Budget(llm_calls=2, tool_calls=1, tokens=100),
        "degraded": False,
        "failures": [],
    }


def test_hypothesize_state_payload_is_case_plus_signals():
    payload = hypothesize_mod._state_payload(_payload_state())
    assert set(payload) == {"case", "screening_signals"}
    # case 为 JSON 形状的全量 dump（screening_signals 独立注入）
    assert payload["case"] == _payload_state()["case"].model_dump(mode="json")
    assert [s["name"] for s in payload["screening_signals"]] == ["KEYWORD", "LOGO_DETECT"]


def test_plan_state_payload_passes_full_case():
    """plan 不再把 case 裁成子集：payload 的原样传全量 dump（渲染层只读需要的字段）。"""
    state = _payload_state()
    payload = plan_mod._state_payload(state)
    assert set(payload) == {
        "hypotheses",
        "evidence",
        "case",
        "measurement_capabilities",
        "required_measurement_coverage",
    }
    assert payload["case"] == state["case"].model_dump(mode="json")
    # 裁剪函数已删（避免与渲染层 _case_lines/_product_lines 两处重复裁剪）
    assert not hasattr(plan_mod, "_case_subset")


def test_reevaluate_state_payload_has_no_queue_key():
    payload = reevaluate_mod._state_payload(_payload_state())
    assert set(payload) == {"hypotheses", "evidence", "pending_tool_calls"}
    assert set(payload["pending_tool_calls"][0]) == {"tool", "priority", "reason"}


def test_decide_state_payload_budget_summary_two_dims_only():
    payload = decide_mod._state_payload(_payload_state())
    assert set(payload) == {"hypotheses", "evidence", "degraded", "failures", "budget"}
    assert payload["budget"]["llm_calls"] == 2 and payload["budget"]["tokens"] == 100
    assert payload["budget"]["limits"] == {"max_llm_calls": 10, "max_tool_calls": 15}


# plan：入口短路


async def test_plan_short_circuit_when_degraded():
    """degraded=True → 不调 LLM，恰返回 ``{"pending_tool_calls": []}``。"""
    backend = _raise_backend()
    out = await plan_node({"degraded": True, "budget": Budget()}, _CONFIG, llm=backend)
    assert out == {"pending_tool_calls": []}
    assert backend.calls == 0


async def test_plan_short_circuit_when_budget_exceeded():
    """预算超限 → 不调 LLM，恰返回空 pending（不动 degraded）。"""
    backend = _raise_backend()
    out = await plan_node({"degraded": False, "budget": Budget(llm_calls=10)}, _CONFIG, llm=backend)
    assert out == {"pending_tool_calls": []}
    assert backend.calls == 0


# hypothesize：入口 + 成功 apply + LLM 失败降级


def _hyp_state() -> dict:
    return {"case": make_case(), "budget": Budget()}


async def test_hypothesize_budget_guard_stops_without_llm():
    backend = _raise_backend()
    state = {"case": make_case(), "budget": Budget(llm_calls=10)}
    out = await hypothesize_node(state, _CONFIG, llm=backend)
    assert backend.calls == 0
    assert out["degraded"] is True
    assert out["hypotheses"] == []
    assert out["budget"] is state["budget"]  # 未记账（不调 LLM）
    failure = out["failures"][0]
    assert failure["step_type"] == STEP_HYPOTHESIZE
    assert failure["severity"] == SEV_CRITICAL
    assert failure["reason"]  # 失败携带可归因原因（预算语义由"未调 LLM + 未记账"承载）


async def test_hypothesize_success_apply():
    """成功路径：H1..Hn 按序编号、PENDING、posterior=None。"""
    backend = NodePayloadBackend(payloads={"hypothesize": hypothesize_json()})
    out = await hypothesize_node(_hyp_state(), _CONFIG, llm=backend)
    hypos = out["hypotheses"]
    assert backend.calls == ["hypothesize"]
    assert [h.id for h in hypos] == ["H1", "H2"]
    # 陈述与先验来自后端载荷（LLM 提供的 prior 被原样采纳，而非节点自造缺省值）
    payload = HypothesizeOutput.model_validate_json(hypothesize_json())
    assert [(h.statement, h.prior) for h in hypos] == [
        (p.statement, p.prior) for p in payload.hypotheses
    ]
    assert all(h.status == HypothesisStatus.PENDING for h in hypos)
    assert all(h.posterior is None for h in hypos)
    assert all(h.evidence_for == [] and h.evidence_against == [] for h in hypos)
    assert out["degraded"] is False
    assert out["budget"].llm_calls == 1  # attempts=1 记账
    assert out["budget"].tokens == 5


async def test_hypothesize_llm_failure_degrades():
    backend = SequenceBackend(contents=["not json", "still not json"], tokens=0)
    out = await hypothesize_node(_hyp_state(), _CONFIG, llm=backend)
    assert out["degraded"] is True
    assert out["hypotheses"] == []
    assert len(backend.calls) == 2
    failure = out["failures"][0]
    assert failure["step_type"] == STEP_HYPOTHESIZE
    assert failure["severity"] == SEV_CRITICAL
    assert "校验" in failure["reason"]  # 语义：schema 校验重试后仍失败（不锁措辞）
    assert out["budget"].llm_calls == 2  # 两次尝试都记账


# reevaluate：入口短路 + apply


async def test_reevaluate_short_circuit_returns_empty():
    backend = _raise_backend()
    assert await reevaluate_node({"degraded": True, "budget": Budget()}, _CONFIG, llm=backend) == {}
    assert await reevaluate_node(
        {"degraded": False, "budget": Budget(llm_calls=10)}, _CONFIG, llm=backend
    ) == {}
    assert backend.calls == 0


async def test_reevaluate_apply_by_id_model_copy_and_new_hypotheses():
    """``_apply``：id 命中走 model_copy（原对象不变、未命中 id 跳过）；新假设从最大 H 号续号。"""
    original = [
        hp("H1", prior=0.5, statement="刻意规避品牌识别"),
        hp("H2", prior=0.3, statement="普通设计"),
    ]
    state = {"hypotheses": original}
    out = ReevaluateOutput(
        hypothesis_updates=[
            HypothesisUpdate(id="H1", posterior=0.9, status="SUPPORTED",
                             evidence_for=["IMAGE_SIMILARITY similarity=0.91"]),
            HypothesisUpdate(id="H99", posterior=0.5, status="REFUTED"),  # 失效 id → 跳过
        ],
        new_hypotheses=[HypothesisProposal(statement="商家系统性上架", prior=0.2)],
    )
    result = reevaluate_mod._apply(state, out)
    updated_h1 = result["hypotheses"][0]
    assert updated_h1.posterior == 0.9 and updated_h1.status == HypothesisStatus.SUPPORTED
    assert updated_h1.evidence_for == ["IMAGE_SIMILARITY similarity=0.91"]
    assert updated_h1 is not original[0]  # model_copy，非原对象
    assert original[0].posterior is None and original[0].status == HypothesisStatus.PENDING
    assert result["hypotheses"][1] is original[1]  # 未更新的保持原引用
    assert [h.id for h in result["hypotheses"]] == ["H1", "H2", "H3"]
    new_h3 = result["hypotheses"][2]
    assert new_h3.statement == "商家系统性上架"
    assert new_h3.status == HypothesisStatus.PENDING and new_h3.posterior is None


async def test_reevaluate_apply_new_hypothesis_from_empty():
    """无现用假设时 new_hypotheses 从 H1 续号。"""
    out = ReevaluateOutput(
        hypothesis_updates=[],
        new_hypotheses=[HypothesisProposal(statement="唯一假设", prior=0.4)],
    )
    result = reevaluate_mod._apply({"hypotheses": []}, out)
    assert [h.id for h in result["hypotheses"]] == ["H1"]


async def test_reevaluate_node_applies_output_and_bumps_budget():
    """节点成功路径：LLM 输出经 apply → hypotheses 全集、llm_calls=1。"""
    state = {
        "hypotheses": [
            hp("H1", prior=0.5, statement="刻意规避品牌识别"),
            hp("H2", prior=0.3, statement="普通设计"),
        ],
        "evidence": [],
        "pending_tool_calls": [],
        "budget": Budget(),
        "degraded": False,
    }
    backend = NodePayloadBackend(payloads={"reevaluate": reevaluate_json()})
    out = await reevaluate_node(state, _CONFIG, llm=backend)
    assert backend.calls == ["reevaluate"]
    hypos = out["hypotheses"]
    assert [h.id for h in hypos] == ["H1", "H2", "H3"]
    # 更新值来自后端载荷（posterior/prior 原样采纳），不钉桩种子具体数值
    payload = ReevaluateOutput.model_validate_json(reevaluate_json())
    assert hypos[0].posterior == payload.hypothesis_updates[0].posterior
    assert hypos[0].status == HypothesisStatus.SUPPORTED
    assert hypos[2].status == HypothesisStatus.PENDING
    assert hypos[2].prior == payload.new_hypotheses[0].prior
    assert out["degraded"] is False
    assert out["budget"].llm_calls == 1


async def test_reevaluate_llm_failure_keeps_hypotheses_and_degrades():
    backend = SequenceBackend(contents=["not json", "not json"], tokens=0)
    state = {
        "hypotheses": [hp("H1", prior=0.5, statement="s")],
        "evidence": [],
        "pending_tool_calls": [],
        "budget": Budget(),
        "degraded": False,
    }
    out = await reevaluate_node(state, _CONFIG, llm=backend)
    assert out["degraded"] is True
    assert "hypotheses" not in out  # 不动推理字段
    assert state["hypotheses"][0].status == HypothesisStatus.PENDING  # 原对象未变
    assert out["failures"][0]["step_type"] == "REEVALUATE"
    assert out["failures"][0]["severity"] == SEV_CRITICAL
    assert out["budget"].llm_calls == 2


# decide：降级态 + 成功采纳


async def test_decide_budget_exceeded_does_not_call_llm():
    """预算超限 → 不调 LLM，HUMAN_REVIEW + R3_BUDGET_EXHAUSTED，budget 未记账。"""
    backend = _raise_backend()
    st = budget_exhausted_state()
    out = await decide_node(st, _CONFIG, llm=backend)
    assert backend.calls == 0
    assert out["decision"].decision == Decision.HUMAN_REVIEW
    assert out["decision"].overrides == ["R3_BUDGET_EXHAUSTED"]
    assert out["degraded"] is False
    assert out["budget"].llm_calls == 10  # 未 bump（没调 LLM）
    assert out["failures"] == []


async def test_decide_degraded_short_circuit_no_llm():
    backend = _raise_backend()
    st = dc_anchor_state()
    st["degraded"] = True
    out = await decide_node(st, _CONFIG, llm=backend)
    assert backend.calls == 0
    assert out["decision"].overrides == ["R5_DEGRADED_OR_FAILED_STEP"]
    assert out["decision"].decision == Decision.HUMAN_REVIEW
    assert out["degraded"] is False  # 节点恒返回 False（降级被消费进 decision）
    assert out["budget"].llm_calls == 0
    assert out["failures"] == []


async def test_decide_llm_failure_r5_override():
    """decide 自身 LLM 失败 → 注入 degraded + critical failure 到 overlay state，R5 落
    overrides；节点仍返回 degraded=False，两次尝试都记账。"""
    backend = _raise_backend()
    st = dc_anchor_state()
    st["degraded"] = False
    out = await decide_node(st, _CONFIG, llm=backend)
    assert backend.calls == 2  # 两次尝试都失败
    assert out["decision"].decision == Decision.HUMAN_REVIEW
    assert out["decision"].overrides == ["R5_DEGRADED_OR_FAILED_STEP"]
    assert out["decision"].risk_level == RiskLevel.HIGH  # 派生自 SUPPORTED 后验
    assert out["degraded"] is False
    assert out["budget"].llm_calls == 2  # attempts=2 记账
    failure = out["failures"][0]
    assert failure["step_type"] == STEP_DECIDE
    assert failure["severity"] == SEV_CRITICAL
    assert "校验" in failure["reason"]  # 语义：schema 校验重试后仍失败（不锁措辞）


async def test_decide_success_path_adopts_reject():
    """成功路径：REJECT 提案过 Gate → 采纳（overrides=[]、dc=0.95）、failures=[]。"""
    backend = NodePayloadBackend(payloads={"decide": decide_reject_json()})
    st = dc_anchor_state()
    st["degraded"] = False
    out = await decide_node(st, _CONFIG, llm=backend)
    assert backend.calls == ["decide"]
    decision = out["decision"]
    assert decision.decision == Decision.REJECT
    assert decision.overrides == []
    assert decision.decision_confidence == 0.95
    assert decision.risk_level == RiskLevel.HIGH
    assert decision.risk_type == [RiskType.POTENTIAL_IP_RISK]
    assert decision.policy == ["POLICY_3.2"]
    assert out["degraded"] is False
    assert out["budget"].llm_calls == 1
    assert out["failures"] == []
