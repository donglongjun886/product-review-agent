"""确定性走查桩（pra/agent/scripted_llm.py）单测。

- 确定性：同 (node, state) → 输出字节一致（可重放）；
- 剧本：各 node 输出都能过对应 OutputModel 的 ``model_validate_json``（schema 强校验）；
- 未知 node → 抛 LLMBackendError（供降级路径测试）；
- 幂等：同证据集重复 reevaluate（应用一次后）不再产出重复更新。

state 一律在本文件直接构造为 JSON 可序列化 dict（结构化直传契约的形状），不经节点
payload 构造器 —— 节点产出 state 与桩分支的集成由图级用例覆盖。
"""

from __future__ import annotations

import pytest

from pra.agent.guardrails.llm_shell import LLMBackendError
from pra.agent.guardrails.schemas import (
    DecisionProposal,
    HypothesizeOutput,
    PlanOutput,
    ReevaluateOutput,
)
from pra.agent.scripted_llm import ScriptedLLMBackend
from pra.domain.models import HypothesisStatus
from helpers import ev, hp, make_case

_IMG_URL = "https://cdn.example.com/products/P_88231/img1.jpg"


def _sim_evidence() -> object:
    return ev("IMAGE_SIMILARITY", source="ImageAnalysisTool",
              value="similarity=0.91, match=某品牌经典鞋款", weight=0.91, ref_id=_IMG_URL)


def _pending_hypotheses() -> list:
    return [
        hp("H1", prior=0.5, statement="普通复古设计，非品牌款"),
        hp("H2", prior=0.4, statement="参考知名品牌经典复古跑鞋设计"),
        hp("H3", prior=0.2, statement="刻意规避品牌识别（品牌字段空缺）"),
        hp("H4", prior=0.15, statement="商家系统性类似上架行为"),
    ]


def _state(
    *,
    hypotheses: list | None = None,
    evidence: list | None = None,
    case=None,
) -> dict:
    """构造节点直传后端的那份 JSON 可序列化 state 子集。"""
    state: dict = {
        "hypotheses": [h.model_dump(mode="json") for h in (hypotheses or [])],
        "evidence": [e.model_dump(mode="json") for e in (evidence or [])],
    }
    if case is not None:
        state["case"] = case.model_dump(mode="json")
    return state


async def test_deterministic_output_bytes_equal():
    backend = ScriptedLLMBackend()
    state = _state(hypotheses=_pending_hypotheses(), evidence=[_sim_evidence()])
    # 同一 state 传两次（且不共享同一 dict 对象）→ 字节一致
    r1 = await backend.complete(
        node="reevaluate", state=dict(state), json_schema={}
    )
    r2 = await backend.complete(
        node="reevaluate", state=dict(state), json_schema={}
    )
    assert r1.content.encode("utf-8") == r2.content.encode("utf-8")
    assert r1.content == r2.content


async def test_hypothesize_script_passes_schema():
    backend = ScriptedLLMBackend()
    resp = await backend.complete(
        node="hypothesize", state=_state(case=make_case()), json_schema={}
    )
    out = HypothesizeOutput.model_validate_json(resp.content)
    assert len(out.hypotheses) == 4
    # prior 具体值属剧本种子，不逐值钉死（换种子数值不应误伤）；结构已由数量与 schema 校验锁定


async def test_plan_script_passes_schema_and_branches():
    """plan 剧本：无外观证据 → call_tools(ImageAnalysisTool)；五类证据齐 → conclude。"""
    backend = ScriptedLLMBackend()
    case = make_case()

    # 分支 1：无 IMAGE_SIMILARITY、有图 → 先做外观比对
    st1 = _state(case=case)
    out1 = PlanOutput.model_validate_json(
        (await backend.complete(node="plan", state=st1, json_schema={})).content
    )
    assert out1.next_action == "call_tools"
    assert [t.tool for t in out1.tools] == ["ImageAnalysisTool"]
    assert out1.tools[0].args["image_urls"] == [case.product.images[0].url]

    # 分支 4：五类关键证据齐 → conclude（tools 空）
    evs = [
        _sim_evidence(),
        ev("PRODUCT_FACT", source="ProductTool", value="brand=null", weight=0.6,
           ref_id="P_88231"),
        ev("MERCHANT_HISTORY", source="MerchantTool", value="5 removals", weight=0.85,
           ref_id="M_5512"),
        ev("CASE_PRECEDENT", source="CaseSearchTool", value="case_1001", weight=0.8,
           ref_id="case_1001"),
        ev("POLICY_REF", source="PolicySearchTool", value="POLICY_3.2", weight=0.9,
           ref_id="POLICY_3.2_v2_c1"),
    ]
    st2 = _state(case=case, evidence=evs)
    out2 = PlanOutput.model_validate_json(
        (await backend.complete(node="plan", state=st2, json_schema={})).content
    )
    assert out2.next_action == "conclude"
    assert out2.tools == []


async def test_reevaluate_script_passes_schema():
    backend = ScriptedLLMBackend()
    state = _state(
        hypotheses=_pending_hypotheses(),
        evidence=[_sim_evidence()],
    )
    state["pending_tool_calls"] = []
    resp = await backend.complete(node="reevaluate", state=state, json_schema={})
    out = ReevaluateOutput.model_validate_json(resp.content)
    by_id = {u.id: u for u in out.hypothesis_updates}
    assert set(by_id) == {"H1", "H2"}
    # 剧本策略语义：强相似下 H1 翻 REFUTED、H2 立 SUPPORTED，且 H2 posterior 取自
    # 证据相似度（round 两位）—— 是派生值而非独立种子
    assert by_id["H1"].status == "REFUTED"
    assert by_id["H2"].status == "SUPPORTED"
    assert by_id["H2"].posterior == round(_sim_evidence().weight, 2)
    assert out.evidence_sufficiency == "INSUFFICIENT"  # 缺 case_pre/policy


async def test_reevaluate_idempotent_after_apply():
    backend = ScriptedLLMBackend()
    hypotheses = _pending_hypotheses()
    state1 = _state(hypotheses=hypotheses, evidence=[_sim_evidence()])
    state1["pending_tool_calls"] = []
    first = ReevaluateOutput.model_validate_json(
        (await backend.complete(node="reevaluate", state=state1, json_schema={})).content
    )
    assert len(first.hypothesis_updates) == 2  # H1→REFUTED、H2→SUPPORTED

    # 模拟 apply 之后：各假设到达 first 输出的目标态（目标值从 first 派生，不钉剧本
    # 种子数值 —— 种子调整不破坏幂等验证）
    updates = {u.id: u for u in first.hypothesis_updates}
    applied = []
    for h in hypotheses:
        u = updates.get(h.id)
        if u is None:
            applied.append(h)
            continue
        applied.append(
            hp(h.id, prior=h.prior, posterior=u.posterior,
               status=HypothesisStatus(u.status), statement=h.statement,
               evidence_for=list(u.evidence_for),
               evidence_against=list(u.evidence_against))
        )
    state2 = _state(hypotheses=applied, evidence=[_sim_evidence()])
    state2["pending_tool_calls"] = []
    second = ReevaluateOutput.model_validate_json(
        (await backend.complete(node="reevaluate", state=state2, json_schema={})).content
    )
    assert second.hypothesis_updates == []  # 无重复更新（幂等）


async def test_decide_script_passes_schema():
    """decide 剧本：固定 HUMAN_REVIEW 提案，Schema 强校验。"""
    backend = ScriptedLLMBackend()
    state = _state(
        hypotheses=[hp("H2", prior=0.4, posterior=0.91,
                       status=HypothesisStatus.SUPPORTED)],
        evidence=[_sim_evidence()],
    )
    resp = await backend.complete(node="decide", state=state, json_schema={})
    out = DecisionProposal.model_validate_json(resp.content)
    assert out.decision == "HUMAN_REVIEW"
    assert out.risk_level == "HIGH"
    # confidence / risk_type / policy 的具体取值属剧本种子，不逐值钉死
    assert len(out.evidence_ids) == 1  # 引用真实证据


async def test_unknown_node_raises_llm_backend_error():
    """未知 node → 抛 LLMBackendError（供节点降级路径测试）。"""
    backend = ScriptedLLMBackend()
    with pytest.raises(LLMBackendError) as exc:
        await backend.complete(node="bogus-node", state={}, json_schema={})
    assert "unknown node" in str(exc.value)
    assert "bogus-node" in str(exc.value)


async def test_empty_state_fallback_hypothesize_plan_conclude():
    """空 state（空事实兜底）：hypothesize 仍产固定假设；plan 无图可查 → conclude。"""
    backend = ScriptedLLMBackend()
    hyp = HypothesizeOutput.model_validate_json(
        (await backend.complete(node="hypothesize", state={}, json_schema={})).content
    )
    assert len(hyp.hypotheses) >= 1  # 空事实兜底仍产出合法假设（schema 已保证非空）
    plan = PlanOutput.model_validate_json(
        (await backend.complete(node="plan", state={}, json_schema={})).content
    )
    assert plan.next_action == "conclude"
    assert plan.tools == []


async def test_default_backend_name_and_instance_shape():
    backend = ScriptedLLMBackend()
    assert backend.name == "scripted-walkthrough"
    a = await backend.complete(node="hypothesize", state={}, json_schema={})
    b = await backend.complete(node="hypothesize", state={}, json_schema={})
    assert a.content == b.content


async def test_feedback_is_ignored_by_fixed_script():
    """桩剧本固定：同一 state 带/不带 feedback → 输出一致（校验失败空转只由壳兜底）。"""
    backend = ScriptedLLMBackend()
    state = _state(case=make_case())
    plain = await backend.complete(node="plan", state=state, json_schema={})
    with_feedback = await backend.complete(
        node="plan", state=state, json_schema={}, feedback=["输出不满足 JSON Schema"]
    )
    assert plain.content == with_feedback.content
