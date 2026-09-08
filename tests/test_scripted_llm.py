"""确定性走查桩（pra/agent/scripted_llm.py）单测。

- 确定性：同 (node, __STATE__) → 输出字节一致（可重放）；
- 走查剧本：各 node 输出都能过对应 OutputModel.model_validate_json（schema 强校验），
  关键字段与 §4.3 剧本一致；
- 未知 node → 抛 LLMBackendError（供降级路径测试）；
- 幂等：同证据集重复 reevaluate（应用一次后）不再产出重复更新。
"""

from __future__ import annotations

import json

import pytest

from pra.agent.guardrails.llm_shell import LLMBackendError
from pra.agent.guardrails.schemas import (
    DecisionProposal,
    HypothesizeOutput,
    PlanOutput,
    ReevaluateOutput,
)
from pra.agent.nodes.decide import _build_messages as decide_messages
from pra.agent.nodes.hypothesize import _build_messages as hypothesize_messages
from pra.agent.nodes.plan import _build_messages as plan_messages
from pra.agent.nodes.reevaluate import _build_messages as reevaluate_messages
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


def _queue(question: str = "外观是否与某知名品牌款高度相似？", status: str = "OPEN") -> list:
    return [{"q": question, "priority": 1, "status": status}]


async def test_deterministic_output_bytes_equal():
    """同 (node, __STATE__) → 字节一致（同输入恒同输出，eval 可重放）。"""
    backend = ScriptedLLMBackend()
    payload = {
        "hypotheses": [h.model_dump(mode="json") for h in _pending_hypotheses()],
        "evidence": [_sim_evidence().model_dump(mode="json")],
    }
    messages = [{"role": "user", "content": "__STATE__ " + json.dumps(payload, ensure_ascii=False)}]
    r1 = await backend.complete(node="reevaluate", messages=messages, json_schema={})
    r2 = await backend.complete(node="reevaluate", messages=messages, json_schema={})
    assert r1.content.encode("utf-8") == r2.content.encode("utf-8")
    assert r1.content == r2.content


async def test_hypothesize_script_passes_schema():
    """hypothesize 剧本：4 假设 + 2 队列，Schema 强校验通过。"""
    backend = ScriptedLLMBackend()
    resp = await backend.complete(node="hypothesize",
                                  messages=hypothesize_messages({"case": make_case()}),
                                  json_schema={})
    out = HypothesizeOutput.model_validate_json(resp.content)
    assert len(out.hypotheses) == 4
    assert len(out.investigation_queue) == 2
    assert [h.prior for h in out.hypotheses] == [0.5, 0.4, 0.2, 0.15]
    assert out.investigation_queue[0].priority == 1
    assert out.investigation_queue[1].priority == 2


async def test_plan_script_passes_schema_and_branches():
    """plan 剧本四分支：无外观证据 → call_tools(ImageAnalysisTool)；五类证据齐 → conclude。"""
    backend = ScriptedLLMBackend()
    case = make_case()

    # 分支 1：无 IMAGE_SIMILARITY、有图 → 先做外观比对
    st1 = {"hypotheses": [], "evidence": [], "case": case}
    out1 = PlanOutput.model_validate_json(
        (await backend.complete(node="plan", messages=plan_messages(st1), json_schema={})).content
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
    st2 = {"hypotheses": [], "evidence": evs, "case": case}
    out2 = PlanOutput.model_validate_json(
        (await backend.complete(node="plan", messages=plan_messages(st2), json_schema={})).content
    )
    assert out2.next_action == "conclude"
    assert out2.tools == []


async def test_reevaluate_script_passes_schema():
    """reevaluate 剧本按证据 flags 更新 H1(REFUTED)/H2(SUPPORTED)，Schema 强校验。"""
    backend = ScriptedLLMBackend()
    state = {
        "hypotheses": _pending_hypotheses(),
        "evidence": [_sim_evidence()],
        "investigation_queue": _queue(),
        "pending_tool_calls": [],
    }
    resp = await backend.complete(node="reevaluate", messages=reevaluate_messages(state),
                                  json_schema={})
    out = ReevaluateOutput.model_validate_json(resp.content)
    by_id = {u.id: u for u in out.hypothesis_updates}
    assert set(by_id) == {"H1", "H2"}
    assert by_id["H1"].status == "REFUTED" and by_id["H1"].posterior == 0.05
    assert by_id["H2"].status == "SUPPORTED" and by_id["H2"].posterior == 0.91
    # "外观"问题 + 强相似 → 队列项置 DONE
    assert [(u.q, u.status) for u in out.queue_updates] == [
        ("外观是否与某知名品牌款高度相似？", "DONE")
    ]
    assert out.evidence_sufficiency == "INSUFFICIENT"  # 缺 case_pre/policy


async def test_reevaluate_idempotent_after_apply():
    """幂等：同证据集下，应用一次脚本结果后再 reevaluate 不再产出重复更新。"""
    backend = ScriptedLLMBackend()
    state1 = {
        "hypotheses": _pending_hypotheses(),
        "evidence": [_sim_evidence()],
        "investigation_queue": _queue(),
        "pending_tool_calls": [],
    }
    first = ReevaluateOutput.model_validate_json(
        (await backend.complete(node="reevaluate", messages=reevaluate_messages(state1),
                                json_schema={})).content
    )
    assert len(first.hypothesis_updates) == 2  # H1→REFUTED、H2→SUPPORTED
    assert len(first.queue_updates) == 1  # "外观"问题 DONE

    # 模拟 apply 之后：H1/H2 已到目标态、队列已 DONE
    applied = [
        hp("H1", prior=0.5, posterior=0.05, status=HypothesisStatus.REFUTED,
           statement="普通复古设计，非品牌款"),
        hp("H2", prior=0.4, posterior=0.91, status=HypothesisStatus.SUPPORTED,
           statement="参考知名品牌经典复古跑鞋设计"),
        hp("H3", prior=0.2, statement="刻意规避品牌识别（品牌字段空缺）"),
        hp("H4", prior=0.15, statement="商家系统性类似上架行为"),
    ]
    state2 = {
        "hypotheses": applied,
        "evidence": [_sim_evidence()],
        "investigation_queue": _queue(status="DONE"),
        "pending_tool_calls": [],
    }
    second = ReevaluateOutput.model_validate_json(
        (await backend.complete(node="reevaluate", messages=reevaluate_messages(state2),
                                json_schema={})).content
    )
    assert second.hypothesis_updates == []  # 无重复更新（幂等）
    assert second.queue_updates == []  # 已 DONE 不再重复产出


async def test_decide_script_passes_schema():
    """decide 剧本：固定 HUMAN_REVIEW 提案，Schema 强校验（§4.3 原文）。"""
    backend = ScriptedLLMBackend()
    state = {
        "hypotheses": [hp("H2", prior=0.4, posterior=0.91,
                          status=HypothesisStatus.SUPPORTED)],
        "evidence": [_sim_evidence()],
        "degraded": False,
        "failures": [],
        "budget": None,
    }
    resp = await backend.complete(node="decide", messages=decide_messages(state), json_schema={})
    out = DecisionProposal.model_validate_json(resp.content)
    assert out.decision == "HUMAN_REVIEW"
    assert out.risk_level == "HIGH"
    assert [t.value for t in out.risk_type] == ["POTENTIAL_IP_RISK", "EVASION_PATTERN"]
    assert out.confidence == 0.91
    assert out.policy == ["POLICY_3.2"]
    assert len(out.evidence_ids) == 1  # 引用真实证据


async def test_unknown_node_raises_llm_backend_error():
    """未知 node → 抛 LLMBackendError（供节点降级路径测试）。"""
    backend = ScriptedLLMBackend()
    with pytest.raises(LLMBackendError) as exc:
        await backend.complete(node="bogus-node", messages=[], json_schema={})
    assert "unknown node" in str(exc.value)
    assert "bogus-node" in str(exc.value)


async def test_empty_state_fallback_hypothesize_plan_conclude():
    """缺 __STATE__（空事实兜底）：hypothesize 仍产固定假设；plan 无图可查 → conclude。"""
    backend = ScriptedLLMBackend()
    hyp = HypothesizeOutput.model_validate_json(
        (await backend.complete(node="hypothesize", messages=[], json_schema={})).content
    )
    assert len(hyp.hypotheses) == 4
    plan = PlanOutput.model_validate_json(
        (await backend.complete(node="plan", messages=[], json_schema={})).content
    )
    assert plan.next_action == "conclude"
    assert plan.tools == []


async def test_default_backend_name_and_instance_shape():
    """桩实例名与无状态性（多次 complete 不引入实例可变状态）。"""
    backend = ScriptedLLMBackend()
    assert backend.name == "scripted-walkthrough"
    a = await backend.complete(node="hypothesize", messages=[], json_schema={})
    b = await backend.complete(node="hypothesize", messages=[], json_schema={})
    assert a.content == b.content
