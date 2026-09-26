"""llm_prompts 渲染层确定性测试（real LLM prompt 约束，全部纯字符串断言）。"""

from __future__ import annotations

from pra.agent.guardrails.schemas import HypothesizeOutput, ReevaluateOutput
from pra.agent.llm_prompts import (
    SYSTEM_PROMPTS,
    build_system_prompt,
    build_user_prompt,
)

# 简单 JSON Schema（模拟 OutputModel 的 model_json_schema() 形状）
_SCHEMA = {
    "title": "OutputProposal",
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["PASS", "REJECT", "HUMAN_REVIEW"],
        },
    },
    "required": ["decision"],
}

# 带既有假设清单的 hypothesize state
_HYP_WITH_EXISTING = {
    "case": {
        "case_id": "CASE_EC_0007",
        "merchant_id": "M_3307",
        "event_type": "UPDATE_PRICE",
        "product": {
            "product_id": "P_44702",
            "title": "云步百搭小白鞋",
            "description": "自主品牌小白鞋，第二版描述。",
            "category": "女鞋/运动鞋",
            "brand": "云步",
            "images": [{"url": "https://cdn/img1.jpg", "source": "主图"}],
        },
    },
    "hypotheses": [
        {
            "id": "H1",
            "statement": "商家存在系统性改标题重上架行为",
            "status": "UNRESOLVED",
            "prior": 0.7,
            "posterior": None,
            "evidence_for": [],
            "evidence_against": [],
        },
        {
            "id": "H2",
            "statement": "商家历史干净无违规记录",
            "status": "REFUTED",
            "prior": 0.2,
            "posterior": 0.05,
            "evidence_for": [],
            "evidence_against": ["MERCHANT_HISTORY removals=0"],
        },
        {
            "id": "H3",
            "statement": "运行中新发现的品牌字段核验维度",
            "status": "PENDING",
            "prior": 0.5,
            "posterior": None,
            "evidence_for": [],
            "evidence_against": [],
        },
    ],
}

_HYP_WITHOUT_EXISTING = {k: v for k, v in _HYP_WITH_EXISTING.items() if k != "hypotheses"}


def test_hypothesize_user_prompt_renders_existing_hypotheses():
    text = build_user_prompt(
        node="hypothesize", state=_HYP_WITH_EXISTING, json_schema=_SCHEMA
    )
    assert "既有假设清单" in text
    assert "H1" in text and "UNRESOLVED" in text
    assert "商家存在系统性改标题重上架行为" in text
    assert "H2" in text and "REFUTED" in text
    assert "商家历史干净无违规记录" in text
    assert "H3" in text and "PENDING" in text
    assert "运行中新发现的品牌字段核验维度" in text
    assert "输出格式要求" in text
    assert "__STATE__" not in text


def test_hypothesize_user_prompt_omits_section_when_no_existing():
    text = build_user_prompt(
        node="hypothesize", state=_HYP_WITHOUT_EXISTING, json_schema=_SCHEMA
    )
    assert "既有假设清单" not in text
    assert "云步百搭小白鞋" in text


def test_output_schema_contracts_unchanged():
    """输出 schema 契约：``hypotheses`` 必填 1..5；``new_hypotheses`` 可选无上限。"""
    hypo_schema = HypothesizeOutput.model_json_schema()
    assert hypo_schema["properties"]["hypotheses"]["minItems"] == 1
    assert hypo_schema["properties"]["hypotheses"]["maxItems"] == 5
    assert "investigation_queue" not in hypo_schema["properties"]
    reeval_schema = ReevaluateOutput.model_json_schema()
    assert "new_hypotheses" in reeval_schema["properties"]
    assert "queue_updates" not in reeval_schema["properties"]
    assert ReevaluateOutput.model_fields["new_hypotheses"].is_required() is False


def test_decide_prompt_budget_caps_two_dims_only():
    """预算上限文案只有 LLM / 工具两维。"""
    text = build_user_prompt(
        node="decide",
        state={
            "budget": {
                "llm_calls": 3,
                "tool_calls": 2,
                "tokens": 900,
                "limits": {
                    "max_llm_calls": 10,
                    "max_tool_calls": 15,
                    "max_tokens": 40000,
                    "max_latency_ms": 30000,
                },
            }
        },
        json_schema=_SCHEMA,
    )
    assert "LLM≤10 次" in text and "工具≤15 次" in text
    assert "token≤" not in text and "时长≤" not in text


def test_prompt_render_smoke_walkthrough():
    rich_state = {
        "case": {
            "case_id": "CASE_EC_0101",
            "merchant_id": "M_5512",
            "event_type": "NEW_LISTING",
            "product": {
                "product_id": "P_77310",
                "title": "1:1 复刻 潮流板鞋",
                "description": "经典复古板鞋版型",
                "category": "女鞋/运动鞋",
                "brand": None,
                "images": [{"url": "https://cdn/img1.jpg", "source": "主图"}],
            },
        },
        "hypotheses": [
            {
                "id": "H1",
                "statement": "商家存在系统性规避行为",
                "status": "SUPPORTED",
                "prior": 0.8,
                "posterior": 0.93,
                "evidence_for": ["MERCHANT_HISTORY removals=5"],
                "evidence_against": [],
            }
        ],
        "evidence": [
            {
                "type": "MERCHANT_HISTORY",
                "weight": 0.85,
                "source": "MerchantTool",
                "ref_id": "M_5512",
                "value": "5 removals / 3 title-relisting, credit=38",
                "extra": {"removals": 5, "title": 3},
            },
            {
                "type": "POLICY_REF",
                "weight": 0.9,
                "source": "PolicySearchTool",
                "ref_id": "POLICY_3.2",
                "value": "POLICY_3.2：外观高度模仿知名品牌",
                "extra": {},
            },
        ],
        "pending_tool_calls": [
            {"tool": "CaseSearchTool", "priority": 1, "reason": "补同类先例"}
        ],
        "required_measurement_coverage": ["- 必需测量覆盖：2/3"],
        "degraded": False,
        "failures": [],
        "budget": {
            "llm_calls": 3,
            "tool_calls": 2,
            "tokens": 900,
            "limits": {"max_llm_calls": 10, "max_tool_calls": 15},
        },
    }
    assert set(SYSTEM_PROMPTS) == {
        "hypothesize", "plan", "reevaluate", "decide", "single_call",
    }
    for node in SYSTEM_PROMPTS:
        sys_prompt = build_system_prompt(node)
        assert isinstance(sys_prompt, str) and len(sys_prompt) > 100
        user = build_user_prompt(node=node, state=rich_state, json_schema=_SCHEMA)
        assert isinstance(user, str) and user
        assert "输出格式要求" in user
        assert "__STATE__" not in user
