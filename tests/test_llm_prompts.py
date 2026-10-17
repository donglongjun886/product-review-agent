"""llm_prompts 渲染层确定性测试（append-only 架构：统一 system + 增量轮次消息）。"""

from __future__ import annotations

from pra.agent.guardrails.schemas import HypothesizeOutput, ReevaluateOutput
from pra.agent.llm_prompts import (
    AGENT_NODES,
    build_system_prompt,
    build_turn_message,
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
    ],
}

_HYP_WITHOUT_EXISTING = {k: v for k, v in _HYP_WITH_EXISTING.items() if k != "hypotheses"}

_RICH_STATE = {
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


def test_agent_system_prompt_byte_identical_across_nodes():
    """四个 agent 节点共用同一份 system（逐字节相等，跨节点前缀缓存的前提）；single_call 独立。"""
    prompts = [build_system_prompt(node) for node in AGENT_NODES]
    assert set(AGENT_NODES) == {"hypothesize", "plan", "reevaluate", "decide"}
    assert len(set(prompts)) == 1
    assert len(prompts[0]) > 50
    single = build_system_prompt("single_call")
    assert single != prompts[0]
    assert "不调用任何工具" in single


def test_turn_message_hypothesize_renders_case_and_existing_hypotheses():
    text = build_turn_message(
        node="hypothesize", state=_HYP_WITH_EXISTING, json_schema=_SCHEMA
    )
    assert "P_44702" in text
    assert "云步百搭小白鞋" in text
    assert "既有假设清单" in text
    assert "H1" in text and "UNRESOLVED" in text
    assert "H2" in text and "REFUTED" in text
    assert "输出格式要求" in text
    assert "__STATE__" not in text


def test_turn_message_hypothesize_omits_section_when_no_existing():
    text = build_turn_message(
        node="hypothesize", state=_HYP_WITHOUT_EXISTING, json_schema=_SCHEMA
    )
    assert "二、既有假设清单" not in text  # 小节标题不出现（指令正文提及该词属正常）
    assert "云步百搭小白鞋" in text


def test_turn_message_instruction_after_content_schema_at_tail():
    """轮次消息结构：增量内容在前，本轮角色指令在后，输出 Schema 要点与修正反馈压尾。"""
    text = build_turn_message(
        node="hypothesize",
        state=_HYP_WITH_EXISTING,
        json_schema=_SCHEMA,
        feedbacks=["JSON 校验失败：decision 字段缺失"],
    )
    case_pos = text.index("P_44702")
    instruction_pos = text.index("初始风险假设生成器")
    schema_pos = text.index("输出格式要求")
    feedback_pos = text.index("上一轮输出校验反馈")
    assert case_pos < instruction_pos < schema_pos < feedback_pos


def test_turn_message_reevaluate_renders_only_new_evidence():
    """reevaluate 增量渲染：只讲 llm_evidence_cursor 之后的证据；cursor=0 → 全量。"""
    state = {**_RICH_STATE, "llm_evidence_cursor": 1}
    text = build_turn_message(node="reevaluate", state=state, json_schema=_SCHEMA)
    assert "POLICY_3.2" in text
    assert "5 removals / 3 title-relisting" not in text  # cursor 之前的旧证据不重复讲
    full = build_turn_message(
        node="reevaluate",
        state={**_RICH_STATE, "llm_evidence_cursor": 0},
        json_schema=_SCHEMA,
    )
    assert "5 removals / 3 title-relisting" in full
    none_new = build_turn_message(
        node="reevaluate",
        state={**_RICH_STATE, "llm_evidence_cursor": 2},
        json_schema=_SCHEMA,
    )
    assert "POLICY_3.2" not in none_new
    assert "无新增证据" in none_new


def test_turn_message_plan_renders_dashboard_gap_and_catalog():
    """plan 增量内容：假设仪表盘快照 + 必需测量覆盖缺口 + 可用工具目录；证据不重复讲。"""
    args_schema = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "检索描述"}},
        "required": ["query"],
    }
    catalog = [
        {"name": "PolicySearchTool", "description": "政策条款检索", "args_schema": args_schema}
    ]
    text = build_turn_message(
        node="plan", state=_RICH_STATE, json_schema=_SCHEMA, tool_catalog=catalog
    )
    assert "H1" in text and "SUPPORTED" in text
    assert "本案必需测量覆盖" in text
    assert "PolicySearchTool" in text and "query" in text
    assert "5 removals / 3 title-relisting" not in text  # 证据已在历史里，不重复
    empty = build_turn_message(node="plan", state={}, json_schema=_SCHEMA)
    assert "conclude" in empty
    assert "无可用取证工具" in empty


def test_turn_message_decide_sections():
    """decide 增量内容：全量假设仪表盘 + 可引用政策/先例 + 覆盖清单 + 运行状态 + 预算。"""
    text = build_turn_message(node="decide", state=_RICH_STATE, json_schema=_SCHEMA)
    assert "H1" in text and "SUPPORTED" in text
    assert "POLICY_REF" in text
    assert "本案必需测量覆盖" in text
    assert "degraded" in text
    assert "900" in text
    assert "LLM≤10" in text and "工具≤15" in text


def test_decide_prompt_budget_caps_two_dims_only():
    """预算上限文案只有 LLM / 工具两维。"""
    text = build_turn_message(
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


def test_decide_prompt_renders_measurement_coverage():
    """decide 上下文带覆盖清单时渲染「本案必需测量覆盖」小节（与 Gate 同源）。"""
    text = build_turn_message(
        node="decide",
        state={
            "required_measurement_coverage": [
                "- 必需测量覆盖：3/3",
                "  - text_compliance：已测（阴性）",
            ],
        },
        json_schema=_SCHEMA,
    )
    assert "本案必需测量覆盖" in text
    assert "text_compliance：已测（阴性）" in text
    assert "以此为准" in text
    without = build_turn_message(node="decide", state={}, json_schema=_SCHEMA)
    assert "三、本案必需测量覆盖" not in without  # 小节标题不出现（指令正文提及该词属正常）


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


def test_turn_message_empty_state_all_agent_nodes_robust():
    """空 state / 缺键：四个 agent 节点都不崩、返回非空文本。"""
    for node in AGENT_NODES:
        text = build_turn_message(node=node, state={}, json_schema={})
        assert isinstance(text, str) and text
        assert "## 输出格式要求" in text
    text = build_turn_message(
        node="decide", state={"hypotheses": "not-a-list"}, json_schema={}
    )
    assert isinstance(text, str) and text


def test_single_call_keeps_one_shot_render():
    """single_call 臂不走对话历史：仍由 build_user_prompt 一次性渲染案件事实 + Schema。"""
    text = build_user_prompt(
        node="single_call", state=_HYP_WITHOUT_EXISTING, json_schema=_SCHEMA
    )
    assert "P_44702" in text
    assert "输出格式要求" in text
