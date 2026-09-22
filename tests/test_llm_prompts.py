"""llm_prompts 渲染层确定性测试（real LLM prompt 约束守护，全部纯字符串断言）。

四节点 prompt 约束：

- hypothesize：禁重复提出既有假设（system 规则 + user 上下文渲染「既有假设清单」）；
  假设必须可取证、允许少提；
- plan：政策条款/先例一次性适用性判定（不重复安排同类检索）；证据已充分（高优先假设已
  全部结论化、无未解决高优先级疑点）应提前 conclude，低优先级/边际 UNRESOLVED 与只会
  重复采集的调用不阻止收尾；
- decide：证据链充分且无矛盾/缺口时本轮直接裁决，人工只留给真正的证据不足/矛盾/政策模糊/
  取证失败；
- reevaluate：外观/视觉类假设判 SUPPORTED 必须引用图像类证据（``IMAGE_SIMILARITY``），
  ``CASE_PRECEDENT`` / ``POLICY_REF`` 只能佐证；``new_hypotheses`` 禁重复、允许为空。

另含输出 schema 契约守护与四节点渲染走查。不联网 / 无 API key / 不调真实 LLM。
"""

from __future__ import annotations

from pra.agent.guardrails.schemas import HypothesizeOutput, ReevaluateOutput
from pra.agent.llm_prompts import (
    SYSTEM_PROMPTS,
    build_system_prompt,
    build_user_prompt,
)

# 简单 JSON Schema（模拟 OutputModel 的 model_json_schema() 形状；只断言分节与文本约束）
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

# 带既有假设清单的 hypothesize state：case 与节点 state 同构，
# 另带 hypotheses（UNRESOLVED / REFUTED / 已新增并存）验证去重清单渲染。
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
            "statement": "外观与经典小白鞋高度相似，存在视觉仿冒风险",
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
    # 行为级：每条既有假设的 id / status / statement 值都渲染进上下文（供 LLM 去重），
    # 不锁具体行格式与分节编号（措辞/排版调整不应误伤）。
    assert "既有假设清单" in text
    assert "H1" in text and "UNRESOLVED" in text
    assert "外观与经典小白鞋高度相似，存在视觉仿冒风险" in text
    assert "H2" in text and "REFUTED" in text
    assert "商家历史干净无违规记录" in text
    assert "H3" in text and "PENDING" in text
    assert "运行中新发现的品牌字段核验维度" in text
    assert "输出格式要求" in text  # Schema 要点分节恒在
    assert "__STATE__" not in text  # 不泄漏 __STATE__ 标记


def test_hypothesize_user_prompt_omits_section_when_no_existing():
    text = build_user_prompt(
        node="hypothesize", state=_HYP_WITHOUT_EXISTING, json_schema=_SCHEMA
    )
    assert "既有假设清单" not in text
    assert "云步百搭小白鞋" in text  # case 事实照常渲染


def test_output_schema_contracts_unchanged():
    """输出 schema 契约守护：``hypotheses`` 仍必填 1..5；``new_hypotheses`` 仍可选无上限；
    调查队列字段已删（零消费者）。"""
    hypo_schema = HypothesizeOutput.model_json_schema()
    assert hypo_schema["properties"]["hypotheses"]["minItems"] == 1
    assert hypo_schema["properties"]["hypotheses"]["maxItems"] == 5
    assert "investigation_queue" not in hypo_schema["properties"]
    reeval_schema = ReevaluateOutput.model_json_schema()
    assert "new_hypotheses" in reeval_schema["properties"]
    assert "queue_updates" not in reeval_schema["properties"]
    assert ReevaluateOutput.model_fields["new_hypotheses"].is_required() is False


def test_decide_prompt_budget_caps_two_dims_only():
    """预算上限文案只剩 LLM / 工具两维：即便 state 仍带旧上限键也不渲染 token/时长维度。"""
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


def test_four_node_prompt_render_smoke_walkthrough():
    # 一个尽量贴近真实节点 state 的富 state（decide 视角字段齐全）
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
                "statement": "外观与经典板鞋高度相似",
                "status": "SUPPORTED",
                "prior": 0.8,
                "posterior": 0.93,
                "evidence_for": ["IMAGE_SIMILARITY sim=0.93"],
                "evidence_against": [],
            }
        ],
        "evidence": [
            {
                "type": "IMAGE_SIMILARITY",
                "weight": 0.93,
                "source": "ImageAnalysisTool",
                "ref_id": "https://cdn/img1.jpg",
                "value": "sim=0.93, match=经典板鞋",
                "extra": {},
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
            {"tool": "ImageAnalysisTool", "priority": 1, "reason": "补视觉证据"}
        ],
        "required_measurement_coverage": ["- 必需测量覆盖：1/4"],
        "degraded": False,
        "failures": [],
        "budget": {
            "llm_calls": 3,
            "tool_calls": 2,
            "tokens": 900,
            "limits": {"max_llm_calls": 10, "max_tool_calls": 15},
        },
    }
    assert set(SYSTEM_PROMPTS) == {"hypothesize", "plan", "reevaluate", "decide"}
    for node in SYSTEM_PROMPTS:
        sys_prompt = build_system_prompt(node)
        assert isinstance(sys_prompt, str) and len(sys_prompt) > 100
        user = build_user_prompt(node=node, state=rich_state, json_schema=_SCHEMA)
        assert isinstance(user, str) and user
        assert "输出格式要求" in user  # Schema 要点分节恒在
        assert "__STATE__" not in user  # 真实模型上下文不带裸 __STATE__ 标记
