"""llm_prompts 渲染层确定性测试（Phase 3 real LLM prompt 迭代的约束守护）。

覆盖 Phase 3 v1 real 跑分诊断出的三条新 prompt 约束（全部为纯字符串断言，
不联网 / 无 API key / 不调真实 LLM）：
- hypothesize：禁重复提出既有假设（system 规则 + user 上下文渲染「既有假设清单」）
  与"假设必须可取证、允许少提"；
- plan：政策条款/先例一次性适用性判定（不重复安排同类检索）；证据已充分（高优先假设
  已全部结论化、无未解决高优先级疑点）应提前 conclude，UNRESOLVED 低优先级/边际假设
  与只会重复采集的工具调用不阻止收尾；
- decide：证据链充分且无矛盾/缺口时应本轮直接裁决（不把可裁决案件推给 HUMAN_REVIEW），
  人工只留给真正的证据不足/矛盾/政策模糊/取证失败场景；
- reevaluate：外观/视觉类假设判 SUPPORTED 必须引用图像类证据（IMAGE_SIMILARITY），
  CASE_PRECEDENT / POLICY_REF 只能佐证；new_hypotheses 禁重复、允许为空。

另含：四节点 user/system 渲染走查冒烟 + 输出 schema 契约不变守护
（HypothesizeOutput.hypotheses 仍 1..5 必填、ReevaluateOutput.new_hypotheses 仍可选）。

语法/import 约定：顶部 ``from __future__ import annotations``；import 一律 pra.*。
"""

from __future__ import annotations

from pra.agent.guardrails.schemas import HypothesizeOutput, ReevaluateOutput
from pra.agent.llm_prompts import (
    SYSTEM_PROMPTS,
    build_system_prompt,
    build_user_prompt,
)

# 简单 JSON Schema（模拟节点 OutputModel 的 model_json_schema() 形状；只断言分节与
# 文本约束，不依赖具体模型字段）。
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

# 带既有假设清单的 hypothesize state：case + screening_signals 与节点 __STATE__ 同构，
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
    "screening_signals": [{"name": "KEYWORD", "result": "HIT", "score": 0.6}],
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


def test_hypothesize_system_no_duplicate_and_provable_rules():
    """hypothesize system：含禁重复既有假设 / 假设可取证且允许少提的显式规则。"""
    prompt = build_system_prompt("hypothesize")
    assert "禁止重复提出假设" in prompt
    assert "既有假设清单" in prompt  # 规则指向 user 上下文里渲染的去重清单
    assert "语义重复即重复" in prompt
    assert "只提出清单之外的" in prompt and "新风险维度" in prompt
    assert "能被后续调查计划的取证工具检验" in prompt  # 假设必须可取证（勿脑补）
    assert "允许**少提**" in prompt  # 没有新维度时宁精勿凑
    # 既有语义关键词不回退（test_litellm_backend 同守护）
    assert "低风险" in prompt and "先验" in prompt


def test_hypothesize_user_prompt_renders_existing_hypotheses():
    """hypothesize user：state 带 hypotheses → 渲染「四、既有假设清单」精简行。"""
    text = build_user_prompt(
        node="hypothesize", state=_HYP_WITH_EXISTING, json_schema=_SCHEMA
    )
    assert "## 四、既有假设清单" in text
    assert "去重参考" in text
    assert "- H1 | status=UNRESOLVED：外观与经典小白鞋高度相似，存在视觉仿冒风险" in text
    assert "- H2 | status=REFUTED：商家历史干净无违规记录" in text
    assert "- H3 | status=PENDING：运行中新发现的品牌字段核验维度" in text
    assert "## 三、机审信号" in text  # 原分节顺序不回退
    assert "## 输出格式要求" in text
    assert "__STATE__" not in text  # 不泄漏 __STATE__ 标记


def test_hypothesize_user_prompt_omits_section_when_no_existing():
    """hypothesize user：state 无 hypotheses（首轮初始生成）→ 不渲染去重清单分节。"""
    text = build_user_prompt(
        node="hypothesize", state=_HYP_WITHOUT_EXISTING, json_schema=_SCHEMA
    )
    assert "既有假设清单" not in text
    assert "## 三、机审信号" in text and "KEYWORD" in text


def test_plan_system_policy_applicability_one_shot():
    """plan system：政策/先例一次性适用性判定 —— 已引用即不再重复安排同类检索。"""
    prompt = build_system_prompt("plan")
    assert "一次性适用性判定" in prompt
    assert "适用性已判定" in prompt
    assert "不要仅为复核同一条款/先例是否适用而重复安排同类检索" in prompt
    # 原有"只计划能带来新证据的调用 / conclude"表述不回退
    assert "新证据" in prompt and "conclude" in prompt and "next_action" in prompt


def test_plan_system_early_conclude_when_evidence_sufficient():
    """plan system：证据已充分（高优先假设已全部结论化、无未解决高优先级疑点）应
    立即提前 conclude；UNRESOLVED 的低优先级/边际假设与只会重复采集的工具不阻止收尾。"""
    prompt = build_system_prompt("plan")
    # 判定基准：高优先假设全部得出基于证据的结论 + 无未解决高优先级疑点 = 证据已充分
    assert "证据已充分 → 提前收尾（conclude）" in prompt
    assert "高优先（high prior）" in prompt and "得出基于证据的结论" in prompt
    assert "没有未解决的高优先级疑点" in prompt
    # 低优先级/边际 UNRESOLVED、或只会重复采集已满足维度/复核已引用条款的工具不阻止
    assert "低优先级/边际假设" in prompt
    assert "不构成阻止 conclude 的理由" in prompt
    # 反对为「看起来有进展」安排多余取证：立即 conclude（tools 空数组）并提前结束循环
    assert "应**立即**输出" in prompt and 'next_action="conclude"' in prompt
    assert "不要为了「看起来有进展」而安排多余取证烧预算" in prompt
    assert "**提前结束**调查循环" in prompt


def test_reevaluate_system_visual_evidence_gate():
    """reevaluate system：外观/视觉类 SUPPORTED 必须引用图像类证据（禁止脑补）。"""
    prompt = build_system_prompt("reevaluate")
    assert "外观/视觉类假设的证据门槛" in prompt
    assert "图像类证据" in prompt and "IMAGE_SIMILARITY" in prompt
    assert "CASE_PRECEDENT / POLICY_REF 只能作佐证" in prompt
    assert "不能单独支撑外观类 SUPPORTED" in prompt
    assert "无视觉证据时该类假设判 UNRESOLVED" in prompt
    assert "仅凭标题文字或先例脑补外观相似结论" in prompt
    assert "禁止仅凭标题" in prompt


def test_reevaluate_system_new_hypotheses_no_duplicate_and_policy_one_shot():
    """reevaluate system：new_hypotheses 禁重复/允许为空 + 政策先例引用一次判定。"""
    prompt = build_system_prompt("reevaluate")
    assert "new_hypotheses 禁重复、允许为空" in prompt
    assert "new_hypotheses **允许为空**" in prompt
    assert "与假设仪表盘既有假设同维度或同表述" in prompt
    assert "政策/先例引用一次判定" in prompt
    assert "适用性已判定" in prompt
    assert "不能替代对应维度的真实取证" in prompt


def test_decide_system_rule_when_evidence_sufficient():
    """decide system：证据链充分且无矛盾/缺口时应本轮直接裁决（不推 HUMAN_REVIEW）；
    人工只留给真正的证据不足/矛盾/政策模糊/取证失败；禁止凭标题/先例脑补事实。"""
    prompt = build_system_prompt("decide")
    assert "证据充分即裁决（硬性）" in prompt
    assert "PASS 侧无证据缺口" in prompt
    assert "REJECT 侧有上下文证据中**真实出现**的政策条款" in prompt
    assert "不要把本可裁决的案件推给 HUMAN_REVIEW" in prompt
    assert "HUMAN_REVIEW 仍只留给真正的证据不足" in prompt
    assert "禁止凭标题、类目或先例脑补上下文没有的事实" in prompt


def test_output_schema_contracts_unchanged():
    """输出 schema 契约不变守护：hypotheses 仍必填 1..5；new_hypotheses 仍可选无上限。"""
    hypo_schema = HypothesizeOutput.model_json_schema()
    assert hypo_schema["properties"]["hypotheses"]["minItems"] == 1
    assert hypo_schema["properties"]["hypotheses"]["maxItems"] == 5
    reeval_schema = ReevaluateOutput.model_json_schema()
    assert "new_hypotheses" in reeval_schema["properties"]
    assert ReevaluateOutput.model_fields["new_hypotheses"].is_required() is False


def test_four_node_prompt_render_smoke_walkthrough():
    """四节点 system/user 渲染走查冒烟：全 node 可渲染、非空、Schema 要点分节恒在。"""
    # 一个尽量贴近真实 __STATE__ 的富 state（decide 视角字段齐全）
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
        "screening_signals": [{"name": "KEYWORD", "result": "HIT", "score": 0.9}],
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
        "investigation_queue": [{"q": "是否仿牌？", "priority": 1, "status": "OPEN"}],
        "pending_tool_calls": [
            {"tool": "ImageAnalysisTool", "priority": 1, "reason": "补视觉证据"}
        ],
        "degraded": False,
        "failures": [],
        "budget": {
            "llm_calls": 3,
            "tool_calls": 2,
            "tokens": 900,
            "limits": {"max_llm_calls": 10, "max_tool_calls": 15, "max_tokens": 40000},
        },
    }
    assert set(SYSTEM_PROMPTS) == {"hypothesize", "plan", "reevaluate", "decide"}
    for node in SYSTEM_PROMPTS:
        sys_prompt = build_system_prompt(node)
        assert isinstance(sys_prompt, str) and len(sys_prompt) > 100
        user = build_user_prompt(node=node, state=rich_state, json_schema=_SCHEMA)
        assert isinstance(user, str) and user
        assert "## 输出格式要求" in user  # Schema 要点分节恒在
        assert "__STATE__" not in user  # 真实模型上下文不带裸 __STATE__ 标记
