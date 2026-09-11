"""确定性走查桩 —— 无 API key 也能端到端。

``ScriptedLLMBackend`` 实现 ``LLMBackend`` Protocol：按 ``node`` 分发并返回固定剧本的
结构化 JSON 文本；同 ``(node, __STATE__)`` → 同输出（无 API key、无网络、无随机、无
实例可变状态），eval 可重放。走查剧本对应复古运动鞋 P_88231 / M_5512 场景，目标
``llm_calls==8 / tool_calls==5``：hypothesize → plan(外观) → tools → reevaluate(①) →
plan(商品+商家) → tools → reevaluate(②) → plan(先例+政策) → tools → reevaluate(③ 收敛)
→ decide（plan 恰 3 次、LLM 总 8 次；三轮工具 1+2+2=5 次）。
"""

from __future__ import annotations

import json

from pra.agent.guardrails.llm_shell import LLMBackend, LLMBackendError, LLMResponse

# 剧本常量（只读，进程内不变）

# hypothesize 固定输出：4 条假设 (statement, prior)
_HYPOTHESES_SCRIPT: tuple = (
    ("普通复古设计，非品牌款", 0.5),
    ("参考知名品牌经典复古跑鞋设计", 0.4),
    ("刻意规避品牌识别（品牌字段空缺）", 0.2),
    ("商家系统性类似上架行为", 0.15),
)

# hypothesize 固定输出：2 条调查问题 (q, priority)
_QUEUE_SCRIPT: tuple = (
    ("外观是否与某知名品牌款高度相似？", 1),
    ("商家历史是否显示系统性类似行为？", 2),
)

# decide 固定提案
_DECIDE_RATIONALE = "证据链充分但涉及仿冒主观判定且政策指引高风险转人工，克制转人工"

# 证据类型常量（与 domain/models.py Evidence.type / tools 输出对齐）
_IMAGE_SIMILARITY = "IMAGE_SIMILARITY"
_PRODUCT_FACT = "PRODUCT_FACT"
_MERCHANT_HISTORY = "MERCHANT_HISTORY"
_CASE_PRECEDENT = "CASE_PRECEDENT"
_POLICY_REF = "POLICY_REF"

_STRONG_SIM_WEIGHT = 0.85  # sim_strong：IMAGE_SIMILARITY 强证据阈值
_CITATION_MAX = 200  # 引用串 value 截断上限（value 已是人读摘要）

_STATE_MARKER = "__STATE__"  # 消息内状态行前缀：__STATE__ {json}


def _citation(ev: dict) -> str:
    """证据引用/摘要串：``f"{type} {value}"``，value 截到 200 字符（不参与分支判定）。"""
    value = ev.get("value")
    value = "" if value is None else (value if isinstance(value, str) else str(value))
    return "{} {}".format(ev["type"], value[:_CITATION_MAX])


def _to_float(value: object) -> float:
    """把 __STATE__ 里的 weight 等安全转 float；缺失/非法/NaN → 0.0。"""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return f if f == f else 0.0  # NaN != NaN → 0.0


def _extract_state(messages) -> dict:
    """从 messages 里解析 ``__STATE__ {json}`` 状态行。

    状态约定（``complete`` 无 state 参数，故由节点把 state 子集挂到消息里）——标记后的
    dict 键：``case``（与 ``pra.domain.models`` 同形状）、``evidence``（元素含
    ``{"type","weight","value","ref_id","extra"}``）、``hypotheses``（元素含
    ``{"id","statement","prior","posterior","status"}``）、``queue`` 或
    ``investigation_queue``（元素含 ``{"q","priority","status"}``）。
    取首个含 ``__STATE__`` 的 str content 解析；失败 → ``{}``（空事实兜底）；其余内容
    一律忽略（仅占位审计）。
    """
    for msg in messages or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, str):
            continue
        idx = content.find(_STATE_MARKER)
        if idx < 0:
            continue
        rest = content[idx + len(_STATE_MARKER):].lstrip()
        if not rest:
            continue
        # 优先整段解析；失败则截取首 '{' 到末 '}' 再试（容忍附带叙述文本）。
        try:
            obj = json.loads(rest)
        except (ValueError, TypeError):
            start, end = rest.find("{"), rest.rfind("}")
            obj = None
            if 0 <= start < end:
                try:
                    obj = json.loads(rest[start : end + 1])
                except (ValueError, TypeError):
                    obj = None
        if isinstance(obj, dict):
            return obj
    return {}


def _evidence_list(state: dict) -> list:
    """state["evidence"] 归一化：只保留 dict 且 type 非空串的元素（保序）。"""
    out = []
    for ev in state.get("evidence") or []:
        if isinstance(ev, dict) and isinstance(ev.get("type"), str) and ev["type"]:
            out.append(ev)
    return out


def _first_of_type(evs: list, ev_type: str):
    """证据列表里 type 的首个元素（保序首现）；无则 None。"""
    for ev in evs:
        if ev["type"] == ev_type:
            return ev
    return None


def _hypothesis_list(state: dict) -> list:
    """state["hypotheses"] 归一化：只保留 dict 元素（保序）。"""
    return [h for h in (state.get("hypotheses") or []) if isinstance(h, dict)]


def _queue_list(state: dict) -> list:
    """调查队列：兼容 ``queue`` / ``investigation_queue`` 两个键名（保序）。"""
    queue = state.get("queue")
    if not isinstance(queue, list):
        queue = state.get("investigation_queue")
    if not isinstance(queue, list):
        return []
    return [item for item in queue if isinstance(item, dict)]


def _case_facts(state: dict) -> dict:
    """抽取 plan 分支需要的案件事实（防御式：缺失给安全空值，不崩）。"""
    case = state.get("case")
    case = case if isinstance(case, dict) else {}
    product = case.get("product")
    product = product if isinstance(product, dict) else {}
    images = product.get("images")
    images = images if isinstance(images, list) else []
    image_urls = [
        img["url"]
        for img in images
        if isinstance(img, dict)
        and isinstance(img.get("url"), str)
        and img["url"]  # 空串/缺失不入列，保证 ImageAnalysisTool 的 min_length=1 不炸
    ]
    product_id = product.get("product_id")
    merchant_id = case.get("merchant_id")
    category = product.get("category")
    return {
        "case": case,
        "product": product,
        "product_id": product_id if isinstance(product_id, str) and product_id else None,
        "merchant_id": (
            merchant_id if isinstance(merchant_id, str) and merchant_id else None
        ),
        "category": category if isinstance(category, str) else None,
        "image_urls": image_urls,
    }


def _risk_filters(category) -> dict:
    """plan 分支 3 的 filters：category（有则带）+ risk_type 词表。"""
    filters = {}
    if category:
        filters["category"] = category
    filters["risk_type"] = ["POTENTIAL_IP_RISK"]
    return filters


class ScriptedLLMBackend:
    """确定性走查桩（LLMBackend）：按 node 返回固定剧本 JSON，忽略消息正文。

    无实例可变状态（同 (node, __STATE__) 恒同输出，可重放）；未知 node →
    ``LLMBackendError``（供降级路径测试）。
    """

    name = "scripted-walkthrough"

    async def complete(self, *, node: str, messages: list, json_schema: dict) -> LLMResponse:
        """按 node 分发：解析 __STATE__ → 生成剧本 payload → JSON 文本返回。

        ``json_schema`` 为 OutputModel 的 JSON Schema（供真实后端约束；桩忽略）。
        """
        state: dict = _extract_state(messages)  # 缺省 {} → 各分支空事实兜底
        if node == "hypothesize":
            payload = self._hypothesize(state)
        elif node == "plan":
            payload = self._plan(state)
        elif node == "reevaluate":
            payload = self._reevaluate(state)
        elif node == "decide":
            payload = self._decide(state)
        else:
            raise LLMBackendError("unknown node: {}".format(node))
        # 桩无真实 provider 响应：tokens=0、usage=None —— 绝不伪造 token 拆分。
        return LLMResponse(
            content=json.dumps(payload, ensure_ascii=False), tokens=0, usage=None
        )

    # -- hypothesize：固定 4 假设 + 2 队列，与案件事实无关 ----------------------

    def _hypothesize(self, state: dict) -> dict:
        hypotheses = [
            {"statement": statement, "prior": prior}
            for statement, prior in _HYPOTHESES_SCRIPT
        ]
        queue = [
            {"q": question, "priority": priority}
            for question, priority in _QUEUE_SCRIPT
        ]
        return {
            "hypotheses": hypotheses,
            "investigation_queue": queue,
            "rationale": "依据品牌字段空缺与典型仿冒模式给出初始假设与调查问题。",
        }

    # -- plan：按证据 type 集合的四分支（1→2→3→4 顺序） ------------------------

    def _plan(self, state: dict) -> dict:
        facts = _case_facts(state)
        evs = _evidence_list(state)
        types = {ev["type"] for ev in evs}
        has_sim = _IMAGE_SIMILARITY in types
        urls = facts["image_urls"]

        # 分支 1：无 IMAGE_SIMILARITY → 先做外观比对（无图 → conclude 兜底）
        if not has_sim:
            if not urls:
                return {
                    "next_action": "conclude",
                    "tools": [],
                    "rationale": "无外观证据且案件无图片可查，本轮无调查动作。",
                }
            return {
                "next_action": "call_tools",
                "tools": [
                    {
                        "tool": "ImageAnalysisTool",
                        "args": {"image_urls": urls},
                        "reason": "验证外观是否对应知名品牌款",
                        "priority": 1,
                    }
                ],
                "rationale": "先做外观比对，验证是否对应知名品牌款。",
            }

        tools = []
        # 分支 2：有 IMAGE_SIMILARITY 但缺 PRODUCT_FACT / MERCHANT_HISTORY → 补齐
        if "PRODUCT_FACT" not in types:
            if facts["product_id"] is not None:
                tools.append(
                    {
                        "tool": "ProductTool",
                        "args": {"product_id": facts["product_id"]},
                        "reason": "核对商品在库事实（品牌空缺/版本漂移）",
                        "priority": 1,
                    }
                )
        if "MERCHANT_HISTORY" not in types:
            if facts["merchant_id"] is not None:
                tools.append(
                    {
                        "tool": "MerchantTool",
                        "args": {"merchant_id": facts["merchant_id"], "window_days": 90},
                        "reason": "核查商家系统性上架/下架历史",
                        "priority": 2,
                    }
                )
        if tools:
            return {
                "next_action": "call_tools",
                "tools": tools,
                "rationale": "补齐商品事实与商家历史，验证规避品牌与系统性上架假设。",
            }
        if "PRODUCT_FACT" not in types or "MERCHANT_HISTORY" not in types:
            # 证据仍缺但 product_id/merchant_id 均缺失（畸形案件事实）→ conclude，
            # 不穿透到分支 3；真实案件两 id 必填，此处仅防御 __STATE__ 畸形载荷。
            return {
                "next_action": "conclude",
                "tools": [],
                "rationale": "商品/商家标识缺失，无法继续补齐事实证据，本轮收尾。",
            }
        # 证据齐（有 PRODUCT_FACT 且 MERCHANT_HISTORY）→ 正常落到分支 3。

        # 分支 3：有 PRODUCT_FACT + MERCHANT_HISTORY 但缺 CASE_PRECEDENT / POLICY_REF
        if "CASE_PRECEDENT" not in types:
            tools.append(
                {
                    "tool": "CaseSearchTool",
                    "args": {
                        "query": "无品牌标识+外观高度模仿+商家多次重上架",
                        "filters": _risk_filters(facts["category"]),
                        "top_k": 5,
                    },
                    "reason": "检索无品牌标识+外观高度模仿+多次重上架的同类先例",
                    "priority": 1,
                }
            )
        if "POLICY_REF" not in types:
            tools.append(
                {
                    "tool": "PolicySearchTool",
                    "args": {
                        "query": "外观高度模仿品牌设计",
                        "filters": _risk_filters(facts["category"]),
                        "top_k": 5,
                        "effective_only": True,
                    },
                    "reason": "检索外观高度模仿品牌设计的政策条款",
                    "priority": 2,
                }
            )
        if tools:
            return {
                "next_action": "call_tools",
                "tools": tools,
                "rationale": "补齐同类先例与政策依据，评估仿冒风险定级。",
            }

        # 分支 4：五类关键证据齐备 → conclude（再查无益）
        return {
            "next_action": "conclude",
            "tools": [],
            "rationale": "外观/商品/商家/先例/政策证据已齐备，无需继续调查。",
        }

    # -- reevaluate：按假设 id + 证据 flags 更新（幂等） ------------------------

    def _reevaluate(self, state: dict) -> dict:
        evs = _evidence_list(state)
        sim_evs = [ev for ev in evs if ev["type"] == _IMAGE_SIMILARITY]
        sim_max = (
            max(_to_float(ev.get("weight")) for ev in sim_evs) if sim_evs else 0.0
        )
        strong = [ev for ev in sim_evs if _to_float(ev.get("weight")) >= _STRONG_SIM_WEIGHT]
        sim_strong = bool(strong)
        # 强证据取 weight 最大者（并列取首现 —— max 保序，确定性）
        strong_ev = max(strong, key=lambda ev: _to_float(ev.get("weight"))) if strong else None
        strong_ref = _citation(strong_ev) if strong_ev is not None else ""

        prod_ev = _first_of_type(evs, _PRODUCT_FACT)
        merch_ev = _first_of_type(evs, _MERCHANT_HISTORY)
        prod = prod_ev is not None
        merch = merch_ev is not None
        case_pre = _first_of_type(evs, _CASE_PRECEDENT) is not None
        policy = _first_of_type(evs, _POLICY_REF) is not None

        # flags 速查：sim_strong / sim_max / prod / merch / case_pre / policy
        hypothesis_updates = []
        for h in _hypothesis_list(state):
            hid = h.get("id")
            status = h.get("status")
            status = status if isinstance(status, str) else "PENDING"
            posterior = h.get("posterior")  # 可为 None（尚未评估）
            if hid == "H1" and sim_strong and status != "REFUTED":
                target_status, target_posterior, fields = "REFUTED", 0.05, {
                    "evidence_against": [strong_ref]
                }
            elif hid == "H2" and sim_strong:
                target_status, target_posterior = "SUPPORTED", round(sim_max, 2)
                fields = {"evidence_for": [strong_ref]}
            elif hid == "H3" and sim_strong and prod and merch:
                target_status, target_posterior = "SUPPORTED", 0.88
                fields = {
                    "evidence_for": [_citation(prod_ev), _citation(merch_ev)]
                }
            elif hid == "H4" and merch:
                target_status, target_posterior = "SUPPORTED", 0.85
                fields = {"evidence_for": [_citation(merch_ev)]}
            else:
                # 未知 id / 条件不满足 → 跳过
                continue
            # 幂等：目标 status/posterior 与当前相同 → 不列入（同一证据集重复调用
            # 输出稳定，不会来回翻；H3/H4 在首轮证据未齐前保持 PENDING 不误更新）
            if status == target_status and posterior == target_posterior:
                continue
            hypothesis_updates.append(
                {
                    "id": hid,
                    "posterior": target_posterior,
                    "status": target_status,
                    "evidence_for": [],
                    "evidence_against": [],
                    **fields,
                }
            )

        # queue_updates：q 含 "外观" 且 sim_strong → DONE；含 "商家历史" 且 merch → DONE
        queue_updates = []
        for item in _queue_list(state):
            q = item.get("q")
            if not isinstance(q, str) or not q:
                continue
            done = ("外观" in q and sim_strong) or ("商家历史" in q and merch)
            if not done:
                continue
            current_status = item.get("status")
            current_status = current_status if isinstance(current_status, str) else "OPEN"
            if current_status == "DONE":  # 幂等：已 DONE 不再重复产出
                continue
            queue_updates.append({"q": q, "status": "DONE"})

        sufficiency = "SUFFICIENT" if (case_pre and policy) else "INSUFFICIENT"
        return {
            "hypothesis_updates": hypothesis_updates,
            "queue_updates": queue_updates,
            "new_hypotheses": [],
            "evidence_sufficiency": sufficiency,
            "conflicts": [],
            "rationale": "按本轮证据批量更新假设状态与调查队列；无新增假设、无矛盾证据。",
        }

    # -- decide：固定提案 -------------------------------------------------------

    def _decide(self, state: dict) -> dict:
        evs = _evidence_list(state)
        return {
            "decision": "HUMAN_REVIEW",
            "risk_level": "HIGH",
            "risk_type": ["POTENTIAL_IP_RISK", "EVASION_PATTERN"],
            "confidence": 0.91,
            "evidence_ids": [_citation(ev) for ev in evs],
            "policy": ["POLICY_3.2"],
            "rationale": _DECIDE_RATIONALE,
        }


__all__ = ["ScriptedLLMBackend"]
