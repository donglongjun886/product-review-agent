"""确定性走查桩 —— 无 API key 也能端到端。

``ScriptedLLMBackend`` 实现 ``LLMBackend`` Protocol：按 ``node`` 分发并返回固定剧本的
结构化 JSON 文本；同 ``(node, state)`` → 同输出（无 API key、无网络、无随机、无
实例可变状态），eval 可重放。走查剧本对应复古运动鞋 P_88231 / M_5512 场景，只调度 4 工具
世界（ProductTool / MerchantTool / CaseSearchTool / PolicySearchTool），目标
``llm_calls==6 / tool_calls==4``：hypothesize → plan(商品事实+商家行为) → tools(2) →
reevaluate(①) → plan(先例+政策) → tools(2) → reevaluate(② 收敛) → decide
（plan 恰 2 次、LLM 总 6 次；两轮工具 2+2=4 次）。
"""

from __future__ import annotations

import json

from pra.agent.guardrails.llm_shell import LLMBackendError, LLMResponse
from pra.domain.measurement import MERCHANT_DIRTY_MIN

# 剧本常量（只读，进程内不变）

# hypothesize 固定输出：4 条假设 (statement, prior)
_HYPOTHESES_SCRIPT: tuple = (
    ("商品在库事实与案件快照一致，无字段冲突", 0.5),
    ("商家历史行为干净，不构成系统性规避", 0.4),
    ("存在同类违规先例可供参照", 0.2),
    ("存在生效政策条款可支撑处置", 0.15),
)

# decide 固定提案
_DECIDE_RATIONALE = "商家画像与同类先例、政策条款成立，但本 listing 缺文本确证，克制转人工"

# 证据类型常量（与 domain/models.py Evidence.type / tools 输出对齐）
_PRODUCT_FACT = "PRODUCT_FACT"
_MERCHANT_HISTORY = "MERCHANT_HISTORY"
_CASE_PRECEDENT = "CASE_PRECEDENT"
_POLICY_REF = "POLICY_REF"

_CITATION_MAX = 200  # 引用串 value 截断上限（value 已是人读摘要）


def _citation(ev: dict) -> str:
    """证据引用/摘要串：``f"{type} {value}"``，value 截到 200 字符（不参与分支判定）。"""
    value = ev.get("value")
    value = "" if value is None else (value if isinstance(value, str) else str(value))
    return "{} {}".format(ev["type"], value[:_CITATION_MAX])


def _to_float(value: object) -> float:
    """把 state 里的 weight 等安全转 float；缺失/非法/NaN → 0.0。"""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return f if f == f else 0.0  # NaN != NaN → 0.0


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


def _merchant_dirty(ev: dict) -> bool:
    """MERCHANT_HISTORY 是否达「系统性规避」阈值（removals/title >= MERCHANT_DIRTY_MIN）。"""
    extra = ev.get("extra")
    extra = extra if isinstance(extra, dict) else {}
    return (
        _to_float(extra.get("removals")) >= MERCHANT_DIRTY_MIN
        or _to_float(extra.get("title")) >= MERCHANT_DIRTY_MIN
    )


def _policy_ids(evs: list) -> list[str]:
    """POLICY_REF 证据 ``extra.policy_id`` 去重列表（保序；缺失/非串跳过）。"""
    ids: list[str] = []
    for ev in evs:
        if ev["type"] != _POLICY_REF:
            continue
        extra = ev.get("extra")
        pid = extra.get("policy_id") if isinstance(extra, dict) else None
        if isinstance(pid, str) and pid and pid not in ids:
            ids.append(pid)
    return ids


def _hypothesis_list(state: dict) -> list:
    """state["hypotheses"] 归一化：只保留 dict 元素（保序）。"""
    return [h for h in (state.get("hypotheses") or []) if isinstance(h, dict)]


def _case_facts(state: dict) -> dict:
    """抽取 plan 分支需要的案件事实（防御式：缺失给安全空值，不崩）。"""
    case = state.get("case")
    case = case if isinstance(case, dict) else {}
    product = case.get("product")
    product = product if isinstance(product, dict) else {}
    product_id = product.get("product_id")
    merchant_id = case.get("merchant_id")
    category = product.get("category")
    return {
        "product_id": product_id if isinstance(product_id, str) and product_id else None,
        "merchant_id": (
            merchant_id if isinstance(merchant_id, str) and merchant_id else None
        ),
        "category": category if isinstance(category, str) else None,
    }


def _risk_filters(category) -> dict:
    """plan 分支 2 的 filters：category（有则带）+ risk_type 词表。"""
    filters = {}
    if category:
        filters["category"] = category
    filters["risk_type"] = ["POTENTIAL_IP_RISK"]
    return filters


class ScriptedLLMBackend:
    """确定性走查桩（LLMBackend）：按 node 返回固定剧本 JSON，忽略 feedback。

    无实例可变状态（同 ``(node, state)`` 恒同输出，可重放）；未知 node →
    ``LLMBackendError``（供降级路径测试）。
    """

    name = "scripted-walkthrough"

    async def complete(
        self,
        *,
        node: str,
        state: dict,
        json_schema: dict,
        feedback: list[str] | None = None,
    ) -> LLMResponse:
        """按 node 分发：读结构化 state → 生成剧本 payload → JSON 文本返回。

        ``json_schema`` 为 OutputModel 的 JSON Schema（供真实后端约束；桩忽略）；
        ``feedback`` 是上一轮校验失败的修正提示，桩的剧本固定、不据它改写输出。
        """
        state = state if isinstance(state, dict) else {}  # 缺省 {} → 各分支空事实兜底
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

    # -- hypothesize：固定 4 假设，与案件事实无关 -------------------------------

    def _hypothesize(self, state: dict) -> dict:
        hypotheses = [
            {"statement": statement, "prior": prior}
            for statement, prior in _HYPOTHESES_SCRIPT
        ]
        return {
            "hypotheses": hypotheses,
            "rationale": "依据品牌字段空缺、商家行为与可引用依据需求给出初始假设。",
        }

    # -- plan：按证据 type 集合的三分支（1→2→3 顺序） --------------------------

    def _plan(self, state: dict) -> dict:
        facts = _case_facts(state)
        evs = _evidence_list(state)
        types = {ev["type"] for ev in evs}

        # 分支 1：本 listing 事实与商家行为证据缺失 → 补齐（本轮 ≤2 条工具调用）
        tools = []
        if "PRODUCT_FACT" not in types and facts["product_id"] is not None:
            tools.append(
                {
                    "tool": "ProductTool",
                    "args": {"product_id": facts["product_id"]},
                    "reason": "核对商品在库事实（品牌空缺/版本漂移）",
                    "priority": 1,
                }
            )
        if "MERCHANT_HISTORY" not in types and facts["merchant_id"] is not None:
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
                "rationale": "补齐商品在库事实与商家行为画像，验证字段一致性与规避模式。",
            }
        if "PRODUCT_FACT" not in types or "MERCHANT_HISTORY" not in types:
            # 证据仍缺但 product_id/merchant_id 均缺失（畸形案件事实）→ conclude，
            # 不穿透到分支 2；真实案件两 id 必填，此处仅防御畸形 state 载荷。
            return {
                "next_action": "conclude",
                "tools": [],
                "rationale": "商品/商家标识缺失，无法继续补齐事实证据，本轮收尾。",
            }
        # 证据齐（有 PRODUCT_FACT 且 MERCHANT_HISTORY）→ 正常落到分支 2。

        # 分支 2：事实齐备但缺可引用依据（同类先例 / 政策条款）
        if "CASE_PRECEDENT" not in types:
            tools.append(
                {
                    "tool": "CaseSearchTool",
                    "args": {
                        "query": "无品牌标识+商家多次重上架+品牌规避",
                        "filters": _risk_filters(facts["category"]),
                        "top_k": 5,
                    },
                    "reason": "检索无品牌标识+商家多次重上架的同类先例",
                    "priority": 1,
                }
            )
        if "POLICY_REF" not in types:
            tools.append(
                {
                    "tool": "PolicySearchTool",
                    "args": {
                        "query": "品牌规避与商家系统性重上架",
                        "filters": _risk_filters(facts["category"]),
                        "top_k": 5,
                        "effective_only": True,
                    },
                    "reason": "检索品牌规避/系统性重上架的政策条款",
                    "priority": 2,
                }
            )
        if tools:
            return {
                "next_action": "call_tools",
                "tools": tools,
                "rationale": "补齐同类先例与政策依据，评估规避风险定级。",
            }

        # 分支 3：事实 + 可引用依据齐备 → conclude（再查无益）
        return {
            "next_action": "conclude",
            "tools": [],
            "rationale": "商品/商家/先例/政策证据已齐备，无需继续调查。",
        }

    # -- reevaluate：按假设 id + 证据 flags 更新（幂等） ------------------------

    def _reevaluate(self, state: dict) -> dict:
        evs = _evidence_list(state)
        prod_ev = _first_of_type(evs, _PRODUCT_FACT)
        merch_ev = _first_of_type(evs, _MERCHANT_HISTORY)
        case_pre_ev = _first_of_type(evs, _CASE_PRECEDENT)
        policy_ev = _first_of_type(evs, _POLICY_REF)
        merch_dirty = merch_ev is not None and _merchant_dirty(merch_ev)

        # flags 速查：prod / merch / merch_dirty / case_pre / policy
        hypothesis_updates = []
        for h in _hypothesis_list(state):
            hid = h.get("id")
            status = h.get("status")
            status = status if isinstance(status, str) else "PENDING"
            posterior = h.get("posterior")  # 可为 None（尚未评估）
            if hid == "H1" and prod_ev is not None:
                target_status, target_posterior, fields = "SUPPORTED", 0.9, {
                    "evidence_for": [_citation(prod_ev)]
                }
            elif hid == "H2" and merch_ev is not None:
                if merch_dirty:
                    target_status, target_posterior, fields = "REFUTED", 0.1, {
                        "evidence_against": [_citation(merch_ev)]
                    }
                else:
                    target_status, target_posterior, fields = "SUPPORTED", 0.9, {
                        "evidence_for": [_citation(merch_ev)]
                    }
            elif hid == "H3" and case_pre_ev is not None:
                target_status, target_posterior, fields = "SUPPORTED", 0.8, {
                    "evidence_for": [_citation(case_pre_ev)]
                }
            elif hid == "H4" and policy_ev is not None:
                target_status, target_posterior, fields = "SUPPORTED", 0.85, {
                    "evidence_for": [_citation(policy_ev)]
                }
            else:
                # 未知 id / 条件不满足 → 跳过
                continue
            # 幂等：目标 status/posterior 与当前相同 → 不列入（同一证据集重复调用
            # 输出稳定，不会来回翻；H3/H4 在证据未齐前保持 PENDING 不误更新）
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

        sufficiency = (
            "SUFFICIENT"
            if all(ev is not None for ev in (prod_ev, merch_ev, case_pre_ev, policy_ev))
            else "INSUFFICIENT"
        )
        return {
            "hypothesis_updates": hypothesis_updates,
            "new_hypotheses": [],
            "evidence_sufficiency": sufficiency,
            "conflicts": [],
            "rationale": "按本轮证据批量更新假设状态；无新增假设、无矛盾证据。",
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
            "policy": _policy_ids(evs),
            "rationale": _DECIDE_RATIONALE,
        }


__all__ = ["ScriptedLLMBackend"]
