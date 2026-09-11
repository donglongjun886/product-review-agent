"""决策收口：LLM 只出提案，最终结论由本模块确定性算出。

输入 ``DecisionProposal`` + state → ``ReviewDecision``；任何改判都写进 ``overrides``
（为空 = 未改判）。顺序：硬规则命中 → REJECT/HIGH/1.0，LLM 不可覆盖；重算
decision_confidence（不采信 LLM 的 ``confidence``）；命中任一弃权条件或无事前提案
（预算耗尽 / 降级 / LLM 失败）→ HUMAN_REVIEW + 归因码；PASS/REJECT 提案不过对应
Gate → HUMAN_REVIEW + 归因码；否则采纳提案，confidence 用重算值。

不变量：``risk_level`` / ``risk_type`` 不参与 Gate 判定（仅供人工队列排序）；
``policy`` 只从 POLICY_REF 证据的 ``extra["policy_id"]`` 读，不采信 LLM 提案。
state 元素是 Pydantic 实例，确定性逻辑只读 extra/weight/ref_id；唯一例外是
``visual_claim_unsupported`` 读 statement 文本做维度归类，命中只导向 HUMAN_REVIEW。
谓词全为纯函数，state 缺 hypotheses/evidence 时安全返回。
"""

from __future__ import annotations

from pra.agent.guardrails.budget import budget_exceeded, snapshot_budget
from pra.agent.guardrails.errors import key_tool_failure
from pra.agent.guardrails.hard_rules import hard_rule_hit
from pra.agent.guardrails.schemas import DecisionProposal
from pra.domain.models import (
    Budget,
    Decision,
    Hypothesis,
    HypothesisStatus,
    RiskLevel,
    RiskType,
    ReviewDecision,
)

# 常量：本地声明，不与其它模块共享可变状态。

HIGH_PRIOR_THRESHOLD = 0.3  # 仅 PASS/REJECT Gate 的「高优先」口径（prior>=0.3）
CONFIDENCE_ABSTAIN_THRESHOLD = 0.7  # REJECT Gate 安全门槛（decision_confidence）
MAX_EXPECTED_EVIDENCE = 8  # dc 公式的完整性分母
CITABLE_TYPES = {"CASE_PRECEDENT", "POLICY_REF"}  # 与 converge 同义（本地重声明）

# overrides 原因码（写进 ReviewDecision.overrides）
R1_HARD_RULE = "R1_HARD_RULE"
R2_REJECT_GATE_FAIL = "R2_REJECT_GATE_FAIL"
R3_BUDGET_EXHAUSTED = "R3_BUDGET_EXHAUSTED"
R3_CRITICAL_CONFLICT = "R3_CRITICAL_CONFLICT"
R3_KEY_TOOL_FAILED = "R3_KEY_TOOL_FAILED"
R3_POLICY_UNCERTAIN = "R3_POLICY_UNCERTAIN"
R3_HYPOTHESES_INDISTINGUISHABLE = "R3_HYPOTHESES_INDISTINGUISHABLE"
R3_VISUAL_CLAIM_UNSUPPORTED = "R3_VISUAL_CLAIM_UNSUPPORTED"  # 视觉声称但无视觉证据
R4_PASS_GATE_FAIL = "R4_PASS_GATE_FAIL"
R5_DEGRADED_OR_FAILED_STEP = "R5_DEGRADED_OR_FAILED_STEP"

# 强相似 / 视觉存在性阈值：镜像 image_analysis.tool 的 0.85 与 0.70，本地声明。
# 存在性用普通档即算「存在视觉证据」，不代表最终相似结论；0.85 强档语义不动。
_SIM_STRONG = 0.85
_SIM_PRESENT = 0.70

# 外观/视觉维度声称关键词表：关键词谓词为主判据，与 llm_prompts.py reevaluate
# prompt 第 8 条的外观表述清单同源（**两处措辞须同步维护**）。外延只覆盖
# 「外观/造型/相似」语义，不含「字样/标题」类文本声称 —— OCR_TEXT 可支撑的文本
# 声称不在拦截面。命中仅做维度归类、不解析数值；归类过宽只多转人工（安全侧），
# 过窄退回现状 —— 宁宽勿窄。只作用于 SUPPORTED 高优先假设，PASS 侧不受影响。
VISUAL_CLAIM_MARKERS: tuple[str, ...] = (
    # llm_prompts reevaluate 例句逐字镜像（须同步维护）
    "外观相似",
    "高度相似",
    "同款外观",
    "视觉仿冒",
    "复刻外观",
    "版型一致",
    "长得像",
    # 外观/造型维度词
    "外观",
    "造型",
    "鞋型",
    "版型",
    "廓形",
    "配色",
    "图案",
    "印花",
    # 含模仿/雷同语义的断言词（覆盖措辞漂移；安全侧从宽）
    "同款",
    "仿冒",
    "复刻",
    "模仿",
)


def _has_citable(evidence) -> bool:
    """是否存在带 ``ref_id`` 的可引用依据（POLICY_REF / CASE_PRECEDENT）。"""
    return any(e.type in CITABLE_TYPES and e.ref_id for e in evidence)


def _strong_similarity(evidence) -> bool:
    """是否存在强相似 IMAGE_SIMILARITY（``weight >= 0.85``）。"""
    return any(e.type == "IMAGE_SIMILARITY" and (e.weight or 0.0) >= _SIM_STRONG for e in evidence)


def _top_supported_posterior(state) -> float:
    """最高 SUPPORTED 假设后验（None 按 0 处理）；无则 0.0。"""
    return max(
        ((h.posterior if h.posterior is not None else 0.0)
         for h in (state.get("hypotheses") or [])
         if h.status == HypothesisStatus.SUPPORTED),
        default=0.0,
    )


# 谓词（全部纯函数、确定性、可单测；空/缺失安全）


def high_priority(hypotheses) -> list:
    """``prior >= HIGH_PRIOR_THRESHOLD``（0.3）的假设。

    只用于 PASS/REJECT Gate 的过滤口径，**不用于收敛判定**；prior 为 None 按 0。
    """
    return [
        h
        for h in (hypotheses or [])
        if h.prior is not None and h.prior >= HIGH_PRIOR_THRESHOLD
    ]


def contradiction_detect(state) -> bool:
    """关键证据矛盾检测：「相似度极高但商家历史干净」。

    True = ∃ 强相似 IMAGE_SIMILARITY（``weight >= 0.85``）∧ ∃ MERCHANT_HISTORY 且
    ``removals == 0 且 title == 0``。extra 缺失/未回填的按「不干净」处理。
    """
    evs = state.get("evidence") or []
    has_strong_sim = _strong_similarity(evs)
    has_clean_merchant = any(
        e.type == "MERCHANT_HISTORY"
        and (e.extra or {}).get("removals") == 0
        and (e.extra or {}).get("title") == 0
        for e in evs
    )
    return bool(has_strong_sim and has_clean_merchant)


def policy_indeterminate(state) -> bool:
    """政策无法确定 —— 只拦 REJECT 侧，不拦 PASS 候选。

    True = 存在 SUPPORTED 高优先假设（拟 REJECT 案需政策依据）∧ 证据链无任何**带
    ref_id** 的 POLICY_REF（CASE_PRECEDENT 不替代政策）。无 SUPPORTED 高优先假设 →
    False：否则弃权清单会抢在两道 Gate 之前把无政策证据的案型全转人工。
    """
    has_supported_high_priority = any(
        h.status == HypothesisStatus.SUPPORTED
        for h in high_priority(state.get("hypotheses") or [])
    )
    if not has_supported_high_priority:
        return False
    return not any(
        e.type == "POLICY_REF" and e.ref_id
        for e in (state.get("evidence") or [])
    )


def indistinguishable_hypotheses(state) -> bool:
    """多假设不可区分：≥2 个高优先 SUPPORTED 假设且 ``evidence_for`` 集合相等。

    其余情况 False；hypotheses 空/缺失 → False。
    """
    supported = [
        h
        for h in high_priority(state.get("hypotheses") or [])
        if h.status == HypothesisStatus.SUPPORTED and h.evidence_for
    ]
    if len(supported) < 2:
        return False
    distinct_sets = {tuple(sorted(set(h.evidence_for))) for h in supported}
    return len(distinct_sets) < len(supported)


def _looks_visual_claim(statement) -> bool:
    """statement 是否落在「外观/视觉维度」（命中任一 ``VISUAL_CLAIM_MARKERS`` 子串）。"""
    text = statement or ""
    return any(marker in text for marker in VISUAL_CLAIM_MARKERS)


def visual_evidence_present(evidence) -> bool:
    """证据链是否存在「视觉相似证据」。

    任一 ``IMAGE_LOGO`` → True（不等于证明侵权）；任一 ``IMAGE_SIMILARITY`` 且
    ``weight >= 0.70`` → True（只读 weight、不依赖 ``extra.strong``，防注入）。
    OCR_TEXT / POLICY_REF / CASE_PRECEDENT / PRODUCT_FACT / MERCHANT_HISTORY 一律
    不算：OCR 是图像来源**文本**证据，替代不了视觉直接测量。
    """
    evs = evidence or []
    if any(e.type == "IMAGE_LOGO" for e in evs):
        return True
    return any(
        e.type == "IMAGE_SIMILARITY" and (e.weight or 0.0) >= _SIM_PRESENT
        for e in evs
    )


def visual_claim_unsupported(state) -> bool:
    """视觉声称无视觉证据 —— 声称维度与证据维度错配。

    True = 存在「高优先 ∧ SUPPORTED ∧ statement 落外观/视觉关键词」的假设，且证据链
    无任何视觉证据。**先例只证明「同类曾被拒」的历史事实，替代不了「本商品与某品牌
    款外观相似」的直接测量 —— 先例只能佐证相同证据维度，不能把历史案例事实迁移为
    当前案件事实**，故应转人工而非自动 REJECT。
    """
    visual_supported = [
        h
        for h in high_priority(state.get("hypotheses") or [])
        if h.status == HypothesisStatus.SUPPORTED and _looks_visual_claim(h.statement)
    ]
    if not visual_supported:
        return False
    return not visual_evidence_present(state.get("evidence") or [])


def key_evidence_complete(state) -> bool:
    """关键证据完整 = 无未解决的 critical Tool 失败。"""
    return not key_tool_failure(state, state.get("failures") or [])


def evidence_sufficient(state) -> bool:
    """证据充分：每个 SUPPORTED 高优先假设 ``evidence_for`` 非空，且无关键 Tool 失败。"""
    if key_tool_failure(state, state.get("failures") or []):
        return False
    supported = [
        h
        for h in high_priority(state.get("hypotheses") or [])
        if h.status == HypothesisStatus.SUPPORTED
    ]
    return all(bool(h.evidence_for) for h in supported)


def pass_gate(state) -> bool:
    """PASS Gate：四项全满足才放行。

    ① 高优先假设非空（空 hypotheses 防御：不误 PASS）；② 全部 REFUTED 且各自
    ``evidence_against`` 非空（「有反驳证据」而非「没查到」）；③ 无未解决 critical
    Tool 失败；④ 无关键矛盾。
    """
    hp = high_priority(state.get("hypotheses") or [])
    if not hp:
        return False
    if not all(h.status == HypothesisStatus.REFUTED and h.evidence_against for h in hp):
        return False
    if not key_evidence_complete(state):
        return False
    if contradiction_detect(state):
        return False
    return True


def reject_gate(state, dc: float) -> bool:
    """REJECT Gate：自动拒绝的安全门槛，五项全满足才放行。

    ① ∃ 高优先 SUPPORTED；② evidence_sufficient；③ ∃ 带 ``ref_id`` 的 POLICY_REF /
    CASE_PRECEDENT；④ ``dc >= 0.7``（**只约束自动 REJECT 侧**）；⑤ 无关键矛盾。
    """
    hp = high_priority(state.get("hypotheses") or [])
    if not any(h.status == HypothesisStatus.SUPPORTED for h in hp):
        return False
    if not evidence_sufficient(state):
        return False
    if not _has_citable(state.get("evidence") or []):
        return False
    if dc < CONFIDENCE_ABSTAIN_THRESHOLD:
        return False
    if contradiction_detect(state):
        return False
    return True


def finalize_decision_confidence(state) -> float:
    """确定性重算 decision_confidence（「自动判安全把握」，非违规概率）。

    ``top = max(SUPPORTED 假设 posterior, 缺省 0.0)``；
    ``completeness = min(len(evidence) / MAX_EXPECTED_EVIDENCE, 1.0)``；
    ``citation = 1.0 if ∃ 带 ref_id 的可引用依据 else 0.0``；
    ``conflict = 0.2 if contradiction_detect else 0.0``；
    ``c = 0.45*top + 0.25*completeness + 0.20*citation + 0.10 - conflict``，
    返回 ``round(clip(c, 0..1), 2)``。空/缺失 → 0.10 基线。
    """
    evs = state.get("evidence") or []
    top = _top_supported_posterior(state)
    completeness = min(len(evs) / MAX_EXPECTED_EVIDENCE, 1.0)
    citation = 1.0 if _has_citable(evs) else 0.0
    conflict = 0.2 if contradiction_detect(state) else 0.0
    c = 0.45 * top + 0.25 * completeness + 0.20 * citation + 0.10 - conflict
    return round(min(max(c, 0.0), 1.0), 2)


def finalize_risk_level(state, proposal) -> RiskLevel:
    """风险等级：proposal 声明了 ``risk_level`` 则用之，否则按最高 SUPPORTED posterior
    派生（>=0.8 HIGH / >=0.5 MEDIUM / >=0.2 LOW / 其余 NONE）。

    仅供展示与人工队列排序，**不参与 Gate 判定**。
    """
    if proposal is not None and proposal.risk_level:
        return RiskLevel(proposal.risk_level)
    top = _top_supported_posterior(state)
    if top >= 0.8:
        return RiskLevel.HIGH
    if top >= 0.5:
        return RiskLevel.MEDIUM
    if top >= 0.2:
        return RiskLevel.LOW
    return RiskLevel.NONE


def finalize_risk_type(state, proposal) -> list:
    """风险类型：proposal 有非空 ``risk_type`` 则用之，否则按证据派生。

    强 IMAGE_SIMILARITY（``weight >= 0.85``）→ POTENTIAL_IP_RISK；MERCHANT_HISTORY
    且 ``extra.removals >= 3`` 或 ``extra.title >= 3`` → EVASION_PATTERN（extra 缺失
    按 0）。顺序固定：IP 先、EVASION 后；无命中 → ``[]``。
    """
    if proposal is not None and proposal.risk_type:
        return list(proposal.risk_type)
    risk: list = []
    evs = state.get("evidence") or []
    if _strong_similarity(evs):
        risk.append(RiskType.POTENTIAL_IP_RISK)
    for e in evs:
        if e.type == "MERCHANT_HISTORY":
            extra = e.extra or {}
            if (extra.get("removals") or 0) >= 3 or (extra.get("title") or 0) >= 3:
                risk.append(RiskType.EVASION_PATTERN)
                break
    return risk


# 组装与 overlay 收口


def build_decision(
    state: dict,
    proposal,
    *,
    decision: Decision,
    risk_level: RiskLevel,
    risk_type: list,
    decision_confidence: float,
    overrides: list,
) -> ReviewDecision:
    """组装 ``ReviewDecision``，取值全部来自确定性参数与 state。

    evidence / hypothesis_trace 直接引用 state 的同型对象列表（浅拷贝容器、元素同一
    实例）；policy 取排序去重的 POLICY_REF ``extra["policy_id"]``（缺 key 则跳过）；
    budget_used 为 ``snapshot_budget`` 快照。
    """
    evidence = list(state.get("evidence") or [])
    policy = sorted(
        {
            str(e.extra["policy_id"])
            for e in evidence
            if e.type == "POLICY_REF"
            and (e.extra or {}).get("policy_id") is not None
        }
    )
    budget = state.get("budget")
    return ReviewDecision(
        decision=decision,
        risk_level=risk_level,
        risk_type=list(risk_type),
        decision_confidence=decision_confidence,
        evidence=evidence,
        policy=policy,
        hypothesis_trace=list(state.get("hypotheses") or []),
        budget_used=snapshot_budget(budget) if budget is not None else Budget(),
        overrides=list(overrides),
    )


def _abstention_codes(state) -> list:
    """HUMAN_REVIEW 弃权清单：顺序固定，命中码全量收集。

    只认 ``state["degraded"]`` 与未解决的 critical Tool 失败；``severity="warn"`` 的
    失败只进审计、不触发。
    """
    codes: list = []
    budget = state.get("budget")
    if budget is not None and budget_exceeded(budget) is not None:
        codes.append(R3_BUDGET_EXHAUSTED)
    if contradiction_detect(state):
        codes.append(R3_CRITICAL_CONFLICT)
    if key_tool_failure(state, state.get("failures") or []):
        codes.append(R3_KEY_TOOL_FAILED)
    if policy_indeterminate(state):
        codes.append(R3_POLICY_UNCERTAIN)
    if indistinguishable_hypotheses(state):
        codes.append(R3_HYPOTHESES_INDISTINGUISHABLE)
    if visual_claim_unsupported(state):
        # 视觉声称 SUPPORTED 但无视觉证据 → 声称维度错配，转人工
        codes.append(R3_VISUAL_CLAIM_UNSUPPORTED)
    if state.get("degraded"):
        codes.append(R5_DEGRADED_OR_FAILED_STEP)
    return codes


def _human_review(state, proposal, dc: float, overrides: list) -> ReviewDecision:
    """转人工的统一组装：risk/risk_type 用 ``finalize_risk_*``，decision_confidence
    用确定性 dc。
    """
    return build_decision(
        state,
        proposal,
        decision=Decision.HUMAN_REVIEW,
        risk_level=finalize_risk_level(state, proposal),
        risk_type=finalize_risk_type(state, proposal),
        decision_confidence=dc,
        overrides=overrides,
    )


def run_decision_overlay(state: dict, proposal) -> ReviewDecision:
    """decide 收口主函数：顺序固定，overrides 全量写入。

    ``proposal`` 为 None（预算耗尽 / 降级 / LLM 失败）时无提案。顺序：① 硬规则命中 →
    直接 REJECT/HIGH/1.0；② 重算 ``dc``；③ 弃权清单任一命中 → HUMAN_REVIEW；
    ④ proposal 为 None 且无弃权码 → 补 ``R5_DEGRADED_OR_FAILED_STEP``；⑤ PASS 未过
    pass_gate → +R4_PASS_GATE_FAIL、REJECT 未过 reject_gate → +R2_REJECT_GATE_FAIL；
    ⑥ 采纳：decision 与 risk 用提案、decision_confidence 用 **dc**、overrides=[]。

    空 hypotheses 防御：两道 Gate 都不通过 → PASS/REJECT 提案均被降 HUMAN。
    """
    # 1) 硬规则优先（不可被 LLM 覆盖；防漏放）
    hit = hard_rule_hit(state)
    if hit is not None:
        return build_decision(
            state,
            proposal,
            decision=Decision.REJECT,
            risk_level=RiskLevel.HIGH,
            risk_type=list(hit.risk_types),
            decision_confidence=1.0,
            overrides=[R1_HARD_RULE],
        )

    # 2) 确定性重算 decision_confidence（LLM confidence 仅参考）
    dc = finalize_decision_confidence(state)

    # 3) 弃权清单（命中码全量写入 overrides）
    overrides = _abstention_codes(state)

    # 4) proposal None 兜底：无弃权码时补 R5
    if proposal is None and not overrides:
        overrides.append(R5_DEGRADED_OR_FAILED_STEP)

    if overrides:
        return _human_review(state, proposal, dc, overrides)

    # 5) PASS / REJECT Gate：校验 LLM 提案
    if proposal.decision == "PASS" and not pass_gate(state):
        return _human_review(state, proposal, dc, [R4_PASS_GATE_FAIL])
    if proposal.decision == "REJECT" and not reject_gate(state, dc):
        return _human_review(state, proposal, dc, [R2_REJECT_GATE_FAIL])

    # 6) 采纳：HUMAN 提案或 Gate 通过
    return build_decision(
        state,
        proposal,
        decision=Decision(proposal.decision),
        risk_level=RiskLevel(proposal.risk_level),
        risk_type=list(proposal.risk_type),
        decision_confidence=dc,
        overrides=[],
    )
