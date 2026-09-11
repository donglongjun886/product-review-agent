"""决策收口：LLM 只出提案，最终结论由本模块用**事实**确定性算出。

输入 = ``DecisionProposal`` + state 的**事实通道**（``case`` / ``evidence`` /
``measurement_capabilities`` / ``failures`` / ``budget`` / ``degraded``）。
**终裁不读任何 LLM 生成量**：不读 ``prior`` / ``posterior`` / ``Hypothesis.status`` /
``evidence_for`` / 提案里的 ``confidence``。``proposal`` 只决定"要走哪道 Gate"
（PASS 提案走 PASS Gate、REJECT 提案走 REJECT Gate）以及非判定性的展示字段。

顺序：① 硬规则命中 → REJECT/HIGH/1.0，LLM 不可覆盖；② 重算 ``decision_confidence``（事实侧）；
③ 弃权清单（预算/冲突/关键工具失败/**关键测量缺口**/不可测维度/降级）任一命中 → HUMAN_REVIEW
+ 归因码；④ 无提案且无码 → 补 ``R5``；⑤ 提案 PASS 不过 PASS Gate → ``+R4``、提案 REJECT 不过
REJECT Gate → ``+R2``；⑥ 采纳：decision 用提案、confidence 用重算值、overrides=[]。

PASS Gate（全满足才放行）：无维度匹配的**阳性证据** ∧ 无**规则侧阳性**（R-102/R-302 命中）
∧ 本案 required 维度**全部覆盖**（无 NOT_MEASURED、无 UNMEASURABLE）∧ 无关键工具失败 ∧ 无证据冲突。

REJECT Gate（全满足才自动拒绝）：∃ **真实证据链中、与风险维度匹配的硬阳性**（强相似 ≥0.85 /
Logo / 商家达阈值 / 显式 ``MEASUREMENT=POSITIVE``）∧ ∃ 带 ``ref_id`` 的可引用依据 ∧ ``dc>=0.7``
∧ 无证据冲突。**弱相似 0.70~0.85 不是硬阳性** —— 它要靠"商品事实"维度交叉，不能单独撑起自动拒绝。

不变量：``risk_level`` / ``risk_type`` 不参与判定（仅供人工队列排序）；``policy`` 只从
``POLICY_REF`` 证据的 ``extra["policy_id"]`` 读，不采信提案；``high_priority`` 仅保留为
展示/排序工具，Gate 不调用它。
"""

from __future__ import annotations

from pra.agent.guardrails.budget import budget_exceeded, snapshot_budget
from pra.agent.guardrails.errors import key_tool_failure
from pra.agent.guardrails.hard_rules import hard_rule_hit
from pra.agent.guardrails.measurements import (
    ALWAYS_COVERED_DIMENSIONS,
    CoverageReport,
    coverage_report,
    text_evasion_hit,
)
from pra.domain.measurement import (
    DIM_IMAGE_APPEARANCE,
    DIM_MERCHANT_PROFILE,
    DIM_TEXT_COMPLIANCE,
)
from pra.domain.models import (
    Budget,
    Decision,
    ReviewDecision,
    RiskLevel,
    RiskType,
)

# 常量：本地声明，不与其它模块共享可变状态。

HIGH_PRIOR_THRESHOLD = 0.3  # **仅**用于人工队列排序/展示；Gate 不消费
CONFIDENCE_ABSTAIN_THRESHOLD = 0.7  # REJECT Gate 安全门槛（decision_confidence）
CITABLE_TYPES = {"CASE_PRECEDENT", "POLICY_REF"}  # 可引用依据类型

# dc 公式的结构系数（覆盖主项 / 证据强度次项 / 引用 / 基线 / 冲突惩罚）。
# **结构性给定，不用任何数据集拟合**；阈值 CONFIDENCE_ABSTAIN_THRESHOLD 保持不变。
DC_W_COVERAGE = 0.40
DC_W_STRENGTH = 0.30
DC_W_CITATION = 0.20
DC_BASE = 0.10
DC_CONFLICT_PENALTY = 0.20

# overrides 原因码（写进 ReviewDecision.overrides）
R1_HARD_RULE = "R1_HARD_RULE"
R2_REJECT_GATE_FAIL = "R2_REJECT_GATE_FAIL"
R3_BUDGET_EXHAUSTED = "R3_BUDGET_EXHAUSTED"
R3_KEY_TOOL_FAILED = "R3_KEY_TOOL_FAILED"
R3_EVIDENCE_CONFLICT = "R3_EVIDENCE_CONFLICT"
R3_MEASUREMENT_MISSING = "R3_MEASUREMENT_MISSING"
R3_DIMENSION_UNMEASURABLE = "R3_DIMENSION_UNMEASURABLE"
R3_POSITIVE_INSUFFICIENT = "R3_POSITIVE_INSUFFICIENT"
R4_PASS_GATE_FAIL = "R4_PASS_GATE_FAIL"
R5_DEGRADED_OR_FAILED_STEP = "R5_DEGRADED_OR_FAILED_STEP"

# 强相似阈值：镜像 image_analysis.tool 的 0.85（本地声明，与证据层同源）
_SIM_STRONG = 0.85
_SIM_PRESENT = 0.70


def _has_citable(evidence) -> bool:
    """是否存在带 ``ref_id`` 的可引用依据（POLICY_REF / CASE_PRECEDENT）。"""
    return any(e.type in CITABLE_TYPES and e.ref_id for e in evidence)


def _strong_similarity(evidence) -> bool:
    """是否存在强相似 IMAGE_SIMILARITY（``weight >= 0.85``）。"""
    return any(e.type == "IMAGE_SIMILARITY" and (e.weight or 0.0) >= _SIM_STRONG for e in evidence)


def weak_similarity(evidence) -> bool:
    """是否存在**弱相似**（0.70 <= weight < 0.85）—— 需与商品事实交叉，不能单独撑起 REJECT。"""
    return any(
        e.type == "IMAGE_SIMILARITY" and _SIM_PRESENT <= (e.weight or 0.0) < _SIM_STRONG
        for e in evidence
    )


# 谓词（全部纯函数、确定性、可单测；空/缺失安全）


def high_priority(hypotheses) -> list:
    """``prior >= HIGH_PRIOR_THRESHOLD``（0.3）的假设。

    **仅供人工队列排序/展示**；两道 Gate 都不消费本函数（终裁不读 LLM 生成的 prior）。
    """
    return [
        h
        for h in (hypotheses or [])
        if h.prior is not None and h.prior >= HIGH_PRIOR_THRESHOLD
    ]


def contradiction_detect(state) -> bool:
    """证据冲突：「相似度极高但商家历史干净」。

    True = ∃ 强相似 IMAGE_SIMILARITY（``weight >= 0.85``）∧ ∃ MERCHANT_HISTORY 且
    ``removals == 0 且 title == 0``。extra 缺失/未回填的按「不干净」处理。

    这是"证据互相矛盾"的特定形态，归入弃权与 REJECT 拦截；**不再兼作弱相似的安全阀**
    （弱相似的应然处置见 ``weak_similarity`` 与 measurements 的 required set）。
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


def key_evidence_complete(state) -> bool:
    """关键证据完整 = 无未解决的 critical Tool 失败。"""
    return not key_tool_failure(state, state.get("failures") or [])


def coverage_of(state) -> CoverageReport:
    """从 state 的事实通道构造覆盖报告（不读 hypotheses）。"""
    return coverage_report(
        state.get("case"),
        state.get("evidence") or [],
        state.get("measurement_capabilities"),
    )


def _rule_positive_dims(case) -> frozenset[str]:
    """规则侧阳性维度：平台规则命中（R-102 品牌词 / R-302 规避词）→ ``text_compliance``。

    规则命中**阻塞 PASS**，但**不单独授权 REJECT**（R-102 的既定语义是"交 Agent 上下文调查"，
    官方店/适配词/授权产品场景会被误杀）—— REJECT 仍需证据链里的硬阳性。
    """
    from pra.agent.guardrails.measurements import text_compliance_positive

    if case is None:
        return frozenset()
    return frozenset({DIM_TEXT_COMPLIANCE}) if text_compliance_positive(case) else frozenset()


def pass_gate(state) -> bool:
    """PASS Gate：五项全满足才放行（全部读事实通道）。

    ① 无维度匹配的阳性证据；② 无规则侧阳性（R-102/R-302）；③ required 维度全部覆盖
    （无 NOT_MEASURED / UNMEASURABLE）；④ 无未解决 critical Tool 失败；⑤ 无证据冲突。

    空 state 防御：``case`` 缺失时 required 为空、其余条件也空 —— 若不放行这道守卫会
    **vacuous PASS**（什么都还没查就放行），故显式返回 False。
    """
    if state.get("case") is None:
        return False
    cov = coverage_of(state)
    if cov.positive:
        return False
    if _rule_positive_dims(state.get("case")):
        return False
    if cov.missing or cov.unmeasurable:
        return False
    if not key_evidence_complete(state):
        return False
    return not contradiction_detect(state)


def reject_gate(state, dc: float) -> bool:
    """REJECT Gate：四项全满足才自动拒绝。

    ① ∃ **足以授权自动拒绝的阳性**：证据链中与本 listing 直接相关的硬阳性
    （``cov.reject_positive``：强相似/Logo；商家行为维度还需本 listing 外观信号佐证），
    **或**平台规则层命中规避词（R-302，文本自证）；
    ② ∃ 带 ``ref_id`` 的可引用依据；③ ``dc >= 0.7``；④ 无证据冲突。
    **不读 LLM 的 ``SUPPORTED`` / ``evidence_for`` / ``posterior``。**

    刻意**不**把"仅商家历史脏"当作授权：商家行为是**针对该商家**的画像，不能单独作为
    本 listing 违规的确证（reviewer 语义：疑似规避但图/文本无确证 → 克制转人工）。
    """
    cov = coverage_of(state)
    if not (cov.reject_positive or text_evasion_hit(state.get("case"))):
        return False
    if not _has_citable(state.get("evidence") or []):
        return False
    if dc < CONFIDENCE_ABSTAIN_THRESHOLD:
        return False
    return not contradiction_detect(state)


def finalize_decision_confidence(state) -> float:
    """确定性重算 ``decision_confidence``（「自动终裁的把握」，非违规概率）。

    ``coverage = |required ∩ covered| / |required|``；
    ``strength = mean(该维度决定性证据强度)``（有阳性取阳性最大、否则取阴性测量可信度）；
    ``citation = 1.0 if ∃ 带 ref_id 的可引用依据 else 0.0``；
    ``conflict = 1.0 if contradiction_detect else 0.0``；
    ``c = 0.40*coverage + 0.30*strength + 0.20*citation + 0.10 - 0.20*conflict``，clip 到 0..1。

    ``required`` 为空（无 case 等异常态）→ 退化到 0.10 基线。**不读 LLM 的 posterior。**
    """
    cov = coverage_of(state)
    return _dc_from(cov, citation=_has_citable(state.get("evidence") or []),
                    conflict=contradiction_detect(state))


def _dc_from(cov: CoverageReport, *, citation: bool, conflict: bool) -> float:
    """dc 的纯函数核心（供 overlay 复用同一次覆盖计算，避免重复求值）。

    维度强度缺省：**恒覆盖的确定性规则维度**（如 text_compliance，由纯函数即时求值、
    无命中即在构造上确定）给 1.0；其余缺省 0.0（未测维度不该贡献把握）。
    """
    if not cov.required:
        return round(max(DC_BASE - (DC_CONFLICT_PENALTY if conflict else 0.0), 0.0), 2)
    covered_req = [d for d in cov.required if d in cov.covered]
    coverage = len(covered_req) / len(cov.required)
    strengths = [
        float(cov.strength.get(d, 1.0 if d in ALWAYS_COVERED_DIMENSIONS else 0.0))
        for d in covered_req
    ]
    strength = sum(strengths) / len(strengths) if strengths else 0.0
    c = (
        DC_W_COVERAGE * coverage
        + DC_W_STRENGTH * strength
        + DC_W_CITATION * (1.0 if citation else 0.0)
        + DC_BASE
        - (DC_CONFLICT_PENALTY if conflict else 0.0)
    )
    return round(min(max(c, 0.0), 1.0), 2)


def finalize_risk_level(state, proposal) -> RiskLevel:
    """风险等级：**仅供展示与人工队列排序**，不参与 Gate。

    提案声明了 ``risk_level`` → 用之（展示）；否则按证据链中最高**阳性**权重派生
    （>=0.8 HIGH / >=0.5 MEDIUM / >=0.2 LOW / 其余 NONE）。不读假设 posterior。
    """
    if proposal is not None and proposal.risk_level:
        return RiskLevel(proposal.risk_level)
    top = max(
        (float(e.weight or 0.0) for items in coverage_of(state).positive.values() for e in items),
        default=0.0,
    )
    if top >= 0.8:
        return RiskLevel.HIGH
    if top >= 0.5:
        return RiskLevel.MEDIUM
    if top >= 0.2:
        return RiskLevel.LOW
    return RiskLevel.NONE


def finalize_risk_type(state, proposal) -> list:
    """风险类型：**仅供展示与人工队列排序**，不参与 Gate。

    提案有非空 ``risk_type`` → 用之；否则按阳性证据所在维度派生：外观 → POTENTIAL_IP_RISK、
    商家行为 → EVASION_PATTERN、文本合规 → FALSE_CLAIM。顺序固定，无命中 → ``[]``。
    """
    if proposal is not None and proposal.risk_type:
        return list(proposal.risk_type)
    positive_dims = set(coverage_of(state).positive)
    risk: list = []
    if DIM_IMAGE_APPEARANCE in positive_dims:
        risk.append(RiskType.POTENTIAL_IP_RISK)
    if DIM_MERCHANT_PROFILE in positive_dims:
        risk.append(RiskType.EVASION_PATTERN)
    if DIM_TEXT_COMPLIANCE in positive_dims:
        risk.append(RiskType.FALSE_CLAIM)
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
    """组装 ``ReviewDecision``，取值全部来自确定性参数与 state 的事实通道。

    evidence / hypothesis_trace 直接引用 state 的同型对象列表（浅拷贝容器、元素同一实例；
    后者仅供审计/展示，不参与判定）；policy 取排序去重的 POLICY_REF ``extra["policy_id"]``；
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


def abstention_codes(state, cov: CoverageReport) -> list:
    """HUMAN_REVIEW 弃权清单：顺序固定，命中码全量收集（只读事实通道）。

    只认 ``state["degraded"]`` 与未解决的 critical Tool 失败；``severity="warn"`` 的失败只进
    审计、不触发。**关键测量缺口/不可测维度**是本次新增的两类弃权（可补救 vs 环境缺失）。
    """
    codes: list = []
    budget = state.get("budget")
    if budget is not None and budget_exceeded(budget) is not None:
        codes.append(R3_BUDGET_EXHAUSTED)
    if contradiction_detect(state):
        codes.append(R3_EVIDENCE_CONFLICT)
    if key_tool_failure(state, state.get("failures") or []):
        codes.append(R3_KEY_TOOL_FAILED)
    if cov.missing:
        codes.append(R3_MEASUREMENT_MISSING)
    if cov.unmeasurable:
        codes.append(R3_DIMENSION_UNMEASURABLE)
    if state.get("degraded"):
        codes.append(R5_DEGRADED_OR_FAILED_STEP)
    return codes


def _human_review(state, proposal, dc: float, overrides: list) -> ReviewDecision:
    """转人工的统一组装：risk/risk_type 用 ``finalize_risk_*``（展示），confidence 用 dc。"""
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

    ``proposal`` 为 None（预算耗尽 / 降级 / LLM 失败）时无提案。
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

    # 2) 事实侧覆盖 + 确定性重算 decision_confidence（LLM confidence 仅参考）
    cov = coverage_of(state)
    dc = _dc_from(
        cov,
        citation=_has_citable(state.get("evidence") or []),
        conflict=contradiction_detect(state),
    )

    # 3) 弃权清单（命中码全量写入 overrides）
    overrides = abstention_codes(state, cov)

    # 4) proposal None 兜底：无弃权码时补 R5
    if proposal is None and not overrides:
        overrides.append(R5_DEGRADED_OR_FAILED_STEP)

    if overrides:
        return _human_review(state, proposal, dc, overrides)

    # 5) PASS / REJECT Gate：校验 LLM 提案（判据全部来自事实通道）
    evs = state.get("evidence") or []
    if proposal.decision == "PASS" and not pass_gate(state):
        return _human_review(state, proposal, dc, [R4_PASS_GATE_FAIL])
    if proposal.decision == "REJECT" and not reject_gate(state, dc):
        codes = [R2_REJECT_GATE_FAIL]
        # 阳性信号存在但不足以自动拒绝（仅商家画像 / 弱相似 / 无可引用依据）→ 显式归因
        if weak_similarity(evs) or cov.positive:
            codes.append(R3_POSITIVE_INSUFFICIENT)
        return _human_review(state, proposal, dc, codes)

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


__all__ = [
    "CONFIDENCE_ABSTAIN_THRESHOLD",
    "HIGH_PRIOR_THRESHOLD",
    "R1_HARD_RULE",
    "R2_REJECT_GATE_FAIL",
    "R3_BUDGET_EXHAUSTED",
    "R3_DIMENSION_UNMEASURABLE",
    "R3_EVIDENCE_CONFLICT",
    "R3_KEY_TOOL_FAILED",
    "R3_MEASUREMENT_MISSING",
    "R3_POSITIVE_INSUFFICIENT",
    "R4_PASS_GATE_FAIL",
    "R5_DEGRADED_OR_FAILED_STEP",
    "abstention_codes",
    "build_decision",
    "contradiction_detect",
    "coverage_of",
    "finalize_decision_confidence",
    "finalize_risk_level",
    "finalize_risk_type",
    "high_priority",
    "key_evidence_complete",
    "pass_gate",
    "reject_gate",
    "run_decision_overlay",
    "weak_similarity",
]
