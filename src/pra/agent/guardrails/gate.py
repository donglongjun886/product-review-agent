"""决策收口：LLM 只出提案，最终结论由本模块用**事实**确定性算出。

输入 = ``DecisionProposal`` + state 的**事实通道**（``case`` / ``evidence`` /
``measurement_capabilities`` / ``failures`` / ``budget`` / ``degraded``）。
**终裁不读任何 LLM 生成量**：不读 ``prior`` / ``posterior`` / ``Hypothesis.status`` /
``evidence_for`` / 提案里的 ``confidence``。``proposal`` 只决定"要走哪道 Gate"
（PASS 提案走 PASS Gate、REJECT 提案走 REJECT Gate）以及非判定性的展示字段。

顺序：① 硬规则命中 → REJECT/HIGH，LLM 不可覆盖；② 弃权清单（预算/**关键测量缺口**/
不可测维度/降级）任一命中 → HUMAN_REVIEW + 归因码；③ 无提案且无码 → 补 ``R5``；
④ 提案 PASS 不过 PASS Gate → ``+R4``、提案 REJECT 不过 REJECT Gate → ``+R2``；
⑤ 采纳：decision 用提案、overrides=[]。

PASS Gate（全满足才放行）：无维度匹配的**阳性证据** ∧ 无**规则侧阳性**（R-102/R-302 命中）
∧ 本案 required 维度**全部覆盖**（无 NOT_MEASURED、无 UNMEASURABLE）。

REJECT Gate（全满足才自动拒绝）：R-302 规避词命中 ∧ ∃ 带 ``ref_id`` 的可引用依据。
若无 R-302 证据，即便有商家行为画像也不自动拒绝 —— 单一来源的商家历史不得单独定案。

不变量：``risk_level`` / ``risk_type`` 不参与判定（仅供人工队列排序）；``policy`` 只从
``POLICY_REF`` 证据的 ``extra["policy_id"]`` 读，不采信提案。本模块**不含任何读 ``prior``
的出口**（连"仅供参考"的排序工具也不在此）—— 「终裁不读 LLM 生成量」由结构而非注释保证。
"""

from __future__ import annotations

from pra.agent.guardrails.budget import budget_exceeded, snapshot_budget
from pra.agent.guardrails.hard_rules import hard_rule_hit
from pra.agent.guardrails.measurements import (
    RULE_BRAND_WORD,
    RULE_EVASION_WORD,
    CoverageReport,
    coverage_report,
    rule_hit_ids,
)
from pra.domain.measurement import (
    CITABLE_TYPES,
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

# 确定性链产出的裁决一律记 1.0（与 screening 直判同口径）；本字段不参与任何判定分支。
DECISION_CONFIDENCE = 1.0

# overrides 原因码（写进 ReviewDecision.overrides）
R1_HARD_RULE = "R1_HARD_RULE"
R2_REJECT_GATE_FAIL = "R2_REJECT_GATE_FAIL"
R3_BUDGET_EXHAUSTED = "R3_BUDGET_EXHAUSTED"
R3_MEASUREMENT_MISSING = "R3_MEASUREMENT_MISSING"
R3_DIMENSION_UNMEASURABLE = "R3_DIMENSION_UNMEASURABLE"
R3_POSITIVE_INSUFFICIENT = "R3_POSITIVE_INSUFFICIENT"
R4_PASS_GATE_FAIL = "R4_PASS_GATE_FAIL"
R5_DEGRADED_OR_FAILED_STEP = "R5_DEGRADED_OR_FAILED_STEP"


def _has_citable(evidence) -> bool:
    """是否存在带 ``ref_id`` 的可引用依据（POLICY_REF / CASE_PRECEDENT）。"""
    return any(e.type in CITABLE_TYPES and e.ref_id for e in evidence)


# 谓词（全部纯函数、确定性、可单测；空/缺失安全）


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
    官方店/适配词/授权产品场景会被误杀）—— REJECT 仍需证据链里的硬阳性（R-302 例外，见
    ``reject_gate``）。
    """
    hits = rule_hit_ids(case)
    return (
        frozenset({DIM_TEXT_COMPLIANCE})
        if hits & {RULE_BRAND_WORD, RULE_EVASION_WORD}
        else frozenset()
    )


def pass_gate(state) -> bool:
    """PASS Gate：三项全满足才放行（全部读事实通道）。

    ① 无维度匹配的阳性证据；② 无规则侧阳性（R-102/R-302）；③ required 维度全部覆盖
    （无 NOT_MEASURED / UNMEASURABLE）。

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
    return not (cov.missing or cov.unmeasurable)


def reject_gate(state) -> bool:
    """REJECT Gate：两项全满足才自动拒绝。

    ① 平台规则层命中规避词（R-302，文本自证）；② ∃ 带 ``ref_id`` 的可引用依据。
    **不读 LLM 的 ``SUPPORTED`` / ``evidence_for`` / ``posterior``。**

    刻意**不**把"仅商家历史脏"当作授权：商家行为是**针对该商家**的画像，不能单独作为
    本 listing 违规的确证（reviewer 语义：疑似规避但文本无确证 → 克制转人工）。

    空 state 防御：``case`` 缺失 ⇒ 不得产出任何自动裁决（与 ``pass_gate`` 对称）。
    """
    if state.get("case") is None:
        return False
    if RULE_EVASION_WORD not in rule_hit_ids(state.get("case")):
        return False
    return _has_citable(state.get("evidence") or [])


def _finalize_risk_level(state, proposal) -> RiskLevel:
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


def _finalize_risk_type(state, proposal) -> list:
    """风险类型：**仅供展示与人工队列排序**，不参与 Gate。

    提案有非空 ``risk_type`` → 用之；否则按阳性证据所在维度派生：商家行为 →
    EVASION_PATTERN、文本合规 → FALSE_CLAIM。顺序固定，无命中 → ``[]``。
    """
    if proposal is not None and proposal.risk_type:
        return list(proposal.risk_type)
    positive_dims = set(coverage_of(state).positive)
    risk: list = []
    if DIM_MERCHANT_PROFILE in positive_dims:
        risk.append(RiskType.EVASION_PATTERN)
    if DIM_TEXT_COMPLIANCE in positive_dims:
        risk.append(RiskType.FALSE_CLAIM)
    return risk


# 组装与 overlay 收口


def _build_decision(
    state: dict,
    *,
    decision: Decision,
    risk_level: RiskLevel,
    risk_type: list,
    overrides: list,
) -> ReviewDecision:
    """组装 ``ReviewDecision``，取值全部来自确定性参数与 state 的事实通道。

    evidence / hypothesis_trace 直接引用 state 的同型对象列表（浅拷贝容器、元素同一实例；
    后者仅供审计/展示，不参与判定）；policy 取排序去重的 POLICY_REF ``extra["policy_id"]``；
    budget_used 为 ``snapshot_budget`` 快照；decision_confidence 恒为直判口径常量。
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
        decision_confidence=DECISION_CONFIDENCE,
        evidence=evidence,
        policy=policy,
        hypothesis_trace=list(state.get("hypotheses") or []),
        budget_used=snapshot_budget(budget) if budget is not None else Budget(),
        overrides=list(overrides),
    )


def abstention_codes(state, cov: CoverageReport) -> list:
    """HUMAN_REVIEW 弃权清单：顺序固定，命中码全量收集（只读事实通道）。

    只认预算超限 / 测量缺口 / 不可测维度 / ``state["degraded"]``；
    ``severity="warn"`` 的失败只进审计、不触发。
    """
    codes: list = []
    budget = state.get("budget")
    if budget is not None and budget_exceeded(budget) is not None:
        codes.append(R3_BUDGET_EXHAUSTED)
    if cov.missing:
        codes.append(R3_MEASUREMENT_MISSING)
    if cov.unmeasurable:
        codes.append(R3_DIMENSION_UNMEASURABLE)
    if state.get("degraded"):
        codes.append(R5_DEGRADED_OR_FAILED_STEP)
    return codes


def _human_review(state, proposal, overrides: list) -> ReviewDecision:
    """转人工的统一组装：risk/risk_type 用 ``_finalize_risk_*``（展示），confidence 用常量。"""
    return _build_decision(
        state,
        decision=Decision.HUMAN_REVIEW,
        risk_level=_finalize_risk_level(state, proposal),
        risk_type=_finalize_risk_type(state, proposal),
        overrides=overrides,
    )


def run_decision_overlay(state: dict, proposal) -> ReviewDecision:
    """decide 收口主函数：顺序固定，overrides 全量写入。

    ``proposal`` 为 None（预算耗尽 / 降级 / LLM 失败）时无提案。
    """
    # 1) 硬规则优先（不可被 LLM 覆盖；防漏放）
    hit = hard_rule_hit(state)
    if hit is not None:
        return _build_decision(
            state,
            decision=Decision.REJECT,
            risk_level=RiskLevel.HIGH,
            risk_type=list(hit.risk_types),
            overrides=[R1_HARD_RULE],
        )

    # 2) 事实侧覆盖 + 弃权清单（命中码全量写入 overrides）
    cov = coverage_of(state)
    overrides = abstention_codes(state, cov)

    # 3) proposal None 兜底：无弃权码时补 R5
    if proposal is None and not overrides:
        overrides.append(R5_DEGRADED_OR_FAILED_STEP)

    if overrides:
        return _human_review(state, proposal, overrides)

    # 4) PASS / REJECT Gate：校验 LLM 提案（判据全部来自事实通道）
    if proposal.decision == "PASS" and not pass_gate(state):
        return _human_review(state, proposal, [R4_PASS_GATE_FAIL])
    if proposal.decision == "REJECT" and not reject_gate(state):
        codes = [R2_REJECT_GATE_FAIL]
        # 阳性信号存在但不足以自动拒绝（无可引用依据 / 未命中规避词）→ 显式归因
        if cov.positive:
            codes.append(R3_POSITIVE_INSUFFICIENT)
        return _human_review(state, proposal, codes)

    # 5) 采纳：HUMAN 提案或 Gate 通过
    return _build_decision(
        state,
        decision=Decision(proposal.decision),
        risk_level=RiskLevel(proposal.risk_level),
        risk_type=list(proposal.risk_type),
        overrides=[],
    )


__all__ = [
    # 对外契约 = 归因码 + 判定入口；组装/派生辅助（`_` 前缀）不导出。
    "R1_HARD_RULE",
    "R2_REJECT_GATE_FAIL",
    "R3_BUDGET_EXHAUSTED",
    "R3_DIMENSION_UNMEASURABLE",
    "R3_MEASUREMENT_MISSING",
    "R3_POSITIVE_INSUFFICIENT",
    "R4_PASS_GATE_FAIL",
    "R5_DEGRADED_OR_FAILED_STEP",
    "abstention_codes",
    "coverage_of",
    "pass_gate",
    "reject_gate",
    "run_decision_overlay",
]
