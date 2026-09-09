"""确定性决策 Gate / overlay —— “LLM 只提案、Gate 确定性兜底”的落点（《00》§7.2 /《01》§7）。

decide 节点先让 LLM 产出 ``DecisionProposal``（只产**提案**），本模块
``run_decision_overlay`` 按固定顺序确定性收口（graph-mvp-contracts §5.4 / 01 §7.2，
overrides 全量写入）：

1. R1 硬规则优先（``hard_rule_hit``，不可被 LLM 覆盖、防漏放 —— 覆盖一切含 PASS 提案）；
2. ``dc = finalize_decision_confidence(state)`` 确定性重算 decision_confidence；
3. HUMAN_REVIEW abstention 清单（R3_BUDGET_EXHAUSTED / R3_CRITICAL_CONFLICT /
   R3_KEY_TOOL_FAILED / R3_POLICY_UNCERTAIN / R3_HYPOTHESES_INDISTINGUISHABLE /
   R3_VISUAL_CLAIM_UNSUPPORTED / R5_DEGRADED_OR_FAILED_STEP）—— 任一命中即
   HUMAN_REVIEW（R3_VISUAL_CLAIM_UNSUPPORTED = 外观/视觉声称被判 SUPPORTED 但证据
   链无视觉证据 → 声称维度与证据维度错配，见 ``visual_claim_unsupported``，docs/05）；
4. proposal 为 None（预算/降级/decide 自身 LLM 失败）的兜底 HUMAN_REVIEW；
5. PASS/REJECT Gate 校验 LLM 提案（不满足 → HUMAN_REVIEW + R4_PASS_GATE_FAIL /
   R2_REJECT_GATE_FAIL）；
6. 采纳：HUMAN 提案或 Gate 通过 → build_decision（risk 用 proposal、
   decision_confidence 用确定性 dc、overrides=[]）。

任一改判/归因码都写进 ``ReviewDecision.overrides``（03-decisions T-8）—— 是“谁把
PASS 改成了 HUMAN_REVIEW”的可审计落点；overrides 为空 = overlay 未改判（LLM 提案即终值）。

``decision_confidence``（O-7 已由 confidence 改名）由 ``finalize_decision_confidence``
**确定性重算**（01 §7.5 / T-4）：LLM 提案的 ``confidence`` 只作参考、不作终值；
``risk_level`` / ``risk_type`` 与 Gate 判定解耦（T-10 /《00》§7.5：仅展示与人工队列
排序，不参与路由与 Gate 判定）。``policy`` 字段从 POLICY_REF 证据的
``extra["policy_id"]`` 确定性提取（O-8 由 tools_node 回填；本模块只读取，extra 缺失时
跳过），不信任 LLM 提案的 policy。

本模块全部谓词为纯函数、确定性、可单测；state 里 hypotheses/evidence 缺失或为空时
安全返回（False/[] 等，不抛 KeyError —— ``AgentState`` 是 total=False 的 TypedDict）。
state 元素为 ``pra.domain.models`` 的 Pydantic 实例（Evidence/Hypothesis），直接属性
访问；确定性逻辑只读 extra/weight/ref_id，不解析人读 value 字符串（O-8）。唯一
例外（docs/05 V-1 拍板放行）：``visual_claim_unsupported`` 读假设 ``statement`` 自由
文本做**维度归类**（外观/视觉），不做数值/结论解析 —— 命中只影响 abstention 方向
（HUMAN_REVIEW），不据此直接 REJECT。常量在
本模块本地声明（CITABLE_TYPES 与 converge 同义，本地重声明避免模块间隐式耦合）。

走查锚点（契约 §2.3/§5.2）：5 条证据 + H2 SUPPORTED posterior=0.91 时
``finalize_decision_confidence`` 返回 0.87（0.45*0.91 + 0.25*5/8 + 0.20*1 + 0.10，
round 2，无矛盾扣分），≥ CONFIDENCE_ABSTAIN_THRESHOLD(0.7) —— REJECT Gate 安全门槛。
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

# ---------------------------------------------------------------------------
# 常量（graph-mvp-contracts §5.1；本地声明，不与其它模块共享可变状态）
# ---------------------------------------------------------------------------

HIGH_PRIOR_THRESHOLD = 0.3  # T-1：仅 PASS/REJECT Gate 的“高优先”口径（prior>=0.3）
CONFIDENCE_ABSTAIN_THRESHOLD = 0.7  # T-4：REJECT Gate 安全门槛（decision_confidence）
MAX_EXPECTED_EVIDENCE = 8  # T-4：dc 公式完整性分母
CITABLE_TYPES = {"CASE_PRECEDENT", "POLICY_REF"}  # 与 converge 同义（本地重声明）

# overrides 原因码（03-decisions T-8；写进 ReviewDecision.overrides）
R1_HARD_RULE = "R1_HARD_RULE"
R2_REJECT_GATE_FAIL = "R2_REJECT_GATE_FAIL"
R3_BUDGET_EXHAUSTED = "R3_BUDGET_EXHAUSTED"
R3_CRITICAL_CONFLICT = "R3_CRITICAL_CONFLICT"
R3_KEY_TOOL_FAILED = "R3_KEY_TOOL_FAILED"
R3_POLICY_UNCERTAIN = "R3_POLICY_UNCERTAIN"
R3_HYPOTHESES_INDISTINGUISHABLE = "R3_HYPOTHESES_INDISTINGUISHABLE"
R3_VISUAL_CLAIM_UNSUPPORTED = "R3_VISUAL_CLAIM_UNSUPPORTED"  # 视觉声称无视觉证据（docs/05 V-7 甲案）
R4_PASS_GATE_FAIL = "R4_PASS_GATE_FAIL"
R5_DEGRADED_OR_FAILED_STEP = "R5_DEGRADED_OR_FAILED_STEP"

# 强相似阈值（镜像 pra.tools.image_analysis.tool.EVIDENCE_STRONG=0.85，T-11）。
# gate 不 import tools 包（metrics 的 import 约束是 gate/budget/errors/hard_rules；
# gate 自身也保持与工具层解耦 —— 只依赖数值口径），故本地私有声明。
_SIM_STRONG = 0.85

# 视觉证据存在性阈值（docs/05 §2.4 / V-3 拍板：存在性用 0.70 普通档 —— 有任一普通
# 视觉证据即算"存在"，不代表最终相似结论，0.85 强档语义不动）。镜像
# pra.tools.image_analysis.tool.EVIDENCE_MIN_SIM=0.70；本地私有声明（同 _SIM_STRONG）。
_SIM_PRESENT = 0.70

# ---------------------------------------------------------------------------
# 外观/视觉维度声称关键词表（docs/05 §2.2-B / V-1 拍板：B 关键词谓词为主判据、
# C 分层预留二期结构化维度）。
# 来源与同步维护责任（V-5）：与 llm_prompts.py reevaluate prompt 第 8 条的外观
# 表述清单（「外观相似」「高度相似」「同款外观」「视觉仿冒」「复刻外观」「版型
# 一致」「长得像」等）同源 —— prompt 负责减少命中率、Gate 兜底为准入边界，
# **两处措辞须同步维护**（改 llm_prompts 例句时应同步本表）。
# 外延只覆盖「外观/造型/相似」语义，不含「字样/标题/复刻字样」类文本声称
# （V-2：OCR_TEXT 可支撑的文本声称不在本约束拦截面）。
# 关键词命中仅做"维度归类"，不做数值/结论解析（V-1）；归类过宽只把案型多转人工
# （安全侧），过窄退回现状 —— 宁宽勿窄（V-8 误伤容忍度：只作用于 SUPPORTED
# 高优先假设，PASS 侧天然不受影响，见 visual_claim_unsupported docstring）。
# ---------------------------------------------------------------------------
VISUAL_CLAIM_MARKERS: tuple[str, ...] = (
    # llm_prompts reevaluate 例句逐字镜像（V-5 同步维护）
    "外观相似",
    "高度相似",
    "同款外观",
    "视觉仿冒",
    "复刻外观",
    "版型一致",
    "长得像",
    # 外观/造型维度词（docs/05 §2.2-B：外观/造型/鞋型/版型/廓形/配色/图案/印花）
    "外观",
    "造型",
    "鞋型",
    "版型",
    "廓形",
    "配色",
    "图案",
    "印花",
    # 含模仿/雷同语义的断言词（覆盖措辞漂移；安全侧从宽，V-8）
    "同款",
    "仿冒",
    "复刻",
    "模仿",
)


def _has_citable(evidence) -> bool:
    """是否存在带 ref_id 的可引用依据（POLICY_REF / CASE_PRECEDENT，证据级判断）。"""
    return any(e.type in CITABLE_TYPES and e.ref_id for e in evidence)


def _strong_similarity(evidence) -> bool:
    """是否存在强相似 IMAGE_SIMILARITY（weight>=0.85，T-11 强档）。"""
    return any(e.type == "IMAGE_SIMILARITY" and (e.weight or 0.0) >= _SIM_STRONG for e in evidence)


def _top_supported_posterior(state) -> float:
    """最高 SUPPORTED 假设后验（None 按 0 处理，models.py 口径）；无则 0.0。"""
    return max(
        ((h.posterior if h.posterior is not None else 0.0)
         for h in (state.get("hypotheses") or [])
         if h.status == HypothesisStatus.SUPPORTED),
        default=0.0,
    )


# ---------------------------------------------------------------------------
# 谓词（§5.2：全部纯函数、确定性、可单测；空/缺失安全）
# ---------------------------------------------------------------------------


def high_priority(hypotheses) -> list:
    """prior>=HIGH_PRIOR_THRESHOLD(0.3) 的假设（T-1：“高优先”口径）。

    只用于 PASS/REJECT Gate 的过滤口径，**不用于收敛判定**（converge.is_converged
    不过滤 prior）；prior 为 None 按 0 处理 → 不属高优先。空/缺失返回 []。
    """
    return [
        h
        for h in (hypotheses or [])
        if h.prior is not None and h.prior >= HIGH_PRIOR_THRESHOLD
    ]


def contradiction_detect(state) -> bool:
    """关键证据矛盾检测（03-decisions T-4(e) v1 启发式，01 §7.5 conflict 项）。

    “相似度极高但商家历史干净”→ 关键矛盾 True：∃ IMAGE_SIMILARITY 强相似
    （weight>=0.85）∧ ∃ MERCHANT_HISTORY 且干净（extra.removals==0 and
    extra.title==0）。extra 缺失/未回填的 MERCHANT_HISTORY 按“不干净”处理
    （确定性只读回填后的 extra；不干净 → 不触发矛盾 → 不误转人工）。
    走查商家 5 removals 不干净 → False。空 evidence → False。
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
    """政策无法确定（只约束需政策支撑的 REJECT/风险案，不拦 PASS 候选）。

    = 存在 SUPPORTED 的 high_priority 假设（拟 REJECT 案，REJECT Gate ③ 要求可引用
    政策依据）∧ 证据链无任何**带 ref_id** 的 POLICY_REF → True。

    只认 POLICY_REF（政策条款证据）；CASE_PRECEDENT 不替代政策。无 SUPPORTED
    高优先假设（干净案，PASS 候选）→ False —— 00 §7.2 PASS Gate 不需要政策引用，
    01 §7.5 "PASS 由 pass_gate 判定、干净商品低风险置信是正常态"；若这里也判 True，
    abstention 清单会在 PASS/REJECT Gate 之前把所有无政策证据的案型一律转人工，
    pass_gate 将不可达（code review R-A P2-① 拍板：政策缺失只拦 REJECT 侧）。
    确定性逻辑读 ref_id，不解析 value 字符串。
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
    """多假设不可区分（MVP 启发式）：≥2 个 high_priority SUPPORTED 假设，
    evidence_for 非空且**集合相等**（互斥假设被同一组证据支持，无法区分）。

    其余情况 False（02 校准，勿过度收紧）；hypotheses 空/缺失 → False。
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
    """statement 是否落在「外观/视觉维度」（docs/05 §2.2-B 关键词谓词）。

    纯维度归类：命中任一 ``VISUAL_CLAIM_MARKERS`` 子串即 True；不解析数值/结论
    （V-1：statement 自由文本解析仅限维度判定，不做强度判定）。None/空 → False。
    """
    text = statement or ""
    return any(marker in text for marker in VISUAL_CLAIM_MARKERS)


def visual_evidence_present(evidence) -> bool:
    """证据链是否存在「视觉相似证据」（docs/05 §2.3 权威清单 / V-2/V-3/V-9 口径）。

    - 任一 ``IMAGE_LOGO`` → True（logo 类：检出即视觉证据，单独满足存在性，V-9，
      不等于直接证明侵权）；
    - 任一 ``IMAGE_SIMILARITY`` 且 weight>=_SIM_PRESENT(0.70) → True（V-3 存在性
      阈值 = Evidence Presence Threshold：普通档即算"存在视觉证据"，不代表最终
      相似结论，0.85 强档语义不动）；weight<0.70 **不算**存在 —— 只读 weight、
      不依赖 extra.strong（防御注入：与 quality_filter 的 EVIDENCE_MIN_SIM=0.70
      同口径，见 docs/05 §4.3 测试点 3）。
    - OCR_TEXT / POLICY_REF / CASE_PRECEDENT / PRODUCT_FACT / MERCHANT_HISTORY
      一律不算（V-2 / §2.3：OCR 是图像来源**文本**证据，证明不了鞋型/配色/版型
      相似；先例/政策/事实/商家史均替代不了视觉直接测量）。
    空/缺失 → False。
    """
    evs = evidence or []
    if any(e.type == "IMAGE_LOGO" for e in evs):
        return True
    return any(
        e.type == "IMAGE_SIMILARITY" and (e.weight or 0.0) >= _SIM_PRESENT
        for e in evs
    )


def visual_claim_unsupported(state) -> bool:
    """视觉声称无视觉证据（docs/05 §3.1 伪码语义 / V-1~V-11 拍板）—— 确定性兜底。

    = 存在 高优先(prior>=HIGH_PRIOR_THRESHOLD 0.3) ∧ status=SUPPORTED ∧ statement
      落外观/视觉维度关键词 的假设，且证据链无任何视觉证据 → True。

    EC_0007 归因（docs/05 §1.1）：先例只证明「同类曾被拒」的历史事实，替代不了
    「本商品与某品牌款外观相似」的直接测量 —— 决策 6 / V-6：**先例只能佐证相同
    证据维度，不能把历史案例事实迁移为当前案件事实**；该假设被判 SUPPORTED 属
    「声称维度与证据维度错配」，应转人工而非自动 REJECT。

    只挑 SUPPORTED 高优先假设 → PASS 案（全 REFUTED）与低优先假设永不命中
    （V-10 / §3.3：只约束 REJECT 风险侧、不扩张 PASS 侧、减少误伤）。
    命中只导向 abstention（overlay 步骤 3 → HUMAN_REVIEW，码与既有 R3 并列全量
    收集），**不直接 REJECT**（V-7 甲案）；纯函数只读 state，不改写
    hypotheses/evidence（审计双视角 = hypothesis_trace 保留 LLM 原判 + overrides
    记拦截原因）。hypotheses/evidence 空/缺失 → False（纯函数空安全）。
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
    """关键证据完整 = 无未解决的关键（critical）Tool 失败（O-2：errors 口径）。"""
    return not key_tool_failure(state, state.get("failures") or [])


def evidence_sufficient(state) -> bool:
    """证据充分：每个 SUPPORTED high_priority 假设 evidence_for 非空
    ∧ not key_tool_failure（关键证据缺失 → 不充分）。无 SUPPORTED 高优先假设时
    all() 为空真 —— 充分性由 reject_gate 的“∃ SUPPORTED”前提兜底。"""
    if key_tool_failure(state, state.get("failures") or []):
        return False
    supported = [
        h
        for h in high_priority(state.get("hypotheses") or [])
        if h.status == HypothesisStatus.SUPPORTED
    ]
    return all(bool(h.evidence_for) for h in supported)


def pass_gate(state) -> bool:
    """PASS Gate（《00》§7.2-4 / 01 §7.2）：高优先假设非空 且 全部被充分证伪。

    ① high_priority 非空（空 hypotheses 防御：不误 PASS）；
    ② 全部 REFUTED 且各自 evidence_against 非空（“有反驳证据”而非“没查到”）；
    ③ key_evidence_complete（无未解决 critical Tool 失败）；
    ④ not contradiction_detect（无未解决关键矛盾）。
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
    """REJECT Gate（《00》§7.2-2 / 01 §7.2）：自动拒绝的安全门槛，全部满足才放行。

    ① ∃ high_priority SUPPORTED（高风险假设成立）；
    ② evidence_sufficient（SUPPORTED 高优先假设都有支持证据且无关键缺失）；
    ③ ∃ citable —— POLICY_REF / CASE_PRECEDENT 带 ref_id（明确政策依据/先例）；
    ④ dc >= CONFIDENCE_ABSTAIN_THRESHOLD(0.7)（decision_confidence 安全门槛，T-4；
       **只约束自动 REJECT 侧**，PASS 不因低 dc 转人工）；
    ⑤ not contradiction_detect（无关键矛盾）。
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
    """确定性重算 decision_confidence（01 §7.5 / T-4；O-7 字段名已由 confidence 改名）。

    公式（“自动判安全把握”，非违规概率）：
        top         = max(SUPPORTED 假设 posterior, 缺省 0.0)
        completeness= min(len(evidence) / MAX_EXPECTED_EVIDENCE, 1.0)
        citation    = 1.0 if ∃ citable(POLICY_REF|CASE_PRECEDENT 带 ref_id) else 0.0
        conflict    = 0.2 if contradiction_detect else 0.0
        c = 0.45*top + 0.25*completeness + 0.20*citation + 0.10 - conflict
    返回 round(clip(c, 0..1), 2)。

    走查锚点：5 条证据 + H2 SUPPORTED posterior=0.91 →
    0.45*0.91 + 0.25*(5/8) + 0.20*1 + 0.10 = 0.86575 → round2 = **0.87**（≥0.7）。
    hypotheses/evidence 空/缺失 → top=0、completeness=0、citation=0 → 0.10 基线。
    """
    evs = state.get("evidence") or []
    top = _top_supported_posterior(state)
    completeness = min(len(evs) / MAX_EXPECTED_EVIDENCE, 1.0)
    citation = 1.0 if _has_citable(evs) else 0.0
    conflict = 0.2 if contradiction_detect(state) else 0.0
    c = 0.45 * top + 0.25 * completeness + 0.20 * citation + 0.10 - conflict
    return round(min(max(c, 0.0), 1.0), 2)


def finalize_risk_level(state, proposal) -> RiskLevel:
    """风险等级：proposal 声明了 risk_level → 用之；否则按最高 SUPPORTED posterior 派生。

    派生阈值：>=0.8 HIGH / >=0.5 MEDIUM / >=0.2 LOW / 其余 NONE。
    仅用于展示/人工队列排序，**不参与 Gate 判定**（T-10）。pydantic str 枚举按值
    转换，故此处显式转 RiskLevel 以返回枚举实例。
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
    """风险类型：proposal 有 risk_type（非空）→ 用之；否则按证据派生：

    - ∃ 强 IMAGE_SIMILARITY（weight>=0.85）→ POTENTIAL_IP_RISK；
    - ∃ MERCHANT_HISTORY 且 extra.removals>=3 或 extra.title>=3 → EVASION_PATTERN
      （走查 5 removals / 3 title-relisting 命中）；extra 缺失按 0 处理。
    派生顺序固定（IP 先、EVASION 后）；无命中 → []（如 PASS 案）。元素为 RiskType 枚举。
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


# ---------------------------------------------------------------------------
# 组装与 overlay 收口（§5.3 / §5.4）
# ---------------------------------------------------------------------------


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
    """组装 ReviewDecision（契约 §5.3）—— 全部取值来自确定性参数与 state：

    - evidence / hypothesis_trace：直接引用 state 同型对象列表（浅拷贝容器、
      元素同一实例），不派生不截断；
    - policy：排序去重的 POLICY_REF 证据 ``extra["policy_id"]``（O-8 由 tools_node
      回填，这里只读取；extra 缺失或未回填 policy_id 时跳过该条）；
    - budget_used：``snapshot_budget(state["budget"])`` 快照（补 latency_ms 的副本）。

    ``proposal`` 形参仅按接口保留（overlay 各分支已把它解析成显式判定参数传入），
    本函数体不依赖它。decision/risk_level 给字符串或枚举皆可（pydantic str 枚举按值
    转换）；risk_type 元素为 RiskType 枚举。字段名 ``decision_confidence``（非
    confidence，O-7）。
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
    """HUMAN_REVIEW abstention 清单（overlay 步骤 3；顺序固定，命中码全量收集）。

    O-2 口径：failures 非空不再一律转人工 —— R5 仅由 LLM 步失败（state["degraded"]，
    severity=critical）触发；未解决的 critical Tool 失败已由 key_tool_failure 以
    R3_KEY_TOOL_FAILED 计；severity="warn" 的失败只进审计，不触发。
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
        # docs/05 V-7 甲案：视觉声称 SUPPORTED 但无视觉证据 → 声称维度错配，转人工
        codes.append(R3_VISUAL_CLAIM_UNSUPPORTED)
    if state.get("degraded"):
        codes.append(R5_DEGRADED_OR_FAILED_STEP)
    return codes


def _human_review(state, proposal, dc: float, overrides: list) -> ReviewDecision:
    """转人工的统一组装：risk/risk_type 用 finalize_risk_*（proposal 声明了则用之，
    否则按最高 SUPPORTED posterior / 证据派生），decision_confidence 用确定性 dc。"""
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
    """decide 收口主函数（契约 §5.4）—— 顺序固定，overrides 全量写入。

    ``proposal: DecisionProposal | None``；None = 预算耗尽/降级/decide 自身 LLM
    失败（decide_node 在 can_llm=False 时不调 LLM）时的空提案。执行顺序：

    1. R1 硬规则命中 → 直接 REJECT/HIGH/1.0/overrides=[R1_HARD_RULE]（覆盖一切）；
    2. dc = finalize_decision_confidence(state)（确定性重算值）；
    3. abstention 清单（``_abstention_codes``）—— 任一命中即 HUMAN_REVIEW
       （risk/risk_type 用 finalize_risk_*：proposal 或派生）；
    4. proposal None 兜底：overrides 已有对应码则用之，无则补
       R5_DEGRADED_OR_FAILED_STEP，同样 HUMAN_REVIEW；
    5. 校验提案：PASS 且 not pass_gate → HUMAN_REVIEW + R4_PASS_GATE_FAIL；
       REJECT 且 not reject_gate(state, dc) → HUMAN_REVIEW + R2_REJECT_GATE_FAIL；
    6. 采纳：HUMAN 提案或 Gate 通过 → build_decision(decision=Decision(proposal.
       decision)，risk 用 proposal，decision_confidence=**dc**，overrides=[])。

    空 hypotheses 防御：high_priority 为空 → pass_gate=False → PASS 提案被降
    HUMAN；REJECT 提案亦无法过 Gate ① → HUMAN（不会空态误 PASS / 误 REJECT）。
    """
    # 1) R1 硬规则优先（不可被 LLM 覆盖；防漏放，覆盖一切含 PASS 提案）
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

    # 2) 确定性重算 decision_confidence（01 §7.5；LLM confidence 仅参考）
    dc = finalize_decision_confidence(state)

    # 3) abstention 清单（命中码全量写入 overrides）
    overrides = _abstention_codes(state)

    # 4) proposal None 兜底：无对应 abstention 码时补 R5
    if proposal is None and not overrides:
        overrides.append(R5_DEGRADED_OR_FAILED_STEP)

    if overrides:
        return _human_review(state, proposal, dc, overrides)

    # 5) PASS / REJECT Gate：校验 LLM 提案（此时 proposal 必非 None）
    if proposal.decision == "PASS" and not pass_gate(state):
        return _human_review(state, proposal, dc, [R4_PASS_GATE_FAIL])
    if proposal.decision == "REJECT" and not reject_gate(state, dc):
        return _human_review(state, proposal, dc, [R2_REJECT_GATE_FAIL])

    # 6) 采纳：HUMAN 提案或 Gate 通过 → risk 用 proposal、dc 为确定性重算值
    return build_decision(
        state,
        proposal,
        decision=Decision(proposal.decision),
        risk_level=RiskLevel(proposal.risk_level),
        risk_type=list(proposal.risk_type),
        decision_confidence=dc,
        overrides=[],
    )
