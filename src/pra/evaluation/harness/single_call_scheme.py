"""Baseline 2：单次 LLM 调用（确定性 mock，可注入 llm_fn）。

一次调用 = 基础输入全量（商品快照 + 机审 OCR 文本 + screening_signals）→ 结构化决策
JSON。**不给商家历史 / 案例库 / 政策库** —— 那是 Agent 经工具调查才拿到的证据，给了
等于作弊。输出是 ReviewDecision 形状子集（decision / risk_level / risk_type /
decision_confidence / policy）；``budget_used / overrides / hypothesis_trace`` 不适用。

默认 ``DefaultSingleCallMock`` 只扫基础输入**表面字段**，不读 expected、不读图片
相似度 / 商家历史 / 在库事实；需要这些证据的案会判错或转人工。表面启发式：
1. 标题/描述明示仿冒词（复刻/高仿/1:1/同款/原单）→ REJECT（conf 0.88，文本自证）；
2. 仅图片 OCR 含仿冒词 → REJECT 候选 conf 0.60（证据弱，后处理转 HUMAN）；
3. 含知名品牌词但无仿冒词 → HUMAN（授权/真伪单次调用无法核验）；
4. brand 或 category 空缺 → HUMAN（关键事实缺失，与 R-301 同口径）；
5. 机审信号非 PASS → HUMAN；6. 否则 → PASS（conf 0.78）。
后处理：``decision_confidence < ctx.abstain_confidence_threshold``（默认 0.7）的
REJECT 候选确定性改记 HUMAN_REVIEW。

2b 变体（RAG-in-prompt）：``SingleCallScheme(extra_context=[…])`` 把政策/先例**文本
摘要**预塞进 prompt（仍不给工具）；None 时与 2a 行为完全一致。预塞的只能是评测世界里
可查的事实文本，不得含 expected 答案；注入键 ``extra_context`` 由 mock 显式读取，
不进 EvalRecord 证据链。

**泄漏边界**：``run`` 把整份 case 输入（含图片 url 字面）喂给 llm_fn；当前 eval 数据
的 url 含语义段（``eval/viol_*`` / ``logo_*``），现有 mock 不读 url → 无实际影响；
**换真实 LLM 前必须先改中性 URL**，否则模型会从 url 读到类别信号（等同注入 GT）。
"""

from __future__ import annotations

import re
from collections.abc import Callable

from pra.evaluation.dataset.schema import EvalCase
from pra.evaluation.harness.base import EvalContext, EvalRecord, SchemeRunner
from pra.screening.rule_engine.terms import BRAND_TERMS, EVASION_TERMS

# mock 名称常量（审计/报告标识）
DEFAULT_MOCK_NAME = "single-call-mock-v1"
RAG_MOCK_NAME = "single-call-rag-mock-v1"

__all__ = [
    "ContextAwareSingleCallMock",
    "DefaultSingleCallMock",
    "SingleCallScheme",
    "apply_abstain_threshold",
    "rag_decision_from_context",
]


# 文本词命中辅助（确定性；与 screening rules 的 _term_hits 同口径但本地实现，
# 避免依赖 screening 私有函数 —— 词表仍复用 terms 单一来源）

_ASCII_LATIN = re.compile(r"[a-z ]+\Z")


def _fold(text: str) -> str:
    return (text or "").lower().replace("：", ":")


def _scan_hits(text: str, terms: frozenset[str]) -> list[str]:
    """折叠文本中命中 terms 的词（纯拉丁按词边界；中文/符号词子串）—— 排序保序。"""
    folded = _fold(text)
    matched: list[str] = []
    for term in terms:
        tf = _fold(term)
        if _ASCII_LATIN.fullmatch(tf):
            pat = re.compile(rf"(?<![a-z0-9_]){re.escape(tf)}(?![a-z0-9_])")
        else:
            pat = re.compile(re.escape(tf))
        if pat.search(folded):
            matched.append(term)
    return sorted(matched)


# 默认 mock：确定性表面审查员


class DefaultSingleCallMock:
    """默认单次 LLM mock —— 纯函数式表面启发式（确定性、可重放）。

    ``__call__(case_json: dict) -> dict``：入参为 ``ProductReviewCase.model_dump(
    mode="json")`` 形状；返回 ReviewDecision 子集 dict（policy 恒 []，无政策库访问；
    signals 为逐项表面信号，供审计）。
    """

    name = DEFAULT_MOCK_NAME

    def __call__(self, case_json: dict) -> dict:
        return _surface_predict(case_json)


def _surface_text(case_json: dict) -> tuple[str, list[str], list[str], list[str]]:
    """抽取基础输入表面文本：折叠标题+描述、OCR 文本、各自词命中（顺序固定）。"""
    product = case_json.get("product") or {}
    title = str(product.get("title") or "")
    desc = str(product.get("description") or "")
    text = _fold(f"{title}\n{desc}")
    ocr_parts: list[str] = []
    for img in product.get("images") or []:
        ocr = (img or {}).get("ocr_text")
        if isinstance(ocr, str) and ocr.strip():
            ocr_parts.append(ocr)
    return text, ocr_parts, _scan_hits(text, EVASION_TERMS), _scan_hits(text, BRAND_TERMS)


def _surface_predict(case_json: dict) -> dict:
    """确定性表面启发式（规则见模块 docstring；signals 供审计与报告）。

    :param case_json: ProductReviewCase 的 JSON 形状（model_dump(mode="json")）。
    """
    product = case_json.get("product") or {}
    brand = product.get("brand")
    category = product.get("category")
    brand_missing = brand is None or (isinstance(brand, str) and not brand.strip())
    cat_missing = not str(category or "").strip()

    _text, ocr_parts, text_evasion, text_brand = _surface_text(case_json)
    ocr_hay = _fold("\n".join(ocr_parts))
    ocr_evasion = _scan_hits(ocr_hay, EVASION_TERMS)
    ocr_brand = _scan_hits(ocr_hay, BRAND_TERMS)
    signals: dict = {
        "text_evasion": text_evasion,
        "text_brand": text_brand,
        "ocr_evasion": ocr_evasion,
        "ocr_brand": ocr_brand,
        "brand_missing": brand_missing,
        "category_missing": cat_missing,
    }

    # 机审信号（screening_signals）是否有非 PASS 项
    flagged = [
        s.get("name")
        for s in (case_json.get("screening_signals") or [])
        if (s or {}).get("result") != "PASS"
    ]
    signals["signal_flagged"] = flagged

    def _out(
        decision: str,
        confidence: float,
        risk_level: str,
        risk_type: list,
        rationale: str,
        reasons: list,
    ) -> dict:
        return {
            "decision": decision,
            "risk_level": risk_level,
            "risk_type": risk_type,
            "confidence": confidence,
            "policy": [],  # 单次调用无政策库访问权限（§12.2：不给案例/政策库）
            "rationale": rationale,
            "signals": signals,
            "reasons": reasons,  # 表面观察理由（人读，进 detail/evidence 审计）
        }

    if text_evasion:  # ① 标题/描述明示仿冒 —— 文本自证，直接 REJECT
        return _out(
            "REJECT", 0.88, "HIGH", ["POTENTIAL_IP_RISK"],
            "标题/描述明示仿冒词，单次调用即可确定性判定违规。",
            [f"文本含仿冒词: {', '.join(text_evasion)}（明示仿冒，无需外部证据）"],
        )
    if ocr_evasion:  # ② 仅 OCR 文本含仿冒词 —— 弱证据 REJECT 候选（后处理转 HUMAN）
        return _out(
            "REJECT", 0.60, "MEDIUM", ["POTENTIAL_IP_RISK"],
            "仅图片 OCR 文本命中仿冒词，但单次调用无法与商品/商家事实交叉核验，置信不足。",
            [f"OCR 含仿冒词: {', '.join(ocr_evasion)}（证据弱于标题自证）"],
        )
    if text_brand or ocr_brand:  # ③ 品牌词但无仿冒词 —— 授权/真伪无法核验
        hits = sorted(set(text_brand) | set(ocr_brand))
        return _out(
            "HUMAN_REVIEW", 0.50, "MEDIUM", ["POTENTIAL_IP_RISK"],
            "含知名品牌词，但单次调用看不到商家授权/在库事实，无法核验真伪，转人工。",
            [f"品牌词命中: {', '.join(hits)}（需调查授权信息，本方案无调查能力）"],
        )
    if brand_missing or cat_missing:  # ④ 关键事实缺失（R-301 同口径）
        missing = []
        if brand_missing:
            missing.append("brand")
        if cat_missing:
            missing.append("category")
        return _out(
            "HUMAN_REVIEW", 0.35, "LOW", [],
            "品牌或类目关键字段空缺，单次调用无法判断是否刻意规避，转人工。",
            [f"关键字段空缺: {', '.join(missing)}（需在库/商家信息核验）"],
        )
    if flagged:  # ⑤ 机审已标非 PASS —— 已有确定性风险信号
        return _out(
            "HUMAN_REVIEW", 0.45, "MEDIUM", [],
            "机审信号已非 PASS，单次调用难以复核，转人工。",
            [f"机审信号非 PASS: {', '.join(flagged)}"],
        )
    # ⑥ 表面干净 → PASS（单次调用视角可放行）
    return _out(
        "PASS", 0.78, "NONE", [],
        "标题/描述/品牌/类目表面干净，无已知风险信号，单次调用视角放行。",
        ["表面字段干净（无仿冒词/品牌词/字段空缺/风险信号）"],
    )


# 2b 变体：RAG-in-prompt（政策/先例文本预塞）
#
# 确定性触发器（规则即文档，仅供 scripted 近似 —— 真实 LLM 的"读了检索文本所以敢
# 下结论"在此用显式规则模拟，语义诚实、可单测、无真 LLM）：
# 基础 raw 决策为 **REJECT 候选但置信 < 0.7**（本会确定性转人工，即规则 ② 的 OCR 弱
# 证据案）时，若注入的 extra_context 中存在**判例行**满足：
#   1) 以 ``判例:`` 开头；2) 含结论标记 ``→ REJECT``（判例是自动拒绝，非转人工）；
#   3) 类目作用域（形如 ``类目[<scope>]``，可省）与案件类目一致；
#   4) 行内含案件**已观测的表面信号词**（OCR/标题仿冒词或品牌词）；
# 则把该 REJECT 候选的置信抬到 0.85（> abstain 门槛 → 不再转人工），理由标注命中行。
# 语义：RAG 让"弱怀疑 + 同型先例"收敛为可自动拒绝；无命中行或先例是转人工则维持原样
# —— 预塞文本不得让 mock 凭空造证据。判定行由 ablation.build_rag_context 从评测世界
# 静态文本（EVAL_PRECEDENTS / EVAL_POLICY_CLAUSES）生成，只含事实、不含 expected。

_CTX_LINE_PREFIX = "判例:"
_CTX_REJECT_MARK = "→ REJECT"
_CTX_SCOPE_RE = re.compile(r"类目\[([^\]]+)\]")
_CONF_ABSTAIN_DEFAULT = 0.7  # 与 EvalContext.abstain_confidence_threshold 默认同值

RAG_UPGRADE_CONFIDENCE = 0.85  # 抬到 > 0.7 → 不再被确定性后处理转人工


def _observed_surface_terms(signals: dict) -> list[str]:
    """把 mock 表面信号里的命中词收敛成排序去重列表（RAG 判例行的匹配词源）。"""
    terms: set[str] = set()
    for key in ("text_evasion", "ocr_evasion", "text_brand", "ocr_brand"):
        for t in signals.get(key) or []:
            if isinstance(t, str) and t:
                terms.add(t)
    return sorted(terms)


def _line_scope_matches(line: str, category: str | None) -> bool:
    """判例行作用域 ``类目[...]`` 与案件类目一致；行无作用域声明则放行。"""
    m = _CTX_SCOPE_RE.search(line)
    if m is None:
        return True
    scope = m.group(1).strip()
    return bool(category) and _fold(scope) == _fold(category)


def rag_decision_from_context(
    base_raw: dict, context_lines: list[str], case_json: dict
) -> dict | None:
    """R-2b 确定性触发器：命中返回升级后的 raw dict，未命中返回 None。

    :param base_raw: ``_surface_predict`` 的输出（含 decision/confidence/signals）。
    :param context_lines: 预塞的判例/政策文本行（extra_context）。
    :param case_json: 案件 JSON（读 product.category 做作用域匹配）。
    """
    if base_raw.get("decision") != "REJECT":
        return None  # 只处理"弱怀疑 REJECT 候选"（规则见模块 docstring）
    if (base_raw.get("confidence") or 0.0) >= _CONF_ABSTAIN_DEFAULT:
        return None  # 本就不转人工 → RAG 无增量
    if not context_lines:
        return None
    signals = base_raw.get("signals") or {}
    observed = _observed_surface_terms(signals)
    if not observed:
        return None  # 无已观测表面词 → 无法与判例特征对位
    product = case_json.get("product") or {}
    category = (product.get("category") or "") or None

    matched_line: str | None = None
    for line in context_lines:
        text = str(line)
        if not text.startswith(_CTX_LINE_PREFIX) or _CTX_REJECT_MARK not in text:
            continue
        if not _line_scope_matches(text, category):
            continue
        hits = _scan_hits(text, frozenset(observed))
        if hits:
            matched_line = text
            break

    if matched_line is None:
        return None
    upgraded = dict(base_raw)
    upgraded["decision"] = "REJECT"
    upgraded["confidence"] = RAG_UPGRADE_CONFIDENCE
    upgraded["risk_level"] = "HIGH"
    if not upgraded.get("risk_type"):
        upgraded["risk_type"] = ["POTENTIAL_IP_RISK"]
    upgraded["policy"] = []
    upgraded["rationale"] = (
        "基础为弱证据 REJECT 候选（本会转人工）；预塞判例文本命中本案表面特征，"
        "RAG-in-prompt 收口为自动拒绝。"
    )
    upgraded["reasons"] = list(base_raw.get("reasons") or []) + [
        f"RAG 命中判例行: {matched_line[:140]}"
    ]
    upgraded["rag_hit_line"] = matched_line
    return upgraded


class ContextAwareSingleCallMock:
    """2b RAG 变体 mock（确定性；name=single-call-rag-mock-v1）。

    与默认 mock 同基础表面审查（复用 ``_surface_predict``）；随后按 R-2b
    （``rag_decision_from_context``）消费 ``case_json["extra_context"]`` 里预塞的
    判例/政策文本 —— 命中 → 把本会转人工的弱 REJECT 候选升级为自动拒绝，未命中 →
    原样返回。不读 expected、不造证据、不加工具。
    """

    name = RAG_MOCK_NAME

    def __call__(self, case_json: dict) -> dict:
        base = _surface_predict(case_json)
        lines = [
            str(x)
            for x in (case_json.get("extra_context") or [])
            if isinstance(x, str) and x.strip()
        ]
        upgraded = rag_decision_from_context(base, lines, case_json)
        return upgraded if upgraded is not None else base


# 置信门槛后处理（REJECT 候选 conf < 门槛 → HUMAN_REVIEW）


def apply_abstain_threshold(
    raw: dict, *, threshold: float
) -> tuple[str, bool]:
    """对 mock/LLM 原始输出的决策做确定性后处理。

    :return: (final_decision, abstained)；abstained=True 表示被门槛改写
        （REJECT 候选 conf < threshold → HUMAN_REVIEW）。
    """
    if raw.get("decision") == "REJECT" and (raw.get("confidence") or 0.0) < threshold:
        return "HUMAN_REVIEW", True
    return raw.get("decision"), False


# SchemeRunner

SingleCallLLMFn = Callable[[dict], dict]


class SingleCallScheme(SchemeRunner):
    """Baseline 2 —— 单次 LLM 调用（默认确定性 mock，可注入 llm_fn）。

    ``extra_context`` 非 None → 2b RAG-in-prompt（静态政策/先例文本经
    ``ContextAwareSingleCallMock`` 消费）；None = 2a Raw Input，**零行为变化**。
    """

    name = "single_call_llm"

    def __init__(
        self,
        llm_fn: SingleCallLLMFn | None = None,
        extra_context: list[str] | None = None,
    ) -> None:
        self._extra_context: list[str] | None = (
            None if extra_context is None else [str(x) for x in extra_context]
        )
        if self._extra_context is None:
            # 2a：现行为（默认 mock / 用户注入 llm_fn），llm_fn 直接消费 case_json
            self._llm_fn: SingleCallLLMFn = llm_fn or DefaultSingleCallMock()
        else:
            # 2b：llm_fn 缺省换 RAG 变体 mock；无论哪种都把预塞文本注入调用入参
            context = list(self._extra_context)
            base_fn: SingleCallLLMFn = llm_fn or ContextAwareSingleCallMock()

            def _inject(case_json: dict) -> dict:
                injected = dict(case_json)
                injected["extra_context"] = context
                return injected

            def _call(case_json: dict) -> dict:
                return base_fn(_inject(case_json))

            self._llm_fn = _call

    @property
    def extra_context(self) -> list[str] | None:
        """预塞文本（None = 2a Raw Input；列表 = 2b RAG-in-prompt）。"""
        return self._extra_context

    async def run(self, case: EvalCase, ctx: EvalContext) -> EvalRecord:
        # 快照 = 整份 case 输入（含图片 url 字面）。当前 eval 数据的 url 带语义段
        # （`eval/viol_*` / `logo_*` / `clean_*` / `bound_*`），本模块 mock 不读 url
        # （只扫 ocr_text / 表面文本）→ 现无实际影响；换真实 LLM 前必须改中性 URL，
        # 否则模型会从 url 直接读到类别信号，等同泄漏 GT 类目。
        case_json = case.input.model_dump(mode="json")
        raw = self._llm_fn(case_json)
        if not isinstance(raw, dict) or "decision" not in raw:
            raise ValueError(
                f"single_call llm_fn 输出非法（缺 decision）: case={case.eval_case_id}"
            )
        decision, abstained = apply_abstain_threshold(
            raw, threshold=ctx.abstain_confidence_threshold
        )
        raw_decision = raw.get("decision")
        confidence = raw.get("confidence")
        reasons = [str(r) for r in (raw.get("reasons") or [])]
        signals = raw.get("signals") or {}
        signals_summary = "; ".join(f"{k}={v}" for k, v in signals.items())
        evidence = [
            {"type": "SURFACE_OBSERVATION", "value": reason, "extra": {}}
            for reason in reasons
        ] + [
            {
                "type": "SURFACE_SIGNALS",
                "value": signals_summary,
                "extra": {"count": len(signals)},
            }
        ]
        return EvalRecord(
            eval_case_id=case.eval_case_id,
            scheme=self.name,
            decision=decision,
            risk_level=str(raw.get("risk_level")) if raw.get("risk_level") else None,
            risk_type=[str(t) for t in (raw.get("risk_type") or [])],
            decision_confidence=float(confidence) if confidence is not None else None,
            evidence=evidence,
            policy=[str(p) for p in (raw.get("policy") or [])],
            tool_calls_actual=[],
            cost={"llm_calls": 1, "tool_calls": 0, "tokens": 0},
            detail={
                "raw_decision": raw_decision,
                "abstained_by_confidence": abstained,
                "threshold": ctx.abstain_confidence_threshold,
                "rationale": raw.get("rationale"),
            },
        )
