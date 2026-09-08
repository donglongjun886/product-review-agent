"""SingleCallLLM —— Baseline 2：单次 LLM 调用（harness/single_call_scheme.py）。

语义（对齐 docs/00-system-design.md §12.2 / docs/02-evaluation.md §3.3）：
- 一次调用 = 基础输入全量（商品快照：标题/描述/属性/类目/品牌/SKU/图片 +
  机审 OCR 文本 + screening_signals）→ 结构化决策 JSON；
- **不给商家历史 / 案例库 / 政策库**（那是 Agent 经工具调查获得的证据，给了等于
  作弊，§12.2 公平性约束）；
- 输出 = ReviewDecision 形状子集（decision / risk_level / risk_type /
  decision_confidence / policy）；``budget_used / overrides / hypothesis_trace``
  恒不适用。

Phase 1 实现（确定性、无真 LLM）：
- ``llm_fn`` 可注入（签名 ``fn(case_json: dict) -> dict``）；默认 ``DefaultSingleCallMock``
  —— "弱但有规则可循的单次审查员"，只扫描基础输入**表面字段**，体现没有调查能力
  的局限（看不到图片相似度 / 商家历史 / 在库事实 → 需要这些证据的案会判错或转
  人工，绝不读取 expected 作弊）。
- 后处理（§4.5 对齐 REJECT Gate 口径）：``decision_confidence < ctx.
  abstain_confidence_threshold(0.7)`` 的 REJECT 候选 → 确定性记 HUMAN_REVIEW。

mock 表面启发式（确定性，规则即文档）：
1. 标题/描述明示仿冒词（复刻/高仿/1:1/同款/原单）→ REJECT（conf 0.88，文本自证）；
2. 仅图片 OCR 文本含仿冒词 → REJECT 候选 conf 0.60（证据弱，后处理转 HUMAN）；
3. 标题/描述/OCR 含知名品牌词但无仿冒词 → HUMAN（授权/真伪单次调用无法核验）；
4. brand 空缺或 category 空缺 → HUMAN（关键事实缺失，R-301 同口径）；
5. 机审信号非 PASS → HUMAN（已有确定性信号，需人工复核）；
6. 否则 → PASS（conf 0.78：表面干净，单次调用视角可放行）。
真实 single-call LLM 的不稳定性主要由"看不到调查证据"体现：多信号/对抗类案在
mock 视角可能表面干净 → 误 PASS（漏放）—— 这是"缺证据"的诚实近似（§12.2 变体
2a：仅商品原始数据）。
"""

from __future__ import annotations

import re
from collections.abc import Callable

from pra.evaluation.dataset.schema import EvalCase
from pra.evaluation.harness.base import EvalContext, EvalRecord, SchemeRunner
from pra.screening.rule_engine.terms import BRAND_TERMS, EVASION_TERMS

# mock 名称常量（审计/报告标识）
DEFAULT_MOCK_NAME = "single-call-mock-v1"

__all__ = ["DefaultSingleCallMock", "SingleCallScheme", "apply_abstain_threshold"]


# ---------------------------------------------------------------------------
# 文本词命中辅助（确定性；与 screening rules 的 _term_hits 同口径但本地实现，
# 避免依赖 screening 私有函数 —— 词表仍复用 terms 单一来源）
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 默认 mock：确定性表面审查员
# ---------------------------------------------------------------------------


class DefaultSingleCallMock:
    """Phase 1 默认单次 LLM mock —— 纯函数式表面启发式（确定性、可重放）。

    ``__call__(case_json: dict) -> dict``：入参为 ``ProductReviewCase.model_dump(
    mode="json")`` 形状；返回 ReviewDecision 子集 dict：decision ∈
    {PASS, REJECT, HUMAN_REVIEW}、risk_level、risk_type、confidence、
    policy（恒 []，无政策库访问）、rationale、signals（逐项表面信号，供审计）。
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


# ---------------------------------------------------------------------------
# 置信门槛后处理（§4.5：REJECT 候选 conf < 门槛 → HUMAN_REVIEW）
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# SchemeRunner
# ---------------------------------------------------------------------------

SingleCallLLMFn = Callable[[dict], dict]


class SingleCallScheme(SchemeRunner):
    """Baseline 2 —— 单次 LLM 调用（Phase 1 = 确定性 mock，可注入 llm_fn）。"""

    name = "single_call_llm"

    def __init__(self, llm_fn: SingleCallLLMFn | None = None) -> None:
        self._llm_fn: SingleCallLLMFn = llm_fn or DefaultSingleCallMock()

    async def run(self, case: EvalCase, ctx: EvalContext) -> EvalRecord:
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
