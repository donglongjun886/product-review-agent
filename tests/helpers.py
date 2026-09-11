"""共享测试构造件：领域对象工厂 + LLMBackend 测试替身 + 常用 state 骨架。

仅被 tests/* 引用；文件名不含 test_ 前缀，pytest 不收集。
"""

from __future__ import annotations

import json
from datetime import datetime

from pra.agent.guardrails.llm_shell import LLMBackendError, LLMResponse
from pra.domain.measurement import (
    ALL_DIMENSIONS,
    DIM_IMAGE_APPEARANCE,
    DIM_LISTING_REGISTRY,
    DIM_MERCHANT_PROFILE,
    VERDICT_NEGATIVE,
    VERDICT_POSITIVE,
    make_measurement,
)
from pra.domain.models import (
    Budget,
    Evidence,
    Hypothesis,
    HypothesisStatus,
    ProductImage,
    ProductInfo,
    ProductReviewCase,
    ScreeningSignal,
    SkuInfo,
)

# domain 对象工厂


def ev(
    type_: str,
    *,
    source: str = "TestTool",
    value: str = "evidence value",
    weight: float = 0.9,
    ref_id: str | None = None,
    extra: dict | None = None,
) -> Evidence:
    # extra 复制为新 dict，防测试间共享引用
    return Evidence(
        type=type_,
        source=source,
        value=value,
        weight=weight,
        ref_id=ref_id,
        extra=dict(extra or {}),
    )


def hp(
    id_: str,
    *,
    prior: float = 0.4,
    posterior: float | None = None,
    status: HypothesisStatus = HypothesisStatus.PENDING,
    evidence_for: list[str] | None = None,
    evidence_against: list[str] | None = None,
    statement: str | None = None,
) -> Hypothesis:
    return Hypothesis(
        id=id_,
        statement=statement or f"statement-{id_}",
        prior=prior,
        posterior=posterior,
        status=status,
        evidence_for=list(evidence_for or []),
        evidence_against=list(evidence_against or []),
    )


def make_case(
    *,
    case_id: str = "CASE_TEST_001",
    brand: str | None = None,
    product_id: str = "P_TEST",
    merchant_id: str = "M_TEST",
    version: int = 3,
) -> ProductReviewCase:
    # 默认 brand=None 不误触 R1 黑名单
    product = ProductInfo(
        product_id=product_id,
        title="复古跑鞋",
        description="复古设计。",
        category="女鞋/运动鞋",
        brand=brand,
        sku_list=[SkuInfo(sku_id="S_1", color="米白", size="38", price=219.0)],
        images=[ProductImage(url="https://cdn.example.com/products/P_TEST/img1.jpg", source="主图")],
        listing_time=datetime(2024, 9, 6, 14, 0, 0),  # naive datetime（DB DATETIME 口径）
        version=version,
    )
    signals = [
        ScreeningSignal(name="KEYWORD", result="PASS", score=0.8),
        ScreeningSignal(name="LOGO_DETECT", result="PASS", score=0.2),
    ]
    return ProductReviewCase(
        case_id=case_id,
        product=product,
        merchant_id=merchant_id,
        event_type="NEW_LISTING",
        screening_signals=signals,
    )


# 常用 state 骨架（确定性纯函数/节点测试复用）


def measurement(
    dimension: str,
    *,
    source: str = "TestTool",
    source_ref: str = "REF_1",
    verdict: str = VERDICT_NEGATIVE,
    weight: float = 0.85,
    value: str = "测量完成",
) -> Evidence:
    """一条 ``MEASUREMENT`` 证据（走领域构造器，保证与生产同形）。"""
    return make_measurement(
        dimension=dimension,
        source=source,
        source_ref=source_ref,
        verdict=verdict,
        weight=weight,
        value=value,
    )


def all_measureable_caps() -> dict[str, bool]:
    """全部维度可测的能力表（评测/演示世界口径）。"""
    return {dim: True for dim in ALL_DIMENSIONS}


def covered_evidence(
    *,
    image_url: str = "https://cdn.example.com/products/P_TEST/img1.jpg",
    product_id: str = "P_TEST",
    merchant_id: str = "M_TEST",
    merchant_removals: int = 0,
    similarity: float | None = None,
) -> list[Evidence]:
    """一套**覆盖完整**的证据链（required 维度全部有一个测量结论）。

    ``similarity=None`` → 外观阴性；给值则产 ``IMAGE_SIMILARITY``（>=0.85 才算阳性）。
    """
    evs: list[Evidence] = [
        ev("PRODUCT_FACT", source="ProductTool", value="brand=山丘, version=3（库中最新）",
           weight=0.6, ref_id=product_id),
        measurement(DIM_LISTING_REGISTRY, source="ProductTool", source_ref=product_id, weight=0.6),
        ev("MERCHANT_HISTORY", source="MerchantTool",
           value=f"0 similar / {merchant_removals} removals / 0 title-relisting, credit=90",
           weight=0.85, ref_id=merchant_id,
           extra={"similar": 0, "removals": merchant_removals, "title": 0, "credit": 90}),
        measurement(
            DIM_MERCHANT_PROFILE,
            source="MerchantTool",
            source_ref=merchant_id,
            verdict=VERDICT_POSITIVE if merchant_removals >= 3 else VERDICT_NEGATIVE,
        ),
        measurement(DIM_IMAGE_APPEARANCE, source="ImageAnalysisTool", source_ref=image_url),
    ]
    if similarity is not None:
        evs.append(
            ev("IMAGE_SIMILARITY", source="ImageAnalysisTool",
               value=f"similarity={similarity:.2f}, match=某品牌经典鞋款",
               weight=similarity, ref_id=image_url)
        )
    return evs


def dc_anchor_state() -> dict:
    """锚点 state：覆盖完整的证据链 + 强相似 0.91 + 商家脏 + 可引用依据。

    相对于旧锚点（5 条证据 / posterior=0.91），本锚点改为**事实侧锚点**：
    required 四维全覆盖 + 两个维度阳性 + 带 ref_id 的 POLICY_REF/CASE_PRECEDENT。
    """
    hypotheses = [
        hp("H1", prior=0.5, status=HypothesisStatus.REFUTED, posterior=0.05,
           evidence_against=["IMAGE_SIMILARITY similarity=0.42, match=某品牌条纹运动鞋"]),
        hp("H2", prior=0.4, status=HypothesisStatus.SUPPORTED, posterior=0.91,
           evidence_for=["IMAGE_SIMILARITY similarity=0.91, match=某品牌经典鞋款"]),
    ]
    image_url = "https://cdn.example.com/products/P_88231/img1.jpg"
    evidence = [
        *covered_evidence(image_url=image_url, product_id="P_88231", merchant_id="M_5512",
                          merchant_removals=5, similarity=0.91),
        ev("CASE_PRECEDENT", source="CaseSearchTool",
           value="case_1001 无品牌标识+外观高度模仿", weight=0.8, ref_id="case_1001"),
        ev("POLICY_REF", source="PolicySearchTool",
           value="POLICY_3.2 v2 条款：外观高度模仿知名品牌设计",
           weight=0.9, ref_id="POLICY_3.2_v2_c1",
           extra={"policy_id": "POLICY_3.2", "policy_version": 2}),
    ]
    return {
        "case": make_case(brand=None, product_id="P_88231", merchant_id="M_5512"),
        "hypotheses": hypotheses,
        "evidence": evidence,
        "budget": Budget(),
        "failures": [],
        "tool_call_history": [],
        "degraded": False,
        "measurement_capabilities": all_measureable_caps(),
    }


def budget_exhausted_state() -> dict:
    return {
        "case": None,
        "hypotheses": [],
        "evidence": [],
        "budget": Budget(llm_calls=10),  # BudgetLimits().max_llm_calls == 10
        "failures": [],
        "tool_call_history": [],
        "degraded": False,
    }


# LLMBackend 测试替身（实现 pra.agent.guardrails.llm_shell.LLMBackend Protocol）

_PLAN_CONCLUDE = {"next_action": "conclude", "tools": [], "rationale": "test"}


def json_dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def plan_conclude_json() -> str:
    return json_dumps(_PLAN_CONCLUDE)

def hypothesize_json() -> str:
    return json_dumps(
        {
            "hypotheses": [
                {"statement": "刻意规避品牌识别", "prior": 0.6, "evidence_hint": []},
                {"statement": "普通设计非品牌款", "prior": 0.3, "evidence_hint": []},
            ],
            "investigation_queue": [{"q": "外观是否高度相似？", "priority": 1}],
            "rationale": "test",
        }
    )


def reevaluate_json() -> str:
    return json_dumps(
        {
            "hypothesis_updates": [
                {
                    "id": "H1",
                    "posterior": 0.9,
                    "status": "SUPPORTED",
                    "evidence_for": ["IMAGE_SIMILARITY similarity=0.91"],
                    "evidence_against": [],
                }
            ],
            "queue_updates": [{"q": "外观是否高度相似？", "status": "DONE"}],
            "new_hypotheses": [{"statement": "商家系统性上架", "prior": 0.2}],
            "evidence_sufficiency": "SUFFICIENT",
            "conflicts": [],
            "rationale": "test",
        }
    )


def decide_reject_json() -> str:
    return json_dumps(
        {
            "decision": "REJECT",
            "risk_level": "HIGH",
            "risk_type": ["POTENTIAL_IP_RISK"],
            "confidence": 0.9,
            "evidence_ids": ["IMAGE_SIMILARITY similarity=0.91, match=某品牌经典鞋款"],
            "policy": ["POLICY_3.2"],
            "rationale": "test",
        }
    )


def decide_human_json() -> str:
    return json_dumps(
        {
            "decision": "HUMAN_REVIEW",
            "risk_level": "HIGH",
            "risk_type": ["POTENTIAL_IP_RISK", "EVASION_PATTERN"],
            "confidence": 0.8,
            "evidence_ids": [],
            "policy": [],
            "rationale": "test",
        }
    )


class SequenceBackend:
    # contents 元素为 JSON 字符串（返回该串）或 None（该次抛后端异常）；calls 记录每次 messages 副本
    name = "test-sequence"

    def __init__(self, contents: list, tokens: int = 7) -> None:
        self._contents = list(contents)
        self._tokens = tokens
        self.calls: list[list] = []  # 每次 complete 收到的 messages（浅拷贝列表）

    async def complete(self, *, node: str, messages: list, json_schema: dict) -> LLMResponse:
        self.calls.append(list(messages))
        if not self._contents:
            raise LLMBackendError("stub 内容耗尽")
        content = self._contents.pop(0)
        if content is None:
            raise LLMBackendError("injected backend failure")
        return LLMResponse(content=content, tokens=self._tokens)


class AlwaysRaiseBackend:
    name = "test-always-raise"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, *, node: str, messages: list, json_schema: dict) -> LLMResponse:
        self.calls += 1
        raise LLMBackendError("injected backend failure")


class NodePayloadBackend:
    name = "test-node-payload"

    def __init__(self, payloads: dict | None = None) -> None:
        self._payloads = dict(payloads or {})
        self.calls: list[str] = []

    async def complete(self, *, node: str, messages: list, json_schema: dict) -> LLMResponse:
        self.calls.append(node)
        if node not in self._payloads:
            raise LLMBackendError(f"test backend unknown node: {node}")
        return LLMResponse(content=self._payloads[node], tokens=5)
