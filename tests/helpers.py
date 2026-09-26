"""共享测试构造件：领域对象工厂 + LLMBackend 测试替身 + 常用 state 骨架 + 真模型缓存探测。

仅被 tests/* 引用；文件名不含 test_ 前缀，pytest 不收集。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pra.agent.guardrails.llm_shell import LLMBackendError, LLMResponse
from pra.domain.measurement import (
    ALL_DIMENSIONS,
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
    SkuInfo,
)
from pra.tools import BGE_MODEL as BGE_DEFAULT_MODEL
from pra.tools.base import Tool

# 真模型缓存探测（只读文件系统；生产路径本身**不做**磁盘预检，靠 fastembed 的
# ``local_files_only=True`` —— 本探测只用于决定真模型用例是否 skip）
_BGE_HF_SOURCE_REPO = "Qdrant"


def bge_model_cached(cache_dir: Path, model_name: str = BGE_DEFAULT_MODEL) -> bool:
    """``cache_dir`` 下是否已有该模型的可加载 onnx（只读，不 import fastembed、不联网）。

    fastembed 两种落盘布局都探：HF snapshot（``models--<org>--<name>/``，blobs 为哈希名、不带
    ``.onnx`` 后缀）与 GCS tar 解包（``fast-<name>/``，onnx 带扩展名）；漏探任一都会在布局不同的
    机器上误 skip。BGE 模型的 HF 源仓库是 Qdrant 官方 ONNX 仓库（**只是模型来源，与向量库选型无关**）。
    """
    if not cache_dir.is_dir():
        return False
    layouts = [
        cache_dir / f"models--{model_name.replace('/', '--')}",
        cache_dir / f"fast-{model_name.rsplit('/', 1)[-1]}",
    ]
    if model_name == BGE_DEFAULT_MODEL:
        name = model_name.rsplit("/", 1)[-1]
        layouts.append(cache_dir / f"models--{_BGE_HF_SOURCE_REPO}--{name}")
    return any(layout.is_dir() and any(layout.glob("**/*.onnx")) for layout in layouts)


# 工具装配辅助：工具列表的**顺序对下游无语义**（tools_node 全程按 name 调度），
# 只被测试断言引用。故测试一律按名取工具，别再写 ``build_tools()[4]`` 这类位置下标 —— 那是脆弱
# 耦合，装配顺序一变就误伤断言。


def tool_by_name(tools: list[Tool], name: str) -> Tool:
    """按 ``tool.name`` 从工具列表取工具；未命中抛 ``KeyError``（附现有工具名，便于定位）。"""
    for tool in tools:
        if getattr(tool, "name", None) == name:
            return tool
    available = [getattr(t, "name", repr(t)) for t in tools]
    raise KeyError(f"工具列表无此 name：{name!r}；现有工具 = {available}")


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
    # 默认 brand=None 不误触品牌黑名单
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
    return ProductReviewCase(
        case_id=case_id,
        product=product,
        merchant_id=merchant_id,
        event_type="NEW_LISTING",
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
    product_id: str = "P_TEST",
    merchant_id: str = "M_TEST",
    merchant_removals: int = 0,
) -> list[Evidence]:
    """一套**覆盖完整**的证据链（required 维度全部有一个测量结论）。

    ``merchant_removals >= MERCHANT_DIRTY_MIN`` → 商家画像阳性（阻塞 PASS 的唯一证据侧阳性）。
    """
    return [
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
    ]


def evasion_case(*, case_id: str = "CASE_TEST_EVASION", product_id: str = "P_88231",
                 merchant_id: str = "M_5512") -> ProductReviewCase:
    """命中 R-302 规避词（``高仿``）的案件 —— REJECT Gate 的文本确证来源。"""
    case = make_case(case_id=case_id, brand=None, product_id=product_id,
                     merchant_id=merchant_id)
    return case.model_copy(
        update={
            "product": case.product.model_copy(
                update={"title": "高仿 1:1 复古跑鞋", "description": "复刻经典款鞋型。"}
            )
        }
    )


def risk_anchor_state() -> dict:
    """锚点 state：命中 R-302 的文本确证 + 商家脏 + 带 ref_id 的可引用依据。

    required 三维全覆盖 + 一个维度阳性 + 带 ref_id 的 POLICY_REF/CASE_PRECEDENT
    ⇒ 硬规则不命中、弃权清单全空、REJECT Gate（R-302 ∧ 可引用依据）满足。
    """
    hypotheses = [
        hp("H1", prior=0.5, status=HypothesisStatus.REFUTED, posterior=0.05,
           evidence_against=["PRODUCT_FACT brand=山丘, version=3（库中最新）"]),
        hp("H2", prior=0.4, status=HypothesisStatus.SUPPORTED, posterior=0.91,
           evidence_for=["MERCHANT_HISTORY 5 removals"]),
    ]
    evidence = [
        *covered_evidence(product_id="P_88231", merchant_id="M_5512", merchant_removals=5),
        ev("CASE_PRECEDENT", source="CaseSearchTool",
           value="case_1001 无品牌标识+外观高度模仿", weight=0.8, ref_id="case_1001"),
        ev("POLICY_REF", source="PolicySearchTool",
           value="POLICY_3.2 v2 条款：外观高度模仿知名品牌设计",
           weight=0.9, ref_id="POLICY_3.2_v2_c1",
           extra={"policy_id": "POLICY_3.2", "policy_version": 2}),
    ]
    return {
        "case": evasion_case(product_id="P_88231", merchant_id="M_5512"),
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
                    "evidence_for": ["MERCHANT_HISTORY 5 removals"],
                    "evidence_against": [],
                }
            ],
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
            "evidence_ids": ["MERCHANT_HISTORY 5 removals"],
            "policy": ["POLICY_3.2"],
            "rationale": "test",
        }
    )


class SequenceBackend:
    # contents 元素为 JSON 字符串（返回该串）或 None（该次抛后端异常）；calls 记录每次收到的 state
    name = "test-sequence"

    def __init__(self, contents: list, tokens: int = 7) -> None:
        self._contents = list(contents)
        self._tokens = tokens
        self.calls: list[dict] = []  # 每次 complete 收到的 state

    async def complete(
        self, *, node: str, state: dict, json_schema: dict, feedback: list[str] | None = None
    ) -> LLMResponse:
        self.calls.append(state)
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

    async def complete(
        self, *, node: str, state: dict, json_schema: dict, feedback: list[str] | None = None
    ) -> LLMResponse:
        self.calls += 1
        raise LLMBackendError("injected backend failure")


class NodePayloadBackend:
    name = "test-node-payload"

    def __init__(self, payloads: dict | None = None) -> None:
        self._payloads = dict(payloads or {})
        self.calls: list[str] = []

    async def complete(
        self, *, node: str, state: dict, json_schema: dict, feedback: list[str] | None = None
    ) -> LLMResponse:
        self.calls.append(node)
        if node not in self._payloads:
            raise LLMBackendError(f"test backend unknown node: {node}")
        return LLMResponse(content=self._payloads[node], tokens=5)


_WALKTHROUGH_DECIDE = {
    "decision": "HUMAN_REVIEW",
    "risk_level": "MEDIUM",
    "risk_type": ["POTENTIAL_IP_RISK"],
    "confidence": 0.8,
    "evidence_ids": [],
    "policy": [],
    "rationale": "test",
}


class WalkthroughBackend:
    """测试用确定性后端：把四节点图驱动到终态（无网络、无真实模型语义，同 state 恒同输出）。

    剧本：``hypothesize`` 固定 2 假设；``plan`` 首轮按案件标识调度 ProductTool + MerchantTool
    （补齐本 listing 事实通道），其后一律 ``conclude``（保证回环必终止，不依赖收敛判定）；
    ``reevaluate`` 恒「无更新 / INSUFFICIENT」；``decide`` 固定 HUMAN_REVIEW 提案（Gate 收口）。
    需真实 MySQL / Chroma 链路或完整调查的用例不适用本替身。
    """

    name = "test-walkthrough"

    def __init__(self) -> None:
        self.plan_calls = 0
        self.calls: list[str] = []  # 每次 complete 收到的 node

    async def complete(
        self, *, node: str, state: dict, json_schema: dict, feedback: list[str] | None = None
    ) -> LLMResponse:
        self.calls.append(node)
        if node == "hypothesize":
            return LLMResponse(content=hypothesize_json(), tokens=7)
        if node == "plan":
            self.plan_calls += 1
            return LLMResponse(content=self._plan(state), tokens=7)
        if node == "reevaluate":
            return LLMResponse(
                content=json_dumps(
                    {
                        "hypothesis_updates": [],
                        "new_hypotheses": [],
                        "evidence_sufficiency": "INSUFFICIENT",
                        "conflicts": [],
                        "rationale": "test",
                    }
                ),
                tokens=7,
            )
        if node == "decide":
            return LLMResponse(content=json_dumps(_WALKTHROUGH_DECIDE), tokens=7)
        raise LLMBackendError(f"test backend unknown node: {node}")

    def _plan(self, state: dict) -> str:
        """首轮按 ``state["case"]`` 的两个标识调度事实工具；其后（或标识缺失）conclude。"""
        if self.plan_calls > 1:
            return json_dumps({"next_action": "conclude", "tools": [], "rationale": "test"})
        case = state.get("case")
        case = case if isinstance(case, dict) else {}
        product = case.get("product")
        product = product if isinstance(product, dict) else {}
        tools: list[dict] = []
        if product.get("product_id"):
            tools.append(
                {
                    "tool": "ProductTool",
                    "args": {"product_id": product["product_id"]},
                    "reason": "核对商品在库事实",
                    "priority": 1,
                }
            )
        if case.get("merchant_id"):
            tools.append(
                {
                    "tool": "MerchantTool",
                    "args": {"merchant_id": case["merchant_id"], "window_days": 90},
                    "reason": "核查商家系统性上架/下架历史",
                    "priority": 2,
                }
            )
        if not tools:
            return json_dumps({"next_action": "conclude", "tools": [], "rationale": "test"})
        return json_dumps(
            {"next_action": "call_tools", "tools": tools, "rationale": "补齐事实通道"}
        )
