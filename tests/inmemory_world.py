"""InMemory 测试世界：4 个调查工具 + 各自的进程内数据源与种子数据。

``build_inmemory_tools()`` 组装 ProductTool / MerchantTool / CaseSearchTool / PolicySearchTool
（顺序即此），四件数据源均在此处**显式**构造 —— CI 不连 MySQL / Chroma，评测与业务用例可重放。
四个 InMemory 实现与种子字面量已从 ``src/pra/tools`` 各 ``tool.py`` 移出，生产装配不再提供默认世界。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pra.tools.base import Tool
from pra.tools.case_search.tool import (
    CaseHit,
    CaseSearchFilters,
    CaseSearchTool,
)
from pra.tools.merchant.tool import (
    MerchantProfile,
    MerchantTool,
)
from pra.tools.policy_search.tool import (
    PolicyClauseHit,
    PolicySearchFilters,
    PolicySearchTool,
)
from pra.tools.product.tool import (
    ProductSnapshot,
    ProductTool,
)

# ---------------------------------------------------------------------------
# 商品事实（ProductTool）
# ---------------------------------------------------------------------------

_DEFAULT_PRODUCTS: Mapping[str, dict[str, Any]] = {
    "P_88231": {
        "product_id": "P_88231",
        "category": "女鞋/运动鞋",
        "brand": None,
        "version": 3,
        "status": "ON_SALE",
    },
}


class InMemoryProductRepository:
    """ProductRepository 的 Mock 默认实现（仅供开发/测试/演示）。

    ``data`` 构造入参可注入自定义种子；不传则用模块级演示数据。
    """

    def __init__(self, data: Mapping[str, dict[str, Any]] | None = None) -> None:
        self._store: dict[str, ProductSnapshot] = {
            pid: ProductSnapshot.model_validate(row) for pid, row in (data or _DEFAULT_PRODUCTS).items()
        }

    async def get_latest(self, product_id: str) -> ProductSnapshot | None:
        return self._store.get(product_id)


# ---------------------------------------------------------------------------
# 商家行为（MerchantTool）
# ---------------------------------------------------------------------------

_DEFAULT_MERCHANTS: Mapping[str, dict[str, Any]] = {
    "M_5512": {
        "merchant_id": "M_5512",
        "similar_product_count": 23,
        "removals": 5,
        "title_relisting_count": 3,
        "credit_score": 62,
    },
}


class InMemoryMerchantRepository:
    """MerchantRepository 的 Mock 默认实现（仅供开发/测试/演示）。

    种子画像按 merchant_id 匹配；``window_days`` 在 mock 中不改变聚合结果（真实实现按其
    截取事件窗口）。
    """

    def __init__(self, data: Mapping[str, dict[str, Any]] | None = None) -> None:
        self._store: dict[str, MerchantProfile] = {
            mid: MerchantProfile.model_validate(row) for mid, row in (data or _DEFAULT_MERCHANTS).items()
        }

    async def get_profile(self, merchant_id: str, window_days: int) -> MerchantProfile | None:
        return self._store.get(merchant_id)


# ---------------------------------------------------------------------------
# 先例检索（CaseSearchTool）
# ---------------------------------------------------------------------------

_DEFAULT_PRECEDENTS: list[dict[str, Any]] = [
    {
        "case_id": "CASE_1832",
        "retrieval_score": 0.86,
        "decision": "REJECT",
        "risk_level": "HIGH",
        "risk_type": ["POTENTIAL_IP_RISK"],
        "summary": "无品牌标识 + 标题含高仿/复刻规避用语 + 商家多次改标题重上架",
        "key_evidence": ["text_evasion_word", "merchant_history>=5_removals"],
        "policy_refs": ["POLICY_3.2"],
        "category": "女鞋/运动鞋",
    },
    {
        "case_id": "CASE_0911",
        "retrieval_score": 0.31,
        "decision": "PASS",
        "risk_level": "NONE",
        "risk_type": [],
        "summary": "普通休闲鞋，无品牌标识且无规避用语",
        "key_evidence": [],
        "policy_refs": [],
        "category": "女鞋/运动鞋",
    },
]


class InMemoryCaseIndex:
    """``CaseIndex`` 的 Mock 默认实现（仅供开发/测试/演示）。

    检索 = 元数据过滤（category 精确 / risk_type 交叠）+ 种子分降序 + top_k；**query 不参与匹配**。
    """

    def __init__(self, precedents: list[dict[str, Any]] | None = None) -> None:
        # 保留原始行（含元数据过滤字段 category），检索过滤后再校验为 CaseHit。
        self._rows: list[dict[str, Any]] = list(precedents or _DEFAULT_PRECEDENTS)

    async def search(self, query: str, filters: CaseSearchFilters, top_k: int) -> list[CaseHit]:
        rows = self._rows
        if filters.category:
            rows = [r for r in rows if r.get("category") == filters.category]
        if filters.risk_type:
            wanted = set(filters.risk_type)
            rows = [r for r in rows if wanted & set(r.get("risk_type", []))]
        ranked = sorted(rows, key=lambda r: r.get("retrieval_score", 0.0), reverse=True)
        return [CaseHit.model_validate(r) for r in ranked[:top_k]]


# ---------------------------------------------------------------------------
# 政策检索（PolicySearchTool）
# ---------------------------------------------------------------------------

_DEFAULT_CLAUSES: list[dict[str, Any]] = [
    {
        "policy_id": "POLICY_3.2",
        "version": 2,
        "clause_id": "POLICY_3.2_v2_c1",
        "title": "标题/描述使用仿冒规避用语",
        "text": "商品标题或描述使用高仿、复刻、1:1 等仿冒规避用语且无品牌授权，判定为高风险，转人工审核处理",
        "category": "女鞋/运动鞋",
        "risk_type": ["POTENTIAL_IP_RISK"],
        "status": "EFFECTIVE",
        "effective_date": "2024-03-01",
    },
    {
        "policy_id": "POLICY_3.1",
        "version": 1,
        "clause_id": "POLICY_3.1_v1_c2",
        "title": "品牌词滥用（旧版）",
        "text": "标题/描述不得出现未授权品牌词（旧版，已失效）",
        "category": "全类目",
        "risk_type": ["POTENTIAL_IP_RISK"],
        "status": "EXPIRED",
        "effective_date": "2023-01-01",
    },
]


class InMemoryPolicyIndex:
    """``PolicyIndex`` 的 Mock 默认实现（仅供开发/测试/演示）。

    检索 = 版本有效性 + 元数据过滤（category / risk_type）+ top_k 截断；**query 不参与匹配**。
    """

    def __init__(self, clauses: list[dict[str, Any]] | None = None) -> None:
        self._rows: list[dict[str, Any]] = list(clauses or _DEFAULT_CLAUSES)

    async def search(
        self,
        query: str,
        filters: PolicySearchFilters,
        top_k: int,
        effective_only: bool,
    ) -> list[PolicyClauseHit]:
        rows = self._rows
        if effective_only:
            rows = [r for r in rows if r.get("status") == "EFFECTIVE"]
        if filters.category:
            rows = [r for r in rows if r.get("category") in (None, filters.category, "全类目")]
        if filters.risk_type:
            wanted = set(filters.risk_type)
            rows = [r for r in rows if wanted & set(r.get("risk_type", []))]
        return [PolicyClauseHit.model_validate(r) for r in rows[:top_k]]


def build_inmemory_tools() -> list[Tool]:
    """组装 InMemory 世界的 4 个调查工具（顺序：product / merchant / case_search / policy_search）。

    :return: 数据源为四个 InMemory 实现的工具列表，可直接交给 tools_node 按名调度。
    """
    tools: list[Tool] = [
        ProductTool(repo=InMemoryProductRepository()),
        MerchantTool(repo=InMemoryMerchantRepository()),
        CaseSearchTool(index=InMemoryCaseIndex()),
        PolicySearchTool(index=InMemoryPolicyIndex()),
    ]
    return tools
