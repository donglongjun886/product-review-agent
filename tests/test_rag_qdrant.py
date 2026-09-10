"""Qdrant 后端索引测试（tests/test_rag_qdrant.py）—— Phase 2 同构等价与装配开关。

对齐 docs/06-rag-phase2-qdrant-bge.md §2.1/§2.3/§4 验收 3/6：
1. **同构等价（核心）**：同一 corpus rows + 同一 embedder（MockHash）下，qdrant 与
   local（Rag*Index）两索引的 ``search`` 返回**同序同 id**；case 的 retrieval_score
   允许 ulp 级浮点尾差（qdrant COSINE 存 float32 vs 纯 Python float64 余弦，
   实测偏差 ~1e-8；个别恰好跨 6 位取整边界时显示差 ≤1e-6 → 断言 ``abs <= 1e-6``）。
   mode 覆盖 hybrid / vector / bm25。
2. 确定性：同索引同 query 两次调用结果相等（同输入同输出）。
3. 协议形状：返回 PolicyClauseHit / CaseHit、retrieval_score ∈ [0,1]、Top-K ≤ top_k。
4. 隔离（R-4）：真实 Case KB 命中 case_id 均为 RAG_CASE_ 前缀。
5. 元数据/版本过滤语义与 local 逐条一致（EXPIRED 排除 / 全类目命中 / category /
   risk_type 交叠）。
6. 装配开关：factory ``backend="qdrant"`` 与 ``build_tools(rag_backend="qdrant")``
   注入 Qdrant*Index（离线可用：embedder 一律 MockHashEmbedder）；
   collection 复用/维度校验、本地持久 ``path=`` 模式。
7. qdrant client 生命周期：每测试自建 ``QdrantClient(":memory:")`` 注入（或默认
   location=":memory:" 自建），collection 名冲突用独立 collection_prefix / 新 client。

**point id 取值域（u64）不在这里**：本文件顶层 ``importorskip("qdrant_client")``，而 CI
只跑 ``uv sync --frozen``（不装 extra）→ 整文件在 CI 上 skip。故该契约放
``tests/test_rag_qdrant_point_id.py``（**不依赖 qdrant-client，任何环境恒跑**）——
进程内模式不校验 id 上界，128 位实现曾在本文件全绿、只在真 server 炸。

约定与现有 tests 一致：**零网络、零模型下载**（qdrant 进程内模式 + MockHash），
embedder 一律 MockHashEmbedder；无 qdrant-client 环境整文件 skip（importorskip）。
"""

from __future__ import annotations

import pytest

qdrant_client = pytest.importorskip("qdrant_client")  # noqa: F841 — 无 qdrant-client 整文件跳过

from pra.domain.models import RiskType
from pra.rag.corpus import load_cases, load_policies
from pra.rag.embedder import MockHashEmbedder
from pra.rag.factory import build_case_index, build_policy_index
from pra.rag.index import RagCaseIndex, RagPolicyIndex
from pra.rag.qdrant_index import QdrantCaseIndex, QdrantPolicyIndex
from pra.tools import build_tools
from pra.tools.case_search.tool import CaseHit, CaseSearchFilters
from pra.tools.policy_search.tool import PolicyClauseHit, PolicySearchFilters

POLICY_ROWS = load_policies()[0]
CASE_ROWS = load_cases()[0]

_EXPIRED_OLD_CLAUSE = "POLICY_2.1_v1_c1"  # 全类目 EXPIRED 旧版（版本过滤测试面）
_SHOE_CATEGORY = "女鞋/运动鞋"
_BAG_CATEGORY = "箱包/女包"
_IP = RiskType.POTENTIAL_IP_RISK
_FALSE_CLAIM = RiskType.FALSE_CLAIM
_EVASION = RiskType.EVASION_PATTERN

# 代表性 query × filter 组合（覆盖：空 filter、category 过滤、risk_type 过滤、
# policy effective_only=True/False、EXPIRED 排除、全类目条款命中、候选空 → []）。
_POLICY_COMBOS: list[tuple[str, PolicySearchFilters, bool, int]] = [
    ("外观高度模仿知名品牌无授权", PolicySearchFilters(), True, 5),
    ("外观高度模仿知名品牌无授权", PolicySearchFilters(category=_SHOE_CATEGORY), True, 5),
    ("外观高度模仿知名品牌无授权", PolicySearchFilters(category=_BAG_CATEGORY), True, 5),
    ("仿冒 高仿 复刻 原单", PolicySearchFilters(risk_type=[_IP]), True, 6),
    ("功效 夸大 根治 去皱", PolicySearchFilters(risk_type=[_FALSE_CLAIM]), True, 6),
    ("永久去皱 根治脚气 功效夸大", PolicySearchFilters(), True, 10),
    ("永久去皱 根治脚气 功效夸大", PolicySearchFilters(), False, 10),
    ("规避 换链接 改标题 重上架", PolicySearchFilters(risk_type=[_EVASION]), True, 6),
    ("外观模仿", PolicySearchFilters(category=_BAG_CATEGORY), True, 20),
    ("不存在类目 数码3C", PolicySearchFilters(category="数码/3C"), True, 5),
]
_CASE_COMBOS: list[tuple[str, CaseSearchFilters, int]] = [
    ("无品牌高相似商家多次上架", CaseSearchFilters(), 5),
    ("无品牌高相似商家多次上架", CaseSearchFilters(category=_SHOE_CATEGORY), 5),
    ("无品牌高相似商家多次上架", CaseSearchFilters(category=_BAG_CATEGORY), 5),
    ("无品牌高相似商家多次上架", CaseSearchFilters(category="服装/卫衣"), 5),
    ("外观高度模仿 相似度", CaseSearchFilters(risk_type=[_IP]), 6),
    ("外观模仿", CaseSearchFilters(category=_SHOE_CATEGORY, risk_type=[_IP]), 6),
    ("夸大宣传 功效 虚假", CaseSearchFilters(category=_BAG_CATEGORY), 6),
    ("规避 多次改标题 重上架", CaseSearchFilters(risk_type=[_EVASION]), 6),
    ("食品 不存在类目", CaseSearchFilters(category="食品"), 5),
]


def _policy_local(mode: str) -> RagPolicyIndex:
    return RagPolicyIndex(POLICY_ROWS, embedder=MockHashEmbedder(), mode=mode)


def _policy_qdrant(mode: str) -> QdrantPolicyIndex:
    # 每实例独立 :memory: client（collection 名含 prefix，互不冲突）
    return QdrantPolicyIndex(
        POLICY_ROWS,
        embedder=MockHashEmbedder(),
        mode=mode,
        qdrant_client=qdrant_client.QdrantClient(":memory:"),
        collection_prefix=f"testp_{mode}",
    )


def _case_local(mode: str) -> RagCaseIndex:
    return RagCaseIndex(CASE_ROWS, embedder=MockHashEmbedder(), mode=mode)


def _case_qdrant(mode: str) -> QdrantCaseIndex:
    return QdrantCaseIndex(
        CASE_ROWS,
        embedder=MockHashEmbedder(),
        mode=mode,
        qdrant_client=qdrant_client.QdrantClient(":memory:"),
        collection_prefix=f"testc_{mode}",
    )


# ---------------------------------------------------------------------------
# 1) 同构等价（核心）：qdrant ≡ local —— 同 id 同序；score 允许 1e-6 ulp 尾差
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["hybrid", "vector", "bm25"])
async def test_policy_qdrant_equivalent_to_local(mode: str) -> None:
    local = _policy_local(mode)
    qd = _policy_qdrant(mode)
    assert local.size == qd.size == len(POLICY_ROWS)
    assert qd.effective_count() == local.effective_count()
    for query, filters, effective_only, top_k in _POLICY_COMBOS:
        lh = await local.search(query, filters, top_k, effective_only)
        qh = await qd.search(query, filters, top_k, effective_only)
        assert [h.clause_id for h in lh] == [h.clause_id for h in qh], (
            f"policy mode={mode} q={query!r} filters={filters} eff={effective_only}: "
            "qdrant 与 local 命中序/id 不一致"
        )


@pytest.mark.parametrize("mode", ["hybrid", "vector", "bm25"])
async def test_case_qdrant_equivalent_to_local(mode: str) -> None:
    local = _case_local(mode)
    qd = _case_qdrant(mode)
    assert local.size == qd.size == len(CASE_ROWS)
    for query, filters, top_k in _CASE_COMBOS:
        lh = await local.search(query, filters, top_k)
        qh = await qd.search(query, filters, top_k)
        assert [h.case_id for h in lh] == [h.case_id for h in qh], (
            f"case mode={mode} q={query!r} filters={filters}: qdrant 与 local 命中序/id 不一致"
        )
        # Qdrant COSINE 存 float32 vs 本地纯 Python float64 余弦 → 允许 ulp 尾差
        # （实测 ~1e-8；个别恰跨 6 位取整边界时显示差 = 相邻两位小数 ≈1e-6，docs/06
        # §2.1 同构口径）。两值均已 6 位取整：diff 先 round 清二进制表示 ulp 再断 ≤1e-6。
        for a, b in zip(lh, qh):
            assert round(abs(a.retrieval_score - b.retrieval_score), 9) <= 1e-6, (
                a.case_id, a.retrieval_score, b.retrieval_score
            )


# ---------------------------------------------------------------------------
# 2) 确定性 + 3) 协议形状 + 4) 隔离
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["hybrid", "vector", "bm25"])
async def test_qdrant_deterministic_same_index_twice(mode: str) -> None:
    """同索引同 query 两次调用结果相等（qdrant 进程内模式确定性）。"""
    for idx, kind in (
        (QdrantPolicyIndex(POLICY_ROWS, embedder=MockHashEmbedder(), mode=mode,
                           qdrant_client=qdrant_client.QdrantClient(":memory:")), "policy"),
        (QdrantCaseIndex(CASE_ROWS, embedder=MockHashEmbedder(), mode=mode,
                         qdrant_client=qdrant_client.QdrantClient(":memory:")), "case"),
    ):
        if kind == "policy":
            a = await idx.search("外观高度模仿知名品牌", PolicySearchFilters(), 5, True)
            b = await idx.search("外观高度模仿知名品牌", PolicySearchFilters(), 5, True)
            assert [(h.clause_id, h.text) for h in a] == [(h.clause_id, h.text) for h in b]
        else:
            a = await idx.search("无品牌高相似", CaseSearchFilters(), 5)
            b = await idx.search("无品牌高相似", CaseSearchFilters(), 5)
            assert [(h.case_id, h.retrieval_score) for h in a] == [(h.case_id, h.retrieval_score) for h in b]


def test_qdrant_missing_client_raises_runtime_error(monkeypatch) -> None:
    """无 qdrant-client 环境构造 qdrant 索引 → RuntimeError 提示 `uv sync --extra rag`。"""
    import builtins

    real_import = builtins.__import__

    def _block_qdrant(name, *args, **kwargs):
        if name == "qdrant_client" or name.startswith("qdrant_client."):
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_qdrant)
    from pra.rag.qdrant_index import _import_qdrant

    with pytest.raises(RuntimeError, match=r"uv sync --extra rag"):
        _import_qdrant()


async def test_qdrant_hit_types_and_topk_shape() -> None:
    p_idx = _policy_qdrant("hybrid")
    c_idx = _case_qdrant("hybrid")
    # policy 协议形状：PolicyClauseHit、Top-K ≤ top_k
    ph = await p_idx.search("仿冒 外观模仿", PolicySearchFilters(), top_k=3, effective_only=True)
    assert ph and all(isinstance(h, PolicyClauseHit) for h in ph)
    assert len(ph) <= 3
    # case 协议形状：CaseHit、retrieval_score ∈ [0,1]、Top-K ≤ top_k
    ch = await c_idx.search("无品牌高相似", CaseSearchFilters(), top_k=10)
    assert ch and all(isinstance(h, CaseHit) for h in ch)
    assert len(ch) <= 10
    assert all(0.0 <= h.retrieval_score <= 1.0 for h in ch)


async def test_qdrant_case_kb_isolation_rag_prefix() -> None:
    """隔离红线 R-4：真实 Case KB 命中 case_id 均为 RAG_CASE_ 前缀。"""
    idx = _case_qdrant("hybrid")
    hits = await idx.search("仿冒 外观高度模仿 无品牌", CaseSearchFilters(), top_k=10)
    assert hits
    assert all(str(h.case_id).startswith("RAG_CASE_") for h in hits)


# ---------------------------------------------------------------------------
# 5) 元数据/版本过滤语义（与 local 逐条一致，直接对 qdrant 索引断言行为）
# ---------------------------------------------------------------------------


async def test_qdrant_policy_version_and_category_semantics() -> None:
    idx = _policy_qdrant("hybrid")
    query = "永久去皱 根治脚气 功效夸大"  # 命中 POLICY_2.1 v1(EXPIRED) 文案

    eff = await idx.search(query, PolicySearchFilters(), top_k=10, effective_only=True)
    assert eff and all(h.status == "EFFECTIVE" for h in eff)
    assert not any(h.clause_id == _EXPIRED_OLD_CLAUSE for h in eff), "effective_only=True 排除 EXPIRED 旧版"

    both = await idx.search(query, PolicySearchFilters(), top_k=10, effective_only=False)
    assert any(h.clause_id == _EXPIRED_OLD_CLAUSE for h in both), "effective_only=False 应含 EXPIRED 旧版"

    # category 过滤语义：具体类目 + 全类目都留（row.category ∈ {None, 目标, 全类目}）
    bag = await idx.search(
        "外观模仿 品牌", PolicySearchFilters(category=_BAG_CATEGORY), top_k=20, effective_only=True
    )
    assert bag
    assert all(h.category in (_BAG_CATEGORY, "全类目") for h in bag)
    assert any(h.category == "全类目" for h in bag), "全类目条款照常匹配"
    assert not any(h.category == _SHOE_CATEGORY for h in bag), "其它具体类目条款不得出现"

    # risk_type 过滤语义：全部命中含该风险类型（给了 filter 时空 risk_type 行被排除）
    ip = await idx.search("仿冒", PolicySearchFilters(risk_type=[_IP]), top_k=20, effective_only=True)
    assert ip
    assert all(_IP in h.risk_type for h in ip)


async def test_qdrant_case_metadata_semantics() -> None:
    idx = _case_qdrant("hybrid")
    # category 精确匹配：命中 case_id 全部落在该类目 corpus 子集内（CaseHit 不带
    # category 字段——过滤在索引侧完成，语义 = InMemory 精确匹配）
    shoe_ids = {c.case_id for c in CASE_ROWS if c.category == _SHOE_CATEGORY}
    hits = await idx.search("无品牌高相似", CaseSearchFilters(category=_SHOE_CATEGORY), top_k=10)
    assert hits and {h.case_id for h in hits} <= shoe_ids
    # category + risk_type 组合：命中 risk_type 全部含目标类型
    ip = await idx.search(
        "外观模仿", CaseSearchFilters(category=_SHOE_CATEGORY, risk_type=[_IP]), top_k=10
    )
    assert ip and all(_IP in h.risk_type for h in ip)
    assert {h.case_id for h in ip} <= shoe_ids
    # 候选空 → 直接返回 []（工具 ok=True）
    none = await idx.search("任何词", CaseSearchFilters(category="不存在/类目"), top_k=5)
    assert none == []


# ---------------------------------------------------------------------------
# 6) 装配开关：factory backend + build_tools rag_backend 透传
# ---------------------------------------------------------------------------


def test_factory_default_backend_is_local_unchanged() -> None:
    """backend 缺省 local：与改动前逐字节等价（仍返回 Rag*Index，未拉起 qdrant 路径）。"""
    assert isinstance(build_policy_index(rows=POLICY_ROWS), RagPolicyIndex)
    assert isinstance(build_case_index(rows=CASE_ROWS), RagCaseIndex)


def test_factory_backend_qdrant_returns_qdrant_index() -> None:
    """factory backend="qdrant"：延迟 import 构造 Qdrant*Index（离线可用，mock embedder）。"""
    p_idx = build_policy_index(
        rows=POLICY_ROWS, embedder=MockHashEmbedder(), backend="qdrant"
    )
    c_idx = build_case_index(
        rows=CASE_ROWS, embedder=MockHashEmbedder(), backend="qdrant"
    )
    assert isinstance(p_idx, QdrantPolicyIndex)
    assert isinstance(c_idx, QdrantCaseIndex)


async def test_build_tools_rag_backend_qdrant_injects_qdrant_index() -> None:
    tools = build_tools("rag", rag_backend="qdrant")
    assert [t.name for t in tools] == [
        "ProductTool", "ImageAnalysisTool", "OCRTool", "MerchantTool",
        "CaseSearchTool", "PolicySearchTool",
    ]
    assert isinstance(tools[4]._index, QdrantCaseIndex)
    assert isinstance(tools[5]._index, QdrantPolicyIndex)
    # 其余 4 工具事实世界不受影响（仍 InMemory）
    assert type(tools[0]._repo).__name__ == "InMemoryProductRepository"
    # qdrant 后端经工具契约可检索（离线，默认 location=":memory:" 自建 client）
    hits = await tools[4]._index.search("无品牌高相似", CaseSearchFilters(), top_k=3)
    assert hits and all(str(h.case_id).startswith("RAG_CASE_") for h in hits)


# ---------------------------------------------------------------------------
# 7) collection 生命周期：复用 / 维度校验 / 本地持久 path= 模式
# ---------------------------------------------------------------------------


def test_qdrant_collection_reuse_same_client_same_dim() -> None:
    """同 client + 同 prefix：已存在 collection（dim 一致）复用之——两次构造不报错。"""
    client = qdrant_client.QdrantClient(":memory:")
    QdrantCaseIndex(
        CASE_ROWS, embedder=MockHashEmbedder(), qdrant_client=client, collection_prefix="reuse"
    )
    # 复用同 collection（dim 一致）→ 不抛错（幂等 upsert 覆盖）
    subset = CASE_ROWS[:3]
    idx2 = QdrantCaseIndex(
        subset, embedder=MockHashEmbedder(), qdrant_client=client, collection_prefix="reuse"
    )
    assert idx2.size == len(subset)


def test_qdrant_collection_dim_mismatch_raises() -> None:
    """已存在 collection 但维度不一致 → 拒绝复用（报错提示换 prefix/清理）。"""
    client = qdrant_client.QdrantClient(":memory:")
    name = "dimcase_case_256"  # QdrantCaseIndex(dim=256, prefix="dimcase") 会用的 collection 名
    client.create_collection(
        collection_name=name,
        vectors_config=qdrant_client.models.VectorParams(size=64, distance=qdrant_client.models.Distance.COSINE),
    )
    with pytest.raises(ValueError, match="维度"):
        QdrantCaseIndex(
            CASE_ROWS, embedder=MockHashEmbedder(), qdrant_client=client, collection_prefix="dimcase"
        )


async def test_qdrant_local_persistence_path_mode(tmp_path) -> None:
    """``location=<目录>`` = qdrant path= 本地持久模式（原生支持，docs/06 P2-3）。"""
    loc = tmp_path / "kb"
    idx = QdrantPolicyIndex(
        POLICY_ROWS,
        embedder=MockHashEmbedder(),
        mode="hybrid",
        location=str(loc),
        collection_prefix="persist",
    )
    hits = await idx.search("外观高度模仿知名品牌", PolicySearchFilters(), top_k=5, effective_only=True)
    assert hits and all(h.status == "EFFECTIVE" for h in hits)
    # qdrant-client path= 在目录下落库（collection 持久化产物存在，非空内存库）
    assert loc.exists() and any(loc.iterdir())
