"""RAG 检索行为验收测试 —— 真 Chroma（进程内库）+ 真 BGE（生产同款编码器）。"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import pytest
from helpers import bge_model_cached

from pra.domain.measurement import DEFAULT_EVIDENCE_WEIGHT
from pra.tools.base import ToolContext
from pra.tools.case_search.tool import (
    CaseSearchArgs,
    CaseSearchFilters,
    CaseSearchTool,
)
from pra.tools.policy_search.tool import (
    PolicySearchArgs,
    PolicySearchFilters,
    PolicySearchTool,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODEL_CACHE = os.environ.get(
    "PRA_RAG2_MODEL_CACHE", str(_REPO_ROOT / ".cache" / "model_cache")
)

_RRF_K = 60.0
_RRF_UPPER_BOUND = 2.0 / _RRF_K

_UNREACHABLE_HOST = "127.0.0.1"
_UNREACHABLE_PORT = 59999


def _rag_deps_importable() -> bool:
    """rag extra 是否可 import（用 ``find_spec`` 探测顶层名，不 import 模块）。"""
    return all(
        importlib.util.find_spec(name) is not None
        for name in ("chromadb", "llama_index", "fastembed", "jieba", "bm25s")
    )


_REAL_READY = _rag_deps_importable() and bge_model_cached(Path(_MODEL_CACHE))

pytestmark = pytest.mark.skipif(
    not _REAL_READY,
    reason=(
        "真 Chroma + 真 BGE 未就绪（缺 rag extra，或 BGE 模型未缓存到 "
        f"{_MODEL_CACHE}）→ 整模块跳过；装上 `uv sync --extra rag --extra observability` "
        "并预热模型后重跑"
    ),
)


@pytest.fixture(scope="module")
def embedder() -> Any:
    """生产同款 BGE 编码器（只读本地缓存；模块级复用）。"""
    from pra.rag.embedding import production_embedder

    return production_embedder(cache_dir=_MODEL_CACHE)


@pytest.fixture(scope="module")
def policy_rows() -> Any:
    from pra.rag.corpus import load_policies

    return load_policies()[0]


@pytest.fixture(scope="module")
def case_rows() -> Any:
    from pra.rag.corpus import load_cases

    return load_cases()[0]


def _build_index(kind: str, *, rows: Any, embedder: Any, prefix: str) -> Any:
    """装配一个 chroma 索引（恒 hybrid）：进程内库 + 真编码器（prefix 隔离 collection）。"""
    from pra.rag.chroma_store import ChromaConfig
    from pra.rag.factory import build_case_index, build_policy_index

    build = build_policy_index if kind == "policy" else build_case_index
    config = ChromaConfig(ephemeral=True, collection_prefix=prefix)
    return build(rows=rows, embedding_model=embedder, config=config)


@pytest.fixture(scope="module")
def policy_hybrid(policy_rows: Any, embedder: Any) -> Any:
    return _build_index(
        "policy", rows=policy_rows, embedder=embedder, prefix="pytest_rag_ret_pol_hyb"
    )


@pytest.fixture(scope="module")
def case_hybrid(case_rows: Any, embedder: Any) -> Any:
    return _build_index(
        "case", rows=case_rows, embedder=embedder, prefix="pytest_rag_ret_case_hyb"
    )


def _ranked(index: Any, kind: str, path: str, query: str, filters: Any, *,
            effective_only: bool = False, top_k: int | None = None) -> list[tuple[int, float]]:
    """直调索引内部单路检索器 → ``[(行索引, 该路原始分)]``。"""
    from pra.rag.deps import llama
    from pra.rag.index import _case_candidates, _policy_candidates

    if kind == "policy":
        candidates = _policy_candidates(index._rows, filters, effective_only)
    else:
        candidates = _case_candidates(index._rows, filters)
    if not candidates:
        return []
    bundle = llama().QueryBundle(query_str=query)
    if path == "bm25":
        ranked = index._rank_bm25(index._sub_context(candidates), bundle, len(candidates))
    elif kind == "policy":
        ranked = index._rank_vector(
            index._full_context(), bundle, index._filters_of(filters, effective_only)
        )
    else:
        ranked = index._rank_vector(index._full_context(), bundle, index._filters_of(filters))
    return ranked if top_k is None else ranked[:top_k]


def _policy_ids(index: Any, ranked: list[tuple[int, float]]) -> list[str]:
    return [index._rows[i].clause_id for i, _ in ranked]


def _case_ids(index: Any, ranked: list[tuple[int, float]]) -> list[str]:
    return [index._rows[i].case_id for i, _ in ranked]


def _tool_ctx() -> ToolContext:
    from pra.domain.models import Budget

    return ToolContext(run_id="rag-retrieval-test", case_id="CASE_TEST_RAG", budget=Budget())


def test_reseed_on_existing_collection_overwrites_same_id(
    policy_rows: Any, embedder: Any
) -> None:
    """同一 collection 上重建必须覆盖同 id 的旧记录（写前先删）。"""
    from pra.rag.chroma_store import ChromaConfig, _node_id, make_chroma_client
    from pra.rag.factory import build_policy_index

    prefix = "pytest_rag_ret_idem"
    base = ChromaConfig(ephemeral=True, collection_prefix=prefix)
    shared = ChromaConfig(
        ephemeral=True, collection_prefix=prefix, client=make_chroma_client(base)
    )
    build_policy_index(rows=policy_rows, embedding_model=embedder, config=shared)
    marker = "重建改写标记字"
    patched = [policy_rows[0].model_copy(update={"text": marker}), *policy_rows[1:]]
    second = build_policy_index(
        rows=patched, embedding_model=embedder, config=shared
    )
    expected_ids = [_node_id(second.collection_name, r.clause_id) for r in patched]
    collection = shared.client.get_collection(second.collection_name)
    assert set(collection.get(include=[])["ids"]) == set(expected_ids)
    stored = collection.get(ids=[expected_ids[0]])["documents"]
    assert stored and marker in stored[0], f"同 id 未被覆盖：{stored}"


def test_reseed_on_shrunk_corpus_purges_stale_node_ids(
    policy_rows: Any, embedder: Any
) -> None:
    """语料缩减后重建：collection 内 id 集合与当前语料严格一致，残留旧 id 被清除。"""
    from pra.rag.chroma_store import ChromaConfig, _node_id, make_chroma_client
    from pra.rag.factory import build_policy_index

    prefix = "pytest_rag_ret_stale"
    base = ChromaConfig(ephemeral=True, collection_prefix=prefix)
    shared = ChromaConfig(
        ephemeral=True, collection_prefix=prefix, client=make_chroma_client(base)
    )
    full = build_policy_index(
        rows=policy_rows, embedding_model=embedder, config=shared
    )
    shrunk_rows = policy_rows[:-1]
    shrunk = build_policy_index(
        rows=shrunk_rows, embedding_model=embedder, config=shared
    )
    full_ids = {_node_id(full.collection_name, r.clause_id) for r in policy_rows}
    expected_ids = {_node_id(shrunk.collection_name, r.clause_id) for r in shrunk_rows}
    assert len(full_ids) == len(expected_ids) + 1, "前置：全量应比缩减多一 node"
    collection = shared.client.get_collection(shrunk.collection_name)
    stored_ids = set(collection.get(include=[])["ids"])
    assert stored_ids == expected_ids, f"残留旧 node id：{sorted(stored_ids - expected_ids)}"
    assert collection.count() == len(shrunk_rows)


def test_empty_corpus_raises_instead_of_building_empty_index(embedder: Any) -> None:
    """空语料必须显式失败（``ValueError``），不得静默建成空索引。"""
    from pra.rag.chroma_store import ChromaConfig
    from pra.rag.factory import build_case_index, build_policy_index

    config = ChromaConfig(ephemeral=True)
    with pytest.raises(ValueError):
        build_policy_index(rows=[], embedding_model=embedder, config=config)
    with pytest.raises(ValueError):
        build_case_index(rows=[], embedding_model=embedder, config=config)


async def test_bm25_recalls_expected_policy_clause(policy_hybrid: Any) -> None:
    """BM25 路能按词面把「该命中的条款」召回进 Top-K。"""
    ranked = _ranked(
        policy_hybrid, "policy", "bm25",
        "无品牌授权，鞋靴整体外观高度模仿知名品牌在售款",
        PolicySearchFilters(), effective_only=True, top_k=5,
    )
    ids = _policy_ids(policy_hybrid, ranked)
    assert "POLICY_1.4_v1_c1" in ids, f"BM25 未召回预期条款；实际 Top-5={ids}"


async def test_bm25_no_global_tokenizer_patch(policy_hybrid: Any) -> None:
    """BM25 分词由检索器自持，不得被全局替换。"""
    import bm25s

    from pra.rag import bm25 as bm25_mod

    ranked = _ranked(
        policy_hybrid, "policy", "bm25", "外观高度模仿知名品牌",
        PolicySearchFilters(), effective_only=True, top_k=3,
    )
    assert ranked, "BM25 检索应有命中（空结果会让本用例失去意义）"
    assert not hasattr(bm25_mod, "_TOKENIZER_LOCK")
    assert not hasattr(bm25_mod, "bm25_tokenizer_context")
    assert bm25s.tokenize.__module__ == "bm25s.tokenization", (
        f"bm25s.tokenize 被替换：{bm25s.tokenize!r}"
    )


async def test_vector_recalls_semantically_matching_clause(policy_hybrid: Any) -> None:
    """Vector 路能按语义把「措辞不同但同义」的条款召回进 Top-K。"""
    ranked = _ranked(
        policy_hybrid, "policy", "vector",
        "商品把 PU 材质宣称为真皮，属于成分虚假宣传",
        PolicySearchFilters(), effective_only=True, top_k=5,
    )
    ids = _policy_ids(policy_hybrid, ranked)
    assert "POLICY_2.3_v1_c1" in ids, f"Vector 未召回语义相关条款；实际 Top-5={ids}"


async def test_hybrid_scores_are_rrf_fusion_not_similarity(case_hybrid: Any) -> None:
    """hybrid 的 ``retrieval_score`` 是 RRF 融合分，不是语义相似度。"""
    hits = await case_hybrid.search(
        "无品牌外观高度模仿知名品牌，商家多次改标题重上架",
        CaseSearchFilters(),
        5,
    )
    scores = [h.retrieval_score for h in hits]
    assert scores, "hybrid 检索必须返回命中"
    assert all(0.0 < s <= _RRF_UPPER_BOUND + 1e-9 for s in scores), (
        f"RRF 分必须落在 (0, {_RRF_UPPER_BOUND:.4f}]，实际={scores}"
    )
    assert scores[0] > 1.0 / _RRF_K, (
        f"Top-1 RRF 分 {scores[0]} 未超过单路首位贡献 {1.0 / _RRF_K:.4f} —— 融合疑似退化成单路"
    )


async def test_vector_scores_are_similarity_scale_unlike_rrf(case_hybrid: Any) -> None:
    """vector 路的分与 hybrid 的 RRF 分量纲不同。"""
    ranked = _ranked(
        case_hybrid, "case", "vector",
        "无品牌外观高度模仿知名品牌，商家多次改标题重上架",
        CaseSearchFilters(), top_k=5,
    )
    scores = [score for _, score in ranked]
    assert scores, "vector 检索必须返回命中"
    assert max(scores) > _RRF_UPPER_BOUND, (
        f"vector 的库口径分应远大于 RRF 上界 {_RRF_UPPER_BOUND:.4f}，实际={scores}"
    )


async def test_policy_effective_only_excludes_expired_clause(
    policy_hybrid: Any, policy_rows: Any
) -> None:
    """``effective_only`` 语义：True 时 EXPIRED 条款不得出现，False 时可出现。"""
    query = "禁止销售仿冒、假冒注册商标的商品"
    top_k = len(policy_rows)

    loose = _ranked(policy_hybrid, "policy", "bm25", query, PolicySearchFilters(),
                    effective_only=False, top_k=top_k)
    assert "POLICY_1.1_v1_c1" in _policy_ids(policy_hybrid, loose), (
        "effective_only=False 时失效条款应可被召回（对照基准）"
    )

    strict = _ranked(policy_hybrid, "policy", "bm25", query, PolicySearchFilters(),
                     effective_only=True, top_k=top_k)
    assert "POLICY_1.1_v1_c1" not in _policy_ids(policy_hybrid, strict), (
        "effective_only=True 时 EXPIRED 条款不得出现"
    )
    assert all(policy_hybrid._rows[i].status == "EFFECTIVE" for i, _ in strict), (
        "effective_only=True 结果必须全为 EFFECTIVE"
    )


async def test_policy_category_filter_keeps_full_category_clauses(
    policy_hybrid: Any, policy_rows: Any
) -> None:
    """``category`` 过滤含「全类目」语义：指定具体类目时，全类目条款也应保留。"""
    ranked = _ranked(
        policy_hybrid, "policy", "bm25", "改标题、描述后重新上架规避审核",
        PolicySearchFilters(category="箱包/女包"), effective_only=True, top_k=len(policy_rows),
    )
    cats = {policy_hybrid._rows[i].category for i, _ in ranked}
    assert cats <= {"箱包/女包", "全类目"}, f"category 过滤漏了全类目语义：{cats}"
    assert "POLICY_4.2_v2_c1" in _policy_ids(policy_hybrid, ranked), "全类目条款应被保留"


async def test_policy_risk_type_filter_is_overlap_not_equality(
    policy_hybrid: Any, policy_rows: Any
) -> None:
    """``risk_type`` 过滤是**交叠非空**，不是相等匹配。"""
    ranked = _ranked(
        policy_hybrid, "policy", "bm25", "材质成分虚假宣传 PU 冒充实皮",
        PolicySearchFilters(risk_type=["FALSE_CLAIM"]), effective_only=True,
        top_k=len(policy_rows),
    )
    assert ranked, "risk_type 过滤后应仍有命中"
    assert all(
        {t.value for t in policy_hybrid._rows[i].risk_type} & {"FALSE_CLAIM"}
        for i, _ in ranked
    ), "存在与过滤类型无交集的条款"
    assert "POLICY_2.3_v1_c1" in _policy_ids(policy_hybrid, ranked), (
        "多风险类型条款应被交叠匹配命中"
    )


async def test_case_filters_apply_before_retrieval(case_hybrid: Any, case_rows: Any) -> None:
    """case 的 ``category`` + ``risk_type`` 过滤与检索**组合**：过滤收窄候选后仍要召回预期先例。"""
    hits = await case_hybrid.search(
        "无品牌标识外观高度模仿，商家多次改标题重上架",
        CaseSearchFilters(category="女鞋/运动鞋", risk_type=["EVASION_PATTERN"]),
        len(case_rows),
    )
    ids = {h.case_id for h in hits}
    expected = {
        r.case_id
        for r in case_rows
        if r.category == "女鞋/运动鞋" and "EVASION_PATTERN" in {t.value for t in r.risk_type}
    }
    assert ids, "带过滤的 case 检索应仍有命中"
    assert ids <= expected, f"过滤结果出现越界先例：{ids - expected}"
    assert "RAG_CASE_0001" in ids, "组合过滤下预期先例被漏召回"


async def test_filter_plus_retrieval_no_silent_recall_loss(policy_hybrid: Any) -> None:
    """**带过滤时**的检索不得静默漏召回。"""
    ranked = _ranked(
        policy_hybrid, "policy", "bm25", "鞋靴整体外观高度模仿知名品牌且无授权",
        PolicySearchFilters(category="女鞋/运动鞋"), effective_only=True, top_k=10,
    )
    assert "POLICY_1.4_v1_c1" in _policy_ids(policy_hybrid, ranked), (
        "带 category 过滤时预期的类目条款被漏召回"
    )


async def test_policy_where_pushdown_equals_python_candidates(
    policy_hybrid: Any, policy_rows: Any
) -> None:
    """向量路下推的 ``where`` 与 BM25 路的 Python 谓词**必须给出同一打分域**。"""
    from pra.rag.index import _policy_candidates

    for category in (None, "女鞋/运动鞋", "箱包/女包", "全类目", "不存在的类目"):
        for effective_only in (False, True):
            for risks in (None, ["FALSE_CLAIM"], ["EVASION_PATTERN", "FALSE_CLAIM"]):
                filters = PolicySearchFilters(category=category, risk_type=risks)
                expected = {
                    policy_rows[i].clause_id
                    for i in _policy_candidates(policy_rows, filters, effective_only)
                }
                ranked = _ranked(
                    policy_hybrid, "policy", "vector",
                    "外观高度模仿知名品牌 / 材质虚假宣传 / 改标题重上架",
                    filters, effective_only=effective_only, top_k=len(policy_rows),
                )
                assert set(_policy_ids(policy_hybrid, ranked)) == expected, (
                    "where 下推与 Python 谓词不一致（带过滤的静默漏召回）："
                    f"category={category!r} effective_only={effective_only} risk_type={risks}"
                )


async def test_case_where_pushdown_equals_python_candidates(
    case_hybrid: Any, case_rows: Any
) -> None:
    """case 侧同理；差异点：case **没有** ``effective_only``，``category`` 是**精确相等**。"""
    from pra.rag.index import _case_candidates

    for category in (None, "女鞋/运动鞋", "箱包/女包", "不存在的类目"):
        for risks in (None, ["EVASION_PATTERN"], ["FALSE_CLAIM", "EVASION_PATTERN"]):
            filters = CaseSearchFilters(category=category, risk_type=risks)
            expected = {
                case_rows[i].case_id for i in _case_candidates(case_rows, filters)
            }
            ranked = _ranked(
                case_hybrid, "case", "vector",
                "无品牌外观高度模仿，商家多次改标题重上架",
                filters, top_k=len(case_rows),
            )
            assert set(_case_ids(case_hybrid, ranked)) == expected, (
                "case 侧 where 下推与 Python 谓词不一致："
                f"category={category!r} risk_type={risks}"
            )


async def test_index_search_returns_hit_models(policy_hybrid: Any, case_hybrid: Any) -> None:
    """索引检索返回的是工具契约的 Hit 模型（而非内部 node / dict）。"""
    from pra.tools.case_search.tool import CaseHit
    from pra.tools.policy_search.tool import PolicyClauseHit

    p_hits = await policy_hybrid.search("外观模仿", PolicySearchFilters(), 3, True)
    c_hits = await case_hybrid.search("外观模仿", CaseSearchFilters(), 3)
    assert p_hits and all(isinstance(h, PolicyClauseHit) for h in p_hits)
    assert c_hits and all(isinstance(h, CaseHit) for h in c_hits)


async def test_policy_search_tool_produces_policy_ref_evidence(policy_hybrid: Any) -> None:
    """Policy 两层：``ChromaPolicyIndex.search`` → ``PolicyClauseHit``；工具 → ``POLICY_REF``。"""
    tool = PolicySearchTool(index=policy_hybrid)
    result = await tool.call(
        PolicySearchArgs(query="外观高度模仿知名品牌设计", top_k=3, effective_only=True),
        _tool_ctx(),
    )
    assert result.ok and result.hits, "注入真索引后工具应拿到命中"
    assert all(type(h).__name__ == "PolicyClauseHit" for h in result.hits)
    evidences = tool.to_evidence(result)
    assert evidences, "命中应转为证据"
    for ev, hit in zip(evidences, result.hits):
        assert ev.type == "POLICY_REF"
        assert ev.source == "PolicySearchTool"
        assert ev.weight == DEFAULT_EVIDENCE_WEIGHT
        assert ev.ref_id == hit.clause_id


async def test_case_search_tool_produces_case_precedent_evidence(case_hybrid: Any) -> None:
    """Case 两层：``ChromaCaseIndex.search`` → ``CaseHit``；工具 → ``CASE_PRECEDENT``。"""
    tool = CaseSearchTool(index=case_hybrid)
    result = await tool.call(
        CaseSearchArgs(query="无品牌 + 高相似 + 商家多次重上架", top_k=3),
        _tool_ctx(),
    )
    assert result.ok and result.hits, "注入真索引后工具应拿到命中"
    assert all(type(h).__name__ == "CaseHit" for h in result.hits)
    assert all(h.retrieval_score > 0.0 for h in result.hits)
    evidences = tool.to_evidence(result)
    assert evidences, "命中应转为证据"
    for ev, hit in zip(evidences, result.hits):
        assert ev.type == "CASE_PRECEDENT"
        assert ev.source == "CaseSearchTool"
        assert ev.weight == hit.retrieval_score
        assert ev.ref_id == hit.case_id


async def test_unreachable_chroma_host_raises_instead_of_silent_empty(
    policy_rows: Any, embedder: Any
) -> None:
    """真基础设施异常必须上抛，而不是静默返回空结果。"""
    from pra.rag.chroma_store import ChromaConfig
    from pra.rag.factory import build_policy_index

    config = ChromaConfig(host=_UNREACHABLE_HOST, port=_UNREACHABLE_PORT)
    try:
        index = build_policy_index(
            rows=policy_rows, embedding_model=embedder, config=config
        )
        await index.search("外观模仿", PolicySearchFilters(), 5, True)
    except Exception:  # noqa: BLE001
        return
    pytest.fail("不可达 Chroma 未被上抛（构建与检索都成功）→ 疑似静默降级为空结果")
