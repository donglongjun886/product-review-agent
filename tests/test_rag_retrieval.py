"""RAG 检索行为验收测试 —— 真 Chroma（进程内库）+ 真 BGE（生产同款编码器）。

覆盖「检索行为」这一层（此前被整体删除、只剩 corpus 完整性与装配契约的部分）：建库重建幂等 /
BM25 召回 / Vector 召回 / Hybrid 的 RRF 融合正确性 / metadata filter（含「过滤器 + 检索」组合的红线场景）/
PolicyHit·CaseHit → Evidence 两层契约 / 真基础设施异常上抛 / RAG 工具可被正常调用。

运行面（用户拍板）：真 Chroma **进程内库**（``ChromaConfig(ephemeral=True)`` →
``chromadb.EphemeralClient``，无需 Docker 服务端）+ 生产同款编码器
``production_embedder(cache_dir=<仓库 .cache/model_cache>)``。未就绪（缺 rag extra / 模型未缓存）
即**整模块 skip**：就绪判定只用 ``importlib.util.find_spec`` + 只读文件系统探测
（``helpers.bge_model_cached``），**收集期绝不 import chromadb / fastembed / llama_index / jieba /
bm25s**（否则会污染 ``tests/test_rag_default_path_no_extra.py`` 的子进程 ``sys.modules`` 断言）。

术语红线：hybrid 的 ``retrieval_score`` 是 **RRF 融合分**（``Σ 1/(60+rank)``，上界
``2/60 ≈ 0.0333``），**不是语义相似度**；vector 是库口径 ``exp(-distance)``（⊂ ``(0, 1]``），两者量纲不同。
"""

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
#: 真模型缓存目录：优先 PRA_RAG2_MODEL_CACHE，缺省为仓库内 .cache/model_cache（只读，不下载）。
_MODEL_CACHE = os.environ.get(
    "PRA_RAG2_MODEL_CACHE", str(_REPO_ROOT / ".cache" / "model_cache")
)

#: RRF 常数 k=60（Cormack 2009）：单路首位贡献 1/60，两路都排首位 2/60。
_RRF_K = 60.0
_RRF_UPPER_BOUND = 2.0 / _RRF_K

#: 一个本机必然无服务端监听的端口（用于「不可达 → 必须上抛」用例）。
_UNREACHABLE_HOST = "127.0.0.1"
_UNREACHABLE_PORT = 59999


def _rag_deps_importable() -> bool:
    """rag extra 是否可 import —— 用 ``find_spec`` 探测顶层名，**不 import 任何模块**。"""
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


# ---------------------------------------------------------------------------
# 装配夹具（真索引：进程内 EphemeralClient + 真编码器；模块级复用）
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def embedder() -> Any:
    """生产同款 BGE 编码器（只读本地缓存；模块级复用，避免重复加载模型）。"""
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
    """直调索引内部单路检索器 → ``[(行索引, 该路原始分)]``。

    索引层没有模式开关（``search`` 恒 hybrid）；单路只作验证 / 调试入口，这里直接调用两路
    私有检索器，不构成对外参数面（BM25 / Vector 是 hybrid 的内部路径，不是可选模式）。
    """
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
        # case 无 effective_only 语义（其 ``_filters_of`` 只收 filters）
        ranked = index._rank_vector(index._full_context(), bundle, index._filters_of(filters))
    return ranked if top_k is None else ranked[:top_k]


def _policy_ids(index: Any, ranked: list[tuple[int, float]]) -> list[str]:
    return [index._rows[i].clause_id for i, _ in ranked]


def _case_ids(index: Any, ranked: list[tuple[int, float]]) -> list[str]:
    return [index._rows[i].case_id for i, _ in ranked]


def _tool_ctx() -> ToolContext:
    from pra.domain.models import Budget

    return ToolContext(run_id="rag-retrieval-test", case_id="CASE_TEST_RAG", budget=Budget())


# ---------------------------------------------------------------------------
# 0. 建库写入（delete-then-add）
# ---------------------------------------------------------------------------


def test_reseed_on_existing_collection_overwrites_same_id(
    policy_rows: Any, embedder: Any
) -> None:
    """同一 collection 上重建必须覆盖同 id 的旧记录（写前先删）。

    断言：改写首条 policy 的正文后重建，库里该 node id 的 document 变成新正文；库内 id 集合仍
    等于语料算出的 id 集合（同语料 → 同 id）。意义：库的 ``add`` 对**已存在的 id 静默跳过**
    （既不覆盖也不抛错，chromadb 1.5.9 实测：重复 ``add`` 后 ``count`` 仍为 1）⇒ 删掉写前的
    ``delete_nodes`` 这条断言必失败，库里留下的是改写前的旧正文。断言必须直接读库：BM25 路读
    内存 node、不经库，用检索观察不到。
    """
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
    """语料缩减后重建：collection 内 id 集合与当前语料严格一致，残留旧 id 被清除。

    断言：先按全量 policy 建库、再用去掉末行的语料重建，collection 的 id 集合 == 当前语料算出的
    id 集合，且 ``count`` == 当前行数。意义：不清理差集时被删行的 node 会永久残留，占掉向量路
    ``similarity_top_k`` 的名额（``_to_ranked`` 会把它丢弃）。
    """
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
    """空语料必须显式失败（``ValueError``），不得静默建成空索引。

    断言：``rows=[]`` 构造 policy / case 索引均抛 ``ValueError``。意义：静默不建库会让「检索不到」
    与「没有语料」混为一谈（旧行为是 ``collection_name=""`` / ``_dim=0`` 的空壳索引），本仓红线是
    数据不可用时显式失败，不静默降级。
    """
    from pra.rag.chroma_store import ChromaConfig
    from pra.rag.factory import build_case_index, build_policy_index

    config = ChromaConfig(ephemeral=True)
    with pytest.raises(ValueError):
        build_policy_index(rows=[], embedding_model=embedder, config=config)
    with pytest.raises(ValueError):
        build_case_index(rows=[], embedding_model=embedder, config=config)


# ---------------------------------------------------------------------------
# 1 / 2. BM25 与 Vector 的召回
# ---------------------------------------------------------------------------


async def test_bm25_recalls_expected_policy_clause(policy_hybrid: Any) -> None:
    """BM25 路能按词面把「该命中的条款」召回进 Top-K。

    断言：查询「无品牌授权，鞋靴整体外观高度模仿知名品牌在售款」的 Top-5 含
    ``POLICY_1.4_v1_c1``（鞋靴外观高度模仿……条款）。意义：证明 BM25 路不是空转（此前被删的
    用例正是这一层）—— 分词 / 索引 / 召回任一环节坏掉，目标条款都会掉出 Top-K。
    """
    ranked = _ranked(
        policy_hybrid, "policy", "bm25",
        "无品牌授权，鞋靴整体外观高度模仿知名品牌在售款",
        PolicySearchFilters(), effective_only=True, top_k=5,
    )
    ids = _policy_ids(policy_hybrid, ranked)
    assert "POLICY_1.4_v1_c1" in ids, f"BM25 未召回预期条款；实际 Top-5={ids}"


async def test_bm25_no_global_tokenizer_patch(policy_hybrid: Any) -> None:
    """BM25 分词是检索器自持的（``bm25s.tokenize`` 归库所有，不得再被全局替换）。

    断言：跑一次真实检索后，``rag/bm25.py`` 不再暴露全局替换件（``_TOKENIZER_LOCK`` /
    ``bm25_tokenizer_context``），且 ``bm25s.tokenize`` 仍指向库自身实现。意义：把「删掉猴补丁
    这条决定」钉死 —— 重新引入全局替换会让本用例立刻变红。
    """
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
    """Vector 路能按语义把「措辞不同但同义」的条款召回进 Top-K。

    断言：查询「商品把 PU 材质宣称为真皮，属于成分虚假宣传」的 Top-5 含 ``POLICY_2.3_v1_c1``
    （材质/成分虚假声称）。意义：目标条款正文与查询措辞并不逐字相同，只有语义编码生效才会召回
    —— 用真 BGE 验证向量路真的做了语义检索，而非退化到词面匹配。
    """
    ranked = _ranked(
        policy_hybrid, "policy", "vector",
        "商品把 PU 材质宣称为真皮，属于成分虚假宣传",
        PolicySearchFilters(), effective_only=True, top_k=5,
    )
    ids = _policy_ids(policy_hybrid, ranked)
    assert "POLICY_2.3_v1_c1" in ids, f"Vector 未召回语义相关条款；实际 Top-5={ids}"


# ---------------------------------------------------------------------------
# 3. Hybrid / RRF 融合正确性
# ---------------------------------------------------------------------------


async def test_hybrid_scores_are_rrf_fusion_not_similarity(case_hybrid: Any) -> None:
    """hybrid 的 ``retrieval_score`` 是 RRF 融合分（``Σ 1/(60+rank)``），不是语义相似度。

    断言：命中分全 ⊂ ``(0, 2/60]``（上界 ≈ 0.0333），且 Top-1 > ``1/60``。意义：``1/60`` 是
    「只被单路排到首位」的贡献，Top-1 严格大于它即证明该文档被**两路同时**排到靠前名次（融合
    确实发生）；若某处把检索分误当语义相似度返回，分数会落到 0.5~0.9 量级、直接越过上界被拦下。
    """
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
    """对照反证：vector 路的分是库口径 ``exp(-distance)``（⊂ ``(0, 1]``），量纲与 hybrid 的 RRF 分完全不同。

    断言：同一查询在 vector 路的 Top 分 **大于** RRF 上界 ``2/60``。意义：与上一条互为反证
    —— 若 hybrid 误用了 vector 的相似度分（或反之），两条用例必有一条失败；同时钉住「不得把
    RRF 分称作语义相似度」这条术语红线。vector 路经内部检索器直调（索引层无模式开关）。
    """
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


# ---------------------------------------------------------------------------
# 4. metadata filter 正确性
# ---------------------------------------------------------------------------


async def test_policy_effective_only_excludes_expired_clause(
    policy_hybrid: Any, policy_rows: Any
) -> None:
    """``effective_only`` 语义：True 时 EXPIRED 条款不得出现，False 时可出现。

    断言：对指向失效条款正文的查询「禁止销售仿冒、假冒注册商标的商品」（= ``POLICY_1.1_v1_c1``
    正文），``effective_only=False`` 的返回**含**该 EXPIRED 条款，``effective_only=True`` 的返回
    **不含**它且全部为 EFFECTIVE。意义：用一个「本可命中」的失效条款作对照，才能证明它消失确由
    ``effective_only`` 造成，而非查询根本没召回（把过滤与召回解耦，避免空断言）。
    """
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
    """``category`` 过滤含「全类目」语义：指定具体类目时，全类目条款也应保留。

    断言：``category="箱包/女包"`` 时返回条款 category 全 ∈ {箱包/女包, 全类目}，且含
    ``POLICY_4.2_v2_c1``（全类目 + 改标题重上架 —— 查询正指向它）。意义：把「全类目」当普通类目
    精确匹配会把平台通用条款整批漏掉，本断言钉住 include-全类目 的过滤语义。
    """
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
    """``risk_type`` 过滤是**交叠非空**，不是相等匹配。

    断言：``risk_type=["FALSE_CLAIM"]`` 时返回条款的 risk_type 均与 {FALSE_CLAIM} 有交集，且含
    ``POLICY_2.3_v1_c1``（risk_type = [FALSE_CLAIM, FIELD_CONFLICT]，多标一个也要命中）。意义：
    相等匹配会漏掉「多风险类型」条款 —— 正是「漏召回只在带过滤时暴露」的典型场景。
    """
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
    """case 的 ``category`` + ``risk_type`` 过滤与检索**组合**：过滤收窄候选后仍要召回预期先例。

    断言：``category="女鞋/运动鞋"`` + ``risk_type=["EVASION_PATTERN"]`` 时，返回先例全落在此类目
    ∩ 风险类型集合内（用 corpus 现算期望集合），且含 ``RAG_CASE_0001``。意义：这是「过滤器 + 检索」
    组合红线 —— 过滤把候选收窄后，若向量路候选集处理有误（历史 bug 即「非候选行抢占名额 → 结果
    静默变子集甚至空」），预期先例会消失；本断言把它钉死。
    """
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
    """红线：**带过滤时**的检索不得静默漏召回（漏召回只在带过滤时暴露）。

    断言：``category="女鞋/运动鞋"`` + ``effective_only=True`` + 指向该条款的查询下，Top-K 仍含
    ``POLICY_1.4_v1_c1``（女鞋/运动鞋且 EFFECTIVE）。意义：过滤走**库侧 ``where`` 下推 + 同一套语义的
    Python 谓词**（BM25 路），若两处语义错位，目标条款会被静默剔除；本用例专门覆盖「过滤 + 检索」这一
    red-line 组合（与无过滤的召回用例互为补充）。
    """
    ranked = _ranked(
        policy_hybrid, "policy", "bm25", "鞋靴整体外观高度模仿知名品牌且无授权",
        PolicySearchFilters(category="女鞋/运动鞋"), effective_only=True, top_k=10,
    )
    assert "POLICY_1.4_v1_c1" in _policy_ids(policy_hybrid, ranked), (
        "带 category 过滤时预期的类目条款被漏召回"
    )


# ---------------------------------------------------------------------------
# 4b. 过滤下推的等价性守护（向量路 ``where`` ⟷ BM25 路 Python 谓词）
# ---------------------------------------------------------------------------


async def test_policy_where_pushdown_equals_python_candidates(
    policy_hybrid: Any, policy_rows: Any
) -> None:
    """向量路下推的 ``where`` 与 BM25 路的 Python 谓词**必须给出同一打分域**。

    断言：遍历 ``category``（无 / 真实类目 / 全类目 / 不存在）× ``effective_only`` × ``risk_type``
    （无 / 单值 / 双值）的全组合，**向量路**取 ``top_k=len(rows)`` 的命中集合
    **逐组等于** ``_policy_candidates`` 算出的候选集合。

    意义：向量路把过滤**下推给 Chroma**（``where``），BM25 路只能在 Python 候选集上打分
    （``bm25s`` 内存索引没有 ``where``）—— 两处语义一旦不一致就会「带过滤时静默漏召回」，
    而漏召回**只在带过滤时暴露**（无过滤时全绿）。这是这条下推唯一的守护：
    ``risk_type`` 的「交叠非空」靠 ``rt_*`` 整数键 + ``$or`` 表达（Chroma 无数组交叠操作符），
    ``category`` 的三态靠 ``$in [值, 全类目]``；键名与写入侧不同源、或语义错一处，都会有某组集合不等。

    ``top_k`` 取语料行数 ⇒ 命中全部返回、不被截断，集合比较才有意义。
    （``category="不存在的类目"`` 一组候选为空，覆盖的是「空候选短路」，属弱校验，留作边界。）
    """
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
    """case 侧同理；差异点：case **没有** ``effective_only``，``category`` 是**精确相等**。

    断言：``category``（无 / 真实类目 / 不存在）× ``risk_type``（无 / 单值 / 双值）全组合下，
    向量路取 ``top_k=len(rows)`` 的命中集合逐组等于 ``_case_candidates`` 的候选集合。
    意义：case 的 ``category`` 若被误写成 policy 的三态（``$in [值, 全类目]``），带过滤时会**多召回**
    全类目先例 —— 本用例把它钉死（预期集合按 case 自己的精确相等语义现算）。
    """
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


# ---------------------------------------------------------------------------
# 5. 两层契约：索引 → Hit；工具 → Evidence
# ---------------------------------------------------------------------------


async def test_index_search_returns_hit_models(policy_hybrid: Any, case_hybrid: Any) -> None:
    """索引检索返回的是工具契约的 Hit 模型（而非内部 node / dict）。

    断言：``ChromaPolicyIndex.search`` 返回 ``PolicyClauseHit``；``ChromaCaseIndex.search`` 返回
    ``CaseHit``。意义：工具层只认 Hit 契约，索引若透出内部结构会让上层协议形同虚设。
    分数语义不在此锁定 —— hybrid 的 RRF 分口径见
    :func:`test_hybrid_scores_are_rrf_fusion_not_similarity`。
    """
    from pra.tools.case_search.tool import CaseHit
    from pra.tools.policy_search.tool import PolicyClauseHit

    p_hits = await policy_hybrid.search("外观模仿", PolicySearchFilters(), 3, True)
    c_hits = await case_hybrid.search("外观模仿", CaseSearchFilters(), 3)
    assert p_hits and all(isinstance(h, PolicyClauseHit) for h in p_hits)
    assert c_hits and all(isinstance(h, CaseHit) for h in c_hits)


async def test_policy_search_tool_produces_policy_ref_evidence(policy_hybrid: Any) -> None:
    """Policy 两层：``ChromaPolicyIndex.search`` → ``PolicyClauseHit``；工具 → ``POLICY_REF``。

    断言：注入真索引的 ``PolicySearchTool.call`` 拿到 ≥1 个 ``PolicyClauseHit``；``to_evidence``
    每条证据 type=``POLICY_REF``、source=``PolicySearchTool``、weight==DEFAULT_EVIDENCE_WEIGHT、
    ref_id==命中条款的 clause_id。意义：「RAG 怎么检索」与「Evidence 怎么产生」是两个层次，本用例
    同时钉住命中类型与证据映射（可追溯引用 = ref_id 必填、政策依据 = 统一默认权重）。
    """
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
    """Case 两层：``ChromaCaseIndex.search`` → ``CaseHit``；工具 → ``CASE_PRECEDENT``。

    断言：命中均为 ``CaseHit``（检索分为 hybrid/RRF 的非零正数）；证据 type=``CASE_PRECEDENT``、
    source=``CaseSearchTool``、weight==该 hit 的 ``retrieval_score``、ref_id==case_id。意义：证据
    侧 weight 直接复用检索分（术语：这是检索分，不是语义相似度），两条断言一起把「检索」与
    「证据产生」两层焊住。⚠️ 此处 weight 能原样透传，前提是 ``Evidence.weight`` 已去掉 ``le=1``。
    """
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


# ---------------------------------------------------------------------------
# 6. 真基础设施异常必须上抛
# ---------------------------------------------------------------------------


async def test_unreachable_chroma_host_raises_instead_of_silent_empty(
    policy_rows: Any, embedder: Any
) -> None:
    """真基础设施异常必须上抛，而不是静默返回空结果。

    断言：``ChromaConfig(host=<不可达>, port=<不可达>)`` 下，「构建索引 或 首次 ``search``」必抛
    异常。意义：RAG 不可用时若静默返回 ``[]``，Agent 会把「检索不到」误当「没有相关政策/先例」
    并据此决策 —— 本仓红线是「失败要显式、可观测」（由 tools_node 记 warn failure），绝不静默降级。
    """
    from pra.rag.chroma_store import ChromaConfig
    from pra.rag.factory import build_policy_index

    config = ChromaConfig(host=_UNREACHABLE_HOST, port=_UNREACHABLE_PORT)
    try:
        index = build_policy_index(
            rows=policy_rows, embedding_model=embedder, config=config
        )
        await index.search("外观模仿", PolicySearchFilters(), 5, True)
    except Exception:  # noqa: BLE001 — 任意异常都算「已上抛」，见上 docstring
        return
    pytest.fail("不可达 Chroma 未被上抛（构建与检索都成功）→ 疑似静默降级为空结果")
