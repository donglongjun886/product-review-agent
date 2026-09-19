"""RAG 检索行为验收测试 —— 真 Chroma（进程内库）+ 真 BGE（生产同款编码器）。

覆盖「检索行为」这一层（此前被整体删除、只剩 corpus 完整性与装配契约的部分）：BM25 召回 /
Vector 召回 / Hybrid 的 RRF 融合正确性 / metadata filter（含「过滤器 + 检索」组合的红线场景）/
PolicyHit·CaseHit → Evidence 两层契约 / 真基础设施异常上抛 / RAG 工具可被正常调用。

运行面（用户拍板）：真 Chroma **进程内库**（``ChromaConfig(ephemeral=True)`` →
``chromadb.EphemeralClient``，无需 Docker 服务端）+ 生产同款编码器
``production_embedder(cache_dir=<仓库 .cache/model_cache>)``。未就绪（缺 rag extra / 模型未缓存）
即**整模块 skip**：就绪判定只用 ``importlib.util.find_spec`` + 只读文件系统探测
（``helpers.bge_model_cached``），**收集期绝不 import chromadb / fastembed / llama_index / jieba /
bm25s**（否则会污染 ``tests/test_rag_default_path_no_extra.py`` 的子进程 ``sys.modules`` 断言）。

术语红线：hybrid 的 ``retrieval_score`` 是 **RRF 融合分**（``Σ 1/(60+rank)``，上界
``2/60 ≈ 0.0333``），**不是语义相似度**；vector 才是 ``1 − cosine distance``，两者量纲不同。
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any

import pytest
from helpers import bge_model_cached

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


def _build_index(kind: str, mode: str, *, rows: Any, embedder: Any, prefix: str) -> Any:
    """按 kind/mode 装配一个 chroma 索引：进程内库 + 真编码器（prefix 隔离各自的 collection）。"""
    from pra.rag.chroma_store import ChromaConfig
    from pra.rag.factory import build_case_index, build_policy_index

    build = build_policy_index if kind == "policy" else build_case_index
    config = ChromaConfig(ephemeral=True, collection_prefix=prefix)
    return build(rows=rows, embedding_model=embedder, mode=mode, config=config)


@pytest.fixture(scope="module")
def policy_bm25(policy_rows: Any, embedder: Any) -> Any:
    return _build_index(
        "policy", "bm25", rows=policy_rows, embedder=embedder, prefix="pytest_rag_ret_pol_bm25"
    )


@pytest.fixture(scope="module")
def policy_vector(policy_rows: Any, embedder: Any) -> Any:
    return _build_index(
        "policy", "vector", rows=policy_rows, embedder=embedder, prefix="pytest_rag_ret_pol_vec"
    )


@pytest.fixture(scope="module")
def case_hybrid(case_rows: Any, embedder: Any) -> Any:
    return _build_index(
        "case", "hybrid", rows=case_rows, embedder=embedder, prefix="pytest_rag_ret_case_hyb"
    )


@pytest.fixture(scope="module")
def case_vector(case_rows: Any, embedder: Any) -> Any:
    return _build_index(
        "case", "vector", rows=case_rows, embedder=embedder, prefix="pytest_rag_ret_case_vec"
    )


def _tool_ctx() -> ToolContext:
    from pra.domain.models import Budget

    return ToolContext(run_id="rag-retrieval-test", case_id="CASE_TEST_RAG", budget=Budget())


# ---------------------------------------------------------------------------
# 1 / 2. BM25 与 Vector 的召回
# ---------------------------------------------------------------------------


async def test_bm25_recalls_expected_policy_clause(policy_bm25: Any) -> None:
    """BM25 路能按词面把「该命中的条款」召回进 Top-K。

    断言：查询「无品牌授权，鞋靴整体外观高度模仿知名品牌在售款」的 Top-5 含
    ``POLICY_1.4_v1_c1``（鞋靴外观高度模仿……条款）。意义：证明 BM25 路不是空转（此前被删的
    用例正是这一层）—— 分词 / 索引 / 召回任一环节坏掉，目标条款都会掉出 Top-K。
    """
    hits = await policy_bm25.search(
        "无品牌授权，鞋靴整体外观高度模仿知名品牌在售款",
        PolicySearchFilters(),
        5,
        True,
    )
    ids = [h.clause_id for h in hits]
    assert "POLICY_1.4_v1_c1" in ids, f"BM25 未召回预期条款；实际 Top-5={ids}"


async def test_vector_recalls_semantically_matching_clause(policy_vector: Any) -> None:
    """Vector 路能按语义把「措辞不同但同义」的条款召回进 Top-K。

    断言：查询「商品把 PU 材质宣称为真皮，属于成分虚假宣传」的 Top-5 含 ``POLICY_2.3_v1_c1``
    （材质/成分虚假声称）。意义：目标条款正文与查询措辞并不逐字相同，只有语义编码生效才会召回
    —— 用真 BGE 验证向量路真的做了语义检索，而非退化到词面匹配。
    """
    hits = await policy_vector.search(
        "商品把 PU 材质宣称为真皮，属于成分虚假宣传",
        PolicySearchFilters(),
        5,
        True,
    )
    ids = [h.clause_id for h in hits]
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


async def test_vector_scores_are_similarity_scale_unlike_rrf(case_vector: Any) -> None:
    """对照反证：vector 模式的分是 ``1 − cosine distance``，量纲与 hybrid 的 RRF 分完全不同。

    断言：同一查询在 vector 模式的 Top 分 **大于** RRF 上界 ``2/60``。意义：与上一条互为反证
    —— 若 hybrid 误用了 vector 的相似度分（或反之），两条用例必有一条失败；同时钉住「不得把
    RRF 分称作语义相似度」这条术语红线。
    """
    hits = await case_vector.search(
        "无品牌外观高度模仿知名品牌，商家多次改标题重上架",
        CaseSearchFilters(),
        5,
    )
    scores = [h.retrieval_score for h in hits]
    assert scores, "vector 检索必须返回命中"
    assert max(scores) > _RRF_UPPER_BOUND, (
        f"vector 相似度分应远大于 RRF 上界 {_RRF_UPPER_BOUND:.4f}，实际={scores}"
    )


def test_fuse_rrf_ranks_double_route_doc_above_single_route_doc() -> None:
    """``fuse_rrf`` 的核心性质：两路都排靠前的文档，名次优于只被一路命中的文档。

    断言：``both`` 同时位于 vector / bm25 首位 → 分 = ``2/60``；``v_only`` / ``b_only`` 各只被
    一路排到次位 → 分 = ``1/61``，故 ``score(both) > score(v_only) == score(b_only)``。意义：
    这是「RRF 真的在融合，而非取两路并集」的最小可判据。索引侧两路排名都覆盖全部候选，
    端到端观察不到「单路命中」的差异，故必须在 ``fuse_rrf`` 这一层用合成输入验证 ——
    ``fuse_rrf(ranked, row_index)`` 入参是「各路排名 id 列表 + node_id→行索引映射」，
    返回 ``[(行索引, RRF 分)]``。
    """
    from pra.rag.retrieval import fuse_rrf

    ranked = {"vector": ["id_both", "id_v"], "bm25": ["id_both", "id_b"]}
    row_index = {"id_both": 0, "id_v": 1, "id_b": 2}
    scores = dict(fuse_rrf(ranked, row_index))
    assert scores[0] > scores[1], f"双路命中应优于单路命中：{scores}"
    assert scores[0] > scores[2], f"双路命中应优于单路命中：{scores}"
    assert scores[0] == pytest.approx(2.0 / _RRF_K, abs=1e-6)
    assert scores[1] == pytest.approx(1.0 / (_RRF_K + 1.0), abs=1e-6)
    assert scores[2] == pytest.approx(1.0 / (_RRF_K + 1.0), abs=1e-6)


# ---------------------------------------------------------------------------
# 4. metadata filter 正确性
# ---------------------------------------------------------------------------


async def test_policy_effective_only_excludes_expired_clause(
    policy_bm25: Any, policy_rows: Any
) -> None:
    """``effective_only`` 语义：True 时 EXPIRED 条款不得出现，False 时可出现。

    断言：对指向失效条款正文的查询「禁止销售仿冒、假冒注册商标的商品」（= ``POLICY_1.1_v1_c1``
    正文），``effective_only=False`` 的返回**含**该 EXPIRED 条款，``effective_only=True`` 的返回
    **不含**它且全部为 EFFECTIVE。意义：用一个「本可命中」的失效条款作对照，才能证明它消失确由
    ``effective_only`` 造成，而非查询根本没召回（把过滤与召回解耦，避免空断言）。
    """
    query = "禁止销售仿冒、假冒注册商标的商品"
    top_k = len(policy_rows)

    loose = await policy_bm25.search(query, PolicySearchFilters(), top_k, False)
    assert "POLICY_1.1_v1_c1" in [h.clause_id for h in loose], (
        "effective_only=False 时失效条款应可被召回（对照基准）"
    )

    strict = await policy_bm25.search(query, PolicySearchFilters(), top_k, True)
    assert "POLICY_1.1_v1_c1" not in [h.clause_id for h in strict], (
        "effective_only=True 时 EXPIRED 条款不得出现"
    )
    assert all(h.status == "EFFECTIVE" for h in strict), (
        "effective_only=True 结果必须全为 EFFECTIVE"
    )


async def test_policy_category_filter_keeps_full_category_clauses(
    policy_bm25: Any, policy_rows: Any
) -> None:
    """``category`` 过滤含「全类目」语义：指定具体类目时，全类目条款也应保留。

    断言：``category="箱包/女包"`` 时返回条款 category 全 ∈ {箱包/女包, 全类目}，且含
    ``POLICY_4.2_v2_c1``（全类目 + 改标题重上架 —— 查询正指向它）。意义：把「全类目」当普通类目
    精确匹配会把平台通用条款整批漏掉，本断言钉住 include-全类目 的过滤语义。
    """
    hits = await policy_bm25.search(
        "改标题、描述后重新上架规避审核",
        PolicySearchFilters(category="箱包/女包"),
        len(policy_rows),
        True,
    )
    cats = {h.category for h in hits}
    assert cats <= {"箱包/女包", "全类目"}, f"category 过滤漏了全类目语义：{cats}"
    assert "POLICY_4.2_v2_c1" in [h.clause_id for h in hits], "全类目条款应被保留"


async def test_policy_risk_type_filter_is_overlap_not_equality(
    policy_bm25: Any, policy_rows: Any
) -> None:
    """``risk_type`` 过滤是**交叠非空**，不是相等匹配。

    断言：``risk_type=["FALSE_CLAIM"]`` 时返回条款的 risk_type 均与 {FALSE_CLAIM} 有交集，且含
    ``POLICY_2.3_v1_c1``（risk_type = [FALSE_CLAIM, FIELD_CONFLICT]，多标一个也要命中）。意义：
    相等匹配会漏掉「多风险类型」条款 —— 正是「漏召回只在带过滤时暴露」的典型场景。
    """
    hits = await policy_bm25.search(
        "材质成分虚假宣传 PU 冒充实皮",
        PolicySearchFilters(risk_type=["FALSE_CLAIM"]),
        len(policy_rows),
        True,
    )
    assert hits, "risk_type 过滤后应仍有命中"
    assert all({t.value for t in h.risk_type} & {"FALSE_CLAIM"} for h in hits), (
        "存在与过滤类型无交集的条款"
    )
    assert "POLICY_2.3_v1_c1" in [h.clause_id for h in hits], (
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


async def test_filter_plus_retrieval_no_silent_recall_loss(policy_bm25: Any) -> None:
    """红线：**带过滤时**的检索不得静默漏召回（漏召回只在带过滤时暴露）。

    断言：``category="女鞋/运动鞋"`` + ``effective_only=True`` + 指向该条款的查询下，Top-K 仍含
    ``POLICY_1.4_v1_c1``（女鞋/运动鞋且 EFFECTIVE）。意义：category 走的是 store 下推 + Python
    二次复核的双重路径，若下推结果与候选集不一致且无补位，目标条款会被静默剔除；本用例专门覆盖
    「过滤 + 检索」这一 red-line 组合（与无过滤的召回用例互为补充）。
    """
    hits = await policy_bm25.search(
        "鞋靴整体外观高度模仿知名品牌且无授权",
        PolicySearchFilters(category="女鞋/运动鞋"),
        10,
        True,
    )
    assert "POLICY_1.4_v1_c1" in [h.clause_id for h in hits], (
        "带 category 过滤时预期的类目条款被漏召回"
    )


# ---------------------------------------------------------------------------
# 5. 两层契约：索引 → Hit；工具 → Evidence
# ---------------------------------------------------------------------------


async def test_index_search_returns_hit_models(policy_bm25: Any, case_hybrid: Any) -> None:
    """索引检索返回的是工具契约的 Hit 模型（而非内部 node / dict）。

    断言：``ChromaPolicyIndex.search`` 返回 ``PolicyClauseHit``；``ChromaCaseIndex.search`` 返回
    ``CaseHit``，且 ``retrieval_score ∈ [0, 1]``（该约束带 ``le=1``，越界会直接 ValidationError）。
    意义：工具层只认 Hit 契约，索引若透出内部结构会让上层协议形同虚设。
    """
    p_hits = await policy_bm25.search("外观模仿", PolicySearchFilters(), 3, True)
    c_hits = await case_hybrid.search("外观模仿", CaseSearchFilters(), 3)
    assert p_hits and all(type(h).__name__ == "PolicyClauseHit" for h in p_hits)
    assert c_hits and all(type(h).__name__ == "CaseHit" for h in c_hits)
    assert all(0.0 <= h.retrieval_score <= 1.0 for h in c_hits), (
        "CaseHit.retrieval_score 必须落在 [0, 1]"
    )


async def test_policy_search_tool_produces_policy_ref_evidence(policy_bm25: Any) -> None:
    """Policy 两层：``ChromaPolicyIndex.search`` → ``PolicyClauseHit``；工具 → ``POLICY_REF``。

    断言：注入真索引的 ``PolicySearchTool.call`` 拿到 ≥1 个 ``PolicyClauseHit``；``to_evidence``
    每条证据 type=``POLICY_REF``、source=``PolicySearchTool``、weight==0.9、ref_id==命中条款的
    clause_id。意义：「RAG 怎么检索」与「Evidence 怎么产生」是两个层次，本用例同时钉住命中类型与
    证据映射（可追溯引用 = ref_id 必填、政策依据 = 强权重 0.9）。
    """
    tool = PolicySearchTool(index=policy_bm25)
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
        assert ev.weight == 0.9
        assert ev.ref_id == hit.clause_id


async def test_case_search_tool_produces_case_precedent_evidence(case_hybrid: Any) -> None:
    """Case 两层：``ChromaCaseIndex.search`` → ``CaseHit``；工具 → ``CASE_PRECEDENT``。

    断言：命中均为 ``CaseHit`` 且 ``retrieval_score ⊂ [0,1]``；证据 type=``CASE_PRECEDENT``、
    source=``CaseSearchTool``、weight==该 hit 的 ``retrieval_score``、ref_id==case_id。意义：证据
    侧 weight 直接复用检索分（术语：这是检索分，不是语义相似度），两条断言一起把「检索」与
    「证据产生」两层焊住。
    """
    tool = CaseSearchTool(index=case_hybrid)
    result = await tool.call(
        CaseSearchArgs(query="无品牌 + 高相似 + 商家多次重上架", top_k=3),
        _tool_ctx(),
    )
    assert result.ok and result.hits, "注入真索引后工具应拿到命中"
    assert all(type(h).__name__ == "CaseHit" for h in result.hits)
    assert all(0.0 <= h.retrieval_score <= 1.0 for h in result.hits)
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


def test_invalid_mode_raises_value_error(policy_rows: Any, embedder: Any) -> None:
    """非法 ``mode`` 必须抛 ``ValueError``（而不是静默退回某个默认模式）。

    断言：以 ``mode="similarity"`` 构造索引抛 ``ValueError``。意义：模式是检索语义开关，静默兜底
    会让调用方以为跑的是它请求的模式（历史兼容分支正是被判定为「防御假想问题」而删除）。
    """
    from pra.rag.factory import build_policy_index

    with pytest.raises(ValueError):
        build_policy_index(rows=policy_rows, embedding_model=embedder, mode="similarity")


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
            rows=policy_rows, embedding_model=embedder, mode="vector", config=config
        )
        await index.search("外观模仿", PolicySearchFilters(), 5, True)
    except Exception:  # noqa: BLE001 — 任意异常都算「已上抛」，见上 docstring
        return
    pytest.fail("不可达 Chroma 未被上抛（构建与检索都成功）→ 疑似静默降级为空结果")
