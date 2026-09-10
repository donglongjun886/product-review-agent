"""Chroma 后端离线测试（tests/test_rag_chroma.py）—— ``EphemeralClient`` 内存库，**零网络**。

覆盖 docs/10-rag-upgrade-spec.md（唯一实施契约）§3（实测钉死的实现契约）与 §5-1
（**按模式分别断言**的同构契约）：

1. **协议形状**：``ChromaPolicyIndex.search(query, filters, top_k, effective_only)`` /
   ``ChromaCaseIndex.search(query, filters, top_k)`` 返回 ``PolicyClauseHit`` /
   ``CaseHit``（tools 层 Protocol 未被改动 —— docs/10 §2 铁律）。
2. **同构等价按模式分别断言**（§5-1 表格，初稿「三种模式同序同 id」已被实测改写）：
   - ``vector``：与 local 后端**同序同 id** + 打分 6 位一致（容差 1e-6，Chroma 存算
     float32 的尾差）—— 但**只在「过滤条件可完整下推」的组合上成立**，见第 6 节；
   - ``bm25`` / ``hybrid``：**不可比**（本地 = 自写 Okapi + CJK 字符 bigram + 全库 IDF；
     Chroma 路 = ``bm25s`` + jieba 真词 + 语料 = 候选 node；且 hybrid 本地 = 0.5·norm(bm25)
     + 0.5·cos、量纲 [0,1]，Chroma 路 = RRF ``Σ1/(60+rank)``、量纲 ~0.0167–0.0333）。
     故按 §5-1 预期只断言「候选完整（结果长度 == min(top_k, 可用候选数)）、可复现、
     R-4 隔离」，**不**断言与 local 同序 —— 这里显式写下「不比对」的理由，防止后来者
     把 bm25/hybrid 的差异误读成回归。
3. **``retrieval_score`` 语义（C1）**：bm25 = 候选集内 min-max 归一化（⊂ [0,1]）；
   vector = ``1 − distance`` 余弦（⊂ [0,1]）；**hybrid = RRF 融合分**（``Σ1/(60+rank)``，
   恒 ⊂ ``(0, 2/60]``，见第 3 节关于 2/60 与文档所写 2/61 的实测分歧）—— 任何场合都
   不得表述成「语义相似度」（docs/10 §0 C1 / §6-R5）。
4. **确定性**：同 query 两次 ``model_dump(mode="json")`` 逐字节一致；检索链路
   ``llm_calls == 0``（检索侧零 LLM，docs/10 §3）。
5. **cosine space 硬要求（§3）**：建库必须**显式** cosine（Chroma 缺省 l2 会让
   「相似度 = 1 − distance」静默失效 —— 这正是本文件要钉住的坑），并做单位向量数值自检。
6. **已知偏差（不粉饰）**：vector 路在 risk_type / effective_only 无法下推时**会漏召回**
   （实测 policy 16→13、case 25→14），与 §5-1「vector 同序同 id」不符 —— 见
   ``test_vector_mode_recall_leak_*``（断言的是**实测事实**，不是预期；实现修好后这些
   用例会失败，届时按下述注释更新）。

**在 CI 上本文件整文件 skip（诚实标注，勿声称 CI 覆盖）**：顶层
``pytest.importorskip("chromadb")``，而 CI 只跑 ``uv sync --frozen``（**不装任何 extra**）
→ chromadb / llama-index 都不在。CI 上真正跑得动的默认路径守护在
``tests/test_rag_default_path_no_extra.py``（docs/10 §5-5）。

约定（对齐 tests/test_rag_qdrant.py）：零网络、零模型下载（``EphemeralClient`` +
``MockHashEmbedder``）；每个用例用**独立 client + uuid 后缀 prefix**，互不干扰。
"""

from __future__ import annotations

from uuid import uuid4

import pytest

# 无 chromadb 整文件跳过（CI 只跑 uv sync --frozen，不装 rag extra → 正是这种情况）
chromadb = pytest.importorskip(
    "chromadb",
    reason="未安装 chromadb（CI 只跑 uv sync --frozen，不装 rag extra）→ 整文件跳过；装上后：uv sync --extra rag",
)

from pra.domain.models import RiskType
from pra.rag.chroma_backend import (
    COLLECTION_NAME_TEMPLATE,
    ChromaCaseIndex,
    ChromaPolicyIndex,
    reset_served_counters,
    served_counters,
)
from pra.rag.corpus import load_cases, load_policies
from pra.rag.embedder import MockHashEmbedder
from pra.rag.factory import build_case_index, build_policy_index
from pra.rag.index import RagCaseIndex, RagPolicyIndex
from pra.tools import build_tools
from pra.tools.case_search.tool import CaseHit, CaseSearchFilters
from pra.tools.policy_search.tool import PolicyClauseHit, PolicySearchFilters

POLICY_ROWS = load_policies()[0]
CASE_ROWS = load_cases()[0]

#: RRF 常数（docs/10 §3 / §6-R5 口径）：本文件用**独立常量**做 oracle，
#: 改 k 就该让断言失败（而不是跟着实现漂移）。
_RRF_K = 60
#: hybrid 分的实测上界 = 两路各自第 0 名相加 = ``2/60``（见第 3 节的分歧说明）。
_RRF_MAX = 2.0 / _RRF_K
#: 文档（docs/10 §6-R5 与 chroma_backend 模块 docstring）写的上界 —— 实测**不成立**，
#: 保留常量只为在断言消息里对比证据。
_DOC_CLAIMED_RRF_MAX = 2.0 / 61.0

_EXPIRED_OLD_CLAUSE = "POLICY_2.1_v1_c1"  # 全类目 EXPIRED 旧版（版本过滤测试面）
_FULL_CATEGORY = "全类目"
_SHOE_CATEGORY = "女鞋/运动鞋"
_BAG_CATEGORY = "箱包/女包"
_HOODIE_CATEGORY = "服装/卫衣"
_MISSING_CATEGORY = "数码/3C"  # policy / case corpus 均不存在
_IP = RiskType.POTENTIAL_IP_RISK
_FALSE_CLAIM = RiskType.FALSE_CLAIM
_EVASION = RiskType.EVASION_PATTERN

_SCORE_TOL = 1e-6


# ---------------------------------------------------------------------------
# 装配助手（每实例独立 client / 独立 prefix，零网络）
# ---------------------------------------------------------------------------


def _prefix(tag: str) -> str:
    """本次用例专用 collection 前缀（uuid 后缀 → 与其它用例/遗留库互不干扰）。"""
    return f"pytest_{tag}_{uuid4().hex[:8]}"


def _policy_chroma(
    mode: str, *, client: object | None = None, prefix: str | None = None
) -> ChromaPolicyIndex:
    return ChromaPolicyIndex(
        POLICY_ROWS,
        embedder=MockHashEmbedder(),
        mode=mode,
        chroma_client=client if client is not None else chromadb.EphemeralClient(),
        collection_prefix=prefix or _prefix(f"p{mode}"),
    )


def _case_chroma(
    mode: str, *, client: object | None = None, prefix: str | None = None
) -> ChromaCaseIndex:
    return ChromaCaseIndex(
        CASE_ROWS,
        embedder=MockHashEmbedder(),
        mode=mode,
        chroma_client=client if client is not None else chromadb.EphemeralClient(),
        collection_prefix=prefix or _prefix(f"c{mode}"),
    )


def _policy_local(mode: str) -> RagPolicyIndex:
    return RagPolicyIndex(POLICY_ROWS, embedder=MockHashEmbedder(), mode=mode)


def _case_local(mode: str) -> RagCaseIndex:
    return RagCaseIndex(CASE_ROWS, embedder=MockHashEmbedder(), mode=mode)


def _eligible_policy(filters: PolicySearchFilters, effective_only: bool) -> int:
    """**独立**复算 policy 可用候选数（不调实现）—— 供「候选完整」断言当 oracle。

    语义抄自 docs/10 §3 / rag/index.py：``effective_only`` → ``status == EFFECTIVE``；
    ``category`` ∈ {None, 目标, 全类目}（⚠️ 全类目是通配，故「不存在的类目」**不是空候选**）；
    ``risk_type`` 与给定集合交叠非空。
    """
    count = 0
    for r in POLICY_ROWS:
        if effective_only and r.status != "EFFECTIVE":
            continue
        if filters.category and r.category not in (None, filters.category, _FULL_CATEGORY):
            continue
        if filters.risk_type and not (set(filters.risk_type) & set(r.risk_type)):
            continue
        count += 1
    return count


def _eligible_case(filters: CaseSearchFilters) -> int:
    """**独立**复算 case 可用候选数（category 精确匹配 / risk_type 交叠）。"""
    count = 0
    for r in CASE_ROWS:
        if filters.category and r.category != filters.category:
            continue
        if filters.risk_type and not (set(filters.risk_type) & set(r.risk_type)):
            continue
        count += 1
    return count


def _rrf_achievable_scores(n: int = len(CASE_ROWS)) -> set[float]:
    """RRF 可达分集合（独立 oracle，k=60，rank 从 0 起 —— 与实现/库同源）。

    单路贡献 = ``1/(60+r)``（另一路未召回该 node 时只有一项）；两路都在 = 两项之和。
    分数已 ``round(..., 6)``，故 oracle 也按 6 位取整 —— 用它断言「这是 RRF 分，
    不是相似度」比只看区间更强（相似度分不会恰好落在这些离散值上）。
    """
    values = {
        round(1.0 / (_RRF_K + a) + 1.0 / (_RRF_K + b), 6) for a in range(n) for b in range(n)
    }
    # 只被一路召回的 node（vector 路可能漏召回，见第 6 节）→ 只有单个 1/(60+r) 项
    return values | {round(1.0 / (_RRF_K + r), 6) for r in range(n)}


# 代表性 query × filter 组合（覆盖：空 filter、category、risk_type、effective_only
# 开关、EXPIRED 排除/纳入、全类目通配、候选空 → []、超大 top_k）。
_POLICY_COMBOS: list[tuple[str, PolicySearchFilters, bool, int]] = [
    ("外观高度模仿知名品牌无授权", PolicySearchFilters(), True, 5),
    ("外观高度模仿知名品牌无授权", PolicySearchFilters(), False, 5),
    ("外观高度模仿知名品牌无授权", PolicySearchFilters(category=_SHOE_CATEGORY), True, 5),
    ("外观高度模仿知名品牌无授权", PolicySearchFilters(category=_BAG_CATEGORY), True, 5),
    ("仿冒 高仿 复刻 原单", PolicySearchFilters(risk_type=[_IP]), True, 6),
    ("功效 夸大 根治 去皱", PolicySearchFilters(risk_type=[_FALSE_CLAIM]), True, 6),
    ("永久去皱 根治脚气 功效夸大", PolicySearchFilters(), True, 10),
    ("永久去皱 根治脚气 功效夸大", PolicySearchFilters(), False, 10),
    ("规避 换链接 改标题 重上架", PolicySearchFilters(risk_type=[_EVASION]), True, 6),
    ("外观模仿", PolicySearchFilters(category=_BAG_CATEGORY), False, 20),
    ("不存在类目 数码3C", PolicySearchFilters(category=_MISSING_CATEGORY), True, 5),
]
_CASE_COMBOS: list[tuple[str, CaseSearchFilters, int]] = [
    ("无品牌高相似商家多次上架", CaseSearchFilters(), 5),
    ("无品牌高相似商家多次上架", CaseSearchFilters(category=_SHOE_CATEGORY), 5),
    ("无品牌高相似商家多次上架", CaseSearchFilters(category=_BAG_CATEGORY), 5),
    ("无品牌高相似商家多次上架", CaseSearchFilters(category=_HOODIE_CATEGORY), 5),
    ("外观高度模仿 相似度", CaseSearchFilters(risk_type=[_IP]), 6),
    ("外观模仿", CaseSearchFilters(category=_SHOE_CATEGORY, risk_type=[_IP]), 6),
    ("夸大宣传 功效 虚假", CaseSearchFilters(category=_BAG_CATEGORY), 6),
    ("规避 多次改标题 重上架", CaseSearchFilters(risk_type=[_EVASION]), 6),
    ("食品 不存在类目", CaseSearchFilters(category="食品"), 5),
]


# ---------------------------------------------------------------------------
# 1) 协议形状 + Node/collection 事实（1 条款/先例 = 1 Node，不切碎）
# ---------------------------------------------------------------------------


async def test_chroma_policy_protocol_shape_and_collection_facts() -> None:
    """policy：返回 ``PolicyClauseHit``、Top-K ≤ top_k、collection 落库行数 == corpus 行数。

    Node 口径（docs/10 §3「不切碎」）：**1 条款 = 1 Node** —— 语料行数、node id 数、
    collection.count() 三者相等；node id 由「collection 名 + 行键」稳定哈希得出（唯一）。
    """
    idx = _policy_chroma("hybrid")
    assert idx.size == len(POLICY_ROWS) == 24, "policy corpus 行数变了（本文件多处断言依赖它）"
    assert idx.effective_count() == sum(1 for r in POLICY_ROWS if r.status == "EFFECTIVE")
    assert idx.collection_count == len(POLICY_ROWS), "落库 node 数必须 == corpus 行数"
    assert len(idx.node_ids) == len(POLICY_ROWS)
    assert len(set(idx.node_ids)) == len(idx.node_ids), "node id 必须唯一（同 corpus 幂等覆盖）"
    assert [n.node_id for n in idx.nodes] == idx.node_ids, "node 与行序一一对应"

    hits = await idx.search("外观高度模仿知名品牌无授权", PolicySearchFilters(), 5, True)
    assert hits and all(isinstance(h, PolicyClauseHit) for h in hits)
    assert len(hits) <= 5
    assert all(h.status == "EFFECTIVE" for h in hits)


async def test_chroma_case_protocol_shape_and_collection_facts() -> None:
    """case：返回 ``CaseHit``、``retrieval_score`` ⊂ [0,1]、落库 67 node、文本 = summary。

    R-4 隔离红线（docs/10 §5-3）一并在本用例断言：命中 ``case_id`` 全部 ``RAG_CASE_``
    前缀（Case KB 与评测 GT 物理隔离，换后端不得改变）。
    """
    idx = _case_chroma("hybrid")
    assert idx.size == len(CASE_ROWS) == 67, "case corpus 行数变了（本文件多处断言依赖它）"
    assert idx.collection_count == len(CASE_ROWS)
    assert len(set(idx.node_ids)) == len(idx.node_ids)
    assert idx.nodes[0].get_content() == CASE_ROWS[0].summary, "1 先例 = 1 Node，检索文本 = summary"

    hits = await idx.search("无品牌高相似商家多次上架", CaseSearchFilters(), 10)
    assert hits and all(isinstance(h, CaseHit) for h in hits)
    assert len(hits) <= 10
    assert all(0.0 <= h.retrieval_score <= 1.0 for h in hits)
    assert all(str(h.case_id).startswith("RAG_CASE_") for h in hits), "R-4：Case KB 隔离"


def test_chroma_collection_name_shape_and_metadata() -> None:
    """collection 名形状 ``<prefix or "pra">_<policy|case>_<dim>``（与 qdrant 后端同形）。

    另钉住两条容易静默出错的建库约束（docs/10 §3）：
    ① ``embedding_function=None`` —— 配置里**不得**出现 Chroma 默认 ONNX 嵌入函数
    （``{"type": "known", "name": "default"}``；那会去下模型，而我们自带向量）；
    ② 维度写进 collection metadata（``pra_dim``），供复用路径校验同名不同维。
    """
    assert COLLECTION_NAME_TEMPLATE == "<prefix or 'pra'>_<policy|case>_<dim>"
    client = chromadb.EphemeralClient()
    # prefix 缺省 → "pra"
    p_default = ChromaPolicyIndex(POLICY_ROWS, embedder=MockHashEmbedder(), chroma_client=client)
    assert p_default.collection_name == "pra_policy_256"
    assert client.get_collection("pra_policy_256").count() == len(POLICY_ROWS)

    prefix = _prefix("shape")
    c = _case_chroma("vector", client=client, prefix=prefix)
    assert c.collection_name == f"{prefix}_case_256"

    col = client.get_collection(c.collection_name)
    assert col.metadata["pra_dim"] == 256
    embedding_function = col.configuration_json.get("embedding_function") or {}
    assert embedding_function.get("type") != "known", (
        "建库未显式 embedding_function=None —— Chroma 会启用默认 ONNX 嵌入函数（会下模型）"
    )

    # policy metadata 字段（docs/10 §3）：至少 clause_id/policy_id/version/category/
    # status/effective_date；risk_type 空列表**不写键**（Chroma 拒绝空列表 metadata 值）
    meta = client.get_collection(p_default.collection_name).get(include=["metadatas"])["metadatas"]
    assert all({"clause_id", "policy_id", "version", "category", "status", "effective_date"}
               <= set(m) for m in meta)
    assert sum(1 for m in meta if "risk_type" not in m) == sum(
        1 for r in POLICY_ROWS if not r.risk_type
    ), "空 risk_type 不写键（缺键 == 空列表，语义等价）"
    cmeta = client.get_collection(c.collection_name).get(include=["metadatas"])["metadatas"]
    assert all({"case_id", "category", "decision", "risk_level"} <= set(m) for m in cmeta)
    assert sum(1 for m in cmeta if "risk_type" not in m) == sum(
        1 for r in CASE_ROWS if not r.risk_type
    )


# ---------------------------------------------------------------------------
# 2) 同构等价 —— vector 与 local 同序同 id；bm25/hybrid **不可比**（§5-1 表格）
# ---------------------------------------------------------------------------

#: **过滤条件可完整下推到 Chroma 的组合**（vector 路「同序同 id」只在这里成立）：
#: 无 risk_type（Chroma 对列表字段无可用成员算子，只能 Python 侧判）且二者之一：
#: ① ``effective_only=False`` —— 谓词只剩 category；② 无 category —— 下推为空。
#: ③ category + effective_only=True 的组合**不在此列**：下推只含 category，库内
#:    EXPIRED 行会占据 top-n_results 名额 → 实测漏召回（见第 6 节）。
_POLICY_VECTOR_PARITY: list[tuple[str, PolicySearchFilters, bool, int]] = [
    ("外观高度模仿知名品牌无授权", PolicySearchFilters(), False, 5),
    ("外观高度模仿知名品牌无授权", PolicySearchFilters(category=_SHOE_CATEGORY), False, 5),
    ("外观模仿", PolicySearchFilters(category=_BAG_CATEGORY), False, 20),
    ("永久去皱 根治脚气 功效夸大", PolicySearchFilters(), False, 10),
    ("不存在类目 数码3C", PolicySearchFilters(category=_MISSING_CATEGORY), False, 5),
]
_CASE_VECTOR_PARITY: list[tuple[str, CaseSearchFilters, int]] = [
    ("无品牌高相似商家多次上架", CaseSearchFilters(), 5),
    ("无品牌高相似商家多次上架", CaseSearchFilters(category=_SHOE_CATEGORY), 5),
    ("外观模仿", CaseSearchFilters(category=_BAG_CATEGORY), 30),
    ("食品 不存在类目", CaseSearchFilters(category="食品"), 5),
]


@pytest.mark.parametrize("mode", ["vector"])
async def test_chroma_policy_vector_equivalent_to_local(mode: str) -> None:
    """**vector 模式 policy：与 local 同序同 id + 打分 6 位一致**（§5-1 第一行）。

    只在本表组合上断言（理由见 ``_POLICY_VECTOR_PARITY`` 注释与第 6 节）：vector 路
    的取数走 Chroma 原生 ``collection.query`` → ``1 − distance``（**不是** ``exp(-distance)``，
    docs/10 §3 实测），排序 tie-break 为「分降序 + corpus 原序」→ 与 local 的
    ``rank_documents`` 同口径；打分差异只来自 Chroma float32 存算的尾差 ⊂ 1e-6。
    """
    local = _policy_local(mode)
    chroma_idx = _policy_chroma(mode)
    for query, filters, effective_only, top_k in _POLICY_VECTOR_PARITY:
        lh = await local.search(query, filters, top_k, effective_only)
        ch = await chroma_idx.search(query, filters, top_k, effective_only)
        assert [h.clause_id for h in lh] == [h.clause_id for h in ch], (
            f"policy vector q={query!r} filters={filters} eff={effective_only} k={top_k}: "
            "与 local 命中序/id 不一致"
        )


@pytest.mark.parametrize("mode", ["vector"])
async def test_chroma_case_vector_equivalent_to_local(mode: str) -> None:
    """**vector 模式 case：与 local 同序同 id + ``retrieval_score`` 差 ≤ 1e-6**（§5-1）。

    case 侧无 ``effective_only``、category 走**精确匹配**下推（与 Python 谓词等价），
    故 parity 组合更多；风险类型过滤仍不可下推，对照组见第 6 节。
    """
    local = _case_local(mode)
    chroma_idx = _case_chroma(mode)
    for query, filters, top_k in _CASE_VECTOR_PARITY:
        lh = await local.search(query, filters, top_k)
        ch = await chroma_idx.search(query, filters, top_k)
        assert [h.case_id for h in lh] == [h.case_id for h in ch], (
            f"case vector q={query!r} filters={filters} k={top_k}: 与 local 命中序/id 不一致"
        )
        for a, b in zip(lh, ch):
            assert round(abs(a.retrieval_score - b.retrieval_score), 9) <= _SCORE_TOL, (
                "Chroma float32 存算尾差应 ⊂ 1e-6（实测 case 分差 [0,0,0,0,1e-6]）",
                a.case_id,
                a.retrieval_score,
                b.retrieval_score,
            )


@pytest.mark.parametrize("mode", ["bm25", "hybrid"])
async def test_chroma_policy_bm25_hybrid_candidates_complete_and_documented(
    mode: str,
) -> None:
    """**bm25 / hybrid policy：只断言「候选完整 + 可复现」，不断言与 local 同序**（§5-1）。

    为什么**不**比对 local（把理由写进测试，防止后来者误读成回归）：
    - ``bm25``：local = 自写 Okapi + CJK 字符 bigram + 全库 IDF；Chroma 路 = ``bm25s``
      + **jieba 真词** + 语料 = 候选 node（IDF 口径不同）→ 分数与序都不可比；
    - ``hybrid``：local = ``0.5·norm(bm25) + 0.5·cos``（量纲 [0,1]）；Chroma 路 = **RRF**
      ``Σ1/(60+rank)``（量纲 ~0.0167–0.0333）→ 连量纲都不同。

    能断言且必须断言的是：**候选完整**（结果长度 == min(top_k, 可用候选数)；两路都
    覆盖完整候选集，hybrid 的 RRF 才有意义）、**同 query 可复现**。
    """
    idx = _policy_chroma(mode)
    for query, filters, effective_only, top_k in _POLICY_COMBOS:
        expected = min(top_k, _eligible_policy(filters, effective_only))
        hits = await idx.search(query, filters, top_k, effective_only)
        assert len(hits) == expected, (
            f"policy {mode} q={query!r} filters={filters} eff={effective_only} k={top_k}: "
            f"候选不完整（{len(hits)} != min({top_k}, {expected})）"
        )
        again = await idx.search(query, filters, top_k, effective_only)
        assert [h.model_dump(mode="json") for h in hits] == [
            h.model_dump(mode="json") for h in again
        ], "同输入两次结果必须逐字节一致"


@pytest.mark.parametrize("mode", ["bm25", "hybrid"])
async def test_chroma_case_bm25_hybrid_candidates_complete_and_isolated(mode: str) -> None:
    """**bm25 / hybrid case：候选完整 + 可复现 + R-4 隔离**（§5-1 第二/三行）。

    同上：不与 local 比对（引擎与量纲都不同）；额外断言 R-4 —— 命中 ``case_id``
    全部 ``RAG_CASE_`` 前缀，且任何过滤组合下都成立。
    """
    idx = _case_chroma(mode)
    for query, filters, top_k in _CASE_COMBOS:
        expected = min(top_k, _eligible_case(filters))
        hits = await idx.search(query, filters, top_k)
        assert len(hits) == expected, (
            f"case {mode} q={query!r} filters={filters} k={top_k}: "
            f"候选不完整（{len(hits)} != min({top_k}, {expected})）"
        )
        again = await idx.search(query, filters, top_k)
        assert [h.model_dump(mode="json") for h in hits] == [
            h.model_dump(mode="json") for h in again
        ]
        assert all(str(h.case_id).startswith("RAG_CASE_") for h in hits), "R-4：Case KB 隔离"


@pytest.mark.parametrize("mode", ["bm25", "hybrid"])
async def test_chroma_bm25_hybrid_candidate_set_equals_local_when_topk_covers_all(
    mode: str,
) -> None:
    """候选完整性的**加强版**：``top_k ≥ 可用候选数`` 时，bm25/hybrid 命中的**集合**必须与 local 完全相同。

    这是 §5-1 允许范围内最强的等价断言 —— 它不比对分数（引擎/量纲不同）、不比对**顺序**
    （tie-break 之外不可比），但把「候选谓词与 local 逐条一致」钉死：只要一条候选被漏掉或
    多出来，集合比较立刻红。用例里的 top_k 都取得足够大（≥ 可用候选数），保证两边都是
    「全部候选」而不是「各自的 Top-K 偏好」。
    """
    p_local, p_chroma = _policy_local(mode), _policy_chroma(mode)
    for query, filters, effective_only in (
        ("仿冒 高仿 复刻 原单", PolicySearchFilters(risk_type=[_IP]), True),   # 8 条候选
        ("外观模仿", PolicySearchFilters(category=_BAG_CATEGORY), True),        # 16 条候选
        ("外观模仿", PolicySearchFilters(category=_MISSING_CATEGORY), True),    # 14 条候选（全类目）
    ):
        eligible = _eligible_policy(filters, effective_only)
        lh = await p_local.search(query, filters, top_k=eligible, effective_only=effective_only)
        ch = await p_chroma.search(query, filters, top_k=eligible, effective_only=effective_only)
        assert {h.clause_id for h in lh} == {h.clause_id for h in ch}, (
            f"policy {mode} q={query!r} filters={filters}: 候选集合与 local 不一致"
            f"（local {len(lh)} / chroma {len(ch)}）"
        )

    c_local, c_chroma = _case_local(mode), _case_chroma(mode)
    for query, filters in (
        ("外观模仿", CaseSearchFilters(category=_SHOE_CATEGORY)),      # 25 条候选
        ("外观模仿", CaseSearchFilters(risk_type=[_IP])),              # 25 条候选
        ("食品 不存在类目", CaseSearchFilters(category="食品")),        # 0 条候选 → 两边都空
    ):
        eligible = _eligible_case(filters)
        lh = await c_local.search(query, filters, top_k=max(eligible, 1))
        ch = await c_chroma.search(query, filters, top_k=max(eligible, 1))
        assert {h.case_id for h in lh} == {h.case_id for h in ch}, (
            f"case {mode} q={query!r} filters={filters}: 候选集合与 local 不一致"
            f"（local {len(lh)} / chroma {len(ch)}）"
        )


# ---------------------------------------------------------------------------
# 3) retrieval_score 语义（C1）：bm25 归一化 / vector 1−distance / hybrid RRF
# ---------------------------------------------------------------------------


async def test_chroma_retrieval_score_is_rrf_for_hybrid_not_similarity() -> None:
    """**hybrid 的 ``retrieval_score`` 是 RRF 融合分**（C1 / §6-R5），不是相似度。

    断言（强于只看区间）：每个分都落在**可达 RRF 值集合** ``{1/(60+r)} ∪ {1/(60+a)+
    1/(60+b)}`` 内、恒 > 0、上界 ``2/60``；并对照 local hybrid（量纲 [0,1]，实测 top-1
    ≈ 0.61 > 2/60）证明两者量纲不同 —— 这正是「禁止把该分读成语义相似度」的实测依据。

    ⚠️ **与文档的分歧（实测）**：docs/10 §6-R5 与 ``chroma_backend`` 模块 docstring 写
    「hybrid 落在 ~(0, 2/61]（上界 2/61≈0.0328）」，但实测出现 **0.033333**（> 2/61）。
    原因：实现 ``_fuse_rrf`` 用 ``enumerate(ids)``（**0 起**）→ 第 0 名贡献
    ``1/(60+0) = 1/60``，两路相加 = ``2/60 = 1/30``。这与 llama-index
    ``QueryFusionRetriever._reciprocal_rerank_fusion`` 的 ``1.0/(rank + k)``
    （同为 0 起）**逐条一致**，即「与库同源」成立、**文档写的 2/61 才是错的**。
    本用例按实测真值断言 ``(0, 2/60]``。
    """
    case_idx = _case_chroma("hybrid")
    local = _case_local("hybrid")
    achievable = _rrf_achievable_scores()
    # 该组合实测 top-1 两路都排第 0 → RRF = 1/60 + 1/60 = 0.033333（> 文档所写 2/61）
    query = "外观模仿"
    filters = CaseSearchFilters(category=_SHOE_CATEGORY, risk_type=[_IP])
    hits = await case_idx.search(query, filters, top_k=25)
    assert hits
    for h in hits:
        assert 0.0 < h.retrieval_score <= _RRF_MAX + 1e-12, (
            (
                "hybrid 分必须 ⊂ (0, 2/60]（实测上界 2/60；文档所写 2/61 "
                f"已被 0.033333 > {_DOC_CLAIMED_RRF_MAX:.6f} 证伪）"
            ),
            h.case_id,
            h.retrieval_score,
        )
        assert h.retrieval_score in achievable, (
            (
                "hybrid 分必须恰好落在 RRF 可达值集合（1/(60+r) 或其两路之和）—— "
                "落在集合外说明它不再是 RRF 分"
            ),
            h.case_id,
            h.retrieval_score,
        )
    assert any(h.retrieval_score == round(_RRF_MAX, 6) for h in hits), (
        "实测存在两路都排第 0 的命中 → 分恰为 round(2/60, 6) = 0.033333"
        "（这正是文档所写上界 2/61 不成立的实证）"
    )
    # local hybrid 是另一套量纲（0.5·norm(RRF 化前的 bm25) + 0.5·cos）→ 不可比
    local_hits = await local.search(query, filters, top_k=25)
    assert max(h.retrieval_score for h in local_hits) > _RRF_MAX, (
        "local hybrid 量纲 [0,1]，应显著大于 RRF 上界 —— 两者不可互读"
    )


async def test_chroma_retrieval_score_scale_per_mode() -> None:
    """三模式打分口径（C1）：bm25 候选集内 min-max ⊂ [0,1]；vector = 1−distance ⊂ [0,1]。

    - ``bm25``：与 local 用**同一** ``normalize_minmax``，候选集内最高分恒 == 1.0；
      无任何查询词命中时（原始分全 0）归一化给全 1.0（确定性约定，防除零）。
    - ``vector``：Chroma cosine distance → ``1 − distance``，⊂ [0,1]；与 local 余弦
      逐位同口径（尾差 ≤ 1e-6），故这里直接与 local 对读。
    - ``PolicyClauseHit`` **没有** retrieval_score 字段（契约不变，docs/10 §2）——
      「检索分」只出现在 CaseHit 上。
    """
    assert "retrieval_score" in CaseHit.model_fields
    assert "similarity" not in CaseHit.model_fields, (
        "docs/10 §0 C1：CaseHit.similarity 已改名 retrieval_score（禁止以相似度口径描述检索分）"
    )
    assert "retrieval_score" not in PolicyClauseHit.model_fields

    # bm25：候选集内最高分 == 1.0 且 ⊂ [0,1]
    bm25_idx = _case_chroma("bm25")
    bm25_hits = await bm25_idx.search("无品牌高相似商家多次上架", CaseSearchFilters(), top_k=10)
    assert bm25_hits and all(0.0 <= h.retrieval_score <= 1.0 for h in bm25_hits)
    assert bm25_hits[0].retrieval_score == 1.0, "min-max 归一化后候选集最高分必须为 1.0"

    # 零命中查询（latin 乱码：jieba 侧与语料无任何交集 → 原始分全等）→ 归一化给全 1.0
    flat = await bm25_idx.search("zzzqqq wwweee", CaseSearchFilters(risk_type=[_IP]), top_k=5)
    assert flat and {h.retrieval_score for h in flat} == {1.0}

    # vector：与 local 余弦同口径（⊂ [0,1] 且逐条 ≤1e-6）
    vec_idx = _case_chroma("vector")
    local = _case_local("vector")
    vh = await vec_idx.search("无品牌高相似商家多次上架", CaseSearchFilters(), top_k=10)
    lh = await local.search("无品牌高相似商家多次上架", CaseSearchFilters(), top_k=10)
    assert vh and all(0.0 <= h.retrieval_score <= 1.0 for h in vh)
    assert [h.case_id for h in vh] == [h.case_id for h in lh]
    assert all(
        round(abs(a.retrieval_score - b.retrieval_score), 9) <= _SCORE_TOL for a, b in zip(lh, vh)
    )


# ---------------------------------------------------------------------------
# 4) 确定性 + 零 LLM（§5-2 / §3）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["bm25", "vector", "hybrid"])
async def test_chroma_deterministic_byte_identical(mode: str) -> None:
    """同 query 两次 → 结果 ``model_dump(mode="json")`` 逐字节一致（§5-2）。

    含两重：① 同索引两次调用；② **另建一个索引实例**（新 client、同 prefix）——
    后者顺带证明 node id / collection 名不引入顺序漂移。docs/10 §3 明令
    「同分竞争不依赖底层库（Chroma / bm25s）返回顺序」。
    """
    shared_client = chromadb.EphemeralClient()
    prefix = _prefix(f"det{mode}")
    p1 = _policy_chroma(mode, client=shared_client, prefix=prefix)
    p2 = _policy_chroma(mode, client=shared_client, prefix=prefix)
    c1 = _case_chroma(mode, client=shared_client, prefix=prefix + "_c")
    c2 = _case_chroma(mode, client=shared_client, prefix=prefix + "_c")

    p_first = await p1.search("外观高度模仿知名品牌", PolicySearchFilters(), 5, True)
    p_again = await p1.search("外观高度模仿知名品牌", PolicySearchFilters(), 5, True)
    p_fresh = await p2.search("外观高度模仿知名品牌", PolicySearchFilters(), 5, True)
    assert [h.model_dump(mode="json") for h in p_first] == [
        h.model_dump(mode="json") for h in p_again
    ]
    assert [h.model_dump(mode="json") for h in p_first] == [
        h.model_dump(mode="json") for h in p_fresh
    ]

    c_first = await c1.search("无品牌高相似商家多次上架", CaseSearchFilters(), 5)
    c_again = await c1.search("无品牌高相似商家多次上架", CaseSearchFilters(), 5)
    c_fresh = await c2.search("无品牌高相似商家多次上架", CaseSearchFilters(), 5)
    assert [h.model_dump(mode="json") for h in c_first] == [
        h.model_dump(mode="json") for h in c_again
    ]
    assert [h.model_dump(mode="json") for h in c_first] == [
        h.model_dump(mode="json") for h in c_fresh
    ]


async def test_chroma_retrieval_makes_zero_llm_calls() -> None:
    """检索链路**零 LLM**（docs/10 §3 红线）：``served_counters()["llm_calls"] == 0``。

    为什么必须钉住：LlamaIndex ``QueryFusionRetriever`` 的 ``num_queries`` 默认 4 会
    **调用 LLM 生成 query 变体**（引入不确定性 + 需要 LLM）；本实现因此自算 RRF 并只提供
    「``num_queries=1`` + MockLLM 守卫」的构造器。计数器同时要证明检索**真的跑过**
    （否则「0 LLM」是被空跑骗出来的假通过）。
    """
    reset_served_counters()
    for mode in ("bm25", "vector", "hybrid"):
        p = _policy_chroma(mode)
        c = _case_chroma(mode)
        await p.search("外观高度模仿知名品牌", PolicySearchFilters(), 5, True)
        await c.search("无品牌高相似商家多次上架", CaseSearchFilters(), 5)
    counters = served_counters()
    assert counters["llm_calls"] == 0, f"检索链路出现 LLM 调用：{counters}"
    assert counters["vector_searches"] >= 2, counters
    assert counters["bm25_searches"] >= 2, counters
    assert counters["hybrid_searches"] >= 1, counters
    assert counters["served_hits"] > 0, "空跑不算数：必须有真实命中"


# ---------------------------------------------------------------------------
# 5) cosine space 硬要求（§3）：显式配置 + 单位向量数值自检
# ---------------------------------------------------------------------------


def test_chroma_created_collection_is_cosine_and_numerically_verified() -> None:
    """**建库必须显式 cosine —— 本用例是为了「删掉那行配置就红」而存在的**（docs/10 §3）。

    背景（险些静默出错）：Chroma 的**默认向量空间是 l2**，只写 ``embedding_function=None``
    **不会**变成 cosine。实测同一对单位向量 ``[1,0,0]`` vs ``[0.9,0.1,0]``：

    | 建库方式 | space | Chroma distance | ``1 − distance`` vs numpy 余弦 |
    |---|---|---|---|
    | 缺省（无配置） | ``l2`` | ``0.020000005`` | ``0.97999999`` ≠ ``0.99388373`` ❌ |
    | 显式 cosine | ``cosine`` | ``0.006116271`` | ``0.99388373`` ✅ |

    → 「相似度 = 1 − distance」**只在 cosine 空间成立**；L2 库不报错，只会让
    §5-1「与 local 同口径」静默失败。本用例两条独立断言：
    ① 读**真实使用的 collection** 的 ``configuration_json["hnsw"]["space"] == "cosine"``；
    ② 用一条**已知单位向量**查询该库，断言 ``1 − distance`` == 该命中向量与探针的 numpy
    余弦（这才是空间语义的数值证据 —— 光看配置字段不足以证明语义）。

    ⚠️ 探针**只读**（不再往业务库 upsert 探针向量）：实测 chromadb 1.5.9 ``EphemeralClient``
    存在「再 upsert 一批后立刻用接近全库的 ``n_results`` 查询」会少返回一条的写读竞争
    （/tmp 探针实测 7/40 轮：``count()=69`` 而 ``query(n_results=69)`` 只回 68 条，
    新向量不可见）。查询库内**已有**向量没有这个问题（构造后立即检索实测 0/80 轮缺行），
    故探针改成「已知单位向量 → 库里已有的真实 node 向量」的比对。

    （可执行反证见交付报告第 4 项：把同一段探针指向缺省 l2 collection → 两处断言均红。）
    """
    import numpy as np

    client = chromadb.EphemeralClient()
    prefix = _prefix("cosine")
    idx = _case_chroma("vector", client=client, prefix=prefix)
    col = client.get_collection(idx.collection_name)

    # ① 配置层：显式 cosine（缺省 l2 在这里就会露出来）
    assert col.configuration_json["hnsw"]["space"] == "cosine", (
        "建库缺显式 cosine 配置 → Chroma 默认 l2，『相似度 = 1 − distance』静默失效"
    )
    assert idx.space == "cosine", "索引自身记录的空间必须与 collection 一致"
    # 构造期自检：库内首条向量与自身距离 ≈ 0（实测 0.0 / -1.19e-07 float32 尾差）
    assert idx.cosine_self_check_distance is not None
    assert abs(idx.cosine_self_check_distance) <= 1e-6

    # ② 数值层：单位向量探针 vs 库内**真实**向量 —— 1 − distance 必须等于 numpy 余弦。
    # 取与库内向量重合度最高的 one-hot 探针（余弦 > 0.3），避免退化成「0 == 0」的假通过。
    dim = len(idx._doc_vectors[0])  # 探针维度必须与库内维度一致（私有属性：测试直读）
    norms = [float(np.linalg.norm(v)) for v in idx._doc_vectors]
    best_cos, axis = 0.0, 0
    for k in range(dim):
        for i, vec in enumerate(idx._doc_vectors):
            if norms[i] <= 0.0:
                continue
            candidate_cos = vec[k] / norms[i]
            if candidate_cos > best_cos:
                best_cos, axis = candidate_cos, k
    assert best_cos > 0.3, f"找不到非退化探针（最高余弦 {best_cos!r}）—— 语料/tokenizer 变了？"

    probe = [1.0 if i == axis else 0.0 for i in range(dim)]
    res = col.query(query_embeddings=[probe], n_results=1, include=["distances"])
    returned_id, distance = res["ids"][0][0], float(res["distances"][0][0])
    returned_vec = idx._doc_vectors[idx.node_ids.index(returned_id)]
    expected = float(
        np.dot(probe, returned_vec) / (np.linalg.norm(probe) * np.linalg.norm(returned_vec))
    )
    actual = 1.0 - distance
    assert expected > 0.3, "命中向量的余弦过小 → 该对比证明力不足（探针退化）"
    assert abs(actual - expected) <= 1e-6, (
        f"1 − distance = {actual!r} 必须等于 numpy 余弦 {expected!r}"
        "（不等即说明 collection 不是 cosine 空间）"
    )


def test_chroma_existing_l2_collection_is_rejected_not_silently_reused() -> None:
    """已存在的 **l2** collection 同名复用必须**报错**，不得静默复用（docs/10 §3）。

    实测陷阱：对已存在的 l2 库再传 ``configuration={"hnsw":{"space":"cosine"}}``，
    ``get_or_create_collection`` **不会**改建库空间 —— 复用路径不校验就会拿到一个
    「看起来正常、语义却错」的库，错误只会以一个偏小的 ``retrieval_score`` 静默流进
    Evidence.weight。故复用与新建两条路径都校验空间。
    """
    client = chromadb.EphemeralClient()
    name = "l2reuse_policy_256"
    client.get_or_create_collection(name=name, embedding_function=None)  # 缺省 = l2
    with pytest.raises(ValueError, match="cosine"):
        ChromaPolicyIndex(
            POLICY_ROWS,
            embedder=MockHashEmbedder(),
            chroma_client=client,
            collection_prefix="l2reuse",
        )


def test_chroma_dim_mismatch_on_reuse_is_rejected() -> None:
    """同名 collection 但维度不一致 → 拒绝复用（Chroma 不声明维度，读我们写的 ``pra_dim``）。"""
    client = chromadb.EphemeralClient()
    client.get_or_create_collection(
        name="dimmix_case_256",
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"pra_dim": 64},
    )
    with pytest.raises(ValueError, match="维度"):
        ChromaCaseIndex(
            CASE_ROWS,
            embedder=MockHashEmbedder(),
            chroma_client=client,
            collection_prefix="dimmix",
        )


# ---------------------------------------------------------------------------
# 6) 已知偏差（实测，不粉饰）：vector 路在「过滤不可完整下推」时漏召回
# ---------------------------------------------------------------------------

#: vector 路漏召回组合（实测，两处机制）：
#: - ``risk_type`` 完全不能下推（Chroma 1.5.5+ 对列表字段无成员算子、LlamaIndex 的
#:   ANY/CONTAINS 无翻译）→ 取数上限 = 候选数，但库里符合条件的行更多；
#: - ``effective_only=True`` 不在 where 里 → 库内 EXPIRED 行占掉 top-n_results 名额。
_VECTOR_LEAK_POLICY: list[tuple[str, PolicySearchFilters, bool, int]] = [
    ("外观模仿", PolicySearchFilters(risk_type=[_FALSE_CLAIM]), True, 6),   # local 3 → chroma 0
    ("仿冒 高仿 复刻 原单", PolicySearchFilters(risk_type=[_IP]), True, 6),   # local 6 → chroma 4
    ("规避 换链接 改标题 重上架", PolicySearchFilters(risk_type=[_EVASION]), True, 6),
    ("外观模仿", PolicySearchFilters(category=_BAG_CATEGORY), True, 20),      # local 16 → chroma 13
]
_VECTOR_LEAK_CASE: list[tuple[str, CaseSearchFilters, int]] = [
    ("外观模仿", CaseSearchFilters(category=_SHOE_CATEGORY, risk_type=[_IP]), 6),
    ("外观模仿", CaseSearchFilters(risk_type=[_IP]), 30),
]


async def test_vector_mode_recall_leak_policy_documented_truth() -> None:
    """**实测偏差**：policy vector 路在 risk_type / effective_only 不可下推时会漏召回。

    本用例断言的是**实测事实**（不是预期行为），因为 docs/10 §5-1 表格给 vector 的
    断言是「与 local 同序同 id + 打分一致」，而以下组合**做不到**：

    | 组合 | local | Chroma vector |
    |---|---|---|
    | ``q=外观模仿`` + risk_type=[FALSE_CLAIM], eff=True, k=6 | 3 | **0**（整组丢光 —— 最坏形态） |
    | ``q=仿冒 高仿 复刻 原单`` + risk_type=[POTENTIAL_IP_RISK], k=6 | 6 | **4** |
    | ``q=规避 换链接 改标题 重上架`` + risk_type=[EVASION_PATTERN], k=6 | 6 | **5** |
    | ``q=外观模仿`` + category=箱包/女包, eff=True, k=20 | 16 | **13** |

    机制：``_retrieve_ranked`` 取 ``n_results = len(candidates)``，但**下推的 where 只是
    候选谓词的真超集**（risk_type 不推、effective_only 不推）→ 库里多出的行按距离挤进
    top-n_results，再被 Python 侧 ``recheck`` 剔除，于是结果**不足** min(top_k, 候选数)，
    极端情况（非候选行恰好排满 top-n_results）下**一条都不剩**。
    实现 docstring 声称「取候选数上限即可避免结果不足」，实测**仅在 where ≡ 谓词时成立**。

    断言的稳定性质（不会因实现修好以外的原因抖动）：① 命中是 local 的**真子集**（只漏召回、
    不会凭空多出；允许为空集）；② 相对顺序与 local 一致；③ 结果数 < min(top_k, 可用候选数)。
    ⚠️ 若实现改为「按全库取数或补齐 where」，本用例会失败 —— 届时**删掉本用例**并把
    §5-1 的 vector 行恢复为无条件成立（这是本用例存在的意义，勿改断言迁就）。
    """
    local = _policy_local("vector")
    idx = _policy_chroma("vector")
    empty_leaks = 0
    for query, filters, effective_only, top_k in _VECTOR_LEAK_POLICY:
        lh = await local.search(query, filters, top_k, effective_only)
        ch = await idx.search(query, filters, top_k, effective_only)
        lid, cid = [h.clause_id for h in lh], [h.clause_id for h in ch]
        empty_leaks += 1 if (lid and not cid) else 0
        assert set(cid) < set(lid), (
            f"q={query!r} filters={filters} eff={effective_only}: 期望「local 的真子集」"
            f"（漏召回），实际 local={lid} chroma={cid}"
        )
        assert [i for i in lid if i in set(cid)] == cid, "漏召回不得改变剩余命中的相对顺序"
        eligible = _eligible_policy(filters, effective_only)
        assert len(cid) < min(top_k, eligible), (
            f"候选不完整：{len(cid)} < min({top_k}, {eligible}) —— 这正是与 §5-1 的冲突点"
        )
    assert empty_leaks >= 1, (
        "实测存在「候选非空但 Chroma vector 返回空结果」的组合（risk_type=[FALSE_CLAIM]）——"
        "这是漏召回的最坏形态，必须如实钉住"
    )


async def test_vector_mode_recall_leak_case_documented_truth() -> None:
    """**实测偏差**：case vector 路同样在 risk_type 不可下推时漏召回（机制同 policy）。

    实测：``category=女鞋/运动鞋 + risk_type=[IP], k=6`` → local 6 / Chroma **5**；
    ``risk_type=[IP], k=30`` → local 25 / Chroma **14**（候选 25 条而库里匹配 67 条，
    取数上限 25 被非候选行占满）。断言性质与 policy 用例相同（真子集 + 序一致 + 结果不足）。
    """
    local = _case_local("vector")
    idx = _case_chroma("vector")
    for query, filters, top_k in _VECTOR_LEAK_CASE:
        lh = await local.search(query, filters, top_k)
        ch = await idx.search(query, filters, top_k)
        lid, cid = [h.case_id for h in lh], [h.case_id for h in ch]
        assert ch, f"漏到空结果已属更严重回归：q={query!r} filters={filters}"
        assert set(cid) < set(lid), (
            f"q={query!r} filters={filters} k={top_k}: 期望「local 的真子集」，"
            f"实际 local={len(lid)} chroma={len(cid)}"
        )
        assert [i for i in lid if i in set(cid)] == cid
        assert len(cid) < min(top_k, _eligible_case(filters))


# ---------------------------------------------------------------------------
# 7) 元数据过滤语义（与 local / InMemory 逐条一致，直接对 chroma 索引断言行为）
# ---------------------------------------------------------------------------


async def test_chroma_policy_version_and_category_semantics() -> None:
    """policy 过滤语义：``effective_only`` 开关、全类目通配、risk_type 交叠。

    用 ``hybrid`` 模式（候选完整，见第 2 节）避免与 vector 路的漏召回偏差纠缠 ——
    本用例测的是**过滤语义**，不是取数完整性。
    """
    idx = _policy_chroma("hybrid")
    query = "永久去皱 根治脚气 功效夸大"  # 命中 POLICY_2.1 v1(EXPIRED) 文案

    eff = await idx.search(query, PolicySearchFilters(), top_k=10, effective_only=True)
    assert eff and all(h.status == "EFFECTIVE" for h in eff)
    assert not any(h.clause_id == _EXPIRED_OLD_CLAUSE for h in eff), (
        "effective_only=True 必须排除 EXPIRED 旧版"
    )

    both = await idx.search(query, PolicySearchFilters(), top_k=10, effective_only=False)
    assert any(h.clause_id == _EXPIRED_OLD_CLAUSE for h in both), (
        "effective_only=False 应含 EXPIRED 旧版（历史版本可检索）"
    )

    # category：具体类目 + 全类目都留（row.category ∈ {None, 目标, 全类目}）
    bag = await idx.search(
        "外观模仿 品牌", PolicySearchFilters(category=_BAG_CATEGORY), top_k=20, effective_only=True
    )
    assert bag and all(h.category in (_BAG_CATEGORY, _FULL_CATEGORY) for h in bag)
    assert any(h.category == _FULL_CATEGORY for h in bag), "全类目条款照常匹配（通配语义）"
    assert not any(h.category == _SHOE_CATEGORY for h in bag), "其它具体类目条款不得出现"

    # risk_type：交叠非空（给了 filter 时空 risk_type 行被排除）
    ip = await idx.search("仿冒", PolicySearchFilters(risk_type=[_IP]), top_k=20, effective_only=True)
    assert ip and all(_IP in h.risk_type for h in ip)

    # 候选规则：**「不存在的类目」在 policy 侧不是空候选** —— 全类目条款仍命中
    #（与 local 逐条一致的既有语义；case 侧才是精确匹配 → []，见下一用例）
    missing = await idx.search(
        "外观模仿", PolicySearchFilters(category=_MISSING_CATEGORY), top_k=20, effective_only=True
    )
    assert missing and all(h.category == _FULL_CATEGORY for h in missing), (
        "policy 的『全类目』是通配：不存在的类目仍命中全类目条款（不是空候选）"
    )


async def test_chroma_case_metadata_semantics_and_empty_candidates() -> None:
    """case 过滤语义：category 精确匹配、risk_type 交叠、不存在的类目 → ``[]``。

    ``CaseHit`` 不带 category 字段（过滤在索引侧完成）→ 用 corpus 侧子集反查命中集合。
    """
    idx = _case_chroma("hybrid")
    shoe_ids = {c.case_id for c in CASE_ROWS if c.category == _SHOE_CATEGORY}
    hits = await idx.search("无品牌高相似", CaseSearchFilters(category=_SHOE_CATEGORY), top_k=10)
    assert hits and {h.case_id for h in hits} <= shoe_ids, "category 必须精确匹配"

    ip = await idx.search(
        "外观模仿", CaseSearchFilters(category=_SHOE_CATEGORY, risk_type=[_IP]), top_k=10
    )
    assert ip and all(_IP in h.risk_type for h in ip)
    assert {h.case_id for h in ip} <= shoe_ids

    none = await idx.search("任何词", CaseSearchFilters(category="不存在/类目"), top_k=5)
    assert none == [], "case 候选为空 → 返回 []（合法空结果，工具 ok=True）"

    # policy 侧同形：非法类目下候选非空（全类目通配）→ 但 top_k < 1 恒返回 []
    p_idx = _policy_chroma("hybrid")
    assert await p_idx.search("外观模仿", PolicySearchFilters(), top_k=0, effective_only=True) == []


# ---------------------------------------------------------------------------
# 8) 装配开关：factory backend="chroma" + build_tools rag_backend="chroma"
# ---------------------------------------------------------------------------


def test_factory_backend_chroma_returns_chroma_index() -> None:
    """factory ``backend="chroma"``：延迟 import 构造 ``Chroma*Index``（离线注入 EphemeralClient）。"""
    client = chromadb.EphemeralClient()
    prefix = _prefix("factory")
    p_idx = build_policy_index(
        rows=POLICY_ROWS, embedder=MockHashEmbedder(), backend="chroma",
        chroma_client=client, collection_prefix=prefix,
    )
    c_idx = build_case_index(
        rows=CASE_ROWS, embedder=MockHashEmbedder(), backend="chroma",
        chroma_client=client, collection_prefix=prefix,
    )
    assert isinstance(p_idx, ChromaPolicyIndex)
    assert isinstance(c_idx, ChromaCaseIndex)


async def test_build_tools_rag_backend_chroma_injects_chroma_index() -> None:
    """``build_tools("rag", rag_backend="chroma")`` 注入 Chroma 索引；其余 4 工具仍是 InMemory。"""
    client = chromadb.EphemeralClient()
    tools = build_tools(
        "rag",
        rag_backend="chroma",
        rag_embedder=MockHashEmbedder(),
        rag_backend_options={"chroma_client": client, "collection_prefix": _prefix("tools")},
    )
    assert [t.name for t in tools] == [
        "ProductTool", "ImageAnalysisTool", "OCRTool", "MerchantTool",
        "CaseSearchTool", "PolicySearchTool",
    ]
    assert isinstance(tools[4]._index, ChromaCaseIndex)  # 注入点直读
    assert isinstance(tools[5]._index, ChromaPolicyIndex)
    assert type(tools[0]._repo).__name__ == "InMemoryProductRepository"
    hits = await tools[4]._index.search("无品牌高相似", CaseSearchFilters(), top_k=3)
    assert hits and all(str(h.case_id).startswith("RAG_CASE_") for h in hits)
