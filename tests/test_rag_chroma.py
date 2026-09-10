"""Chroma 后端离线测试（tests/test_rag_chroma.py）—— ``EphemeralClient`` 内存库，**零网络**。

覆盖 docs/10-rag-upgrade-spec.md（唯一实施契约）§3（实测钉死的实现契约）与 §5-1
（**按模式分别断言**的同构契约）：

1. **协议形状**：``ChromaPolicyIndex.search(query, filters, top_k, effective_only)`` /
   ``ChromaCaseIndex.search(query, filters, top_k)`` 返回 ``PolicyClauseHit`` /
   ``CaseHit``（tools 层 Protocol 未被改动 —— docs/10 §2 铁律）。
2. **同构等价按模式分别断言**（§5-1 表格，初稿「三种模式同序同 id」曾被实测改写成
   「只在下推 ≡ 谓词时成立」；**漏召回 bug 已修**，故 vector 行恢复为**无条件**成立，
   且断言覆盖**过滤组合**，见第 2 节与第 6 节）：
   - ``vector``：与 local 后端**同序同 id** + 打分 6 位一致（容差 1e-6，Chroma 存算
     float32 的尾差）—— 组合表含 ``risk_type`` / ``effective_only`` / category 过滤；
   - ``bm25`` / ``hybrid``：**不可比**（本地 = 自写 Okapi + CJK 字符 bigram + 全库 IDF；
     Chroma 路 = ``bm25s`` + jieba 真词 + 语料 = 候选 node；且 hybrid 本地 = 0.5·norm(bm25)
     + 0.5·cos、量纲 [0,1]，Chroma 路 = RRF ``Σ1/(60+rank)``、量纲 ~0.0167–0.0333）。
     故按 §5-1 预期只断言「候选完整（结果长度 == min(top_k, 可用候选数)）、可复现、
     R-4 隔离」，**不**断言与 local 同序 —— 这里显式写下「不比对」的理由，防止后来者
     把 bm25/hybrid 的差异误读成回归。
3. **``retrieval_score`` 语义（C1）**：bm25 = 候选集内 min-max 归一化（⊂ [0,1]）；
   vector = ``1 − distance`` 余弦（⊂ [0,1]）；**hybrid = RRF 融合分**（``Σ1/(60+rank)``，
   恒 ⊂ ``(0, 2/60]`` —— rank **从 0 起**，故上界是 ``2/60 = 1/30``，不是 ``2/61``；
   实现 docstring 与本套件均已按实测更正）—— 任何场合都不得表述成「语义相似度」
   （docs/10 §0 C1 / §6-R5）。
4. **确定性**：同 query 两次 ``model_dump(mode="json")`` 逐字节一致；检索链路
   ``llm_calls == 0``（检索侧零 LLM，docs/10 §3）。
5. **cosine space 硬要求（§3）**：建库必须**显式** cosine（Chroma 缺省 l2 会让
   「相似度 = 1 − distance」静默失效 —— 这正是本文件要钉住的坑），并做单位向量数值自检。
6. **两道回归闸（本轮补，都是「上一版没测所以漏了」的直接产物）**：
   - **向量路候选完整性**：vector 取数改为「精确候选 id 集 + 覆盖率自检 + 已存向量兜底」，
     本文件断言**过滤组合**下同序同 id、结果长度 == ``min(top_k, 可用候选数)``，
     且 ``served_counters()["vector_bruteforce_fallbacks"] == 0``（兜底没被默默用上）；
   - **R7：BM25 只索引正文**。⚠️ **不要把断言写成「命中 0 条」**（那是错的，会让正确实现变红）：
     ``BM25Retriever`` 对全零分查询仍按 ``similarity_top_k`` 返回节点，且仓库既有 ``normalize_minmax``
     对「空/等值集」按既定约定给全 1.0 → 实测 ``search("RAG_CASE_0037", CaseSearchFilters(), 5)``
     **返回 5 条、分全 1.0**（``[('RAG_CASE_0001', 1.0), ('RAG_CASE_0002', 1.0), …]``）。
     正确口径是「**没有信号**」，用两层断言：
     ① 机制级（定向、便宜）``test_chroma_build_nodes_embed_text_is_body_only``：``_build_nodes`` 产出的
        node 其 ``get_content(metadata_mode=EMBED)`` 逐字等于正文、不含任何 metadata 字面值，且
        **``node_to_metadata_dict`` → ``metadata_dict_to_node`` 往返后仍成立**（``BM25Retriever`` 正
        这样重建节点）；``test_chroma_index_nodes_carry_metadata_exclusions`` 再证明索引构造路径确实用了这套 node；
     ② 黑盒判别式 ``test_chroma_*_bm25_ignores_metadata_literals``：纯 metadata 字面值查询的结果必须与
        「无信号乱码查询」**逐字节一致**，且该字面值的 BM25 **原始分恒 0**（正文查询作对照：原始分 > 0 且分数非平坦）。
     ⚠️ 反例字面值必须**token 级**不在正文里：``全类目`` / ``女鞋/运动鞋`` / ``箱包/女包`` 本来就出现在正文
     （命中是正确行为），``POLICY_5.3`` 会被 jieba 拆出单字符 ``'3'`` 命中「≥3 次」表述 —— 用例内会现场自检。
7. **R6：BM25 分词器受控替换的并发正确性**（第 9 节，docs/10 §6-R6）。``llama-index-retrievers-bm25
   0.8.0`` 无 tokenizer 注入点 → 本模块只能**受控替换** ``bm25s.tokenize``（模块级全局符号）并用
   ``_TOKENIZER_LOCK`` 串行化；因此「补丁的安装/撤销是否被锁覆盖」是**必须实测**的性质：
   - **主闸（黑盒）**：8 线程并发跑 ``mode="bm25"`` 检索 → 无异常、每轮结果 == 单线程 golden
     （id 序 + 6 位分）、结束后 ``bm25s.tokenize`` **复原为原符号**（不泄漏）；
   - **机制级定位工具**：观测代理锁断言「取锁那一刻补丁**尚未**安装」（补丁必须在锁内装）。
   ⚠️ **不得**据此声称该实现「天然线程安全」—— 它仍是全局符号替换，**已验证边界**见 docs/10 §6-R6
   （同进程内本模块的调用点互不干扰；非本模块代码在持锁窗口内直接调 ``bm25s.tokenize`` 仍会看到替身）。
   实测（2026-09-10）：修复前 8×15 轮**每次运行**都复现（120 轮里 1 次 ``ValueError`` + 符号泄漏），
   修复后 3/3 运行干净 —— 本套件的两道断言正是据此写的。

**在 CI 上本文件整文件 skip（诚实标注，勿声称 CI 覆盖）**：顶层
``pytest.importorskip("chromadb")``，而 CI 只跑 ``uv sync --frozen``（**不装任何 extra**）
→ chromadb / llama-index 都不在。CI 上真正跑得动的默认路径守护在
``tests/test_rag_default_path_no_extra.py``（docs/10 §5-5）。

约定（对齐 tests/test_rag_qdrant.py）：零网络、零模型下载（``EphemeralClient`` +
``MockHashEmbedder``）；每个用例用**独立 client + uuid 后缀 prefix**，互不干扰。
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Self
from uuid import uuid4

import pytest

# 无 chromadb 整文件跳过（CI 只跑 uv sync --frozen，不装 rag extra → 正是这种情况）
chromadb = pytest.importorskip(
    "chromadb",
    reason="未安装 chromadb（CI 只跑 uv sync --frozen，不装 rag extra）→ 整文件跳过；装上后：uv sync --extra rag",
)
bm25s = pytest.importorskip(
    "bm25s",
    reason="未安装 bm25s（llama-index-retrievers-bm25 的引擎，随 rag extra 装入）→ 整文件跳过",
)

from pra.domain.models import RiskType
from pra.rag import chroma_backend
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
from pra.rag.vectors import cosine_similarity
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
#: 上一轮被证伪、后由实现与本套件一并更正的历史上界（``2/61``）—— 仅作对照常量保留，
#: 断言**不再**引用它（真值是 :data:`_RRF_MAX`）。
_DOC_CLAIMED_RRF_MAX_LEGACY = 2.0 / 61.0

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

#: 进入任何检索之前的 ``bm25s.tokenize`` 原符号（R6：整个模块「受控替换」它，
#: 用例断言跑完必须复原为这一份；本模块锁的观测代理也用它判别「补丁是否已安装」）。
_REAL_BM25S_TOKENIZE = bm25s.tokenize


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

#: **vector 模式与 local 的同序同 id 组合（含过滤组合）** —— 逐条参数化，**不再只是「干净查询」**。
#:
#: 为什么必须带过滤组合（2026-09-10 缺陷复盘）：最初只测「无 risk_type + 下推 ≡ 谓词」的组合，
#: 于是 vector 路的漏召回 bug 全绿通过 —— 只有当 **下推的 where 只是候选谓词的真超集** 时
#: 才会暴露（``risk_type`` 无法下推：Chroma 列表字段没有成员算子；``effective_only`` 也不在
#: where 里）。当时实测：policy ``外观模仿``+risk_type=[FALSE_CLAIM]+eff k=6 → local 3 / chroma **0**；
#: policy 品牌词+eff（**无任何 store 过滤**）k=30 → local 21 / chroma **19**（EXPIRED 行抢位）；
#: case 外观模仿+risk_type=[IP] k=30 → local 25 / chroma **14**。
#: 这些组合现在全部逐条断言（下面每条都标注了「修复前的实测值」）。
_VECTOR_PARITY_POLICY: list[tuple[str, str, PolicySearchFilters, bool, int]] = [
    # (用例后缀, query, filters, effective_only, top_k)
    ("no_filter_eff", "外观高度模仿知名品牌无授权", PolicySearchFilters(), True, 5),
    ("no_filter_all", "永久去皱 根治脚气 功效夸大", PolicySearchFilters(), False, 10),
    ("category_only", "外观模仿", PolicySearchFilters(category=_BAG_CATEGORY), False, 20),
    # ↓ 修复前：local 21 / chroma 19（EXPIRED 行抢占 n_results 名额，无任何 store 过滤也中招）
    ("brand_eff_k30", "外观高度模仿知名品牌无授权", PolicySearchFilters(), True, 30),
    # ↓ 修复前：local 3 / chroma 0（risk_type 不可下推 → 非候选行排满 top-N，真候选一条没取到）
    ("risk_type_false_claim", "外观模仿", PolicySearchFilters(risk_type=[_FALSE_CLAIM]), True, 6),
    # ↓ 修复前：local 6 / chroma 4
    ("risk_type_ip", "外观模仿", PolicySearchFilters(risk_type=[_IP]), True, 6),
    # ↓ 修复前：local 6 / chroma 5
    ("risk_type_evasion", "规避 换链接 重上架", PolicySearchFilters(risk_type=[_EVASION]), True, 6),
    # ↓ 修复前：local 16 / chroma 13（保住了前缀但漏了 3 条）
    ("category_eff_k20", "外观模仿", PolicySearchFilters(category=_BAG_CATEGORY), True, 20),
    ("category_plus_risk_type", "外观模仿",
     PolicySearchFilters(category=_SHOE_CATEGORY, risk_type=[_IP]), True, 20),
    ("missing_category", "不存在类目 数码3C", PolicySearchFilters(category=_MISSING_CATEGORY), True, 20),
]
_VECTOR_PARITY_CASE: list[tuple[str, str, CaseSearchFilters, int]] = [
    ("no_filter", "无品牌高相似商家多次上架", CaseSearchFilters(), 5),
    ("category_only", "无品牌高相似商家多次上架", CaseSearchFilters(category=_SHOE_CATEGORY), 5),
    ("category_all_candidates", "外观模仿", CaseSearchFilters(category=_BAG_CATEGORY), 30),
    ("no_match_category", "食品 不存在类目", CaseSearchFilters(category="食品"), 5),
    # ↓ 修复前：local 25 / chroma 14（候选 25 条而库里匹配 67 条，取数上限被非候选行占满）
    ("risk_type_k30", "外观模仿", CaseSearchFilters(risk_type=[_IP]), 30),
    # ↓ 修复前：local 6 / chroma 5
    ("category_plus_risk_type", "外观模仿",
     CaseSearchFilters(category=_SHOE_CATEGORY, risk_type=[_IP]), 10),
]


@pytest.mark.parametrize(
    ("label", "query", "filters", "effective_only", "top_k"),
    _VECTOR_PARITY_POLICY,
    ids=[case[0] for case in _VECTOR_PARITY_POLICY],
)
async def test_chroma_policy_vector_equivalent_to_local(
    label: str, query: str, filters: PolicySearchFilters, effective_only: bool, top_k: int
) -> None:
    """**vector 模式 policy：与 local 同序同 id + 打分 6 位一致（§5-1 第一行，无条件）**。

    vector 路走 Chroma 原生 ``collection.query`` → ``1 − distance``（**不是** ``exp(-distance)``，
    docs/10 §3 实测），排序 tie-break = 「分降序 + corpus 原序」→ 与 local ``rank_documents``
    同口径；分差只来自 Chroma float32 存算尾差（⊂ 1e-6，故这里对 policy 只断 id 序）。

    组合表覆盖**过滤组合**（``risk_type`` / ``effective_only`` / category，见
    ``_VECTOR_PARITY_POLICY`` 注释里的「修复前实测值」）——「干净查询」测不出漏召回，
    这正是当初漏掉这个 bug 的原因。同时断言结果长度 == ``min(top_k, 可用候选数)``
    （候选完整，不只是「同 id 截断后的前缀」）。
    """
    local = _policy_local("vector")
    chroma_idx = _policy_chroma("vector")
    lh = await local.search(query, filters, top_k, effective_only)
    ch = await chroma_idx.search(query, filters, top_k, effective_only)
    assert [h.clause_id for h in lh] == [h.clause_id for h in ch], (
        f"policy vector[{label}] q={query!r} filters={filters} eff={effective_only} k={top_k}: "
        f"与 local 命中序/id 不一致\nlocal ={[(h.clause_id, getattr(h, 'retrieval_score', '-')) for h in lh]}\n"
        f"chroma={[h.clause_id for h in ch]}"
    )
    eligible = _eligible_policy(filters, effective_only)
    assert len(ch) == min(top_k, eligible), (
        f"policy vector[{label}]: 候选不完整 {len(ch)} != min({top_k}, {eligible})"
    )
    assert all(h.status == "EFFECTIVE" for h in ch) if effective_only else True


@pytest.mark.parametrize(
    ("label", "query", "filters", "top_k"),
    _VECTOR_PARITY_CASE,
    ids=[case[0] for case in _VECTOR_PARITY_CASE],
)
async def test_chroma_case_vector_equivalent_to_local(
    label: str, query: str, filters: CaseSearchFilters, top_k: int
) -> None:
    """**vector 模式 case：与 local 同序同 id + ``retrieval_score`` 差 ≤ 1e-6（无条件）**。

    与 policy 同理，组合覆盖 ``risk_type`` / category 过滤（``_VECTOR_PARITY_CASE`` 注释里
    标注了每条修复前的实测偏差）；额外断言结果长度 == ``min(top_k, 可用候选数)``。
    """
    local = _case_local("vector")
    chroma_idx = _case_chroma("vector")
    lh = await local.search(query, filters, top_k)
    ch = await chroma_idx.search(query, filters, top_k)
    assert [h.case_id for h in lh] == [h.case_id for h in ch], (
        f"case vector[{label}] q={query!r} filters={filters} k={top_k}: 与 local 命中序/id 不一致"
    )
    eligible = _eligible_case(filters)
    assert len(ch) == min(top_k, eligible), (
        f"case vector[{label}]: 候选不完整 {len(ch)} != min({top_k}, {eligible})"
    )
    for a, b in zip(lh, ch):
        assert round(abs(a.retrieval_score - b.retrieval_score), 9) <= _SCORE_TOL, (
            "Chroma float32 存算尾差应 ⊂ 1e-6（实测 case 分差 [0,0,0,0,1e-6]）",
            label,
            a.case_id,
            a.retrieval_score,
            b.retrieval_score,
        )


async def test_chroma_vector_path_covers_candidates_without_bruteforce_fallback() -> None:
    """**vector 路按「精确候选 id 集」取数，正常路径不得动用兜底**（漏召回 bug 的回归闸）。

    修复后的机制（docs/10 复盘）：候选 node id 集合直接交给 Chroma ``ids=``（``where`` 仍下推，
    但只作收窄），``n_results = min(候选数, collection.count())`` → 「返回 ⊇ 候选」由构造保证；
    另有两道保险：覆盖率自检（重试 3 次后仍缺即抛 ``RuntimeError``）与**按已存向量补算 cos 的兜底**
    （``served_counters()["vector_bruteforce_fallbacks"]``）。

    本用例断言的是「**兜底没有被默默用上**」：risk_type 过滤（正是修复前会漏的那类查询）下
    ① 结果长度 == min(top_k, 可用候选数)、② 分数与 local 逐条 ≤1e-6（若走了兜底补算，
    分数仍可能接近但会经纯 Python 余弦而非 Chroma float32 —— 更重要的是
    ③ ``vector_bruteforce_fallbacks`` 必须恒为 **0**：一旦它变成非 0，说明精确取数/覆盖率那道闸
    在真实环境里失效了，结果正确只是「兜底替它干活」的假象。
    """
    reset_served_counters()
    idx = _policy_chroma("vector")
    local = _policy_local("vector")
    probes = [
        ("外观模仿", PolicySearchFilters(risk_type=[_FALSE_CLAIM]), True, 6),
        ("外观模仿", PolicySearchFilters(risk_type=[_IP]), True, 6),
        ("规避 换链接 重上架", PolicySearchFilters(risk_type=[_EVASION]), True, 6),
        ("外观模仿", PolicySearchFilters(category=_BAG_CATEGORY), True, 20),
    ]
    for query, filters, effective_only, top_k in probes:
        ch = await idx.search(query, filters, top_k, effective_only)
        lh = await local.search(query, filters, top_k, effective_only)
        assert len(ch) == min(top_k, _eligible_policy(filters, effective_only)), (
            f"q={query!r} filters={filters}: 候选不完整（结果靠兜底补齐即说明精确取数失效）"
        )
        assert [h.clause_id for h in ch] == [h.clause_id for h in lh]
    counters = served_counters()
    assert counters["vector_bruteforce_fallbacks"] == 0, (
        "vector 路动用了「按已存向量补算」兜底 —— 精确 id 取数 / 覆盖率自检在真实环境失效："
        f"{counters}"
    )
    assert counters["vector_searches"] >= len(probes), f"空跑不算数：{counters}"


def test_chroma_vector_bruteforce_fallback_scores_missing_candidates() -> None:
    """兜底路径**本身可用**（直接调用，绕开「让 Chroma 少返」这种不可控触发）。

    修复引入的最后一道防线：万一覆盖率自检 3 次重试后仍缺候选，``_rank_vector`` 会改走
    ``_score_missing_by_stored_vectors``（按 Chroma 里**已存的 doc 向量**现算 ``cosine_similarity``）。
    这段代码正常路径**永不执行**（上面那条 ``fallbacks == 0`` 就是证明），因此它的正确性只能直测：

    ① 补算出的分 == 用同一 embedder 现算的余弦（口径与 local 同源，不是第二套公式）；
    ② 取不到向量的 id → **抛 RuntimeError，绝不静默少返**（静默少返正是被修掉的那个 bug）。
    """
    idx = _case_chroma("vector")
    sub_ctx = idx._sub_context([0, 5, 7])
    query_bundle = idx._llama["QueryBundle"](query_str="外观模仿")
    missing = [idx.node_ids[0], idx.node_ids[7]]
    scored = idx._score_missing_by_stored_vectors(sub_ctx, query_bundle, missing)
    assert set(scored) == set(missing)
    embedder = MockHashEmbedder()
    query_vec = embedder.embed("外观模仿")
    for node_id, score in scored.items():
        row = CASE_ROWS[idx.node_ids.index(node_id)]
        expected = cosine_similarity(query_vec, embedder.embed(row.summary))
        assert abs(score - expected) <= _SCORE_TOL, (
            node_id,
            score,
            expected,
        )

    with pytest.raises(RuntimeError, match="兜底失败|拒绝返回子集"):
        idx._score_missing_by_stored_vectors(sub_ctx, query_bundle, ["pra-不存在的-node-id"])


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

    上界的来历（**本轮已修文档**）：rank 从 **0** 起（``_fuse_rrf`` 用 ``enumerate(ids)``）→ 第 0 名
    贡献 ``1/(60+0) = 1/60``，两路都排首位即 ``2/60 = 1/30 ≈ 0.03333``（实测 0.033333），
    **不是 2/61**。实现（``_fuse_rrf`` / ``_rank_hybrid`` / ``ChromaCaseIndex`` 三处 docstring）
    与本测试套件上一轮都按实测值更正为 ``2/60``；上与 llama-index
    ``QueryFusionRetriever._reciprocal_rerank_fusion`` 的 ``1.0/(rank + k)``（同 0 起）逐条一致。
    本用例把该口径钉住：任何把它写回 ``2/61`` 的改动都会让「实测存在 == round(2/60, 6) 的命中」变红。
    """
    case_idx = _case_chroma("hybrid")
    local = _case_local("hybrid")
    achievable = _rrf_achievable_scores()
    # 该组合实测 top-1 两路都排第 0 → RRF = 1/60 + 1/60 = 0.033333（= 上界 2/60）
    query = "外观模仿"
    filters = CaseSearchFilters(category=_SHOE_CATEGORY, risk_type=[_IP])
    hits = await case_idx.search(query, filters, top_k=25)
    assert hits
    for h in hits:
        assert 0.0 < h.retrieval_score <= _RRF_MAX + 1e-12, (
            "hybrid 分必须 ⊂ (0, 2/60]（rank 从 0 起 → 两路首位相加 = 2/60 = 0.0333）",
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
        "实测存在两路都排第 0 的命中 → 分恰为 round(2/60, 6) = 0.033333（上界即 2/60，不是 2/61）"
    )
    assert max(h.retrieval_score for h in hits) > _DOC_CLAIMED_RRF_MAX_LEGACY, (
        "实测最高分必须 > 历史文档所写的 2/61 —— 这条同时钉住「上界写 2/61 是错的」这一更正依据"
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
# 6) R7 守卫：BM25 索引文本 = 正文，metadata 字面值不得成为检索信号
# ---------------------------------------------------------------------------

#: 纯 metadata 字面值查询（**修复前实测命中 67/67 case、24/24 policy**）。
#: 取自 node metadata 的实际取值：``case_id`` / ``risk_type`` / ``decision`` / ``risk_level``
#: （policy 侧另有 ``clause_id`` / ``status``）。
#: ⚠️ **挑选纪律（实测踩过）**：不能凭「看起来像 metadata」就拿来当反例 —— 必须是
#: **token 级**在正文里不出现的字面值（下面每个用例都会现场自检这一点）：
#: - 类目字面值 ``女鞋/运动鞋`` / ``箱包/女包`` 与 ``全类目`` 本来就出现在正文里
#:   （实测 case 正文含 ``运动鞋``，policy 正文含 ``类目``）→ 命中是**正确行为**，不能当反例；
#: - ``POLICY_5.3`` 会被 jieba 拆成 ``['policy', '_', '5', '3', '5.3']``，其中单字符 ``'3'``
#:   命中政策正文里的「≥3 次」等表述 → 原始分非 0 属**分词假阳性**，与 R7 无关（已剔除）。
_CASE_METADATA_LITERALS = ("RAG_CASE_0037", "RAG_CASE_0001", "POTENTIAL_IP_RISK",
                           "EVASION_PATTERN", "REJECT", "HUMAN_REVIEW", "PASS", "HIGH", "LOW")
_POLICY_METADATA_LITERALS = ("POLICY_1.1_v2_c1", "POLICY_1.5", "POTENTIAL_IP_RISK",
                             "EVASION_PATTERN", "REJECT", "EFFECTIVE", "EXPIRED",
                             # ``RAG_CASE_0037`` 不是 policy 的 metadata 取值，列在此处只作**跨 KB 对照**：
                             # 「与本 KB 任何字段都无关的字面值同样不得产生信号」（与零信号基线同效）。
                             "RAG_CASE_0037")

#: 「毫无信号」的对照查询：latin 乱码，与全库正文/词表零交集（实测原始分恒 0）。
#: 口径与主 agent 的独立实测一致（``"zzzqqqxxx"``）；本套件另验证过 ``"zzzqqq wwweee"`` 同效。
_NO_SIGNAL_QUERY = "zzzqqqxxx"


def _jieba_tokens_of(text: str) -> list[str]:
    """与被测 BM25 索引**同一分词器**切词（复用模块内 ``_jieba_tokens``，不自造第二套）。"""
    from pra.rag.chroma_backend import _jieba_tokens

    return _jieba_tokens(text)


def _body_tokens(rows: list[Any], *, kind: str) -> set[str]:
    """该 KB **正文**（BM25 索引文本）的 jieba 词集合 —— 用于自检「反例确实不在正文里」。"""
    from pra.rag.chroma_backend import _jieba_tokens

    text = " ".join(
        (f"{r.title}。{r.text}" if kind == "policy" else r.summary) for r in rows
    )
    return set(_jieba_tokens(text))


def _bm25_raw_scores(idx: object, candidates: list[int], query: str) -> list[float]:
    """直接向 BM25 检索器要**原始分**（不经 ``normalize_minmax``）—— 「有没有信号」的直接证据。

    走模块内部件（``_sub_context`` / ``_make_bm25_retriever`` / ``_bm25_retrieve``）是**有意为之**：
    ``search()`` 的分数已被归一化成 [0,1]，而「纯 metadata 字面值是否命中」这件事只体现在原始分上；
    R7 的验收口径本就该看检索器视野里有没有这个词，而不是看归一化后的包装值。
    """
    from pra.rag.chroma_backend import _make_bm25_retriever

    sub_ctx = idx._sub_context(candidates)  # 测试内省（有意读私有件，见 docstring）
    query_bundle = idx._llama["QueryBundle"](query_str=query)
    retriever = _make_bm25_retriever(sub_ctx, len(candidates))
    nodes = idx._bm25_retrieve(retriever, query_bundle)
    return [float(node.score or 0.0) for node in nodes]


@pytest.mark.parametrize("literal", _CASE_METADATA_LITERALS)
async def test_chroma_case_bm25_ignores_metadata_literals(literal: str) -> None:
    """**R7 守卫（case）**：纯 metadata 字面值查询在 ``bm25`` 模式下**不产生任何检索信号**。

    背景（R7 用户拍板，实现 ``_build_nodes`` 用 ``TextNode(excluded_embed_metadata_keys=...)``）：
    ``BM25Retriever`` 内部索引的是 ``node.get_content(metadata_mode=MetadataMode.EMBED)`` ——
    不排除 metadata 时索引文本是「``case_id: RAG_CASE_0001`` / ``category: …`` / ``decision: REJECT``
    / ``risk_level: HIGH`` / ``risk_type: ['POTENTIAL_IP_RISK']`` + 正文」，于是**按 metadata 字面值
    就能命中**（修复前实测 ``RAG_CASE_0037`` 命中 67/67 篇、且该篇被排到首位）—— 这是伪检索：
    用「RAG_CASE_0037」这类字样去搜，本不该有任何先例因为「恰好被引用到 id」而浮上来。

    ⚠️ **实测口径纠正（勿照抄「命中 0 篇」）**：修复后 ``search()`` 仍返回**全部候选**，但
    （a）**BM25 原始分全部为 0**、（b）分数完全平坦（全 1.0）、（c）结果与「零信号查询」**逐字节一致**。
    「1.0」不是「命中」而是两段既有约定的合力：① ``BM25Retriever`` 对零分查询仍按其
    ``similarity_top_k`` 返回节点；② ``_rank_bm25`` 用仓库既有 ``normalize_minmax``，而
    「全等值集 → 全 1.0」是它防除零的既定确定性约定。故本用例断言**原始分 == 0** 这一直接证据，
    再用「平坦 + 与对照查询逐字节一致」把可观测语义钉住 —— metadata 一旦重回索引文本，
    该字面值所属先例会被区分出来，三处断言同时破。
    """
    assert not (set(_jieba_tokens_of(literal)) & _body_tokens(CASE_ROWS, kind="case")), (
        f"{literal!r} 的 token 出现在正文里 → 它不是「纯 metadata 字面值」反例（请换一个）"
    )
    idx = _case_chroma("bm25")
    all_rows = list(range(len(CASE_ROWS)))
    raw = _bm25_raw_scores(idx, all_rows, literal)
    assert raw and max(raw) == 0.0, (
        f"纯 metadata 字面值 {literal!r} 在 BM25 索引里产生了非零原始分（max={max(raw)}）——"
        "说明 metadata 又进了检索文本（R7 回归）"
    )

    literal_hits = await idx.search(literal, CaseSearchFilters(), top_k=len(CASE_ROWS))
    assert len(literal_hits) == len(CASE_ROWS), "零分查询仍会返回候选（见 docstring 的实测口径）"
    assert {h.retrieval_score for h in literal_hits} == {1.0}, "零信号查询的归一化分必须平坦"
    control = await idx.search(_NO_SIGNAL_QUERY, CaseSearchFilters(), top_k=len(CASE_ROWS))
    assert [h.model_dump(mode="json") for h in literal_hits] == [
        h.model_dump(mode="json") for h in control
    ], f"字面值查询 {literal!r} 的结果必须与零信号查询逐字节一致（字面值不得有信号）"

    # 退化区间的具体形态（主 agent 实测口径：该查询返回前 5 行且分全 1.0）
    top5 = await idx.search(literal, CaseSearchFilters(), top_k=5)
    assert [(h.case_id, h.retrieval_score) for h in top5] == [
        (row.case_id, 1.0) for row in CASE_ROWS[:5]
    ], "零信号查询应退化为「corpus 原序 + 全 1.0」（不是按字面值相关度排序）"

    # 跨后端对照（**仅在全零分退化区间**成立：无任何区分度 → 两边都退化为 corpus 原序 + 全 1.0）。
    # 这不是对 §5-1「bm25 与 local 不可比」的反例 —— 那条针对的是有信号的检索结果。
    local = _case_local("bm25")
    local_literal = await local.search(literal, CaseSearchFilters(), top_k=len(CASE_ROWS))
    assert [h.model_dump(mode="json") for h in local_literal] == [
        h.model_dump(mode="json") for h in literal_hits
    ], "零信号退化区间下 chroma 与 local 都应「无区分度」"

    # 对照：正文查询必须是**有区分度**的（否则上面的「平坦」可能只是整条链路都失灵）
    body_raw = _bm25_raw_scores(idx, all_rows, "无品牌高相似商家多次上架")
    assert max(body_raw) > 0.0, "正文查询应当有非零 BM25 原始分 —— 否则本用例前提不成立"
    body = await idx.search("无品牌高相似商家多次上架", CaseSearchFilters(), top_k=10)
    assert body and len({h.retrieval_score for h in body}) > 1, "正文查询应当有区分度（非平坦）"


@pytest.mark.parametrize("literal", _POLICY_METADATA_LITERALS)
async def test_chroma_policy_bm25_ignores_metadata_literals(literal: str) -> None:
    """**R7 守卫（policy）**：同上一路（``clause_id`` / ``policy_id`` / ``risk_type`` / ``status``）。

    修复前实测：``POLICY_1.1_v2_c1`` 这类字面值命中 24/24 篇。修复后：BM25 原始分全 0、
    ``search()`` 结果平坦且与零信号查询逐字节一致（口径说明见 case 侧用例 docstring）。
    ⚠️ ``PolicyClauseHit`` **没有** ``retrieval_score`` 字段（C1 契约只改 ``CaseHit``），
    故 policy 侧的「有没有信号」只能靠原始分 + 命中序与对照查询一致来断言。
    """
    assert not (set(_jieba_tokens_of(literal)) & _body_tokens(POLICY_ROWS, kind="policy")), (
        f"{literal!r} 的 token 出现在正文里 → 它不是「纯 metadata 字面值」反例（请换一个）"
    )
    idx = _policy_chroma("bm25")
    all_rows = list(range(len(POLICY_ROWS)))
    raw = _bm25_raw_scores(idx, all_rows, literal)
    assert raw and max(raw) == 0.0, (
        f"纯 metadata 字面值 {literal!r} 在 BM25 索引里产生了非零原始分（max={max(raw)}）——"
        "说明 metadata 又进了检索文本（R7 回归）"
    )
    literal_hits = await idx.search(literal, PolicySearchFilters(), top_k=len(POLICY_ROWS),
                                    effective_only=False)
    assert len(literal_hits) == len(POLICY_ROWS)
    control = await idx.search(_NO_SIGNAL_QUERY, PolicySearchFilters(),
                               top_k=len(POLICY_ROWS), effective_only=False)
    assert [h.model_dump(mode="json") for h in literal_hits] == [
        h.model_dump(mode="json") for h in control
    ], f"字面值查询 {literal!r} 的结果必须与零信号查询逐字节一致（字面值不得有信号）"
    # 跨后端对照（**仅在全零分退化区间**成立，理由见 case 侧用例）
    local = _policy_local("bm25")
    local_literal = await local.search(literal, PolicySearchFilters(), top_k=len(POLICY_ROWS),
                                       effective_only=False)
    assert [h.model_dump(mode="json") for h in local_literal] == [
        h.model_dump(mode="json") for h in literal_hits
    ], "零信号退化区间下 chroma 与 local 都应「无区分度」"

    body_raw = _bm25_raw_scores(idx, all_rows, "仿冒 高仿 复刻")
    assert max(body_raw) > 0.0, "正文查询应当有非零 BM25 原始分 —— 否则本用例前提不成立"


def test_chroma_build_nodes_embed_text_is_body_only() -> None:
    """**(a) 机制级 R7 守卫（定向、便宜）**：``_build_nodes`` 产出的 node，其 EMBED 文本 == 正文。

    直接按签名调 ``_build_nodes(rows, *, kind, collection, llama)``（``llama`` 用实现自己的
    ``_import_llama()`` 装配面，不做 monkeypatch），逐条断言三种口径：
    ① ``node.get_content(metadata_mode=MetadataMode.EMBED)`` **逐字等于正文**
       （policy ``f"{title}。{text}"`` / case ``summary``）—— 这就是 ``BM25Retriever``
       索引时取的那一份文本（安装源码 ``bm25s.tokenize([node.get_content(metadata_mode=EMBED) ...])``）；
    ② 该 node 的 metadata 字面值（标量 + 列表逐元素，如 ``case_id`` / ``POTENTIAL_IP_RISK`` /
       ``REJECT`` / ``HIGH`` / ``女鞋/运动鞋``）**都不出现在 EMBED 文本里**；
    ③ **JSON 往返后仍成立**：``node_to_metadata_dict`` → ``metadata_dict_to_node`` 是
       ``BM25Retriever`` 重建节点的真实路径（``metadata_dict_to_node(node_dict)``），
       若 ``excluded_*_metadata_keys`` 不随 ``_node_content`` 往返存活，R7 就是假的 —— 这里直接测。

    ⚠️ **本断言的安全性有实测前提**：本 corpus 实测「正文含自身 metadata 字面值」的 node 数为
    **case 0/67、policy 0/24**（否则 ② 会是误报 —— 例如把类目名写进 summary）。语料若改动
    导致某 node 的正文天然含某字面值，请在该 node 上跳过那个字面值并在注释里说明，不要放宽断言。

    ⚠️ **诚实标注本用例的边界**：它只证明「node 暴露给 EMBED 的文本是正文」。它**不**证明
    检索器真的走 EMBED（那是库内部行为）——端到端性质由黑盒用例
    ``test_chroma_*_bm25_ignores_metadata_literals``（字面值查询 ≡ 无信号查询）兜底；
    两者互为补充：本用例定位「哪一层坏了」，黑盒用例证明「用户看到的行为对不对」。
    """
    from llama_index.core.schema import MetadataMode
    from llama_index.core.vector_stores.utils import (
        metadata_dict_to_node,
        node_to_metadata_dict,
    )

    from pra.rag.chroma_backend import _build_nodes, _import_llama

    llama = _import_llama()

    def _literals(metadata: dict) -> list[str]:
        """metadata 的全部字面值（标量 + 列表逐元素），统一转 str 供子串断言。"""
        out: list[str] = []
        for value in metadata.values():
            if isinstance(value, list):
                out.extend(str(item) for item in value)
            else:
                out.append(str(value))
        return [item for item in out if item]

    for kind, rows, body_of in (
        ("case", CASE_ROWS, lambda r: r.summary),
        ("policy", POLICY_ROWS, lambda r: f"{r.title}。{r.text}"),
    ):
        nodes, node_ids = _build_nodes(
            rows, kind=kind, collection=f"unit_{kind}_256", llama=llama
        )
        assert len(nodes) == len(node_ids) == len(rows), "1 行 = 1 Node（不切碎）"
        for node, row in zip(nodes, rows):
            body = body_of(row)
            embed_text = node.get_content(metadata_mode=MetadataMode.EMBED)
            assert embed_text == body, (
                f"{kind} node {node.node_id}: EMBED 文本 != 正文 —— metadata 进了检索文本（R7 回归）\n"
                f"EMBED={embed_text[:120]!r}\n正文={body[:120]!r}"
            )
            for literal in _literals(node.metadata):
                assert literal not in embed_text, (
                    f"{kind} node {node.node_id}: metadata 字面值 {literal!r} 出现在 EMBED 文本里"
                )
            # ③ 往返（BM25Retriever 的重建路径）后 exclusion 仍生效
            rebuilt = metadata_dict_to_node(node_to_metadata_dict(node))
            assert rebuilt.get_content(metadata_mode=MetadataMode.EMBED) == body, (
                f"{kind} node {node.node_id}: JSON 往返后 EMBED 文本不再是正文 —— "
                "excluded_embed_metadata_keys 未随 _node_content 存活（BM25Retriever 正是这样重建节点的）"
            )
            # metadata 本体仍完整（R-4 隔离 / 过滤 / 审计都靠它）；只做「非空 + 含行键」的最小断言
            assert node.metadata
            assert str(node.metadata.get("case_id") or node.metadata.get("clause_id")) == (
                row.case_id if kind == "case" else row.clause_id
            )


def test_chroma_index_nodes_carry_metadata_exclusions() -> None:
    """**装配层守卫**：真实索引里的 node 确实带上了 ``excluded_*_metadata_keys``（不是只有单测路径）。

    ``_build_nodes`` 单测（上一条）覆盖「文本口径」；本条覆盖「**索引构造路径确实用了这套 node**」——
    即 ``Chroma*Index.nodes`` 的每个 node 都有非空 exclusion 且覆盖其全部 metadata 键，
    否则前面的机制单测可能测的是「另一条没人用的构造分支」。同时断言 metadata 本体未被裁剪。
    """
    case_idx = _case_chroma("bm25")
    policy_idx = _policy_chroma("bm25")
    for idx, rows in ((case_idx, CASE_ROWS), (policy_idx, POLICY_ROWS)):
        assert len(idx.nodes) == len(rows)
        for node in idx.nodes:
            assert set(node.excluded_embed_metadata_keys) == set(node.metadata), (
                "node 的 EMBED exclusion 必须覆盖全部 metadata 键（否则 metadata 会回到检索文本）"
            )
            assert set(node.excluded_llm_metadata_keys) == set(node.metadata)
            assert node.metadata.get("case_id") or node.metadata.get("clause_id")
            assert node.excluded_embed_metadata_keys, "exclusion 为空 = R7 未生效"


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


# ---------------------------------------------------------------------------
# 9) R6：BM25 分词器受控替换的**并发正确性**（docs/10 §6-R6）
# ---------------------------------------------------------------------------


#: 并发规模：8 线程同时冲进临界区（barrier 对齐起点），每线程固定跑自己的 query。
#: 实测（修复前）：8×15 轮下**每次运行**都能复现（120 轮里 1 次抛错 + 全局符号泄漏），
#: 故本用例对「补丁顺序」有牙齿；修复后 3/3 运行 0 异常、无泄漏。
_R6_THREADS = 8
_R6_ROUNDS = 12


class _PatchOrderProbe:
    """``_TOKENIZER_LOCK`` 的观测代理：只记录「取锁那一刻补丁是否已安装」。

    独占语义与 ``threading.RLock`` 完全一致（内部就是 RLock），只在 ``acquire`` 前记一笔
    —— 用来把**顺序不变量**「补丁必须在锁内安装」写成可判别断言（机制级定位工具）。
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        #: 每次取锁时 ``bm25s.tokenize`` 是否已被替换为 jieba 替身。
        self.patched_at_acquire: list[bool] = []

    def acquire(self, *args: Any, **kwargs: Any) -> bool:
        self.patched_at_acquire.append(bm25s.tokenize is not _REAL_BM25S_TOKENIZE)
        return self._lock.acquire(*args, **kwargs)

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def _bm25_hits(index: Any, query: str, top_k: int) -> list[tuple[str, float]]:
    """``mode="bm25"`` 的检索结果 → ``(case_id, retrieval_score)``（用于逐项对比）。

    用 **case KB**：``CaseHit`` 契约带 ``retrieval_score``，而 ``PolicyClauseHit`` 按契约**不含分**
    （docs/10 §5-4）→ 分数是判别「分词器被污染」的更强信号（id 序可能不变而分数漂移）。
    """
    hits = asyncio.run(index.search(query, CaseSearchFilters(), top_k))
    return [(h.case_id, h.retrieval_score) for h in hits]


def test_r6_bm25_tokenizer_patch_installed_under_lock() -> None:
    """机制级：``bm25s.tokenize`` 的替换**必须在持锁窗口内**完成（R6 顺序不变量）。

    断言「取锁那一刻补丁尚未安装」——若有人把调用点写回 ``with _jieba_tokenizer(), _TOKENIZER_LOCK:``
    （补丁先于取锁），本用例立刻变红（实测：回退顺序 → ``patched_at_acquire`` 出现 ``True``）。
    这是**定位工具**，不单独证明并发安全；真正的用户可见性质由
    :func:`test_r6_concurrent_bm25_searches_do_not_pollute_tokenizer` 守住。
    """
    index = _case_chroma("bm25")
    probe = _PatchOrderProbe()
    original_lock = chroma_backend._TOKENIZER_LOCK
    chroma_backend._TOKENIZER_LOCK = probe
    try:
        hits = _bm25_hits(index, "外观高度模仿知名品牌无授权", 5)
    finally:
        chroma_backend._TOKENIZER_LOCK = original_lock

    assert hits, "空结果会让本用例失去意义（先确认检索真的跑了）"
    # 两个调用点（建索引 / 检索）各取一次锁 → 两次都必须在「补丁未安装」时取到锁
    assert probe.patched_at_acquire == [False, False], (
        f"补丁先于取锁安装 → 临界区未覆盖补丁（R6）：{probe.patched_at_acquire}"
    )
    assert bm25s.tokenize is _REAL_BM25S_TOKENIZE, "检索结束后 bm25s.tokenize 必须已复原"


def test_r6_concurrent_bm25_searches_do_not_pollute_tokenizer() -> None:
    """主闸（黑盒）：8 线程并发跑 ``mode="bm25"`` 检索，**结果必须与单线程逐项一致**。

    断言三件事（都是用户可见性质，不绑内部实现）：
    ① 无异常 —— 补丁窗口错位时实测抛
       ``ValueError: The maximum token ID in the query (…) is higher than the number of tokens in the index.``
       （线程在自以为的 jieba 上下文里用了真分词器）；
    ② 每线程每轮结果 == 该 query 的**单线程 golden**（id 序 + 6 位检索分）—— tokenizer 状态
       被别的线程污染就会立刻表现为分数/名次漂移或异常；
    ③ 全部线程结束后 ``bm25s.tokenize`` **复原为原符号** —— 错位时实测被 jieba 替身
       **进程级永久替换**（泄漏）。

    ⚠️ 如实标注（勿夸大）：本用例验证的是「**本实现 + 本模块锁**在并发下不互相污染」，
    **不等于**该实现「天然线程安全」—— 全局符号替换仍是有代价的做法，边界见 docs/10 §6-R6。
    """
    index = _case_chroma("bm25")
    queries = [
        "无品牌高相似商家多次上架",
        "外观高度模仿 相似度",
        "规避 多次改标题 重上架",
        "夸大宣传 功效 虚假",
    ]
    golden = {q: _bm25_hits(index, q, 5) for q in queries}
    assert all(golden.values()), "golden 不得为空（否则本用例无法判别）"

    errors: list[str] = []
    mismatches: list[str] = []
    barrier = threading.Barrier(_R6_THREADS)

    def worker(tid: int) -> None:
        query = queries[tid % len(queries)]
        barrier.wait(timeout=30)  # 起点对齐 → 最大化临界区重叠
        for rnd in range(_R6_ROUNDS):
            try:
                got = _bm25_hits(index, query, 5)
            except Exception as exc:  # noqa: BLE001 —— 任何异常都算失败（要看到具体类型/文本）
                errors.append(f"t{tid}r{rnd} {query!r}: {type(exc).__name__}: {exc}")
                continue
            if got != golden[query]:
                mismatches.append(f"t{tid}r{rnd} {query!r}: {got} != {golden[query]}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(_R6_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)

    assert not errors, f"并发检索抛异常（{len(errors)} 次），前 3 条：{errors[:3]}"
    assert not mismatches, f"并发结果与单线程 golden 不一致（{len(mismatches)} 次）：{mismatches[:3]}"
    assert bm25s.tokenize is _REAL_BM25S_TOKENIZE, (
        "并发结束后 bm25s.tokenize 未复原 → 全局符号泄漏（R6）"
    )
    assert served_counters()["llm_calls"] == 0
