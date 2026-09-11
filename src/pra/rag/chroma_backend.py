"""ChromaDB + LlamaIndex 检索后端（rag/chroma_backend.py）—— ChromaPolicyIndex / ChromaCaseIndex。

docs/10-rag-upgrade-spec.md（唯一实施契约）§1/§3 的落地实现：

```
Query → Hybrid Retrieval（Policy KB / Case KB 各自独立）
          ├── VectorRetriever : ChromaDB(cosine) + 自带 embedding（Mock / BGE）
          └── BM25Retriever   : llama-index-retrievers-bm25（引擎 bm25s）+ jieba 分词
        → RRF（Reciprocal Rank Fusion）
        → Top-K → PolicySearchTool / CaseSearchTool（现有契约一行不改）
```

业务边界仍由 tools 层 ``PolicyIndex`` / ``CaseIndex`` Protocol 守护（docs/10 §2 铁律）：
本模块只是「向量库 + 检索器 + 融合」的实现替换，Agent / Gate / Evidence / Evaluation 零改动。

Node / Chunk 口径（§3「不切碎」）
--------------------------------
**1 政策条款 = 1 Node**、**1 先例 = 1 Node**，不做任何切分；node id 由「collection 名 +
corpus 行键」稳定哈希得出（同 corpus 重建幂等 upsert 覆盖）。metadata 按 §3：

- policy → ``clause_id / policy_id / version / category / risk_type / status / effective_date``
- case   → ``case_id / category / decision / risk_level / risk_type``

⚠️ **实测约束（Chroma 1.5.9）**：metadata 里**空列表值不被接受** ——
``ValueError: Expected metadata list value for key 'risk_type' to be non-empty in add.``。
而 corpus 真实存在 ``risk_type == []`` 的行（67 case 中 16 条、24 policy 中 1 条）。
故 ``risk_type`` **仅在非空时写入**（语义等价：缺键 == 空列表，检索侧一律用
``wanted & set(...)`` 判定），本模块以 ``_RISK_TYPE_KEY`` 常量集中标注该处理。

三种检索模式（与仓库其它实现同名：``"bm25" | "vector" | "hybrid"``）
------------------------------------------------------------------
- ``vector``：Chroma cosine 距离 → **相似度 = 1 − distance**（docs/10 §3 实测口径：用
  ``[1,0,0]`` vs ``[0.9,0.1,0]`` 验证 distance 0.006116271 ↔ 1−cos 0.9938837）。
  ⚠️ **实测（本轮）**：``llama-index-vector-stores-chroma 0.6.0`` 的 ``ChromaVectorStore``
  把分数算成 ``exp(-distance)``（安装源码 ``_query``：``similarity_score = math.exp(-distance)``；
  同一对向量 store 返 0.993902 ≠ 1−distance 0.9938837）—— 那是**另一套映射、不是余弦**。
  故本模块**自建 ``BaseRetriever`` 子类**（``_ChromaCosineRetriever``）：装配仍走 LlamaIndex
  （``TextNode`` / ``MetadataFilters``），但取数直接调 **Chroma 原生 ``collection.query``**
  拿 ``distance`` 自己算 ``1 − distance``。这一层「只多一层薄适配、不改余弦几何」是
  docs/10 §3 钉死的口径所要求的（§5 验收 1「同构等价」亦以此为前提）。
- ``bm25``：``BM25Retriever``（引擎 ``bm25s``，``jieba`` 分词，见下「jieba 接入」）；
  **在本模块自建的 RRF 里作为一路排名**（见下 hybrid）。
- ``hybrid``：**RRF（Reciprocal Rank Fusion）** —— 两路排名列表 → ``Σ_r 1/(60 + rank_r)``。
  ⚠️ **本模块自己算 RRF**（:func:`_fuse_rrf`），**不调用** ``QueryFusionRetriever`` 现成的融合：
  实测该实现在融合时**原地改写共享 node 的 ``score``**（源码
  ``reranked_nodes[-1].score = score``），而 node 对象跨检索共享 → **同一 query 连查两次结果
  不同**（实测 top-1 从 ``RAG_CASE_0064`` 0.020393 变成 ``RAG_CASE_0020`` 0.022821）——
  直接违反 §3「确定性不可退化」。融合定义/常数与它一致（同 k=60），并已提供
  ``_make_fusion_retriever``（**必传 ``num_queries=1`` + ``MockLLM()`` 守卫**）供外部/测试
  引用的「配置正确」构造：⚠️ ``num_queries`` 默认 4 会**调用 LLM 生成 query 变体**（源码
  ``if self.num_queries > 1: queries.extend(self._get_queries(...))``），本项目检索侧
  **禁止任何 LLM 调用**；且不传 ``llm`` 时该库回落 ``Settings.llm`` → 本环境直接
  ``ImportError: llama-index-llms-openai package not found``（项目未装该集成）。

``retrieval_score`` 口径（docs/10 §0 C1 / §6-R5）
------------------------------------------------
写进 ``CaseHit.retrieval_score`` 的是**检索分，不是语义相似度**：
``bm25`` 模式 = 候选集内 min-max 归一化的 BM25 分；``vector`` 模式 = ``1 − distance``；
**``hybrid`` 模式 = RRF 融合分**（``Σ 1/(k+rank)``，``k=60``，落在 ~(0, **2/60 = 1/30**]）。
任何文档/注释/报告都**不得**把它表述成「语义相似度」。

Metadata 过滤：**两路过滤位置不统一（实测钉死，如实标注）**
--------------------------------------------------------
- 向量路：经 LlamaIndex ``MetadataFilters`` 下推到 Chroma ``where``（store 侧过滤）。
- BM25 路：``BM25Retriever`` 无等价下推 —— 改为**按候选集重建 retriever**（构造期只喂
  候选 node），即 Python 侧先过滤候选。

→ 这正是 docs/10 §3 / §6-R2 记录的**已知口径分裂**（原 docs/06 P2-4 刻意只在 Python 侧
单实现以避免双实现漂移）；本模块不掩盖它。为把「同语义」这条底线守住，向量路在
store 下推**之后**仍会用**同一份** ``_match_*`` Python 谓词复核一遍（下推只为收窄候选，
不参与判定）—— 两路最终判定是同一段代码，不存在双实现漂移。

两处**实测**得到的 Chroma 过滤能力边界（决定了下推能推多少）：
1. **列表字段的成员判定**：``{"risk_type": {"$in": [X]}}`` / ``$eq`` **恒不命中**
   （Chroma 的 ``$in`` 面向标量）；把列表嵌进 ``$in`` 会报
   ``ValueError: Expected operand value to be a str, int, float, or bool``。实测可用的只有
   ``{"risk_type": {"$contains": X}}``（X 为标量）。
2. ``llama_index.vector_stores.chroma`` 的 ``_transform_chroma_filter_operator`` 只认
   ``!= == > < >= <= in nin`` —— ``FilterOperator.ANY/ALL/CONTAINS`` 直接抛
   ``ValueError: Filter operator any not supported``（``contains`` 同理）。

故向量路的下推范围为 **category（``FilterOperator.IN``，含「全类目」）**；
``risk_type`` 交叠判定在 Python 侧与 BM25 路共用同一谓词（能力边界来自上面两条实测，
非实现偷懒）。文档务必同步此段。

jieba 接入（C3 的意图：成熟库 + 中文分词）
------------------------------------------
``llama-index-retrievers-bm25 0.8.0`` 实测**没有任何 tokenizer 注入点**：
``from_defaults`` 里 ``tokenizer`` 参数已标记 deprecated（传了只 warning 且**不使用**），
构造与检索两处都硬编码 ``bm25s.tokenize(..., token_pattern=self.token_pattern, stemmer=...)``
（见安装源码），且该函数**只接受 ``str``**。因此本模块在**构造前后**把
``bm25s.tokenize`` 替换为 jieba 实现（``_install_jieba_tokenizer`` / 可恢复的
``_jieba_tokenizer`` 上下文管理器），使索引与查询**都**走 jieba 切词：

- jieba 精确模式切词 → 词表按首次出现顺序稳定编号（确定性）；
- 不做停用词/词干化（``skip_stemming=True`` / ``language=""`` 显式关掉英文 Stemmer ——
  中文文本用英文词干器无意义且引入额外依赖行为）；
- 缺失查询词由同一切词器编号进同一词表，不会给 bm25s 发越界 id
  （否则会 ``ValueError: The maximum token ID in the query ... is higher than the number
  of tokens in the index``）。

这是对第三方库全局符号的**受控替换**（本模块独占，构造与检索均在同一锁内进行），
在模块 docstring 如实记录 —— 不假装 bm25s 原生支持中文分词。

确定性（不可退化）
------------------
检索链路**零 LLM、零随机**；排序 tie-break 恒为 ``(score 降序, corpus 原序 idx 升序)``，
**不依赖** Chroma / bm25s 的返回顺序（两者顺序都不可信）；分数统一 ``round(..., 6)``。
模块级 ``SERVED_COUNTERS`` 记录本进程内经本模块发出的 llm 调用数（恒为 0，供验收断言）。

客户端与 collection
-------------------
- 默认客户端 ``chromadb.HttpClient(host="127.0.0.1", port=8001)``（本机服务端已部署；
  从不动连接，构造不联网 —— 见返回报告 §4）；可外部注入任意 client（测试）或使用
  ``EphemeralClient()``（内存，离线恒跑）。
- collection 名沿用仓库既有形状 ``<prefix or "pra">_<policy|case>_<dim>``（与
  ``rag/qdrant_index.py`` 对齐，便于测试复用）；**建库必须同时满足两条**（§3 实测，缺一
  不可，两条都是「静默出错」型陷阱）：

  1. ``embedding_function=None`` —— 否则 Chroma 静默启用**默认 ONNX 嵌入函数**并去下模型
     （我们自带向量）；
  2. ``configuration={"hnsw": {"space": "cosine"}}`` —— **Chroma 默认空间是 ``l2``**，
     只关嵌入函数**不会**变成 cosine。实测同一对单位向量 ``[1,0,0]`` vs ``[0.9,0.1,0]``：
     默认 l2 → distance ``0.020000005``（``1 − distance`` = 0.979999995 而 numpy 余弦
     0.993883735，**静默错**）；显式 cosine → distance ``0.006116271`` → ``1 − distance``
     = 0.993883729 ✅。

  空间由 :func:`_verify_collection_space` **断言**（新建 + 复用两条路径都查，读
  ``collection.configuration["hnsw"]["space"]``/旧式 ``metadata["hnsw:space"]``），
  并在 upsert 后做一次自身距离 ≈ 0 的 cosine 自检；不符即抛 ValueError（含清理指引），
  **绝不带着未知/错误空间继续检索**。⚠️ 实测：对**已存在**的 l2 库再传 cosine 配置
  ``get_or_create_collection`` **不会**改建库空间 —— 复用路径不校验就会拿到语义错库。
  已存在时另校验 ``pra_dim`` 一致（同名不同维即报错，不静默复用）。
- 空候选集 → 返回 ``[]``（合法空结果，工具 ok=True），与其它实现一致。
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from typing import Any

# 顶层只 import 仓库内模块 + 标准库；chroma / llama_index / jieba 一律延迟 import
# （默认 local 路径零额外依赖，构造/检索时才拉起 —— 对齐 rag/qdrant_index.py 的约定）。
from pra.rag.corpus.schema import CasePrecedentRecord, PolicyClauseRecord
from pra.rag.embedder import Embedder, MockHashEmbedder
from pra.rag.llama_embedding import LlamaIndexEmbeddingAdapter
from pra.rag.retrieval import (
    DEFAULT_WEIGHTS,
    MODES,
    RetrievalMode,
    normalize_minmax,
)
from pra.rag.vectors import cosine_similarity
from pra.tools.case_search.tool import CaseHit, CaseSearchFilters
from pra.tools.policy_search.tool import PolicyClauseHit, PolicySearchFilters

__all__ = [
    "CHROMA_DEFAULT_HOST",
    "CHROMA_DEFAULT_PORT",
    "COLLECTION_NAME_TEMPLATE",
    "SERVED_COUNTERS",
    "ChromaCaseIndex",
    "ChromaPolicyIndex",
    "check_cosine_self_check",
    "check_cosine_space",
    "delete_collection",
    "make_chroma_client",
    "reset_served_counters",
    "served_counters",
]

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 默认 Chroma 服务端地址（deploy/chroma：宿主 8001 → 容器 8000；`/api/v1` 已废弃，只用 /api/v2）。
CHROMA_DEFAULT_HOST = "127.0.0.1"
CHROMA_DEFAULT_PORT = 8001

#: collection 名形状（与 rag/qdrant_index.py 的 `_collection_name` 同形）。
COLLECTION_NAME_TEMPLATE = "<prefix or 'pra'>_<policy|case>_<dim>"

_DEFAULT_COLLECTION_PREFIX = "pra"
_FULL_CATEGORY = "全类目"
#: metadata 维度键（Chroma 不声明向量维度，故把本索引声明的 dim 存进 collection metadata 复用校验）。
_DIM_KEY = "pra_dim"
#: 风险类型列表键（空列表不写 —— Chroma 拒绝空列表 metadata 值，见模块 docstring）。
_RISK_TYPE_KEY = "risk_type"

#: RRF 常数 k=60（与 ``QueryFusionRetriever._reciprocal_rerank_fusion`` 同源：Cormack 2009）。
_RRF_K = 60.0

#: 本进程内经本模块发出的计数（验收断言用）：llm 必须恒为 0（检索侧零 LLM）。
SERVED_COUNTERS: dict[str, int] = {
    "llm_calls": 0,
    "vector_searches": 0,
    "bm25_searches": 0,
    "hybrid_searches": 0,
    "served_hits": 0,
    #: 向量路在 Chroma 少返候选时，由本模块按已存向量补算 cos 的次数（正常应恒为 0）。
    "vector_bruteforce_fallbacks": 0,
}

#: 向量路覆盖率自检的重试次数（第三方少返是已知风险；重试后仍不覆盖即抛）。
_VECTOR_COVERAGE_ATTEMPTS = 3

_LATIN_RUN = re.compile(r"[0-9A-Za-z_]+")


# ---------------------------------------------------------------------------
# 延迟 import / 客户端 / collection 装配
# ---------------------------------------------------------------------------


def _import_chroma() -> tuple[Any, Any]:
    """延迟 import ``chromadb``（仅构造路径调用；模块顶层不 import）。

    缺包时报错并提示安装 extra（与 ``rag/qdrant_index._import_qdrant`` 同风格）。
    返回 ``(chromadb, ChromaNotFoundError)``。
    """
    try:
        import chromadb
        from chromadb.errors import NotFoundError as ChromaNotFoundError
    except ImportError as exc:  # pragma: no cover — 触发路径仅在显式开启 chroma 后端
        raise RuntimeError(
            "chroma 后端需要 chromadb 与 llama-index 集成包：请运行 `uv sync --extra rag` 安装；"
            '默认本地检索用 backend="local" 即可（无需该依赖）'
        ) from exc
    return chromadb, ChromaNotFoundError


def _import_llama() -> dict[str, Any]:
    """延迟 import LlamaIndex 装配面（仅构造路径调用；模块顶层不 import）。

    只 import ``llama-index-core`` + ``llama-index-retrievers-bm25`` +
    ``llama-index-vector-stores-chroma`` 三个具体集成（**不装/不引伞包** ``llama-index``，
    docs/10 §3：「伞包会拖进 llms-openai / embeddings-openai 等不用的集成」）。
    """
    try:
        from llama_index.core.base.base_retriever import BaseRetriever
        from llama_index.core.llms import MockLLM
        from llama_index.core.retrievers import QueryFusionRetriever
        from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
        from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
        from llama_index.core.vector_stores.types import FilterOperator
        from llama_index.retrievers.bm25 import BM25Retriever
        from llama_index.vector_stores.chroma import ChromaVectorStore
    except ImportError as exc:  # pragma: no cover — 触发路径仅在显式开启 chroma 后端
        raise RuntimeError(
            "chroma 后端需要 llama-index-core / llama-index-vector-stores-chroma / "
            "llama-index-retrievers-bm25：请运行 `uv sync --extra rag` 安装；"
            '默认本地检索用 backend="local" 即可（无需该依赖）'
        ) from exc
    return {
        "BaseRetriever": BaseRetriever,
        "MockLLM": MockLLM,
        "QueryFusionRetriever": QueryFusionRetriever,
        "NodeWithScore": NodeWithScore,
        "QueryBundle": QueryBundle,
        "TextNode": TextNode,
        "MetadataFilter": MetadataFilter,
        "MetadataFilters": MetadataFilters,
        "FilterOperator": FilterOperator,
        "BM25Retriever": BM25Retriever,
        "ChromaVectorStore": ChromaVectorStore,
    }


def make_chroma_client(
    chroma_client: Any | None = None,
    *,
    host: str = CHROMA_DEFAULT_HOST,
    port: int = CHROMA_DEFAULT_PORT,
    ephemeral: bool = False,
) -> Any:
    """取用/自建 Chroma 客户端（外部注入优先）。

    - ``chroma_client`` 非空 → 原样返回（测试注入 ``EphemeralClient`` 或自有 client）；
    - ``ephemeral=True`` → ``chromadb.EphemeralClient()``（进程内内存库，**离线可用**）；
    - 否则 → ``chromadb.HttpClient(host=host, port=port)``（默认本机 ``127.0.0.1:8001``
      服务端）。

    **构造不联网**：``HttpClient`` 只做参数装配，首个请求才建立连接（故无服务端时构造
    安全、检索时才失败）。匿名遥测显式关闭（用本机服务端不应产生外发流量）。
    """
    if chroma_client is not None:
        return chroma_client
    chromadb, _not_found = _import_chroma()
    if ephemeral:
        return chromadb.EphemeralClient()
    try:
        from chromadb.config import Settings as ChromaSettings

        return chromadb.HttpClient(
            host=host, port=port, settings=ChromaSettings(anonymized_telemetry=False)
        )
    except Exception:  # noqa: BLE001  # pragma: no cover —— 兼容无 Settings 参数的 chromadb 变体
        return chromadb.HttpClient(host=host, port=port)


def _collection_name(prefix: str | None, kind: str, dim: int) -> str:
    """collection 名：``<prefix or "pra">_<policy|case>_<dim>``（与 qdrant 后端同形）。"""
    return f"{prefix or _DEFAULT_COLLECTION_PREFIX}_{kind}_{dim}"


def _node_id(collection: str, key: str) -> str:
    """node id = ``sha256(collection + '\x1f' + row_key)`` 前 16 字节 hex（稳定、幂等覆盖）。

    以 collection 名参与哈希，避免 policy / case KB 在同一 collection 前缀下 node id 撞车；
    同 corpus 重复构造 → 同 id → upsert 幂等覆盖（不产生重复点）。
    """
    digest = hashlib.sha256(f"{collection}\x1f{key}".encode()).hexdigest()[:32]
    return f"pra-{digest}"


def _resolve_dim(embedder: Embedder, doc_vectors: list[list[float]]) -> int:
    """解析检索向量维度（embedder 声明优先，否则取首条 doc 向量的长度）。"""
    dim = getattr(embedder, "dim", None)
    if isinstance(dim, int) and dim > 0:
        if doc_vectors and len(doc_vectors[0]) != dim:
            raise ValueError(
                f"embedder 声明维度 {dim} 与 doc 向量长度 {len(doc_vectors[0])} 不一致"
            )
        return dim
    if not doc_vectors:
        raise ValueError("无法确定向量维度：corpus 为空且 embedder 未声明 dim")
    return len(doc_vectors[0])


#: collection 的期望向量空间（**必须显式 cosine**）：Chroma 默认是 ``l2``，
#: 而本模块「相似度 = 1 − distance」的整套口径只在 cosine 空间成立 —— 见
#: :func:`_verify_collection_space` 的实测记录。
_REQUIRED_SPACE = "cosine"

#: cosine 自检容差（实测 distance==0 时精确为 0；浮点尾差留 1e-6）。
_SELF_CHECK_TOL = 1e-6


def _collection_space(collection: Any) -> str | None:
    """读 collection 的向量空间（优先新式 ``configuration["hnsw"]["space"]``，回落旧式 metadata）。

    - 新式（chromadb ≥ 0.6/1.x）：``collection.configuration["hnsw"]["space"]``，
      实测（1.5.9）建库时传 ``configuration={"hnsw":{"space":"cosine"}}`` 或
      ``metadata={"hnsw:space":"cosine"}`` 都会在 config 里体现为 ``"cosine"``；
    - 旧式：``collection.metadata["hnsw:space"]``（老版本 API）。
    都读不到 → ``None``（调用方按「无法确认」处理，不假定 cosine）。
    """
    cfg = getattr(collection, "configuration", None) or {}
    hnsw = cfg.get("hnsw") if isinstance(cfg, dict) else None
    if isinstance(hnsw, dict) and hnsw.get("space"):
        return str(hnsw["space"]).lower()
    meta = getattr(collection, "metadata", None) or {}
    for key in ("hnsw:space", "space"):
        if meta.get(key):
            return str(meta[key]).lower()
    return None


def _verify_collection_space(collection: Any, name: str, *, context: str) -> None:
    """**断言 collection 真的是 cosine 空间**，不是 cosine 就报错（绝不静默降级）。

    为什么必须有这道闸（docs/10 §3 更正后钉死，本模块实测复现）：
    ``embedding_function=None`` **只管住默认 ONNX 嵌入函数，不管向量空间** ——
    Chroma 的默认空间是 **l2**。实测同一对单位向量 ``[1,0,0]`` vs ``[0.9,0.1,0]``：

    - 只传 ``embedding_function=None`` → ``space='l2'``，distance **0.020000005**
      （=``2(1−cos)``，单位向量下 ``l2`` 距离与余弦不同），``1 − distance`` = 0.979999995
      而 numpy 余弦 = 0.993883735 → **静默错**；
    - ``configuration={"hnsw": {"space": "cosine"}}`` → ``space='cosine'``，
      distance **0.006116271** → ``1 − distance`` = 0.993883729 ≈ numpy 0.993883735 ✅；
    - ``metadata={"hnsw:space": "cosine"}`` → 同上 ✅。

    更隐蔽的一点（本模块实测）：**对已存在的 l2 collection**，
    ``get_or_create_collection(..., configuration={"hnsw":{"space":"cosine"}})`` /
    ``metadata={"hnsw:space":"cosine"}`` **都不会改建库空间**（仍是 ``l2``）——
    复用路径不校验就会拿到一个「看起来正常、语义却错」的库。故复用与新建**两条路径都校验**。

    读不到空间（``None``）时也报错：宁可让调用方显式清理/换名前缀，也不在未知空间上跑检索。
    """
    space = _collection_space(collection)
    if space != _REQUIRED_SPACE:
        raise ValueError(
            f"collection {name!r} 的向量空间是 {space!r}（{context}），"
            f"必须是 {_REQUIRED_SPACE!r}：Chroma 默认空间是 l2，"
            "而本索引的『相似度 = 1 − distance』只在 cosine 空间成立。"
            "修法：显式建库 configuration={'hnsw': {'space': 'cosine'}}"
            f"（已存在且空间不符的库不会被自动改建 —— 请删除后重建：delete_collection({name!r})，"
            "或换 collection_prefix 用新库）"
        )


def check_cosine_self_check(
    collection: Any, sample_vector: list[float], *, name: str
) -> float:
    """cosine 自检：用「库内已有向量」查库，返回自身距离（cosine 空间必须 ≈ 0）。

    取一条**已入库**的向量（通常 ``_doc_vectors[0]``）作查询：cosine 空间里它与自身的
    距离应为 0。实测（MockHashEmbedder，真服务端）出现两类值：``0.0`` 与
    ``-1.1920929e-07``（float32 尾差，Chroma 存 float32）—— 故用 ``abs() <= _SELF_CHECK_TOL``
    判定，并让调用方据此回带精确数值（返回原值，便于打印实测证据）。

    真值来源仍是 :func:`_verify_collection_space` 的 configuration 判定（本项自身距离在
    **l2 空间同样为 0**，单独看不区分空间）；本函数是第二道防线：抓「库未 upsert / 被清空 /
    空间被外力改掉后数值不可信」这类问题，并给出可打印的实测数字。
    「单位向量自检」（``1 − distance`` vs numpy 余弦）需要一条**已知不同**的单位向量对，
    由验收脚本在独立 collection 上做（本函数不往业务库里塞探针向量）。
    """
    result = collection.query(
        query_embeddings=[list(sample_vector)], n_results=1, include=["distances"]
    )
    distances = (result.get("distances") or [[]])[0]
    if not distances:
        raise ValueError("cosine 自检失败：collection 查询未返回任何结果（库是否已 upsert？）")
    value = float(distances[0])
    if abs(value) > _SELF_CHECK_TOL:
        raise ValueError(
            f"cosine 自检失败：collection {name!r} 内向量与自身距离 {value!r}"
            f"（cosine 空间应为 0，容差 {_SELF_CHECK_TOL}）—— 库可能未正确 upsert 或空间异常"
        )
    return value


def check_cosine_space(collection: Any, name: str) -> str:
    """检索前复检空间（默认对**每次** ``search`` 调用执行，返回空间名）。

    读 ``configuration["hnsw"]["space"]``（:func:`_collection_space`）并断言为 cosine。
    成本可忽略（框架本地字段读取，无网络）；收益是「空间错误」永远在检索前暴露，而不是
    让错误的 ``1 − distance`` 静默流进 ``CaseHit.retrieval_score`` 与 Evidence.weight。
    """
    _verify_collection_space(collection, name, context="检索前复检")
    return _collection_space(collection) or _REQUIRED_SPACE


def _open_collection(
    *, client: Any, name: str, dim: int, kind: str, prefix: str
) -> tuple[Any, bool]:
    """``get_or_create_collection``（**必须 ``embedding_function=None`` + 显式 cosine 空间**）。

    实测（docs/10 §3）两条独立约束，缺一不可：

    1. **``embedding_function=None``**：不显式关掉，Chroma 会启用**默认 ONNX 嵌入函数**
       并尝试下载模型 —— 本项目自带向量，必须关闭；
    2. **``configuration={"hnsw": {"space": "cosine"}}``**：Chroma 默认空间是 **l2**，
       只关嵌入函数**不会**变成 cosine（实测见 :func:`_verify_collection_space`）。

    返回 ``(collection, reused)``：两条路径都校验空间（复用路径尤其重要 —— 实测对已存在的
    l2 库再传 cosine 配置**不会改建库空间**）；维度写入 collection metadata 供复用校验。
    同名不同前缀由调用方负责（``name`` 已含 prefix+kind+dim）。
    """
    _import_chroma()  # 缺包早失败（提示装 extra）；本函数只用注入进来的 client
    try:
        existing_collection = client.get_collection(name=name)
    except Exception as exc:
        if "not found" not in str(exc).lower() and type(exc).__name__ != "NotFoundError":
            raise
    else:
        _verify_collection_space(existing_collection, name, context="已存在，复用")
        return existing_collection, True
    collection = client.get_or_create_collection(
        name=name,
        embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},  # ★ 显式 cosine：默认是 l2
        metadata={_DIM_KEY: int(dim), "pra_kind": kind, "pra_prefix": prefix},
    )
    _verify_collection_space(collection, name, context="本次新建")
    return collection, False


def _validate_collection_dim(collection: Any, name: str, dim: int) -> None:
    """复用已有 collection 时校验维度（Chroma 不声明维度 → 读我们写入的 metadata）。"""
    existing = (collection.metadata or {}).get(_DIM_KEY)
    if existing is not None and int(existing) != int(dim):
        raise ValueError(
            f"collection {name!r} 已存在且维度 {existing} != 期望 {dim}："
            "同名不同维不可复用（换 collection_prefix 或清理旧库）"
        )


def delete_collection(
    name: str,
    *,
    chroma_client: Any | None = None,
    host: str = CHROMA_DEFAULT_HOST,
    port: int = CHROMA_DEFAULT_PORT,
) -> bool:
    """删除 collection（测试清理用）；不存在返回 False（幂等，不抛）。"""
    client = make_chroma_client(chroma_client, host=host, port=port)
    try:
        client.delete_collection(name)
        return True
    except Exception as exc:
        if "not found" in str(exc).lower() or type(exc).__name__ == "NotFoundError":
            return False
        raise


def served_counters() -> dict[str, int]:
    """本进程内经本模块发出的计数快照（``llm_calls`` 必须恒为 0 —— 检索侧零 LLM）。"""
    return dict(SERVED_COUNTERS)


def reset_served_counters() -> None:
    """清零 ``SERVED_COUNTERS``（测试/验收断言前调用）。"""
    for key in SERVED_COUNTERS:
        SERVED_COUNTERS[key] = 0


# ---------------------------------------------------------------------------
# jieba 分词接入（bm25s.tokenize 的受控替换，见模块 docstring）
# ---------------------------------------------------------------------------

#: 全局替换锁：bm25s.tokenize 是模块级符号，构造与检索期间独占。
_TOKENIZER_LOCK = threading.RLock()


def _jieba_tokens(text: str) -> list[str]:
    """jieba 精确模式切词（确定性；拉丁/数字串拆出并小写，中文词原样）。

    不做停用词过滤/词干化：中文语料上英文词干器无意义，且过滤器会引入版本相关行为。
    单字符词（如「的」）保留 —— 其文档频率高、IDF 低，对排序影响可忽略；宁可不裁剪也
    不引入一份需要维护的停用词表。
    """
    import jieba

    tokens: list[str] = []
    for piece in jieba.cut(text, cut_all=False):
        piece = piece.strip()
        if not piece:
            continue
        for latin in _LATIN_RUN.findall(piece):
            tokens.append(latin.lower())
        if _LATIN_RUN.sub("", piece):
            tokens.append(piece)
    return tokens


def _jieba_bm25s_tokenize(texts: Any, **_kwargs: Any) -> Any:
    """``bm25s.tokenize`` 的 jieba 替身（签名兼容：忽略 token_pattern/stemmer/stopwords 等）。

    行为：逐文档 jieba 切词 → 词表「首次出现即编号」（跨文档共享，确定性）；查询与索引
    走同一函数 → 缺失查询词也进词表，**不会**触发 bm25s 的越界 token id 报错。
    返回 ``bm25s.tokenization.Tokenized``（``BM25Retriever`` 只用其 ``ids``）。
    """
    from bm25s.tokenization import Tokenized

    single = isinstance(texts, str)
    items = [texts] if single else list(texts)
    vocab: dict[str, int] = {}
    ids: list[list[int]] = []
    for text in items:
        doc_ids: list[int] = []
        for tok in _jieba_tokens(str(text)):
            if tok not in vocab:
                vocab[tok] = len(vocab)
            doc_ids.append(vocab[tok])
        ids.append(doc_ids)
    return Tokenized(ids=ids, vocab=vocab)


@contextmanager
def _jieba_tokenizer():
    """在上下文内把 ``bm25s.tokenize`` 换成 jieba 实现（退出即恢复原符号）。

    ``BM25Retriever`` 构造与 ``retrieve`` 两处都硬编码调用 ``bm25s.tokenize``，且**没有**
    tokenizer 注入点（0.8.0 实测），故只能在此上下文内完成索引与查询 —— 保证「索引分词」
    与「查询分词」是同一个分词器（否则词表不一致）。

    🔴 **调用方必须写成 ``with _TOKENIZER_LOCK, _jieba_tokenizer():``（锁在前、补丁在后）**：
    多个上下文管理器按「左→右 __enter__、右→左 __exit__」执行，若写成
    ``with _jieba_tokenizer(), _TOKENIZER_LOCK:``，则打补丁在取锁**之前**、恢复在放锁
    **之后** —— 临界区不覆盖补丁的安装/撤销，两个并发线程下必然出错（实测，R6）：
    线程 B 会把「A 打的补丁」当成 ``original`` 存下，A 退出即恢复真身 → **B 在自以为的
    jieba 上下文内用真分词器检索 jieba 建的索引**（实测
    ``ValueError: The maximum token ID in the query (56) is higher than the number of
    tokens in the index.``）；B 退出再把补丁写回 → ``bm25s.tokenize`` **进程级永久泄漏**。
    锁在前则补丁窗口 ⊆ 持锁窗口，两线程的补丁窗口互不相交，save/restore 自然成栈。
    """
    import bm25s

    original = bm25s.tokenize
    bm25s.tokenize = _jieba_bm25s_tokenize
    try:
        yield
    finally:
        bm25s.tokenize = original


# ---------------------------------------------------------------------------
# corpus 行 → LlamaIndex Node（1 条款 / 1 先例 = 1 Node，不切碎）
# ---------------------------------------------------------------------------


def _iso_or_empty(value: date | None) -> str:
    """``date`` → ISO 串（None → 空串）。Chroma metadata 不收 None（写库前会被换成 ""）。"""
    return value.isoformat() if value is not None else ""


def policy_node_metadata(row: PolicyClauseRecord) -> dict[str, Any]:
    """Policy 条款 → node metadata（docs/10 §3 规定字段；空 risk_type 不写键）。"""
    meta: dict[str, Any] = {
        "clause_id": row.clause_id,
        "policy_id": row.policy_id,
        "version": int(row.version),
        "category": row.category,
        "status": row.status,
        "effective_date": _iso_or_empty(row.effective_date),
    }
    if row.risk_type:
        meta[_RISK_TYPE_KEY] = [t.value for t in row.risk_type]
    return meta


def case_node_metadata(row: CasePrecedentRecord) -> dict[str, Any]:
    """Case 先例 → node metadata（docs/10 §3 规定字段；空 risk_type 不写键）。"""
    meta: dict[str, Any] = {
        "case_id": row.case_id,
        "category": row.category,
        "decision": row.decision.value,
        "risk_level": row.risk_level.value,
    }
    if row.risk_type:
        meta[_RISK_TYPE_KEY] = [t.value for t in row.risk_type]
    return meta


def _policy_text(row: PolicyClauseRecord) -> str:
    """Policy 检索文本 = ``title。text``（与 ``rag/index.py`` 同口径）。"""
    return f"{row.title}。{row.text}"


def _case_text(row: CasePrecedentRecord) -> str:
    """Case 检索文本 = ``summary``（与 ``rag/index.py`` 同口径）。"""
    return row.summary


def _build_nodes(
    rows: list[Any], *, kind: str, collection: str, llama: dict[str, Any]
) -> tuple[list[Any], list[str]]:
    """corpus 行 → (TextNode 列表, node id 列表)。**1 行 = 1 Node，不切分**。

    **检索文本 = 正文**（policy：``title。text``；case：``summary``）—— 与 local 后端
    （``rag/index.py`` 的 ``_texts``）及本模块向量路 embed 的文本**同一份**。

    节点 metadata **不参与任何检索文本**（R7 决策）：
    ``TextNode(excluded_embed_metadata_keys=<全部 metadata 键>)`` 使
    ``node.get_content(metadata_mode=MetadataMode.EMBED)`` 只返回正文。这一点对 BM25 路是
    **必需**的 —— ``BM25Retriever`` 用的正是 ``MetadataMode.EMBED``（安装源码
    ``bm25s.tokenize([node.get_content(metadata_mode=MetadataMode.EMBED) ...])``），
    不排除就会把 ``case_id`` / ``category`` / ``decision`` / ``risk_level`` / ``risk_type``
    的字面值（如 ``RAG_CASE_0001`` / ``POTENTIAL_IP_RISK``）索引进去，出现「按 metadata
    字面值就能命中」的伪检索。实测（本模块，见返回报告 R7 证据）：不排除时 EMBED 文本以
    ``case_id: RAG_CASE_0001`` / ``category: …`` / ``decision: REJECT`` / ``risk_level: HIGH`` /
    ``risk_type: ['POTENTIAL_IP_RISK', …]`` 开头再接正文；排除后 EMBED 文本 == 正文。
    该排除设置**随 ``node_to_metadata_dict`` 的 ``_node_content`` JSON 往返存活**（已实测），
    故 ``BM25Retriever`` 由 metadata 重建节点时排除仍然生效。

    注：``metadata`` 本身仍完整保留（含 R-4 隔离所需字段），只是不进检索文本。
    """
    nodes: list[Any] = []
    node_ids: list[str] = []
    for row in rows:
        key = row.clause_id if kind == "policy" else row.case_id
        nid = _node_id(collection, key)
        node_ids.append(nid)
        meta = policy_node_metadata(row) if kind == "policy" else case_node_metadata(row)
        text = _policy_text(row) if kind == "policy" else _case_text(row)
        nodes.append(
            llama["TextNode"](
                id_=nid,
                text=text,
                metadata=meta,
                # R7：metadata 不进检索文本（EMBED/LLM 两个模式都排除；ALL 仍保留供审计）。
                excluded_embed_metadata_keys=list(meta.keys()),
                excluded_llm_metadata_keys=list(meta.keys()),
            )
        )
    return nodes, node_ids


# ---------------------------------------------------------------------------
# 候选过滤（policy / case 各一份谓词；向量路与 BM25 路共用，绝无双实现）
# ---------------------------------------------------------------------------


def _policy_candidates(rows: list[PolicyClauseRecord], filters: PolicySearchFilters, effective_only: bool) -> list[int]:
    """Policy 候选行索引（语义与 ``rag/index.py`` / ``rag/qdrant_index.py`` 逐条一致）。

    ``effective_only`` → ``status == "EFFECTIVE"``；``category`` ∈ {None, 值, 全类目}；
    ``risk_type`` 与给定集合交叠非空。
    """
    candidates: list[int] = []
    for i, r in enumerate(rows):
        if effective_only and r.status != "EFFECTIVE":
            continue
        if filters.category and r.category not in (None, filters.category, _FULL_CATEGORY):
            continue
        if filters.risk_type:
            wanted = set(filters.risk_type)
            if not (wanted & set(r.risk_type)):
                continue
        candidates.append(i)
    return candidates


def _case_candidates(rows: list[CasePrecedentRecord], filters: CaseSearchFilters) -> list[int]:
    """Case 候选行索引（category **精确匹配**、risk_type 交叠非空 —— 与既有实现一致）。"""
    candidates: list[int] = []
    for i, r in enumerate(rows):
        if filters.category and r.category != filters.category:
            continue
        if filters.risk_type:
            wanted = set(filters.risk_type)
            if not (wanted & set(r.risk_type)):
                continue
        candidates.append(i)
    return candidates


def _policy_store_filters(filters: PolicySearchFilters, llama: dict[str, Any]) -> Any | None:
    """Policy 下推过滤（向量路）：仅 ``category``（含「全类目」）可下推。

    ``risk_type`` 不下推 —— 实测 Chroma ``$in``/``$eq`` 对**列表字段**恒不命中，
    而 LlamaIndex 的 ``FilterOperator.ANY/CONTAINS`` 在 chroma 集成里**无翻译**会抛
    ValueError（详见模块 docstring「两处实测得到的 Chroma 过滤能力边界」）。
    """
    if not filters.category:
        return None
    return llama["MetadataFilters"](
        filters=[
            llama["MetadataFilter"](
                key="category",
                value=[filters.category, _FULL_CATEGORY],
                operator=llama["FilterOperator"].IN,
            )
        ]
    )


def _case_store_filters(filters: CaseSearchFilters, llama: dict[str, Any]) -> Any | None:
    """Case 下推过滤（向量路）：仅 ``category``（精确匹配可下推）。

    ``risk_type`` 同 policy：Chroma 列表字段无可用成员操作符 → 留在 Python 侧。
    """
    if not filters.category:
        return None
    return llama["MetadataFilters"](
        filters=[
            llama["MetadataFilter"](
                key="category",
                value=filters.category,
                operator=llama["FilterOperator"].EQ,
            )
        ]
    )


# ---------------------------------------------------------------------------
# 检索器装配（向量 / BM25 / RRF 融合）
# ---------------------------------------------------------------------------


@dataclass
class _RetrievalContext:
    """一次检索需要的装配面（两个索引类共用；避免 policy / case 双实现）。"""

    kind: str
    rows: list[Any]
    row_keys: list[str]
    node_ids: list[str]
    nodes: list[Any]
    collection: Any | None
    embed_model: Any
    vector_store: Any
    llama: dict[str, Any]
    #: node_id → corpus 原序行索引（排序 tie-break 用原序，不依赖底层库返回顺序）。
    row_index_by_key: dict[str, int] = field(default_factory=dict)


def _filters_to_chroma_where(store_filters: Any | None) -> dict | None:
    """LlamaIndex ``MetadataFilters`` → Chroma ``where`` dict（本模块自实现的**显式**下推）。

    支持本项目实际用到的两种算子（``in`` / ``eq``，可组合成 ``$and``）；出现别的算子即
    报错（不静默忽略 —— 静默会变成「下推了但没推」的隐性错误）。

    ⚠️ 为什么不用 ``llama_index.vector_stores.chroma.base._to_chroma_filter``：本模块的向量路
    直接读 **Chroma 原生返回的 distance**（见 :func:`_make_vector_retriever` 的实测理由），
    不再经过 ``ChromaVectorStore.query``，故 where 子句也由本函数显式转换（行为等价、可审计）。
    """
    if store_filters is None:
        return None
    items = list(getattr(store_filters, "filters", []) or [])
    clauses: list[dict] = []
    for item in items:
        op = getattr(item, "operator", None)
        key = getattr(item, "key", None)
        value = getattr(item, "value", None)
        if key is None:
            raise ValueError(f"不支持的 MetadataFilters 元素（无 key）: {item!r}")
        name = getattr(op, "value", op)
        if name in (None, "==", "eq"):
            clauses.append({key: value})
        elif name in ("in",):
            clauses.append({key: {"$in": list(value)}})
        else:
            raise ValueError(f"chroma where 下推不支持算子 {name!r}（key={key!r}）")
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _make_vector_retriever(
    ctx: _RetrievalContext,
    similarity_top_k: int,
    store_filters: Any | None,
    candidate_ids: list[str] | None = None,
) -> Any:
    """向量检索器：Chroma（cosine 距离）+ ``MetadataFilters`` 下推，分数 = ``1 − distance``。

    ⚠️ **实测**：``ChromaVectorStore.query`` 把分数算成 ``similarity = exp(-distance)``
    （读安装源码 ``_query`` 可见 ``similarity_score = math.exp(-distance)``；实测
    ``[1,0,0]`` vs ``[0.9,0.1,0]`` → store 给 0.993902，而 ``1 − distance`` = 0.9938837）。
    那是**另一套映射、不是余弦**，会破坏「向量分 = 余弦相似度」这一 docs/10 §3 钉死的口径。

    故本检索器**只借用标准 LlamaIndex 装配面**（``BaseRetriever`` 子类 + ``MetadataFilters``
    下推语义 + ``TextNode``），但取数走 **Chroma 原生 ``collection.query``**：拿 ``distance``
    自己算 ``1 − distance``。节点对象由本模块的 node id → TextNode 表直接映射（node id
    由行键派生，逐行唯一），**不经** ``_node_content`` JSON 反序列化（那一步在语料含
    相同内容行时会与 ``node.hash`` 去重冲突）。

    ★ **``candidate_ids`` = 候选集的 node id（本模块给的精确集合）** —— 这是漏召回 bug 的修复点。

    原缺陷（另一个 agent 测出、本模块复现）：先用 ``MetadataFilters`` 把 ``where`` 下推，
    再按 ``n_results = len(candidates)`` 取 top-N —— 但**下推的 ``where`` 只是候选谓词的
    超集**（``risk_type`` 无法下推：Chroma 列表字段没有成员算子；``effective_only`` 也不在
    ``where`` 里）。于是**非候选行按距离抢占 top-N 名额**，被 Python 复核剔除后没有补位，
    真候选从未被打分 → 结果是 local 的**真子集**，最坏为**空**。实测（真服务端 + Mock 嵌入，
    修复前）：policy ``外观模仿`` + ``risk_type=[FALSE_CLAIM]`` + ``effective_only`` k=6 →
    local 3 / chroma **0**；policy ``品牌词`` + ``effective_only``（无任何 store 过滤！）k=30 →
    local 21 / chroma **19**（只因 EXPIRED 行抢位）。

    修法：把**精确候选 node id 集合**交给 Chroma（``ids=`` 过滤，等价 ``$in``，但走原生 ids
    参数），即「store 返回的集合 **⊇** 候选集」由构造保证（请求的就是候选本身），
    ``n_results = min(candidate_count, collection.count())``；随后 Python 复核 + ``top_k``
    截断照旧。**并断言覆盖率**（见 :meth:`_ChromaCosineRetriever._retrieve`）：返回 id 集合
    必须覆盖候选集合，否则重试，重试后仍不覆盖即抛。
    """

    class _ChromaCosineRetriever(ctx.llama["BaseRetriever"]):
        """Chroma cosine 检索器：分数 = ``1 − distance``（**不是** ``exp(-distance)``）。"""

        def __init__(self) -> None:
            super().__init__()
            self._collection = ctx.collection
            self._emb = ctx.embed_model
            self._k = int(similarity_top_k)
            self._where = _filters_to_chroma_where(store_filters)
            # Chroma 返回的是 **node id**（`_node_id()` 派生），不是 corpus 行键 —— 映射键必须用 node id。
            self._by_id = {nid: node for nid, node in zip(ctx.node_ids, ctx.nodes)}
            #: 精确候选 id（NodeWithScore 只允许这些 id 出现；None = 未限定，取 ``where`` 命中的 top-N）。
            self._candidate_ids = list(candidate_ids) if candidate_ids is not None else None

        def _query_once(self, query_embedding: list[float]) -> tuple[list[str], list[float]]:
            """一次 Chroma 查询 → (ids, distances)。

            ``n_results`` 的取值原则：**只要候选**（``len(candidate_ids)``），并夹到
            ``collection.count()`` —— 实测 ``n_results`` 大于库内条数不会报错（返回全部），
            但显式夹住可让「要多少」与「只可能有多少」一致，便于覆盖率断言归因。
            """
            kwargs: dict[str, Any] = {
                "query_embeddings": [list(query_embedding)],
                "include": ["distances"],
            }
            if self._candidate_ids is not None:
                kwargs["ids"] = list(self._candidate_ids)
                kwargs["n_results"] = max(1, min(len(self._candidate_ids), int(self._collection.count())))
            else:
                kwargs["n_results"] = self._k
            if self._where is not None:
                kwargs["where"] = self._where
            result = self._collection.query(**kwargs)
            return (
                list((result.get("ids") or [[]])[0]),
                list((result.get("distances") or [[]])[0]),
            )

        def _retrieve(self, query_bundle: Any) -> list[Any]:
            if self._collection is None or not self._by_id:
                return []
            query_embedding = self._emb.get_query_embedding(query_bundle.query_str)
            # 覆盖率自检（**不假定「要了 N 就一定拿到 N」**）：chromadb 1.5.9 实测存在少返
            # （另一 agent 测到 count()=69 而 query(n_results=69) 只回 68；本模块 ephemeral
            #  紧接 upsert 也见过）。重试 _VECTOR_COVERAGE_ATTEMPTS 次，仍缺真候选即抛。
            wanted = set(self._candidate_ids) if self._candidate_ids is not None else None
            ids: list[str] = []
            distances: list[float] = []
            for attempt in range(1, _VECTOR_COVERAGE_ATTEMPTS + 1):
                ids, distances = self._query_once(query_embedding)
                if wanted is None or wanted.issubset(set(ids)):
                    break
                if attempt == _VECTOR_COVERAGE_ATTEMPTS:
                    missing = sorted(wanted - set(ids))
                    raise RuntimeError(
                        f"向量路覆盖率不足：{len(missing)} 个候选 node 未从 Chroma 取回 "
                        f"（如 {missing[:5]}），已重试 {_VECTOR_COVERAGE_ATTEMPTS} 次；"
                        f"collection={getattr(self._collection, 'name', '?')!r} "
                        f"count={self._collection.count()} requested={len(wanted)} got={len(ids)}。"
                        "这会让结果静默变成子集 —— 拒绝返回，请检查 collection 是否被清理/与索引不一致。"
                    )
            out: list[Any] = []
            for nid, distance in zip(ids, distances):
                node = self._by_id.get(nid)
                if node is None:  # 防御：候选集外的 node id（不应发生）
                    continue
                out.append(
                    ctx.llama["NodeWithScore"](
                        node=node, score=_cosine_from_distance(distance)
                    )
                )
            return out

    return _ChromaCosineRetriever()


def _make_bm25_retriever(ctx: _RetrievalContext, top_k: int) -> Any:
    """BM25 检索器（``bm25s`` 引擎 + jieba 分词）：**只喂候选 node**（Python 侧过滤）。

    ``similarity_top_k=top_k`` 取候选集内的 Top-K（候选集本身已是过滤后集合）；
    ``skip_stemming=True`` / ``language=""`` 关掉英文 Stemmer 与英文停用词（中文语料无意义）。
    ``token_pattern=""``：本模块的 jieba 替身**忽略**该参数（见 :func:`_jieba_bm25s_tokenize`），
    传空串只为显式标注「不启用 bm25s 的正则切词」。

    **索引文本 = 正文**（R7 决策，见 :func:`_build_nodes`）：该库内部取
    ``node.get_content(metadata_mode=MetadataMode.EMBED)``，本模块构造 node 时用
    ``excluded_embed_metadata_keys`` 把 metadata 全部排除，故 BM25 路与向量路
    （``title。text`` / ``summary``）**检索同一份文本** —— metadata 字面值
    （``case_id`` / ``risk_type`` 等）不再进入 BM25 词表（实测证据见返回报告 R7 段）。
    """
    # ⚠️ 顺序不可颠倒：锁**在**补丁之前（见 :func:`_jieba_tokenizer` docstring，R6 实测）。
    with _TOKENIZER_LOCK, _jieba_tokenizer():
        return ctx.llama["BM25Retriever"](
            nodes=list(ctx.nodes),
            similarity_top_k=min(top_k, len(ctx.nodes)),
            skip_stemming=True,
            language="",
            token_pattern="",
            verbose=False,
        )


def _make_fusion_retriever(
    ctx: _RetrievalContext, vector_retriever: Any, bm25_retriever: Any, top_k: int
) -> Any:
    """RRF 融合检索器 = ``QueryFusionRetriever(mode="reciprocal_rerank", num_queries=1)``。

    ⚠️ ``num_queries=1`` **必填**：默认 4 会调用 LLM 生成 query 变体（读安装源码
    ``_retrieve``：``if self.num_queries > 1: queries.extend(self._get_queries(...))``）——
    本项目检索侧**零 LLM 调用**，故 1 + 显式 ``MockLLM()`` 守卫。不传 ``llm`` 时该库回落
    ``Settings.llm`` → 本环境 ``ImportError: llama-index-llms-openai package not found``
    （项目刻意不装该集成），故必须显式传守卫对象；``num_queries=1`` 下它**永不被调用**。

    **本函数是给外部/测试用的「配置正确」构造器，不参与本模块检索链路** —— 原因见
    :func:`_fuse_rrf` 的 docstring（实测该实现会原地改写共享 node 的 score，跨查询污染）。
    """
    return ctx.llama["QueryFusionRetriever"](
        retrievers=[vector_retriever, bm25_retriever],
        llm=ctx.llama["MockLLM"](),
        mode="reciprocal_rerank",
        similarity_top_k=min(top_k, len(ctx.nodes)),
        num_queries=1,
        use_async=False,
        verbose=False,
    )


def _fuse_rrf(
    ranked: dict[str, list[str]], row_index: dict[str, int]
) -> list[tuple[int, float]]:
    """RRF（Reciprocal Rank Fusion）：``score(d) = Σ_r 1 / (k + rank_r(d))``，``k=60``。

    口径与 ``QueryFusionRetriever._reciprocal_rerank_fusion`` **逐条一致**（同 k=60、同
    「按该路分数降序定 rank」定义，docstring 亦同源引用 Cormack 2009），但**由本模块自己算**：

    ⚠️ **实测（本轮，见返回报告 §5）**：``QueryFusionRetriever`` 在融合时会**原地改写**
    它拿到的 ``NodeWithScore.node.score``（源码 ``reranked_nodes[-1].score = score``），而
    这些 node 对象是**跨检索共享的同一批实例**（我方向量路 / BM25 路给的是同一批 node）
    → 融合分被写回共享对象，**下一次检索的融合结果会随此前的调用序列漂移**。实测证据：

    - 每次新建 retriever + 每次新建 fusion（``num_queries=1``）连查 3 次：
      ``[('RAG_CASE_0066', 0.030118), ...]`` / ``[('RAG_CASE_0061', 0.016667), ...]`` /
      ``[('RAG_CASE_0061', 0.016667), ...]`` —— **第一次就与后两次不同**；
    - 复用同一 fusion 对象连查 2 次：top-1 从 ``RAG_CASE_0064``(0.020393) 变成
      ``RAG_CASE_0020``(0.022821)（分 > 1/(60+1)，说明同一 node 被重复计入）；
    - 复用同一对 retriever 对象两次新建 fusion：top-1 从 ``RAG_CASE_0020`` 变成
      ``RAG_CASE_0061``。

    docs/10 §3 的**确定性红线**（「不得退化」）直接排斥这种行为，故 hybrid 路取「同一融合
    定义 + 本模块自算」：排名输入来自两路检索器（同一 query、覆盖全部候选），融合与 tie-break
    由本模块显式完成 —— 逐次运行结果一致（验收已断言两次运行逐字节相同）。

    另注：该库的融合用 ``node.hash``（内容哈希）去重，而本 corpus **实测存在内容相同的行**
    （67 case 中 2 对哈希相同）→ 依赖内容哈希去重会把不同先例合并成一个；本模块用
    **node id**（由行键派生，逐行唯一）作融合键，不受此影响。

    ⚠️ **上界是 ``2/60``（``≈0.0333``），不是 ``2/61``**：rank 从 **0** 起（``enumerate(ids)``），
    故首位贡献 ``1/(60+0) = 1/60``；两路都排首位即 ``2/60 = 1/30``（实测 0.033333）。
    与 llama-index ``_reciprocal_rerank_fusion`` 的 ``1.0 / (rank + k)`` 同式（其 rank 亦从 0 起）。

    返回 ``[(行索引, 6 位 RRF 分)]``，排序 key = ``(分降序, corpus 原序 idx 升序)``。
    """
    fused: dict[str, float] = {}
    for ids in ranked.values():
        for offset, nid in enumerate(ids):
            fused[nid] = fused.get(nid, 0.0) + 1.0 / (_RRF_K + offset)
    out = [(row_index[nid], round(score, 6)) for nid, score in fused.items()]
    out.sort(key=lambda t: (-t[1], t[0]))
    return out


# ---------------------------------------------------------------------------
# Policy / Case 两个公开索引类（构造/检索签名与既有实现对齐）
# ---------------------------------------------------------------------------


class _ChromaIndexBase:
    """两个索引类的共用装配（collection / node / 向量 upsert / 三模式检索）。"""

    _kind = ""

    def _setup(
        self,
        rows: Iterable[dict | Any],
        *,
        record_type: type,
        text_of,
        embedder: Embedder | None,
        mode: str,
        weights: tuple[float, float],
        chroma_client: Any | None,
        host: str,
        port: int,
        ephemeral: bool,
        collection_prefix: str | None,
    ) -> None:
        self._rows: list[Any] = _normalize_rows(rows, record_type)
        self.mode: RetrievalMode = _validate_mode(mode)
        self.weights: tuple[float, float] = tuple(weights)  # 兼容既有构造签名（RRF 不用权重）
        self.embedder: Embedder = embedder or MockHashEmbedder()
        self._texts: list[str] = [text_of(r) for r in self._rows]
        self._row_keys: list[str] = [
            (r.clause_id if self._kind == "policy" else r.case_id) for r in self._rows
        ]
        self._llama = _import_llama()
        self._client = make_chroma_client(
            chroma_client, host=host, port=port, ephemeral=ephemeral
        )
        self._embed_model = LlamaIndexEmbeddingAdapter(self.embedder)
        self._doc_vectors: list[list[float]] = [
            self._embed_model.get_text_embedding(t) for t in self._texts
        ]
        self._dim = _resolve_dim(self.embedder, self._doc_vectors)
        self.collection_prefix = collection_prefix
        self.collection_name = (
            _collection_name(collection_prefix, self._kind, self._dim) if self._rows else ""
        )
        self._collection = None
        #: ``ChromaVectorStore``（LlamaIndex 官方 store 包装）——**装配面保留**：外部/测试可
        #: 用它走标准 ``VectorStoreQuery`` 通路（docs/10 §4 要求的 vector-stores-chroma 集成
        #: 确实被构造并被引用）；本模块自己的向量取数走原生 ``collection.query``（理由见模块
        #: docstring 的 ``vector`` 段：该 store 的 ``exp(-distance)`` 不是余弦）。
        self._vector_store = None
        self._nodes: list[Any] = []
        self._node_ids: list[str] = []
        if self._rows:
            self._seed()

    # -- 装配 ---------------------------------------------------------------

    def _seed(self) -> None:
        """建/复用 collection（``embedding_function=None`` **+ 显式 cosine 空间**）+ 幂等 upsert。

        node 向量 = 文本向量（与 local 后端同一 embedder/同一文本，逐位一致）；节点按
        「1 行 = 1 node」写入，metadata 见 ``*_node_metadata``。

        空间闸（docs/10 §3 更正）：``_open_collection`` 对**新建与复用两条路径**都断言
        collection 是 cosine（默认 l2 会让「相似度 = 1 − distance」静默失效）；upsert 后
        再做一次 cosine 自检（自身距离须 ≈ 0），失败即抛 —— **绝不带着未知空间继续检索**。
        """
        collection, reused = _open_collection(
            client=self._client,
            name=self.collection_name,
            dim=self._dim,
            kind=self._kind,
            prefix=self.collection_prefix or _DEFAULT_COLLECTION_PREFIX,
        )
        if reused:
            _validate_collection_dim(collection, self.collection_name, self._dim)
        self._nodes, self._node_ids = _build_nodes(
            self._rows,
            kind=self._kind,
            collection=self.collection_name,
            llama=self._llama,
        )
        collection.upsert(
            ids=list(self._node_ids),
            embeddings=[list(v) for v in self._doc_vectors],
            metadatas=[node.metadata for node in self._nodes],
            documents=[node.get_content() for node in self._nodes],
        )
        self.collection_space = _collection_space(collection)
        self._self_check_distance = check_cosine_self_check(
            collection, self._doc_vectors[0], name=self.collection_name
        )
        self._collection = collection
        self._vector_store = self._llama["ChromaVectorStore"](chroma_collection=collection)

    # -- size / 统计 ---------------------------------------------------------

    @property
    def size(self) -> int:
        """corpus 行数（policy：**含 EXPIRED 历史版**；case：先例数）——与 qdrant 后端同义。"""
        return len(self._rows)

    @property
    def space(self) -> str | None:
        """collection 实际向量空间（构造后应为 ``"cosine"``；空 KB 未建库则为 ``None``）。

        供测试/审计读取（对应 ``configuration["hnsw"]["space"]``）。
        """
        return self.collection_space

    @property
    def cosine_self_check_distance(self) -> float | None:
        """构造期 cosine 自检值：库内首条向量与自身的距离（cosine 空间应 ≈ 0；空库为 None）。"""
        return self._self_check_distance

    @property
    def collection_count(self) -> int:
        """collection 内 node 数（真服务端落库校验用；空库为 0）。"""
        return int(self._collection.count()) if self._collection is not None else 0

    @property
    def node_ids(self) -> list[str]:
        """本索引写入的 node id（顺序与 corpus 行一致；测试/审计用）。"""
        return list(self._node_ids)

    @property
    def nodes(self) -> list[Any]:
        """LlamaIndex ``TextNode`` 列表（1 行 = 1 node；测试/审计用）。"""
        return list(self._nodes)

    # -- 装配子件 -----------------------------------------------------------

    def _sub_context(self, candidates: list[int]) -> _RetrievalContext:
        """候选子集上的检索上下文（node / node id / 行索引映射都只含候选）。

        ``row_index_by_key`` 把 node id 直接映射回 **corpus 原序行索引** —— 后续排序
        tie-break 用原序，因此不依赖 Chroma / bm25s 的返回顺序。
        """
        sub_ids = [self._node_ids[i] for i in candidates]
        return _RetrievalContext(
            kind=self._kind,
            rows=[self._rows[i] for i in candidates],
            row_keys=[self._row_keys[i] for i in candidates],
            node_ids=[self._node_ids[i] for i in candidates],
            nodes=[self._nodes[i] for i in candidates],
            collection=self._collection,
            embed_model=self._embed_model,
            vector_store=self._vector_store,
            llama=self._llama,
            row_index_by_key={nid: candidates[offset] for offset, nid in enumerate(sub_ids)},
        )

    def _bm25_retrieve(self, retriever: Any, query_bundle: Any) -> list[Any]:
        """BM25 检索（**在 jieba 上下文内** —— 查询与索引必须同一分词器）。

        ⚠️ 必须与构造期同一个 ``bm25s.tokenize``：索引词表由 jieba 建立，若查询仍走 bm25s
        原分词器，未登录词会拿到越界 token id →
        ``ValueError: The maximum token ID in the query (379) is higher than the number of
        tokens in the index.``（本模块首次实测即此错，见返回报告 §5）。

        ⚠️ 顺序不可颠倒：锁**在**补丁之前（见 :func:`_jieba_tokenizer` docstring，R6 实测）。
        """
        with _TOKENIZER_LOCK, _jieba_tokenizer():
            return retriever.retrieve(query_bundle)

    def _rank_vector(
        self,
        sub_ctx: _RetrievalContext,
        query_bundle: Any,
        top_k: int,
        store_filters: Any | None,
        *,
        count_search: bool = True,
    ) -> list[tuple[int, float]]:
        """向量路排名：``[(行索引, 1 − distance)]``（**精确候选 id 集** + store 侧 category 下推）。

        与 local 后端的等价性由三件事保证（本模块实测：同序同 id，分差 ≤ 1e-6）：

        1. **打分域 = 精确候选集** —— 检索器只对 ``sub_ctx.node_ids``（候选）取距离，
           不再让非候选行抢 top-N 名额（漏召回 bug 的根因，见 :func:`_make_vector_retriever`
           docstring）；
        2. **覆盖率自检** —— 返回 id 必须覆盖候选 id，否则重试/抛（第三方少返是已知风险）；
        3. **兜底补算** —— 万一覆盖率自检失败但需继续（例如部分候选确实取不回），
           缺失候选按**本模块已存向量**（Chroma ``get`` 回来的 doc 向量）与 query 向量
           现算 cos（复用 ``rag/vectors.cosine_similarity``，不新写第二套余弦），
           并累加 ``SERVED_COUNTERS["vector_bruteforce_fallbacks"]`` 使其可见。

        ``top_k`` 传候选数上限时返回全部候选（hybrid 的 RRF 需要完整排名列表）。
        ``count_search``：本次是否记一次 ``vector_searches``（hybrid 会调本函数一次 +
        另计一次 ``hybrid_searches``；重试/兜底不重复计数）。
        """
        retriever = _make_vector_retriever(
            sub_ctx, top_k, store_filters, candidate_ids=list(sub_ctx.node_ids)
        )
        try:
            nodes = retriever.retrieve(query_bundle)
        except RuntimeError:
            # 覆盖率重试仍不足 → 已存向量兜底（结果仍完整；计数可见）。
            nodes = []
        if count_search:
            SERVED_COUNTERS["vector_searches"] += 1
        scored: dict[str, float] = {
            n.node.node_id: n.score for n in nodes if n.node.node_id in sub_ctx.row_index_by_key
        }
        missing = [nid for nid in sub_ctx.node_ids if nid not in scored]
        if missing:
            scored.update(self._score_missing_by_stored_vectors(sub_ctx, query_bundle, missing))
            SERVED_COUNTERS["vector_bruteforce_fallbacks"] += 1
        ranked = [(sub_ctx.row_index_by_key[nid], score) for nid, score in scored.items()]
        ranked.sort(key=lambda t: (-round(t[1], 6), t[0]))
        return [(i, round(s, 6)) for i, s in ranked]

    def _score_missing_by_stored_vectors(
        self, sub_ctx: _RetrievalContext, query_bundle: Any, missing: list[str]
    ) -> dict[str, float]:
        """兜底：对未取回的候选，用 Chroma 里**已存 doc 向量**与 query 向量现算余弦。

        - 向量来源 ``collection.get(ids=missing, include=["embeddings"])``（本地读取，无 ANN 近似）；
        - 余弦用仓库既有 ``rag/vectors.cosine_similarity``（纯 Python，零第三方依赖；
          与 local numpy/纯 Python 后端同源码），再按 ``_cosine_from_distance`` 同口径 clamp 到 [0,1]；
        - 任一步失败（缺向量/长度不符）即抛，**不静默少返** —— 少返正是本次要修的 bug。
        """
        raise_on_missing = (
            f"向量路兜底失败：{len(missing)} 个候选取不到向量（如 {missing[:5]}）——"
            "拒绝返回子集结果"
        )
        if self._collection is None:
            raise RuntimeError(raise_on_missing)
        got = self._collection.get(ids=list(missing), include=["embeddings"])
        # 注意：embeddings 是 numpy 数组列表 —— 不能用 ``or []``（数组真值歧义，实测
        # ``ValueError: The truth value of an array with more than one element is ambiguous``）。
        got_ids = list(got.get("ids") if got.get("ids") is not None else [])
        raw_emb = got.get("embeddings")
        got_vecs = [list(v) for v in ([] if raw_emb is None else raw_emb)]
        if len(got_ids) != len(missing) or len(got_vecs) != len(missing):
            raise RuntimeError(
                f"{raise_on_missing}（collection.get 只回 {len(got_ids)}/{len(missing)} 条）"
            )
        query_embedding = sub_ctx.embed_model.get_query_embedding(query_bundle.query_str)
        out: dict[str, float] = {}
        for nid, vec in zip(got_ids, got_vecs):
            raw = float(cosine_similarity(list(query_embedding), vec))
            if math.isnan(raw):  # 防御：退化输入（cosine_similarity 自身已 clamp 0~1）
                raw = 0.0
            out[nid] = max(0.0, min(1.0, raw))
        return out

    def _rank_bm25(
        self, sub_ctx: _RetrievalContext, query_bundle: Any, top_k: int
    ) -> list[tuple[int, float]]:
        """BM25 路排名：候选集内 min-max 归一化 BM25 分（Python 侧过滤 = 只喂候选 node）。

        归一化口径与 local 后端 ``rank_documents(mode="bm25")`` 同函数（``normalize_minmax``）：
        候选集内最高分 = 1.0，分数恒 ⊂ [0,1]（``CaseHit.retrieval_score`` 的 ``le=1`` 约束成立）。
        """
        retriever = _make_bm25_retriever(sub_ctx, top_k)
        nodes = self._bm25_retrieve(retriever, query_bundle)
        SERVED_COUNTERS["bm25_searches"] += 1
        pairs = [
            (sub_ctx.row_index_by_key[n.node.node_id], float(n.score or 0.0))
            for n in nodes
            if n.node.node_id in sub_ctx.row_index_by_key
        ]
        pairs.sort(key=lambda t: (-t[1], t[0]))
        norm = normalize_minmax([s for _i, s in pairs])
        ranked = [(i, s) for (i, _raw), s in zip(pairs, norm)]
        ranked.sort(key=lambda t: (-round(t[1], 6), t[0]))
        return [(i, round(s, 6)) for i, s in ranked]

    def _rank_hybrid(
        self,
        sub_ctx: _RetrievalContext,
        query_bundle: Any,
        top_k: int,
        store_filters: Any | None,
    ) -> list[tuple[int, float]]:
        """Hybrid 路：两路排名 → **RRF 融合**（``Σ_r 1/(60 + rank_r)``，``k=60``）。

        两路各自返回**全部候选**（``top_k`` = 候选数）→ 在完整排名列表上融合
        （:func:`_fuse_rrf`，含「为什么不用 ``QueryFusionRetriever`` 现成融合」的实测理由）。
        ⚠️ 该分是**融合排名分，不是相似度**（上界 **2/60 = 1/30 ≈ 0.0333**，docs/10 §6-R5）；排序 key
        = ``(分降序, corpus 原序 idx 升序)``。
        """
        vec_ranked = self._rank_vector(sub_ctx, query_bundle, top_k, store_filters)
        bm25_ranked = self._rank_bm25(sub_ctx, query_bundle, top_k)
        SERVED_COUNTERS["hybrid_searches"] += 1
        # RRF 的输入是「两路各自的排名列表」：把 (行索引, 分) 排名还原为 node id 顺序。
        id_by_row = {row: nid for nid, row in sub_ctx.row_index_by_key.items()}
        ranked_lists = {
            "vector": [id_by_row[row] for row, _score in vec_ranked],
            "bm25": [id_by_row[row] for row, _score in bm25_ranked],
        }
        return _fuse_rrf(ranked_lists, sub_ctx.row_index_by_key)

    # -- 检索核心 -----------------------------------------------------------

    def _retrieve_ranked(
        self,
        query: str,
        candidates: list[int],
        *,
        top_k: int,
        store_filters: Any | None,
        recheck,
    ) -> list[tuple[int, float]]:
        """三模式检索 → ``[(行索引, 6 位检索分)]``（已按 ``(分降序, 原序)`` 排序、已截断 Top-K）。

        向量路在 store 下推之后用 ``recheck``（**与 BM25 路同一份** Python 谓词）复核一遍：
        下推只用于收窄候选，判定只有一份实现，杜绝双实现语义漂移（docs/10 §6-R2）。

        ``top_k`` 截断在**排序之后**统一做；各路内部按需取「候选数」以保证 BM25 的
        min-max 与 RRF 的排名列表覆盖完整候选集。
        """
        if top_k < 1 or not candidates:
            return []
        if self._collection is not None:
            check_cosine_space(self._collection, self.collection_name)  # 检索前空间闸
        sub_ctx = self._sub_context(candidates)
        query_bundle = self._llama["QueryBundle"](query_str=query)
        # 三路都取「候选数」上限：先拿到**完整候选排名**，再复核过滤、最后截断 Top-K。
        # （若这里就按 top_k 预截断，store 返回的前 top_k 里一旦有被 recheck 剔除的行，
        #   结果会不足 top_k —— 实测 policy vector + effective_only 就是这样少一条。）
        full_k = len(candidates)
        if self.mode == "bm25":
            ranked = self._rank_bm25(sub_ctx, query_bundle, full_k)
        elif self.mode == "vector":
            ranked = self._rank_vector(sub_ctx, query_bundle, full_k, store_filters)
        else:
            ranked = self._rank_hybrid(sub_ctx, query_bundle, full_k, store_filters)
        ranked = [item for item in ranked if recheck(item[0])]
        return ranked[:top_k]


def _cosine_from_distance(distance: float) -> float:
    """Chroma cosine ``distance`` → 余弦相似度 ``1 − distance``（docs/10 §3 实测口径）。

    实测（本模块首次运行即复现）：``[1,0,0]`` vs ``[0.9,0.1,0]`` → ``distance = 0.006116271``
    而 ``1 − cos = 0.00611627``；``[1,0,0]`` vs ``[1,1,0]`` → ``distance = 0.29289323``
    而 ``1 − cos = 0.29289322``。故相似度 = ``1 − distance``（Chroma 对入库向量做过
    L2 归一化，查询向量不做 —— 其 cosine 距离即 `1 − 余弦`，与 local numpy 余弦同口径）。

    夹到 ``[0,1]`` 仅作防御：浮点尾差可能给出 ``-1e-9`` / ``1+1e-9``，而
    ``CaseHit.retrieval_score`` 约束 ``ge=0, le=1``（不夹会让整个检索抛 ValidationError）。
    NaN（零向量等退化输入）按 0 计。

    ⚠️ 这是**向量路的检索分**（余弦相似度）；hybrid 路的分数是 RRF 融合分，二者不同量纲。
    """
    value = float(distance)
    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, 1.0 - value))


def _validate_mode(mode: str) -> RetrievalMode:
    """与 ``rag/index.py`` / ``rag/qdrant_index.py`` 同口径：非法模式即报错（不静默降级）。"""
    if mode not in MODES:
        raise ValueError(f"未知检索模式: {mode!r}（可选: {list(MODES)}）")
    return mode  # type: ignore[return-value]


def _normalize_rows(rows: Iterable[Any], record_type: type) -> list[Any]:
    """rows（dict 或 record 模型）→ 强校验的 record 列表（与既有实现同源码）。"""
    out: list[Any] = []
    for row in rows:
        out.append(record_type.model_validate(row) if isinstance(row, dict) else row)
    for row in out:
        if not isinstance(row, record_type):
            raise TypeError(f"corpus 行须为 {record_type.__name__} 或 dict: got {type(row)!r}")
    return out


class ChromaPolicyIndex(_ChromaIndexBase):
    """``PolicyIndex`` Protocol 的 Chroma + LlamaIndex 实现。

    构造参数与 ``RagPolicyIndex`` / ``QdrantPolicyIndex`` 对齐，另加 Chroma 装配参数
    （``chroma_client`` 注入 / ``host``+``port`` 自建 / ``ephemeral`` 离线内存库 /
    ``collection_prefix``）。检索签名与工具契约逐字一致：

    ``async def search(query, filters, top_k, effective_only) -> list[PolicyClauseHit]``

    过滤语义（effective_only / category / risk_type）与既有实现**逐条一致**；差异只在
    「向量库 + 检索器 + 融合」（Chroma / LlamaIndex RRF）。
    """

    _kind = "policy"

    def __init__(
        self,
        rows: Iterable[dict | PolicyClauseRecord],
        *,
        embedder: Embedder | None = None,
        mode: RetrievalMode = "hybrid",
        weights: tuple[float, float] = DEFAULT_WEIGHTS,
        chroma_client: Any | None = None,
        host: str = CHROMA_DEFAULT_HOST,
        port: int = CHROMA_DEFAULT_PORT,
        ephemeral: bool = False,
        collection_prefix: str | None = None,
    ) -> None:
        self._setup(
            rows,
            record_type=PolicyClauseRecord,
            text_of=_policy_text,
            embedder=embedder,
            mode=mode,
            weights=weights,
            chroma_client=chroma_client,
            host=host,
            port=port,
            ephemeral=ephemeral,
            collection_prefix=collection_prefix,
        )

    def effective_count(self) -> int:
        """当前生效条款数（``status == "EFFECTIVE"``）——与既有实现同义。"""
        return sum(1 for r in self._rows if r.status == "EFFECTIVE")

    async def search(
        self,
        query: str,
        filters: PolicySearchFilters,
        top_k: int,
        effective_only: bool,
    ) -> list[PolicyClauseHit]:
        """检索政策条款（三模式；无命中 → ``[]``，工具 ok=True）。

        流程：a. Python 侧候选过滤（与既有实现同一谓词）→ 候选空即返回 ``[]``；
        b. 按模式装配检索器（向量路 store 侧下推 category；BM25 路只喂候选 node）；
        c. 打分/融合（RRF）；d. ``(分降序, corpus 原序)`` 排序 + 6 位取整 → Top-K。
        """
        candidates = _policy_candidates(self._rows, filters, effective_only)
        ranked = self._retrieve_ranked(
            query,
            candidates,
            top_k=top_k,
            store_filters=_policy_store_filters(filters, self._llama),
            recheck=lambda i: i in set(candidates),
        )
        SERVED_COUNTERS["served_hits"] += len(ranked)
        return [
            PolicyClauseHit.model_validate(self._rows[i].model_dump(mode="json"))
            for i, _score in ranked
        ]


class ChromaCaseIndex(_ChromaIndexBase):
    """``CaseIndex`` Protocol 的 Chroma + LlamaIndex 实现。

    ``async def search(query, filters, top_k) -> list[CaseHit]`` —— 与工具契约逐字一致。

    ``CaseHit.retrieval_score`` 口径（docs/10 §0 C1，**勿误读**）：它是**检索分，不是语义
    相似度** —— ``bm25`` 模式 = 候选集内 min-max 归一化 BM25 分；``vector`` 模式 =
    ``1 − distance``（余弦相似度）；**``hybrid`` 模式 = RRF 融合分**（``Σ 1/(k+rank)``，
    ``k=60``，落在 ~(0, **2/60 = 1/30**]）。取值恒 ⊂ ``[0,1]``（``CaseHit`` 的 ``ge=0, le=1`` 约束成立）。
    """

    _kind = "case"

    def __init__(
        self,
        rows: Iterable[dict | CasePrecedentRecord],
        *,
        embedder: Embedder | None = None,
        mode: RetrievalMode = "hybrid",
        weights: tuple[float, float] = DEFAULT_WEIGHTS,
        chroma_client: Any | None = None,
        host: str = CHROMA_DEFAULT_HOST,
        port: int = CHROMA_DEFAULT_PORT,
        ephemeral: bool = False,
        collection_prefix: str | None = None,
    ) -> None:
        self._setup(
            rows,
            record_type=CasePrecedentRecord,
            text_of=_case_text,
            embedder=embedder,
            mode=mode,
            weights=weights,
            chroma_client=chroma_client,
            host=host,
            port=port,
            ephemeral=ephemeral,
            collection_prefix=collection_prefix,
        )

    async def search(
        self, query: str, filters: CaseSearchFilters, top_k: int
    ) -> list[CaseHit]:
        """检索先例（三模式；无命中 → ``[]``，工具 ok=True）。流程同 ``ChromaPolicyIndex``。"""
        candidates = _case_candidates(self._rows, filters)
        ranked = self._retrieve_ranked(
            query,
            candidates,
            top_k=top_k,
            store_filters=_case_store_filters(filters, self._llama),
            recheck=lambda i: i in set(candidates),
        )
        SERVED_COUNTERS["served_hits"] += len(ranked)
        hits: list[CaseHit] = []
        for i, score in ranked:
            row = self._rows[i]
            hits.append(
                CaseHit.model_validate(
                    {
                        **row.model_dump(mode="json"),
                        "retrieval_score": score,  # 检索分（hybrid=RRF 分），**非语义相似度**
                    }
                )
            )
        return hits
