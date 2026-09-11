"""生产入口的 RAG 接线 —— 装配惰性 / 首次检索才构建 / 参数透传 / 真 Chroma e2e。

契约：
- ``build_production_tools()`` 把 CaseSearchTool / PolicySearchTool 指向真实 RAG
  （``LazyCaseIndex`` / ``LazyPolicyIndex``），但**装配期零 import、零 IO**：不建库、不连
  Chroma、不加载模型，失败留到首次检索（由 ``tools_node`` 记 warn failure）；
- 首次 ``search`` 才调用 builder，之后复用同一实例；**构建失败不缓存**（下次重试）；
- 默认 ``build_tools()``（以及 conftest 钉回的测试路径）仍是 InMemory 种子 —— 评测确定性红线。

真 Chroma 段需要 rag extra + 服务端可达 + BGE 模型缓存，缺一即 **运行期 skip**（不在收集期
import fastembed，避免污染其他用例的 ``sys.modules`` 断言）。CI 只跑 ``uv sync --frozen``
（不装 extra）→ 必然 skip；CI 上真正跑得动的守护是默认路径零 extra 契约测试。
"""

from __future__ import annotations

import importlib.util
import socket
from pathlib import Path
from uuid import uuid4

import pytest

import pra.tools as tools_pkg
from pra.tools import build_tools

# 模块导入期抓真实装配函数：conftest 的 autouse fixture 会在测试期把 ``pra.tools`` 上的
# 同名属性换成 ``build_tools``（保证 CI 不连库/不连 Chroma），此处保留真身供本文件使用。
_REAL_BUILD_PRODUCTION_TOOLS = tools_pkg.build_production_tools

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BGE_CACHE = _REPO_ROOT / ".cache" / "model_cache"
_CHROMA_HOST = "127.0.0.1"
_CHROMA_PORT = 8001


# ---------------------------------------------------------------------------
# 生产装配：惰性注入
# ---------------------------------------------------------------------------


def test_production_assembly_injects_lazy_rag_indices():
    """生产装配 = 默认 6 工具；商品/商家换真库、案例/政策换惰性 RAG；**装配期不构建**。"""
    from pra.rag.lazy_index import LazyCaseIndex, LazyPolicyIndex

    prod = _REAL_BUILD_PRODUCTION_TOOLS()
    default = build_tools()
    assert [t.name for t in prod] == [t.name for t in default]
    assert type(prod[0]._repo).__name__ == "MySQLProductRepository"
    assert type(prod[3]._repo).__name__ == "MySQLMerchantRepository"
    assert isinstance(prod[4]._index, LazyCaseIndex)
    assert isinstance(prod[5]._index, LazyPolicyIndex)
    # 惰性的全部意义：装配完还没建库（未 import 后端、未连服务端、未加载模型）
    assert prod[4]._index.index is None and prod[4]._index.is_built is False
    assert prod[5]._index.index is None and prod[5]._index.is_built is False
    # 其余两个仍是 Mock 桩（本轮未动 image/ocr）
    assert type(prod[1]).__name__ == type(default[1]).__name__ == "ImageAnalysisTool"
    assert type(prod[2]).__name__ == type(default[2]).__name__ == "OCRTool"


def test_default_tools_keep_inmemory_knowledge_sources():
    """默认 ``build_tools()`` 的两个知识库工具仍是 InMemory 种子（不改评测可重放口径）。"""
    from pra.tools.case_search.tool import InMemoryCaseIndex
    from pra.tools.policy_search.tool import InMemoryPolicyIndex

    default = build_tools()
    assert isinstance(default[4]._index, InMemoryCaseIndex)
    assert isinstance(default[5]._index, InMemoryPolicyIndex)


# ---------------------------------------------------------------------------
# 惰性代理语义（不依赖 extra / 服务端）
# ---------------------------------------------------------------------------


async def test_lazy_index_defers_build_then_caches():
    from pra.rag.lazy_index import LazyCaseIndex
    from pra.tools.case_search.tool import CaseSearchFilters

    calls: list[int] = []

    class _FakeIndex:
        async def search(self, query, filters, top_k):
            return [("hit", query, top_k)]

    def _builder():
        calls.append(1)
        return _FakeIndex()

    lazy = LazyCaseIndex(_builder)
    assert lazy.is_built is False and lazy.index is None and calls == []
    assert await lazy.search("q1", CaseSearchFilters(), 3) == [("hit", "q1", 3)]
    assert calls == [1] and lazy.is_built
    assert isinstance(lazy.index, _FakeIndex)
    assert await lazy.search("q2", CaseSearchFilters(), 4) == [("hit", "q2", 4)]
    assert calls == [1], "构建必须只发生一次（成功后复用实例）"


async def test_lazy_policy_index_passes_effective_only_through():
    from pra.rag.lazy_index import LazyPolicyIndex
    from pra.tools.policy_search.tool import PolicySearchFilters

    seen: dict = {}

    class _FakeIndex:
        async def search(self, query, filters, top_k, effective_only):
            seen.update(query=query, filters=filters, top_k=top_k, effective_only=effective_only)
            return ["policy-hit"]

    filters = PolicySearchFilters(category="女鞋/运动鞋")
    lazy = LazyPolicyIndex(lambda: _FakeIndex())
    assert await lazy.search("外观模仿", filters, 5, False) == ["policy-hit"]
    assert seen == {
        "query": "外观模仿",
        "filters": filters,
        "top_k": 5,
        "effective_only": False,
    }


async def test_lazy_build_failure_is_not_cached_and_is_retried():
    """构建失败原样上抛、不缓存失败 —— 服务端短暂不可达可自愈（下次检索重试）。"""
    from pra.rag.lazy_index import LazyCaseIndex
    from pra.tools.case_search.tool import CaseSearchFilters

    attempts: list[int] = []

    class _FakeIndex:
        async def search(self, query, filters, top_k):
            return ["ok"]

    def _builder():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("chroma 不可达")
        return _FakeIndex()

    lazy = LazyCaseIndex(_builder)
    with pytest.raises(RuntimeError, match="chroma 不可达"):
        await lazy.search("q", CaseSearchFilters(), 3)
    assert lazy.is_built is False and attempts == [1]
    assert await lazy.search("q", CaseSearchFilters(), 3) == ["ok"]
    assert attempts == [1, 1] and lazy.is_built


async def test_lazy_default_production_builders_target_chroma_bge(monkeypatch):
    """生产 builder 的真实口径：``backend="chroma"`` + ``BgeEmbedder`` + hybrid。

    用假 factory 记录调用参数 —— 不 import 后端、不连服务端，CI 恒跑；真链路在下面的 e2e。
    """
    from pra.rag import factory
    from pra.rag.embedder import BgeEmbedder

    seen: list[tuple[str, dict]] = []
    sentinel = object()

    def _fake_build_case(**kwargs):
        seen.append(("case", kwargs))
        return sentinel

    def _fake_build_policy(**kwargs):
        seen.append(("policy", kwargs))
        return sentinel

    monkeypatch.setattr(factory, "build_case_index", _fake_build_case)
    monkeypatch.setattr(factory, "build_policy_index", _fake_build_policy)
    # 预检放行：只验证「builder 装配成 chroma + BGE + hybrid」，不加载模型/不下载（CI 恒跑）
    monkeypatch.setattr(BgeEmbedder, "available", classmethod(lambda cls: True))
    monkeypatch.setattr(BgeEmbedder, "model_ready", lambda self: True)

    assert tools_pkg._build_production_case_index() is sentinel
    assert tools_pkg._build_production_policy_index() is sentinel
    assert [kind for kind, _ in seen] == ["case", "policy"]
    for _, kwargs in seen:
        assert kwargs["backend"] == "chroma"
        assert kwargs["mode"] == "hybrid"
        assert isinstance(kwargs["embedder"], BgeEmbedder)


def test_production_embedder_fails_fast_instead_of_downloading(monkeypatch):
    """**生产请求期绝不下载模型**：缺 rag extra / 模型未缓存都快速报错（否则请求线程会挂在
    首次下载上——这正是「MySQL e2e 与演示脚本子进程挂死」的根因）。"""
    from pra.rag.embedder import BgeEmbedder

    monkeypatch.setattr(BgeEmbedder, "available", classmethod(lambda cls: False))
    with pytest.raises(RuntimeError, match="rag extra"):
        tools_pkg._production_rag_embedder()

    monkeypatch.setattr(BgeEmbedder, "available", classmethod(lambda cls: True))
    monkeypatch.setattr(BgeEmbedder, "model_ready", lambda self: False)
    with pytest.raises(RuntimeError, match="未缓存"):
        tools_pkg._production_rag_embedder()


# ---------------------------------------------------------------------------
# 真 Chroma + BGE 端到端（缺条件运行期 skip）
# ---------------------------------------------------------------------------


def _e2e_skip_reason() -> str | None:
    missing = [
        name
        for name in ("chromadb", "llama_index", "fastembed")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        return (
            f"未安装 rag extra（缺 {missing}）→ 跳过真 Chroma e2e；"
            "装上：uv sync --extra rag --extra observability"
        )
    try:
        with socket.create_connection((_CHROMA_HOST, _CHROMA_PORT), timeout=1):
            pass
    except OSError:
        return (
            f"Chroma 服务端不可达（{_CHROMA_HOST}:{_CHROMA_PORT}）→ 跳过；"
            "起服务：cd deploy/chroma && docker compose up -d"
        )
    from pra.rag.embedder import BgeEmbedder

    if not BgeEmbedder(cache_dir=str(_BGE_CACHE)).model_ready():
        return (
            f"BGE 模型未缓存（{_BGE_CACHE}）→ 跳过真模型 e2e；"
            "首次需联网下载（HF_ENDPOINT=https://hf-mirror.com）"
        )
    return None


def _delete_prefix(prefix: str) -> None:
    """只删本测试前缀的 collection（Chroma 是共享单实例，绝不动别人的库）。"""
    from pra.rag.chroma_backend import make_chroma_client

    client = make_chroma_client(host=_CHROMA_HOST, port=_CHROMA_PORT)
    for coll in list(client.list_collections()):
        if coll.name.startswith(prefix):
            client.delete_collection(coll.name)


def _demo_case():
    """走查案件（与 ``scripts/demo_api.build_demo_case`` 同口径）。

    图片必须是 Mock 图像源认得的那张（``P_88231/img1.jpg``）—— 否则产不出 IMAGE_SIMILARITY，
    scripted plan 卡在「先做外观比对」分支，永远不会去调 CaseSearch / PolicySearch。
    """
    from helpers import make_case

    from pra.domain.models import ProductImage

    case = make_case(
        case_id=f"CASE_PROD_RAG_{uuid4().hex[:8]}",
        brand=None,
        product_id="P_88231",
        merchant_id="M_5512",
    )
    return case.model_copy(
        update={
            "product": case.product.model_copy(
                update={
                    "images": [
                        ProductImage(
                            url="https://cdn.example.com/products/P_88231/img1.jpg", source="主图"
                        )
                    ]
                }
            )
        }
    )


async def _run_graph(tools) -> dict:
    from pra.agent.graph import build_agent_graph
    from pra.agent.state import build_initial_state

    graph = build_agent_graph(tools=tools, checkpointer=None)
    return await graph.ainvoke(
        build_initial_state(_demo_case()),
        {"configurable": {"thread_id": f"prod-rag-{uuid4().hex[:8]}"}},
    )


async def test_production_rag_reaches_real_knowledge_base(monkeypatch):
    """生产装配 → 惰性构建真 Chroma 库 → 图内检索 → CASE_PRECEDENT / POLICY_REF 证据。

    判别器：真实 KB 的先例 id 全为 ``RAG_CASE_`` 前缀（与评测 GT 隔离），政策条款来自真实
    corpus；同一案件换回 ``build_tools()`` 默认世界则得 InMemory 种子（``CASE_1832`` /
    ``POLICY_3.2_v2_c1``）—— 后半段是本用例的回退反证。

    用 uuid 前缀隔离 collection（共享服务端上自建自删）；embedder 走生产默认 ``BgeEmbedder``
    （缓存目录指到仓库内 ``.cache/model_cache``）。
    """
    reason = _e2e_skip_reason()
    if reason:
        pytest.skip(reason)

    from pra.rag import factory

    monkeypatch.setenv("PRA_EMBED_CACHE_DIR", str(_BGE_CACHE))
    prefix = f"pytest_prod_rag_{uuid4().hex[:8]}"
    real_case, real_policy = factory.build_case_index, factory.build_policy_index
    monkeypatch.setattr(
        factory, "build_case_index", lambda **kw: real_case(collection_prefix=prefix, **kw)
    )
    monkeypatch.setattr(
        factory, "build_policy_index", lambda **kw: real_policy(collection_prefix=prefix, **kw)
    )

    try:
        prod_tools = _REAL_BUILD_PRODUCTION_TOOLS()
        assert prod_tools[4]._index.is_built is False, "装配期不得构建"
        state = await _run_graph(prod_tools)
        assert prod_tools[4]._index.is_built is True, "首次检索后应已构建"
        assert type(prod_tools[4]._index.index).__name__ == "ChromaCaseIndex"

        case_hits = [e for e in state["evidence"] if e.type == "CASE_PRECEDENT"]
        policy_hits = [e for e in state["evidence"] if e.type == "POLICY_REF"]
        assert case_hits, "生产 RAG 世界应产出 CASE_PRECEDENT 证据"
        assert policy_hits, "生产 RAG 世界应产出 POLICY_REF 证据"
        assert all(str(e.ref_id).startswith("RAG_CASE_") for e in case_hits), (
            "先例必须来自真实 Case KB（RAG_CASE_ 前缀 = 与评测 GT 隔离）"
        )
        assert not any(str(e.ref_id).startswith("CASE_18") for e in case_hits), (
            "不得回落到 InMemory 种子 CASE_1832"
        )

        # 回退反证：同一案件走默认 InMemory 世界 → 命中种子先例/政策，而非真实 KB
        memory_state = await _run_graph(build_tools())
        memory_cases = [e.ref_id for e in memory_state["evidence"] if e.type == "CASE_PRECEDENT"]
        assert memory_cases == ["CASE_1832"], (
            "默认世界必须仍是 InMemory 种子（本断言是上面「读了真实 KB」的回退反证）"
        )
    finally:
        _delete_prefix(prefix)
