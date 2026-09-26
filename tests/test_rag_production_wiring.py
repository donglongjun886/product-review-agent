"""生产入口的 RAG 接线 —— 装配惰性 / 首次检索才构建。

契约：
- ``build_production_tools()`` 把 CaseSearchTool / PolicySearchTool 指向真实 RAG
  （``LazyCaseIndex`` / ``LazyPolicyIndex``），但**装配期零 import、零 IO**：不建库、不连
  Chroma、不加载模型，失败留到首次检索（异常经 ``tools_node`` 上抛）；
- 首次 ``search`` 才调用 builder，之后复用同一实例；**构建失败不缓存**（下次重试）；
- 测试世界 = ``inmemory_world`` 的 InMemory 种子（``build_inmemory_tools()``），conftest 把
  生产入口的工具装配钉到它 —— 评测确定性红线。

缺 rag extra 路径由子进程用例守护（不在收集期 import fastembed，避免污染其他用例的
``sys.modules`` 断言）；CI 只跑 ``uv sync --frozen``（不装 extra）→ 该路径必然走到引导报错。
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import pytest
from helpers import tool_by_name
from inmemory_world import build_inmemory_tools

import pra.tools as tools_pkg

# 模块导入期抓真实装配函数：conftest 的 autouse fixture 会在测试期把 ``pra.tools`` 上的
# 同名属性换成 ``inmemory_world.build_inmemory_tools``（保证 CI 不连库/不连 Chroma），
# 此处保留真身供本文件使用。
_REAL_BUILD_PRODUCTION_TOOLS = tools_pkg.build_production_tools


# ---------------------------------------------------------------------------
# 生产装配：惰性注入
# ---------------------------------------------------------------------------


def test_production_assembly_injects_lazy_rag_indices():
    """生产装配 = 4 工具（名称与测试世界一致）；商品/商家换真库、案例/政策换惰性 RAG；**装配期不构建**。"""
    from pra.rag.lazy_index import LazyCaseIndex, LazyPolicyIndex

    prod = _REAL_BUILD_PRODUCTION_TOOLS()
    default = build_inmemory_tools()
    prod_case = tool_by_name(prod, "CaseSearchTool")
    prod_policy = tool_by_name(prod, "PolicySearchTool")
    assert [t.name for t in prod] == [t.name for t in default]
    assert type(tool_by_name(prod, "ProductTool")._repo).__name__ == "MySQLProductRepository"
    assert type(tool_by_name(prod, "MerchantTool")._repo).__name__ == "MySQLMerchantRepository"
    assert isinstance(prod_case._index, LazyCaseIndex)
    assert isinstance(prod_policy._index, LazyPolicyIndex)
    # 惰性的全部意义：装配完还没建库（未 import 后端、未连服务端、未加载模型）——
    # 该契约的行为守护在 tests/test_rag_default_path_no_extra.py（子进程查 sys.modules）。


def test_default_tools_keep_inmemory_knowledge_sources():
    """测试世界 ``build_inmemory_tools()`` 的两个知识库工具仍是 InMemory 种子（评测可重放口径）。"""
    from inmemory_world import InMemoryCaseIndex, InMemoryPolicyIndex

    default = build_inmemory_tools()
    assert isinstance(tool_by_name(default, "CaseSearchTool")._index, InMemoryCaseIndex)
    assert isinstance(tool_by_name(default, "PolicySearchTool")._index, InMemoryPolicyIndex)


# ---------------------------------------------------------------------------
# 惰性代理语义（不依赖 extra / 服务端）
# ---------------------------------------------------------------------------


async def test_lazy_index_defers_build_then_caches():
    """惰性代理：装配期不建库、首次 search 才建、成功后复用同一实例。"""
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
    assert calls == [], "装配期不得构建"
    assert await lazy.search("q1", CaseSearchFilters(), 3) == [("hit", "q1", 3)]
    assert calls == [1]
    assert await lazy.search("q2", CaseSearchFilters(), 4) == [("hit", "q2", 4)]
    assert calls == [1], "构建必须只发生一次（成功后复用实例）"


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
    assert attempts == [1]
    assert await lazy.search("q", CaseSearchFilters(), 3) == ["ok"]
    assert attempts == [1, 1], "失败不得缓存 —— 第二次检索必须重试"


def test_production_embedder_fails_fast_instead_of_downloading(monkeypatch, tmp_path):
    """**生产请求期绝不下载模型**：缓存目录里没有模型时，``local_files_only=True`` 让 fastembed
    只读本地并**立刻**抛错（否则请求线程会挂在首次下载上——这正是 e2e / 演示脚本子进程挂死的根因）。

    非空转：不打桩任何被测函数，真实调用 ``production_embedder()``，只把
    ``PRA_EMBED_CACHE_DIR``（生产代码实际读取的缓存目录开关）指到一个空的临时目录。
    """
    monkeypatch.setenv("PRA_EMBED_CACHE_DIR", str(tmp_path))
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="不在请求期下载模型"):
        tools_pkg.production_embedder()
    assert time.monotonic() - started < 15, "必须是本地立即失败，而不是卡在下载/联网超时上"
    assert not any(tmp_path.rglob("*.onnx")), "失败路径不得在缓存目录留下模型下载产物"


#: 子进程脚本：屏蔽 ``llama_index``（模拟「缺 rag extra」）后调 ``production_embedder``。
#: 「缺 extra」也必须汇到同一条带安装指引的 ``RuntimeError``（import 必须留在 ``try`` 内）。
_NOEXTRA_CHILD_SCRIPT = """
import importlib.abc, json, sys


class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "llama_index" or name.startswith("llama_index."):
            raise ModuleNotFoundError(f"No module named '{name}'")
        return None


sys.meta_path.insert(0, _Block())
for _k in [k for k in sys.modules if k.split(".")[0] == "llama_index"]:
    del sys.modules[_k]
from pra.tools import production_embedder

try:
    production_embedder(cache_dir="/tmp/pra-noextra-guard")
except BaseException as _e:
    print("PRA_NOEXTRA:" + json.dumps({"type": type(_e).__name__, "msg": str(_e)}))
else:
    print("PRA_NOEXTRA:" + json.dumps({"type": "NONE", "msg": ""}))
"""


def test_missing_extra_embedder_raises_guided_runtime_error():
    """缺 rag extra（``llama_index`` 不可导入）时 ``production_embedder`` 必须抛带安装指引的
    ``RuntimeError``（含 ``uv sync --extra rag``），而非裸 ``ModuleNotFoundError``。

    子进程模拟「缺 extra」：装了 extra 的开发 venv 里 ``llama_index`` 已在 ``sys.modules``，
    进程内屏蔽不可靠（先例见 ``test_rag_default_path_no_extra`` 的 docstring）。本用例是这条路径
    在装了 extra 的环境下唯一的守护 —— import 一旦被挪出 ``try`` 会再次静默复发。
    """
    proc = subprocess.run(
        [sys.executable, "-c", _NOEXTRA_CHILD_SCRIPT],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, f"缺 extra 子进程失败：\n{proc.stdout}\n{proc.stderr}"
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("PRA_NOEXTRA:")]
    payload = json.loads(lines[-1][len("PRA_NOEXTRA:") :])
    assert payload["type"] == "RuntimeError", f"缺 extra 应抛 RuntimeError，实为 {payload['type']}"
    assert "uv sync --extra rag" in payload["msg"], "缺 extra 的异常须带安装指引（承重）"
