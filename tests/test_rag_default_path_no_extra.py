"""默认路径「零额外依赖」契约测试 —— CI 上恒跑。

契约：默认路径 ``build_policy_index()`` / ``build_case_index()`` / ``build_tools()`` 必须仍是
``local`` 后端，绝不因为新增 backend 开关把 ``chromadb`` / ``llama_index`` / ``jieba`` /
``bm25s`` 拉进 ``sys.modules``。

为什么这是 CI 上唯一跑得动的 RAG 守护：CI 只跑 ``uv sync --frozen``（不装任何 extra），
而 ``tests/test_rag_chroma.py`` / ``tests/test_rag_chroma_server.py`` 顶层
``pytest.importorskip("chromadb")`` → 整文件 skip，在 CI 上不执行任何一行断言
（故不得声称「CI 覆盖了 chroma」）。

断言为什么用子进程：同进程里别的测试文件可能已经 ``import chromadb``，进程内查
``sys.modules`` 会变成永真/永假的空断言。子进程是全新解释器：先跑默认路径，再查
``sys.modules``，断言才不是 vacuous。
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

#: 必须**不出现**在默认路径进程 ``sys.modules`` 里的顶层模块名（前缀匹配：
#: ``llama_index.core`` / ``bm25s.tokenization`` 等同属违规）。
_FORBIDDEN_TOP_LEVEL = ("chromadb", "llama_index", "jieba", "bm25s")

_PAYLOAD_MARKER = "PRA_DEFAULT_PATH_GUARD_JSON:"

#: 子进程脚本：跑**默认**装配 + 一次真实检索，然后检查 sys.modules（同一进程内先后关系，
#: 保证「默认路径跑过之后仍未引入额外依赖」而不是「压根没跑」）。
#: 用普通字符串 + 占位替换（而非 f-string）—— 脚本里全是 dict/集合字面量，f-string 会把
#: 它们当格式字段解析（曾因此收集期 NameError）。
_CHILD_SCRIPT_TEMPLATE = """
import asyncio, importlib.util, json, sys

FORBIDDEN = __FORBIDDEN_TOP_LEVEL__


def installed(name):
    # find_spec 在残缺安装 / 自定义 meta_path finder 下可能直接抛（如实按「未安装」处理）
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def main() -> dict:
    from pra.rag.corpus import load_cases, load_policies
    from pra.rag.factory import build_case_index, build_policy_index
    from pra.tools import build_production_tools, build_tools

    policy_rows = load_policies()[0]
    case_rows = load_cases()[0]
    policy_index = build_policy_index()          # 默认 backend="local"
    case_index = build_case_index()              # 默认 backend="local"
    tools_memory = build_tools()                 # 默认 data_source="memory"
    tools_rag = build_tools("rag")               # 默认 rag_backend="local"
    prod_tools = build_production_tools()        # 生产装配：惰性 RAG，装配期不得 import 后端

    async def search():
        from pra.tools.case_search.tool import CaseSearchFilters
        hits = await tools_rag[4]._index.search("无品牌高相似", CaseSearchFilters(), top_k=3)
        return [h.case_id for h in hits]

    hits = asyncio.run(search())

    forbidden = {}
    for name in FORBIDDEN:
        mods = sorted(k for k in list(sys.modules) if k == name or k.startswith(name + "."))
        if mods:
            forbidden[name] = mods

    return {
        "forbidden_present": forbidden,
        "chromadb_installed": installed("chromadb"),
        "llama_index_installed": installed("llama_index"),
        "policy_type": type(policy_index).__name__,
        "case_type": type(case_index).__name__,
        "policy_size": policy_index.size,
        "case_size": case_index.size,
        "policy_effective": policy_index.effective_count(),
        "corpus_rows": [len(policy_rows), len(case_rows)],
        "rag_tool_index_types": [
            type(tools_rag[4]._index).__name__,
            type(tools_rag[5]._index).__name__,
        ],
        "prod_tool_index_types": [
            type(prod_tools[4]._index).__name__,
            type(prod_tools[5]._index).__name__,
        ],
        "prod_tool_built": [prod_tools[4]._index.is_built, prod_tools[5]._index.is_built],
        "tool_names": [t.name for t in tools_memory],
        "hits": hits,
    }


print(__PAYLOAD_MARKER__ + json.dumps(main(), ensure_ascii=False))
"""

_CHILD_SCRIPT = (
    _CHILD_SCRIPT_TEMPLATE.replace("__FORBIDDEN_TOP_LEVEL__", repr(_FORBIDDEN_TOP_LEVEL))
    .replace("__PAYLOAD_MARKER__", repr(_PAYLOAD_MARKER))
)


def _module_installed(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # noqa: BLE001 — 探测失败一律按「不可导入」处理（只降级，不掩盖）
        return False


def _run_default_path_child() -> dict:
    # 全新解释器；子进程 stdout 最后一行回传证据 JSON
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_SCRIPT],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, (
        f"默认路径子进程失败（rc={proc.returncode}）：\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(_PAYLOAD_MARKER)]
    assert lines, f"子进程未回传证据行：\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    return json.loads(lines[-1][len(_PAYLOAD_MARKER):])


def test_default_path_pulls_in_no_extra_dependencies() -> None:
    """默认路径**与生产装配**之后，``chromadb`` / ``llama_index`` / ``jieba`` / ``bm25s``
    不得进入 ``sys.modules``（生产装配经 ``Lazy*Index`` 惰性构建，首次检索才拉起后端）。"""
    payload = _run_default_path_child()
    assert payload["forbidden_present"] == {}, (
        "默认路径引入了额外依赖（CI 上 chromadb/llama_index 根本没装 → 会 ImportError）："
        f"{payload['forbidden_present']}"
    )


def test_default_path_guard_is_not_vacuous() -> None:
    """反「空断言」守卫：默认路径必须真的建了索引并检索出结果（24/67 行、21 条生效条款、6 个工具）。"""
    payload = _run_default_path_child()
    assert payload["policy_type"] == "RagPolicyIndex"
    assert payload["case_type"] == "RagCaseIndex"
    assert payload["rag_tool_index_types"] == ["RagCaseIndex", "RagPolicyIndex"], (
        "build_tools('rag') 默认后端必须仍是 local（默认后端不变）"
    )
    assert payload["prod_tool_index_types"] == ["LazyCaseIndex", "LazyPolicyIndex"], (
        "生产装配必须注入惰性 RAG 代理（装配期不建库/不 import 后端）"
    )
    assert payload["prod_tool_built"] == [False, False], (
        "生产装配后索引必须尚未构建（首次检索才建）"
    )
    assert payload["policy_size"] == 24 and payload["case_size"] == 67
    assert payload["corpus_rows"] == [24, 67]
    assert payload["policy_effective"] == 21
    assert payload["tool_names"] == [
        "ProductTool", "ImageAnalysisTool", "OCRTool", "MerchantTool",
        "CaseSearchTool", "PolicySearchTool",
    ]
    assert payload["hits"] and all(h.startswith("RAG_CASE_") for h in payload["hits"])


def test_default_index_path_stays_local_in_process() -> None:
    """进程内兜底（不依赖任何 extra）：默认实现的类型没被 backend 开关换掉，保证 v1/v2 回归
    digest 零变化；不碰 ``sys.modules``（同进程里别的测试可能已 import 过 chromadb）。"""
    import asyncio

    from pra.rag.corpus import load_cases, load_policies
    from pra.rag.factory import build_case_index, build_policy_index
    from pra.rag.index import RagCaseIndex, RagPolicyIndex
    from pra.tools.case_search.tool import CaseSearchFilters
    from pra.tools.policy_search.tool import PolicySearchFilters

    policy_index = build_policy_index()
    case_index = build_case_index()
    assert isinstance(policy_index, RagPolicyIndex)
    assert isinstance(case_index, RagCaseIndex)
    assert policy_index.size == len(load_policies()[0]) == 24
    assert case_index.size == len(load_cases()[0]) == 67

    async def _search() -> tuple[int, int]:
        ph = await policy_index.search(
            "外观高度模仿知名品牌", PolicySearchFilters(), top_k=3, effective_only=True
        )
        ch = await case_index.search("无品牌高相似", CaseSearchFilters(), top_k=3)
        return len(ph), len(ch)

    policy_hits, case_hits = asyncio.run(_search())
    assert policy_hits == 3 and case_hits == 3


@pytest.mark.parametrize("module_name", _FORBIDDEN_TOP_LEVEL)
def test_extra_module_really_exists_when_installed(module_name: str) -> None:
    if not _module_installed(module_name):
        pytest.skip(f"{module_name} 未安装（CI 只跑 uv sync --frozen，不装 extra）→ 无法给出反证")

    proc = subprocess.run(
        [sys.executable, "-c", f"import {module_name}"],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, (
        f"{module_name} 已安装却无法 import → 「默认路径未引入」的子进程用例失去意义：\n"
        f"{proc.stderr}"
    )
