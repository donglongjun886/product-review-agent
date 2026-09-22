"""默认路径「零额外依赖」契约测试 —— CI 上恒跑。

契约：``import pra.rag`` / ``import pra.tools`` 以及默认 ``build_tools()``（memory 世界）时，
``chromadb`` / ``llama_index`` / ``jieba`` / ``bm25s`` / ``fastembed`` 一律**不得**进入
``sys.modules``（真实检索后端统一在「显式装配 chroma 索引」或「首次检索」时才延迟 import）。

为什么这是 CI 上唯一跑得动的 RAG 守护：CI 只跑 ``uv sync --frozen``（不装任何 extra），
真实检索（chroma）的用例需要 ``chromadb`` / ``fastembed`` 与已缓存模型，在 CI 上不执行任何
一行断言（故不得声称「CI 覆盖了 chroma」）。

注意：默认 ``build_policy_index()`` / ``build_case_index()`` **已不再是零依赖路径** —— 它们只保留
chroma 后端（缺 rag extra 时构造即抛），故不在本文件的守护范围内；本文件钉的是「默认 memory 装配
与 import 路径不引入重依赖」这条懒加载红线。

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
#: ``llama_index.core`` / ``bm25s.tokenization`` / ``fastembed.…`` 等同属违规）。
_FORBIDDEN_TOP_LEVEL = ("chromadb", "llama_index", "jieba", "bm25s", "fastembed")

_PAYLOAD_MARKER = "PRA_DEFAULT_PATH_GUARD_JSON:"

#: 子进程脚本：跑**默认**装配（import 包 + memory 世界 + 生产装配），然后检查 sys.modules
#: （同一进程内先后关系，保证「默认路径跑过之后仍未引入额外依赖」而不是「压根没跑」）。
#: 用普通字符串 + 占位替换（而非 f-string）—— 脚本里全是 dict/集合字面量，f-string 会把
#: 它们当格式字段解析（曾因此收集期 NameError）。
_CHILD_SCRIPT_TEMPLATE = """
import importlib.util, json, sys

FORBIDDEN = __FORBIDDEN_TOP_LEVEL__


def installed(name):
    # find_spec 在残缺安装 / 自定义 meta_path finder 下可能直接抛（如实按「未安装」处理）
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def main() -> dict:
    import pra.rag   # 包导入：不得拉起重依赖
    import pra.tools
    from pra.rag.corpus import load_cases, load_policies
    from pra.tools import build_production_tools, build_tools

    policy_rows = load_policies()[0]
    case_rows = load_cases()[0]
    tools_memory = build_tools()           # 默认世界：InMemory 种子
    prod_tools = build_production_tools()  # 生产装配：惰性 RAG，装配期不得 import 后端
    # 按名取工具（装配顺序对下游无语义：ToolRegistry / tools_node 全程按名调度）
    memory_by_name = {t.name: t for t in tools_memory}
    prod_by_name = {t.name: t for t in prod_tools}

    forbidden = {}
    for name in FORBIDDEN:
        mods = sorted(k for k in list(sys.modules) if k == name or k.startswith(name + "."))
        if mods:
            forbidden[name] = mods

    return {
        "forbidden_present": forbidden,
        "chromadb_installed": installed("chromadb"),
        "llama_index_installed": installed("llama_index"),
        "fastembed_installed": installed("fastembed"),
        "rag_loaded": "pra.rag" in sys.modules,
        "tools_loaded": "pra.tools" in sys.modules,
        "tool_names": [t.name for t in tools_memory],
        "memory_index_types": [
            type(memory_by_name["CaseSearchTool"]._index).__name__,
            type(memory_by_name["PolicySearchTool"]._index).__name__,
        ],
        "prod_tool_index_types": [
            type(prod_by_name["CaseSearchTool"]._index).__name__,
            type(prod_by_name["PolicySearchTool"]._index).__name__,
        ],
        "prod_tool_built": [
            prod_by_name["CaseSearchTool"]._index.is_built,
            prod_by_name["PolicySearchTool"]._index.is_built,
        ],
        "corpus_rows": [len(policy_rows), len(case_rows)],
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
    """``import pra.rag`` / ``pra.tools`` + 默认 memory 装配**与生产装配**之后，
    ``chromadb`` / ``llama_index`` / ``jieba`` / ``bm25s`` / ``fastembed`` 不得进入
    ``sys.modules``（生产装配经 ``Lazy*Index`` 惰性构建，首次检索才拉起后端）。"""
    payload = _run_default_path_child()
    assert payload["forbidden_present"] == {}, (
        "默认路径引入了额外依赖（CI 上 chromadb/llama_index/fastembed 根本没装 → 会 ImportError）："
        f"{payload['forbidden_present']}"
    )


def test_default_path_guard_is_not_vacuous() -> None:
    """反「空断言」守卫：默认路径必须真的 import 了 pra.rag/pra.tools、建了 6 个 memory 工具
    并读出了语料，生产装配注入惰性代理且尚未建库。"""
    payload = _run_default_path_child()
    assert payload["rag_loaded"] and payload["tools_loaded"], (
        "子进程实际未 import pra.rag/pra.tools → 本用例会空跑（先修脚本）"
    )
    assert payload["tool_names"] == [
        "ProductTool", "ImageAnalysisTool", "OCRTool", "MerchantTool",
        "CaseSearchTool", "PolicySearchTool",
    ]
    assert payload["memory_index_types"] == ["InMemoryCaseIndex", "InMemoryPolicyIndex"], (
        "build_tools() 默认（memory）仍须注入 InMemory 索引"
    )
    assert payload["prod_tool_index_types"] == ["LazyCaseIndex", "LazyPolicyIndex"], (
        "生产装配必须注入惰性 RAG 代理（装配期不建库/不 import 后端）"
    )
    assert payload["prod_tool_built"] == [False, False], (
        "生产装配后索引必须尚未构建（首次检索才建）"
    )
    assert len(payload["corpus_rows"]) == 2 and all(r > 0 for r in payload["corpus_rows"]), (
        "两路 InMemory 索引必须真的读出了种子语料（行数规模不锁，随语料演进自由变化）"
    )


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
