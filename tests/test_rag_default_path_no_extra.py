"""默认路径「零额外依赖」契约测试（tests/test_rag_default_path_no_extra.py）—— **CI 上恒跑**。

**为什么必须有它**（docs/10 §5-5）：RAG 换成 Chroma + LlamaIndex + bm25s + jieba 之后，
本项目最要紧的一条承诺是「**不装 extra 也能用**」—— 默认路径（``build_policy_index()`` /
``build_case_index()`` / ``build_tools()``）必须仍是 ``local`` 后端，绝不因为新增 backend
开关而把 chromadb / llama-index / jieba / bm25s 拉进 ``sys.modules``。CI 只跑
``uv sync --frozen``（**不装任何 extra**），所以：

**诚实标注（不得声称「CI 覆盖了 chroma」）**：
- ``tests/test_rag_chroma.py`` / ``tests/test_rag_chroma_server.py`` 顶层
  ``pytest.importorskip("chromadb")`` → 在 CI 上**整文件 skip**（chromadb 不在依赖里），
  且真服务端集成用例还额外依赖本机 Docker 服务；**这两份文件在 CI 上不执行任何一行断言**；
- 因此 CI 上真正跑得动的 RAG 守护只有本文件（+ ``tests/test_rag_qdrant_point_id.py``
  的纯 Python u64 契约）。

**断言为什么用子进程**：pytest 单进程里其它测试文件（或在开发机上）可能已经
``import chromadb`` → 进程内 ``sys.modules`` 检查会变成**永真/永假的空断言**。子进程是
**全新解释器**：先跑默认路径，再检查 ``sys.modules``，断言才是真的（不是 vacuous）。
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
    from pra.tools import build_tools

    policy_rows = load_policies()[0]
    case_rows = load_cases()[0]
    policy_index = build_policy_index()          # 默认 backend="local"
    case_index = build_case_index()              # 默认 backend="local"
    tools_memory = build_tools()                 # 默认 data_source="memory"
    tools_rag = build_tools("rag")               # 默认 rag_backend="local"

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
    """该模块在本环境是否可导入（``find_spec`` 直接抛时按不可导入处理 —— 残缺安装/自定义
    ``meta_path`` finder 会出现这种情况，不能让它把用例变成 error）。"""
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # noqa: BLE001 — 探测失败一律按「不可导入」处理（只降级，不掩盖）
        return False


def _run_default_path_child() -> dict:
    """在**全新解释器**里跑默认装配路径并取回证据（子进程 stdout 最后一行 JSON）。"""
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
    """默认 ``build_policy_index()`` / ``build_case_index()`` / ``build_tools()`` 之后，
    ``chromadb`` / ``llama_index`` / ``jieba`` / ``bm25s`` **不得进入 ``sys.modules``**（docs/10 §5-5）。

    这是「不装 extra 也能用」这条公开承诺唯一的 CI 可执行守护：默认路径一旦有人不小心把
    ``chroma_backend``（或它依赖的 jieba/bm25s）提到模块顶层 import，本用例立刻变红 ——
    而真 chroma 用例在 CI 上是 skip 的，抓不到。
    """
    payload = _run_default_path_child()
    assert payload["forbidden_present"] == {}, (
        "默认路径引入了额外依赖（CI 上 chromadb/llama_index 根本没装 → 会 ImportError）："
        f"{payload['forbidden_present']}"
    )


def test_default_path_guard_is_not_vacuous() -> None:
    """反「空断言」守卫：默认路径必须**真的建了索引并检索出结果**。

    若上面的用例只证明「什么东西都没跑」，那它守不住任何东西 —— 故这里断言默认装配的
    实际产物：local 实现类型、corpus 行数（24/67）、生效条款数、6 个工具名、以及经
    ``build_tools("rag")``（默认 local 后端）真的检索到 ``RAG_CASE_`` 命中。
    """
    payload = _run_default_path_child()
    assert payload["policy_type"] == "RagPolicyIndex"
    assert payload["case_type"] == "RagCaseIndex"
    assert payload["rag_tool_index_types"] == ["RagCaseIndex", "RagPolicyIndex"], (
        "build_tools('rag') 默认后端必须仍是 local（docs/10 §2：默认后端不变）"
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
    """进程内兜底（CI 恒跑、不依赖任何 extra）：默认实现的**类型**没被 backend 开关换掉。

    ``build_policy_index()`` / ``build_case_index()`` 缺省仍是 ``Rag*Index``（numpy 余弦 +
    自写 BM25）—— 与 chroma / qdrant 分支**同签名不同实现**；这条保证 v1/v2 回归的
    digest 零变化（docs/10 §2 / §5-6）。本用例只做类型与行为断言，**不**碰 ``sys.modules``
    （同进程里别的测试文件可能已经 import 过 chromadb，那样的断言会是假的）。
    """
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
    """反证「默认路径没 import 它」不是空话：装了 extra 的环境里，**显式 import 必须成功**。

    如果某个模块在本机根本没装，那「不在 ``sys.modules``」就是废话（CI 上正是如此）。
    本用例在**装了 extra 的开发机**上给出反证（``import chromadb`` 等确实可用 → 子进程
    用例抓到的「未引入」是真结果）；CI 上如实 skip（docs/10 §5-5 的诚实要求）。
    """
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
