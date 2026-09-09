"""run_rag_phase2_demo.py —— RAG Phase 2 Demo（BGE 真语义 Embedding + Qdrant 进程内向量库）。

Phase 2 验证脚本（docs/06-rag-phase2-qdrant-bge.md §4 验收 / §5 边界）：把 MVP 的
确定性 mock embedding（词面 hash，非语义）换成 ``BgeEmbedder``（BAAI/bge-small-zh-v1.5
@ fastembed/onnxruntime，dim 512）+ ``QdrantPolicyIndex/QdrantCaseIndex``（qdrant-client
进程内模式），回答 docs/06 §0 的唯一增量问题：替换缝零改动上层？词面检索不到的
同义改写 query 语义检索能否命中？BM25 / Vector / Hybrid 三路真实对比如何（不预设）。

用法::

    # 模型已缓存（离线）时直接跑：
    export HF_ENDPOINT=https://hf-mirror.com
    export PRA_RAG2_MODEL_CACHE=<fastembed 缓存根目录>     # 含 fast-*/ 或 models--*/onnx
    python scripts/run_rag_phase2_demo.py                  # 默认：三模式并排 + Part A~D
    python scripts/run_rag_phase2_demo.py --top-k 5        # 换 Top-K（默认 3）
    python scripts/run_rag_phase2_demo.py --mode bm25      # 只看单一模式
    python scripts/run_rag_phase2_demo.py --location :memory:   # qdrant 进程内（默认）
    python scripts/run_rag_phase2_demo.py --no-probe       # 跳过 Part C probe
    python scripts/run_rag_phase2_demo.py --model-cache <dir>   # 显式给缓存（缺省读 env）

    # 模型未缓存/未装依赖时优雅退出，并提示：
    #   uv sync --extra rag          （需含 fastembed / qdrant-client）
    #   export HF_ENDPOINT=https://hf-mirror.com 后首次联网下载 onnx 模型

展示内容：
- Part A：run_rag_demo 固定 query 集 × 三模式 × Top-K 并排（政策/先例各 4 条，抄自
  run_rag_demo 的验收 query 集，不 import 脚本）；
- Part B（核心叙事）：语义 vs 词面 —— 为 2 个政策条款 + 2 个先例各写一条**同义改写**
  query（与目标检索文本几乎零共享 token，用 ``pra.rag.bm25.tokenize`` 程序化验证交集），
  对比 bm25-only / vector-only / hybrid 三路下目标是否进 Top-K（进 = 命中）；
- Part C：Policy KB / Case KB 各 ~8 条人工标注 expected ids 的 probe query（覆盖
  品牌仿冒 / 规避词 / 虚假宣传 / 材质 / 类目准入 / 商家史 / 规避等主题，keyword 与
  同义改写各半），程序化算 bm25 / vector / hybrid 三路 Recall@3 —— 不预设 Hybrid 最优；
- Part D：经 PolicySearchTool / CaseSearchTool 注入 qdrant 索引跑 1 条 query，打印
  证据 type / ref_id / weight（与 MVP 同引用格式，tools 层零改动验证）。

确定性口径（docs/06 §2.2 / §5）：
- 同进程内 BGE 同输入 ``embed`` 逐位相等；跨进程/平台浮点尾差不入逐字节契约。
- qdrant 默认 ``:memory:`` 每次运行全新实例；打分/排序 tie-break 在 Python 侧
  （6 位取整 → 分降序、corpus 原序 idx 升序），不信任 qdrant 同分点顺序。
- 固定 corpus / query / probe，两次运行输出应逐行一致（脚本以两次运行 diff 自检）。
- 结论边界：单模型（bge-small-zh-v1.5）× 单语料（24/67 条）的定向演示与 probe 观测，
  非大规模评测；qdrant 进程内非分布式（远端 server 未实测）；确定性回归恒以默认
  mock 路径（backend="local"）为准。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from pra.domain.models import Budget
from pra.rag.bm25 import tokenize
from pra.rag.embedder import BgeEmbedder
from pra.rag.factory import build_case_index, build_policy_index
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

MODES = ("bm25", "vector", "hybrid")
MODE_CHOICES = MODES + ("all",)
ENV_CACHE = "PRA_RAG2_MODEL_CACHE"
CACHE_HINT = "https://hf-mirror.com"

# ---------------------------------------------------------------------------
# 固定 query / 目标 / probe 数据（全手工标注；确定性数据不随运行变化）
# ---------------------------------------------------------------------------

# Part A：与 run_rag_demo.py 的验收 query 集一致（抄写内容，不 import 脚本）。
_POLICY_QUERIES = [
    "外观高度模仿知名品牌，无授权",  # 验收 §6.1：IP 条款命中
    "标题含复刻高仿原单 仿冒来源词",
    "无依据功效夸大 增高磁疗 虚假宣传",
    "改标题重上架规避审核 商家多次",
]

_CASE_QUERIES = [
    "无品牌 + 高相似 + 商家多次上架",  # 验收 §6.2：对应先例命中
    "外观高度模仿品牌 换图规避重上架",
    "功效宣传无检测报告 虚假宣称",
    "材质标真皮实为PU 字段冲突",
]

# Part B：语义 vs 词面 定向目标（2 条款 + 2 先例）。
# 每条目标配一条**同义改写** query：改写用词刻意避开目标检索文本的高信号词，使
# bm25（词面 bigram）漏检；语义命中由 BGE 生效与否决定 —— 交集用 bm25.tokenize
# 程序化验证并打印，不依赖人工断言。
_POLICY_SEMANTIC_TARGETS = [
    {
        "clause_id": "POLICY_1.4_v1_c1",
        "label": "IP 仿冒条款 · 鞋靴外观高度模仿知名品牌（无授权）",
        "note": "改写：外形照搬大牌在售款 + 无许可文件；避开 外观/模仿/品牌/授权/鞋靴 等词",
        "paraphrase": "一家网店卖的运动鞋，外形跟某外国名牌当季主打的货几乎分不出来，可店主拿不出那家公司的合作文件",
    },
    {
        "clause_id": "POLICY_4.2_v2_c1",
        "label": "规避条款 · 改标题/描述规避重上架（≥3 次系统性）",
        "note": "改写：被撤下 → 换文字说明 → 反复放回；避开 下架/标题/修改/上架/规避 等词",
        "paraphrase": "卖家把被平台撤掉的货品换一下文字说明又放回店里，而且翻来覆去干过好几轮",
    },
]

_CASE_SEMANTIC_TARGETS = [
    {
        "case_id": "RAG_CASE_0001",
        "label": "高仿先例 · 无标高仿女跑鞋 + 商家多次下架/改名重上架",
        "note": "改写：跑步鞋无标牌 + 同一模具 + 多次被清理/改售卖名；仅余 2 个低信号 bigram 交集（的女/经典）",
        "paraphrase": "这家铺子卖的女式跑步鞋不挂任何标牌，外形却跟某大厂久经市场的经典鞋款几乎出自同一模具，近三个多月已挨过五回清理，还三度更换售卖名称重新摆出",
    },
    {
        "case_id": "RAG_CASE_0011",
        "label": "冒用品牌徽记先例 · 箱包检出奢侈品牌双 G 字样 logo",
        "note": "改写：豪奢品牌双弧线字符徽记 + 无受权分号/许可文书；仅余 1 个低信号 bigram 交集（品牌）",
        "paraphrase": "一只挎包表面识别出某豪奢品牌那种双弧线字符叠印的徽记，店方既不是受权的分号，也交不出许可文书，这是冒用他人标识",
    },
]

# Part C：probe 三路 Recall@3 —— 每条 query 人工标注应命中（expected）1~2 个 id。
# tag: kw=词面友好（BM25 应命中）；para=同义改写（需语义；BM25 大概率漏）—— 混编避免
# 单一口径给 Hybrid 注水，最终数字如实呈现不预设。
_POLICY_PROBES = [
    {
        "query": "标题带“原单”“复刻”“高仿”“A货”等暗示仿冒来源的词",
        "expected": ["POLICY_1.2_v1_c1"],
        "tag": "kw",
        "topic": "规避词",
    },
    {
        "query": "详情页印着他人注册商标图形，未经授权使用",
        "expected": ["POLICY_1.3_v1_c1"],
        "tag": "kw",
        "topic": "品牌仿冒(logo)",
    },
    {
        "query": "鞋类详情标“头层牛皮”，须提供材质质检凭证",
        "expected": ["POLICY_3.1_v1_c1"],
        "tag": "kw",
        "topic": "材质标识",
    },
    {
        "query": "商家宣称增高磁疗改善循环，却拿不出检测报告",
        "expected": ["POLICY_2.1_v2_c1"],
        "tag": "kw",
        "topic": "虚假宣传",
    },
    {
        "query": "卫衣宣称抑菌90天、自发热，须有对应检测报告",
        "expected": ["POLICY_2.2_v1_c1"],
        "tag": "kw",
        "topic": "虚假宣传(服饰功能)",
    },
    {
        "query": "一家网店卖的运动鞋，外形跟某外国名牌当季主打的货几乎分不出来，可店主拿不出那家公司的合作文件",
        "expected": ["POLICY_1.4_v1_c1"],
        "tag": "para",
        "topic": "品牌仿冒(外观)",
    },
    {
        "query": "卖家把被平台撤掉的货品换一下文字说明又放回店里，而且翻来覆去干过好几轮",
        "expected": ["POLICY_4.2_v2_c1"],
        "tag": "para",
        "topic": "规避(改名重上架)",
    },
    {
        "query": "一双再普通不过的鞋，店家却把它摆进专门收纳能让人长高的那种鞋的区域，为了蹭些本不该有的关注",
        "expected": ["POLICY_3.5_v1_c1"],
        "tag": "para",
        "topic": "类目准入(错挂)",
    },
]

_CASE_PROBES = [
    {
        "query": "卫衣印花把品牌图形镜像翻转后规避商标识别",
        "expected": ["RAG_CASE_0012"],
        "tag": "kw",
        "topic": "品牌仿冒(logo 规避)",
    },
    {
        "query": "标题含“复刻”“同款”等仿冒词，且多次改标题重上架要从重",
        "expected": ["RAG_CASE_0010"],
        "tag": "kw",
        "topic": "规避词+从重",
    },
    {
        "query": "同一违规托特包拆成三个SKU多链接铺货规避处罚",
        "expected": ["RAG_CASE_0023"],
        "tag": "kw",
        "topic": "规避(SKU 拆分)",
    },
    {
        "query": "鞋的页面标“头层牛皮”，实检是PU合成革，以次充好",
        "expected": ["RAG_CASE_0018"],
        "tag": "kw",
        "topic": "材质冲突",
    },
    {
        "query": "这家铺子卖的女式跑步鞋不挂任何标牌，外形却跟某大厂久经市场的经典鞋款几乎出自同一模具，近三个多月已挨过五回清理，还三度更换售卖名称重新摆出",
        "expected": ["RAG_CASE_0001"],
        "tag": "para",
        "topic": "高仿+商家史",
    },
    {
        "query": "一只挎包表面识别出某豪奢品牌那种双弧线字符叠印的徽记，店方既不是受权的分号，也交不出许可文书",
        "expected": ["RAG_CASE_0011"],
        "tag": "para",
        "topic": "冒用品牌徽记",
    },
    {
        "query": "外套挂着的小牌写明整件都用天然植物纤维纺成，送去化验却发现棉的成分刚过半，剩下的都是人造纤维",
        "expected": ["RAG_CASE_0019"],
        "tag": "para",
        "topic": "面料成分不符",
    },
    {
        "query": "一双平常的鞋被店家摆进专门卖那种能让人变高的鞋的货区，白得了展示位置，货品本身没别的毛病",
        "expected": ["RAG_CASE_0024"],
        "tag": "para",
        "topic": "类目错挂",
    },
]

# Part D 证据引用演示用的 query（与 MVP run_rag_demo 同 query，验证引用格式零改动）。
_EVIDENCE_POLICY_QUERY = "外观高度模仿知名品牌，无授权"
_EVIDENCE_CASE_QUERY = "无品牌 + 高相似 + 商家多次上架"


# ---------------------------------------------------------------------------
# 打印/排版 helpers
# ---------------------------------------------------------------------------


def _clip(text: str, n: int = 46) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _clip_long(text: str, n: int = 84) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _id_of(hit) -> str:
    """policy 命中取 clause_id；case 命中取 case_id（两类 hit 的 id 属性名不同）。"""
    return hit.clause_id if hasattr(hit, "clause_id") else hit.case_id


def _mode_tag(mode: str) -> str:
    return f"[{mode:<7}]"


# ---------------------------------------------------------------------------
# 索引装配（共享 embedder；每 (KB, mode) 一个实例，构造期一次建 doc 向量）
# ---------------------------------------------------------------------------


class _IndexPool:
    """缓存 (kind, mode) → qdrant 索引；全部索引共享同一个 BgeEmbedder。"""

    def __init__(self, embedder: BgeEmbedder, location: str) -> None:
        self._embedder = embedder
        self._location = location
        self._pool: dict[tuple[str, str], object] = {}

    def get(self, kind: str, mode: str) -> object:
        key = (kind, mode)
        if key not in self._pool:
            build = build_policy_index if kind == "policy" else build_case_index
            self._pool[key] = build(
                embedder=self._embedder,
                mode=mode,  # type: ignore[arg-type]
                backend="qdrant",
                location=self._location,
            )
        return self._pool[key]


async def _search(index: object, kind: str, query: str, top_k: int) -> list:
    """统一检索入口：policy 只查生效条款、无元数据过滤；case 无过滤。"""
    if kind == "policy":
        return await index.search(  # type: ignore[union-attr]
            query, PolicySearchFilters(), top_k=top_k, effective_only=True
        )
    return await index.search(query, CaseSearchFilters(), top_k=top_k)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Part A / B / C / D
# ---------------------------------------------------------------------------


async def _part_a(pool: _IndexPool, kind: str, queries: list[str], top_k: int, modes) -> None:
    if kind == "policy":
        print(f"\n▶ Part A 政策检索（固定 query 集 × {len(modes)} 模式 × Top-{top_k}）")
    else:
        print(f"\n▶ Part A 先例检索（固定 query 集 × {len(modes)} 模式 × Top-{top_k}）")
    if kind == "policy":
        print("  （注：PolicyClauseHit 契约不含 score 字段 —— 与 run_rag_demo 同口径，政策行不打印分；"
              "case 的 similarity 即检索最终分）")
    for query in queries:
        print(f"    query = {query}")
        for mode in modes:
            hits = await _search(pool.get(kind, mode), kind, query, top_k)
            cells = []
            for h in hits:
                if kind == "policy":
                    tags = "/".join(t.value for t in h.risk_type) or "-"
                    cells.append(
                        f"{h.clause_id}[{tags}]《{h.title}》{_clip(h.text)}"
                    )
                else:
                    tags = "/".join(t.value for t in h.risk_type) or "-"
                    cells.append(
                        f"{h.case_id} sim={h.similarity:.3f} {h.decision.value}/{h.risk_level.value}[{tags}] {_clip(h.summary, 40)}"
                    )
            line = " | ".join(cells) if cells else "(无命中)"
            print(f"    {_mode_tag(mode)} " + line)


def _target_text(kind: str, target_id: str) -> str:
    """目标检索文本（与索引构造同口径：policy=title。text；case=summary）。"""
    if kind == "policy":
        from pra.rag.corpus import load_policies

        rows, _ = load_policies()
        row = next(r for r in rows if r.clause_id == target_id)
        return f"{row.title}。{row.text}"
    from pra.rag.corpus import load_cases

    rows, _ = load_cases()
    row = next(r for r in rows if r.case_id == target_id)
    return row.summary


async def _part_b(
    pool: _IndexPool, targets, kind: str, top_k: int, modes
) -> dict[str, int]:
    """语义 vs 词面：同义改写 query 下目标在 bm25/vector/hybrid 的 Top-K 命中情况。

    返回 {"both_hit": 满足 bm25 漏 & vector 中的目标数, "total": 目标数}。
    """
    kb_name = "Policy" if kind == "policy" else "Case"
    plural = "条款" if kind == "policy" else "先例"
    print(f"\n▶ Part B 语义 vs 词面（{kb_name} KB {len(targets)} 个目标{plural}）")
    print("  判定口径：目标进该模式 Top-K = 命中；bm25(词面 bigram) 漏 + vector(语义) 中 = 语义检索生效。")
    both_hit = 0
    for t in targets:
        target_id = t.get("clause_id") or t["case_id"]
        q = t["paraphrase"]
        text = _target_text(kind, target_id)
        inter = sorted(set(tokenize(q)) & set(tokenize(text)))
        print(f"\n  ◇ 目标 {target_id}  {t['label']}")
        print(f"    目标检索文本: {_clip_long(text, 150)}")
        print(f"    同义改写 query: {q}")
        print(f"    改写意图: {t['note']}")
        print(f"    token 交集（改写 ∩ 目标检索文本, bm25.tokenize 程序化验证）= {len(inter)} 个 {inter if inter else '[]（零共享关键词）'}")
        positions: dict[str, int | None] = {}
        top_ids: dict[str, list[str]] = {}
        for mode in modes:
            hits = await _search(pool.get(kind, mode), kind, q, top_k)
            ids = [_id_of(h) for h in hits]
            pos = ids.index(target_id) + 1 if target_id in ids else None
            positions[mode] = pos
            top_ids[mode] = ids
        for mode in modes:
            tag = "命中" if positions[mode] is not None else "漏检"
            pos_txt = f"@{positions[mode]}" if positions[mode] is not None else "(不在 Top-K)"
            ids_txt = ", ".join(top_ids[mode]) if top_ids[mode] else "(无)"
            print(f"    {_mode_tag(mode)} Top-{top_k}: {ids_txt}")
            print(f"      → 目标 {tag} {pos_txt}")
        bm25_miss = positions.get("bm25") is None
        vec_hit = positions.get("vector") is not None
        if bm25_miss and vec_hit:
            both_hit += 1
            verdict = "【语义检索生效：bm25 漏检 → vector 命中】"
        else:
            verdict = "（未满足 bm25 漏 + vector 中 —— 如实记录）"
        print(f"    ✓ 结论: {verdict}")
    return {"both_hit": both_hit, "total": len(targets)}


async def _part_c(
    pool: _IndexPool, probes, kind: str, top_k: int, modes
) -> None:
    """probe 三路 Recall@3（item-level：Σ_q |top3(q) ∩ expected(q)| / Σ_q |expected(q)|）。"""
    kb_name = "Policy" if kind == "policy" else "Case"
    print(f"\n▶ Part C probe 三路 Recall@3（{kb_name} KB · {len(probes)} 条 query × 三模式）")
    print("  Recall@3 = Σ |Top-K ∩ expected| / Σ |expected|（item-level；expected 为人工标注应命中 id）")
    total_exp = sum(len(p["expected"]) for p in probes)
    retrieved: dict[str, int] = {m: 0 for m in modes}
    for i, p in enumerate(probes, 1):
        exp = p["expected"]
        cells = []
        pos_info: dict[str, list[int]] = {}
        for mode in modes:
            hits = await _search(pool.get(kind, mode), kind, p["query"], top_k)
            ids = [_id_of(h) for h in hits]
            poss = [ids.index(e) + 1 for e in exp if e in ids]
            retrieved[mode] += len(poss)
            pos_info[mode] = poss
            cells.append(f"{mode}:{len(poss)}/{len(exp)}@" + (f"{poss}" if poss else "-"))
        print(f"  [{i:>2}/{len(probes)}] [{p['tag']:<4}|{p['topic']}] {p['query']}")
        print(f"        exp={exp}   " + "  |  ".join(cells))
    print(f"  ---- {kb_name} KB Recall@{top_k}（expected 合计 {total_exp}） ----")
    for mode in modes:
        pct = 100.0 * retrieved[mode] / total_exp if total_exp else 0.0
        print(f"    {_mode_tag(mode)} Recall@{top_k} = {retrieved[mode]}/{total_exp} ({pct:.1f}%)")
    # 如实解读（不预设）：打印数字间的相对关系
    vals = {m: retrieved[m] for m in modes}
    print(f"  → 解读（如实，不预设）: "
          f"bm25={vals['bm25']} vector={vals['vector']} hybrid={vals['hybrid']} "
          f"（hybrid {'≥ 单路双方' if vals['hybrid'] >= max(vals['bm25'], vals['vector']) else '未同时优于单路'}；"
          f"N={total_exp} 个标注项，小样本仅作定向观测）")


async def _part_d(pool: _IndexPool) -> None:
    """真实 Tool 注入 qdrant 索引 → Evidence 引用（与 MVP 同格式，验证 tools 层零改动）。"""
    print("\n▶ Part D 证据引用（PolicySearchTool / CaseSearchTool 注入 qdrant 索引 → Evidence）")
    ctx = ToolContext(run_id="rag-phase2-demo", case_id="DEMO_RAG_PH2_0001", budget=Budget())
    policy_tool = PolicySearchTool(index=pool.get("policy", "hybrid"))  # type: ignore[arg-type]
    p_res = await policy_tool.call(
        PolicySearchArgs(query=_EVIDENCE_POLICY_QUERY, top_k=3), ctx
    )
    print(f"  PolicySearchTool  query={_EVIDENCE_POLICY_QUERY} → Evidence:")
    for ev in policy_tool.to_evidence(p_res):
        print(f"    type={ev.type:<12} weight={ev.weight} ref_id={ev.ref_id} | {_clip(ev.value, 72)}")
    case_tool = CaseSearchTool(index=pool.get("case", "hybrid"))  # type: ignore[arg-type]
    c_res = await case_tool.call(CaseSearchArgs(query=_EVIDENCE_CASE_QUERY, top_k=3), ctx)
    print(f"  CaseSearchTool  query={_EVIDENCE_CASE_QUERY} → Evidence:")
    for ev in case_tool.to_evidence(c_res):
        print(f"    type={ev.type:<12} weight={ev.weight:.3f} ref_id={ev.ref_id} | {_clip(ev.value, 72)}")


def _print_boundaries() -> None:
    print("\n▶ 结论边界（docs/06-rag-phase2-qdrant-bge.md §5 口径，如实标注勿当能力承诺）")
    print("  - 语义质量 = 单模型（BAAI/bge-small-zh-v1.5）× 单语料（Policy 24 条 / Case 67 条）的")
    print("    定向演示与 probe 观测，非大规模评测；换模型/语料结果会变。")
    print("  - Qdrant 进程内模式（:memory:）= 真 Qdrant API 但非分布式部署；远端 server 未实测")
    print("    （代码路径同一，仅连接串差异）。")
    print("  - 跨进程/平台 embedding 浮点尾差不入逐字节契约；确定性回归恒以默认 mock 路径为准。")
    print("  - hybrid = 0.5·norm(bm25) + 0.5·vector（默认权重，可配）；不预设 hybrid 更优。")
    print("  - PolicyClauseHit 契约不含 score（政策行不打印分）；CaseHit.similarity = 检索最终分。")


def _resolve_model_cache(model_cache: str | None) -> str:
    """解析模型缓存目录：--model-cache > env PRA_RAG2_MODEL_CACHE；都没有 → 报错退出。"""
    if model_cache:
        return model_cache
    env = os.environ.get(ENV_CACHE)
    if env:
        return env
    print(
        "[FAIL] 未找到 BGE 模型缓存目录：请用 --model-cache <dir> 或导出环境变量 "
        f"{ENV_CACHE}=<dir>（fastembed 缓存根目录，应含 fast-*/ 或 models--*/ 下的 onnx 文件）",
        file=sys.stderr,
    )
    sys.exit(2)


def _ensure_model_ready(cache_dir: str) -> BgeEmbedder:
    """模型就绪预检：fastembed 可 import + 磁盘缓存命中。未就绪 → 带指引退出（绝不静默回退 mock）。"""
    embedder = BgeEmbedder(cache_dir=cache_dir)
    if not embedder.available():
        print(
            "[FAIL] fastembed 不可用（BgeEmbedder.available()=False）。请先安装依赖: "
            "`uv sync --extra rag`（需含 fastembed/onnxruntime）；默认本地检索不依赖它。",
            file=sys.stderr,
        )
        sys.exit(2)
    if not embedder.model_ready():
        print(
            "[FAIL] 语义模型尚未缓存，无法离线运行。请先联网下载一次（~90MB，huggingface.co 被墙时设镜像）：\n"
            f"  1) export HF_ENDPOINT={CACHE_HINT}\n"
            f"  2) export {ENV_CACHE}={cache_dir}\n"
            "  3) 跑一次会触发下载的调用（如本脚本任一 BGE 路径，或 fastembed TextEmbedding 实例化）\n"
            "下载完成后本脚本即纯离线（local_files_only）。\n"
            "注意：本 provider 失败只报错/退出，绝不静默回退 MockHashEmbedder（mock 与真模型维度/语义不可混算）。",
            file=sys.stderr,
        )
        sys.exit(2)
    return embedder


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG Phase 2 Demo：BGE 真语义 Embedding + Qdrant 进程内向量库（docs/06）"
    )
    parser.add_argument("--top-k", type=int, default=3, help="每路 Top-K（默认 3）")
    parser.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        default="all",
        help="检索模式：默认 all 跑 bm25/vector/hybrid 三模式并排；可单选",
    )
    parser.add_argument(
        "--location",
        default=":memory:",
        help="qdrant 进程内 location：:memory:（默认）/ path=<目录> / http(s)://远端",
    )
    parser.add_argument(
        "--model-cache",
        default=None,
        help=f"fastembed 模型缓存根目录（默认读环境变量 {ENV_CACHE}）",
    )
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="跳过 Part C probe 三路 Recall@3",
    )
    return parser.parse_args(argv)


def _resolve_modes(mode: str) -> list[str]:
    return list(MODES) if mode == "all" else [mode]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cache_dir = _resolve_model_cache(args.model_cache)
    embedder = _ensure_model_ready(cache_dir)
    modes = _resolve_modes(args.mode)
    top_k = args.top_k

    print(f"=== RAG Phase 2 Demo（BGE 语义 + Qdrant {args.location}）===")
    print(f"model cache = {cache_dir}  | embedder = BgeEmbedder({embedder.model_name}, dim={embedder.dim})")
    print(f"modes = {modes} | top_k = {top_k} | policy/case 权重 = 0.5/0.5")
    print("模型就绪（离线 local_files_only）→ 开始装配索引（构造期全量 embed corpus）…")

    pool = _IndexPool(embedder, args.location)

    # Part A：固定 query 集三模式并排
    await _part_a(pool, "policy", _POLICY_QUERIES, top_k, modes)
    await _part_a(pool, "case", _CASE_QUERIES, top_k, modes)

    # Part B：语义 vs 词面
    p_stat = await _part_b(pool, _POLICY_SEMANTIC_TARGETS, "policy", top_k, modes)
    c_stat = await _part_b(pool, _CASE_SEMANTIC_TARGETS, "case", top_k, modes)
    print("\n▶ Part B 汇总（bm25 漏检 & vector 命中 的目标数）")
    print(f"  Policy 目标: {p_stat['both_hit']}/{p_stat['total']}；Case 目标: {c_stat['both_hit']}/{c_stat['total']}")
    if p_stat["both_hit"] >= 1 and c_stat["both_hit"] >= 1:
        print("  ✓ 语义命中成立：至少各 1 个 Policy 条款 + Case 先例为「bm25 词面漏检、vector 语义命中」。")
    else:
        print("  ! 未同时满足 Policy 与 Case 的「bm25 漏 + vector 中」—— 如实记录（同义改写不彻底或语义不足）。")

    # Part C：probe 三路 Recall@3
    if not args.no_probe:
        await _part_c(pool, _POLICY_PROBES, "policy", top_k, modes)
        await _part_c(pool, _CASE_PROBES, "case", top_k, modes)
    else:
        print("\n[skip] Part C probe 已按 --no-probe 跳过")

    # Part D：证据引用演示
    await _part_d(pool)

    # 结论边界
    _print_boundaries()
    print("\n[OK] run_rag_phase2_demo 完成（BGE 真语义 + Qdrant 进程内；输出可重放——同输入两次运行应逐行一致）")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_main(argv))
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[FAIL] run_rag_phase2_demo 运行失败: {exc!r}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
