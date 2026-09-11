"""RAG 世界评测入口：InMemory vs RAG + BM25/Vector/Hybrid 三路。

评测默认仍 InMemory（回归不破坏）。对同一 eval_data 分别以 ``tool_world="eval"``（InMemory）
与 ``tool_world="rag"`` + mode ∈ {bm25, vector, hybrid}（``--modes``）跑 ``agent`` 方案，输出
各世界/模式的决策指标与决策序列 digest、与 InMemory 的逐案决策差异、三检索模式并排对比
（如实呈现，不预设 Hybrid 优于单路），以及运行时证据级隔离抽查：RAG 世界引用的先例 ref_id
必须全部是 ``RAG_CASE_*``，不得引用 eval GT / InMemory 种子先例。

``--probe`` 追加检索层报告（不跑 LLM/agent）：人工标注 probe 集在三模式下的 ``Recall@K``
（默认 K=3，``--probe-top-k`` 可改），Policy KB 与 Case KB 并排；``--probe-only`` 只跑该报告。
``--backend {local,chroma,qdrant}``（缺省 local，缺省路径输出与改动前逐字节一致）同时作用于
agent A/B 的 RAG 臂与 probe 报告；``--smoke`` / ``--smoke-limit`` 跑确定性子集。

A/B 隔离（``--chroma-client``，默认 ``ephemeral``）：chroma 臂缺省用
``chromadb.EphemeralClient()``（进程内内存库），每个索引实例都是全新的库、随进程消失，各臂
之间零状态传递，也不要求本机起 Chroma 服务端；只有 ``--chroma-client http`` 才连本机服务端
（127.0.0.1:8001），那时用独占前缀建库并在结束前删除（服务端是共享单实例）。两种客户端实测
结果逐字节一致（probe 报告 diff 为空、vector parity 分差 0.000e+00），但不是同一份存储。

分数口径（量纲不可比，报告不做归一化）：``CaseHit.retrieval_score`` 是检索分 —— local 后端
hybrid 是 [0,1] 加权融合分，chroma 后端 hybrid 是 RRF 分 ``Σ 1/(60 + rank)``（k=60，rank 从
0 起 → 上界 ``2/60 = 1/30 ≈ 0.0333``，实测观测区间约 0.0275~0.0333）；RRF 分是排名融合分，
**不是相似度、不是概率**，禁止跨模式/跨后端比大小。

全链路确定性：无真 LLM / 无 API key / 无 LLM 调用；``chroma`` 后端只在 ``http`` 模式下连
本机服务端。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Protocol

from pra.evaluation.harness.base import EvalContext
from pra.evaluation.runner import EvaluationRunner

MODES = ("bm25", "vector", "hybrid")
BACKENDS = ("local", "chroma", "qdrant")
#: chroma 臂的客户端选择：ephemeral = 进程内内存库（默认），http = 本机服务端。
#: 默认选 ephemeral 的理由 = A/B 隔离（见模块 docstring）。
CHROMA_CLIENTS = ("ephemeral", "http")
CHROMA_DEFAULT_PORT = 8001  # 与 pra.rag.chroma_backend 的服务端默认端口一致（宿主机侧）
DEFAULT_DATA = "eval_data/v1/cases_v1.jsonl"
DIFF_HEAD = 12  # 差异明细打印条数上限
PROBE_TOP_K = 3  # probe Recall@K 的 K（与 phase2 demo 同口径）
PROBE_SOURCE = "scripts/run_rag_phase2_demo.py"  # probe 集来源（复用，不新造）
PROBE_COLLECTION_PREFIX = "pra_eval_probe"  # chroma probe 独占 collection 前缀


def _decision_digest(records) -> str:
    seq = [r.decision for r in records]
    return hashlib.sha256(
        "|".join(seq).encode("utf-8")
    ).hexdigest()[:16]


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.3f}"


def _metrics_row(label: str, result, records) -> str:
    m = result.overall["agent"]
    cost = result.cost_summary.get("agent") or {}
    return (
        f"{label:<16} acc={_fmt(m.accuracy)} prec={_fmt(m.precision)} "
        f"recall={_fmt(m.recall)} fpr={_fmt(m.fpr)} fnr={_fmt(m.fnr)} "
        f"hmr={_fmt(m.human_rate)} auto={_fmt(m.automation)} "
        f"tool={cost.get('tool_calls', 0):.2f} digest={_decision_digest(records)}"
    )


def _transition_label(r) -> str:
    extra = ""
    if r.detail.get("overrides"):
        extra = f"[{','.join(r.detail['overrides'])}]"
    return f"{r.decision}{extra}"


async def _run_ab(configs: list[tuple[str, dict]], args: argparse.Namespace) -> dict[str, tuple]:
    """跑 InMemory(eval) vs 各 (mode, backend) 的 RAG 臂，返回 label → (result, records, ctx)。"""
    results: dict[str, tuple] = {}
    for label, overrides in configs:
        ctx = EvalContext(**overrides)
        runner = EvaluationRunner(data_path=args.data, ctx=ctx)
        result = await runner.run(
            include=("agent",), smoke=args.smoke, smoke_limit=args.smoke_limit
        )
        records = result.records["agent"]
        results[label] = (result, records, ctx)
        print(_metrics_row(label, result, records))
    return results


# ---------------------------------------------------------------------------
# 人工标注 probe 集（复用 run_rag_phase2_demo.py 的 Part C，不新造 probe）
# ---------------------------------------------------------------------------


class _ProbeSource(Protocol):
    """``scripts/run_rag_phase2_demo.py`` 里被复用的两个模块级常量（结构约定）。"""

    _POLICY_PROBES: list[dict]
    _CASE_PROBES: list[dict]


def _load_probe_source() -> _ProbeSource:
    """按路径加载 ``scripts/run_rag_phase2_demo.py``，只取它的人工标注 probe 集。

    按路径 importlib 加载（两脚本同级且仓库无 ``scripts`` 包，这样不污染 ``sys.path``、
    不依赖当前工作目录）。只读模块级常量，不执行其 ``main``；该模块 import 期无副作用。
    """
    path = Path(__file__).resolve().parent / "run_rag_phase2_demo.py"
    spec = importlib.util.spec_from_file_location("_pra_probe_source", path)
    if spec is None or spec.loader is None:  # pragma: no cover —— 文件缺失即报错
        raise RuntimeError(f"无法加载 probe 来源脚本: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def _search(index, kind: str, query: str, top_k: int) -> list:
    """统一检索入口（与 tools 层契约同形）：policy 只查生效条款，case 无过滤。"""
    from pra.tools.case_search.tool import CaseSearchFilters
    from pra.tools.policy_search.tool import PolicySearchFilters

    if kind == "policy":
        return await index.search(query, PolicySearchFilters(), top_k, True)
    return await index.search(query, CaseSearchFilters(), top_k)


def _id_of(hit) -> str:
    """policy 命中取 ``clause_id``；case 命中取 ``case_id``（两类 hit 属性名不同）。"""
    return hit.clause_id if hasattr(hit, "clause_id") else hit.case_id


def _score_of(hit) -> float | None:
    """命中分：case 有 ``retrieval_score``；policy 契约**不含分**（返回 None）。"""
    return getattr(hit, "retrieval_score", None)


def _build_probe_index(backend: str, kind: str, mode: str, options: dict):
    """按 (backend, kind, mode) 装配一个索引（每 combo 独立实例；缺省 = MockHashEmbedder）。"""
    from pra.rag.factory import build_case_index, build_policy_index

    build = build_policy_index if kind == "policy" else build_case_index
    if backend == "local":
        return build(mode=mode, **options)
    return build(mode=mode, backend=backend, **options)


def _chroma_options(args: argparse.Namespace, prefix: str) -> dict:
    """chroma 臂的装配参数（缺省 EphemeralClient，见模块 docstring 「A/B 隔离」段）。

    ``collection_prefix`` 两种模式都传：node id 由 ``hash(collection + row_key)`` 决定，
    固定前缀 → 两次运行 id 稳定（确定性契约）。
    """
    if args.chroma_client == "http":
        return {"collection_prefix": prefix}
    return {"collection_prefix": prefix, "chroma_ephemeral": True}


def _chroma_client_label(args: argparse.Namespace) -> str:
    """报告里如实标注本次 chroma 臂用的客户端（ephemeral 不得被读成「真服务端」）。"""
    if args.chroma_client == "http":
        return f"HttpClient(127.0.0.1:{CHROMA_DEFAULT_PORT} 服务端)"
    return "EphemeralClient(进程内内存库，随进程消失)"


async def _probe_report(
    backend: str, modes: list[str], top_k: int, *, chroma_options: dict,
    client_label: str = "", probe_only: bool = False,
) -> None:
    """三模式 × 两 KB 的 probe Recall@K 并排报告（不预设任何模式最优）。

    只打印实测命中数与分数量纲；不同模式/后端的分数不可比。
    """
    from pra.rag.chroma_backend import served_counters

    probes_by_kind = (
        ("policy", _load_probe_source()._POLICY_PROBES),
        ("case", _load_probe_source()._CASE_PROBES),
    )
    options: dict = dict(chroma_options) if backend == "chroma" else {}

    print("\n" + "=" * 100)
    print(f"人工标注 probe 集 · 三模式 Recall@{top_k} 并排（backend={backend}；不预设任何模式最优）")
    print("=" * 100)
    print(f"probe 来源（**复用，未新造**）: {PROBE_SOURCE} —— Part C 的 _POLICY_PROBES / _CASE_PROBES")
    print("embedder = MockHashEmbedder（确定性、离线；**不是语义模型** —— 词面特征 hash）")
    if backend == "chroma":
        print(f"chroma 客户端 = {client_label}；collection 前缀 = {options.get('collection_prefix')}")

    for kind, probes in probes_by_kind:
        kb = "Policy KB" if kind == "policy" else "Case KB"
        corpus_size = 24 if kind == "policy" else 67
        n_kw = sum(1 for p in probes if p["tag"] == "kw")
        n_para = sum(1 for p in probes if p["tag"] == "para")
        total_exp = sum(len(p["expected"]) for p in probes)
        print("\n" + "-" * 100)
        print(
            f"{kb}（corpus {corpus_size} 条 / probe {len(probes)} 条 = kw {n_kw} + para {n_para} / "
            f"expected 合计 {total_exp} 项）"
        )
        print("  Recall@K = Σ_q |Top-K ∩ expected(q)| / Σ_q |expected(q)|（item-level）")
        retrieved: dict[str, int] = {m: 0 for m in modes}
        score_lo: dict[str, float | None] = {m: None for m in modes}
        score_hi: dict[str, float | None] = {m: None for m in modes}
        for i, p in enumerate(probes, 1):
            cells = []
            for mode in modes:
                index = _build_probe_index(backend, kind, mode, options)
                hits = await _search(index, kind, p["query"], top_k)
                ids = [_id_of(h) for h in hits]
                poss = [ids.index(e) + 1 for e in p["expected"] if e in ids]
                retrieved[mode] += len(poss)
                for h in hits:
                    s = _score_of(h)
                    if s is None:
                        continue
                    score_lo[mode] = s if score_lo[mode] is None else min(score_lo[mode], s)
                    score_hi[mode] = s if score_hi[mode] is None else max(score_hi[mode], s)
                cells.append(
                    f"{mode}={len(poss)}/{len(p['expected'])}@"
                    + (f"{poss}" if poss else "-")
                )
            print(f"  [{i:>2}/{len(probes)}] [{p['tag']:<4}|{p['topic']}] {p['query']}")
            print(f"        exp={p['expected']}  " + "  ".join(cells))
        print(f"  ---- {kb} Recall@{top_k}（expected 合计 {total_exp}） ----")
        for mode in modes:
            hit_n = retrieved[mode]
            pct = 100.0 * hit_n / total_exp if total_exp else 0.0
            if score_lo[mode] is None:
                scale = "分数：policy 契约不含分（不打印）"
            else:
                scale = f"分数区间 [{score_lo[mode]:.6f}, {score_hi[mode]:.6f}]"
            print(f"    {mode:<7} Recall@{top_k} = {hit_n}/{total_exp} ({pct:5.1f}%)   {scale}")
        print(
            "  → 如实记录（**不排序、不判定最优**）: "
            + "  ".join(f"{m}={retrieved[m]}/{total_exp}" for m in modes)
            + f"  （probe {len(probes)} 条，小样本定向观测）"
        )

    # --- 分数口径（量纲差异必须同框声明；不归一化、不当相似度） ----------
    print("\n" + "-" * 100)
    print("分数口径（**量纲不可比 → 本报告不做任何归一化**）")
    print("  - local 后端：hybrid = 0.5·norm(bm25) + 0.5·cos，量纲 [0,1]（加权融合分）；")
    print("    bm25/vector 两列也是各自归一化分 —— 与 chroma 的分数**不是同一把尺**。")
    print("  - chroma 后端：hybrid = **RRF 融合分** Σ 1/(60 + rank)（k=60），rank 从 0 起 → 上界 2/60 ≈ 0.0333；")
    print("    vector = 1 − Chroma cosine distance（余弦相似度）；bm25 = 候选集内 min-max 归一化 bm25s 分。")
    print("  - ⚠️ RRF 分是**排名融合分，不是相似度、不是概率**；禁止跨模式/跨后端比大小。")
    print("  - CaseHit.retrieval_score = 检索分；PolicyClauseHit 契约不含分。")

    # --- 运行期自检（确定性证据 + 零 LLM + collection 清理） --------------------
    print("\n" + "-" * 100)
    print("probe 运行期自检")
    print(f"  - backend={backend} modes={list(modes)} top_k={top_k} probe_only={probe_only}")
    print(f"  - 本进程 chroma_backend 发出的 LLM 调用数（须为 0）: {served_counters()['llm_calls']}")
    if backend == "chroma" and chroma_options.get("chroma_ephemeral"):
        # EphemeralClient = 进程内内存库：**无需清理**（随进程消失），也不该去连服务端。
        print("  - chroma 客户端 = EphemeralClient（进程内）→ 无需清理服务端；本进程结束即释放")
    elif backend == "chroma":
        _cleanup_chroma_collections(chroma_options["collection_prefix"], label="probe")


def _cleanup_chroma_collections(prefix: str, *, label: str) -> None:
    """删除本脚本用 ``prefix`` 在**服务端**建的 collection（两个 KB × dim 256），并回读服务端列表。

    仅 ``--chroma-client http`` 需要：服务端是共享单实例，脚本只用独占前缀建库、用完即删，
    绝不触碰别人的 collection。``ephemeral`` 模式不调本函数 —— 内存库随进程消失，也不应为
    清理而去连服务端（那会重新引入 Docker 依赖）。
    """
    from pra.rag.chroma_backend import delete_collection, make_chroma_client

    for name in (f"{prefix}_policy_256", f"{prefix}_case_256"):
        deleted = delete_collection(name)
        print(f"  - [{label}] 清理 collection {name}: {'已删除' if deleted else '不存在'}")
    names = [c.name for c in make_chroma_client().list_collections()]
    print(f"  - [{label}] 清理后服务端 collection 列表: {names}")


async def _main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    modes = list(args.modes)
    backend = args.backend
    # 独占 collection 前缀（仅 chroma 用）：同一进程内 A/B 与 probe 共用，只建一套库。
    prefix = f"{PROBE_COLLECTION_PREFIX}_{os.getpid()}"
    chroma_options = _chroma_options(args, prefix)
    client_label = _chroma_client_label(args)

    print("=" * 100)
    print("商品审核 Agent · RAG 世界 vs InMemory（agent 方案 · 决策序列 digest + 指标）")
    print("=" * 100)
    print(f"数据集: {args.data}（smoke={args.smoke}）| 世界: eval(InMemory) + rag×{len(modes)} 模式")
    if backend != "local":
        # 缺省（local）不打印这一行：缺省路径输出与改动前逐字节一致。
        print(f"backend: {backend}（RAG 索引装配；chroma = ChromaDB + LlamaIndex + BGE + BM25 + RRF）")
    if backend == "chroma":
        print(f"chroma 客户端 = {client_label}；collection 前缀 = {prefix}")

    if not args.probe_only:
        configs: list[tuple[str, dict]] = [("InMemory(eval)", {})]
        rag_overrides: dict = {"tool_world": "rag", "rag_backend": backend}
        if backend == "chroma":
            # A/B 隔离：每臂独立装配（ephemeral 每实例一个全新内存库 → 零状态传递）。
            rag_overrides["rag_backend_options"] = chroma_options
        configs += [
            (f"RAG-{mode}", {**rag_overrides, "rag_mode": mode}) for mode in modes
        ]
        results = await _run_ab(configs, args)
        _report_ab(results, modes)
        if backend == "chroma" and not chroma_options.get("chroma_ephemeral"):
            # 仅 http 客户端需要在共享服务端上清理（ephemeral 随进程消失，不连服务端）。
            _cleanup_chroma_collections(prefix, label="agent-ab")

    if args.probe:
        await _probe_report(
            backend, modes, args.probe_top_k, chroma_options=chroma_options,
            client_label=client_label, probe_only=args.probe_only,
        )

    print("\n" + "=" * 100)
    print("[OK] RAG 世界评测完成（确定性；RAG 接入后的结论边界说明见上方差异与指标）")
    return 0


def _report_ab(results: dict[str, tuple], modes: list[str]) -> None:
    """A/B 报告：逐案差异 / 三路对比 / 证据来源 / 证据级隔离抽查。"""
    # --- InMemory vs RAG 逐案差异 + 三路对比 --------------------------------
    mem_result, mem_records, _ = results["InMemory(eval)"]
    mem_by_id = {r.eval_case_id: r for r in mem_records}
    exp = mem_result.expected

    print("\n" + "-" * 100)
    print("逐案决策差异（相对 InMemory(eval)）")
    for label in [f"RAG-{m}" for m in modes]:
        _result, records, _ctx = results[label]
        diffs = []
        for r in records:
            base = mem_by_id[r.eval_case_id]
            if base.decision != r.decision:
                scene = exp.get(r.eval_case_id, {}).get("scene", "?")
                diffs.append((r.eval_case_id, scene, base.decision, r.decision))
        same = len(records) - len(diffs)
        print(f"\n  {label}：与 InMemory 决策一致 {same}/{len(records)}，差异 {len(diffs)} 条")
        for cid, scene, base, cur in diffs[:DIFF_HEAD]:
            print(f"    · {cid} [{scene}] InMemory {base} → {label} {cur}")
        if len(diffs) > DIFF_HEAD:
            print(f"    … 其余 {len(diffs) - DIFF_HEAD} 条（见上方指标 digest）")
        if not diffs:
            print("    （本数据集上无决策差异 —— 证据内容/引用来源差异见下方隔离抽查）")

    # --- RAG 模式间三路对比（如实报告，不预设 Hybrid 最优） ----------------
    print("\n" + "-" * 100)
    print("三检索模式对比（agent 决策序列两两比对）")
    mode_results = {f"RAG-{m}": results[f"RAG-{m}"][1] for m in modes}
    for i, m1 in enumerate(modes):
        for m2 in modes[i + 1:]:
            r1, r2 = mode_results[f"RAG-{m1}"], mode_results[f"RAG-{m2}"]
            d = sum(1 for a, b in zip(r1, r2) if a.decision != b.decision)
            print(f"  RAG-{m1:<7} vs RAG-{m2:<7}: {d}/{len(r1)} 条决策不同")

    # --- 证据来源差异摘要（决策相同 ≠ 证据相同） ------------------------
    print("\n" + "-" * 100)
    print("证据来源差异摘要（决策一致时仍看引用来源/集合）")
    print(
        f"  {'world':<16} {'REJECT':>6} {'w/policy':>8} {'w/case':>7} "
        f"{'distinct_case':>13} {'distinct_policy':>15}  样例 case refs"
    )
    for label in ["InMemory(eval)"] + [f"RAG-{m}" for m in modes]:
        _result, records, _ = results[label]
        rejects = [r for r in records if r.decision == "REJECT"]
        pol_refs = {
            ev["ref_id"]
            for r in records
            for ev in r.evidence
            if ev.get("type") == "POLICY_REF" and ev.get("ref_id")
        }
        case_refs = {
            ev["ref_id"]
            for r in records
            for ev in r.evidence
            if ev.get("type") == "CASE_PRECEDENT" and ev.get("ref_id")
        }
        w_pol = sum(1 for r in rejects if any(
            ev.get("type") == "POLICY_REF" for ev in r.evidence))
        w_case = sum(1 for r in rejects if any(
            ev.get("type") == "CASE_PRECEDENT" for ev in r.evidence))
        sample = ", ".join(sorted(case_refs)[:4]) or "-"
        print(
            f"  {label:<16} {len(rejects):>6} {w_pol:>8} {w_case:>7} "
            f"{len(case_refs):>13} {len(pol_refs):>15}  {sample}"
        )

    # --- 证据级隔离抽查（运行期） ----------------------------------------
    print("\n" + "-" * 100)
    print("运行时证据级隔离抽查（RAG 世界引用 ref_id 前缀）")
    for label in [f"RAG-{m}" for m in modes]:
        _result, records, _ = results[label]
        refs = set()
        for r in records:
            for ev in r.evidence:
                if ev.get("type") in ("CASE_PRECEDENT", "POLICY_REF") and ev.get("ref_id"):
                    refs.add(ev["ref_id"])
        case_refs = sorted(x for x in refs if x.startswith("RAG_CASE_"))
        other_case_refs = sorted(x for x in refs if not x.startswith("RAG_CASE_") and not x.startswith("POLICY_"))
        bad = [x for x in other_case_refs if not x.startswith("POLICY_")]
        verdict = "PASS（RAG 世界仅引用 RAG_CASE_*/KB POLICY_*，无 eval GT / InMemory 先例）" if not bad else f"FAIL: {bad}"
        print(f"  {label}: CASE 引用 {len(case_refs)} 个（样例 {case_refs[:5]}…）| {verdict}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG 世界评测：InMemory vs RAG + 三模式对比")
    parser.add_argument("--data", default=DEFAULT_DATA, help=f"评测集 JSONL（默认 {DEFAULT_DATA}）")
    parser.add_argument("--modes", nargs="*", default=list(MODES), choices=list(MODES),
                        help="要跑的 RAG 检索模式（默认全三路）")
    parser.add_argument("--smoke", action="store_true", help="冒烟：只跑前 N 条（确定性）")
    parser.add_argument("--smoke-limit", type=int, default=10, help="smoke 上限（默认 10）")
    parser.add_argument("--backend", default="local", choices=list(BACKENDS),
                        help="RAG 索引后端（默认 local = numpy + MockHash；chroma = ChromaDB + LlamaIndex + BM25 + RRF）")
    parser.add_argument("--chroma-client", default=CHROMA_CLIENTS[0], choices=list(CHROMA_CLIENTS),
                        help=("chroma 臂的客户端：ephemeral（默认）= 进程内内存库，每臂独立、"
                              "无需本机服务端；http = 本机 127.0.0.1:8001 服务端（独占前缀建库、"
                              "用完即删）"))
    parser.add_argument("--probe", action="store_true",
                        help=f"追加 probe 三模式 Recall@K 报告（probe 复用 {PROBE_SOURCE}）")
    parser.add_argument("--probe-only", action="store_true",
                        help="只跑 probe 报告（跳过 agent A/B）")
    parser.add_argument("--probe-top-k", type=int, default=PROBE_TOP_K,
                        help=f"probe Recall@K 的 K（默认 {PROBE_TOP_K}）")
    args = parser.parse_args(argv)
    if args.probe_only:
        args.probe = True
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        return asyncio.run(_main(argv))
    except Exception as exc:
        print(f"[FAIL] RAG 世界评测失败: {exc!r}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
