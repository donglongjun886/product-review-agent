"""RAG 检索层验收测试：corpus 完整性 / 命中 / 过滤 / 三模式 / 确定性 / 装配 / 评测。

覆盖：Policy KB 与 Case KB 规模与唯一性校验、≥2 条 EXPIRED；**隔离红线**：Case KB 的
case_id 与 eval_data v1+v2 全部 case 标识（含 InMemory 种子先例）无交集，防评测作弊；
关键 query 命中目标条款/先例；``effective_only`` 与 category（含「全类目」匹配语义）/
risk_type 过滤；bm25/vector/hybrid 三模式可切换与 hybrid 融合权重；同 corpus+query
两次检索一致、HashEmbedder 同输入同输出且维度固定；``build_tools`` 的 memory 默认
不变与 rag 注入点；RAG 世界评测可跑通且默认 InMemory 回归不破。

确定性、无网络、无真 LLM、不写评测数据目录。
"""

from __future__ import annotations

import json
from pathlib import Path

from pra.domain.models import Decision, RiskType
from pra.evaluation.dataset.loader import load_dataset
from pra.evaluation.harness.agent_scheme import EVAL_PRECEDENTS, AgentScheme
from pra.evaluation.harness.base import EvalContext
from pra.evaluation.regression import compute_current_snapshot
from pra.rag.bm25 import tokenize
from pra.rag.corpus import load_cases, load_policies
from pra.rag.embedder import MOCK_DIM, MockHashEmbedder
from pra.rag.factory import build_case_index, build_policy_index
from pra.rag.index import RagCaseIndex, RagPolicyIndex
from pra.rag.retrieval import fuse_scores, normalize_minmax
from pra.tools import build_tools
from pra.tools.case_search.tool import CaseSearchFilters
from pra.tools.policy_search.tool import InMemoryPolicyIndex, PolicySearchFilters

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_V1 = REPO_ROOT / "eval_data" / "v1" / "cases_v1.jsonl"
EVAL_V2 = REPO_ROOT / "eval_data" / "v2" / "cases_v2.jsonl"
BASELINE_FILE = REPO_ROOT / "eval_data" / "v1" / "regression_baseline.json"

# 固定断言目标（corpus 由 scripts/build_rag_corpus.py 确定性生成、git 入库；
# 若重建数据导致目标移位，应先核对检索质量再同步本测试 —— 不允许为过测试而改数据）
_SHOE_MIMIC_POLICY = "POLICY_1.4"  # 鞋靴外观高度模仿知名品牌（女鞋/运动鞋，EFFECTIVE）
_IP_POLICY_FAMILY = {"POLICY_1.1", "POLICY_1.2", "POLICY_1.4", "POLICY_1.5", "POLICY_1.6", "POLICY_4.5"}
_SHOE_FLAGSHIP_CASE = "RAG_CASE_0001"  # demo（P_88231/M_5512 剧情改写）无品牌复古跑鞋先例
_SHOE_CASE_QUERY = "无品牌高相似商家多次上架"


def _eval_case_ids() -> set[str]:
    """eval_data v1+v2 全部 case 标识（eval_case_id / input.case_id / lineage.seed_case_id）。"""
    ids: set[str] = set()
    for path in (EVAL_V1, EVAL_V2):
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ids.add(obj["eval_case_id"])
            ids.add(obj["input"]["case_id"])
            lineage = obj.get("lineage") or {}
            if lineage.get("seed_case_id"):
                ids.add(lineage["seed_case_id"])
    # 追加 InMemory eval 世界种子先例（agent_scheme.EVAL_PRECEDENTS / CASE_1832 等）
    for row in EVAL_PRECEDENTS:
        ids.add(row["case_id"])
    return ids


def test_corpus_data_integrity() -> None:
    policies, pmeta = load_policies()
    cases, cmeta = load_cases()

    assert 20 <= len(policies) <= 40, f"Policy KB 条款数须 20~40，实际 {len(policies)}"
    assert 50 <= len(cases) <= 100, f"Case KB 条数须 50~100，实际 {len(cases)}"

    expired = [p for p in policies if p.status == "EXPIRED"]
    assert len(expired) >= 2, "Policy KB 须含 ≥2 条 EXPIRED（版本过滤测试面）"

    assert len({p.clause_id for p in policies}) == len(policies), "clause_id 须唯一"
    case_ids = [c.case_id for c in cases]
    assert len(case_ids) == len(set(case_ids)), "case_id 须唯一"
    assert all(str(cid).startswith("RAG_CASE_") for cid in case_ids)

    # meta 携带来源/隔离声明（JSON 无注释 → meta 字段，schema 校验即生效）
    assert "isolation_declaration" in pmeta and "isolation_declaration" in cmeta


def test_case_kb_isolated_from_eval_gt() -> None:
    """Case KB 与 eval GT 严格隔离（防检索到 GT = 评测作弊）。"""
    cases, _ = load_cases()
    kb_ids = {c.case_id for c in cases}
    eval_ids = _eval_case_ids()

    overlap = kb_ids & eval_ids
    assert not overlap, f"Case KB 与 eval GT case_id 有交集（红线违反）: {sorted(overlap)[:10]}"
    # InMemory 种子先例 id 亦不得混入（CASE_1832/0911/2033/2120 等）
    assert kb_ids.isdisjoint({r["case_id"] for r in EVAL_PRECEDENTS})


async def test_policy_retrieval_hits_ip_clause() -> None:
    idx = build_policy_index()  # hybrid 默认
    hits = await idx.search(
        "外观高度模仿知名品牌无授权", PolicySearchFilters(), top_k=5, effective_only=True
    )
    assert hits, "验收：query 应命中 IP 政策条款"
    assert all(h.status == "EFFECTIVE" for h in hits)
    top = hits[0]
    assert RiskType.POTENTIAL_IP_RISK in top.risk_type, "首条须为 IP 风险条款"
    assert top.policy_id in _IP_POLICY_FAMILY
    # 加类目过滤后鞋类条款精确居首（可复现目标）
    shoe_hits = await idx.search(
        "外观高度模仿知名品牌无授权",
        PolicySearchFilters(category="女鞋/运动鞋"),
        top_k=5,
        effective_only=True,
    )
    assert shoe_hits[0].policy_id == _SHOE_MIMIC_POLICY


async def test_case_retrieval_hits_relevant_precedent() -> None:
    idx = build_case_index()  # hybrid 默认
    hits = await idx.search(
        _SHOE_CASE_QUERY, CaseSearchFilters(category="女鞋/运动鞋"), top_k=5
    )
    assert hits, "验收：query 应命中对应先例"
    assert hits[0].decision == Decision.REJECT
    assert any(h.case_id == _SHOE_FLAGSHIP_CASE for h in hits), (
        f"女鞋先例检索应含 demo 改写旗舰案 {_SHOE_FLAGSHIP_CASE}"
    )
    # retrieval_score 为检索期分（0~1），可作 CASE_PRECEDENT 证据 weight
    assert all(0.0 <= h.retrieval_score <= 1.0 for h in hits)


async def test_version_filter_effective_only() -> None:
    idx = build_policy_index()
    query = "永久去皱 根治脚气 功效夸大"  # 命中 POLICY_2.1 v1(EXPIRED 旧版) 文案
    both = await idx.search(query, PolicySearchFilters(), top_k=10, effective_only=False)
    effective = await idx.search(query, PolicySearchFilters(), top_k=10, effective_only=True)

    expired_hit = [h for h in both if h.status == "EXPIRED" and h.policy_id == "POLICY_2.1"]
    assert expired_hit and expired_hit[0].version == 1, "effective_only=False 应含 EXPIRED 旧版"
    assert effective and all(h.status == "EFFECTIVE" for h in effective), (
        "effective_only=True 不应含任何 EXPIRED"
    )
    assert not any(h.policy_id == "POLICY_2.1" and h.version == 1 for h in effective)


async def test_category_filter_includes_full_category() -> None:
    idx = build_policy_index()
    # InMemory 语义：row.category ∈ (None, query.category, 全类目) → 具体类目 + 全类目都留
    bag_hits = await idx.search(
        "外观模仿 品牌", PolicySearchFilters(category="箱包/女包"), top_k=20, effective_only=True
    )
    assert bag_hits
    assert all(h.category in ("箱包/女包", "全类目") for h in bag_hits)
    assert any(h.policy_id == "POLICY_1.5" for h in bag_hits)          # 箱包条款
    assert not any(h.policy_id == _SHOE_MIMIC_POLICY for h in bag_hits)  # 鞋类条款不得出现
    assert any(h.category == "全类目" for h in bag_hits)               # 全类目条款照常匹配

    all_hits = await idx.search(
        "外观模仿 品牌", PolicySearchFilters(), top_k=20, effective_only=True
    )
    assert any(h.policy_id == _SHOE_MIMIC_POLICY for h in all_hits)
    assert any(h.policy_id == "POLICY_1.5" for h in all_hits)


async def test_risk_type_filter() -> None:
    idx = build_policy_index()
    hits = await idx.search(
        "功效",
        PolicySearchFilters(risk_type=[RiskType.FALSE_CLAIM]),
        top_k=20,
        effective_only=True,
    )
    assert hits
    assert all(RiskType.FALSE_CLAIM in h.risk_type for h in hits)

    case_idx = build_case_index()
    c_hits = await case_idx.search(
        "外观模仿",
        CaseSearchFilters(category="女鞋/运动鞋", risk_type=[RiskType.POTENTIAL_IP_RISK]),
        top_k=10,
    )
    assert c_hits
    assert all(RiskType.POTENTIAL_IP_RISK in h.risk_type for h in c_hits)


async def test_three_modes_switchable() -> None:
    q_policy = "外观高度模仿知名品牌无授权"
    q_case = _SHOE_CASE_QUERY
    for mode in ("bm25", "vector", "hybrid"):
        p_idx = build_policy_index(mode=mode)
        assert isinstance(p_idx, RagPolicyIndex) and p_idx.mode == mode
        p_hits = await p_idx.search(q_policy, PolicySearchFilters(), top_k=5, effective_only=True)
        assert p_hits, f"policy mode={mode} 应返回结果"

        c_idx = build_case_index(mode=mode)
        assert isinstance(c_idx, RagCaseIndex) and c_idx.mode == mode
        c_hits = await c_idx.search(q_case, CaseSearchFilters(), top_k=5)
        assert c_hits, f"case mode={mode} 应返回结果"
        assert all(0.0 <= h.retrieval_score <= 1.0 for h in c_hits)


def test_hybrid_fusion_weights_effective() -> None:
    """构造两极值：bm25 路看好 doc2、vector 路看好 doc1 → 权重翻转应翻转 argmax。"""
    bm25 = [0.1, 0.9]
    vec = [0.9, 0.1]
    f_bm25 = fuse_scores(bm25, vec, weights=(1.0, 0.0), normalize=False)
    f_vec = fuse_scores(bm25, vec, weights=(0.0, 1.0), normalize=False)
    f_50 = fuse_scores(bm25, vec, weights=(0.5, 0.5), normalize=False)
    assert f_bm25[1] > f_bm25[0]           # 纯 BM25 → doc2
    assert f_vec[0] > f_vec[1]             # 纯 Vector → doc1
    assert abs(f_50[0] - f_50[1]) < 1e-12  # 0.5/0.5 → 平手
    # 归一化口径：等值集 → 全 1（确定性约定）
    assert normalize_minmax([2.0, 2.0]) == [1.0, 1.0]
    assert normalize_minmax([1.0, 3.0]) == [0.0, 1.0]


async def test_retrieval_deterministic() -> None:
    queries = [
        ("policy", "外观高度模仿知名品牌无授权", PolicySearchFilters(), 5),
        ("case", _SHOE_CASE_QUERY, CaseSearchFilters(), 5),
    ]
    for kind, q, filters, top_k in queries:
        idx_a = build_policy_index() if kind == "policy" else build_case_index()
        idx_b = build_policy_index() if kind == "policy" else build_case_index()
        if kind == "policy":
            ha = await idx_a.search(q, filters, top_k=top_k, effective_only=True)
            hb = await idx_b.search(q, filters, top_k=top_k, effective_only=True)
            assert [(h.clause_id, h.text) for h in ha] == [(h.clause_id, h.text) for h in hb]
        else:
            ha = await idx_a.search(q, filters, top_k=top_k)
            hb = await idx_b.search(q, filters, top_k=top_k)
            assert [(h.case_id, h.retrieval_score) for h in ha] == [(h.case_id, h.retrieval_score) for h in hb]


def test_mock_hash_embedder_deterministic_and_fixed_dim() -> None:
    emb = MockHashEmbedder()
    text = "外观高度模仿知名品牌，无品牌授权"
    v1 = emb.embed(text)
    v2 = emb.embed(text)
    assert v1 == v2, "同输入同输出（确定性 mock）"
    assert len(v1) == MOCK_DIM == 256
    other = MockHashEmbedder(dim=128)
    assert len(other.embed(text)) == 128
    assert emb.embed("外观高度模仿") != emb.embed("普通休闲鞋 无品牌 无模仿")
    # tokenize 确定性 + 无内置 hash() 依赖（跨进程可重放）
    assert tokenize("无品牌高相似 商家 image_similarity>=0.85") == tokenize(
        "无品牌高相似 商家 image_similarity>=0.85"
    )


_EXPECTED_TOOLS = [
    "ProductTool",
    "ImageAnalysisTool",
    "OCRTool",
    "MerchantTool",
    "CaseSearchTool",
    "PolicySearchTool",
]


def test_build_tools_memory_unchanged() -> None:
    memory = build_tools("memory")
    assert [t.name for t in memory] == _EXPECTED_TOOLS
    # 默认参数即 memory：两个检索工具仍注入 InMemory 索引
    default = build_tools()
    assert [t.name for t in default] == _EXPECTED_TOOLS
    assert type(memory[4]._index).__name__ == "InMemoryCaseIndex"
    assert isinstance(memory[5]._index, InMemoryPolicyIndex)


def test_build_tools_rag_injection() -> None:
    rag = build_tools("rag")
    assert [t.name for t in rag] == _EXPECTED_TOOLS
    assert type(rag[4]._index).__name__ == "RagCaseIndex"
    assert type(rag[5]._index).__name__ == "RagPolicyIndex"
    # 其余 4 工具不受影响（事实世界仍 InMemory）
    assert type(rag[0]._repo).__name__ == "InMemoryProductRepository"
    assert isinstance(rag[5]._index, RagPolicyIndex)
    assert rag[4].name == "CaseSearchTool" and rag[5].name == "PolicySearchTool"


async def test_rag_world_eval_runs_and_default_regression_intact() -> None:
    # (a) 默认路径（tool_world="eval"）回归：全量 v1 agent 决策序列 == 入库基线
    baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    assert baseline.get("format_version") == 1
    snap = await compute_current_snapshot(EVAL_V1, schemes=("agent",))
    assert snap["decisions"]["agent"] == baseline["decisions"]["agent"], (
        "默认 InMemory 评测路径回归失败：agent 决策序列与基线不一致"
    )

    # (b) RAG 世界（smoke 子集）可跑通、确定性、且证据来源与 eval GT 隔离
    scheme = AgentScheme()
    smoke = load_dataset(EVAL_V1)[:4]
    ctx = EvalContext(tool_world="rag", rag_mode="hybrid")
    recs1 = [await scheme.run(c, ctx) for c in smoke]
    recs2 = [await scheme.run(c, ctx) for c in smoke]
    assert [r.decision for r in recs1] == [r.decision for r in recs2], "RAG 世界确定性重放"
    assert all(r.scheme == "agent" and r.eval_case_id == c.eval_case_id for r, c in zip(recs1, smoke))
    rag_refs = {
        ev["ref_id"]
        for r in recs1
        for ev in r.evidence
        if ev.get("type") == "CASE_PRECEDENT" and ev.get("ref_id")
    }
    assert rag_refs and all(str(x).startswith("RAG_CASE_") for x in rag_refs), (
        "RAG 世界不得引用 eval GT / InMemory 种子先例"
    )
    # (c) 显式 "eval" 与默认 ctx 等价（回归不破坏）
    ctx_default = EvalContext()
    assert ctx_default.tool_world == "eval"
    default_recs = [await scheme.run(c, ctx_default) for c in smoke]
    eval_recs = [await scheme.run(c, EvalContext(tool_world="eval")) for c in smoke]
    assert [r.decision for r in default_recs] == [r.decision for r in eval_recs]
