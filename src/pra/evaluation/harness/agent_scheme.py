"""System 3：完整调查 Agent（走真实图 + eval 世界 + 确定性审查员桩）。

走 ``build_agent_graph``（hypothesize→plan→tools→reevaluate→decide，预算
10/15/40000/30000），终态以确定性 overlay 后的 ``ReviewDecision`` 为评测真值。
不落 DB；每 case 独立 build + compile 一个图、thread_id 唯一 → 天然隔离、可重放。
**scripted（CI 可跑）**：注入确定性 ``EvalScriptedLLMBackend``，工具用与 eval_data/v1
同一份 InMemory 种子世界 —— Agent 经工具拿到 Single-call / Rule 看不到的证据。
**real**：``AgentScheme(llm=<对象>)`` 直接把该对象交给 ``build_agent_graph``；
**非确定性、不可重放**、需 API key，仅作观测对照，不进确定性回归基线。
**结论边界（报告必标注）**：默认工具为 InMemory 种子、LLM 为桩 —— 种子里查不到的
先例/规避史会低估 Agent 上限；另提供 RAG 世界（``tool_world="rag"``）。

确定性"审查员模型"（细则见各方法 docstring）：
- hypothesize 按**表面信号**建假设，prior 由信号确定性给定；
- plan 按证据类型缺口补五类取证（外观→商品→商家→先例/政策）；
- reevaluate：相似>=0.85 / Logo>=0.7 → 外观支持；弱相似(0.70~0.85) → 弱支持；商家
  removals>=3 或改标题>=3 → 系统性支持；在库干净 → 证伪"系统性"；缺证据 →
  UNRESOLVED（"没查到 ≠ 证伪"）。LLM 消息只带 hypotheses+evidence，表面事实经
  hypothesize 固化进假设，本层不自造事实。**已知边界**：品牌维度只判"在库品牌非空"
  （``_product_brand_nonnull``），**不比对案件品牌与在库品牌是否一致** —— 二者不一致
  （漂移/冒名）会被当作"核验通过"证伪；"案件 brand 在案 + 虚构 pid 查无"只产低先验
  （prior 0.2 < 0.3）UNRESOLVED，不挡 PASS；
- decide：**无受支持的"高优先"风险（prior>=0.3）且高优先假设均已证伪 → PASS**；
  受支持的文本仿冒 / 强视觉 / （弱视觉且商家系统性）→ REJECT（再经 REJECT Gate 校验
  可引用依据与 dc）；其余 → HUMAN。低先验 SUPPORTED 与 PASS 相容是刻意行为：交叉判据
  用"假设是否成立"而非"先验"。Gate / abstention overlay 仍做最终收口。

确定性约束：纯函数 + 异步包装；不读 expected、不读外部配置；阈值常量本地声明并与
gate / evidence / tools 同口径；同 (node, __STATE__) → 同 payload。
"""

from __future__ import annotations

import json
import re
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from langgraph.graph.state import CompiledStateGraph

from pra.agent.checkpointer import make_memory_checkpointer
from pra.agent.graph import build_agent_graph
from pra.agent.guardrails.budget import (  # 预算超限维度常量（P2-16 记录侧复用）
    DIM_LLM_CALLS,
    DIM_TOOL_CALLS,
    DIM_TOKENS,
    DIM_LATENCY,
)
from pra.agent.guardrails.llm_shell import LLMResponse
from pra.agent.scripted_llm import (  # __STATE__ 解析/引用串格式
    _citation,
    _extract_state,
)
from pra.agent.state import build_initial_state
from pra.domain.models import ReviewDecision
from pra.evaluation.dataset.schema import EvalCase
from pra.evaluation.harness.base import EvalContext, EvalRecord, SchemeRunner
from pra.observability.tracing import (
    TraceContext,
    experiment_name,
    get_tracer,
    session_id,
)
from pra.screening.rule_engine.terms import BRAND_TERMS, EVASION_TERMS

__all__ = [
    "EVAL_CATEGORIES",
    "EVAL_IMAGE_MATCHES",
    "EVAL_MERCHANTS",
    "EVAL_POLICY_CLAUSES",
    "EVAL_PRECEDENTS",
    "EVAL_PRODUCTS",
    "EVAL_WORLD_LABEL",
    "RAG_WORLD_LABEL",
    "AgentScheme",
    "EvalScriptedLLMBackend",
    "make_eval_world_tools",
    "make_rag_world_tools",
]

# 评测种子世界（与 eval_data/v1 同一份"事实知识"；默认演示种子同源扩展）
# 三方案公平性：Rule / Single-call 只用基础输入（case 快照），本世界知识只能经
# Agent 的 5 个 InMemory 工具获得 —— 多信号/对抗类案需要调查才能发现。

EVAL_CATEGORIES: tuple[str, ...] = ("女鞋/运动鞋", "箱包/女包", "服装/卫衣")

# 图片外观分析种子（url → top_similar / logos / visual_risk）。
EVAL_IMAGE_MATCHES: dict[str, dict[str, Any]] = {
    # 强相似（>=0.85）：外观模仿确证
    "https://cdn.example.com/products/P_88231/img1.jpg": {
        "top_similar": [{"brand_ref": "某品牌经典鞋款", "similarity": 0.91}],
        "logos": [],
        "visual_risk": "外观与某品牌经典复古跑鞋高度相似",
    },
    "https://cdn.example.com/eval/viol_shoe/img1.jpg": {
        "top_similar": [{"brand_ref": "某品牌经典鞋款", "similarity": 0.93}],
        "logos": [],
        "visual_risk": "外观与某品牌经典鞋款高度相似",
    },
    "https://cdn.example.com/eval/viol_bag/img1.jpg": {
        "top_similar": [{"brand_ref": "某品牌托特包", "similarity": 0.95}],
        "logos": [],
        "visual_risk": "外观与某品牌托特包高度相似",
    },
    "https://cdn.example.com/eval/viol_hoodie/img1.jpg": {
        "top_similar": [{"brand_ref": "某品牌卫衣", "similarity": 0.90}],
        "logos": [],
        "visual_risk": "图案/版型与某品牌卫衣高度相似",
    },
    # 弱相似（0.70~0.85 单信号）：需与商家/在库事实交叉才可判
    "https://cdn.example.com/eval/bound_shoe/img1.jpg": {
        "top_similar": [{"brand_ref": "某品牌经典鞋款", "similarity": 0.72}],
        "logos": [],
        "visual_risk": "外观与某品牌经典鞋款存在一定相似",
    },
    "https://cdn.example.com/eval/bound_bag/img1.jpg": {
        "top_similar": [{"brand_ref": "某品牌托特包", "similarity": 0.73}],
        "logos": [],
        "visual_risk": "外观与某品牌托特包存在一定相似",
    },
    # Logo 检测命中（仅工具可得；与文本解耦）
    "https://cdn.example.com/eval/logo_bag/img1.jpg": {
        "top_similar": [],
        "logos": [{"brand": "GUCCI", "confidence": 0.90}],
        "visual_risk": "检测到疑似品牌 Logo",
    },
    # 干净图（无命中 → 无外观证据；"没查到"按低先验 UNRESOLVED 处理）
    "https://cdn.example.com/eval/clean_shoe1/img1.jpg": {},
    "https://cdn.example.com/eval/clean_shoe2/img1.jpg": {},
    "https://cdn.example.com/eval/clean_bag/img1.jpg": {},
    "https://cdn.example.com/eval/clean_hoodie/img1.jpg": {},
}

# 商家行为画像种子（removals/title_relisting_count >= 3 = 系统性信号）
EVAL_MERCHANTS: dict[str, dict[str, Any]] = {
    "M_5512": {  # 脏：5 removals / 3 改标题重上架（默认演示商家，同源）
        "merchant_id": "M_5512",
        "product_total": 120,
        "similar_product_count": 23,
        "removals": 5,
        "title_relisting_count": 3,
        "violations": {"total": 2, "by_type": {"IP_MIMIC": 1, "FALSE_CLAIM": 1}},
        "credit_score": 62,
        "recent_events": [
            {"event_type": "改标题重上架", "ts": "2024-09-01T10:00:00Z"},
            {"event_type": "下架", "ts": "2024-08-20T09:00:00Z"},
        ],
    },
    "M_8801": {  # 脏：7 removals / 4 改标题
        "merchant_id": "M_8801",
        "product_total": 45,
        "similar_product_count": 31,
        "removals": 7,
        "title_relisting_count": 4,
        "violations": {"total": 4, "by_type": {"IP_MIMIC": 3, "EVASION": 1}},
        "credit_score": 38,
        "recent_events": [
            {"event_type": "改标题重上架", "ts": "2024-09-02T10:00:00Z"},
            {"event_type": "改标题重上架", "ts": "2024-08-15T10:00:00Z"},
            {"event_type": "下架", "ts": "2024-08-10T09:00:00Z"},
        ],
    },
    "M_3307": {  # 干净
        "merchant_id": "M_3307",
        "product_total": 28,
        "similar_product_count": 0,
        "removals": 0,
        "title_relisting_count": 0,
        "violations": {"total": 0, "by_type": {}},
        "credit_score": 96,
        "recent_events": [],
    },
    "M_9904": {  # 干净
        "merchant_id": "M_9904",
        "product_total": 12,
        "similar_product_count": 0,
        "removals": 0,
        "title_relisting_count": 0,
        "violations": {"total": 0, "by_type": {}},
        "credit_score": 92,
        "recent_events": [],
    },
    "M_6602": {  # 中性：1 removals（低于系统性阈值 3）
        "merchant_id": "M_6602",
        "product_total": 8,
        "similar_product_count": 1,
        "removals": 1,
        "title_relisting_count": 0,
        "violations": {"total": 1, "by_type": {"FALSE_CLAIM": 1}},
        "credit_score": 85,
        "recent_events": [{"event_type": "下架", "ts": "2024-07-01T09:00:00Z"}],
    },
}

# 商品在库事实种子（ProductTool 用）
EVAL_PRODUCTS: dict[str, dict[str, Any]] = {
    "P_88231": {  # 默认演示商品（在库 brand=None；强相似图 0.91）
        "product_id": "P_88231",
        "merchant_id": "M_5512",
        "title": "新款厚底复古跑鞋 女士百搭运动鞋",
        "description": "经典复古跑鞋设计，轻量缓震，适合日常通勤和运动。",
        "category": "女鞋/运动鞋",
        "brand": None,
        "attributes": {"材质": "PU", "鞋底": "橡胶", "适用人群": "女士"},
        "sku_list": [{"sku_id": "S_1", "color": "米白", "size": "36-40", "price": 129.0}],
        "images": [{"url": "https://cdn.example.com/products/P_88231/img1.jpg", "source": "主图"}],
        "version": 3,
        "listing_time": "2024-09-06 14:00:00",
        "status": "ON_SALE",
    },
    "P_77310": {
        "product_id": "P_77310",
        "merchant_id": "M_5512",
        "title": "复古板鞋 男女同款休闲鞋",
        "description": "经典复古板鞋版型，街头风格。",
        "category": "女鞋/运动鞋",
        "brand": None,
        "attributes": {"鞋面": "帆布", "适用人群": "男女通用"},
        "sku_list": [{"sku_id": "S_1", "color": "黑色", "size": "38-44", "price": 99.0}],
        "images": [{"url": "https://cdn.example.com/eval/viol_shoe/img1.jpg", "source": "主图"}],
        "version": 1,
        "listing_time": "2024-09-10 10:00:00",
        "status": "ON_SALE",
    },
    "P_55208": {  # 自有品牌（潮动）但外观 0.93 强相似 + 脏商家 —— 对抗案核心
        "product_id": "P_55208",
        "merchant_id": "M_5512",
        "title": "潮动轻量缓震跑鞋",
        "description": "轻量缓震，日常跑步通勤皆宜。",
        "category": "女鞋/运动鞋",
        "brand": "潮动",
        "attributes": {"材质": "织物", "适用人群": "女士"},
        "sku_list": [{"sku_id": "S_1", "color": "浅灰", "size": "36-40", "price": 159.0}],
        "images": [{"url": "https://cdn.example.com/eval/viol_shoe/img1.jpg", "source": "主图"}],
        "version": 2,
        "listing_time": "2024-09-12 11:00:00",
        "status": "ON_SALE",
    },
    "P_31240": {
        "product_id": "P_31240",
        "merchant_id": "M_5512",
        "title": "大容量百搭帆布包",
        "description": "简约大容量帆布托特，日常通勤。",
        "category": "箱包/女包",
        "brand": None,
        "attributes": {"材质": "帆布", "容量": "大容量"},
        "sku_list": [{"sku_id": "S_1", "color": "米色", "size": "均码", "price": 49.0}],
        "images": [{"url": "https://cdn.example.com/eval/bound_bag/img1.jpg", "source": "主图"}],
        "version": 1,
        "listing_time": "2024-09-08 09:00:00",
        "status": "ON_SALE",
    },
    "P_66820": {
        "product_id": "P_66820",
        "merchant_id": "M_8801",
        "title": "大容量托特包 通勤手提",
        "description": "简约托特包，多袋设计。",
        "category": "箱包/女包",
        "brand": None,
        "attributes": {"材质": "PU", "容量": "大容量"},
        "sku_list": [{"sku_id": "S_1", "color": "黑色", "size": "均码", "price": 89.0}],
        "images": [{"url": "https://cdn.example.com/eval/viol_bag/img1.jpg", "source": "主图"}],
        "version": 2,
        "listing_time": "2024-09-09 15:00:00",
        "status": "ON_SALE",
    },
    "P_66900": {
        "product_id": "P_66900",
        "merchant_id": "M_8801",
        "title": "简约通勤手提包",
        "description": "简约设计，通勤多用。",
        "category": "箱包/女包",
        "brand": None,
        "attributes": {"材质": "PU", "容量": "中容量"},
        "sku_list": [{"sku_id": "S_1", "color": "棕色", "size": "均码", "price": 79.0}],
        "images": [{"url": "https://cdn.example.com/eval/logo_bag/img1.jpg", "source": "主图"}],
        "version": 1,
        "listing_time": "2024-09-07 12:00:00",
        "status": "ON_SALE",
    },
    "P_90771": {
        "product_id": "P_90771",
        "merchant_id": "M_8801",
        "title": "复古印花宽松卫衣",
        "description": "宽松版型，复古印花。",
        "category": "服装/卫衣",
        "brand": None,
        "attributes": {"材质": "棉", "版型": "宽松"},
        "sku_list": [{"sku_id": "S_1", "color": "灰色", "size": "M-2XL", "price": 129.0}],
        "images": [{"url": "https://cdn.example.com/eval/viol_hoodie/img1.jpg", "source": "主图"}],
        "version": 1,
        "listing_time": "2024-09-11 16:00:00",
        "status": "ON_SALE",
    },
    "P_55190": {  # 干净自有品牌（云步），在库 brand 可查
        "product_id": "P_55190",
        "merchant_id": "M_3307",
        "title": "云步轻弹缓震跑步鞋 女款",
        "description": "自主品牌轻弹缓震跑步鞋，适合日常慢跑。",
        "category": "女鞋/运动鞋",
        "brand": "云步",
        "attributes": {"材质": "织物", "适用人群": "女士"},
        "sku_list": [{"sku_id": "S_1", "color": "白色", "size": "36-40", "price": 199.0}],
        "images": [{"url": "https://cdn.example.com/eval/clean_shoe1/img1.jpg", "source": "主图"}],
        "version": 2,
        "listing_time": "2024-08-01 10:00:00",
        "status": "ON_SALE",
    },
    "P_44702": {
        "product_id": "P_44702",
        "merchant_id": "M_3307",
        "title": "云步百搭小白鞋",
        "description": "自主品牌百搭小白鞋，简约舒适。",
        "category": "女鞋/运动鞋",
        "brand": "云步",
        "attributes": {"材质": "皮革", "适用人群": "女士"},
        "sku_list": [{"sku_id": "S_1", "color": "白色", "size": "35-39", "price": 169.0}],
        "images": [{"url": "https://cdn.example.com/eval/clean_shoe2/img1.jpg", "source": "主图"}],
        "version": 1,
        "listing_time": "2024-08-10 10:00:00",
        "status": "ON_SALE",
    },
    "P_61040": {
        "product_id": "P_61040",
        "merchant_id": "M_3307",
        "title": "简行极简通勤托特包",
        "description": "自主品牌极简托特包，大容量通勤。",
        "category": "箱包/女包",
        "brand": "简行",
        "attributes": {"材质": "帆布", "容量": "大容量"},
        "sku_list": [{"sku_id": "S_1", "color": "米白", "size": "均码", "price": 139.0}],
        "images": [{"url": "https://cdn.example.com/eval/clean_bag/img1.jpg", "source": "主图"}],
        "version": 1,
        "listing_time": "2024-08-20 10:00:00",
        "status": "ON_SALE",
    },
    "P_23411": {
        "product_id": "P_23411",
        "merchant_id": "M_9904",
        "title": "山丘基础款纯色卫衣",
        "description": "自主品牌基础款纯色卫衣，重磅棉质。",
        "category": "服装/卫衣",
        "brand": "山丘",
        "attributes": {"材质": "棉", "版型": "宽松"},
        "sku_list": [{"sku_id": "S_1", "color": "黑色", "size": "M-2XL", "price": 99.0}],
        "images": [{"url": "https://cdn.example.com/eval/clean_hoodie/img1.jpg", "source": "主图"}],
        "version": 1,
        "listing_time": "2024-08-15 10:00:00",
        "status": "ON_SALE",
    },
    "P_44120": {  # 中性商家（M_6602）的干净自有品牌商品
        "product_id": "P_44120",
        "merchant_id": "M_6602",
        "title": "山野宽松纯色卫衣",
        "description": "自主品牌纯色卫衣。",
        "category": "服装/卫衣",
        "brand": "山野",
        "attributes": {"材质": "棉", "版型": "宽松"},
        "sku_list": [{"sku_id": "S_1", "color": "藏青", "size": "M-2XL", "price": 89.0}],
        "images": [{"url": "https://cdn.example.com/eval/clean_hoodie/img1.jpg", "source": "主图"}],
        "version": 1,
        "listing_time": "2024-08-25 10:00:00",
        "status": "ON_SALE",
    },
}

# 先例库种子（CaseSearchTool：category 精确 + risk_type 交叠 + retrieval_score 排序）
EVAL_PRECEDENTS: list[dict[str, Any]] = [
    {
        "case_id": "CASE_1832",
        "retrieval_score": 0.86,
        "decision": "REJECT",
        "risk_level": "HIGH",
        "risk_type": ["POTENTIAL_IP_RISK"],
        "summary": "无品牌标识 + 外观高度模仿知名品牌复古跑鞋 + 商家多次改标题重上架",
        "key_evidence": ["image_similarity>=0.85", "merchant_history>=5_removals"],
        "policy_refs": ["POLICY_3.2"],
        "category": "女鞋/运动鞋",
    },
    {
        "case_id": "CASE_0911",
        "retrieval_score": 0.31,
        "decision": "PASS",
        "risk_level": "NONE",
        "risk_type": [],
        "summary": "普通休闲鞋，无品牌标识且无相似外观",
        "key_evidence": [],
        "policy_refs": [],
        "category": "女鞋/运动鞋",
    },
    {
        "case_id": "CASE_2033",
        "retrieval_score": 0.88,
        "decision": "REJECT",
        "risk_level": "HIGH",
        "risk_type": ["POTENTIAL_IP_RISK"],
        "summary": "无品牌标识 + 外观高度模仿知名品牌手袋 + 商家多次改标题重上架",
        "key_evidence": ["image_similarity>=0.85", "merchant_history>=5_removals"],
        "policy_refs": ["POLICY_4.1"],
        "category": "箱包/女包",
    },
    {
        "case_id": "CASE_2120",
        "retrieval_score": 0.84,
        "decision": "REJECT",
        "risk_level": "HIGH",
        "risk_type": ["POTENTIAL_IP_RISK"],
        "summary": "卫衣图案/版型高度模仿知名潮牌 + 商家系统性重上架",
        "key_evidence": ["image_similarity>=0.85", "merchant_history>=5_removals"],
        "policy_refs": ["POLICY_5.2"],
        "category": "服装/卫衣",
    },
]

# 政策条款种子（PolicySearchTool：effective_only + category/risk_type 过滤）
EVAL_POLICY_CLAUSES: list[dict[str, Any]] = [
    {
        "policy_id": "POLICY_3.2",
        "version": 2,
        "clause_id": "POLICY_3.2_v2_c1",
        "title": "外观高度模仿知名品牌",
        "text": "商品外观高度模仿知名品牌设计且无品牌授权，判定为高风险，转人工审核处理",
        "category": "女鞋/运动鞋",
        "risk_type": ["POTENTIAL_IP_RISK"],
        "status": "EFFECTIVE",
        "effective_date": "2024-03-01",
    },
    {
        "policy_id": "POLICY_4.1",
        "version": 1,
        "clause_id": "POLICY_4.1_v1_c1",
        "title": "箱包外观高度模仿知名品牌",
        "text": "箱包类商品外观高度模仿知名品牌设计且无授权，判定为高风险，转人工审核处理",
        "category": "箱包/女包",
        "risk_type": ["POTENTIAL_IP_RISK"],
        "status": "EFFECTIVE",
        "effective_date": "2024-05-01",
    },
    {
        "policy_id": "POLICY_5.2",
        "version": 1,
        "clause_id": "POLICY_5.2_v1_c1",
        "title": "服装外观/图案模仿知名品牌",
        "text": "服装外观或图案高度模仿知名品牌且无授权，判定为高风险，转人工审核处理",
        "category": "服装/卫衣",
        "risk_type": ["POTENTIAL_IP_RISK"],
        "status": "EFFECTIVE",
        "effective_date": "2024-06-01",
    },
]

# 评测世界标识（报告"结论边界"标注用）
EVAL_WORLD_LABEL = "InMemory 种子世界 v1（含 P_88231/M_5512 演示种子扩展）"
# RAG 世界标识：真实 Policy KB / Case KB + 确定性 mock embedding + BM25 + 余弦
# （mode 由运行 ctx 注入，报告里拼上实际 mode —— 见 run_rag_eval.py）
RAG_WORLD_LABEL = "RAG 世界（真实 Policy/Case KB · 确定性 mock embedding + BM25 + 余弦）"


def make_eval_world_tools():
    """构造评测世界的 5 个 InMemory 工具（比 ``pra.tools.build_tools`` 少一个 ``OCRTool``）。

    **刻意不补 OCRTool**：评测集把 OCR 文本当基础输入（``input.images[].ocr_text``）直接
    消费，现有 v1/v2 案件既不带图片、也无 ``OCRTool`` 期望 → 补进来只会多一个无数据的
    工具。工具清单与生产各自维护，**不存在"同构"约束**，对标时以本函数为准。

    注入本模块评测种子（默认演示种子 P_88231 / M_5512 / POLICY_3.2 / CASE_1832 已并入
    EVAL_* 常量）—— 数据源与 eval_data/v1 同一份事实，杜绝"评测集与工具世界漂移"。
    """
    # 延迟 import：避免 evaluation 包导入期拉起全部工具子包（防环/省启动）
    from pra.tools.base import Tool
    from pra.tools.case_search.tool import CaseSearchTool, InMemoryCaseIndex
    from pra.tools.image_analysis.tool import (
        ImageAnalysisTool,
        MockImageAnalysisProvider,
    )
    from pra.tools.merchant.tool import InMemoryMerchantRepository, MerchantTool
    from pra.tools.policy_search.tool import InMemoryPolicyIndex, PolicySearchTool
    from pra.tools.product.tool import InMemoryProductRepository, ProductTool

    tools: list[Tool] = [
        ProductTool(repo=InMemoryProductRepository(EVAL_PRODUCTS)),
        ImageAnalysisTool(provider=MockImageAnalysisProvider(EVAL_IMAGE_MATCHES)),
        MerchantTool(repo=InMemoryMerchantRepository(EVAL_MERCHANTS)),
        CaseSearchTool(index=InMemoryCaseIndex(EVAL_PRECEDENTS)),
        PolicySearchTool(index=InMemoryPolicyIndex(EVAL_POLICY_CLAUSES)),
    ]
    return tools


def make_rag_world_tools(
    *, mode: str = "hybrid", backend: str = "local", backend_options: dict | None = None
):
    """构造 RAG 世界的 Agent 工具（评测 RAG 单独模式）。

    与 ``make_eval_world_tools`` 只差两个"知识库检索"工具：CaseSearchTool /
    PolicySearchTool 注入**真实 RAG 索引**（确定性 mock embedding + BM25 + 余弦，三模式
    可切换）；Product / Image / Merchant 仍沿用 eval 世界种子（事实锚点，两世界共用 →
    差异只归因于检索数据源）。

    :param mode: "bm25" / "vector" / "hybrid"（默认 hybrid 0.5/0.5）。
    :param backend: 索引后端（缺省 "local" 行为不变）—— "local"（既有 numpy 实现）/
        "qdrant" / "chroma"（ChromaDB + LlamaIndex；缺 ``rag`` extra 依赖时构造即抛，
        不静默降级）。只影响索引装配，零判定逻辑改动。
    :param backend_options: 后端装配参数透传（缺省 None = 不传 → 装配与改动前逐字节
        等价）；键名与 ``pra.rag.factory`` 构造参数逐字对应（如 chroma 的
        ``collection_prefix``），非法键由 factory 抛错。
    """
    # 延迟 import：避免 evaluation 包导入期拉起 pra.rag（防环/省启动）
    from pra.rag.factory import build_case_index, build_policy_index
    from pra.tools.base import Tool
    from pra.tools.case_search.tool import CaseSearchTool
    from pra.tools.image_analysis.tool import (
        ImageAnalysisTool,
        MockImageAnalysisProvider,
    )
    from pra.tools.merchant.tool import InMemoryMerchantRepository, MerchantTool
    from pra.tools.policy_search.tool import PolicySearchTool
    from pra.tools.product.tool import InMemoryProductRepository, ProductTool

    options = dict(backend_options or {})
    tools: list[Tool] = [
        ProductTool(repo=InMemoryProductRepository(EVAL_PRODUCTS)),
        ImageAnalysisTool(provider=MockImageAnalysisProvider(EVAL_IMAGE_MATCHES)),
        MerchantTool(repo=InMemoryMerchantRepository(EVAL_MERCHANTS)),
        CaseSearchTool(index=build_case_index(mode=mode, backend=backend, **options)),
        PolicySearchTool(index=build_policy_index(mode=mode, backend=backend, **options)),
    ]
    return tools


# 表面信号 / 证据统计辅助（纯函数）

_STYLE_WORDS = frozenset({"复古", "经典", "潮流", "同款", "ins风", "韩版"})

_T_IMAGE_SIM = "IMAGE_SIMILARITY"
_T_IMAGE_LOGO = "IMAGE_LOGO"
_T_PRODUCT = "PRODUCT_FACT"
_T_MERCHANT = "MERCHANT_HISTORY"
_T_CASE = "CASE_PRECEDENT"
_T_POLICY = "POLICY_REF"

_SIM_STRONG = 0.85  # 与 image_analysis.EVIDENCE_STRONG 同口径
_SIM_MIN = 0.70  # 与 image_analysis.EVIDENCE_MIN_SIM 同口径（评测审查员读证据视图的下限）
_LOGO_CONF = 0.70  # Logo 视为强视觉信号的置信下限
_HIGH_PRIOR = 0.3  # 与 gate.HIGH_PRIOR_THRESHOLD 同口径（Gate 只看 >=0.3 的假设）
_MERCHANT_DIRTY = 3  # removals/title >= 3 = 系统性（gate extra 口径一致）

# 假设 statement 标记（hypothesize 写、reevaluate/decide 读 —— 表面事实经它跨节点传递）
_MARK_VISUAL = "外观模仿风险"
_MARK_BRAND_MISSING = "案件品牌空缺"
_MARK_BRAND = "品牌规避风险"
_MARK_MERCHANT = "商家行为风险"
_MARK_TEXT = "仿冒词明示风险"

_ASCII_LATIN = re.compile(r"[a-z ]+\Z")


def _fold_text(text: str) -> str:
    return (text or "").lower().replace("：", ":")


def _hits_in(text: str, terms: frozenset[str]) -> list[str]:
    """词表命中（纯拉丁词边界 / 中文子串；与 screening rules 同口径）。"""
    folded = _fold_text(text)
    matched: list[str] = []
    for term in terms:
        tf = _fold_text(term)
        if _ASCII_LATIN.fullmatch(tf):
            pat = re.compile(rf"(?<![a-z0-9_]){re.escape(tf)}(?![a-z0-9_])")
        else:
            pat = re.compile(re.escape(tf))
        if pat.search(folded):
            matched.append(term)
    return sorted(matched)


def _case_surface(case: dict) -> dict:
    """case（JSON 形状）→ 表面信号（hypothesize 用；与 Single-call mock 同源口径）。"""
    product = case.get("product") or {}
    title = str(product.get("title") or "")
    desc = str(product.get("description") or "")
    text = f"{title}\n{desc}"
    ocr_parts = [
        img.get("ocr_text")
        for img in product.get("images") or []
        if isinstance(img, dict) and isinstance(img.get("ocr_text"), str) and img["ocr_text"].strip()
    ]
    ocr_hay = "\n".join(ocr_parts)
    brand = product.get("brand")
    brand_missing = brand is None or (isinstance(brand, str) and not brand.strip())
    urls = [
        i["url"]
        for i in (product.get("images") or [])
        if isinstance(i, dict) and isinstance(i.get("url"), str) and i["url"]
    ]
    return {
        "brand_missing": brand_missing,
        "text_evasion": _hits_in(text, EVASION_TERMS),
        "ocr_evasion": _hits_in(ocr_hay, EVASION_TERMS),
        "text_brand": _hits_in(text, BRAND_TERMS),
        "ocr_brand": _hits_in(ocr_hay, BRAND_TERMS),
        "style_words": [w for w in sorted(_STYLE_WORDS) if w in _fold_text(text)],
        "image_urls": urls,
    }


def _evidence_list(state: dict) -> list[dict]:
    return [e for e in (state.get("evidence") or []) if isinstance(e, dict)]


def _hypothesis_list(state: dict) -> list[dict]:
    return [h for h in (state.get("hypotheses") or []) if isinstance(h, dict)]


def _ev_types(evs: list[dict]) -> set:
    return {e.get("type") for e in evs if isinstance(e.get("type"), str)}


def _visible_sim_evidence(evs: list[dict], *, min_sim: float = _SIM_MIN) -> list[dict]:
    """审查员"看得见"的 IMAGE_SIMILARITY 证据（仅相似度证据，weight >= min_sim）。

    sweep 的最小侵入注入点：真实图 tools_node 的 quality_filter 常量（pra.agent，
    业务层不动）先以 0.70 兜底，评测审查员模型读证据视图时再按
    ``EvalContext.evidence_thresholds.min_sim`` 过滤 —— 只在评测侧模拟"更低/更高
    证据下限"的校准视图，原始 state 证据不动（审计可溯）。
    只返回 IMAGE_SIMILARITY 类型条目（相似度分档/证据引用只针对相似度证据）。
    """
    return [
        e for e in evs
        if e.get("type") == _T_IMAGE_SIM and float(e.get("weight") or 0.0) >= min_sim
    ]


def _sim_stats(
    evs: list[dict], *, strong: float = _SIM_STRONG, min_sim: float = _SIM_MIN
) -> tuple[float, bool, bool]:
    """(sim_max, sim_strong, any_sim)；similarity 即 IMAGE_SIMILARITY.weight。

    ``strong``/``min_sim`` 为 sweep 注入的相似度分档阈值（默认 0.85/0.70 保持现行为）：
    sim_strong = 可见强相似中 sim_max >= strong；any_sim = 可见证据里存在相似命中。
    """
    sims = [float(e.get("weight") or 0.0) for e in _visible_sim_evidence(evs, min_sim=min_sim)]
    sim_max = max(sims) if sims else 0.0
    return sim_max, sim_max >= strong, bool(sims)


def _logo_conf(evs: list[dict]) -> float:
    confs = [float(e.get("weight") or 0.0) for e in evs if e.get("type") == _T_IMAGE_LOGO]
    return max(confs) if confs else 0.0


def _merchant_flags(evs: list[dict]) -> tuple[bool, bool, bool]:
    """MERCHANT_HISTORY → (dirty, clean, known)。dirty=removals>=3 或 title>=3。"""
    for e in evs:
        if e.get("type") != _T_MERCHANT:
            continue
        extra = e.get("extra") or {}
        removals = int(extra.get("removals") or 0)
        title = int(extra.get("title") or 0)
        dirty = removals >= _MERCHANT_DIRTY or title >= _MERCHANT_DIRTY
        clean = removals == 0 and title == 0
        return dirty, clean, True
    return False, False, False


def _product_found(evs: list[dict]) -> bool:
    return any(e.get("type") == _T_PRODUCT for e in evs)


def _product_brand_nonnull(evs: list[dict]) -> bool:
    """PRODUCT_FACT 表明在库品牌非空（value 形如 brand=云步, version=… / brand=null, …）。

    **已知边界**：只判"在库品牌非空"，**不比对案件品牌与在库品牌是否一致** ——
    二者不一致（漂移/冒名）时 reevaluate 的 BRAND 分支会按"在库可查 → 案件空缺/存疑
    被证伪"（REFUTED）放行。当前 v2 数据不可达（虚构 pid 案全是 brand 空缺）；
    real/扩展数据可达。要比对一致性须另立规则 + 新 family，本模块不加。
    """
    for e in evs:
        if e.get("type") == _T_PRODUCT and re.search(
            r"brand=(?!null\b)\S+", str(e.get("value") or "")
        ):
            return True
    return False


def _has_citable(evs: list[dict]) -> bool:
    return any(e.get("type") in {_T_CASE, _T_POLICY} and e.get("ref_id") for e in evs)


def _risk_filters(category) -> dict:
    """先例/政策检索的元数据过滤：类目（有则带）+ 风险类型词表。"""
    filters: dict = {}
    if category:
        filters["category"] = category
    filters["risk_type"] = ["POTENTIAL_IP_RISK"]
    return filters


def _dim_of(statement: str) -> str:
    if _MARK_TEXT in statement:
        return "TEXT"
    if _MARK_VISUAL in statement:
        return "VISUAL"
    if _MARK_BRAND in statement:
        return "BRAND"
    if _MARK_MERCHANT in statement:
        return "MERCHANT"
    return "OTHER"


# EvalScriptedLLMBackend —— 确定性"审查员模型"（LLMBackend 协议）


class EvalScriptedLLMBackend:
    """确定性审查员模型：同 (node, __STATE__) → 同输出（tokens=0，可重放）。

    要点：reevaluate / decide 的 __STATE__ 只含 hypotheses + evidence（节点契约不带
    case）→ 本层不自造事实，只消费假设标记与证据；hypothesize 阶段已把表面信号固化
    进假设 prior/statement。

    可注入参数（均默认 None/默认值 → 行为逐字节不变）：
    - ``allowed_tools``：装配层裁剪（组件级 Ablation）—— plan 只排程该子集内的工具，
      图工具注册由 AgentScheme 另行过滤；None = 全工具。
    - ``evidence_thresholds``：证据阈值覆盖（sweep）—— min_sim 过滤审查员读到的
      相似度证据视图，strong 决定强相似分档；None = 0.70/0.85。
    """

    name = "eval-scripted-reviewer"

    def __init__(
        self,
        *,
        allowed_tools: set[str] | None = None,
        evidence_thresholds: dict | None = None,
    ) -> None:
        self._allowed_tools: frozenset[str] | None = (
            None if allowed_tools is None else frozenset(allowed_tools)
        )
        overrides = dict(evidence_thresholds or {})
        self._min_sim: float = float(overrides.get("min_sim", _SIM_MIN))
        self._strong: float = float(overrides.get("strong", _SIM_STRONG))

    async def complete(self, *, node: str, messages: list, json_schema: dict) -> LLMResponse:
        from pra.agent.guardrails.llm_shell import LLMBackendError

        state: dict = _extract_state(messages)  # __STATE__ {json}（缺省 {} → 兜底）
        if node == "hypothesize":
            payload = self._hypothesize(state)
        elif node == "plan":
            payload = self._plan(state)
        elif node == "reevaluate":
            payload = self._reevaluate(state)
        elif node == "decide":
            payload = self._decide(state)
        else:
            raise LLMBackendError(f"unknown node: {node}")
        return LLMResponse(content=json.dumps(payload, ensure_ascii=False), tokens=0)

    # hypothesize：按表面信号生成假设（prior = 信号强度的确定性映射）

    def _hypothesize(self, state: dict) -> dict:
        case = state.get("case")
        surface = _case_surface(case if isinstance(case, dict) else {})
        hyps: list[dict] = []
        queue: list[dict] = []

        # ① 外观模仿（有图才建；有风格词嫌疑才高先验 —— 避免对干净商品无谓高怀疑）
        if surface["image_urls"]:
            if surface["text_evasion"]:
                p_vis = 0.85
            elif surface["text_brand"] or surface["ocr_brand"]:
                p_vis = 0.65
            elif surface["style_words"]:
                p_vis = 0.45
            else:
                p_vis = 0.22
            hyps.append(
                {
                    "statement": "商品外观高度模仿某知名品牌款（外观模仿风险，需图片比对确认）",
                    "prior": p_vis,
                    "evidence_hint": ["IMAGE_SIMILARITY", "IMAGE_LOGO"],
                }
            )
            queue.append({"q": "外观是否与某知名品牌款高度相似？", "priority": 1})

        # ② 品牌核验（案件 brand 空缺 → 高先验 + statement 标记「案件品牌空缺」）
        if surface["brand_missing"]:
            stmt_brand = "案件品牌空缺，涉嫌刻意规避品牌识别（品牌规避风险）"
            p_brand = 0.65
        else:
            stmt_brand = "商品品牌真实性与在库一致性核验（品牌规避风险）"
            p_brand = 0.2
        hyps.append({"statement": stmt_brand, "prior": p_brand, "evidence_hint": ["PRODUCT_FACT"]})
        queue.append({"q": "商品品牌是否真实可核验？", "priority": 2})

        # ③ 商家行为（恒建；0.35 默认先验）
        hyps.append(
            {
                "statement": "商家存在系统性违规上架/改标题重上架历史（商家行为风险）",
                "prior": 0.35,
                "evidence_hint": ["MERCHANT_HISTORY"],
            }
        )
        queue.append({"q": "商家历史是否显示系统性类似上架行为？", "priority": 3})

        # ④ 文本明示仿冒（标题/描述/OCR 命中才建）
        if surface["text_evasion"] or surface["ocr_evasion"]:
            hyps.append(
                {
                    "statement": "标题/描述/OCR 含仿冒词，明示仿冒意图（仿冒词明示风险）",
                    "prior": 0.85,
                    "evidence_hint": [],
                }
            )
            queue.append({"q": "仿冒词是否与在库/商家事实印证？", "priority": 4})

        return {
            "hypotheses": hyps,
            "investigation_queue": queue,
            "rationale": "按案件表面信号（品牌空缺/文本词/图片存在性）确定性生成待验证假设。",
        }

    # plan：缺口驱动取证计划（每轮 ≤3 条；重复的 (tool,args) 由 dedup 过滤）

    def _plan(self, state: dict) -> dict:
        case = state.get("case")
        case = case if isinstance(case, dict) else {}
        product = case.get("product")
        product = product if isinstance(product, dict) else {}
        evs = _evidence_list(state)
        types = _ev_types(evs)
        urls = [
            img["url"]
            for img in (product.get("images") or [])
            if isinstance(img, dict) and isinstance(img.get("url"), str) and img["url"]
        ]
        product_id = product.get("product_id")
        merchant_id = case.get("merchant_id")
        category = product.get("category")

        # 有序候选：外观 → 在库商品 → 商家历史 → 先例 → 政策。商品/商家证据齐后才
        # 排先例/政策（REJECT 可引用依据属于"确认风险后"的取证）；已执行成功的
        # 同 (tool,args) 由确定性 dedup 兜底过滤 —— 干净图（IA 无命中）不会让计划
        # 卡死在反复重排 ImageAnalysis，下一轮会自动补先例/政策并收敛。
        facts_ready = _T_PRODUCT in types and _T_MERCHANT in types
        candidates: list[dict] = []
        if urls and _T_IMAGE_SIM not in types and _T_IMAGE_LOGO not in types:
            candidates.append(
                {"tool": "ImageAnalysisTool", "args": {"image_urls": urls},
                 "reason": "验证外观是否对应知名品牌款", "priority": 1}
            )
        if product_id and _T_PRODUCT not in types:
            candidates.append(
                {"tool": "ProductTool", "args": {"product_id": product_id},
                 "reason": "核对商品在库事实（品牌真实性/版本漂移）", "priority": 2}
            )
        if merchant_id and _T_MERCHANT not in types:
            candidates.append(
                {"tool": "MerchantTool", "args": {"merchant_id": merchant_id, "window_days": 90},
                 "reason": "核查商家系统性上架/下架历史", "priority": 3}
            )
        if facts_ready and _T_CASE not in types:
            candidates.append(
                {"tool": "CaseSearchTool",
                 "args": {"query": "外观高度模仿+商家多次重上架",
                          "filters": _risk_filters(category), "top_k": 5},
                 "reason": "检索同类外观模仿+多次重上架的裁决先例", "priority": 4}
            )
        if facts_ready and _T_POLICY not in types:
            candidates.append(
                {"tool": "PolicySearchTool",
                 "args": {"query": "外观高度模仿品牌设计",
                          "filters": _risk_filters(category), "top_k": 5, "effective_only": True},
                 "reason": "检索外观高度模仿品牌设计的政策条款", "priority": 5}
            )

        if self._allowed_tools is not None:
            # 装配层裁剪（组件级 Ablation）：plan 的 tool schema 侧不给被裁工具 ——
            # 候选仍按原缺口逻辑生成，但只排程允许子集内工具（不动判定逻辑）。
            candidates = [c for c in candidates if c["tool"] in self._allowed_tools]

        if not candidates:
            # 已无新证据可补（商品/商家缺失的畸形案由上面条件自然收尾）
            return {
                "next_action": "conclude",
                "tools": [],
                "rationale": "当前证据缺口无对应工具可补（商品/商家标识缺失或证据已齐备），收尾。",
            }
        return {
            "next_action": "call_tools",
            "tools": candidates[:3],  # 每轮 ≤3 条
            "rationale": "按证据缺口排本轮取证（外观/在库/商家/先例/政策）。",
        }

    # reevaluate：证据 → 假设状态（确定性；只列变化项，幂等）

    def _reevaluate(self, state: dict) -> dict:
        evs = _evidence_list(state)
        hyps = _hypothesis_list(state)
        sim_max, sim_strong, any_sim = _sim_stats(
            evs, strong=self._strong, min_sim=self._min_sim
        )
        logo = _logo_conf(evs)
        merch_dirty, merch_clean, merch_known = _merchant_flags(evs)
        prod_found = _product_found(evs)
        prod_brand_ok = _product_brand_nonnull(evs)
        has_citable = _has_citable(evs)

        updates: list[dict] = []
        for h in hyps:
            dim = _dim_of(str(h.get("statement") or ""))
            current_status = h.get("status")
            current_posterior = h.get("posterior")
            target: tuple | None = None  # (status, posterior, evidence_for, evidence_against)

            if dim == "VISUAL":
                if sim_strong or logo >= _LOGO_CONF:
                    posterior = max(sim_max, logo)
                    refs = [
                        self._citation(e) for e in _visible_sim_evidence(evs, min_sim=self._min_sim)
                    ] + [
                        self._citation(e) for e in evs if e["type"] == _T_IMAGE_LOGO
                    ]
                    target = ("SUPPORTED", round(posterior, 2), refs, [])
                elif any_sim:  # 弱相似(0.70~0.85)：弱支持（不构成"高度模仿"确证）
                    refs = [
                        self._citation(e) for e in _visible_sim_evidence(evs, min_sim=self._min_sim)
                    ]
                    target = ("SUPPORTED", round(sim_max, 2), refs, [])
                else:
                    target = ("UNRESOLVED", None, [], [])  # 无命中/无可比 → 查无结论
            elif dim == "BRAND":
                case_missing = _MARK_BRAND_MISSING in str(h.get("statement") or "")
                # 已知边界（P2-3）：prod_brand_ok 只证"在库品牌非空"，不比对案件品牌
                # 与在库品牌是否一致 —— 漂移/冒名（不一致）也会走 REFUTED"核验通过"。
                # v2 无 family 覆盖（不可达）；real/扩展数据可达。加比对规则需新 family
                # + 设计拍板，此处仅标注。
                if not prod_found:
                    # 在库未核验 → 不臆断；"案件 brand 在案 + 虚构 pid 查无"走这里 →
                    # UNRESOLVED 且 prior 0.2(<0.3) → 不挡 PASS（低优先 unresolved
                    # 不触发 HUMAN）—— 同上，属已知边界，不在此扩张规则。
                    target = ("UNRESOLVED", None, [], [])  # 在库未核验 → 不臆断
                elif prod_brand_ok:
                    # 在库品牌可查 → 案件"空缺/存疑"被证伪（快照字段缺失 ≠ 规避）
                    refs = [self._citation(e) for e in evs if e["type"] == _T_PRODUCT]
                    target = ("REFUTED", 0.1, [], refs)
                elif case_missing:
                    # 在库亦空缺 → 规避嫌疑成立
                    refs = [self._citation(e) for e in evs if e["type"] == _T_PRODUCT]
                    target = ("SUPPORTED", 0.7, refs, [])
                else:
                    # 案件品牌声明存在但在库无法佐证（虚构/漂移嫌疑）→ 存疑转人工
                    target = ("UNRESOLVED", None, [], [])
            elif dim == "MERCHANT":
                if merch_known and merch_dirty:
                    refs = [self._citation(e) for e in evs if e["type"] == _T_MERCHANT]
                    target = ("SUPPORTED", 0.85, refs, [])
                elif merch_known:
                    # 干净(0/0)或中性(0<removals<3)：未达"系统性"阈值 → 证伪"系统性"主张
                    posterior = 0.1 if merch_clean else 0.25
                    refs = [self._citation(e) for e in evs if e["type"] == _T_MERCHANT]
                    target = ("REFUTED", posterior, [], refs)
                else:
                    target = ("UNRESOLVED", None, [], [])  # 商家查无 → 无法结论
            elif dim == "TEXT":
                # 文本自证仿冒 + 支撑证据（先例/政策/视觉/商家任一同证即充分）
                refs = [
                    self._citation(e)
                    for e in evs
                    if e["type"] in {_T_CASE, _T_POLICY, _T_IMAGE_SIM, _T_MERCHANT, _T_PRODUCT}
                ]
                if has_citable or refs:
                    target = ("SUPPORTED", 0.9, refs, [])
                else:
                    target = ("UNRESOLVED", None, [], [])

            if target is None:
                continue  # 未知维度（防御：非本后端生成的假设不动）
            status, posterior, ev_for, ev_against = target
            # ReevaluateOutput.HypothesisUpdate.posterior 为必填 float → UNRESOLVED 给 0.0
            # 占位（"未证实也未证伪"无后验语义；gate/decide 不消费 UNRESOLVED 的 posterior）
            posterior = 0.0 if posterior is None else posterior
            if status == current_status and posterior == current_posterior:
                continue  # 幂等：无变化不产出（防无限翻转）
            updates.append(
                {
                    "id": h.get("id"),
                    "status": status,
                    "posterior": posterior,
                    "evidence_for": list(ev_for),
                    "evidence_against": list(ev_against),
                }
            )

        return {
            "hypothesis_updates": updates,
            "queue_updates": [],
            "new_hypotheses": [],
            "evidence_sufficiency": "SUFFICIENT" if has_citable else "INSUFFICIENT",
            "conflicts": [],
            "rationale": "按当前证据确定性综合假设；证据不足一律 UNRESOLVED，不把'没查到'当'证伪'。",
        }

    # decide：由假设终态 + 证据推提案（overlay/Gate 仍做最终收口）

    def _decide(self, state: dict) -> dict:
        evs = _evidence_list(state)
        hyps = _hypothesis_list(state)
        _sim_max, sim_strong, _ = _sim_stats(
            evs, strong=self._strong, min_sim=self._min_sim
        )
        logo = _logo_conf(evs)
        merch_dirty, _, _ = _merchant_flags(evs)
        has_citable = _has_citable(evs)
        visible_sim = _visible_sim_evidence(evs, min_sim=self._min_sim)

        def _pri(h: dict) -> float:
            try:
                return float(h.get("prior") or 0.0)
            except (TypeError, ValueError):
                return 0.0

        supported_high = [
            h for h in hyps if h.get("status") == "SUPPORTED" and _pri(h) >= _HIGH_PRIOR
        ]
        # 全部 SUPPORTED 假设（含低先验弱视觉 —— 交叉判据用"是否成立"而非"先验"）
        visual_supported = any(
            _dim_of(str(h.get("statement") or "")) == "VISUAL" and h.get("status") == "SUPPORTED"
            for h in hyps
        )
        text_flag = any(_dim_of(str(h.get("statement") or "")) == "TEXT" for h in supported_high)
        merchant_supported = any(
            _dim_of(str(h.get("statement") or "")) == "MERCHANT" for h in supported_high
        )
        visual_strong = sim_strong or logo >= _LOGO_CONF

        risk_types: list[str] = []
        # 视觉风险类型按"审查员可见证据视图"派生（sweep 抬 min_sim → 弱相似不再记 IP 风险）
        if visible_sim or any(e.get("type") == _T_IMAGE_LOGO for e in evs):
            risk_types.append("POTENTIAL_IP_RISK")
        if merch_dirty:
            risk_types.append("EVASION_PATTERN")

        # auto_reject 判定：原第 4 子句 (V∧B∧M) 被第 3 子句 (V∧M) 蕴含、恒死，已删
        # —— 行为零变化。
        auto_reject = (
            text_flag
            or visual_strong
            or (visual_supported and merchant_supported)
        )
        if supported_high:
            if auto_reject:
                if has_citable:
                    return {
                        "decision": "REJECT",
                        "risk_level": "HIGH",
                        "risk_type": list(dict.fromkeys(risk_types)),
                        "confidence": 0.9,
                        "evidence_ids": [self._citation(e) for e in evs],
                        "policy": [
                            str(e.get("extra", {}).get("policy_id"))
                            for e in evs
                            if e["type"] == _T_POLICY and (e.get("extra") or {}).get("policy_id")
                        ],
                        "rationale": "高优先风险假设获证据支持且政策/先例可引用，确定性提议拒绝。",
                    }
                return {
                    "decision": "HUMAN_REVIEW",
                    "risk_level": "HIGH",
                    "risk_type": list(dict.fromkeys(risk_types)),
                    "confidence": 0.6,
                    "evidence_ids": [self._citation(e) for e in evs],
                    "policy": [],
                    "rationale": "存在受支持的风险假设但政策依据缺失，克制转人工。",
                }
            return {
                "decision": "HUMAN_REVIEW",
                "risk_level": "MEDIUM",
                "risk_type": list(dict.fromkeys(risk_types)),
                "confidence": 0.5,
                "evidence_ids": [self._citation(e) for e in evs],
                "policy": [],
                "rationale": "存在未决风险关注但未达自动拒绝门槛，转人工复核。",
            }

        unresolved_high = [
            h for h in hyps
            if h.get("status") in ("PENDING", "UNRESOLVED") and _pri(h) >= _HIGH_PRIOR
        ]
        if unresolved_high:
            return {
                "decision": "HUMAN_REVIEW",
                "risk_level": "LOW",
                "risk_type": [],
                "confidence": 0.4,
                "evidence_ids": [self._citation(e) for e in evs],
                "policy": [],
                "rationale": "高优先假设仍未证实或证伪，证据不足转人工。",
            }
        return {
            "decision": "PASS",
            "risk_level": "NONE",
            "risk_type": [],
            "confidence": 0.8,
            "evidence_ids": [self._citation(e) for e in evs],
            "policy": [],
            "rationale": "无受支持的风险假设且高优先假设均已证伪，确定性提议放行。",
        }

    @staticmethod
    def _citation(ev: dict) -> str:
        return _citation(ev)


# SchemeRunner：走真实图（build_agent_graph + eval 世界 + eval 审查员后端）


def _validate_budget_limit_keys(overrides: dict, model_cls: type) -> None:
    """按 pydantic 模型字段白名单校验预算覆盖键；未知键抛 ValueError。

    pydantic v2 的 ``model_copy(update=…)`` **不校验键**：未知键会静默挂成实例多余
    属性而覆盖不生效 —— 拼错字（如 ``max_llm_call`` 少个 s）会让档位实验静默以生产
    默认 10 跑。装配层失败要响亮：非空 overrides 的每个键都必须命中
    ``model_cls.model_fields``。
    """
    allowed = set(model_cls.model_fields)
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(
            "AgentScheme budget_limits 含未知键（拼错字会被 model_copy 静默挂成多余"
            f"属性而不生效，已拒绝）：{unknown}；合法键 = BudgetLimits 字段"
            f"（无别名映射）：{sorted(allowed)}"
        )


def _budget_hit_dim_from_snapshot(budget) -> str | None:
    """从决策预算快照重算“首个撞限维度”（不改 overrides 码）。

    与 ``pra.agent.guardrails.budget.budget_exceeded`` 同阈值、同判定顺序
    （LLM_CALLS→TOOL_CALLS→TOKENS→LATENCY），但 latency 用**快照已冻结的
    ``latency_ms``** 而非实时墙钟 —— EvalRecord 保持不含进程相关量、可逐字节重放。
    只写进 EvalRecord.detail 供审计归因（区分真实跑分时 token / llm_calls / latency
    哪维先撞限）；``R3_BUDGET_EXHAUSTED`` 码字面与语义不变。
    """
    if budget is None:
        return None
    limits = budget.limits
    if budget.llm_calls >= limits.max_llm_calls:
        return DIM_LLM_CALLS
    if budget.tool_calls >= limits.max_tool_calls:
        return DIM_TOOL_CALLS
    if budget.tokens >= limits.max_tokens:
        return DIM_TOKENS
    if budget.latency_ms >= limits.max_latency_ms:
        return DIM_LATENCY
    return None


def _root_trace_context(
    case: EvalCase, ctx: EvalContext, initial_state: dict, *, backend_name: str = "unknown"
) -> TraceContext:
    """按确定性规则包装评测路径的 root trace（每案一条）。

    ``trace_id = uuid5(NAMESPACE_URL, f"{experiment}:{eval_case_id}:agent:{backend_name}")``
    —— **确定性**：同 experiment + 同案 + 同方案 + **同 LLM 后端**重跑落同一条 trace。

    **为什么把 ``backend_name`` 纳入 trace_id**：``run_evaluation_real.py`` 会在同一
    进程里对同一批 case 跑 scripted 对照臂 + real 臂（两臂 experiment 相同）—— 若
    trace_id 不含后端名，两臂会落进同一条 trace（实测每 trace 出现 2 个 root
    observation、real/scripted generation 交织，按 trace 汇总 token 会混入 0-token 的
    桩 generation）。纳入后端名后两臂各自成 trace，scripted 臂的确定性不变。

    ``session_id`` 取 ``PRA_LANGFUSE_SESSION``（一次评测 run 一个值）；``version`` =
    experiment 名。只读：不写 state、不参与任何判定。
    """
    experiment = experiment_name()
    metadata: dict[str, Any] = {
        "case_id": case.input.case_id,
        "eval_case_id": case.eval_case_id,
        "scene": case.scene,
        "scheme": "agent",
        "experiment": experiment,
        "llm_backend": backend_name,
        "tool_world": ctx.tool_world,
        "rag_mode": ctx.rag_mode,
        "source": "evaluation",
    }
    limits = getattr(initial_state.get("budget"), "limits", None)
    if limits is not None and hasattr(limits, "model_dump"):
        metadata["budget_limits"] = limits.model_dump()
    tags = [
        "env:local",
        "scheme:agent",
        f"experiment:{experiment}",
        "source:evaluation",
    ]
    if ctx.tool_world:
        tags.append(f"tool_world:{ctx.tool_world}")
    return TraceContext(
        trace_id=uuid5(
            NAMESPACE_URL, f"{experiment}:{case.eval_case_id}:agent:{backend_name}"
        ).hex,
        name="review",
        session_id=session_id(),
        version=experiment,
        metadata=metadata,
        tags=tags,
        input={"case_id": case.input.case_id, "eval_case_id": case.eval_case_id},
    )


class AgentScheme(SchemeRunner):
    """System 3 —— 完整调查 Agent（scripted 模式：eval 世界 + eval 审查员桩）。

    每 case 独立 build + compile 一个图（checkpointer=InMemory、thread_id 唯一），
    天然隔离、可重放；运行后把进程级 LLM 后端恢复为默认桩（防污染后续进程）。
    装配参数均默认 None → 行为逐字节不变：
    - ``allowed_tools``：**装配层裁剪** —— 图工具注册与 plan 的 tool schema 都只给该
      子集（不动判定逻辑）；None = eval 世界全 5 工具。裁剪后 plan 不再排程被裁工具，
      证据链自然缺该类证据 → 决策差异即"该组件必要性"的归因。
    - ``ctx.tool_world`` == "rag"：CaseSearch/PolicySearch 注入真实 RAG 索引，其余事实
      工具沿用 eval 世界；检索模式随 ``ctx.rag_mode``（None → hybrid）。
    - ``ctx.evidence_thresholds``：见 ``EvalContext``（sweep 注入相似度分档）。
    - ``llm``：**real 模式注入** —— 非 None 时 ``run()`` 直接把它当 LLMBackend 交给
      ``build_agent_graph``（跳过确定性桩）；须实现
      ``pra.agent.guardrails.llm_shell.LLMBackend`` Protocol（``name`` 属性 +
      ``async complete(*, node, messages, json_schema)``）；run 的 finally 仍统一恢复
      ``set_llm_backend(None)``。real 非确定性 / 不可重放 / 需 API key。
    - ``budget_limits``：**评测侧预算覆盖**（None = 默认 10/15/40000/30000）—— 键为
      ``BudgetLimits`` 字段名（``max_llm_calls`` / ``max_tool_calls`` / ``max_tokens`` /
      ``max_latency_ms``），对 ``budget.limits`` 做 model_copy 覆盖（不改生产对象）；
      **未知键抛 ValueError**（按 model_fields 白名单校验），拼错字不会被静默忽略。
      用途：real 模式放宽墙钟护栏（真实 LLM 每案 ~9 次串行调用天然 >30s，不放宽则每案
      都被 LATENCY 超限截胡转人工；scripted 毫秒级跑完不触发）；调 LLM 调用预算档
      （抬高后仍打满 ⇒ 收敛问题，涨到收敛即止 ⇒ 预算紧）。**生产护栏恒为
      10/15/40000/30000**，本覆盖只作用于评测装配层注入的 initial_state。
    """

    name = "agent"

    def __init__(
        self,
        allowed_tools: set[str] | None = None,
        *,
        llm: object | None = None,  # Phase 3 real 模式：注入 LLMBackend（None=确定性桩）
        budget_limits: dict | None = None,  # 评测侧预算覆盖（None=默认 10/15/40000/30000；real 放宽 latency / 调 llm 档用）
    ) -> None:
        # 审查员后端改为**每次 run 按 ctx 装配**（阈值/裁剪随 EvalContext 变），
        # 不在构造期缓存 —— sweep/ablation 同进程换 ctx 重跑才能生效。
        self._allowed_tools: frozenset[str] | None = (
            None if allowed_tools is None else frozenset(allowed_tools)
        )
        # real 模式注入的后端对象（None → run() 按 ctx 装配 EvalScriptedLLMBackend）；
        # 每次 run 仍统一经 build_agent_graph(llm=...) 注入并在 finally 恢复默认桩。
        self._llm: object | None = llm
        # 评测侧预算覆盖（None = 不覆盖，默认 10/15/40000/30000；real 模式放宽
        # max_latency_ms / 调 max_llm_calls 档用 —— 只改每次 run 初始 state 的
        # budget.limits，见 run()）。
        self._budget_limits: dict | None = dict(budget_limits or {}) or None
        if self._budget_limits is not None:
            # P2-14：装配期即白名单校验（拼错字 → ValueError，不静默以生产默认跑）
            from pra.domain.models import BudgetLimits  # 惰性 import：仅覆盖路径需要

            _validate_budget_limit_keys(self._budget_limits, BudgetLimits)

    @staticmethod
    def _apply_budget_limits(state: dict, overrides: dict | None) -> dict:
        """对 ``build_initial_state`` 产物覆盖 ``budget.limits``（只动评测装配层）。

        overrides 为 ``BudgetLimits`` 字段名→值的 dict（None/空 = 原样返回）；用
        model_copy 逐层拷贝，不改生产 Budget/BudgetLimits 对象与默认值。

        **键白名单校验**：未知键（如拼错的 ``max_llm_call``）先按
        ``BudgetLimits.model_fields`` 校验并抛 ValueError —— pydantic v2 的
        ``model_copy(update=…)`` 对未知键静默挂属性而不生效，若放任会令预算档位
        实验静默以生产默认 10 跑；装配层失败要响亮。
        """
        if not overrides:
            return state
        from pra.domain.models import BudgetLimits  # 惰性 import：仅覆盖路径需要

        _validate_budget_limit_keys(overrides, BudgetLimits)
        budget = state.get("budget")
        if budget is None:
            return state  # 防御：初始 state 恒有 budget，缺省不覆盖
        limits = budget.limits
        updated = limits.model_copy(update=dict(overrides))
        state["budget"] = budget.model_copy(update={"limits": updated})
        return state

    async def run(self, case: EvalCase, ctx: EvalContext) -> EvalRecord:
        from pra.agent.guardrails import llm_shell

        tools = None
        if ctx.tool_world == "eval":
            tools = make_eval_world_tools()
        elif ctx.tool_world == "rag":
            # RAG 世界（R-4/R-6）：先例/政策检索注入真实 RAG 索引，检索模式可切换
            # （EvalContext.rag_mode，默认 None → hybrid）、索引后端可切换
            # （EvalContext.rag_backend，默认 "local" → 装配不变）—— 评测默认路径不动。
            tools = make_rag_world_tools(
                mode=ctx.rag_mode or "hybrid",
                backend=ctx.rag_backend,
                backend_options=ctx.rag_backend_options,
            )
        if tools is not None and self._allowed_tools is not None:
            # 工具注册层裁剪（只保留允许子集；连同 plan 侧裁剪 = 完整装配裁剪）
            tools = [t for t in tools if t.name in self._allowed_tools]
        if self._llm is not None:
            # Phase 3 real 模式：调用方注入对象即为 LLM 后端（构造与 tools 配置由
            # 调用方负责，本层不额外处理）；工具仍按 ctx.tool_world 装配（同上）。
            backend = self._llm
        else:
            # 确定性审查员桩（Phase 1/2 默认；同 case 同 ctx → 同输出，可重放）
            backend = EvalScriptedLLMBackend(
                allowed_tools=set(self._allowed_tools) if self._allowed_tools is not None else None,
                evidence_thresholds=ctx.evidence_thresholds,
            )
        graph: CompiledStateGraph = build_agent_graph(
            tools=tools,
            checkpointer=make_memory_checkpointer(),
            llm=backend,
        )
        try:
            initial_state = build_initial_state(case.input)
            if self._budget_limits is not None:
                # 评测侧预算覆盖（real 放宽 max_latency_ms / 调 max_llm_calls 档用；
                # 键 = BudgetLimits 字段名；None 分支原样返回）
                initial_state = self._apply_budget_limits(initial_state, self._budget_limits)
            # Root trace：每案一条，trace_id 确定性 uuid5
            # （含 LLM 后端名 —— scripted 对照臂与 real 臂各自成 trace）；
            # **不 per-case flush**（320 次太慢）—— 由评测入口整轮结束后
            # ``tracing.flush_tracer()`` 统一刷出（S5 CLI 收尾调用）。
            root_ctx = _root_trace_context(
                case, ctx, initial_state, backend_name=getattr(backend, "name", "unknown")
            )
            with get_tracer().trace_root(root_ctx) as root:
                final_state = await graph.ainvoke(
                    initial_state,
                    {"configurable": {"thread_id": f"eval-agent-{case.eval_case_id}"}},
                )
                _decision = final_state.get("decision")
                if _decision is not None:
                    root.update(
                        output={
                            "decision": _decision.decision.value,
                            "risk_level": _decision.risk_level.value,
                        }
                    )
        finally:
            llm_shell.set_llm_backend(None)  # 恢复默认桩（本后端仅评测期生效）

        decision: ReviewDecision | None = final_state.get("decision")
        if decision is None:
            raise RuntimeError(
                f"Agent 图执行完成但终态缺少 decision（eval_case_id={case.eval_case_id}）"
            )
        return self._transcribe(case, final_state, decision)

    # 终态 → EvalRecord 转录（只读 ReviewDecision / state 摘要）

    @staticmethod
    def _transcribe(case: EvalCase, final_state: dict, decision: ReviewDecision) -> EvalRecord:
        history = final_state.get("tool_call_history") or []
        ok_calls = [r for r in history if isinstance(r, dict) and r.get("status") == "ok"]
        tool_names = list(dict.fromkeys(str(r.get("tool")) for r in ok_calls))  # 去重保序

        evidence = [
            {
                "type": e.type,
                "source": e.source,
                "value": e.value,
                "weight": e.weight,
                "ref_id": e.ref_id,
                "extra": dict(e.extra or {}),
            }
            for e in decision.evidence
        ]
        budget = decision.budget_used
        trace = [
            {
                "id": h.id,
                "status": h.status.value,
                "prior": h.prior,
                "posterior": h.posterior,
                "statement": h.statement,
            }
            for h in decision.hypothesis_trace
        ]
        return EvalRecord(
            eval_case_id=case.eval_case_id,
            scheme="agent",
            decision=decision.decision.value,
            risk_level=decision.risk_level.value,
            risk_type=[t.value for t in decision.risk_type],
            decision_confidence=decision.decision_confidence,
            evidence=evidence,
            policy=list(decision.policy),
            tool_calls_actual=tool_names,
            cost={
                "llm_calls": budget.llm_calls,
                "tool_calls": budget.tool_calls,
                "tokens": budget.tokens,
            },
            detail={
                "overrides": list(decision.overrides),
                # P2-16：R3 命中时附"哪一维先撞限"（LLM_CALLS/TOOL_CALLS/TOKENS/
                # LATENCY）供真实跑分归因 —— 用快照冻结值重算（确定性），不改
                # R3_BUDGET_EXHAUSTED 码字面/语义。真实跑分 tokens 口径 =
                # usage.total_tokens（input+output、含缓存命中、失败重试全额累计），
                # 故 TOKENS 是真实第二截胡源，需与 llm_calls 维度区分（见
                # llm_shell/litellm_backend docstring 口径注记）。
                "budget_hit_dim": _budget_hit_dim_from_snapshot(budget),
                "hypothesis_trace": trace,
                "tool_history_count": len(history),
            },
        )
