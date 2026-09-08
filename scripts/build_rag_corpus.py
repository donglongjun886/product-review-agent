"""build_rag_corpus.py —— 确定性生成 RAG 知识库数据（Policy KB / Case KB） + 自检。

用途（rag-implementation-plan.md R-3/R-4/R-5）：
- 生成 ``src/pra/rag/corpus/policies.json`` 与 ``cases.json``（静态、git 入库、可评审；
  幂等：同输入重跑产出逐字节一致，已存在文件将被覆盖）；
- 内置**隔离自检**（红线 R-4）：Case KB 的 case_id 与 eval_data/v1 + v2 全部
  case 标识（eval_case_id / input.case_id / lineage.seed_case_id）断言**无交集**，
  违反即非零退出 —— 防"检索到 eval GT = 评测作弊"。

数据来源（与 schema.py docstring 同口径）：
- Policy KB：**手写** 24 条款（品牌/IP、虚假宣传/功效、类目准入与标识、规避行为、
  处置与复核五族；其中 3 条 EXPIRED 历史版测版本过滤；编号有意避开 eval mock
  世界的 POLICY_3.2/4.1/5.2，避免跨世界引用混淆）；
- Case KB：手写种子 26 条（demo 剧情 P_88231/M_5512 同源**改写** + 合成各案型）
  + 程序化变体 42 条（固定 seed，随机源 ``random.Random`` 全程确定性）；
  case_id 统一前缀 ``RAG_CASE_``（与 eval 的 EC_*/EC_V2_*/CASE_EC_* 无交集）；
  剧情不与任何 eval 违规案一一对应。

用法::

    uv run python scripts/build_rag_corpus.py            # 生成 + 校验 + 隔离自检
    uv run python scripts/build_rag_corpus.py --no-write  # 只自检（不覆盖文件）

退出码 0 = 生成成功且自检通过；非零 = 自检失败（见输出）。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from pra.rag.corpus.schema import CaseCorpus, PolicyCorpus  # 导入期校验 shape

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "src" / "pra" / "rag" / "corpus"
POLICIES_FILE = CORPUS_DIR / "policies.json"
CASES_FILE = CORPUS_DIR / "cases.json"

EVAL_FILES = [
    REPO_ROOT / "eval_data" / "v1" / "cases_v1.jsonl",
    REPO_ROOT / "eval_data" / "v2" / "cases_v2.jsonl",
]

GEN_SEED = 20260908  # 程序化变体的固定 seed（勿改 —— 改了 corpus 即变）
VARIANTS_TARGET = 42  # 程序化变体条数（确定性）

# ---------------------------------------------------------------------------
# Policy KB —— 手写（来源与隔离声明见模块 docstring；编号避开 eval mock 世界 id）
# ---------------------------------------------------------------------------

_POLICY_ROWS: list[dict] = [
    # ---- 1.x 品牌 / 知识产权（POTENTIAL_IP_RISK）----
    {
        "policy_id": "POLICY_1.1", "version": 1,
        "clause_id": "POLICY_1.1_v1_c1", "title": "仿冒商品禁售（旧版）",
        "text": "禁止销售仿冒、假冒注册商标的商品；商家应确保所售商品为授权正品或自有设计，仿冒商品一经查实下架处理（旧版，已失效）。",
        "category": "全类目", "risk_type": ["POTENTIAL_IP_RISK"],
        "status": "EXPIRED", "effective_date": "2023-03-01",
    },
    {
        "policy_id": "POLICY_1.1", "version": 2,
        "clause_id": "POLICY_1.1_v2_c1", "title": "仿冒/高仿/复刻商品禁售",
        "text": "禁止销售仿冒、高仿、复刻他人品牌的商品；无品牌授权或自有版权的同款设计不得上架，判定为仿冒的强制下架并计入商家违规记录。",
        "category": "全类目", "risk_type": ["POTENTIAL_IP_RISK"],
        "status": "EFFECTIVE", "effective_date": "2024-06-01",
    },
    {
        "policy_id": "POLICY_1.1", "version": 2,
        "clause_id": "POLICY_1.1_v2_c2", "title": "仿冒认定要素",
        "text": "仿冒认定包含：使用注册商标相同或近似标识、直接复制他人商品外观或图案、冒用他人品牌背书（详情页/包装出现他人品牌信息）。",
        "category": "全类目", "risk_type": ["POTENTIAL_IP_RISK"],
        "status": "EFFECTIVE", "effective_date": "2024-06-01",
    },
    {
        "policy_id": "POLICY_1.2", "version": 1,
        "clause_id": "POLICY_1.2_v1_c1", "title": "高仿/复刻兜售词限制",
        "text": "标题与描述不得使用“原单”“尾单”“复刻”“高仿”“A货”等暗示仿冒来源的词；未经授权不得使用他人品牌名做引流。",
        "category": "全类目", "risk_type": ["POTENTIAL_IP_RISK"],
        "status": "EFFECTIVE", "effective_date": "2024-01-01",
    },
    {
        "policy_id": "POLICY_1.3", "version": 1,
        "clause_id": "POLICY_1.3_v1_c1", "title": "未授权品牌 logo/商标使用",
        "text": "未经权利人授权，商品图片、视频、包装不得出现他人注册商标或近似图形；翻转、镜像、裁剪等变形处理规避识别的同样违规。",
        "category": "全类目", "risk_type": ["POTENTIAL_IP_RISK"],
        "status": "EFFECTIVE", "effective_date": "2024-03-01",
    },
    {
        "policy_id": "POLICY_1.4", "version": 1,
        "clause_id": "POLICY_1.4_v1_c1", "title": "鞋靴外观高度模仿知名品牌",
        "text": "鞋靴类商品整体外观（鞋型、配色、细节组合）高度模仿知名品牌在售款且无品牌授权，判定为高风险违规；外观相似度不低于 0.85 为强参考信号。",
        "category": "女鞋/运动鞋", "risk_type": ["POTENTIAL_IP_RISK"],
        "status": "EFFECTIVE", "effective_date": "2024-03-01",
    },
    {
        "policy_id": "POLICY_1.5", "version": 1,
        "clause_id": "POLICY_1.5_v1_c1", "title": "箱包外观高度模仿知名品牌",
        "text": "箱包类商品（含手袋、托特包、双肩包）外观高度模仿知名品牌设计（廓形、印花、五金组合）且无授权，判定为高风险违规。",
        "category": "箱包/女包", "risk_type": ["POTENTIAL_IP_RISK"],
        "status": "EFFECTIVE", "effective_date": "2024-05-01",
    },
    {
        "policy_id": "POLICY_1.6", "version": 1,
        "clause_id": "POLICY_1.6_v1_c1", "title": "服装图案/版型模仿知名品牌",
        "text": "服装（含卫衣、外套）的图案、印花、版型组合高度模仿知名潮牌设计且无授权，判定为高风险违规。",
        "category": "服装/卫衣", "risk_type": ["POTENTIAL_IP_RISK"],
        "status": "EFFECTIVE", "effective_date": "2024-06-01",
    },
    # ---- 2.x 虚假宣传 / 功效夸大（FALSE_CLAIM）----
    {
        "policy_id": "POLICY_2.1", "version": 1,
        "clause_id": "POLICY_2.1_v1_c1", "title": "功效夸大禁止（旧版）",
        "text": "禁止功效类夸大表述，如“永久去皱”“根治脚气”等绝对化功效词（旧版，已失效）。",
        "category": "全类目", "risk_type": ["FALSE_CLAIM"],
        "status": "EXPIRED", "effective_date": "2023-05-01",
    },
    {
        "policy_id": "POLICY_2.1", "version": 2,
        "clause_id": "POLICY_2.1_v2_c1", "title": "无依据功效夸大禁止",
        "text": "禁止无依据的功效夸大宣称：如鞋类“增高 5cm”“磁疗改善循环”、服装“抑菌 90 天”、箱包“防弹”等；宣称须能出具权威检测或临床依据，否则下架处理。",
        "category": "全类目", "risk_type": ["FALSE_CLAIM"],
        "status": "EFFECTIVE", "effective_date": "2024-04-01",
    },
    {
        "policy_id": "POLICY_2.2", "version": 1,
        "clause_id": "POLICY_2.2_v1_c1", "title": "服饰功能宣称需检测依据",
        "text": "服饰商品宣称抗菌、防紫外线、发热、凉感等功能须提供对应检测报告；无报告仅以功能词描述视为虚假宣传。",
        "category": "服装/卫衣", "risk_type": ["FALSE_CLAIM"],
        "status": "EFFECTIVE", "effective_date": "2024-05-01",
    },
    {
        "policy_id": "POLICY_2.3", "version": 1,
        "clause_id": "POLICY_2.3_v1_c1", "title": "材质/成分虚假声称",
        "text": "商品宣称材质或成分与实际不符视为虚假宣传：如将 PU 标注为真皮、将混纺标注为纯棉、以次充好，按违规处置。",
        "category": "全类目", "risk_type": ["FALSE_CLAIM", "FIELD_CONFLICT"],
        "status": "EFFECTIVE", "effective_date": "2024-02-01",
    },
    # ---- 3.x 类目准入 / 材质标识（FIELD_CONFLICT）----
    {
        "policy_id": "POLICY_3.1", "version": 1,
        "clause_id": "POLICY_3.1_v1_c1", "title": "鞋类真皮标识凭证",
        "text": "鞋类商品标题/详情宣称“真皮”“头层牛皮”的，须提交材质质检或授权标识凭证；无法证明真皮却宣称的，判定虚假标识。",
        "category": "女鞋/运动鞋", "risk_type": ["FIELD_CONFLICT"],
        "status": "EFFECTIVE", "effective_date": "2024-03-01",
    },
    {
        "policy_id": "POLICY_3.3", "version": 1,
        "clause_id": "POLICY_3.3_v1_c1", "title": "箱包材质与五金标识一致",
        "text": "箱包材质（真皮/PU/帆布）与五金材质标注须与实物及质检一致；“纯皮”“原版五金”等表述需可核验凭证。",
        "category": "箱包/女包", "risk_type": ["FIELD_CONFLICT"],
        "status": "EFFECTIVE", "effective_date": "2024-05-01",
    },
    {
        "policy_id": "POLICY_3.4", "version": 1,
        "clause_id": "POLICY_3.4_v1_c1", "title": "服装面料成分标注规范",
        "text": "服装须按洗标与质检标注面料成分及含量（如棉含量百分比）；标注与实检不符判定违规。",
        "category": "服装/卫衣", "risk_type": ["FIELD_CONFLICT"],
        "status": "EFFECTIVE", "effective_date": "2024-01-01",
    },
    {
        "policy_id": "POLICY_3.5", "version": 1,
        "clause_id": "POLICY_3.5_v1_c1", "title": "类目错挂/蹭类目流量",
        "text": "商品须按真实属性挂载正确类目；借他类目流量（如普通鞋挂“增高鞋”类目、普通服饰挂“防护服”类目）判定类目错挂违规。",
        "category": "全类目", "risk_type": ["FIELD_CONFLICT"],
        "status": "EFFECTIVE", "effective_date": "2024-01-01",
    },
    # ---- 4.x 规避行为（EVASION_PATTERN）----
    {
        "policy_id": "POLICY_4.2", "version": 1,
        "clause_id": "POLICY_4.2_v1_c1", "title": "改标题重上架（旧版）",
        "text": "因违规被下架的商品不得通过修改标题后重新上架规避审核（旧版，已失效）。",
        "category": "全类目", "risk_type": ["EVASION_PATTERN"],
        "status": "EXPIRED", "effective_date": "2023-07-01",
    },
    {
        "policy_id": "POLICY_4.2", "version": 2,
        "clause_id": "POLICY_4.2_v2_c1", "title": "改标题/描述规避重上架",
        "text": "因违规被下架的商品不得通过修改标题、描述、属性后重新上架规避审核；同一商品多次（≥3 次）改标题重上架视为系统性规避，从重处理。",
        "category": "全类目", "risk_type": ["EVASION_PATTERN"],
        "status": "EFFECTIVE", "effective_date": "2024-01-15",
    },
    {
        "policy_id": "POLICY_4.3", "version": 1,
        "clause_id": "POLICY_4.3_v1_c1", "title": "换图/换主图规避",
        "text": "不得通过更换主图、详情图或翻转镜像图片等方式规避图片审核；图片与实物、标题明显不符亦属违规。",
        "category": "全类目", "risk_type": ["EVASION_PATTERN"],
        "status": "EFFECTIVE", "effective_date": "2024-02-01",
    },
    {
        "policy_id": "POLICY_4.4", "version": 1,
        "clause_id": "POLICY_4.4_v1_c1", "title": "SKU 拆分/多链接铺货规避",
        "text": "不得将同一违规商品拆分为多个 SKU 或多链接铺货，或复用违规图文到新链接以规避单链接处罚；关联链接一并处置。",
        "category": "全类目", "risk_type": ["EVASION_PATTERN"],
        "status": "EFFECTIVE", "effective_date": "2024-02-01",
    },
    {
        "policy_id": "POLICY_4.5", "version": 1,
        "clause_id": "POLICY_4.5_v1_c1", "title": "规避叠加仿冒从重处置",
        "text": "规避审核行为与品牌模仿、虚假宣传等风险叠加出现时（如高仿鞋改标题重上架），按从重档处置：直接拒绝并累计商家信用分扣减。",
        "category": "全类目", "risk_type": ["EVASION_PATTERN", "POTENTIAL_IP_RISK"],
        "status": "EFFECTIVE", "effective_date": "2024-03-01",
    },
    {
        "policy_id": "POLICY_4.6", "version": 1,
        "clause_id": "POLICY_4.6_v1_c1", "title": "规避行为认定要素",
        "text": "认定规避需满足：存在被处置记录、再次上架、信息变更但实质商品未变；单次改名不构成，商家历史（下架≥3 或改标题≥3）为系统性信号参考。",
        "category": "全类目", "risk_type": ["EVASION_PATTERN"],
        "status": "EFFECTIVE", "effective_date": "2024-03-01",
    },
    # ---- 5.x 处置与复核 ----
    {
        "policy_id": "POLICY_5.1", "version": 1,
        "clause_id": "POLICY_5.1_v1_c1", "title": "商家系统性违规处置",
        "text": "同一商家多链接、多类目系统性违规（如相似高仿商品 20+ 链接在架）触发店铺级处置；情节严重暂停上新资格。",
        "category": "全类目", "risk_type": ["EVASION_PATTERN"],
        "status": "EFFECTIVE", "effective_date": "2024-04-01",
    },
    {
        "policy_id": "POLICY_5.3", "version": 1,
        "clause_id": "POLICY_5.3_v1_c1", "title": "证据不足转人工复核",
        "text": "外观真伪等需主观判定的情形证据不足时，转人工复核；人工复核结论回流沉淀为后续同类商品判断的先例依据。",
        "category": "全类目", "risk_type": [],
        "status": "EFFECTIVE", "effective_date": "2024-01-01",
    },
]

# ---------------------------------------------------------------------------
# Case KB —— 手写种子（demo 改写 + 合成案型；不来自 eval GT）
# ---------------------------------------------------------------------------

_CASE_SEEDS: list[dict] = [
    # demo 剧情改写（P_88231 / M_5512 同源：无品牌复古跑鞋 + 强相似 + 脏商家；改写非照抄）
    {
        "category": "女鞋/运动鞋", "decision": "REJECT", "risk_level": "HIGH",
        "risk_type": ["POTENTIAL_IP_RISK", "EVASION_PATTERN"],
        "summary": "无品牌标识的女款复古跑鞋，外观高度模仿某知名品牌经典复古跑鞋（相似度 0.91，无 logo）；商家近 90 天 5 次被下架、3 次改标题重上架，系统性规避明显。",
        "key_evidence": ["image_similarity>=0.91", "merchant_removals=5", "title_relisting=3"],
        "policy_refs": ["POLICY_1.4", "POLICY_4.2", "POLICY_4.5"],
    },
    {
        "category": "箱包/女包", "decision": "REJECT", "risk_level": "HIGH",
        "risk_type": ["POTENTIAL_IP_RISK", "EVASION_PATTERN"],
        "summary": "无品牌标识的大容量托特包，廓形与印花组合高度模仿某知名品牌在售手袋（相似度 0.95）；同商家历史 7 次下架、4 次改标题重上架。",
        "key_evidence": ["image_similarity>=0.95", "merchant_removals=7", "title_relisting=4"],
        "policy_refs": ["POLICY_1.5", "POLICY_4.2"],
    },
    {
        "category": "服装/卫衣", "decision": "REJECT", "risk_level": "HIGH",
        "risk_type": ["POTENTIAL_IP_RISK", "EVASION_PATTERN"],
        "summary": "复古印花宽松卫衣，图案与版型高度模仿某知名潮牌（相似度 0.9，无授权）；商家多次换主图重上架规避审核。",
        "key_evidence": ["image_similarity>=0.9", "image_swap_relisting>=3"],
        "policy_refs": ["POLICY_1.6", "POLICY_4.3"],
    },
    # 干净自有品牌（PASS）
    {
        "category": "女鞋/运动鞋", "decision": "PASS", "risk_level": "NONE",
        "risk_type": [],
        "summary": "自有品牌轻弹缓震跑步鞋，无品牌模仿、无风险词，商家历史干净（无下架无改标题），机审与人工复核一致放行。",
        "key_evidence": ["image_similarity=0.0", "merchant_clean"],
        "policy_refs": [],
    },
    {
        "category": "箱包/女包", "decision": "PASS", "risk_level": "NONE",
        "risk_type": [],
        "summary": "自有品牌极简通勤托特包，帆布材质，无相似外观无违规词，商家信用良好，正常放行。",
        "key_evidence": ["image_similarity=0.0", "merchant_clean"],
        "policy_refs": [],
    },
    {
        "category": "服装/卫衣", "decision": "PASS", "risk_level": "NONE",
        "risk_type": [],
        "summary": "自主品牌基础款纯色卫衣，无印花无品牌关联，面料成分标注齐全，商家无历史违规，正常放行。",
        "key_evidence": ["fabric_label_ok", "merchant_clean"],
        "policy_refs": [],
    },
    # 边界案（弱相似：需交叉证据）
    {
        "category": "女鞋/运动鞋", "decision": "PASS", "risk_level": "NONE",
        "risk_type": [],
        "summary": "自有设计休闲鞋与某品牌经典款存在 0.72 弱相似（共性仅为通用板鞋元素），商家干净无系统性行为，判定不构成高度模仿，放行。",
        "key_evidence": ["image_similarity=0.72", "merchant_clean"],
        "policy_refs": ["POLICY_1.4"],
    },
    {
        "category": "箱包/女包", "decision": "HUMAN_REVIEW", "risk_level": "MEDIUM",
        "risk_type": ["POTENTIAL_IP_RISK"],
        "summary": "无品牌托特包与某品牌款相似度 0.8（弱偏中），商家历史中性（1 次下架），无其他佐证，转人工比对确认是否构成高度模仿。",
        "key_evidence": ["image_similarity=0.8", "merchant_removals=1"],
        "policy_refs": ["POLICY_1.5", "POLICY_5.3"],
    },
    {
        "category": "女鞋/运动鞋", "decision": "HUMAN_REVIEW", "risk_level": "MEDIUM",
        "risk_type": ["POTENTIAL_IP_RISK", "EVASION_PATTERN"],
        "summary": "无品牌鞋与知名款相似度 0.78（弱相似），但商家历史 4 次下架达系统性阈值，交叉证据下转人工综合判定。",
        "key_evidence": ["image_similarity=0.78", "merchant_removals=4"],
        "policy_refs": ["POLICY_1.4", "POLICY_4.6"],
    },
    # 高仿 + 规避词叠加
    {
        "category": "女鞋/运动鞋", "decision": "REJECT", "risk_level": "HIGH",
        "risk_type": ["POTENTIAL_IP_RISK", "EVASION_PATTERN"],
        "summary": "标题含“复刻”“同款”等暗示仿冒词，无品牌标识、外观与知名跑鞋高度相似（0.89）；商家 6 次改标题重上架，规避叠加仿冒从重拒绝。",
        "key_evidence": ["title_evasion_word", "image_similarity>=0.89", "title_relisting=6"],
        "policy_refs": ["POLICY_1.2", "POLICY_4.5"],
    },
    # 未授权 logo
    {
        "category": "箱包/女包", "decision": "REJECT", "risk_level": "HIGH",
        "risk_type": ["POTENTIAL_IP_RISK"],
        "summary": "包面检测到某奢侈品牌双 G 字样 logo（置信 0.9），商品无授权凭证亦非二手转售说明，判定未授权使用品牌商标。",
        "key_evidence": ["logo_detect>=0.9", "no_authorization"],
        "policy_refs": ["POLICY_1.3", "POLICY_1.1"],
    },
    {
        "category": "服装/卫衣", "decision": "REJECT", "risk_level": "HIGH",
        "risk_type": ["POTENTIAL_IP_RISK"],
        "summary": "卫衣胸前印花经镜像翻转规避识别的某品牌图形商标，图像鉴权确认来源品牌，判定未授权 logo 变形规避使用。",
        "key_evidence": ["logo_flipped_detect", "image_similarity>=0.86"],
        "policy_refs": ["POLICY_1.3"],
    },
    # 自有品牌但外观强相似（对抗案）
    {
        "category": "女鞋/运动鞋", "decision": "REJECT", "risk_level": "HIGH",
        "risk_type": ["POTENTIAL_IP_RISK"],
        "summary": "标称自有品牌“轻缓跑鞋”，但鞋型与知名品牌款相似度达 0.93；自有品牌标识不豁免高度模仿，判定高风险拒绝。",
        "key_evidence": ["image_similarity>=0.93", "own_brand_claimed"],
        "policy_refs": ["POLICY_1.4"],
    },
    {
        "category": "服装/卫衣", "decision": "REJECT", "risk_level": "HIGH",
        "risk_type": ["POTENTIAL_IP_RISK"],
        "summary": "标称自创品牌卫衣整体版型与印花构图高度复刻某潮牌（0.9），仅替换了名称，判定换名不换样的仿冒。",
        "key_evidence": ["image_similarity>=0.9", "own_brand_claimed"],
        "policy_refs": ["POLICY_1.6", "POLICY_1.1"],
    },
    # 虚假宣传 / 功效夸大
    {
        "category": "女鞋/运动鞋", "decision": "REJECT", "risk_level": "HIGH",
        "risk_type": ["FALSE_CLAIM"],
        "summary": "普通运动鞋宣称“增高 5cm”“磁疗缓解疲劳”，无任何临床或检测依据，功效夸大且涉医疗暗示，拒绝。",
        "key_evidence": ["claim_no_evidence", "medical_word"],
        "policy_refs": ["POLICY_2.1"],
    },
    {
        "category": "服装/卫衣", "decision": "REJECT", "risk_level": "MEDIUM",
        "risk_type": ["FALSE_CLAIM"],
        "summary": "卫衣宣称“抑菌 90 天”“自发热”但无法提供对应检测报告，功能宣称无依据，判定虚假宣传。",
        "key_evidence": ["claim_no_report"],
        "policy_refs": ["POLICY_2.2"],
    },
    {
        "category": "箱包/女包", "decision": "HUMAN_REVIEW", "risk_level": "MEDIUM",
        "risk_type": ["FALSE_CLAIM"],
        "summary": "背包宣称“军工级防弹面料”且商家无法出具报告，宣传夸大属实但需人工确认材质与话术定性后处置。",
        "key_evidence": ["claim_no_report", "extreme_word"],
        "policy_refs": ["POLICY_2.1", "POLICY_5.3"],
    },
    # 材质虚假 / 标识冲突
    {
        "category": "女鞋/运动鞋", "decision": "REJECT", "risk_level": "MEDIUM",
        "risk_type": ["FIELD_CONFLICT", "FALSE_CLAIM"],
        "summary": "详情页宣称“头层牛皮”的鞋，质检实为 PU 合成革；材质标识与实物冲突且构成以次充好。",
        "key_evidence": ["material_conflict_leather_pu"],
        "policy_refs": ["POLICY_3.1", "POLICY_2.3"],
    },
    {
        "category": "服装/卫衣", "decision": "REJECT", "risk_level": "MEDIUM",
        "risk_type": ["FIELD_CONFLICT"],
        "summary": "卫衣洗标标注棉含量 100%，实检棉 60% 聚酯 40%；面料成分标注与实检不符。",
        "key_evidence": ["fabric_conflict_cotton_60"],
        "policy_refs": ["POLICY_3.4"],
    },
    {
        "category": "女鞋/运动鞋", "decision": "PASS", "risk_level": "NONE",
        "risk_type": [],
        "summary": "商品如实标注 PU 材质且提供质检凭证，无功效宣称无品牌模仿，商家干净，正常放行。",
        "key_evidence": ["material_label_ok", "merchant_clean"],
        "policy_refs": [],
    },
    # 规避（纯规避/类目错挂）
    {
        "category": "女鞋/运动鞋", "decision": "REJECT", "risk_level": "MEDIUM",
        "risk_type": ["EVASION_PATTERN"],
        "summary": "普通休闲鞋此前因违规被下架，商家连续 3 次仅改标题（加空格/换字序）后重新上架，实质商品未变，判定系统性规避。",
        "key_evidence": ["title_relisting=3", "relisting_after_removal"],
        "policy_refs": ["POLICY_4.2", "POLICY_4.6"],
    },
    {
        "category": "服装/卫衣", "decision": "REJECT", "risk_level": "MEDIUM",
        "risk_type": ["EVASION_PATTERN"],
        "summary": "下架卫衣通过更换主图（正反镜像）后重新上架，规避图片审核，判定换图规避。",
        "key_evidence": ["image_swap_relisting=2", "mirror_image"],
        "policy_refs": ["POLICY_4.3"],
    },
    {
        "category": "箱包/女包", "decision": "REJECT", "risk_level": "MEDIUM",
        "risk_type": ["EVASION_PATTERN"],
        "summary": "同一违规托特包拆分为 3 个不同 SKU 链接分别上架，图文复用规避单链接处罚，关联链接一并下架。",
        "key_evidence": ["sku_split=3", "multi_link_relisting"],
        "policy_refs": ["POLICY_4.4"],
    },
    {
        "category": "女鞋/运动鞋", "decision": "REJECT", "risk_level": "LOW",
        "risk_type": ["FIELD_CONFLICT"],
        "summary": "普通休闲鞋挂载“增高鞋”类目获取错误流量，商品本身无违规但类目错挂，勒令改挂正确类目。",
        "key_evidence": ["category_misplaced"],
        "policy_refs": ["POLICY_3.5"],
    },
    {
        "category": "箱包/女包", "decision": "HUMAN_REVIEW", "risk_level": "LOW",
        "risk_type": ["EVASION_PATTERN"],
        "summary": "商家 2 次换标题重上架未达系统性阈值（<3），商品本身干净，单次改名不构成规避，转人工复核商家整体历史。",
        "key_evidence": ["title_relisting=2"],
        "policy_refs": ["POLICY_4.6", "POLICY_5.3"],
    },
]

# ---------------------------------------------------------------------------
# Case KB —— 程序化变体（固定 seed；文本模板按案型组装，天然与 eval GT 剧情不同源）
# ---------------------------------------------------------------------------

_CATS = ("女鞋/运动鞋", "箱包/女包", "服装/卫衣")

_GOODS = {
    "女鞋/运动鞋": ("复古跑鞋", "板鞋", "休闲鞋", "缓震跑步鞋"),
    "箱包/女包": ("托特包", "双肩包", "斜挎小方包", "通勤手提包"),
    "服装/卫衣": ("宽松卫衣", "连帽卫衣", "印花外套", "基础款卫衣"),
}


def _gen_variants(seed: int, target: int) -> list[dict]:
    """程序化变体：按案型模板确定性组装（random.Random(seed) —— 跨进程可重放）。"""
    rng = random.Random(seed)
    rows: list[dict] = []
    pattern_idx = 0
    while len(rows) < target:
        pattern = pattern_idx % 5
        pattern_idx += 1
        category = _CATS[pattern_idx % len(_CATS)]
        goods = _GOODS[category]
        item = goods[rng.randrange(len(goods))]

        if pattern == 0:  # IP 强模仿 + 规避史
            sim = round(rng.uniform(0.85, 0.97), 2)
            rem = rng.randint(3, 9)
            rel = rng.randint(3, 6)
            risks = ["POTENTIAL_IP_RISK", "EVASION_PATTERN"]
            summary = (
                f"无品牌标识的{item}，外观高度模仿某知名品牌款（相似度 {sim}），"
                f"商家近 90 天 {rem} 次下架、{rel} 次改标题重上架"
            )
            rows.append({
                "category": category, "decision": "REJECT", "risk_level": "HIGH",
                "risk_type": risks, "summary": summary,
                "key_evidence": [f"image_similarity>={sim}", f"merchant_removals={rem}"],
                "policy_refs": ["POLICY_1.4", "POLICY_1.5", "POLICY_1.6", "POLICY_4.2"],
            })
        elif pattern == 1:  # 弱相似边界：按商家历史分 PASS/HUMAN
            sim = round(rng.uniform(0.7, 0.84), 2)
            rem = rng.randint(0, 4)
            if rem >= 3:
                decision, level, risks = "HUMAN_REVIEW", "MEDIUM", ["POTENTIAL_IP_RISK"]
                verdict = f"商家历史 {rem} 次下架达系统性参考线，转人工综合判定"
            else:
                decision, level, risks = "PASS", "NONE", []
                verdict = "商家历史干净，判定不构成高度模仿，放行"
            summary = f"自有设计的{item}与某品牌款相似度 {sim}（弱相似），{verdict}"
            rows.append({
                "category": category, "decision": decision, "risk_level": level,
                "risk_type": risks, "summary": summary,
                "key_evidence": [f"image_similarity={sim}", f"merchant_removals={rem}"],
                "policy_refs": ["POLICY_1.4", "POLICY_1.5", "POLICY_1.6", "POLICY_5.3"],
            })
        elif pattern == 2:  # 虚假/功效宣称（无报告）
            claim = rng.choice([
                "增高 3cm", "抗菌 90 天", "自发热保暖", "磁疗促进循环", "防紫外线 50+",
            ])
            summary = f"{item}宣称“{claim}”但无任何检测或临床依据，功效夸大"
            rows.append({
                "category": category, "decision": "REJECT", "risk_level": "MEDIUM",
                "risk_type": ["FALSE_CLAIM"], "summary": summary,
                "key_evidence": ["claim_no_report"],
                "policy_refs": ["POLICY_2.1", "POLICY_2.2"],
            })
        elif pattern == 3:  # 干净自有品牌（PASS）
            summary = f"自有品牌{item}，无相似外观、无风险词、无规避史，商家信用良好，正常放行"
            rows.append({
                "category": category, "decision": "PASS", "risk_level": "NONE",
                "risk_type": [], "summary": summary,
                "key_evidence": ["image_similarity=0.0", "merchant_clean"],
                "policy_refs": [],
            })
        else:  # 纯规避（商品干净）
            rem = rng.randint(3, 7)
            summary = f"{item}曾因轻微违规被下架，商家 {rem} 次改标题/换图重上架，实质商品未变，判定系统性规避"
            rows.append({
                "category": category, "decision": "REJECT", "risk_level": "MEDIUM",
                "risk_type": ["EVASION_PATTERN"], "summary": summary,
                "key_evidence": [f"title_relisting={rem}", "relisting_after_removal"],
                "policy_refs": ["POLICY_4.2", "POLICY_4.3", "POLICY_4.6"],
            })
    return rows


# ---------------------------------------------------------------------------
# 装配 + 自检
# ---------------------------------------------------------------------------


def _finalize_cases(curated: list[dict], variants: list[dict]) -> list[dict]:
    """给每条先例赋 case_id（RAG_CASE_%04d，稳定序：curated 在前、variants 在后）。"""
    rows = list(curated) + list(variants)
    out: list[dict] = []
    seen: set[str] = set()
    for i, row in enumerate(rows, start=1):
        cid = f"RAG_CASE_{i:04d}"
        if cid in seen:
            raise RuntimeError(f"case_id 重复: {cid}")
        seen.add(cid)
        out.append({"case_id": cid, **row})
    return out


def _eval_case_ids() -> set[str]:
    """eval_data/v1 + v2 全部 case 标识（红线 R-4 自检用）。"""
    ids: set[str] = set()
    for path in EVAL_FILES:
        if not path.exists():
            continue
        for lineno, line in enumerate(path.open(encoding="utf-8"), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError as exc:
                raise SystemExit(f"eval 数据解析失败 {path}:{lineno}: {exc}")
            ids.add(str(obj.get("eval_case_id", "")))
            inp = obj.get("input") or {}
            ids.add(str(inp.get("case_id", "")))
            lineage = obj.get("lineage") or {}
            if lineage.get("seed_case_id"):
                ids.add(str(lineage["seed_case_id"]))
    return {i for i in ids if i}


def _isolation_check(case_ids: set[str]) -> tuple[bool, set[str]]:
    """Case KB case_id ∩ eval case 标识 → (通过?, 交集)。"""
    eval_ids = _eval_case_ids()
    overlap = case_ids & eval_ids
    return not overlap, overlap


def _write(envelope: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def build() -> tuple[dict, dict]:
    """组装 policies / cases 信封（含 meta 来源与隔离声明）。"""
    policies = list(_POLICY_ROWS)
    cases = _finalize_cases(list(_CASE_SEEDS), _gen_variants(GEN_SEED, VARIANTS_TARGET))

    policy_meta = {
        "source": "手写（电商平台商品内容治理常见政策为蓝本；非 eval 世界种子复制）",
        "isolation_declaration": "Policy KB 独立于 eval mock 世界；编号避开 eval 的 POLICY_3.2/4.1/5.2，避免跨世界引用混淆",
        "generated_by": "scripts/build_rag_corpus.py",
        "counts": {"clauses": len(policies),
                   "effective": sum(1 for r in policies if r["status"] == "EFFECTIVE"),
                   "expired": sum(1 for r in policies if r["status"] == "EXPIRED")},
    }
    case_meta = {
        "source": "手写种子（demo P_88231/M_5512 剧情改写 + 合成各案型） + 程序化变体（固定 seed）",
        "isolation_declaration": "红线 R-4：Case KB 与 eval_data/v1、v2 的 ground-truth 严格隔离 —— "
                                 "case_id 统一 RAG_CASE_ 前缀（与 EC_*/EC_V2_*/CASE_EC_*/CASE_EC_V2_* 无交集），"
                                 "剧情不与任何 eval 违规案一一对应（防检索到 GT = 评测作弊）",
        "generated_by": "scripts/build_rag_corpus.py",
        "seed": GEN_SEED,
        "counts": {"total": len(cases), "curated": len(_CASE_SEEDS),
                   "variants": len(cases) - len(_CASE_SEEDS)},
    }
    return (
        {"meta": policy_meta, "policies": policies},
        {"meta": case_meta, "cases": cases},
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG corpus 确定性生成 + 隔离自检")
    parser.add_argument("--no-write", action="store_true", help="只跑校验/自检，不写文件")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    policy_env, case_env = build()

    # 1) schema 强校验（Pydantic 信封模型）
    PolicyCorpus.model_validate(policy_env)
    CaseCorpus.model_validate(case_env)

    # 2) 隔离自检（红线 R-4）
    case_ids = {r["case_id"] for r in case_env["cases"]}
    ok, overlap = _isolation_check(case_ids)

    print("RAG corpus 自检")
    print(f"  Policy KB : {policy_env['meta']['counts']['clauses']} 条款"
          f"（EFFECTIVE {policy_env['meta']['counts']['effective']} / "
          f"EXPIRED {policy_env['meta']['counts']['expired']}）")
    print(f"  Case KB   : {case_env['meta']['counts']['total']} 条"
          f"（手写 {case_env['meta']['counts']['curated']} + 变体 {case_env['meta']['counts']['variants']}）")
    if ok:
        print("  隔离自检   : PASS（Case KB case_id 与 eval_data/v1+v2 全部 case 标识无交集）")
    else:
        print(f"  隔离自检   : FAIL —— 交集 {sorted(overlap)[:10]}", file=sys.stderr)

    if args.no_write:
        return 0 if ok else 1

    _write(policy_env, POLICIES_FILE)
    _write(case_env, CASES_FILE)
    print(f"  已写入    : {POLICIES_FILE}")
    print(f"  已写入    : {CASES_FILE}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
