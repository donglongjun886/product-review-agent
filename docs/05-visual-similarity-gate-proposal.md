# 视觉相似类假设的证据存在性 Gate 约束 —— 设计提案（docs/05-visual-similarity-gate-proposal.md v1）

> **状态声明（重要）**：本文为设计提案，**V-1 ~ V-11 已拍板并实施落地**（见下方「实施状态注记」；
> 落地前全文未实施 / 待拍板）。本文为纯设计文档（不含实现代码），保留作拍板依据与设计
> rationale；实现代码位于 `src/pra/agent/guardrails/gate.py` 等。背景：Phase 3 real
> LLM 首次跑分（v1 35 案）暴露唯一
> 误杀 EC_0007（truth=PASS 被判 REJECT，见 docs/02-evaluation.md §8「Phase 3 实现状态」块）；
> 并行已做 prompt 层缓解，本提案 = **Gate 侧确定性兜底的准备设计**：R3 abstention 扩展 ——
> 「外观/视觉相似类假设被判 SUPPORTED 但证据集不含视觉证据 → 确定性拦截」。与
> docs/03-decisions.md 的 T-* / docs/02-evaluation.md 的 P-* 编号惯例对齐，本文待拍板项编为
> **V-1 ~ V-11**（见 §5）。

> **实施状态注记**：V-1 ~ V-11 已全部拍板并落地（与 §5 各行推荐结论一致）：Gate 侧新增
> `R3_VISUAL_CLAIM_UNSUPPORTED` abstention（gate.py —— 关键词谓词
> `visual_claim_unsupported` / 存在性谓词 `visual_evidence_present` / 本地关键词表
> `VISUAL_CLAIM_MARKERS`，V-5 与 llm_prompts 同步维护）+ 确定性单测
> （tests/test_gate_visual.py）+ v1 回归基线重录 + docs 00/01/03 定向同步。
> 实现 commit：`feat(agent/gate): ②b Gate 视觉兜底 —— R3_VISUAL_CLAIM_UNSUPPORTED`（2026-09-09，
> 与 B-1 收敛 prompt、B-2 评测 --llm-budget 同轮落地；SHA 见 job-tracker content-governance/project-status.md §13.6）。

---

## 1. 动机与现状

### 1.1 EC_0007 归因（Gate 视角）

- 案例：无任何 `IMAGE_SIMILARITY` 证据（无视觉测量）。LLM 仅凭标题「小白鞋」+「高度相似→
  REJECT」的同类先例，把「外观与经典小白鞋高度相似」假设脑补为 SUPPORTED（posterior=0.9）
  → 提案 REJECT → 自动放行。
- 该 REJECT 提案通过了 `reject_gate`（gate.py:220-241）**全部五个条件**，Gate 未拦：
  ① 高优先 SUPPORTED 成立（LLM 判的）；② `evidence_sufficient` 通过——假设的 `evidence_for`
  非空（引用了 `CASE_PRECEDENT`）且无关键 Tool 失败；③ `_has_citable`（gate.py:81-83）通过——
  `CASE_PRECEDENT` 带 `ref_id` 属可引用集 `CITABLE_TYPES={CASE_PRECEDENT, POLICY_REF}`
  （gate.py:62）；④ dc≥0.7 通过——`finalize_decision_confidence`（gate.py:244-265）里
  `citation=1.0`（有先例）+ top=0.9，仅先例一条证据时 0.45×0.9 + 0.25×1/8 + 0.20×1 + 0.10
  ≈ 0.74 ≥ 0.7（随证据增多单调不降）；
  ⑤ 无关键矛盾（无强相似则 `contradiction_detect` 恒 False）。
- 结论：Gate 缺的不是「强度」检查而是「**声称维度与证据维度的一致性**」检查——先例能证明
  「同类曾被拒」的历史事实，**替代不了**「本商品与某品牌款视觉相似」的直接测量。

### 1.2 现有确定性检查查了什么、缺什么

| 现有谓词（gate.py 行） | 查什么 | 对 EC_0007 的作用 |
|---|---|---|
| `_has_citable`（81-83） | ∃ POLICY_REF/CASE_PRECEDENT 带 ref_id | 通过（被先例满足）——只查「可引用」，不查「引用能否支撑声称的维度」 |
| `_strong_similarity`（86-88） | ∃ IMAGE_SIMILARITY 且 weight≥0.85（`_SIM_STRONG`，78 行） | False，但**只被 `contradiction_detect`（119-136）与 `finalize_risk_type`（287-307）消费**；REJECT Gate 本身不要求任何视觉证据存在 |
| `policy_indeterminate`（139-161） | SUPPORTED 高优先 ∧ 无带 ref_id 的 POLICY_REF | **未触发**——EC_0007 能走到自动 REJECT，说明它未命中（命中即 R3_POLICY_UNCERTAIN 先转人工）；「有政策定性」与「有视觉测量」是两回事 |
| `indistinguishable_hypotheses`（164-178） | ≥2 个高优先 SUPPORTED 且 evidence_for 集合相同 | False（单假设） |
| `evidence_sufficient`（186-197） | SUPPORTED 高优先假设 evidence_for 非空 | 通过（被 LLM 塞的先例引用满足） |

> 注：EC_0007 未触发 abstention 清单（`_abstention_codes`，gate.py:361-382）——尤其
> `policy_indeterminate` 未命中 ⇒ 证据链存在带 ref_id 的 POLICY_REF（否则 R3_POLICY_UNCERTAIN
> 会在 Gate 前把提案转人工）；REJECT Gate ③ `_has_citable` 又被 CASE_PRECEDENT 满足。即
> 「政策定性 + 先例引用」齐备，唯缺「与声称维度一致的视觉测量」——这是现有五检查（存在性 /
> 可引用性 / 充分性 / 置信 / 矛盾）的共同盲区：没有一条核对「声称的维度」与「证据的维度」。
> 补充口径：上述 POLICY_REF / CASE_PRECEDENT 均为 **real 运行中经 RAG 检索动态产生**——
> EC_0007 静态 case 记录（eval_data/v1/cases_v1.jsonl）`expected.evidence=[]`、
> `applicable_policy=[]`，无任何预置证据/政策；推演只说明 Gate 放行时刻链上状态，**不代表
> case 文件自带政策引用**。

---

## 2. 概念定义（本提案核心口径，全部待拍板）

### 2.1 risk_type 口径：新增 `VISUAL_SIMILARITY` 枚举 vs 复用现有枚举

现状（models.py:67-77）：`RiskType` 仅 4 成员 `POTENTIAL_IP_RISK / EVASION_PATTERN /
FALSE_CLAIM / FIELD_CONFLICT`，**无 `VISUAL_SIMILARITY`**。案件级 `risk_type` 来源：
① LLM 提案 `DecisionProposal.risk_type`（schemas.py:136-148，受控词表内取值）；② 派生兜底
`finalize_risk_type`（gate.py:287-307，∃ 强 IMAGE_SIMILARITY → POTENTIAL_IP_RISK）。
注意：**单条 Hypothesis 无 risk_type 字段**（models.py:198-218 仅 statement/prior/posterior/
status/evidence_*）——风险类型是案件级口径，不与假设一一绑定。

**建议：不新增 `VISUAL_SIMILARITY` 枚举成员（理由）**
1. **语义错位**：RiskType 是「业务违规类别」（00 §7.3：评测标签/政策库元数据/决策输出三方共用
   的可统计词表）；「外观相似」在本案是**假设的取证维度/声称渠道**，不是新的违规类别——政策库里
   视觉模仿类（POLICY_1.3~1.6，policies.json）`risk_type` 元数据全部标 `POTENTIAL_IP_RISK`。
2. **连锁成本高**：新增成员需同步 00 §7.3 词表、models.py 枚举、schemas.py 校验、RAG 政策元数据、
   v1/v2 评测集 `risk_type` 标注与预期、report 按类型指标分层、回归基线——与本提案「最小 Gate
   兜底」目标不符。若将来 metrics 要按维度拆分，用派生口径即可，不必进 schema 词表。
3. **现有词表可覆盖**：`finalize_risk_type` 已把强 IMAGE_SIMILARITY 派生为 POTENTIAL_IP_RISK，
   视觉类是它的最主要子类。
- **反方保留（V-4）**：若产品需要「纯视觉相似（无品牌词/无 logo/无商家史）」作为独立风险类别以
  支撑专门政策/人工队列，则新增合理——但须一次性完成 00 §7.3 与全链口径连锁更新。

### 2.2 「外观/视觉维度」的机器判定口径（二选一或分层）

- 现状：假设只有自由文本 `statement`，无结构化维度；假设维度 ≠ 案件 risk_type（§2.1）。
- **方案 A（结构化维度）**：给 `HypothesisProposal`/`Hypothesis` 加受控 `risk_dimension`
  （含 VISUAL），LLM 标注。稳，但 schema/prompt/模型三处变更，且「把兜底押在要被兜底的对象
  上」——LLM 标错维度即绕开 Gate。
- **方案 B（假设表述关键词谓词，默认方向）**：Gate 侧对 SUPPORTED 高优先假设的 `statement`
  做确定性关键词判定（如 外观/造型/鞋型/版型/廓形/配色/图案/印花/高度相似/模仿知名品牌/经典
  小白鞋…），命中即视为「视觉维度声称」。优点：零 schema 变更、纯确定性、与 prompt 的维度示例
  措辞同源（llm_prompts.py:111 显式示例「外观相似」维度）。缺点：中文自由文本措辞漂移、负向句
  误伤（如「外观与品牌款无相似」被判 SUPPORTED 的异常态）、新词漏判。
- **方案 C（分层，推荐）**：以 B 为主判据，A 作二期结构化增强；关键词白名单集中一处、随 prompt
  措辞同步维护。关键权衡：漏判代价 = 退回现状（仍可能误杀）；误判代价 = 多转人工（安全侧，比
  误杀轻），且只作用于 REJECT 侧 SUPPORTED 假设，PASS 案（全 REFUTED）不受影响（00 §7.4 口径）。
- 纪律冲突（V-7 也问）：现有纪律「确定性逻辑只读 extra/weight/ref_id，不解析人读字符串」
  （gate.py:31、03-decisions §4.2 漂移项 3）——本谓词需读假设 `statement` 自由文本，须单独拍板
  放行（statement 是断言本体而非证据摘要，且只做维度判定、不做强度判定）。

### 2.3 视觉证据类型权威清单（建议）

| 证据类型 | 出处 | 是否算「视觉相似证据」 | 口径 |
|---|---|---|---|
| `IMAGE_SIMILARITY` | image_analysis/tool.py:32 | **是（核心）** | 唯一能直接支撑「与某品牌款外观高度相似」的测量证据；extra 回填 `{similarity, strong}`（guardrails/evidence.py） |
| `IMAGE_LOGO` | image_analysis/tool.py:33 | **是（logo 类）** | weight=confidence，支撑「带品牌 logo」类声称 |
| `OCR_TEXT` | pra/tools/ocr | **否**（建议口径） | 属「图像来源**文本**证据」：能证明图内含文字（复刻字样/品牌词），无法证明鞋型/配色/版型相似；单一 OCR 不得支撑「外观高度相似」类 SUPPORTED |
| `CASE_PRECEDENT / POLICY_REF / PRODUCT_FACT / MERCHANT_HISTORY` | — | 否 | 先例=「同类曾被拒」历史、政策=定性依据、商品/商家=事实与行为史，均替代不了视觉直接测量 |

- 标题字段（title）不是图像证据——EC_0007 正是「标题 + 先例」冒充视觉证据。
- 权威清单落点建议：guardrails 常量层本地声明（对齐 `CITABLE_TYPES` 的本地声明做法，gate.py:62），
  或并入 03-decisions §5 常量总表。
- 关键词外延建议只覆盖「外观/造型/相似」语义，**不含**「字样/标题/复刻词」类文本声称，避免把
  OCR 可支撑的文本声称也拦了（V-5）。

### 2.4 与 0.70/0.85 阈值及 `_strong_similarity` 的关联

- 证据链内的 IMAGE_SIMILARITY 必然 weight≥0.70：`quality_filter`（guardrails/evidence.py）把
  <EVIDENCE_MIN_SIM(0.70) 的弱命中在入链前丢弃（03 T-11 三档语义：<0.70 不作证据 / 0.70~0.85
  普通 / ≥0.85 Strong）。故「链上 ∃ IMAGE_SIMILARITY」≈「weight≥0.70」。
- 0.85 = gate `_SIM_STRONG`（gate.py:78，镜像 EVIDENCE_STRONG）→ `_strong_similarity`
  （gate.py:86-88）与 `contradiction_detect`「高相似」分界、`finalize_risk_type` 派生
  POTENTIAL_IP_RISK、scripted 理想行为 `sim_strong` 同一档位。
- 本约束是**存在性谓词**（防「无任何视觉测量就脑补 SUPPORTED」），建议档位 **≥0.70（普通证据
  即算「有视觉证据」）**；强度差异继续由 posterior / dc / risk 派生表达。是否要求 ≥0.85 强档
  才放行 REJECT → V-3。

---

## 3. 确定性约束草案

### 3.1 检查点与谓词（示意伪码，非实现代码）

```text
VISUAL_EVIDENCE_TYPES = {IMAGE_SIMILARITY, IMAGE_LOGO}   # §2.3 建议清单

def visual_claim_unsupported(state) -> bool:             # 建议命名（草案）
    """视觉维度声称被判 SUPPORTED，但证据链无任何视觉证据 → True。"""
    vis = [h for h in high_priority(state.hypotheses)          # gate.py:106 同口径
           if h.status == SUPPORTED and _looks_visual(h.statement)]  # §2.2-B 关键词谓词
    if not vis: return False          # 干净案 / PASS 候选不拦（对齐 policy_indeterminate 纪律）
    return not any(e.type in VISUAL_EVIDENCE_TYPES for e in state.evidence)
```

### 3.2 命中后行为与原因码归属（三候选 + 推荐）

| 候选 | 落点 | 命中行为 | 评价 |
|---|---|---|---|
| **甲（推荐）** | 谓词加入 `_abstention_codes`（gate.py:361-382，紧随 R3_POLICY_UNCERTAIN / R3_HYPOTHESES_INDISTINGUISHABLE 判定）；新码 `R3_VISUAL_CLAIM_UNSUPPORTED`（草案名）声明于 R3 常量块（gate.py:67-71 相邻） | overlay 步骤 3（run_decision_overlay gate.py:436-443）→ HUMAN_REVIEW，overrides 带专属码 | 归因清晰：「非证据不足，而是声称维度与证据错配」；PASS/REJECT 提案一律被 abstention 拦（PASS 态本就不会命中，见 3.3） |
| 乙 | `reject_gate`（gate.py:220-241）插条件⑥ | REJECT 提案 → R2_REJECT_GATE_FAIL → HUMAN_REVIEW | 归因与政策/证据不足混在同一码，可审计性差 |
| 丙 | reevaluate apply / overlay 预步把该假设 SUPPORTED→UNRESOLVED「降级不可引用」 | 两 Gate 均不过 → HUMAN_REVIEW | 效果等价但**违反 gate 纯函数只读纪律**（gate.py:28-32）；假设状态所有者是 reevaluate 节点，若要降级应落在那里而非 Gate |

**推荐甲**；同时记录：该假设「在本案不可作为 REJECT 依据」只体现为转人工，不写回假设状态——
`hypothesis_trace` 保留 LLM 原判、overrides 记拦截原因，保留「谁提的 / 被谁拦」双视角审计。

### 3.3 与 dc 0.7 / R3_HYPOTHESES_INDISTINGUISHABLE / R3_POLICY_UNCERTAIN 的交互

- **dc≥0.7**：位置甲在 overlay 步骤 3 先于步骤 5 的 dc 判断（gate.py:436 vs 448），命中时无论 dc
  高低一律 HUMAN——与 R3_POLICY_UNCERTAIN 同理，属「不可自动判」清单，与置信门槛正交。
- **R3_HYPOTHESES_INDISTINGUISHABLE**：`_abstention_codes` 命中码全量收集（gate.py:361 语义），
  可并列出现；若多个互斥 SUPPORTED 假设引用同一先例，视觉谓词往往更先解释「为何这批 SUPPORTED
  不可信」（evidence_for 相同正是 INDISTINGUISHABLE 的触发面）。
- **R3_POLICY_UNCERTAIN**：互补而非重叠——它管「无政策引用」，本谓词管「声称渠道错配」。
  EC_0007 中 policy_indeterminate 未触发（链上有带 ref_id 的 POLICY_REF，否则早被转人工——
  该 POLICY_REF/CASE_PRECEDENT 均为 real 运行经 RAG 检索动态产生，静态记录无预置证据，
  见 §1.2 注），政策与先例两条引用类规则都放行，唯独无人核对声称维度 → 本谓词补齐这一环。
- **对 PASS Gate 与「REJECT 必引政策依据」语义**：只挑 SUPPORTED 高优先假设 → PASS 案（全
  REFUTED）永不命中，不改「干净商品低风险置信是正常态」（00 §7.4）。「REJECT 必引政策依据」
  语义不变（本约束不替代政策引用，也不放宽它），只是给 REJECT 侧增加一条**前置的证据一致性
  要求**——其是否构成对 00 §7.2-2「明确政策依据或高度相似先例」的语义扩展（先例须能支撑声称
  维度）→ V-6。

---

## 4. 影响面与验证

### 4.1 对确定性 scripted 行为的影响

- scripted 剧本（scripted_llm.py）里，外观类假设 SUPPORTED 的唯一路径是 `sim_strong` 成立
  （H2「参考知名品牌经典复古跑鞋设计」仅当 ∃ IMAGE_SIMILARITY weight≥0.85，:390-392；H1 REFUTED
  同源 :386-389），彼时证据链必含强 IMAGE_SIMILARITY → 谓词恒 False；H3/H4 表述（「刻意规避品牌
  识别 / 商家系统性类似上架行为」，:48-53）不含视觉关键词；队列「外观是否与某知名品牌款高度
  相似？」DONE 判定同理需 sim_strong（:419-432）。
- **预期结论：v1 35 案 / v2 320 案 scripted 重放 decisions 数组零变化**；影响集中在 real LLM
  路径（EC_0007 形态：REJECT → HUMAN_REVIEW，消误杀、FPR↓，代价是 human_review_rate 上升）。
  但改动后**仍须重跑存档**（守护未来剧本演进，如无图 conclude 路径 :256-262 时关键词假设被
  误判 SUPPORTED 的回归）。
- 重放成本：确定性桩零 API 调用，`uv run python scripts/run_evaluation.py` + `run_regression.py`
  分钟级，**低成本可重复**。

### 4.2 回归 / 口径记录要求

- 刷新 `eval_data/v1/regression_baseline.json`（35 案 decisions 数组）与
  `eval_data/v2/regression_baseline.json`（320 案三方案，digest `387a345c…`，已 git 入库；
  守护断言见 `tests/test_regression_v2.py` —— v1/v2 三方案决策序列全比对 + digest 逐字节
  重放 + `(320,42)` 数据字节锁；`scripts/run_regression.py --data v2` 手动比对路径）。
  若本约束（或任何改判定逻辑的变更）改变 scripted 决策，须用
  `run_regression.py --record --data v2` / `--data v1` 重录两条基线并同步本段 digest。
- 报告快照记录（对齐 02 §7.1「screening 行为快照」做法）：谓词/关键词表/阈值版本 + 变化 case
  清单（预期空）+ HUMAN_REVIEW 增量 + scripted 重放逐字节一致断言。
- 口径记录：本约束是**证据一致性**而非**证据强度**——转人工原因写「视觉声称无视觉证据」，
  勿与 `evidence_sufficient` 的「证据不足」混淆。

### 4.3 建议新增的确定性测试点

1. EC_0007 形态：SUPPORTED 外观假设 + CASE_PRECEDENT(ref_id) + 无视觉证据 → 谓词 True →
   HUMAN_REVIEW + `R3_VISUAL_CLAIM_UNSUPPORTED`（REJECT 提案被 abstention 拦下）。
2. 同形态 + 一条 weight=0.72 IMAGE_SIMILARITY → False（0.70 普通档即算有视觉证据，放行语义）。
3. 防御注入：weight=0.65 的 IMAGE_SIMILARITY（低于 0.70，正常被 quality_filter 丢弃）→ 谓词口径
   以 weight≥0.70 判定，避免依赖 extra.strong 时被伪造 extra 绕过（gate 只读 weight，对齐纪律）。
4. 负向句守卫：statement「外观与品牌款明显不同」被判 SUPPORTED 的异常态是否命中关键词（V-8）。
5. PASS 侧守卫：高优先全 REFUTED + 外观词只出现在 evidence_against → 不命中。
6. 空 hypotheses / evidence → False（纯函数空安全，gate.py:28-32）。
7. 与 R3_POLICY_UNCERTAIN / R3_HYPOTHESES_INDISTINGUISHABLE 并列命中 → overrides 多码全量收集。
8. scripted v1/v2 全量重放：decisions 与 regression baseline 逐字节一致。

---

## 5. 待拍板问题清单（Open Questions，V-1 ~ V-11）

| ID | 问题 | 本稿倾向（待确认） |
|---|---|---|
| V-1 | 「外观/视觉维度」判定：关键词谓词（B）还是结构化 risk_dimension（A）？statement 自由文本解析如何与「确定性不解析人读文本」纪律共存？ | B 为主 + C 分层预留；statement 例外放行 |
| V-2 | OCR_TEXT 算不算视觉相似证据？「图内复刻字样/品牌词」类文本声称是否纳入本约束？ | OCR 不算视觉相似证据；关键词外延不含字样/标题类 |
| V-3 | 视觉证据存在性阈值：≥0.70（普通档即算）vs ≥0.85（与 scripted sim_strong 完全对齐、更保守）？多证据混合口径（0.72 相似 + 先例 + 商家史）是否够支撑 REJECT？ | 存在性用 ≥0.70 |
| V-4 | 是否引入 `VISUAL_SIMILARITY` 枚举？引入则 00 §7.3 词表、政策元数据、v1/v2 标签、report 类型指标需一次性连锁更新 | 不引入；metrics 要拆分时用派生口径 |
| V-5 | prompt 修复（约束 LLM 自举）与 Gate 确定性拦截，谁为主？关键词表与 prompt 措辞的同步维护责任？ | Gate 兜底为准入边界，prompt 负责减少命中率 |
| V-6 | 本约束是否构成对「REJECT 必引政策依据 / 高度相似先例」语义的显式扩张（先例须能支撑声称的维度）？00 §7.2-2 措辞是否要补 | 语义上应扩张，文档措辞待改 |
| V-7 | 命中后行为：甲（abstention 新 R3 码，只读 state）还是丙（降级假设不可引用，写回 hypothesis_trace）？新 R3 码命名与 03-decisions T-8 overrides 词表登记、01 §7 overlay 顺序图补行？ | 甲 + 审计双视角 |
| V-8 | 关键词谓词误伤容忍度：负向句/跨语言标题/措辞漂移；被误命中转人工的代价 vs 漏判代价 | 只作用于 SUPPORTED 高优先，误判代价在安全侧 |
| V-9 | IMAGE_LOGO 单独命中是否足够支撑「logo 类」SUPPORTED？其 confidence 阈值档位？ | 是；档位沿用 0.85 或单列 |
| V-10 | 本谓词是否也约束 PASS Gate 侧（若未来假设状态机允许 SUPPORTED 存在于 PASS 案）？ | 不约束，保持现状语义 |
| V-11 | real 路径验证口径：非确定性 LLM 重跑如何对比（prompt 版本/模型/temperature 锁定）？是否只以确定性单测 + scripted 回归为准入 | 单测 + scripted 回归为准；real 抽样仅观察 |
