# T-1~T-12 决策记录（v2 —— 含复核修订）

> **v2 修订说明（本轮）**：① 决策机制从"单一 confidence<0.7 → HUMAN_REVIEW"升级为 **Decision Gate 口径**
> （decision_confidence 安全门槛与 risk 分离；PASS/REJECT/HUMAN_REVIEW 三个 Gate，见 T-4）；② T-7 补
> "Budget 是 Guardrail 非目标 + Budget Utilization 指标"；③ T-11 补三档语义（<0.70 不作证据 /
> 0.70~0.85 普通 / ≥0.85 Strong）+ threshold sweep；④ T-8/D 确认 DECIDED 为图唯一终态；⑤ 命名统一：
> 相似度强阈值代码名 **`EVIDENCE_STRONG`**（本表 v1 旧名 `SIM_HIGH_CONTRADICT`，同值同义）；⑥ 代码
> `BudgetLimits` 默认值按 T-7 修订为 10/15/40000/30000。v1 中"[B] 待定 / 🔧 未改代码"等表述为当时状态，
> 已随决策与落地推进更新。

> 本文档对《01-agent-loop.md》第 9 章列出的 12 项待定项（T-1~T-12）做**最终定稿**，并核查待定项取值与现有实现
> （`src/pra/domain/models.py`、`src/pra/agent/state.py`、`src/pra/tools/base.py`、`src/pra/tools/image_analysis/tool.py`）的一致性。
>
> **判定标准**：取值若改变**对外口径或业务语义**（成本上界、误伤/漏放边界）→ 归 `[B]` 需评审确认；
> 若只是内部实现参数/机制、对业务语义无影响 → 归 `[A]` 可直接落地。
>
> **效力与边界**：本表自生效起作为 graph/domain 实现的**参数与 schema 修订权威依据**；《01-agent-loop.md》与现有代码
> 凡与本表冲突处，以本表为准（01 文档已在本轮随本表同步修订）。
>
> **符号约定**：`[A]` = 已定，直接按最终值落地；`[B]` = 需评审确认（选项见 §3）；`[已拍板]` = B 类已选定；
> 🔧 = 涉及代码/文档修订的落地动作。

---

## 1. 拍板总览表

| ID | 待定项 | 分类 | 最终取值（[A]） / 待拍板（[B]） |
|---|---|---|---|
| T-1 | prior 初始值来源 + 高优先阈值 | [A] | hypothesize LLM 输出 prior（0..1，不归一化）；确定性钳制；必须含 ≥1 条低风险假设；`HIGH_PRIOR_THRESHOLD=0.3`（仅用于 PASS 门控与 prompt 强调，**不用于收敛判定**，见 T-4 修正） |
| T-2 | 假设/队列/每轮工具数量上限 | [A] | `MAX_HYPOTHESES=5`、`MAX_QUEUE=8`、`MAX_TOOLS_PER_PLAN=3`，全部配置化 |
| T-3 | HypothesisStatus 词表 | [A]（含一致性冲突，见 §4.1） | `PENDING / SUPPORTED / REFUTED / UNRESOLVED`；🔧 代码 `UNVERIFIED` → 改 `UNRESOLVED` |
| T-4 | confidence（→ decision_confidence 口径）+ 矛盾/收敛确定性定义 | [A]（v2 修订，见 §2.4） | 区分 **decision_confidence**（自动决策安全门槛）与 **risk_level/risk confidence**；三个 **Decision Gate**（PASS/REJECT/HUMAN_REVIEW，§2.4）；确定性公式产出 decision_confidence（落库 `ReviewDecision.decision_confidence`，O-7 已改名）；abstention 阈值 0.7 **仅约束 REJECT Gate**；PASS 走 Gate 判定 |
| T-5 | Evidence.weight 来源 | [A] | 工具转换器写默认权重（5.1~5.6 各工具列），reevaluate 不改 weight；weight 仅供审计/展示，不参与 v1 任何公式 |
| T-6 | hypothesize 是否可重跑 | [A] | 只入口执行 1 次；运行中新假设走 `reevaluate.new_hypotheses`（追加，PENDING）；不加 `decide→hypothesize` 边 |
| T-7 | 预算阈值与走查余量 | [已拍板] | **B：10 / 15 / 40000 / 30000** —— Guardrail 上界非目标，正常案件明显低于上限；v2 补：Trace 记录四组占用率、Evaluation 增 **Budget Utilization** 指标（见 §3.1 拍板结果） |
| T-8 | 运行终态语义 + decision.overrides | [A]（v2 复核 D） | **DECIDED 为图唯一终态**；PASS/REJECT/HUMAN_REVIEW 是 decision 取值而非图终态；预算耗尽/工具失败/降级经 `overrides` 记录，不新增图终态（见 §2.8） |
| T-9 | 图内通道字段去留 | [A]（代码已落地） | `pending_tool_calls / degraded / failures` 已补进 `AgentState`（state.py）；`run_id/case_id` → thread、`status` → DB |
| T-10 | risk_level 词表与映射 | [A]（v2 补 E） | 采用代码 `NONE/LOW/MEDIUM/HIGH`（PASS→NONE）；**risk_level ≠ decision**：HIGH risk + 证据不足 = HUMAN_REVIEW，而非 REJECT（见 §2.9） |
| T-11 | ImageAnalysis 相似度阈值 | [已拍板] | **B：`EVIDENCE_MIN_SIM=0.70` / `EVIDENCE_STRONG=0.85`**（代码常量名；v1 旧名 SIM_HIGH_CONTRADICT 同值同义）；v2 补三档语义 + threshold sweep（见 §3.2 拍板结果） |
| T-12 | 字段冲突（FIELD_CONFLICT）检测归属 | [A] | 二期（guardrails 确定性检测器）；v1 不新增确定性检测，冲突由 reevaluate/decide 的 LLM 综合（词表字段保留） |

---

## 2. [A] 已定项明细（10 项）

### 2.1 T-1 — prior 来源与高优先阈值

- **最终取值**：
  1. prior 由 **hypothesize 的 LLM 结构化输出**给出（0..1）；decide overlay 侧的确定性代码只做**范围钳制**（越界 clamp 并记 failure），**不做归一化**（假设集不必和为 1，先验是"独立怀疑度"而非概率分布）。
  2. hypothesize 输出约束：必须包含 ≥1 条**低风险/无违规**假设（作为可证伪对象，PASS Gate 判定前提，见 §2.4(b)）。
  3. `HIGH_PRIOR_THRESHOLD = 0.3`（配置常量），仅用于两处：PASS/REJECT Gate 的"高优先"口径（§2.4(b)）与 plan/reevaluate 的 **prompt 强调**（提示把高优先假设排在验证队列前面）。
- **理由**：机制完全不对外（decision 输出只带 prior 数值，不带"谁给的"）；LLM 出 prior 贴合"每个案件假设质量不同"，确定性钳制保证可计算性；0.3 与《00》§3 示例（H1=0.5/H2=0.4 高优先，H3=0.2/H4=0.15 低优先）一致。
- **重要澄清（相对 01 §6.3 的修正之一）**：**收敛判定 `is_converged` 不再用 prior 阈值过滤假设**——01 §6.3 的 `high = prior >= 0.3` 会把走查里的 H3/H4（prior 0.2/0.15）排除出"待收敛集合"，与《00》§4.4 走查（H3/H4 在第 2 轮仍被继续调查）矛盾。修正后的收敛定义见 T-4。prior 阈值只保留 PASS 门控用途。
- **落地动作**：🔧 无代码改动（hypothesize/guardrails 实现时照此实现）。

### 2.2 T-2 — 数量上限

- **最终取值**：`MAX_HYPOTHESES = 5`（走查用 4，留 1 余量）；`MAX_QUEUE = 8`；`MAX_TOOLS_PER_PLAN = 3`（一轮最多建议 3 个工具）。三个常量放 `guardrails/` 或配置层。
- **理由**：纯成本/上下文长度控制，不改业务语义；与《00》§4.4 走查（4 假设、2 轮工具、第 2/3 轮各 2 个工具）兼容。
- **落地动作**：🔧 无代码改动（常量实现时落位）。

### 2.3 T-3 — HypothesisStatus 词表（一致性冲突详见 §4.1）

- **最终取值**：`PENDING（生成待验证）/ SUPPORTED（证据支持）/ REFUTED（证据证伪）/ UNRESOLVED（已查证但证据不足，未能证实也未证伪）`，初始 `PENDING`。
- **理由**：UNRESOLVED 的语义"查过、没定论"与 PENDING"还没查"形成无歧义区分，直接对应《00》§7.2-4"区分证明无风险与没查到风险"（UNRESOLVED 的高优先假设 → 不可 PASS → HUMAN_REVIEW）；代码里现用 `UNVERIFIED` 与 PENDING 字面语义重叠（详见 §4.1），是必须消除的歧义源。
- **落地动作**：🔧 `domain/models.py` 枚举成员 `UNVERIFIED` → `UNRESOLVED`（含 docstring）；后续 guardrails 状态集合与 eval 标签统一用 `UNRESOLVED`。

### 2.4 T-4 — decision_confidence / risk 分离 + 三个 Decision Gate + 矛盾/收敛确定性定义

**v2 修订核心**：把初版"单一 confidence < 0.7 → HUMAN_REVIEW"的简化，升级为
**"证据是否足以支持安全自动决策"**的 Gate 判定。最终取值分五个子决定：

**(a) 两个量分离（写进《00》§7.1/§7.4 与 01 §7.5）**：

| 量 | 一句话定义 | 落点 |
|---|---|---|
| **decision_confidence** | 对"自动决策（不放人工）"的安全性把握 —— **安全门槛量，非模型判"违规"的真实概率** | `ReviewDecision.decision_confidence` 字段（O-7 已拍板改名；语义=自动决策安全门槛） |
| **risk_level / risk confidence** | 风险本身的高低（NONE/LOW/MEDIUM/HIGH）与强度（最高 SUPPORTED 假设 posterior） | `risk_level` + `hypothesis_trace[].posterior`；**不参与 Gate/路由**（T-10） |

**(b) 三个 Decision Gate（写进《00》§7.1/§7.2 与 01 §7.2 overlay）**：

- **PASS Gate**：可自动 PASS ⇔ ① 高优先风险假设全部被**充分证据**证伪（REFUTED 且有反驳证据，非"没查到"）② 关键证据完整 ③ 无未解决关键矛盾。
- **REJECT Gate**：可自动 REJECT ⇔ ① 高风险假设成立 ② 证据充分 ③ 存在明确政策依据/先例（citable）④ `decision_confidence ≥ 0.7` ⑤ 无关键矛盾。
- **HUMAN_REVIEW（abstention）**：任一成立即转人工——证据不足 / 关键证据冲突 / 政策无法确定 / **多个风险假设无法区分** / 拟自动 REJECT 而 decision_confidence<0.7 / **关键 Tool 失败致证据缺失** / Budget Exhausted。

**(c) 确定性 decision_confidence 公式**（产出的是**安全门槛**，与 risk 分离；LLM 提案 confidence 只作参考不作终值）：

```python
def finalize_decision_confidence(state) -> float:
    top = max((h.posterior or 0.0 for h in hypotheses if h.status == "SUPPORTED"), default=0.0)
    completeness = min(len(evidence) / MAX_EXPECTED_EVIDENCE, 1.0)    # MAX_EXPECTED_EVIDENCE = 8
    citation = 1.0 if any(e.type in CITABLE_TYPES and e.ref_id for e in evidence) else 0.0
    conflict = 0.2 if contradiction_detect(state) else 0.0
    return round(min(max(0.45 * top + 0.25 * completeness + 0.20 * citation + 0.10 - conflict, 0.0), 1.0), 2)
```

- **0.7 的用法**：`CONFIDENCE_ABSTAIN_THRESHOLD = 0.7` 只作为 **REJECT Gate** 的安全门槛
  （判违规需要高置信）；**PASS 不因低 decision_confidence 转人工**——干净商品低风险置信是正常态，
  PASS 由 PASS Gate 判定（证伪充分性），不是 abstention 信号。0.7 是 v1 工程初始值，validation set 校准（《00》§7.6/§11.5）。

**(d) 收敛判定 `is_converged`（写进 01 §6.3，证据级可引用、不过滤 prior）**：

```python
def is_converged(state) -> bool:
    open_hp = any(h.status in {"PENDING", "UNRESOLVED"} for h in hypotheses)          # 全部假设，不限 prior
    supported_wo_citation = any(h.status == "SUPPORTED" for h in hypotheses) and \
                            not any(e.type in CITABLE_TYPES and e.ref_id for e in evidence)
    return not open_hp and not supported_wo_citation
```

**(e) 矛盾启发式 v1（唯一一条，配置化）**：存在 `IMAGE_SIMILARITY` 证据且 `extra.similarity >= EVIDENCE_STRONG`（0.85），
同时存在 `MERCHANT_HISTORY` 证据且 `extra.removals == 0 and extra.title == 0`（O-8 回填字段无 `violations_total`；`extra.title` = value 中 title-relisting 计数，干净=removals 与 title-relisting 均为 0）→ 关键矛盾 →
overlay 记 R3_CRITICAL_CONFLICT 转人工（除非 R1 硬规则 REJECT）。

- **理由**：abstention 的落点是"证据是否足以支撑自动决策"，而不是一个孤立的置信数字；
  decision_confidence 是 Gate 的输入之一（仅约束 REJECT 侧），risk 高低与能否自动判解耦（HIGH risk + 证据不足 = HUMAN_REVIEW）。
- **落地动作**：✅ 已按 O-7 拍板将 DTO 字段改名 `decision_confidence`（`domain/models.py` 已落地；00 §2.2/§7、01 §7.5/§7.6/§8 已同步，DB `decision` 列同口径）。

### 2.5 T-5 — Evidence.weight 来源

- **最终取值**：weight 由**各 Tool 的结果→Evidence 转换器**写入默认值（01 §5.1~5.6 各工具列的默认：PRODUCT_FACT 0.6、IMAGE_SIMILARITY=similarity 数值、IMAGE_LOGO=logo confidence、OCR_TEXT 0.5、MERCHANT_HISTORY 0.85、CASE_PRECEDENT=retrieval_score、POLICY_REF 0.9），clamp 到 [0,1]；**reevaluate 不修改 weight**（只更新 posterior/status/证据链接）。weight 语义=证据强度，仅供展示/审计，**不参与 v1 的任何确定性公式**（T-4 公式只读存在性、可引用性、矛盾性）。
- **理由**：把 weight 留给"谁产出谁标价"，避免 reevaluate 双重调参造成不可复现；v1 公式不依赖 weight，则 weight 标定误差不影响判决策略（可解释性优先）。
- **落地动作**：🔧 无（6 个工具转换器实现时照此写默认值）。

### 2.6 T-6 — hypothesize 是否可重跑

- **最终取值**：**只在图入口执行 1 次**（`set_entry_point("hypothesize")`，此后无回到 hypothesize 的边）；运行中新发现的风险维度由 **reevaluate 输出的 `new_hypotheses`** 追加（新假设以 `PENDING` 进入假设集，同轮返回全集）。
- **理由**：《00》§4.2 的图结构即如此（唯一入口假设生成）；"更新假设"能力由 reevaluate 承担（更新 posterior/status），新增假设是 reevaluate 的附属职责；不加 `decide→hypothesize` 回边，避免假设集反复膨胀与预算失控。
- **落地动作**：🔧 无（graph.py 实现时照此接线）。

### 2.7 T-9 — 图内通道字段去留（pending_tool_calls / degraded / failures）

- **最终取值**：三个〔细化新增〕通道**全部保留**并补进 `AgentState`（`agent/state.py`，graph 实现阶段）：
  - `pending_tool_calls: list[dict]`（plan 写、tools 消费后置 `[]`，覆盖写）；元素字段 `{tool, args, reason, priority}`；
  - `degraded: bool`（LLM 节点失败置 True / 成功置 False，覆盖写）；
  - `failures: list[dict]`（append reducer，元素 `{step_type, tool?, severity: "warn"|"critical", reason, ts}`，O-3 拍板）。
  - **O-2/O-3 拍板（对齐 04 §5.2）**：failures 分级 —— LLM 步失败记 `severity:"critical"`；Tool 失败按是否为 plan 标记的关键取证记 `critical` / 默认 `warn`（含 `tool` 字段）。overlay 仅对 `degraded`（LLM 步失败）与**未解决的 critical Tool 失败**触发 HUMAN_REVIEW（R3_KEY_TOOL_FAILED / R5），`warn` 只进审计/trace —— failures 非空不再一律转人工（R5 语义见 01 §7.2）。
- **同时采纳现有代码的既有方案**：`run_id / case_id` **不进 State**（`state.py` 已按 LangGraph thread 维度处理：thread_id = run/案件身份，由 Checkpointer 与调用方携带）——与 01 §2.2 的 13 字段清单不同，但代码方案更符合 LangGraph 惯用法且避免双份冗余，**以代码为准**（01 后续修订）。
- **落地动作**：🔧 `agent/state.py` 增加 3 通道；reducer 语义见 01 §2.5（overwrite / append）。`case` 之外的身份信息不重复入 State。

### 2.8 T-8 — 运行终态语义 + decision.overrides

- **最终取值（运行状态机，落 DB `review_run.status`，由 worker 层维护，不进图 State）**：

| 状态 | 含义 | 何时置位 |
|---|---|---|
| `PENDING` | 运行记录已建、未启动 | 入队时 |
| `INVESTIGATING` | 执行中 / 中断可恢复（checkpointer 断点续跑） | worker 开始 invoke 时 |
| `DECIDED` | 已产出 `ReviewDecision`（**含 PASS/REJECT/HUMAN_REVIEW 三种结果**；预算/降级导致的 HUMAN_REVIEW 也落 DECIDED） | invoke 返回且 decision 非空 |
| `FAILED` | 未产出决策的异常终止（不可自动恢复） | invoke 抛未捕获异常 |
| `ESCALATED` | **不作为 review_run 主终态**：语义=HUMAN_REVIEW 已被投递人工裁决队列，属下游人工流程状态（review 维度维护） | 人工工作台侧 |
| `BUDGET_EXCEEDED` | **不作为 review_run 主终态**：超限归因记 `decision.overrides=["R3_BUDGET_EXHAUSTED"]`，budget 快照随 `review_result.decision_json`（budget_used）落库；供"超时/超限转人工率"指标（《00》§10.3）统计 | —— |

- **`decision.overrides`**：`ReviewDecision` 增加可选字段 `overrides: list[str] = Field(default_factory=list)`，存放确定性 overlay 的改判/归因原因码（v2 词汇：R1_HARD_RULE / R2_REJECT_GATE_FAIL / R3_BUDGET_EXHAUSTED / R3_CRITICAL_CONFLICT / R3_KEY_TOOL_FAILED / R3_POLICY_UNCERTAIN / R3_HYPOTHESES_INDISTINGUISHABLE / R3_VISUAL_CLAIM_UNSUPPORTED / R4_PASS_GATE_FAIL / R5_DEGRADED_OR_FAILED_STEP）；空=overlay 未改判（LLM 提案即终值）。这是"谁把 PASS/REJECT 改成了 HUMAN_REVIEW"的可审计落点（《00》§8.2-4），且不破坏既有 decision 字段（新增默认空列表，向后兼容）。
- **理由**：图内每轮运行必然以 decide 产出一个 decision 收尾，因此"DECIDED"是唯一的图终态；把 ESCALATED/BUDGET_EXCEEDED 从主状态机剥出为归因/下游状态，避免状态机出现"决策已出但状态未决"的二义；overrides 可选字段保证 00 §2.2 输出形状兼容。
- **v2 复核**：再次确认 **DECIDED 是 Graph 唯一终态**——PASS/REJECT/HUMAN_REVIEW 只是
  `ReviewDecision.decision` 的取值，不是图终态；预算耗尽 / 关键 Tool 失败 / 降级 / Gate 改判全部通过
  `decision.overrides`（R1/R2/R3/R4/R5 原因码）记录，**不新增任何图终态**；ESCALATED / BUDGET_EXCEEDED
  仅存在于 DB 运行/下游状态（03 §2.8 上表）与 trace 归因，graph.py 里不存在对应终结点。
- **落地动作**：🔧 `domain/models.py` ReviewDecision 增 `overrides` 字段（extra="forbid" 下必须显式声明）；worker/DB 层状态机照上表。

### 2.9 T-10 — risk_level 词表与映射

- **最终取值**：
  - 词表采用**代码现有值** `NONE / LOW / MEDIUM / HIGH`（代码比 01 建议多 `NONE`，更合理：PASS 案件无风险量级）：
    - `PASS` → `risk_level=NONE`、`risk_type=[]`；
    - `REJECT / HUMAN_REVIEW` → 默认映射：`POTENTIAL_IP_RISK / EVASION_PATTERN` 起步 `HIGH`（《00》走查即 HIGH）；`FALSE_CLAIM / FIELD_CONFLICT` 默认 `MEDIUM`，允许 LLM 提案按证据强弱给 LOW/MEDIUM/HIGH。
  - **该映射仅用于展示与人工队列排序，不参与任何路由/overlay 判定**（避免把展示口径变成判定逻辑；overlay 决策只认 decision/decision_confidence/证据，不认 risk_level）。
  - **v2 落地口径（代码 `gate.py::finalize_risk_level` 已落地，契约 §5.3）**：上表"默认映射"是 **prompt 指导**（供 LLM 提案时参考，非强制）；当提案**未声明** `risk_level` 时，确定性终值按最高 SUPPORTED posterior 派生：`≥0.8 → HIGH / ≥0.5 → MEDIUM / ≥0.2 → LOW / 其余 → NONE`。低危案（派生为 NONE）可产出 `risk_level=NONE` 的 HUMAN_REVIEW —— risk_level 纯展示字段（只进队列排序），无路由影响。
- **v2 补强（写进《00》§7.5）**：**risk_level ≠ decision** —— 决策由 §2.4(b) 的 Decision Gate 判定（证据 + 政策依据 + decision_confidence），
  risk_level 高低不改变决策：**HIGH risk + 证据不足 = HUMAN_REVIEW（而非 HIGH → REJECT）**；LOW risk + 证据矛盾同样 HUMAN_REVIEW。
- **理由**：词表与映射是输出格式问题，不改变判决策略；NONE 让 PASS 在统计/队列里语义干净。
- **落地动作**：🔧 无（代码已含 NONE；映射表写入 decide 实现注释）。

### 2.10 T-12 — 字段冲突（FIELD_CONFLICT）检测归属

- **最终取值**：v1 **不做**确定性字段冲突检测器；`FIELD_CONFLICT` 词表成员保留（代码已有）；OCRTool 与商品字段（title/description/attributes）的交叉冲突，v1 由 reevaluate/decide 的 LLM 在证据综合时自然发现并使用该 risk_type。**二期**（对应《00》§14.2 支撑场景）在 `guardrails/` 实现确定性检测器（OCR 文本 vs 商品字段比对，不占 LLM）。
- **理由**：《00》§14.2 已把"字段信息冲突"列为第二阶段支撑场景，非 MVP 阻塞项；v1 词表先占位保证三方共用词表成立（《00》§7.3）。
- **落地动作**：🔧 无。

---

## 3. [B] 拍板项（2 项 —— 均已拍板；保留问题与选项作为决策记录）

### 3.1 T-7 — 预算阈值与走查余量（影响成本上界/成本口径）

**问题**：《00》§8.1 定死预算上限 `8 次 LLM / 12 次 Tool / 40000 token / 30s`；但主链路走查（hypothesize + plan×3 + reevaluate×3 + decide）恰好消耗 **8/8 次 LLM**、零余量——而"schema 校验失败→重试 1 次"是既定设计，任何一次重试就会把单案推到超限转人工。要不要偏离《00》数值、以什么口径落地？

**选项**：

| 选项 | 取值 | 代价/收益 |
|---|---|---|
| A | 完全沿用《00》：8 / 12 / 40000 / 30000 | 成本口径与总设计逐字一致；但走查贴顶、无重试余量，LLM 重试即触发"预算转人工" |
| B（推荐） | 上调 LLM/Tool：**10 / 15** / 40000 / 30000 | 上限是"可花费上界"而非目标：为主链路(8)留 2 次重试余量，常态成本不变；只在真正超深调查时才多花；代价=偏离《00》字面数字（可解释为"给既定重试策略留容差"） |
| C | 双档配置：线上默认 8/12（与《00》一致），eval/演示走查用 10/15 | 指标口径与成本叙事两不误；代价=多一份配置矩阵与口径说明 |

**推荐 B**：理由见上表；若强求与《00》逐字一致则选 A；C 作为折中（推荐在 02-evaluation.md 前先定，避免评测跑分与线上口径不一致）。

**✅ 已拍板：选 B —— `10 / 15 / 40000 / 30000`。**
理由：预算上限是"可花费上界"而非目标值 —— 主链路常态仍为 8 次 LLM（《00》§4.4 走查），
多出的 2/3 次只用于覆盖既定的"schema 校验失败 → 重试 1 次"策略（plan/reevaluate 各轮均可触发），
避免"任何一次重试即把单案推到预算超限转人工"；token/latency 维持《00》原值（40000/30s），
成本叙事（P50/P95 token 与延迟）不受 LLM/Tool 次数上界调整影响。落地：上限经运行配置注入
`Budget.limits`（graph 阶段 guardrails 常量层 / worker 初始化），`models.py` `BudgetLimits` 默认已
按 T-7 修订为 10/15/40000/30000（与原 docstring"与设计一致 8/12"不再相符，docstring 同步更新）。
决策时间/来源：设计评审（已复核）

**v2 复核（写进《00》§8.1/§10.3/§11.3）**：
- **Budget 是 Guardrail，不是目标调用次数**：正常案件实际调用应明显低于上限（主链路 8/5 次）；
  上限余量留给 schema 重试、工具失败恢复、防无限循环。
- **Trace 记录四组占用率**：`llm_calls/max_llm_calls`、`tool_calls/max_tool_calls`、
  `tokens/max_tokens`、`latency/max_latency`（review_trace 逐步 tokens/latency / decision_json.budget_used）。
- **Evaluation 增 Budget Utilization 指标**（《00》§11.3）：观察占用率分布，证明 Agent 不是为耗完预算而调工具；
  若某类案件占用率普遍贴顶，应视为 plan 选择策略或上限设置的问题信号。

---

### 3.2 T-11 — ImageAnalysis 相似度阈值（影响"多像算仿冒证据"的业务口径）

**问题**：图片与品牌款的相似度到什么程度才形成 `IMAGE_SIMILARITY` 证据（`EVIDENCE_MIN_SIM`），以及"高相似/Strong Evidence"分界（代码名 `EVIDENCE_STRONG`，v1 旧名 `SIM_HIGH_CONTRADICT`）取多少？阈值直接决定哪些外观案件进入证据链、进而影响 REJECT 误伤与漏放（《00》§7.2-2"防误伤商家"红线）。

**选项**（格式：产证据阈值 `EVIDENCE_MIN_SIM` / Strong/高相似分界 `EVIDENCE_STRONG`）：

| 选项 | 取值 | 代价/收益 |
|---|---|---|
| A | 0.60 / 0.85（01 原默认） | 召回最广：0.60~0.85 的中相似都进证据链交 LLM/reevaluate 综合；误伤风险与上下文噪声最大 |
| B（推荐） | **0.70 / 0.85** | 业界常用"高相似"分界：完整覆盖核心 Hard Case（0.91）与评测高分段（0.8+），滤掉"风格撞车"的弱相似；误伤/漏放平衡点 |
| C | 0.85 / 0.85 | 只认高度相似为证据，最防误伤；但低相似规避、擦边仿款会被漏掉（与《00》"多弱信号组合"案型冲突，弱相似将彻底丢失） |
| D | 分级双阈值：≥0.85 强证据（可作 REJECT 直接依据）；0.60~0.85 弱证据（仅进 reevaluate 综合，不能单独支撑 REJECT） | 表达力最强、最贴近"克制判违规"；代价=引入证据强度分级，影响 5.2 转换器、矛盾启发式与 02-eval 证据标签口径 |

**推荐 B**（v1 单阈值最简且平衡）；若 02-eval 的 IP 样本相似度区间较宽（0.6~0.9 都有）建议改选 D。选定后 `EVIDENCE_MIN_SIM / EVIDENCE_STRONG` 作为配置常量落地，两处阈值都做成配置项以便评测 sweep 调参。

**✅ 已拍板：选 B —— `EVIDENCE_MIN_SIM = 0.70` / `EVIDENCE_STRONG = 0.85`。**
理由：0.70 为业界常用"高相似"分界 —— 完整覆盖核心 Hard Case（0.91）与评测高分段（0.8+），
同时滤掉"风格撞车"的弱相似命中（防误伤商家，对齐《00》§7.2-2 红线）；0.85 作为 Strong Evidence 档
与矛盾启发式的"高相似"分界（与 T-4(e) 矛盾检测一致：`extra.similarity >= 0.85` 且商家历史干净才判矛盾）。
落地：代码常量已同步（`src/pra/tools/image_analysis/tool.py`：`EVIDENCE_MIN_SIM=0.70`、`EVIDENCE_STRONG=0.85`；
文档统一使用代码名 `EVIDENCE_STRONG`，本表 v1 旧名 `SIM_HIGH_CONTRADICT` 同值同义，仅在别名注记中出现）；
工具仍只返回原始相似度，证据过滤/矛盾检测由下游 tools_node/guardrails 确定性层引用这两值。
决策时间/来源：设计评审（已复核）

**v2 复核（写进《00》§7.6/§11.5/§13.3 与 01 §5.7）**：
- **三档语义**：`similarity < 0.70` 不作 IMAGE_SIMILARITY 证据；`0.70 ~ 0.85` 普通证据；
  `≥ 0.85` **Strong Evidence**（也是矛盾启发式"高相似"判据）。
- **数值定位**：0.70/0.85 是 **v1 工程初始值、非理论最优**，阈值**配置化**（不写死）。
- **Evaluation 支持 threshold sweep**：对 `EVIDENCE_MIN_SIM / EVIDENCE_STRONG` 扫
  `0.60 / 0.65 / 0.70 / 0.75 / 0.80 / 0.85 / 0.90`，在 validation set 上观察
  Recall / Precision / FPR / Human Review Rate，选 operating point（《00》§11.5）。

---

## 4. 一致性核查：待定项取值 vs 现有实现（domain/models.py · agent/state.py · tools/base.py）

### 4.1 T-3 专项冲突：UNVERIFIED vs UNRESOLVED

- **冲突描述**：《01-agent-loop.md》§2.4 假设状态词表建议 `UNRESOLVED`（语义=已核查但证据不足以证实/证伪）；现有 `src/pra/domain/models.py` `HypothesisStatus` 枚举用 **`UNVERIFIED`**，其 docstring 却写着同样的语义"已核查但证据不足以证实/证伪"。即**两个名字、同一语义**。
- **为什么是问题**：`UNVERIFIED` 字面意为"尚未验证"，与 `PENDING`（生成待验证/尚未查）在字面上重叠，而 docstring 实义是"已查未决"——名字与语义打架，后续实现者极易把两者混用（例如收敛判定把 UNVERIFIED 当"还没查"处理，导致提前收敛漏查）。
- **对齐建议**：**改代码**：`models.py` 枚举成员 `UNVERIFIED` → **`UNRESOLVED`**（docstring 同步）。
- **理由**：① 01 文档、guardrails 状态集合（`{PENDING, UNRESOLVED}` 即"未收敛"集合）、以及即将落地的收敛谓词/评测标签都以 `UNRESOLVED` 为基准，改代码一处成本最低；② `UNRESOLVED`（悬而未决）与 `PENDING`（待验证）字面语义无歧义地区分"没查"与"查了没定论"，与《00》§7.2-4"证明无风险 vs 没查到风险"的叙事直接呼应；③ 目前无任何代码引用 `UNVERIFIED`（domain 层无消费方），重命名零破坏。
- **影响面**：后续 guardrails / nodes / eval 标签统一用 `UNRESOLVED`；02-evaluation.md 未开始，无存量标签需迁移。

### 4.2 其它待定项与代码的字段级漂移表

| # | 对象 | 01-agent-loop 契约 | 现有代码 | 冲突/漂移说明 | 最终口径（本表拍板） | 落地动作 |
|---|---|---|---|---|---|---|
| 1 | `HypothesisStatus` 词表 | UNRESOLVED | UNVERIFIED | **命名冲突（T-3，§4.1）** | UNRESOLVED | 🔧 models.py 重命名 |
| 2 | `Hypothesis.posterior` | 默认 `0.0`（01 §2.4） | `float \| None = None`（未 reevaluate 前为 None） | 形态差异 | **以代码为准**：初始 None、reevaluate 后数值；聚合公式对 None 按 0 处理（T-4(a) 已内置） | 无需改代码；01 后续修订 |
| 3 | `Evidence` 字段 | `evidence_id / type(枚举) / source_tool / value / weight / ref_id / extra` | `type: str / source / value / weight / ref_id`（无 evidence_id、无 extra） | ①字段名 source_tool→source（代码与《00》§2.2 一致）；②缺 evidence_id/extra；③type 为开放 str | **字段名 `source` 为准**（对齐《00》§2.2）；`evidence_id` 不进 DTO（ToolNode 运行时按 `E_nn` 分配用于引用与 result_ref，DB `evidence` 主键承载持久化身份）；`type` 保持开放 str、运行时收敛到 7 个受控值（`guardrails` 常量集 + 单测，**不升级为 Pydantic Enum**——避免 contract 卡死未来新证据类型）；**增 `extra: dict`** 承载 similarity/removals 等数值（T-4(b)/T-11 的确定性读取依赖） | 🔧 models.py：Evidence 增 `extra: dict = Field(default_factory=dict)`（extra="forbid" 下必须显式声明）；dedup key = `(type, source, ref_id)`（**O-1 拍板修订**：ref 优先稳定业务标识 image_url/product_id/merchant_id/case_id/clause_id，ref_id=None 时回退 value，见 01 §2.5） |
| 4 | `ReviewDecision.overrides` | 01 建议可选（T-8） | 无该字段 | T-8 扩展 | 增加 `overrides: list[str] = []`（overlay 原因码 R1..R5） | 🔧 models.py 增字段 |
| 5 | `AgentState` 身份字段 run_id/case_id | 在 State 内（01 §2.2 #1/#2） | 迁出为 LangGraph thread 维度 | **有意分歧**（代码 docstring 已说明） | **以代码为准**：身份归 thread_id，State 不冗余 | 无（graph.py 接线时映射） |
| 6 | `AgentState.status` | 在 State 内（01 §2.2 #4） | 不在 State（DB `review_run.status`） | T-8 落地分歧 | **以 DB 承载**：worker 层维护 PENDING/INVESTIGATING/DECIDED/FAILED（见 §2.8），不进图 State | 无（worker/DB 实现时照 §2.8） |
| 7 | `AgentState` 图内通道 pending_tool_calls / degraded / failures | 建议新增 | 无（graph 阶段 TODO） | T-9 | **保留并补进 AgentState**（见 §2.7） | 🔧 state.py 增 3 通道 |
| 8 | `investigation_queue` / `tool_call_history` 元素形态 | 01 §2.4 子模型（InvestigationItem/ToolCallRecord） | `list[dict]`（与《00》§3 JSON 一致） | 形态差异 | **保持 dict**（对齐《00》§4.2 草图与现有 state.py）；字段约束由写入方遵守 + 单测保证，不引入子模型 | 无需改代码 |
| 9 | `RiskLevel` 词表 | T-10 建议 LOW/MEDIUM/HIGH | `NONE/LOW/MEDIUM/HIGH` | 代码多 NONE | **以代码为准**（PASS→NONE，见 §2.9） | 无需改代码 |
| 10 | `Budget.latency_ms` | 01 未细化 | 含 latency_ms（§3 运行态 ∪ §2.2 budget_used 并集，一型两用） | 代码自洽扩展 | **以代码为准**（ReviewDecision.budget_used 直接引用运行期实例） | 无需改代码 |

### 4.3 一致性核查结论

- **仅 1 处真正冲突**：T-3 的 UNVERIFIED/UNRESOLVED（§4.1），建议改代码。
- **5 处"以代码为准"的形态漂移**（漂移项 2/5/6/8/10 及 RiskLevel）：代码均已自洽且更贴《00》原文或 LangGraph 惯用法；01 文档已在本轮（v2）随本表同步（01 §2 AgentState/子模型与代码对齐）。
- **代码改动均已执行**：漂移项 1（UNRESOLVED 重命名）、3（Evidence.extra）、4（overrides）、7（state.py 三通道）+ 本轮 BudgetLimits 默认值（§4.4 ✅）。
- **无"待定项取值与代码互相矛盾且必须改文档"的项**：T-1/T-2/T-4/T-5/T-6/T-9/T-10/T-12 的取值均不触碰现有 DTO 字段，落地即实现层常量/逻辑。

### 4.4 代码落地动作清单（v1 项已执行；v2 增补项见末行）

| 顺序 | 文件 | 动作 | 状态 |
|---|---|---|---|
| 1 | `src/pra/domain/models.py` | `HypothesisStatus.UNVERIFIED` → `UNRESOLVED`（docstring 同步） | ✅ 已执行 |
| 2 | `src/pra/domain/models.py` | `Evidence` 增 `extra: dict = Field(default_factory=dict)`（承载结构化数值） | ✅ 已执行 |
| 3 | `src/pra/domain/models.py` | `ReviewDecision` 增 `overrides: list[str] = Field(default_factory=list)`（overlay 原因码） | ✅ 已执行 |
| 4 | `src/pra/agent/state.py` | `AgentState` 增 `pending_tool_calls: list[dict]`（覆盖写）、`degraded: bool`（覆盖写）、`failures`（append）；run_id/case_id 线程维度、status 落 DB | ✅ 已执行 |
| 5 | guardrails 常量层（`src/pra/agent/guardrails/`：常量表 + `gate.py`） | 按 §5 常量表落位；T-7/T-11 值 10/15/40000/30000 与 0.70/0.85 | ✅ 已执行（常量层与 `gate.py` 均已落地） |
| 6 | `src/pra/domain/models.py` | **v2：`BudgetLimits` 默认值 8/12 → `10/15`（max_llm_calls/max_tool_calls），docstring 注明"Guardrail 上界非目标、运行时可由配置覆盖"** | ✅ 本轮执行 |

> v1 章节 4.3/4.1 的漂移结论仍有效；其中"01 文档需后续同步"已在本轮完成（01 §2/§6/§7/§9 与《00》§7/§8/§11/§12/§13 已按 v2 修订）。

---

## 5. 可落地参数常量总表（[A] 已定 + [B] 已拍板）

> 供 guardrails/配置实现直接引用；`[B]` 项确认后回填并解锁对应实现。

| 常量 | 取值 | 来源 | 备注 |
|---|---|---|---|
| `HIGH_PRIOR_THRESHOLD` | 0.3 | T-1 [A] | 仅 PASS/REJECT Gate 的"高优先"口径 + prompt 强调；不用于收敛判定 |
| `MAX_HYPOTHESES` | 5 | T-2 [A] | hypothesize 输出上限 |
| `MAX_QUEUE` | 8 | T-2 [A] | 调查队列上限 |
| `MAX_TOOLS_PER_PLAN` | 3 | T-2 [A] | 每轮 plan 建议工具数上限 |
| `MAX_EXPECTED_EVIDENCE` | 8 | T-4 [A] | decision_confidence 公式完整性分母 |
| `CONFIDENCE_ABSTAIN_THRESHOLD` | 0.7 | T-4 [A]（《00》§7.6 工程初始值） | **自动 REJECT 的安全门槛（decision_confidence，非模型概率）**，仅约束 REJECT Gate；PASS 不走置信 abstention；validation set 校准 |
| `EVIDENCE_MIN_SIM` | 0.70 | T-11 [已拍板 B] | 产 IMAGE_SIMILARITY 证据的下限（三档语义下界）；v1 工程初始值，配置化 + threshold sweep |
| `EVIDENCE_STRONG` | 0.85 | T-11 [已拍板 B] | **Strong Evidence 档 + 矛盾启发式"高相似"判据**；文档统一此名（v1 旧名 `SIM_HIGH_CONTRADICT` 同值同义）；代码常量已落地（image_analysis/tool.py） |
| `CITABLE_TYPES` | {CASE_PRECEDENT, POLICY_REF} | T-4 [A] | 收敛 / REJECT Gate 的可引用依据集合 |
| 预算 `max_llm_calls / max_tool_calls / max_tokens / max_latency_ms` | 10 / 15 / 40000 / 30000 | T-7 [已拍板 B] | **Guardrail 上界非目标**：主链路常态 8/5 次明显低于上限，余量留重试/恢复/防死循环；Trace 记四组占用率，Evaluation 看 Budget Utilization（《00》§11.3）；`models.py` `BudgetLimits` 默认已改 10/15/40000/30000，运行时可由配置覆盖 |

---

## 6. [B] 项决策记录摘要（2 项 —— 均已拍板，见 §3.1 / §3.2 拍板结果）

**T-7 预算上限**：问题=《00》定 8/12/40000/30s，主链路走查恰耗满 8 次 LLM、零余量，与"schema 校验重试 1 次"既定策略冲突。选项：A 沿用 8/12（与《00》逐字一致，无重试余量）/ **B 推荐 10/15**（上限≠目标，为主链路留 2 次重试余量，常态成本不变）/ C 双档配置（线上 8/12、eval/演示 10/15）。

**T-11 图片相似度阈值**：问题=相似度到什么程度才算"命中品牌款"证据（产证据阈值）与"高相似矛盾"分界。选项：A 0.60/0.85（召回最广，误伤风险最大）/ **B 推荐 0.70/0.85**（业界惯用高相似分界，覆盖 Hard Case 0.91 与评测 0.8+，误伤/漏放平衡）/ C 0.85/0.85（最防误伤，漏低相似规避）/ D 分级双阈值 0.85 强证据 + 0.60~0.85 弱证据（表达最强，复杂度最高）。

---

## 7. 结束状态（v2）

- [A] 已定：T-1、T-2、T-3、T-4、T-5、T-6、T-8、T-9、T-10、T-12（10 项，最终值见 §2 / §1 总览）。
- [已拍板] T-7 → **B（10 / 15 / 40000 / 30000，Guardrail 语义 + Budget Utilization）**、T-11 → **B（EVIDENCE_MIN_SIM=0.70 / EVIDENCE_STRONG=0.85，三档语义 + threshold sweep）**（§3.1 / §3.2；已复核）。
- v2 修订落点：① decision_confidence 与 risk 分离 + 三个 Decision Gate（T-4/《00》§7/01 §7）；② Budget Guardrail 语义 + 四组占用率 + Budget Utilization 指标（T-7/《00》§8/§10.3/§11.3）；③ 相似度三档 + 工程初始值 + threshold sweep（T-11/《00》§7.6/§11.5）；④ DECIDED 图唯一终态复核（T-8）；⑤ risk_level ≠ decision 示例（T-10/《00》§7.5）；⑥ 评测公平性（《00》§12.0）、成本/调查效率指标含 **Marginal Evidence Gain / Investigation Efficiency**（《00》§11.3，契约落 01 §2.4/§5.8 tool_call_history）、**Ablation Evaluation**（《00》§13.4）；⑦ 命名统一 EVIDENCE_STRONG、01 §2 与代码对齐、`BudgetLimits` 默认 10/15。
- 一致性冲突 1 处（T-3）已通过代码改名解决；代码落地动作 §4.4 已执行（含本轮 BudgetLimits 默认值修订）。
- [O 拍板] **O-1~O-10 已拍板并落地**（去重 key 稳定 ref / failures severity 分级与 R5 口径 / `args_model`+`parse_args` / `decision_confidence` 改名 / extra 由 tools_node backfill 等）：每条 O 的拍板结果与落地记录见 **docs/04-graph-design.md §10**；代码同步见 models.py / state.py / tools/base.py / tools 各 tool.py。
- 遗留/待办：01-agent-loop.md 若有与 O 表冲突的旧表述以 04 §10 为准（guardrails 常量层与 `tools_node` 的 `backfill_extra`/`quality_filter` 接线已落地，见 §4.4 行 5）。
