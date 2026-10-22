# 调研：LLM agent 护栏（harness）与评测的业内做法

- 调研日期：2026-10-19
- 调研问题：
  1. 主流护栏框架是"代码判模型"还是"模型判模型"？各自的设计哲学是什么？
  2. "模型提案 + 确定性/人类终裁"架构在业内叫什么名字、有哪些实例？
  3. 工具调用权限与证据约束（allowlist、引用要求）的通行做法是什么？
  4. 主流 agent 基准（SWE-bench、τ-bench、AgentBench、GAIA）如何评分——确定性判卷还是 LLM-as-judge？
  5. LLM-as-judge 的已知偏差与业内缓解做法是什么？
  6. holdout/dev 纪律、contamination 防护、多次抽样一致性、metamorphic testing 有哪些应用实例？
  7. 用第二模型审计真值标签有没有先例？
- 背景：本项目是电商商品审核 agent，核心设计是"模型只出提案、确定性代码终裁"（硬规则不可被模型覆盖、
  自动拒绝必须有可引用政策依据、证据不足转人工），评测为 selective prediction 口径 +
  320 案冻结考卷 / 160 案练习册 + 异源模型交叉审计真值 + 确定性判卷。
  与 `research-over-abstention.md`（selective prediction 理论、过度弃权、评测迭代纪律、
  人机分工）互补，本文只覆盖护栏 harness 与评测口径两块，不重复该文内容。
  每条做法标注与本项目的对应关系：【已采用】/【值得考虑】/【不适合本项目】+ 一句话理由。

## 第一部分：护栏 / harness

### 1. 护栏框架的两条路线："代码判模型" vs "模型判模型"

业内护栏工具明确分成两派，分野在于**裁决者是不是确定性代码**。

**代码判模型（可编程护栏）：**

- **NVIDIA NeMo Guardrails**：核心概念是 "programmable rails"——开发者用专用建模语言 Colang
  定义对话流与安全约束，rails"由用户定义、独立于底层 LLM、可解释"，运行时包裹在应用代码与
  LLM 之间，与训练期对齐（embedded guardrails）显式对立。即：控制逻辑是**程序**，
  不是另一个模型的判断（[arXiv 2310.10501 摘要](https://arxiv.org/abs/2310.10501)，
  [官方仓库](https://github.com/NVIDIA-NeMo/Guardrails)）。
- **Guardrails AI**：核心是 validator——对模型输入/输出跑确定性校验组件（格式、取值域、
  PII、规则约束等），失败后执行可配置的 on-fail 动作：`reask`（带失败原因重问模型）、
  `fix`、`filter`、`refrain`（拒答返回 None）、`exception`、`noop`
  （[官方文档：Validator OnFail Actions](https://guardrailsai.com/guardrails/docs/concepts/validator_on_fail_actions)）。
  要点：模型输出不直接生效，先过代码校验管线，"重试"也是一种代码驱动的控制流。
- **OpenAI Agents SDK guardrails**：input / output / tool 三层 guardrail 函数，裁决语义是
  `tripwire_triggered` 布尔值，触发即抛异常、终止 agent 执行；input guardrail 可设
  blocking 模式，保证"贵模型/工具副作用在裁决通过前不发生"
  （[官方文档：Guardrails](https://openai.github.io/openai-agents-python/guardrails/)）。
  注意其 guardrail 函数**内部**可以跑一个便宜模型做分类，但对外的裁决与阻断机制是
  确定性代码——"模型可以做检查员，但执法的是代码"。

**模型判模型（安全分类器）：**

- **Llama Guard**（Meta）：一个微调过的 LLM，对输入/输出做 safe/unsafe 分类并可细分类目，
  定位是"LLM-based input-output safeguard"（[arXiv 2312.06674](https://arxiv.org/abs/2312.06674)）。
- **OpenAI Moderation API**：托管的审核模型端点（`omni-moderation-latest`，多模态），
  返回 flagged 布尔 + 类目分数，供开发者自行决定处置
  （[官方 Moderation 指南](https://platform.openai.com/docs/guides/moderation)）。

两派并不互斥：分类器模型产出的是**信号**，确定性代码决定**处置**。OpenAI Agents SDK 的
官方示例（用小模型当 guardrail 内部实现、tripwire 做阻断）就是两派的合体形态。

**对本项目的启示**：
- 本项目的架构（模型提案、Gate/硬规则终裁）正是 NeMo / Guardrails AI / OpenAI SDK 代表的
  "代码判模型"路线，而且是其中走得最远的一档：多数框架只对输入/输出做格式与安全校验，
  本项目把**业务终裁本身**放进了确定性代码。业内的成熟先例说明这条路线是主流而非异类。【已采用】
- Llama Guard / Moderation API 这类"模型判模型"适合当**信号源**（风险初筛、分流），
  不适合当终裁——用它们做自动拒绝与本项目正确性契约"不允许模型说了算的自动拒绝"直接冲突。【不适合本项目】
- Guardrails AI 的 `reask`（校验失败时把失败原因喂回模型重试）值得对照：本项目判定输出
  若格式/证据校验失败，当前处理是显式失败还是重试，值得确认是否已有等价机制；
  "带失败原因的重试"比"一次失败即终止"能挽回一部分合法判定。【值得考虑】

### 2. "模型提案 + 确定性/人类终裁"的业内形态与名字

这个架构在业内没有统一的名字，但以几种成熟形态反复出现：

- **护栏 tripwire（代码终裁）**：OpenAI Agents SDK 的 tripwire 语义即"模型可以继续跑，
  但最终输出/工具调用必须过裁决；裁决不过，执行被异常终止"，且 blocking 模式显式处理
  "副作用不可回滚"的问题（[官方文档](https://openai.github.io/openai-agents-python/guardrails/)）。
  这与"模型提案、代码终裁"同构：提案可以先生成，生效前必须过闸。
- **human-in-the-loop 审批闸（人终裁）**：LangChain/LangGraph 的 HITL middleware 在
  **工具调用粒度**上暂停执行，把待执行的调用与参数呈现给人，人可 approve / edit / reject，
  按工具分别配置哪些需要审批（[官方文档](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)）。
  即"模型提议动作、人决定动作是否生效"，而非"人替模型做全部工作"。
- **监管语境的 human oversight**：EU AI Act（Regulation (EU) 2024/1689）Article 14 要求
  高风险 AI 系统必须设计为"可被自然人有效监督"，监督者需能理解系统能力边界、识别异常、
  并有能力**干预或中断**系统（[EUR-Lex 法规原文](https://eur-lex.europa.eu/eli/reg/2024/1689/oj)，
  条文镜像 [ai-act-law.eu/article/14](https://ai-act-law.eu/article/14/)）。
  监管语言里的 oversight 不是"事后签字"，而是内建于运行机制。
- **金融模型风险管理（独立验证）**：美联储 SR 11-7《模型风险管理监管指引》要求模型必须
  经过独立于开发方的验证（independent validation）与"有效挑战"（effective challenge），
  模型产出不能直接成为决策（[Federal Reserve SR 11-7](https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm)）。
  这是"提案与终裁分离"在金融业制度化最早的形态。

**对本项目的启示**：
- "模型提案 + 确定性终裁"在业内的对应物是：工程界叫 guardrail/tripwire 与 HITL 审批闸，
  监管界叫 human oversight，金融界叫独立模型验证。三条线都要求**生效前有一个独立于
  提案模型的裁决层**——本项目设计有跨领域背书。【已采用】
- LangChain 的 HITL 把审批闸放在**工具调用粒度**而非"整案粒度"：本项目目前是整案转人工，
  若未来出现"大部分证据齐、只差一个关键核查"的形态，可以参考"单动作审批"而非"整案移交"
  来降低人工成本。【值得考虑】
- EU AI Act 强调监督者要"有能力干预和中断"：对本项目的推论是人工复核界面必须呈现
  可行动的完整证据与调查轨迹（本项目契约已要求），且人必须有推翻权——这已是设计现状，
  但可引用 Art.14 作为对外说明设计动机的权威依据。【已采用】

### 3. 工具调用权限与证据约束的通行做法

- **工具调用的前后校验**：OpenAI Agents SDK 的 tool guardrail 挂在工具上、每次调用都执行，
  input guardrail 可在工具执行前跳过调用/替换输出/触发 tripwire；且与人工审批串行
  （默认审批后再跑 input 校验，可配置为审批前先校验）
  （[官方文档：Tool guardrails](https://openai.github.io/openai-agents-python/guardrails/)）。
  要点：**权限约束长在工具上，不长在 prompt 里**——模型无法通过说服自己绕过。
- **按工具分级审批**：LangChain HITL middleware 用 `interruptOn` 按工具名单配置哪些调用
  需要人审，低风险工具自动执行（[官方文档](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)）。
  这是 allowlist 思想的运行时形态：默认信任集显式列出，集外一律升级。
- **引用约束（citation-grounded）**：ALCE（Gao et al., EMNLP 2023）是第一个系统的
  LLM 引用质量基准，核心发现之一是**生成带引用的文本不等于引用真的支撑了论断**——
  引用可以是编造的来源，也可以是真实来源但不支撑该 claim，因此需要自动评估
  "每个 claim 是否被其引用支撑"（[arXiv 2305.14627](https://arxiv.org/abs/2305.14627)）。
  要点：引用要求必须配套**支撑性校验**，否则只是格式要求。

**对本项目的启示**：
- "权限长在工具/代码上、不长在 prompt 里"是业内共识，本项目的硬规则与 Gate 已经是这个
  形态。【已采用】
- ALCE 的教训直接适用于本项目"自动拒绝必须有可引用政策依据"：仅检查"模型给了引用"
  不够，需要确定性校验"被引用的条款文本确实存在且与判定相关"。本项目若已在判卷/终裁
  侧校验引用的政策条款 ID 属于冻结规则书，则与业内最佳实践齐平；若只校验了引用格式，
  值得补上支撑性校验。【已采用，支撑性校验需自查】

## 第二部分：评测口径

### 4. Agent 基准的评分方式：确定性判卷是主流

- **SWE-bench**（ICLR 2024）：判卷完全由单元测试承担——候选补丁必须让 `FAIL_TO_PASS`
  测试（修复前红、修复后绿）通过且 `PASS_TO_PASS` 测试（既有功能）不回归，两组全过才算
  resolved（[arXiv 2310.06770](https://arxiv.org/abs/2310.06770)，
  [OpenAI 的 SWE-bench 说明](https://openai.com/index/introducing-swe-bench-verified/)）。【已采用：确定性判卷】
- **τ-bench**（Sierra，ICLR 2025）：评测的是 tool-agent-user 三方交互，判卷方法是
  **比对对话结束后的数据库状态与标注的目标状态**（外加必需输出检查），完全不看对话过程
  的"理由写得像不像"；并提出 `pass^k` 指标衡量多次独立试验的一致性（gpt-4o 单次成功率
  <50%，retail 域 pass^8 <25%）（[arXiv 2406.12045 摘要](https://arxiv.org/abs/2406.12045)）。【已采用 + 见第 6 节】
- **GAIA**（Meta/HuggingFace，ICLR 2024）：466 道有**唯一无歧义答案**的题，按 quasi-exact
  match 自动判卷；其中 300 题的答案不公之于众，只留作排行榜评分用——基准作者主动制造
  holdout（[arXiv 2311.12983 摘要](https://arxiv.org/abs/2311.12983)，
  评分方式见 [Princeton HAL 的 GAIA 页](https://hal.cs.princeton.edu/reliability/benchmark/gaia/)）。【已采用】
- **AgentBench**（清华等，ICLR 2024 spotlight）：8 个交互环境（OS、数据库、知识图谱、
  网购等），各环境用各自的确定性指标（成功率、F1、游戏奖励等）而非模型打分
  （[arXiv 2308.03688 摘要](https://arxiv.org/abs/2308.03688)）。【已采用】

共同点：主流 agent 基准的主指标**没有一个是 LLM-as-judge**——能确定终态的比对终态
（SWE-bench 的测试、τ-bench 的数据库），能唯一答案的做 exact match（GAIA）。
设计题目时刻意让答案"可被确定性判卷"（GAIA 明确把"答案无歧义、易评分"作为选题标准）。

**对本项目的启示**：
- 本项目的确定性判卷（与冻结规则书真值比对）与主流 agent 基准的主指标口径完全一致，
  LLM-as-judge 在一线基准里从来不是主指标。【已采用】
- GAIA 的选题哲学值得反向学习：它把"能否确定性判卷"作为**出题阶段的筛选条件**
  （答案必须唯一无歧义），而不是出题后再找评分方法。本项目构建新题时，
  "该案真值能否追溯到冻结规则书条款、是否唯一"应继续作为入卷门槛。【已采用，可作明文门槛】
- τ-bench 的"只看终态、不看过程"提示：本项目终裁也只看判定结果与证据是否成立，
  不奖励"推理过程写得漂亮"，方向一致。【已采用】

### 5. LLM-as-judge 的已知偏差与缓解

- **偏差分类学的定义性文献**：Zheng et al.（NeurIPS 2023，MT-Bench / Chatbot Arena）
  命名并测量了三类系统性偏差——position bias（成对比较中偏好特定位置的答案，
  交换顺序后 GPT-4 判定一致率仅约 65%）、verbosity bias（偏好更长的答案）、
  self-enhancement bias（偏好自己生成的答案）；同时给出缓解：交换位置两次判、
  提供参考答案引导、将评分任务拆成独立维度（[arXiv 2306.05685](https://arxiv.org/abs/2306.05685)）。
- **自偏好有因果证据**：Panickssery et al.（NeurIPS 2024）证明 LLM 评审能识别自己的
  生成物，且识别能力与自偏好呈线性关系、受控实验支持因果解读——能力越强的模型
  自偏好可能越强（[arXiv 2404.13076](https://arxiv.org/abs/2404.13076)）。
  推论：审计/评审若用与被审对象同源同族的模型，结论系统性不可信。
- **缓解：多模型陪审团**：Verga et al.（NeurIPS 2024）"Replacing Judges with Juries"
  用多个不同来源的小模型组成评审团，比单一强模型评审更接近人类判断，且对个体偏差
  更稳健（[arXiv 2404.18796](https://arxiv.org/abs/2404.18796)）。
- **兜底缓解永远是确定性校验**：以上缓解都只是降低 LLM 评审的偏差，不是消除；
  一线基准（第 4 节）的应对是绕开——主指标不用 LLM 评审。

**对本项目的启示**：
- 本项目"判卷必须是确定性比对、LLM-as-judge 不得作主指标"的契约，与偏差文献和一线基准
  实践完全对齐。【已采用】
- Panickssery 的自偏好因果证据，直接论证了本项目"交叉审计用**异源**模型"的必要性：
  审计模型与生产模型若同源，它会在倾向性上包庇同类判定。【已采用，异源是刚性的】
- 交叉审计的定位（只标记分歧、不修改真值）正是"陪审团"的正确用法：模型分歧是**信号**
  （提示该案真值可能有问题、需人工复核），而不是**裁决**。【已采用】

### 6. holdout 纪律、contamination 防护与一致性测量

- **基准作者主动 holdout**：GAIA 扣留 300 题答案不公开，仅供官方排行榜评分
  （[arXiv 2311.12983 摘要](https://arxiv.org/abs/2311.12983)）；
  OpenAI 在 SWE-bench Verified 讨论中明确指出：SWE-bench 取材公开 GitHub，
  互联网语料预训练的模型"很可能已被污染"，这是静态公开基准的固有局限
  （[OpenAI 官方博客](https://openai.com/index/introducing-swe-bench-verified/)）。
- **contamination 的定量证据**：
  - GSM1k（Scale AI，2024）：仿 GSM8K 风格全新命制 1250 题，部分模型在新题上准确率
    比在 GSM8K 上掉最多 13%，且掉幅与"记忆信号"相关——证明部分基准分数含记忆成分
    （[arXiv 2405.00332](https://arxiv.org/abs/2405.00332)）。
  - GSM-Symbolic（Apple，ICML 2025）：把 GSM8K 题目符号化模板化后重采样，
    仅改动题目里的数值就造成模型表现明显波动，加入无关干扰从句后准确率暴跌
    （最高 65%）——"同题换皮"即可击穿表面高分（[arXiv 2410.05229](https://arxiv.org/abs/2410.05229)）。
- **一致性指标 pass^k**：τ-bench 的 pass^k = k 次独立试验**全部成功**的概率，
  与 pass@k（至少成功一次）方向相反，专门刻画 agent 的不可靠性；gpt-4o 在 retail 域
  pass^8 <25% 说明单次成功率掩盖了巨大的方差
  （[arXiv 2406.12045 摘要](https://arxiv.org/abs/2406.12045)）。
- **metamorphic testing（蜕变测试）在 LLM 评测的应用**：不依赖标准答案，而是检验
  "相关输入的输出之间必须满足的关系"（如改写措辞不应改变判定）。Cho et al.
  （ICSME 2025）整理了 191 条 NLP 任务的蜕变关系并给出自动化测试工具 LLMORPH
  （[论文 PDF](https://valerio-terragni.github.io/assets/pdf/cho-icsme-2025.pdf)）；
  安全评测方向已有专门实证研究，为 LLM 安全选择题定义 7 条蜕变关系做系统性测试
  （[Springer：Safety Evaluation for LLMs with Metamorphic Testing](https://link.springer.com/article/10.1007/s11390-026-5705-z)）。

**对本项目的启示**：
- 本项目 320 案考卷冻结、不进入 prompt/练习册，与 GAIA 扣答案、GSM1k 新题验证是同一
  纪律；需要持续保证考卷文本不出现在任何会进入训练语料的渠道（日志、文档、公开仓库）。【已采用】
- **pass^k 值得引入**：本项目每次评测是单次抽样，LLM 的判定方差会被单次结果掩盖。
  对考卷做 k 次重复采样并报告 pass^k（k 次判定全部与真值一致的比例）或至少报告
  逐案一致率，能揭示"碰运气判对"的案件，成本只是多跑几遍。【值得考虑】
- **metamorphic testing 值得引入为补充层**：对练习册案件做措辞级改写
  （同案换表述、换证据排列顺序、加无关噪音字段），判定不应改变——这能在不动真值体系
  的前提下暴露"模型记住了表面形式而非学会了规则"。GSM-Symbolic 证明这类测试
  检出率高、构造便宜。它不能替代真值考卷（蜕变关系只查一致性不查正确性），
  定位是 dev 阶段的健壮性回归测试。【值得考虑】

### 7. 交叉审计真值的先例

- **模型标记 + 人复核的标签审计**：Northcutt et al.（NeurIPS 2021 Datasets & Benchmarks，
  该 track 最佳论文提名）用 confident learning 在 10 个主流基准测试集里找标签错误，
  平均错误率约 3.4%（ImageNet 验证集约 5.8%、2916/50000 张被确认错标），修正后
  部分模型排名翻转（ResNet-18 反超 ResNet-50）——证明"基准真值不审就会系统性扭曲
  结论"。方法学形态正是本项目采用的：**模型负责标记可疑样本，人/规则负责最终确认**
  （[arXiv 2103.14749](https://arxiv.org/abs/2103.14749)，
  [cleanlab 论文列表](https://www.curtisnorthcutt.com/resources/pdf/northcutt-confident-learning-for-machines-and-humans.pdf)）。【已采用】
- **整卷人工审计的工业先例**：OpenAI 与 SWE-bench 作者合作，由 93 名开发者对 1699 个
  样本做三人独立标注、取最高严重度保守集成，结果 **68.3% 的样本因问题描述欠定或
  判卷测试不公而被剔除**，产出 SWE-bench Verified 500 题；修正后 GPT-4o 分数从 16%
  升到 33.2%——真值审计能成倍改变结论（[OpenAI 官方博客](https://openai.com/index/introducing-swe-bench-verified/)）。【已采用】

**对本项目的启示**：
- "用第二模型审标签、只标记不改真值"有直接的学术先例（Northcutt：模型标记 + 保守确认），
  且被证明不改审计就会扭曲结论——本项目的交叉审计不是过度工程，是基准卫生的底线配置。【已采用】
- SWE-bench Verified 的数字（68.3% 样本有问题）提醒：考卷冻结不等于考卷正确。
  交叉审计标记出的分歧案应有人工复核的固定流程，且复核结论要回溯到规则书条款
  （本项目"真值可追溯到冻结规则书"已提供这个锚点）。【已采用】

## 结论速览

1. 护栏框架分两派："代码判模型"（NeMo Guardrails、Guardrails AI、OpenAI Agents SDK
   tripwire）与"模型判模型"（Llama Guard、Moderation API）；前者是本项目路线的业内对应，
   后者只能当信号源，不能当终裁。
2. "模型提案 + 独立终裁"在工程界（tripwire、HITL 审批闸）、监管界（EU AI Act Art.14
   human oversight）、金融界（SR 11-7 独立验证）有三重先例；LangChain 把审批闸放到
   工具调用粒度，是值得参考的降人工成本的形态。
3. 权限与证据约束的共识是"长在代码上、不长在 prompt 里"；ALCE 证明引用要求必须配套
   支撑性校验，否则只是格式约束。
4. 主流 agent 基准（SWE-bench、τ-bench、GAIA、AgentBench）主指标全部是确定性判卷；
   GAIA 把"可确定性判卷"作为出题筛选条件，并主动扣留答案做 holdout。
5. LLM-as-judge 的三类偏差（立场/冗长/自偏好）有定义性测量，自偏好有因果证据——
   交叉审计必须用异源模型，且模型只能标记分歧、人定真值（与 Northcutt 标签审计、
   SWE-bench Verified 人工审计的先例一致）。
6. 两个值得本项目新增的机制：**pass^k 多次抽样一致性**（τ-bench 口径，揭示单次评测
   掩盖的判定方差）和 **metamorphic testing**（措辞改写不应改判，dev 阶段的健壮性
   回归层，不替代真值考卷）。
