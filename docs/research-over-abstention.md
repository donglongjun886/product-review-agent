# 调研：LLM agent 过度弃权（over-abstention）的优化实践

- 调研日期：2026-10-15
- 调研问题：
  1. Selective prediction 的 risk-coverage 理论基础与分场景弃权策略
  2. LLM 过度弃权/过度拒绝的测量与校准对策
  3. Prompt 层减少过度谨慎的具体手法
  4. 评测驱动的 prompt 迭代纪律（防 dev 集过拟合）
  5. 内容审核行业的自动化率与人机分工实践
- 背景：本项目是电商商品审核 agent，selective classification 口径（coverage / selective risk）。
  当前风险 0% 但覆盖率仅 52%，根因是对"商家有轻微下架前科的干净商品"过度弃权。
  修复限定在 prompt 层；业务上误拒成本 ≫ 多余转人工成本。

## 1. Selective prediction 理论基础

- 弃权（reject option）的源头是 Chow 1970：在已知类别后验时，最优拒绝规则就是对置信度设阈值，
  换取 error-reject trade-off（[JMLR 11 综述引用, El-Yaniv & Wiener 2010](https://www.jmlr.org/papers/v11/el-yaniv10a.html)）。
- El-Yaniv & Wiener 2010 形式化了 selective classification：分类器是 (预测器 f, 选择器 g) 二元组，
  用 risk-coverage (RC) 曲线刻画"牺牲多少覆盖率换多少准确率"，是整个领域的定义性文献
  （被引 900+）：[JMLR 论文页](https://www.jmlr.org/papers/v11/el-yaniv10a.html)。
- Geifman & El-Yaniv 2017 给出深度学习版：后验置信度（softmax 最大响应）+ 选定阈值，
  以高概率保证选择性风险不超过目标值（[arXiv 引用综述](https://arxiv.org/html/2110.14914v2)）；
  2019 年 SelectiveNet（ICML）把拒绝头联合训练进网络，直接优化"给定目标风险下的最大覆盖率"
  （[ICML oral PDF](https://geifmany.github.io/papers/icml_oral.pdf)）。
- 分子群/分场景弃权是有文献依据的方向：
  - Shah et al. 2022（ICML，Selective Regression Under Fairness Criteria）证明全局统一阈值可能
    扩大子群间性能差距，需要按子群处理（[MIT PDF](https://sia.mit.edu/wp-content/uploads/2022/08/2022-shah-bu-lee-das-panda-sattigeri-wornell-icml.pdf)）。
  - Casacuberta et al. 2025（Selective Omniprediction and Fair Abstention）研究按域子集分布设置
    弃权策略（[Oxford ORA PDF](https://ora.ox.ac.uk/objects/uuid:f7e23d48-5f82-4ccf-bee9-deee345d4209/files/s1r66j349x)）。
  - 更一般地，公平性文献中"为不同敏感组设不同决策阈值"是标准做法
    （[TU/e 公平性综述](https://pure.tue.nl/ws/files/356540689/20250506_Weerts_hf.pdf)，引 Hardt et al.）。

**对本项目的启示**：
- RC 曲线就是 coverage vs selective risk 的正式理论框架——52% 覆盖率 @ 0% 风险说明当前工作点
  远在曲线左下角的保守区，有巨大的"免费覆盖率"可挖。
- 公平性文献给"分场景弃权"背书：干净题（64%）和违规题（90%）覆盖率差 26 个点，正是
  全局单一弃权策略在子群上失配的表现。在 prompt 层对不同子群给不同的弃权规则表述，
  是"分群阈值"在 prompt-only 约束下的合法实现。
- 注意 Chow 规则假定有可靠的置信度；LLM 的弃权是"语言行为"而非校准过的概率，
  所以不能套用数值阈值，只能改弃权规则本身（prompt）——这与本项目修复限定一致。

## 2. LLM 过度弃权/过度拒绝的测量与对策

- 过度拒绝（over-refusal / exaggerated safety）是被系统性测量的问题：
  - **XSTest**（Röttger et al., NAACL 2024）：250 条"字面危险、语义无害"的安全提示 +
    200 条不安全对照，专测模型对安全提示的误拒。论文发现加上安全 system prompt 会显著
    增加过度拒绝——**prompt 措辞本身就是过度谨慎的来源之一**
    （[arXiv 2308.01263](https://arxiv.org/abs/2308.01263)，[NAACL 2024](https://aclanthology.org/2024.naacl-long.301/)）。
  - **OR-Bench**（Cui et al., ICML 2025）：8 万条"看似有毒实则无害"的 prompt，
    10 个类别，用于大规模测量各模型家族的过度拒绝（[arXiv 2405.20947](https://arxiv.org/abs/2405.20947)）。
- 弃权行为的通用测量：
  - 综述 The Art of Refusal（Wen et al. 2024）从 query / model / human values 三个维度
    整理了弃权方法、基准与指标（[arXiv 2407.18418](https://arxiv.org/abs/2407.18418)）。
  - **Abstain-QA**（Madhusudhan et al., COLING 2025）：LLM 的弃权校准很差；但 CoT、
    上下文学习、以及**在 prompt 中显式强调弃权选项**都能实质改善弃权行为
    （[arXiv 2407.16221](https://arxiv.org/html/2407.16221v1)，[COLING 2025 PDF](https://aclanthology.org/2025.coling-main.627.pdf)）。
  - **AbstentionBench**（Meta, 2025）：20 个前沿模型上弃权仍是未解问题；推理微调平均
    使弃权能力退化 24%；精心设计的 system prompt 能提升弃权但不解决根本校准问题——
    反向读就是：**prompt 是双向杠杆，往哪边推取决于措辞**
    （[arXiv 2506.09038](https://arxiv.org/abs/2506.09038)）。
- 工业界降误拒的对策：
  - Anthropic Constitutional Classifiers：护栏分类器对无害流量的拒答增量只有 +0.38%；
    下一代（Constitutional Classifiers++）用轻量探针先行、只对被标记的流量调用重分类器，
    生产流量拒答率降 10 倍（[Anthropic 博客](https://www.anthropic.com/research/constitutional-classifiers)，
    [CC++ 论文](https://arxiv.org/html/2601.04603v1)）。要点：降误拒靠的是**分级/分层处理**，
    而不是把单一系统调得更宽松。
  - OpenAI safe-completions（GPT-5）：硬性拒答边界在意图模糊的双重用途场景上太脆，
    改为"以输出为中心"的安全训练——在策略约束内尽可能帮助，而非二元拒绝
    （[arXiv 2508.09224](https://arxiv.org/html/2508.09224v1)，[OpenAI 官方](https://openai.com/index/gpt-5-safe-completions/)）。

**对本项目的启示**：
- 本项目的过度弃权本质就是 XSTest/OR-Bench 测的同一个病：表面风险特征（商家前科）
  触发对干净样本的误杀。可借鉴它们的评测构造思路：把"有轻微前科的干净商家"作为显式的
  over-abstention 对照子集在 dev 集里单独计分（本项目已按干净/违规分别报覆盖率，方向正确）。
- prompt 是双向杠杆的证据充分：安全措辞会增加误拒（XSTest），弃权指令会改变弃权率
  （Abstain-QA、AbstentionBench）。因此 prompt 层修复"用更精确的放行条件措辞来降低误弃权"
  有文献支撑，不是玄学。
- Anthropic 的分层思路（轻探针先行、只升级可疑流量）对应本项目的"确定性 Gate + 模型调查"
  架构：过度弃权出现在模型侧，Gate 侧不动即可控制风险不变。

## 3. Prompt 层的具体手法

- **显式弃权语义**：Abstain-QA 显示在 prompt 中明确写出"何时该弃权"能实质改善弃权校准
  （[arXiv 2407.16221](https://arxiv.org/html/2407.16221v1)）。推论：反过来，明确写出
  "何时**不该**弃权"（正面定义放行条件）是同一机制的对称用法。该对称用法本身未找到
  直接一手来源，属合理外推。
- **示例驱动**：Abstain-QA 与 AbstentionBench 都显示 in-context examples 能移动弃权倾向
  （来源同上）。在 prompt 里放"有前科但判 PASS"的正例，是把目标子群的弃权规则
  显式化的直接手段（此具体用法未找到一手来源）。
- **举证要求**：让模型"弃权必须引用具体证据缺口、禁止以画像先验为由弃权"——未找到
  一手来源；最接近的是 Abstain-QA 的显式弃权 prompt 与研究 agent 评测中的 groundedness
  检查（claims 必须有检索来源支撑，见 Anthropic 评测博客
  [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)）。
- **警惕负向措辞堆叠**：XSTest 的核心教训是"安全导向的 system prompt 会系统性地抬高误拒率"
  （[arXiv 2308.01263](https://arxiv.org/abs/2308.01263)）。prompt 里每多一句"宁可错杀"式的
  兜底警告，都在把模型往弃权方向推；修复时应优先**删除/改写**这类措辞，而不是再加新规则。
- **成本不对称要显式写出**：OpenAI safe-completions 论文论证"二元拒绝边界在意图模糊时脆弱"
  （[arXiv 2508.09224](https://arxiv.org/html/2508.09224v1)）；把业务成本不对称（误拒 ≫ 多余转人工）
  翻译成 prompt 中的决策原则属于常识做法，但"在审核 prompt 中显式声明成本矩阵"这一具体手法
  未找到一手来源。

**对本项目的启示**：
- 最有文献支撑的三招：① 正面定义"满足哪些条件即应 PASS"；② 加"有前科但干净"的正例；
  ③ 要求弃权必须列出"缺哪条具体证据"，禁止以商家画像单独作为弃权理由——
  注意 ③ 与项目正确性契约"画像不得单独构成自动拒绝依据"完全同构，等于把硬约束
  翻译成模型可执行的弃权门槛。
- 先做减法：检查 system prompt 里所有"谨慎/存疑即转人工"式的宽泛兜底句，
  它们是 XSTest 意义上已知的过度弃权诱因。

## 4. 评测驱动的 prompt 迭代纪律

- Hamel Husain（Parlance Labs，AI Evals 课程）的规范流程（[AI Evals FAQ](https://hamel.dev/blog/posts/evals-faq/)）：
  - 数据三分：train（写进 prompt 的示例）/ dev（迭代调 prompt 用）/ test（封存，只用于最终验收）。
    **每用 dev 结果改一次 prompt，dev 信息就泄漏一次**；多轮后在 dev 上好、新数据上差，就是过拟合，
    即使从未把 dev 样本直接写进 prompt。
  - 若 test 明显差于 dev，需换新一批未接触过的 test 重做。
  - 判断指标优先 binary（pass/fail）而非 Likert 打分。
  - 自动化 prompt 调优（如 GEPA）之前必须先人工看 disagreement，否则会掩盖非 prompt 类问题
    （缺上下文、标签错误等）。
  - 每种 failure mode 建独立的小 evaluator，比"一个 judge 抓所有错"更易对齐、更可行动。
  - TPR/TNR 要同时跟踪：抓更多错往往以更多误报为代价，接受水平按业务后果定；
    失败案例稀少时，很小的误报率也会制造大量无效人工复核——**这正是本项目 48% 转人工的成因**。
- Anthropic 的 agent 评测方法论（[Demystifying evals for AI agents, 2026-01](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)）：
  - grader 分三类（代码/模型/人工），能确定性判的就不要用模型当裁判——与本项目契约一致。
  - 区分 capability eval（爬坡用，初始低通过率）与 regression eval（防退化，近 100% 通过率）；
    优化 agent 时必须同时跑，防止修一个失败模式引入另一个。
  - eval 套件是活的资产，需要长期维护与专人负责。

**对本项目的启示**：
- 本项目 v3-dev 160 案练习 / v3 考卷 320 案封存的拆分，正是 Hamel 三分法的落地；纪律上再补一条：
  记录"用 dev 迭代了多少轮"，轮数越多对 dev 分数的信任折扣越大。
- 把"过度弃权"按 Hamel 的建议做成独立 evaluator：单独统计"干净题上的转人工率"作为
  回归指标（已在做），防止任何 prompt 改动在压低弃权的同时引入风险。

## 5. 内容审核行业实践

- Meta 2025 年 Q1 诚信报告：主动提高自动化系统的处置阈值，要求更高置信度才移除内容；
  对低严重度违规改为依赖用户举报，自动化聚焦高严重度类别
  （[Meta Transparency Center, Integrity Reports Q1 2025](https://transparency.meta.com/reports/integrity-reports-q1-2025/)）。
  要点：平台按"严重度 × 置信度"分档自动化，而不是追求统一的自动化率。
- Digital Trust & Safety Partnership（DTSP，成员含 Meta/Google/Microsoft 等）发布过
  AI 自动化在 Trust & Safety 的最佳实践白皮书（[DTSP PDF](https://dtspartnership.org/wp-content/uploads/2024/09/DTSP_Best-Practices-for-AI-Automation-in-Trust-Safety.pdf)）。
- 行业通行的三段式：高置信违规自动处置 / 灰区转人工 / 高置信干净自动放行；
  具体阈值与自动化率数字各平台基本不公开（未找到一手来源的具体数值）。

**对本项目的启示**：
- Meta 的做法验证了"按违规严重度分档自动化"的合理性：IP 仿冒属于高严重度类目，
  对这类目提高覆盖率（压低误弃权）的业务代价可控，因为硬规则与证据 Gate 已经兜住底线。
- "灰区转人工"不是失败而是设计形态；优化目标应该是把"轻微前科但证据干净"从灰区
  重新分类为"高置信干净"，而不是消灭转人工本身。

## 结论速览

1. 本项目的问题在文献里有名字（over-refusal / exaggerated safety）、有基准（XSTest、OR-Bench）、
   有理论框架（selective classification 的 RC 曲线）。
2. prompt 是双向杠杆：措辞既会制造过度谨慎（XSTest），也能校准弃权（Abstain-QA）。
3. 分场景弃权（按子群给不同弃权规则）有公平性文献背书；在 prompt-only 约束下，
   实现方式是"正面定义放行条件 + 目标子群正例 + 弃权举证要求"。
4. dev/考卷拆分的纪律正确，需防"迭代轮数"造成的隐性过拟合；过度弃权应作为独立回归指标。
5. 行业实践支持按严重度分档自动化；灰区转人工是正常设计，优化的是灰区的归类而非存在。
