# AGENTS.md —— product-review-agent（每会话注入，**预算 ≤3200 字符**）

> 本文件**入库**、随代码版本化，只放「任何贡献者 / 任何 agent 都必须知道」的契约；**代码即事实**，
> 此处只写红线。私有记忆（状态 / 决策台账 / 口径细节）在 gitignored 的 `.workbuddy/` 与 `.memory/`；
> 本机与 harness 坑在同目录 `AGENTS.local.md`（DSH 一并注入）。
> 自检：`python3 -c "import pathlib as p;print(len(p.Path('AGENTS.md').read_text(encoding='utf-8')))"`

## 项目一句话

电商商品内容治理的**复杂风险调查 Agent**：机审处理确定性异常，Agent 只处理复杂低置信案件，
动态取证后输出 `PASS / REJECT / HUMAN_REVIEW`。Python 3.12 + uv + LangGraph + Pydantic + FastAPI +
MySQL；全链路确定性可重放（scripted LLM + InMemory 世界）。

## 评测口径

- **0.964 不是能力得分**，它衡量 scripted 世界下与 GT（≈审查员可判定函数）的**实现一致性**；讲它必须
  紧跟 real 对照 —— 引用以 `docs/02-evaluation.md` §14 为准（重构后 real 320 单次 acc **0.850**）。
- real LLM 共三次**单次抽样**（v1 35 案 0.200 / v2 重构前 0.109 / 重构后 0.850）：**不可重放、不重复
  采样、不为改分调 prompt / 规则 / GT**。**不得与 scripted 混读**。
- 默认装配、`build_agent_graph()` 缺省与评测世界**仍是 InMemory / Mock** ⇒ 讲评测数字**不得说成
  「已接真库 / 真实 RAG」**。
- 评测脚本 `run_*.py` **封板后别随手重跑**；scripted 路径下 **token=0 / cost 空 / latency≈0 是真实
  情况，绝不伪造**。

## 代码不变量

1. **工具世界四份清单**：`build_tools()` = 6、`build_production_tools()` = 6（生产世界：商品 / 商家读
   MySQL，案例 / 政策读真实 RAG）、`make_eval_world_tools()` = 5（缺 OCR）；生产图装配唯一处
   `pra/wiring.py`。
2. **RAG 唯一后端 = chroma**（ChromaDB + LlamaIndex + BGE + BM25(jieba) + RRF），原 `local` 后端已整体删除。
3. **检索分数不做量纲适配、不设上界**（三模式互不可比；`bm25` 原始分**无界**）；
   `retrieval_score` / `Evidence.weight` 只余 `ge=0`；前者是**检索分**，**不得说成「语义相似度」**，
   也不参与 Gate 判定。
4. **hybrid 融合用库 `QueryFusionRetriever`**（`RECIPROCAL_RANK` / `num_queries=1` / `use_async=False`）
   —— ⛔ 别再自写融合函数。并列序 = **corpus 原序**。
5. **BM25 分词自持**（`rag/bm25.py`：jieba 切词 + `bm25s` 自建索引）；⛔ 不得全局替换 `bm25s.tokenize`；
   ⛔ 别把 BM25 搬进 Chroma。检索失败就抛异常，不自己补一套检索系统。
6. **过滤语义两处定义必须逐条等价**：向量路下推 Chroma `where`、BM25 路用 Python 谓词 —— 改一处必须
   同步两处（有测试锁死）。
7. **写侧 = `ChromaVectorStore` 的 delete-then-add**；建库必须显式 `space="cosine"`；**metadata 形状一改
   必须 bump collection 版本**（现 `v4`），否则复用旧库 → 新 `where` **静默零命中**。
8. 服务运行时 LLM = **scripted 桩**；real LLM 仅评测脚本用（`set_llm_backend` 注入）。
9. `PRA_LANGFUSE_ENABLED` **未设 = 启用**，`=0` 才是强制关闭；缺凭据回落 `NullTracer`（no-op）。
10. **两道 Gate 不读** `prior` / `posterior` / `Hypothesis.status` / `evidence_for`。
    PASS = required 全覆盖且阴性 ∧ 无阳性 ∧ 无规则阳性（R-102 / R-302）∧ 无关键失败 ∧ 无冲突；
    REJECT = 硬阳性 + 可引用依据 + dc≥0.7 + 无冲突。
11. HUMAN_REVIEW 归因码 = `R2_REJECT_GATE_FAIL` / `R3_*` / `R4_PASS_GATE_FAIL` /
    `R5_DEGRADED_OR_FAILED_STEP`；`R1_HARD_RULE` 是**唯一**直接产出 REJECT 的硬规则码。
    **不为缺席造证据**。
12. **两条业务红线**：① **仅商家历史脏不授权自动 REJECT**（需本 listing 外观 / 文本确证）；
    ② **R-102 品牌词只阻塞 PASS、不授权 REJECT**，**只有 R-302 规避词命中才授权 REJECT**。

## 非目标（别再提议）

- 基础设施：MQ / worker 化、OTel、真实视觉 / OCR 数据源、多标注者抽检、MCP、Redis 幂等、
  MySQL Checkpointer、HTTP 查询端点。
- 评测侧：Budget Utilization、按 scene 分层、单案成本折算、SHOULD 对抗负例、LLM-as-a-Judge、
  为校准补 0.75–0.88 数据带、Ablation 之外的扩张。

## 注释规范

- 只写「做什么 + 参数 / 返回值语义」，一两行说完。**禁止**：契约清单、为自己的写法辩护、跨模块断言
  （"下游 Gate 不读该值"）。**判据：是在描述代码做什么，还是在为自己的写法辩护？后者不写。**
- 不写 `docs/NN`、「拍板 / 本轮 / 用户决策」等内部口吻；枚举型全表只在**定义处**出现一次。
- 公开仓库只维护 `docs/00-system-design.md` + `docs/02-evaluation.md`，别再新建 .md
  （gitignored 的 `.memory/`、`.workbuddy/` 不受此限）。

## 变更纪律

- 中文 conventional commit；**删功能要连「为它服务的件」一起删**（参数 / 分支 / 守卫 / 测试 / 文档），
  动手前先扫它的全部引用。
