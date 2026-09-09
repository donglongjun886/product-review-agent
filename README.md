# product-review-agent

电商平台商品内容治理 · **复杂风险调查 Agent**（Complex Risk Investigation Agent）

> **定位**：传统机审负责**确定性异常**（黑名单 / 关键词 / Logo / 类目 / OCR / 分类模型）——快、便宜、可解释；Agent 只处理**开放性、上下文依赖强、需要多源交叉验证的复杂风险案件**，动态取证后输出三分类裁决：`PASS` 放行 / `REJECT` 违规拒绝上架（须有可引用政策依据）/ `HUMAN_REVIEW` 主动克制地转人工（证据不足、置信不足、预算耗尽、政策模糊或关键工具失败时）。

生产主线：商品上架/变更 → 传统规则 + 模型初筛（同步、快、便宜）→ 三分流（明确正常 / 明确违规 / 复杂低置信）→ 复杂低置信案件投递 **Agent**（LangGraph 编排）→ `PASS / REJECT / HUMAN_REVIEW` → 人工裁决回流案例库 / 策略库 / 评测集。

## 系统总链路

```
HTTP 接入（同步；MQ 异步 worker 为规划）
  │  POST /api/v1/reviews，请求体 = ProductReviewCase
  ▼
ReviewCase（输入快照：商品 / 商家 / 事件类型 / 机审信号）
  ▼
build_initial_state(case) → AgentState（hypotheses / evidence / budget / …）
  ▼
LangGraph 调查子图（5 节点 7 边单回环）：
  hypothesize（风险假设，prior 显式给出）
    → plan（下一步调查计划）
    → tools（6 个调查工具：product 商品事实 / image_analysis 图片分析 /
             ocr / merchant 商家历史 / case_search 先例 / policy_search 政策）
    → reevaluate（证据综合、假设 posterior 更新）
    → 未收敛 → 回到 plan（收敛循环，受预算 / 降级 / 去重护栏约束）
    → 收敛或护栏触发 → decide
  ▼
Decision Gate：LLM 只产 DecisionProposal 提案，确定性 overlay 收口
  （R1 硬规则强制 REJECT → abstention 清单 → PASS/REJECT Gate）
  ▼
PASS / REJECT / HUMAN_REVIEW（DECIDED 唯一终态）
  ▼
五表落库（MySQL）：review_case / review_run / review_trace / review_evidence / review_result
```

Agent 采用「**假设验证**」而非「直接分类」：hypothesize 显式给出风险假设的先验，reevaluate 依据工具取证结果更新为后验（SUPPORTED / REFUTED / UNRESOLVED），证据链与假设轨迹全程可回溯，供解释与评测重放。

## 目录结构

```
src/pra/
├── agent/        复杂风险调查子图（LangGraph）：state（AgentState + 证据去重 reducer）/
│                 graph（build_agent_graph，5 节点 7 边单回环）/
│                 nodes（hypothesize / plan / reevaluate / decide）/
│                 tools_node（工具执行 + 质量过滤 + extra 回填）/
│                 guardrails（budget 预算 / converge 收敛 / dedup 去重 / errors /
│                            evidence / hard_rules R1 / gate 决策 overlay /
│                            llm_shell 后端抽象 / metrics / schemas）/
│                 checkpointer（InMemorySaver）/ scripted_llm（确定性 LLM 桩）
├── api/          HTTP 接入面：app（create_app）/ routes（POST /api/v1/reviews、
│                 GET /api/v1/health）/ schemas（ReviewRunResult）/ service（run_review）
├── infra/        MySQL 接入 + 落库编排：db（Settings / engine）/ rdb_models /
│                 persist_service（run_and_persist，五表闭环）
├── domain/       领域契约（Pydantic v2，extra="forbid"）：ProductReviewCase /
│                 Evidence / Hypothesis / ReviewDecision / Decision 三分类
├── tools/        6 个调查工具（默认注入 InMemory 数据源，开箱可跑）：
│                 product / image_analysis / ocr / merchant / case_search / policy_search
├── screening/    机审初筛 + 三分流（rule_engine / triage）：terms（规则词表单一来源，
│                 Screening 与 Agent R1 硬规则共用）/ rules（R-101 REJECT、
│                 R-102/301/302 COMPLEX —— R-102 品牌词交 Agent 调查，Q-1 拍板 B）/ engine（triage 纯函数 + RULE_HIT 证据）
├── evaluation/   评测 harness（确定性、不落 DB）：runner 编排 + 三 scheme（Rule 复用 screening
│                 三分流 / Single-call LLM / Agent 图执行）、business + abstention 指标、
│                 ablation（方案级 2a/2b/2c + 组件级）、sweep（Evidence 阈值单参数）、
│                 regression（三方案决策序列 hash 比对基线）
├── rag/          Policy KB + Case KB（RAG_CASE_ 前缀、与评测 GT 隔离）：BM25 / Vector /
│                 Hybrid 三模式检索；实现 tools 层 PolicyIndex / CaseIndex Protocol
│                 （Tool 层零改动，向量库替换经工厂注入点接入）
└── common/       通用工具：雪花 ID / JSON 工具 / 错误码

docs/         设计文档（见「文档索引」）
migrations/   MySQL 核心表 DDL：001_review_core_tables.sql（5 表）/ 002_review_case_triage.sql（增量：review_case.triage_result）
scripts/      demo_walkthrough.py（端到端走查）/ demo_api.py（执行器演示）/
              run_evaluation.py（三方案评测）/ run_ablation.py / run_sweep.py /
              run_regression.py（回归比对）/ run_rag_demo.py（RAG 检索演示）/
              run_evaluation_real.py（真实 LLM 评测对照，需 API key）/ eval_dataset_gen.py
tests/        pytest 用例（确定性 mock，无网络 / API key：screening / evaluation / rag / agent / api / infra）
```

## 评测与结论

评测**方法与口径**（三方案定义 / 指标口径含 abstention / Ablation / Sweep / 里程碑与阶段验收 / 结论边界）
以 [docs/02-evaluation.md](docs/02-evaluation.md) 为权威出处，README 不承载跑分数字，避免随评测集与模型迭代过期；
评测链路**确定性可重放**（scripted LLM + InMemory/RAG 种子数据，无真实 LLM / 网络），复现命令见「快速开始」与
`scripts/`（run_evaluation / run_ablation / run_sweep / run_regression / run_evaluation_real）。

## 快速开始

环境：Python 3.12+ · [uv](https://docs.astral.sh/uv/)。

```bash
# 1. 安装依赖（uv 自动创建 .venv 并同步默认 + dev 组：pytest/ruff/httpx）
uv sync

#   可选组 = 额外能力依赖（向量库化 / 可观测性，见「技术栈」）；基础链路无需额外组
uv sync --extra rag --extra observability
```

### 端到端走查（无 API key）

默认注入**确定性 scripted LLM 桩** + 6 个 InMemory 数据源工具 + InMemory checkpointer，
无任何外部依赖即可端到端跑通调查子图并执行内置断言：

```bash
uv run python scripts/demo_walkthrough.py
```

走查案件为「P_88231 复古运动鞋 / 商家 M_5512 / NEW_LISTING」：打印每个节点关键内容后
输出决策摘要，预期（确定性、可复现）：`HUMAN_REVIEW` / `HIGH` /
`[POTENTIAL_IP_RISK, EVASION_PATTERN]` / `decision_confidence=0.87` / `overrides=[]`。

### 评测 / RAG / 回归（确定性，无需外部依赖）

与走查同构的确定性链路（scripted LLM + 种子数据，无真实 LLM / 网络），CI 可直接复用：

```bash
# 三方案对比（默认 eval_data/v1；--data 可切其他数据集）：按 scheme 逐行打印决策指标与成本
uv run python scripts/run_evaluation.py

# 使用 v2 数据集（路径即数据源；方法 / 口径与里程碑见 docs/02-evaluation.md）
uv run python scripts/run_evaluation.py --data eval_data/v2/cases_v2.jsonl

# RAG 检索演示：固定 query × Policy/Case KB × BM25/Vector/Hybrid 三模式 → Top-K 命中与证据引用
uv run python scripts/run_rag_demo.py

# 回归：三方案决策序列 sha256 与基线快照比对（基线缺失时首次自动记录；此后一致输出 PASS）
uv run python scripts/run_regression.py
```

### Real LLM 评测（需 API key，真实调用有费用）

```bash
# 冒烟：先 --limit 小批量确认链路与成本量级
uv run python scripts/run_evaluation_real.py --limit 10

# 全量 real vs scripted 对照（默认 eval_data/v1 全集），结果写 JSON
uv run python scripts/run_evaluation_real.py --out /tmp/real_v1.json

# 抽样：--data 指定其他数据集 / --world rag 切 RAG 世界（同一 eval 世界，差异只归因于 LLM）
uv run python scripts/run_evaluation_real.py --data eval_data/v2 --limit 10 --out /tmp/real_v2.json
uv run python scripts/run_evaluation_real.py --world rag --limit 10
```

API key / base URL：`--api-key` / `--base-url` 显式传，或写仓库根 `.env` 的
`DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL`（脚本自动注入，值不入日志/报告/JSON）。

### API 演示

```bash
# 执行器级（不经 HTTP、不落库）：直接跑 run_review，验证接入 → 执行链路
uv run python scripts/demo_api.py

# 起 HTTP 服务（默认 http://127.0.0.1:8000）
uv run uvicorn pra.api.app:app --reload

# 存活探针
curl http://127.0.0.1:8000/api/v1/health
# → {"status": "ok"}

# 受理一次审核：请求体 = ProductReviewCase JSON（P_88231 示例字段与走查一致）
curl -X POST http://127.0.0.1:8000/api/v1/reviews \
  -H 'Content-Type: application/json' \
  -d '{
    "case_id": "CASE_20240907_001",
    "product": {
      "product_id": "P_88231",
      "title": "新款厚底复古跑鞋 女士百搭运动鞋",
      "description": "复古厚底设计，舒适百搭，适合日常通勤与运动。",
      "category": "女鞋/运动鞋",
      "brand": null,
      "sku_list": [{"sku_id": "S_1", "color": "米白", "size": "38", "price": 219.0}],
      "images": [{"url": "https://cdn.example.com/products/P_88231/img1.jpg", "source": "主图"}],
      "listing_time": "2024-09-06T14:00:00",
      "version": 3
    },
    "merchant_id": "M_5512",
    "event_type": "NEW_LISTING",
    "screening_signals": [
      {"name": "KEYWORD", "result": "PASS", "score": 0.8},
      {"name": "LOGO_DETECT", "result": "PASS", "score": 0.2},
      {"name": "CATEGORY_RULE", "result": "PASS", "score": 0.95},
      {"name": "DUPLICATE_CHECK", "result": "PASS", "score": 0.1}
    ]
  }'
# 响应信封：{"run_id": "<uuid4 hex>", "review_decision": {decision/risk_level/risk_type/...}}
```

注意：`POST /api/v1/reviews` **受理即 Screening 三分流**（`persist_service.process_review`）：
verdict=COMPLEX 的案件同步执行调查图**并落库**（Agent run，`run_and_persist`）；
PASS/REJECT 的案件由规则**直判落库**（`run_screening_direct`，trigger_type=SCREENING_DIRECT，
无 trace、终裁即 DECIDED），需要本机 MySQL 可达且已建表（含 002 的 review_case.
triage_result 列，见下节）；triage/图执行/落库异常统一返回 500（detail 为人类可读
信息，不暴露堆栈）。

### 数据库（可选，落库闭环需要）

五张核心表：`review_case` / `review_run` / `review_trace` / `review_evidence` / `review_result`。

```bash
# 1) 起本地 MySQL 容器（容器名 mysql-dev；口令自定，需与 DATABASE_URL 一致）
docker run --name mysql-dev -e MYSQL_ROOT_PASSWORD=<your-password> -p 3306:3306 -d mysql:8

# 2) 建库 + 执行核心表 DDL
docker exec -i mysql-dev mysql -uroot -p<your-password> \
  -e "CREATE DATABASE IF NOT EXISTS product_review CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
docker exec -i mysql-dev mysql -uroot -p<your-password> product_review \
  < migrations/001_review_core_tables.sql

# 3) 连接串覆盖（可选）：复制模板并按环境填写真实口令（.env 已被 gitignore，勿提交真实密码）
cp .env.example .env   # 编辑 DATABASE_URL=mysql+aiomysql://<user>:<pass>@127.0.0.1:3306/product_review
```

未配置 `.env` 时，代码内置的**本地开发默认 DSN** 指向 `127.0.0.1:3306/product_review`
（见 `pra.infra.db.Settings`），仅适用于步骤 1 的本地容器；其他环境必须经 `.env` 覆盖。

### 测试

```bash
uv run pytest tests/ -q   # 全确定性 mock，无网络 / 无 API key，CI 可直接复用
```

覆盖：screening 三分流、evaluation（三方案 / 指标 / 数据集 schema / ablation）、rag 检索、
agent guardrails（budget / converge / errors / gate / llm_shell / scripted_llm）、api routes、
persist 落库等。

## 关键设计决策速查

- **LLM 只提案、确定性 Gate 把关**：decide 节点先让 LLM 产 `DecisionProposal` 提案，
  再由 `run_decision_overlay` 确定性收口——R1 硬规则强制 REJECT（不可被 LLM 覆盖）
  → abstention 清单（预算耗尽 / 关键证据冲突 / 关键工具失败 / 政策不确定等）→
  PASS/REJECT Gate 校验；任何改判原因码（`R1_HARD_RULE` / `R2_*` / `R3_*` / `R4_*` /
  `R5_*`）全量写入 `ReviewDecision.overrides`，保证「谁把 PASS 改成了 HUMAN_REVIEW」
  可审计。（docs/01 §7、docs/03 T-8）
- **decision_confidence 是确定性安全门槛、非模型概率**：由证据/假设按固定公式重算
  （LLM 的 confidence 只作参考不作终值）；`0.7`（CONFIDENCE_ABSTAIN_THRESHOLD）只作为
  REJECT 的自动安全门槛——不达标即转 HUMAN_REVIEW；PASS 另有独立的 Gate 校验
  （R4_* 归因），但不设置信门槛。（docs/00 §7.4、docs/03 T-4）
- **预算 10 LLM / 15 Tool / 40k tokens / 30s 是 Guardrail 上界而非目标**：常态案件
  远低于上限（走查 8 LLM / 5 Tool）；超限不是失败，而是「带部分证据转人工止损」，
  归因记 `R3_BUDGET_EXHAUSTED`。（docs/03 T-7）
- **证据去重与回填全确定性**：去重指纹 `(type, source, ref_id)`——ref_id（image_url /
  product_id / merchant_id / case_id / clause_id 等稳定业务标识，O-1）优先、缺失才回退
  value，防同 (type, source) 的多条无 ref 证据互相吞并；`extra` 派生数值
  （similarity / removals / policy_id…）由确定性层统一回填（O-8），证据一旦收集不可篡改。
- **DECIDED 是唯一终态**：decide 之后无任何出边；ESCALATED / BUDGET_EXCEEDED 不再作为
  主终态，超限/降级只记入 overrides 与预算快照。（docs/03 T-8）
- **Checkpoint 与业务表分离**：LangGraph 线程 checkpoint（InMemorySaver）只服务断点续跑 /
  eval 重放；业务真相（五表）由 `persist_service.run_and_persist` 落 MySQL——内存不是
  真相源。（docs/04 §7.2/§7.4；MySQL Checkpointer 为规划）
- **评测确定性重放**：evaluation 全链路 scripted（无真实 LLM / 网络 / 随机），同 case 决策
  序列 sha256 可回归；方案可比基于同一输入快照（docs/02 §3.1），Agent 高分须以「标注-审查员
  同口径」的耦合 + InMemory 种子数据为边界解读（docs/02 §3.4/§7.2），不以「真实 LLM 能力」
  外推。
- **RAG MVP 三模式并排 + 数据隔离**：BM25 / Vector / Hybrid 同一 corpus 并排输出、不预设
  Hybrid 最优；Case KB（RAG_CASE_ 前缀）与评测 GT 数据集物理隔离，互不污染；MVP 的
  embedding 为确定性 mock，只验证检索链路与可重放性，不包装成语义检索。（docs/02 §7.2）

## 文档索引

- [docs/00-system-design.md](docs/00-system-design.md) —— 系统设计总览（业务价值 → 决策难点 → 系统设计 → 验证结果；§15 src 布局）
- [docs/01-agent-loop.md](docs/01-agent-loop.md) —— Agent Loop 细化：节点 / 边 / 状态契约（实现层契约）
- [docs/02-evaluation.md](docs/02-evaluation.md) —— 评测方案：三方案定义（公平性前提）/ 指标口径（含 abstention）/ Ablation / Threshold Sweep / 里程碑与阶段验收（§8）
- [docs/03-decisions.md](docs/03-decisions.md) —— T-1~T-12 参数与语义拍板表（Decision Gate 口径等）
- [docs/04-graph-design.md](docs/04-graph-design.md) —— LangGraph StateGraph 正式设计（graph.py 实现前的最后设计）

## Roadmap（方向与动机）

本节只列**方向与动机**，不标注完成状态、不引用进度事实；评测向的阶段性里程碑与验收规划以
[docs/02-evaluation.md](docs/02-evaluation.md) §8 为准。

- **RAG 向量库化（语义检索）**：以真实 embedding + 向量库（Qdrant）替换确定性 mock embedding；
  动机——mock 只验证检索链路与可重放性（docs/02 §7.2 边界），语义检索价值留待向量库化后验证；
  `pra.rag.factory` 注入点与 pyproject `rag` extra 已为此预留装配位。
- **MQ 异步 worker + 人工审核队列**：动机——HTTP 同步受理受吞吐 / 并发限制，异步化（含 MySQL
  Checkpointer、Redis 幂等）支撑接入解耦、削峰与事件溯源（docs/00 §9.3）。
- **可观测性**：Langfuse（LLM 调用级）+ OpenTelemetry（业务链路 trace）；动机——真实 LLM 接入后
  需要 prompt / token / 成本与行为观测数据驱动迭代。
- **Screening 策略库化**：真实品牌黑名单经规则层注入、词表沉淀为可维护的策略库；动机——规则词表
  已是单一来源，随 R2/R3 归因观测与词表调优需要更结构化的策略管理。

## 技术栈

| 关注点 | 选型 | 状态 |
|---|---|---|
| 语言 / 包管理 | Python 3.12+ · uv | 已用 |
| 服务接入层 | FastAPI（`POST /api/v1/reviews` 同步执行 + 落库） | 已用 |
| 复杂调查编排 | LangGraph StateGraph（5 节点 7 边单回环 + InMemory Checkpointer） | 已用 |
| 领域/校验 | Pydantic v2（契约 DTO，`extra="forbid"`） | 已用 |
| LLM | `LLMBackend` 抽象：默认确定性 scripted 桩（无 key 可跑）；`LiteLLMBackend`（litellm 真后端，四节点完整 prompt）经 `set_llm_backend`/`build_agent_graph(llm=)` 注入 | 已用（桩 + 真后端） |
| 评测 | `pra.evaluation`：三方案 harness + business/abstention 指标 + ablation + sweep + regression（确定性重放） | 已用 |
| RAG | `pra.rag`：Policy KB + Case KB，BM25 / Vector / Hybrid（numpy + mock embedding，无外部依赖） | 已用（确定性 mock embedding；语义向量化见 Roadmap） |
| 数据层 | SQLAlchemy 2.0 async · aiomysql · MySQL 五表（DDL：migrations/001…）；Alembic 依赖就绪 | 已用（DDL 经 migrations/ 直执行） |
| 规划 extras | Redis 幂等 / MQ worker；RAG 向量库化：Qdrant + 本地 embedding；可观测：Langfuse + OpenTelemetry | 规划（pyproject optional groups 已声明） |
