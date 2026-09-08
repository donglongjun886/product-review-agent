# product-review-agent

电商平台商品内容治理 · **复杂风险调查 Agent**（Complex Risk Investigation Agent）

> 传统机审负责**确定性异常**（黑名单 / 关键词 / Logo / 类目 / OCR / 分类模型），Agent 负责**开放性、上下文依赖强、需要多源交叉验证的复杂风险案件**，产出 `PASS / REJECT / HUMAN_REVIEW`，并在证据不足时主动**克制地转人工**。

系统叙事（主线）：商品上架/变更 → 传统规则 + 模型初筛（同步、快、便宜）→ 三分流（明确正常 / 明确违规 / 复杂低置信）→ 复杂低置信投递 **Complex Risk Investigation Agent**（LangGraph 编排，异步、预算受限）→ `PASS / REJECT / HUMAN_REVIEW` → 人工裁决后回流案例库 + 策略库 + 评测集。

## 技术栈

| 关注点 | 选型 |
|---|---|
| 语言 / 包管理 | Python 3.12+ · uv |
| 服务接入层 | FastAPI（同步机审 / 审核工作台接口） |
| 复杂调查编排 | LangGraph（StateGraph + MySQL Checkpointer 可恢复） |
| LLM 调用 | litellm（统一多模型网关） |
| 数据层 | SQLAlchemy 2.0 async · MySQL（aiomysql）· Alembic 迁移 |
| 缓存 / 幂等 | Redis |
| 可选 extras | RAG：Qdrant（qdrant-client）；可观测：Langfuse + OpenTelemetry |

> 模块划分遵循 `docs/00-system-design.md` §15 的 src 布局：`src/cg/{common,domain,screening,agent,tools,rag,evaluation,api,infra}`。

## 快速开始

```bash
# 1. 安装依赖（uv 自动创建 .venv 并同步默认 + dev 组）
uv sync

# 2. 按需启用可选组（RAG 向量库 / 可观测性）
uv sync --extra rag --extra observability

# 3. 校验（骨架阶段暂无业务代码，pytest 需在 tests/ 有用例后运行）
uv run ruff check src/cg
uv run pytest          # 当前骨架阶段：无测试用例
```

## 文档

- [docs/00-system-design.md](docs/00-system-design.md) —— 系统设计总览（业务架构 / §15 包结构 / §16 面试答辩备忘）

## 目录速览

```
src/cg/
├── common/       通用：雪花 ID、JSON 工具、错误码
├── domain/       领域模型：Case / AgentState / Evidence / Decision（Pydantic）
├── screening/    传统机审初筛 + 三分流（rule_engine / triage）
├── agent/        Agent 核心：LangGraph StateGraph（state / graph / nodes / tools_node / guardrails / checkpointer）
├── tools/        6 个 Tool：product / image_analysis / ocr / merchant / case_search / policy_search
├── rag/          政策库 + 案例库：分块、embedding、混合检索、rerank
├── evaluation/   评测 harness + 三方案对比 + Hard Case Benchmark
├── api/          FastAPI 路由 + 审核工作台接口
└── infra/        MySQL / Redis / MQ / OTel / Langfuse / 向量库 集成
```
