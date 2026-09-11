# product-review-agent

[![tests](https://github.com/donglongjun886/product-review-agent/actions/workflows/test.yml/badge.svg)](https://github.com/donglongjun886/product-review-agent/actions/workflows/test.yml)
[![python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

电商商品内容治理的**复杂风险调查 Agent**：用 LangGraph 编排多步取证，对机审难以直判的低置信案件输出
`PASS` / `REJECT` / `HUMAN_REVIEW` 三分类裁决。

## 这是什么

传统机审（黑名单、关键词、Logo/类目规则、OCR、分类模型）快、便宜、可解释，适合处理**确定性异常**。
真正困难的是开放性、上下文依赖强、需要多源交叉验证的案件。本项目把 Agent 定位为机审链路里的
**复杂案件处理节点**——只接管初筛后无法直判的低置信案件，动态取证后给出可回溯、可重放的裁决：

```text
商品上架 / 变更事件
  ▼  规则 + 模型初筛（同步、全量）
  ▼  三分流：明确正常 → PASS 直判 · 明确违规 → REJECT 直判 · 复杂低置信 → 进入 Agent
  ▼  Complex Risk Investigation Agent（预算受限的调查子图）
  ▼  PASS / REJECT / HUMAN_REVIEW → 落库 → 人工裁决回流案例库 / 政策库 / 评测集
```

调查范式是**假设验证**而非直接分类：`hypothesize` 先给出风险假设先验，`reevaluate` 依据取证结果把假设
更新为后验（`SUPPORTED` / `REFUTED` / `UNRESOLVED`），假设轨迹与证据链全程保留、可重放。
证据不足时 Agent 主动克制地转人工，而不是强行下结论。

## 核心特性

- **假设驱动的调查子图**：5 节点 7 边单回环（`hypothesize → plan → tools → reevaluate → decide`），
  未收敛则在预算 / 收敛 / 去重护栏约束下回到 `plan`。
- **LLM 只提案，确定性 Gate 收口**：LLM 产出 `DecisionProposal`，最终裁决由硬规则、abstention 清单与
  PASS/REJECT Gate 决定；每次改判的原因码写入 `ReviewDecision.overrides`，可审计。
- **6 个可插拔调查工具**：商品事实、商家历史、图像分析、OCR、案例检索、政策检索，统一 `Tool` 抽象。
- **确定性可重放**：默认使用 scripted LLM 桩 + InMemory 数据源 + InMemory Checkpointer，
  无 API key、无网络即可跑通全链路与评测。
- **检索增强（RAG）**：政策库与案例库支持 `local`（numpy + MockHash，确定性）与 `chroma`
  （ChromaDB + LlamaIndex + BGE + BM25/jieba + RRF 混合检索）两种后端。
- **评测体系**：三方案对比（Rule / Single-call LLM / Agent）、业务与 abstention 指标、消融、阈值扫描、
  决策序列回归。
- **可选可观测性**：Langfuse 适配层上报 root / node / generation / tool / gate span 树，
  `trace_id` 与落库 `run_id` 同值。

## 快速开始

环境要求：Python 3.12+、[uv](https://docs.astral.sh/uv/)。

```bash
uv sync                                    # 默认依赖 + dev 组（pytest / ruff / httpx）
uv sync --extra rag --extra observability  # 可选：真实 RAG / Langfuse；两个 --extra 需同时写（只写一个会移除另一个）
```

### 端到端走查（无需 API key）

```bash
uv run python scripts/demo_walkthrough.py
```

默认装配 scripted 桩 + InMemory 工具 + InMemory Checkpointer，跑通完整调查子图并执行内置断言。
确定性输出：`decision=HUMAN_REVIEW` / `risk_level=HIGH` /
`risk_type=[POTENTIAL_IP_RISK, EVASION_PATTERN]` / `decision_confidence=0.87` / `overrides=[]`。

### 运行测试

```bash
uv run pytest tests/ -q
```

全部用例均为确定性 mock，**不联网、不需要 API key**。依赖外部条件的用例在不可用时自动 skip：
真库冒烟需本机 MySQL 可达，BGE 真模型需模型缓存，RAG 真链路 e2e 需 `--extra rag` + Chroma 服务端
+ 模型缓存。

### 启动服务并受理一次审核

```bash
uv run uvicorn pra.api.app:app --reload            # http://127.0.0.1:8000
curl http://127.0.0.1:8000/api/v1/health           # {"status":"ok"}

uv run python -c "from scripts.demo_api import build_demo_case; import json; print(json.dumps(build_demo_case().model_dump(mode='json'), ensure_ascii=False))" > case.json
curl -X POST http://127.0.0.1:8000/api/v1/reviews -H 'Content-Type: application/json' -d @case.json
# → {"run_id": "<32-hex>", "review_decision": {"decision": "...", "risk_level": ..., "risk_type": [...], ...}}
```

请求体为 `ProductReviewCase`，关键字段 `case_id` / `product`（`product_id`、`title`、`category`、
`images` …）/ `merchant_id` / `event_type` / `screening_signals`，完整契约见 `src/pra/domain/models.py`。
服务端**受理即分流**：`COMPLEX` 进入调查子图并落库，`PASS` / `REJECT` 由规则直判落库（无 trace）。
不经 HTTP 的执行器级演示见 `scripts/demo_api.py`。

### 数据库（可选）

```bash
# 仅绑回环，勿用 -p 3306:3306 暴露到局域网
docker run --name mysql-dev -e MYSQL_ROOT_PASSWORD=<your-password> -p 127.0.0.1:3306:3306 -d mysql:8
docker exec -i mysql-dev mysql -uroot -p<your-password> -e "CREATE DATABASE IF NOT EXISTS product_review CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
docker exec -i mysql-dev mysql -uroot -p<your-password> product_review < migrations/001_review_core_tables.sql
cp .env.example .env   # 填 DATABASE_URL
```

DDL 位于 `migrations/`：`001_review_core_tables.sql`（审核核心 5 表）、
`002_review_case_triage.sql`（分流增量列）、`003_product_tables.sql`（商品 3 表）、
`004_merchant_tables.sql`（商家 2 表），全部幂等。未配置 `.env` 时代码使用内置本地开发默认 DSN。

## 配置

配置经 pydantic-settings 从仓库根 `.env` 读取，模板见 `.env.example`：

| 变量 | 说明 |
|---|---|
| `DATABASE_URL` | MySQL 连接串（aiomysql 异步驱动） |
| `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` | 仅真实 LLM 评测脚本使用，可选 |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` | Langfuse 凭据，缺省走 `NullTracer` |
| `PRA_LANGFUSE_ENABLED` | 总开关；`0/false/no/off` 关闭，未设置 = 有凭据即启用 |
| `PRA_LANGFUSE_EXPERIMENT` / `PRA_LANGFUSE_SAMPLE` / `PRA_LANGFUSE_SESSION` | trace 版本名 / 采样率 / 会话分组 |

`Settings` 保持 `extra="forbid"`：出现未知键名直接报错，避免拼写错误静默连错库。

## 评测

评测方法与口径（三方案定义、abstention 指标、消融、阈值扫描、结论边界）以
[docs/02-evaluation.md](docs/02-evaluation.md) 为权威出处。

```bash
uv run python scripts/run_evaluation.py      # 三方案对比（默认 v1 数据集）
uv run python scripts/run_regression.py      # 决策序列 sha256 与基线快照比对
uv run python scripts/run_ablation.py        # 方案级 / 组件级消融
uv run python scripts/run_sweep.py           # 阈值扫描
uv run python scripts/run_evaluation_real.py --limit 10   # 真实 LLM 对照（需 API key，有费用）
```

## 检索（Policy KB / Case KB）

默认后端为 `local`（numpy 内存索引 + `MockHashEmbedder`，确定性、无外部依赖）。安装 `--extra rag`
后可切换 `rag_backend="chroma"`：ChromaDB + LlamaIndex + BGE + BM25(jieba) + RRF，检索口径见
[docs/00-system-design.md](docs/00-system-design.md) 与 `src/pra/rag/chroma_backend.py` 模块注释。

```bash
uv run python scripts/run_rag_demo.py                            # 三模式（bm25/vector/hybrid）Top-K 检索演示
uv run python scripts/run_rag_eval.py --backend chroma           # RAG 评测：local / chroma / qdrant
uv run python scripts/run_rag_eval.py --backend chroma --probe   # 人工标注 probe 的 Recall@K（缺省关）
```

`--backend chroma` 默认使用进程内 `EphemeralClient`（每臂独立、无需本机服务端），连接服务端请加
`--chroma-client http`，部署见 [deploy/chroma/README.md](deploy/chroma/README.md)。Qdrant 路径代码与
`deploy/qdrant` 保留可用，但非默认。

## 可观测性（Langfuse，可选）

`pra.observability` 是薄适配层：**无凭据时使用 `NullTracer`，全链路 no-op、零网络、不影响测试**；
配置凭据后上报 root / node / generation / tool / gate span 树，`trace_id` 与 MySQL `review_run.run_id`
同值，审计链可互跳。

```bash
cd deploy/langfuse && docker compose up -d                             # 本地自托管，UI http://localhost:3000
uv run python scripts/demo_langfuse_trace.py                           # 打印 trace UI 链接
uv run python scripts/langfuse_smoke.py                                # 端到端自检（无凭据 exit 0）
```

部署与实测记录见 [deploy/langfuse/README.md](deploy/langfuse/README.md)。

## 项目结构

```text
src/pra/
  agent/           LangGraph 调查子图：state / graph / nodes / tools_node / guardrails / checkpointer / llm 后端
  api/             FastAPI 接入层（POST /api/v1/reviews、GET /api/v1/health）
  domain/          Pydantic 契约（ProductReviewCase / ReviewDecision / Evidence …）
  tools/           6 个调查工具：product / merchant / image_analysis / ocr / case_search / policy_search
  screening/       机审引擎与三分流
  rag/             知识库检索：local / chroma / qdrant、BM25、embedder、惰性索引
  evaluation/      评测 harness、数据集、指标、消融、扫描、回归
  infra/           MySQL 五表落库（SQLAlchemy 2.0 async）
  observability/   Tracer 适配层（Langfuse / Null）
  common/          通用工具
docs/              系统设计与评测口径 · migrations/ DDL · scripts/ 演示与评测脚本 · tests/ pytest 用例
```

## 关键设计

- **裁决收口**：R1 硬规则强制 `REJECT`（LLM 不可覆盖）→ abstention 清单 → PASS/REJECT Gate；
  改判原因码全量写入 `overrides`。
- **`decision_confidence` 是确定性安全门槛，不是模型概率**：按证据与假设以固定公式重算；
  `0.7` 仅作为 REJECT 的安全门槛（不达标转 `HUMAN_REVIEW`），PASS 另有独立 Gate 校验。
- **预算是护栏上界而非目标**：默认 10 次 LLM 调用 / 15 次工具调用 / 40k tokens / 30s；
  超限不是失败，而是带部分证据转人工止损。
- **`DECIDED` 是唯一终态**：`decide` 之后无出边，超限与降级只记入 overrides 与预算快照。
- **证据不可篡改**：去重指纹为 `(type, source, ref_id)`（`ref_id` 优先、缺失回退 `value`），
  收集后不可修改。字段语义见 `src/pra/agent/state.py` 与 `src/pra/agent/guardrails/schemas.py`。

## 技术栈

| 关注点 | 选型 |
|---|---|
| 语言 / 包管理 | Python 3.12+ · uv（`rag` / `observability` / `dev` 依赖组） |
| 接入层 | FastAPI + uvicorn |
| 调查编排 | LangGraph StateGraph（5 节点 7 边单回环）+ InMemory Checkpointer |
| LLM | `LLMBackend` 抽象：默认确定性 scripted 桩；`LiteLLMBackend` 经 `set_llm_backend` / `build_agent_graph(llm=)` 注入 |
| 检索 | local（numpy + MockHash）/ chroma（ChromaDB + LlamaIndex + BGE + BM25 + RRF）/ qdrant |
| 数据层 | SQLAlchemy 2.0 async · aiomysql · MySQL |
| 可观测性 | Langfuse v4（自托管）· Null Object 兜底 |
| 质量 | pytest（含超时守护）· ruff · GitHub Actions |

## 当前实现边界

- **默认全链路为 scripted LLM 桩 + InMemory 数据源**，保证确定性可重放；该路径下 token = 0、
  latency ≈ 0、cost 为空是真实情况，不做填充。
- **生产 / HTTP 入口 `build_production_tools()` 使用真实数据源**：商品、商家接 MySQL，
  案例、政策接真实 RAG（惰性构建，首次检索才建库连服务端；失败记 warn failure，不静默回退种子）。
  默认 `build_tools()` 与评测世界仍是 InMemory；评测世界为 5 个工具，比生产少一个 `OCRTool`。
- `image_analysis` 与 `ocr` 尚未接入真实视觉模型与 OCR 服务，为 Mock 桩。
- 真实 LLM 评测仅通过 `scripts/run_evaluation_real.py` 触发，需 API key、有费用、非确定性；
  已发布的对照结果基于 v1 数据集单次抽样。
- 评测集由单一标注者按与审查员同源的规则构造，未做多标注者交叉校验，因此高分只反映实现一致性。
- 尚未实现：MQ 异步 worker、Redis 幂等与限流、审核工作台、OpenTelemetry 跨服务链路。

## 文档

- [docs/00-system-design.md](docs/00-system-design.md) —— 系统设计总览：业务价值 → 决策难点 → 设计 → 验证
- [docs/02-evaluation.md](docs/02-evaluation.md) —— 评测方案与指标口径

## Roadmap

- **RAG 语料与模型扩展**：扩大语料与模型对比范围、补充检索指令，并配套独立评测口径。
- **异步化接入**：MQ 异步 worker + 人工审核队列，配套 MySQL Checkpointer 续跑与 Redis 幂等。
- **可观测性**：在 Langfuse 之上补充 OpenTelemetry 跨服务链路 trace、采样与容量治理。
- **策略库化**：把品牌黑名单等规则词表沉淀为可维护的策略库，支撑归因观测与词表调优。

## 许可证

[MIT](LICENSE)
