# product-review-agent

[![tests](https://github.com/donglongjun886/product-review-agent/actions/workflows/test.yml/badge.svg)](https://github.com/donglongjun886/product-review-agent/actions/workflows/test.yml)

电商平台商品内容治理 · **复杂风险调查 Agent**（Complex Risk Investigation Agent）

> **为什么需要 Agent**：传统机审（黑名单 / 关键词 / Logo / 类目 / OCR / 分类模型）负责**确定性异常**——快、便宜、可解释；难的是**开放性、上下文依赖强、需要多源交叉验证**的案件。本项目让 Agent 只接管这类**复杂低置信案件**，动态取证后输出三分类裁决：`PASS` 放行 / `REJECT` 违规拒绝（须有可引用政策依据）/ `HUMAN_REVIEW` 转人工（证据或置信不足、预算耗尽、政策模糊、关键工具失败时主动克制）。
> **生产主线**：上架/变更 → 规则 + 模型初筛（同步、便宜）→ 三分流（明确正常 / 明确违规 / **复杂低置信**）→ Agent 调查（LangGraph 编排）→ 三分类裁决 → 人工裁决回流案例库 / 政策库 / 评测集。

## 系统总链路

```text
HTTP 接入（同步；MQ 异步 worker 为规划）—— POST /api/v1/reviews，请求体 = ProductReviewCase
  ▼  build_initial_state(case) → AgentState（hypotheses / evidence / budget / …）
  ▼  LangGraph 调查子图（5 节点 7 边单回环）
     hypothesize → plan → tools（6 个调查工具）→ reevaluate
     └─ 未收敛 → 回 plan（受预算 / 收敛 / 去重护栏约束）  ·  收敛或护栏触发 → decide
  ▼  Decision Gate：LLM 只产 DecisionProposal，确定性 overlay 收口
     （R1 硬规则强制 REJECT → abstention 清单 → PASS/REJECT Gate）
  ▼  PASS / REJECT / HUMAN_REVIEW（DECIDED 唯一终态）→ 五表落库（MySQL）
```

调查范式是**假设验证**而非直接分类：hypothesize 给出风险假设先验，reevaluate 依据取证结果更新为后验（SUPPORTED / REFUTED / UNRESOLVED），证据链与假设轨迹全程可回溯、可重放。

## 口径红线（读结论前必读）

- **默认全链路 = scripted LLM 桩 + InMemory 世界 ⇒ 确定性可重放**（无真实 LLM、无网络）；scripted 路径下 **token=0 / latency≈0 / cost 为空**是真实情况，不填假值。
- **默认 `build_tools()` 与评测世界 = InMemory 世界**（确定性可重放；**评测世界是 5 个工具，比生产少一个 `OCRTool`**，口径见 [docs/02-evaluation.md](docs/02-evaluation.md) §7.2）；**生产 / HTTP 入口 `build_production_tools()` 读真库 + 真实 RAG**：商品/商家接 MySQL，案例/政策接真实 RAG（`rag_backend="chroma"` + BGE + hybrid，经 `Lazy*Index` **惰性构建**——首次检索才建库连 Chroma，不可用时记 warn failure、不静默回退种子）。
- **6 个 Tool 的默认实现仍是 InMemory/Mock 桩**：`image_analysis` / `ocr` 未接真实视觉模型与 OCR 服务。
- **README 不承载跑分数字**（会随数据集与模型迭代过期）：评测方法与口径以 [docs/02-evaluation.md](docs/02-evaluation.md) 为权威出处；Agent 高分须按「GT ≈ 审查员可判定函数、同口径耦合 + 种子数据」的边界解读，**不得**外推成「真实 LLM 能力」。
- **真实 LLM 评测**仅 `scripts/run_evaluation_real.py`（需 API key、有费用、非确定性）。

## 快速开始

环境：Python 3.12+ · [uv](https://docs.astral.sh/uv/)。

```bash
uv sync                                    # 默认依赖 + dev 组（pytest / ruff / httpx）
uv sync --extra rag --extra observability  # 可选：RAG 真实化 / Langfuse（见「技术栈」）
```

### 端到端走查（无 API key）

```bash
uv run python scripts/demo_walkthrough.py
```

默认 scripted 桩 + 6 个 InMemory 工具 + InMemory checkpointer，跑通调查子图并执行内置断言。预期输出（确定性可复现）：`decision=HUMAN_REVIEW` / `risk_level=HIGH` / `risk_type=[POTENTIAL_IP_RISK, EVASION_PATTERN]` / `decision_confidence=0.87` / `overrides=[]`。

### 测试

```bash
uv run pytest tests/ -q     # 全确定性 mock：无网络 / 无 API key
```

外部条件相关的用例**不可用时自动 skip**：真库冒烟需本机 MySQL 可达；BGE 真模型需模型缓存（`PRA_RAG2_MODEL_CACHE`，缺省仓库内 `.cache/model_cache`）；RAG 真链路 e2e 需 `--extra rag` + Chroma 服务端 + 模型缓存 —— 缺一即跳过，**绝不联网下载**。

### 起服务 + 受理一次审核

```bash
uv run uvicorn pra.api.app:app --reload      # 默认 http://127.0.0.1:8000
curl http://127.0.0.1:8000/api/v1/health     # → {"status":"ok"}

uv run python -c "from scripts.demo_api import build_demo_case; import json; print(json.dumps(build_demo_case().model_dump(mode='json'), ensure_ascii=False))" > case.json
curl -X POST http://127.0.0.1:8000/api/v1/reviews -H 'Content-Type: application/json' -d @case.json
# → {"run_id": "<32-hex>", "review_decision": {"decision": "...", "risk_level": ..., "risk_type": [...], ...}}
```

请求体 = `ProductReviewCase`，关键字段 `case_id` / `product`（`product_id` `title` `category` `images` …）/ `merchant_id` / `event_type` / `screening_signals`；完整契约见 `src/pra/domain/models.py`，执行器级演示（不经 HTTP）见 `scripts/demo_api.py`。服务端**受理即 Screening 三分流**：`COMPLEX` 走调查图并落库，`PASS` / `REJECT` 由规则直判落库（无 trace）；**落库需本机 MySQL 可达且已建表**；完整的政策 / 先例证据还需 `--extra rag` + Chroma 服务端（缺则相应工具调用记 warn failure，其余链路正常完成）。

### 落库（可选）

五表 DDL 见 `migrations/`：`001_review_core_tables.sql`（核心 5 表）+ `002_review_case_triage.sql`（增量列，按需执行）。

```bash
# 仅绑回环（勿用 -p 3306:3306 暴露到局域网）；口令务必改掉默认值并只写进 .env
docker run --name mysql-dev -e MYSQL_ROOT_PASSWORD=<your-password> -p 127.0.0.1:3306:3306 -d mysql:8
docker exec -i mysql-dev mysql -uroot -p<your-password> -e "CREATE DATABASE IF NOT EXISTS product_review CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;" && docker exec -i mysql-dev mysql -uroot -p<your-password> product_review < migrations/001_review_core_tables.sql
cp .env.example .env   # 填 DATABASE_URL=mysql+aiomysql://<user>:<pass>@127.0.0.1:3306/product_review
```

未配 `.env` 时代码内置**本地开发默认 DSN**（仅适用本机容器，见 `pra.infra.db.Settings`）；其他环境必须经 `.env` 覆盖，且 `Settings` 保持 `extra="forbid"`（未知键直接报错）。

## 评测

评测方法与口径（三方案定义 / abstention 指标 / Ablation / Sweep / 里程碑与结论边界 / **数据集局限**）以 [docs/02-evaluation.md](docs/02-evaluation.md) 为权威出处；README 不承载跑分数字。真实 LLM 对照（需 API key、有费用、非确定性）：`scripts/run_evaluation_real.py --limit 10`；凭据经 `--api-key` / `--base-url` 或 `.env` 的 `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` 注入。

**评测覆盖边界**：真 LLM **只跑过 v1 35 案单次抽样**（仅证明「真实链路已跑通」，不是模型水平；未跑 v2、未重复采样）；评测集由**单一标注者**按与审查员同源的规则构造（无第二标注者交叉校验），故 scripted 高分只衡量实现一致性。

```bash
uv run python scripts/run_evaluation.py                      # 三方案对比（默认 v1；--data 可切 v2）
uv run python scripts/run_regression.py                      # 决策序列 sha256 与基线快照比对
uv run python scripts/run_ablation.py                        # 方案级 / 组件级消融（run_sweep.py 阈值扫描）
```

## RAG 检索（Policy KB / Case KB）

默认后端是 **`local`**（numpy 内存索引 + `MockHashEmbedder`，确定性、无外部依赖）。装 `--extra rag` 后可切 **`rag_backend="chroma"`**：ChromaDB + LlamaIndex + BGE + BM25(jieba) + RRF，检索口径见 [docs/00-system-design.md](docs/00-system-design.md) 与 `src/pra/rag/chroma_backend.py` 的模块注释。**生产 / HTTP 入口已切真实 RAG**（`build_production_tools()` 装配 chroma + BGE + hybrid，经 `Lazy*Index` 惰性构建：首次检索才建库；缺 `--extra rag` / 服务端不可达 / 模型未缓存时该次工具调用记 warn failure，**不静默回退 InMemory 种子**）。Qdrant 的代码与 `deploy/qdrant` **迁移期保留**（本机容器已卸，`rag_backend="qdrant"` 仍可用）。

```bash
uv run python scripts/run_rag_demo.py                          # 三模式（bm25/vector/hybrid）Top-K 检索演示
uv run python scripts/run_rag_eval.py --backend chroma         # RAG 评测入口：local / chroma / qdrant
uv run python scripts/run_rag_eval.py --backend chroma --probe # 人工标注 probe 的 Recall@K（缺省关）
```

`--backend chroma` 缺省用**进程内 `EphemeralClient`**（每臂独立、**无需本机服务端**），要跑服务端路径加 `--chroma-client http`（部署见 [deploy/chroma/README.md](deploy/chroma/README.md)）。`scripts/run_rag_phase2_demo.py` 是**迁移期保留的遗留演示**（BGE + Qdrant），当前 RAG 评测入口是 `scripts/run_rag_eval.py`。

## 可观测性（Langfuse，可选）

`pra.observability` 是薄适配层：**无凭据 → `NullTracer` 全链路 no-op**（零网络、不影响测试）；配好凭据后上报 **root / node / generation / tool / gate** span 树，`trace_id` 与 MySQL `review_run.run_id` 同值（审计链可互跳），`PRA_LANGFUSE_ENABLED=0` 可强制关闭。

```bash
cd deploy/langfuse && docker compose up -d      # 本地自托管，UI http://localhost:3000
PRA_LANGFUSE_ENABLED=1 uv run python scripts/demo_langfuse_trace.py   # 打印 trace UI 链接
```

部署与实测记录见 [deploy/langfuse/README.md](deploy/langfuse/README.md)，职责边界与口径见 [docs/00-system-design.md](docs/00-system-design.md)。

## 目录结构

```text
src/pra/  agent/（LangGraph 子图：state / graph / nodes / tools_node / guardrails / checkpointer / scripted_llm）
          api/（FastAPI 接入）· infra/（MySQL 五表落库）· domain/（Pydantic 契约）
          tools/（6 个调查工具）· screening/（机审 + 三分流）· evaluation/（评测 harness）
          rag/（KB 检索：local / chroma / qdrant）· observability/（Tracer 适配）· common/（通用工具）
docs/ 设计文档 · migrations/ DDL · scripts/ 演示与评测脚本 · tests/ pytest 用例
（逐层职责与设计动机见 docs/00-system-design.md §15）
```

## 关键设计决策（要点）

- **LLM 只提案、确定性 Gate 收口**：R1 硬规则强制 REJECT（不可被 LLM 覆盖）→ abstention 清单 → PASS/REJECT Gate；改判原因码全量写入 `ReviewDecision.overrides`，可审计。
- **`decision_confidence` 是确定性安全门槛、不是模型概率**：按证据/假设固定公式重算；`0.7` 仅作 REJECT 的安全门槛（不达标转 `HUMAN_REVIEW`），PASS 另有独立 Gate 校验。
- **预算是 Guardrail 上界而非目标**：默认 10 LLM / 15 Tool / 40k tokens / 30s；超限不是失败，而是「带部分证据转人工止损」。
- **`DECIDED` 是唯一终态**：decide 之后无出边，超限 / 降级只记入 overrides 与预算快照。
- **证据去重与派生回填全确定性**：去重指纹 `(type, source, ref_id)`（ref_id 优先、缺失回退 value），证据收集后不可篡改。参数语义明细见 `src/pra/agent/guardrails/schemas.py` 与 `src/pra/agent/state.py`。

## 技术栈

| 关注点 | 选型 / 现状 |
|---|---|
| 语言 / 包管理 | Python 3.12+ · uv（`rag` / `observability` 可选组 + `dev` 组） |
| 接入层 | FastAPI：`POST /api/v1/reviews` 同步执行 + 落库；`GET /api/v1/health` |
| 调查编排 | LangGraph StateGraph：5 节点 7 边单回环 + InMemory Checkpointer |
| LLM | `LLMBackend` 抽象：默认确定性 scripted 桩（无 key 可跑）；`LiteLLMBackend` 真后端经 `set_llm_backend` / `build_agent_graph(llm=)` 注入 |
| 评测 | `pra.evaluation`：三方案 harness + business/abstention 指标 + ablation + sweep + regression |
| RAG | 默认 `local`（numpy + MockHash，确定性）；`--extra rag` 后可 `rag_backend="chroma"`（ChromaDB + LlamaIndex + BGE + BM25(jieba) + RRF，口径见 [docs/00](docs/00-system-design.md)）；Qdrant 迁移期保留 |
| 数据层 | SQLAlchemy 2.0 async · aiomysql · MySQL 五表（DDL 见 `migrations/`） |
| 可观测性 | `pra.observability` 适配层（Null Object 兜底）+ Langfuse v4 本地自托管（`--extra observability`） |

## 文档索引

- [docs/00-system-design.md](docs/00-system-design.md) —— 系统设计总览（业务价值 → 决策难点 → 设计 → 验证；RAG / 可观测性口径；§15 目录结构）
- [docs/02-evaluation.md](docs/02-evaluation.md) —— 评测方案：三方案定义 / 指标口径（含 abstention）/ Ablation / Sweep / 结论边界

## Roadmap（方向与动机）

- **RAG 语料与模型的评测口径扩展**：当前实现与结论边界见 [docs/00-system-design.md](docs/00-system-design.md) 与 [docs/02-evaluation.md](docs/02-evaluation.md)；动机是扩大语料与模型对比、补检索指令，需要独立评测口径，不做能力外推。
- **MQ 异步 worker + 人工审核队列**：HTTP 同步受理受吞吐 / 并发限制；异步化（含 MySQL Checkpointer、Redis 幂等）支撑接入解耦、削峰与事件溯源。
- **可观测性下一步**：OpenTelemetry 跨服务链路 trace 与采样 / 容量治理（Langfuse 已覆盖 LLM 调用级）。
- **Screening 策略库化**：品牌黑名单等规则词表沉淀为可维护的策略库，支撑归因观测与词表调优。

## 许可

[MIT](LICENSE)
