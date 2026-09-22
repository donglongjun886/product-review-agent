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
更新为后验，假设轨迹与证据链全程保留、可重放。先验 / 后验只用于**引导调查与留痕，不参与最终裁决**——
`PASS` / `REJECT` 由 Gate 依据证据事实确定性算出。证据不足时 Agent 主动克制地转人工，而不是强行下结论。

## 核心特性

- **假设驱动的多步调查**：`hypothesize → plan → tools → reevaluate → decide`，未收敛则在预算 / 收敛 /
  去重护栏约束下回到 `plan`。
- **LLM 只提案，确定性 Gate 收口**：LLM 产出决策提案，最终裁决由硬规则、abstention 清单与 PASS/REJECT
  Gate 依据**证据事实**决定；每次改判的原因码写入 `overrides`，可审计。
- **多源取证工具**：商品事实、商家历史、图像分析、OCR、案例检索、政策检索，统一工具抽象。
- **两条路径都开箱可跑**：默认（测试 / 评测 / 离线走查）为确定性桩 + InMemory 数据源，无凭据、无网络
  即可跑通全流程；生产入口走真实网关与真实数据源，**缺凭据显式失败**，不静默回落。
- **评测体系**：三方案对比（Rule / Single-call LLM / Agent）、业务与 abstention 指标、方案级消融、
  决策序列回归。
- **可选检索与观测**：知识库支持向量 + 关键词混合检索，观测层支持 trace 树；两者均为可选依赖。

## 快速开始

环境要求：Python 3.12+、[uv](https://docs.astral.sh/uv/)。

```bash
uv sync                                    # 默认依赖 + dev 组（pytest / ruff / httpx）
uv sync --extra rag --extra observability  # 可选：真实检索 / Langfuse（两个 --extra 需同时写）
uv run pytest tests/ -q                    # 确定性用例，不联网、不需要凭据
uv run python scripts/demo_walkthrough.py  # 离线端到端走查：跑通完整调查子图并执行内置断言
```

## 用法

### HTTP 服务

```bash
cp .env.example .env                        # 填 DATABASE_URL 与生产入口所需凭据
uv run uvicorn pra.api.app:app --reload     # http://127.0.0.1:8000
curl http://127.0.0.1:8000/api/v1/health    # {"status":"ok"}

uv run python -c "from scripts.demo_api import build_demo_case; import json; print(json.dumps(build_demo_case().model_dump(mode='json'), ensure_ascii=False))" > case.json
curl -X POST http://127.0.0.1:8000/api/v1/reviews -H 'Content-Type: application/json' -d @case.json
```

请求体为 `ProductReviewCase`（商品快照 + 商家 + 事件类型 + 机审信号），契约见
[`src/pra/domain/models.py`](src/pra/domain/models.py)。服务端**受理即分流**：明确正常 / 明确违规由规则
直判落库，复杂低置信进入调查子图后落库。不经 HTTP 的执行器级演示见
[`scripts/demo_api.py`](scripts/demo_api.py)（同走生产入口）。

### 数据库（可选）

MySQL 承担落库与审计链；离线走查与评测不需要它，HTTP 入口需要。

```bash
docker run --name mysql-dev -e MYSQL_ROOT_PASSWORD=<your-password> -p 127.0.0.1:3306:3306 -d mysql:8
docker exec -i mysql-dev mysql -uroot -p<your-password> -e "CREATE DATABASE IF NOT EXISTS product_review CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
for f in migrations/*.sql; do docker exec -i mysql-dev mysql -uroot -p<your-password> product_review < "$f"; done
```

DDL 位于 [`migrations/`](migrations/)，全部幂等；Chroma 与 Langfuse 的本地部署见 [`deploy/`](deploy/)。

## 架构

- **调查子图**：`hypothesize → plan → tools → reevaluate → decide` 单回环，`decide` 是唯一终态出口。
- **裁决收口**：硬规则强制 `REJECT`（LLM 不可覆盖）→ abstention 清单 → PASS/REJECT Gate；改判原因码
  全量写入 `overrides`。
- **置信度是确定性门槛，不是模型概率**：按证据事实以固定公式重算，系数结构性给定、不用数据集拟合；
  不达标转人工。模型产出的假设与概率类中间结果**不参与最终裁决**。
- **预算是护栏上界而非目标**：超限不是失败，而是带部分证据转人工止损。
- **证据不可篡改**：去重后不可修改；证据不足转人工，禁止用推测、默认值或空证据补齐。
- **测量边界显式声明**：外观维度在缺少真实视觉服务时声明为不可测，不因缺席而放行。

判定语义、Gate 规则与数据模型口径见 [docs/00-system-design.md](docs/00-system-design.md)。

## 评测

评测方法与口径（三方案定义、指标分子 / 分母、结论边界）以
[docs/02-evaluation.md](docs/02-evaluation.md) 为权威出处：**§11** = 最近一次 scripted 封板结果，
**§12** = 最近一次 real 全量（单次运行）。

```bash
uv run python scripts/run_evaluation.py                 # 三方案对比（默认 v1 数据集）
uv run python scripts/run_regression.py                 # 决策序列 sha256 与基线快照比对
uv run python scripts/run_ablation.py                   # 方案级消融
uv run python scripts/run_evaluation_real.py --limit 10 # 真实 LLM 对照（需凭据、有费用）
```

两条世界口径必须同框引用：**scripted 数字衡量实现一致性，不外推模型能力**；**real 数字为单次运行、
非确定性、不可重放**。评测世界的数据源与工具集也与生产不同，详见 docs/02 §7。

## 配置

配置经 pydantic-settings 从仓库根 `.env` 读取，**全部键与说明以 [`.env.example`](.env.example) 为准**。
两条约束：未知键名直接报错，不静默忽略；生产入口缺 LLM 凭据时装配期显式失败。

可选依赖组 `rag`（真实检索）与 `observability`（Langfuse）只增强能力，不装也能跑默认路径。

## 开发

- **测试**：`uv run pytest tests/ -q`（确定性、不联网）。依赖外部条件的用例在不可用时自动 skip：
  真库冒烟需本机 MySQL，真实向量检索需 `rag` extra + Chroma 服务端 + 模型缓存。
- **静态检查**：`uv run ruff check` —— 仓库未配置 ruff 规则集，改动后对照是否新增告警。
- **检索**：向量 + 关键词混合（需 `rag` extra）。两路分数不可比、不参与裁决，分数口径见
  [docs/00-system-design.md](docs/00-system-design.md) §6。
- **常用脚本**：`scripts/demo_walkthrough.py`（离线走查）、`scripts/run_rag_demo.py`（检索三模式）、
  `scripts/demo_langfuse_trace.py`（观测 trace）。

## 项目结构

```text
src/pra/
  agent/          调查子图：state / graph / nodes / tools_node / guardrails / checkpointer / LLM 后端
  api/            FastAPI 接入层
  domain/         Pydantic 契约（案件、裁决、证据）
  tools/          调查工具（商品 / 商家 / 图像 / OCR / 案例检索 / 政策检索）
  screening/      机审引擎与三分流
  rag/            知识库检索（向量 + 关键词混合，惰性构建）
  evaluation/     评测 harness、数据集、指标、消融、回归
  infra/          落库与持久化（SQLAlchemy 2.0 async）
  observability/  Tracer 适配层（Langfuse / Null）
docs/ · migrations/ · scripts/ · tests/ · deploy/
```

## 技术栈

| 关注点 | 选型 |
|---|---|
| 语言 / 包管理 | Python 3.12+ · uv |
| 服务与编排 | FastAPI + uvicorn · LangGraph |
| LLM | 后端抽象：生产走真实网关，测试 / 评测走确定性桩 |
| 检索 | ChromaDB + LlamaIndex + BGE + BM25（可选依赖） |
| 数据与观测 | MySQL（SQLAlchemy async）· Langfuse（可选） |
| 质量 | pytest（含超时守护）· ruff · GitHub Actions |

## Roadmap

- **真实视觉 / OCR 接入**：图像分析与 OCR 目前为桩，生产据此声明外观维度不可测。
- **RAG 语料与模型扩展**：扩大语料与模型对比范围，并配套独立评测口径。
- **异步化接入**：MQ worker + 人工审核队列，配套续跑与幂等。
- **审核工作台与跨服务链路观测**：在现有 trace 之上补跨服务链路与容量治理。

## 文档

- [docs/00-system-design.md](docs/00-system-design.md) —— 设计口径与决策依据：业务约束、判定机制、边界与「不做什么」
- [docs/02-evaluation.md](docs/02-evaluation.md) —— 评测口径与最近一次结果：三方案定义、指标分子 / 分母、最近一次 scripted / real 运行

> 两份文档只写**口径与决策**；字段、表结构、目录、配置值与当前实现状态以代码与本 README 为准。
> 评测结果只保留**最近一次**运行，重跑即覆盖（历次数字见版本库）。

## 许可证

[MIT](LICENSE)
