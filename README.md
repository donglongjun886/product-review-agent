# product-review-agent

[![tests](https://github.com/donglongjun886/product-review-agent/actions/workflows/test.yml/badge.svg)](https://github.com/donglongjun886/product-review-agent/actions/workflows/test.yml)
[![python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

机器审核能挡掉大部分明显违规，但总有一批案子说不准：标题干净、图片可疑、商家还有前科。
这个项目处理的就是这批案子 —— 它自己决定去查什么，查完给出三种结论：**通过、拒绝，或者交给人**。

## 它怎么工作

先看平台已有的规则和机审结果：明显正常的直接放行，明显违规的直接拒绝，**剩下拿不准的才进入调查**。

调查是一个循环：**提风险假设 → 定查证计划 → 调工具取证 → 复核假设**。没查清就再来一轮，
直到有把握，或者到达调查次数上限。

最后一步不交给模型。结论由固定规则和证据决定：证据够不够、有没有硬命中、有没有互相打架。
模型只提建议。查不完也不会硬判，而是把已经查到的东西交给人。

```text
商品上架
  ▼  规则 + 机审结果初筛
  ▼  拿不准 → 进入调查循环
  ▼  通过 / 拒绝 / 交给人
```

## 快速开始

需要 Python 3.12+ 和 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync                                     # 装依赖
uv run pytest -q                            # 跑测试：不需要网络，也不需要密钥
uv run python scripts/demo_walkthrough.py   # 离线跑通一次完整调查
```

要真实检索或 Langfuse 观测时，再装可选依赖（两个 `--extra` 要同时写）：

```bash
uv sync --extra rag --extra observability
```

## 跑一次真实的调查

```bash
cp .env.example .env      # 填 DATABASE_URL（MySQL）和 DEEPSEEK_API_KEY（模型网关）
uv run python scripts/demo_api.py
```

它会打印一份裁决摘要：结论、风险等级与类型、置信度、证据条数，以及这次用掉的模型调用数和 token。

也可以起 HTTP 服务：

```bash
uv run uvicorn pra.api.app:app --reload    # http://127.0.0.1:8000
curl http://127.0.0.1:8000/api/v1/health   # {"status":"ok"}
```

`POST /api/v1/reviews` 的请求体是一条商品案件：商品快照 + 商家 + 事件类型 + 机审结果。
字段定义见 [`src/pra/domain/models.py`](src/pra/domain/models.py)。

## 评测

```bash
uv run python scripts/run_evaluation.py                  # 三个方案对比：规则 / 单次模型 / Agent
uv run python scripts/run_regression.py                  # 和基线比对，确认结论没变
uv run python scripts/run_evaluation_real.py --limit 10   # 用真实模型跑：要密钥、有费用
```

评测怎么做、最近一次结果是什么，见 [docs/02-evaluation.md](docs/02-evaluation.md)。

## 目录

```text
src/pra/
  agent/        调查流程：状态、图、节点、护栏、检查点
  tools/        六个取证工具：商品 / 商家 / 图片 / OCR / 案例 / 政策
  screening/    规则初筛与三分流
  rag/          政策库与案例库检索
  evaluation/   评测
  api/  infra/  domain/  observability/
docs/  migrations/  scripts/  tests/  deploy/
```

需要落库、向量库或观测服务时：建表 SQL 在 [`migrations/`](migrations/)，本地部署见 [`deploy/`](deploy/)。

## 文档

- [docs/00-system-design.md](docs/00-system-design.md) —— 系统设计与判定规则
- [docs/02-evaluation.md](docs/02-evaluation.md) —— 评测怎么做与最近结果

## 许可证

[MIT](LICENSE)
