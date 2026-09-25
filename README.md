# product-review-agent

[![tests](https://github.com/donglongjun886/product-review-agent/actions/workflows/test.yml/badge.svg)](https://github.com/donglongjun886/product-review-agent/actions/workflows/test.yml)
[![python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

机器审核能挡掉大部分明显违规，但总有一批案子说不准：文本合规没有硬命中、商品在库事实对不上、
商家有违规前科，案例与政策里又没有能直接套用的先例。这个项目处理的就是这批案子 ——
它自己决定去查什么，查完给出三种结论：**通过、拒绝，或者交给人**。

## 它怎么工作

先看平台已有的规则和机审结果：明显正常的直接放行，明显违规的直接拒绝，**剩下拿不准的才进入调查**。

调查是一个循环：**提风险假设 → 定查证计划 → 调工具取证 → 复核假设**。没查清就再来一轮，
直到有把握，或者到达调查次数上限。取证沿四条线走：**文本合规、商品在库事实、商家行为、案例与政策**。

最后一步不交给模型。结论由固定规则和证据决定：证据够不够、有没有硬命中、有没有可引用的依据。
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
```

要真实检索或 Langfuse 观测时，再装可选依赖（两个 `--extra` 要同时写）：

```bash
uv sync --extra rag --extra observability
```

配好 `LANGFUSE_*`（`.env` 或环境变量）+ `cd deploy/langfuse && docker compose up -d` 后，
可跑一次观测接线自检（发一条合成 trace 再从服务端读回断言；无凭据时退出码 0 并提示启用方式）：

```bash
uv run --extra observability python scripts/langfuse_smoke.py
```

## 跑一次真实的调查

```bash
cp .env.example .env                       # 填 DATABASE_URL（MySQL）和 DEEPSEEK_API_KEY（模型网关）
uv run uvicorn pra.api.app:app --reload    # http://127.0.0.1:8000
curl http://127.0.0.1:8000/api/v1/health   # {"status":"ok"}
```

`POST /api/v1/reviews` 受理一条商品案件并同步跑完整个调查，返回
`{"run_id": ..., "review_decision": {...}}`：结论、风险等级与类型、置信度与证据链。
请求体 = 商品快照 + 商家 + 事件类型 + 机审结果，字段定义见
[`src/pra/domain/models.py`](src/pra/domain/models.py)。

## 评测

```bash
uv run python scripts/run_evaluation.py --limit 10 --out /tmp/eval.json   # 三臂：single / agent 要 API key，有费用
```

评测只有一个脚本，跑三条臂：`rule`（确定性初筛，零模型零工具）、`single`（一次模型直判，只喂案件
快照）、`agent`（完整调查，允许调工具）。事实来源是**生产装配**：商品与商家读真 MySQL，案例与政策
走真 RAG。只有 `agent` 用工具，`rule` 与 `single` 只看案件快照。真实调用有费用、结果不确定、不可
重放，不进 CI。案源是 `eval_data/v2/cases_v2.jsonl`，它的真值仍是旧口径标的，**待按规则书重标**。
评测口径见 [docs/02-evaluation.md](docs/02-evaluation.md)。

## 目录

```text
src/pra/
  agent/        调查流程：状态、图、节点、护栏、检查点
  tools/        四个取证工具：商品 / 商家 / 案例 / 政策
  screening/    规则初筛与三分流
  rag/          政策库与案例库检索
  evaluation/   评测：数据集、指标、记录
  api/  infra/  domain/  observability/
docs/  migrations/  scripts/  tests/  deploy/
```

需要落库、向量库或观测服务时：建表 SQL 在 [`migrations/`](migrations/)，本地部署见 [`deploy/`](deploy/)。

## 文档

- [docs/00-system-design.md](docs/00-system-design.md) —— 系统设计与判定规则
- [docs/02-evaluation.md](docs/02-evaluation.md) —— 评测口径：三臂、事实来源与真值路线

## 许可证

[MIT](LICENSE)
