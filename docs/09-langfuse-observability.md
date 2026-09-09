# Langfuse 可观测性接入（设计定稿）

> 状态：**进行中**（2026-09-09 用户拍板解冻，**仅此一项**，不扩到 OTel 全家桶）。
> S2 适配层已提交 `288feb8`；S3 埋点进行中；S4/S5 未开始 —— 逐阶段实况见 §10。
> 口径约定：全文数字均标注来源与口径；未实测的一律写「未验证」，不预支结论。
> 配套：`deploy/langfuse/README.md`（本机 Docker 部署与镜像源实测）、
> `docs/07-project-status.md`（内部记忆，gitignore；Langfuse 决策节待落笔）。

## 1. 定位（勿偏移）

本项目已有五层各司其职的结构，Langfuse 只补其中**一格**，不越界：

| 层 | 回答的问题 | 本项目落点 |
|---|---|---|
| LangGraph | Agent **怎么执行**（节点/路由/循环） | `src/pra/agent/graph.py`（5 节点 + 2 条件边） |
| Tools / RAG | 获取**什么外部信息** | `src/pra/agent/tools_node.py` + `src/pra/tools/` |
| Guardrails / Gate | 如何**安全收敛**（预算、降级、确定性终裁） | `src/pra/agent/guardrails/`（`gate.run_decision_overlay`） |
| Evaluation | 效果**好不好**（acc / HRR / abstention） | `src/pra/evaluation/`（口径权威出处 `docs/02-evaluation.md`） |
| **Langfuse** | **实际怎么跑的、为什么成功/失败** | 本文档（`src/pra/observability/`） |
| MySQL | 业务**审计 / 持久化 / 结果溯源** | `review_run` / `review_trace` / `review_result` / `review_evidence` |

三条硬边界（写进 `src/pra/observability/__init__.py` 模块 docstring）：

1. Langfuse **不替代 Evaluation**：acc / HRR / 消融 / 回归 digest 仍由评测侧算，
   Langfuse 只提供「这次跑的 span 树长什么样」的观测证据。
2. Langfuse **不替代 MySQL `review_trace`**：业务审计与合规取证仍在 MySQL（§2）。
3. Langfuse **不是业务真相源**：裁决值唯一来自 `review_result`；观测数据可丢、
   可采样、可清理，丢了不影响任何业务结论。

## 2. 职责边界（MySQL `review_trace` vs Langfuse）

| 维度 | MySQL `review_trace` | Langfuse |
|---|---|---|
| 目的 | 业务审计、合规取证、结果溯源 | 研发观测、failure analysis、实验对比 |
| 粒度 | **每步一行摘要**（节点/工具），`seq` 连续自增 | **完整 span 树**（root → node → generation/tool/gate） |
| 内容 | 轻量状态摘要 + 计数 + `latency_ms` + `tokens` 差分 | prompt/response **全文** + model + usage + latency + 错误堆栈 |
| 生命周期 | **永久**（业务档案，不清理） | 可清理（按环境/时间/采样率淘汰） |
| 关联键 | `run_id`（`review_trace.run_id → review_run.run_id`，`UNIQUE(run_id, seq)`） | `trace_id`（32-hex，与 `run_id` 同值 → 两侧可互跳） |
| 是否可关闭 | **不可关闭**（业务链路必备） | **可关闭**（无凭据 → `NullTracer`，零网络零开销） |

**明确不做**：

- 不把 prompt / response **全文**塞进 `review_trace`（审计表存摘要，全文属观测域；
  也避免业务库膨胀与敏感内容入库）。
- 不为接 Langfuse 而**删除或重写** `review_trace`：两者并行，`review_trace` 的
  表结构、写入时机、`seq` 语义一律不动。

## 3. 接入方式：最小侵入手动埋点（不用 LangChain/LangGraph 自动 callback）

**为什么不用自动回调**：本项目的 LLM 是**裸 litellm 后端**
（`src/pra/agent/litellm_backend.py` 的 `LiteLLMBackend`，经 `set_llm_backend()` 注入到
`src/pra/agent/guardrails/llm_shell.py`），它**不是 LangChain Runnable / ChatModel**。
LangChain 的 `CallbackHandler` 只能挂在 LangGraph 图与 LangChain 组件上，**拿不到
`backend.complete()` 这一层**——能拿到的只有图结构（节点进出），拿不到
prompt/response/usage/重试次数。自动埋点在本项目会得到「节点骨架 + 空 generation」，
恰好漏掉最需要看的 LLM 细节。

**因此采用** `src/pra/observability/` 适配层 + **四处显式埋点**：

| 文件 | 角色 |
|---|---|
| `src/pra/observability/tracing.py` | 薄接口（`Tracer` / `Observation` / `TraceContext`）+ `NullTracer` + 确定性采样；**不 import 任何 tracing SDK** |
| `src/pra/observability/langfuse_backend.py` | **唯一 import SDK 的模块**（惰性 import，SDK 缺失 → `NullTracer`） |
| `src/pra/observability/__init__.py` | 对外导出（业务侧只依赖薄接口） |
| `tests/test_observability_tracing.py` | 16 用例（假 client，无网络、无 SDK 依赖） |

四类观测（与 `tracing.py` 模块 docstring 表一致）：

| 接口 | 埋点位置 | 记录内容 |
|---|---|---|
| `trace_root` | 三个图调用点（§4.1） | case_id / run_id / experiment / tags / session |
| `node_span` | `graph.py` 的 5 个 `add_node` 包装（§4.2） | 节点名 + 入参摘要 |
| `llm_generation` | `llm_shell.py` 内层 `backend.complete()`，**每次真实调用一条**（§4.3） | model / input / output / usage / latency / error |
| `tool_span` | `tools_node.py` 的 `await tool.call()`（§4.4） | tool 名 / args / output / latency / status |

**四条不变式**（`tests/test_observability_tracing.py` 守护，实测 16 passed）：

1. 无凭据 → `NullTracer`，且**不 import `langfuse`**（断言 `sys.modules` 无该键）；
2. 所有观测调用**绝不抛异常**（观测失败不得影响业务）；
3. 采样判定**确定性**（同 key 同结果），保证评测可重放；
4. 埋点**不写 `AgentState`、不参与路由**（`add_edge` / 条件边 / 决策序列零变化）。

> 编号说明：`tracing.py` / `test_observability_tracing.py` 的 docstring 引用「docs/09 §3 / §4」
> 是 S2 阶段草稿编号；本文定稿后 **§3 = 接入方式、§4 = 埋点位置、§5 = 关联**。
> 代码注释未同步修改（S2 已提交、范围冻结），以本文为准。

## 4. 埋点位置（逐个落点）

### 4.1 Root trace：三处

| # | 文件 | 调用点 | trace_id 口径 | source tag |
|---|---|---|---|---|
| 1 | `src/pra/api/service.py` | `await app.ainvoke(build_initial_state(case), config)` | **`run_id`** | `source:http` |
| 2 | `src/pra/infra/persist_service.py` | `app.astream(..., stream_mode="updates")` | **`run_id`** | `source:http` |
| 3 | `src/pra/evaluation/harness/agent_scheme.py` | `await graph.ainvoke(initial_state, {"configurable": {"thread_id": f"eval-agent-{case.eval_case_id}"}})` | **uuid5（确定性）** | `source:evaluation` |

- HTTP / 落库路径：`trace_id = run_id`。`run_id` 由 `run_review()` 里 `uuid4().hex`
  生成 = **32 位 hex**，同时是 LangGraph `thread_id`，并与 MySQL
  `review_run.run_id`（`VARCHAR(64)` 主键）**一一对应** → Langfuse 侧看到的 trace
  可直接用 `run_id` 反查 MySQL 审计链（`review_run → review_trace / review_evidence
  / review_result`）。
- root observation 名默认 `review`；整体 input 放 **root observation**（v4 中 trace 级
  input/output 已废弃，见 §6）。output 在终态写回（decision 摘要）。
- 三处均为「包住图调用」的 `with tracer.trace_root(ctx):`，不改调用参数、不改返回。

### 4.2 Node span：5 个

在 `src/pra/agent/graph.py` 的 `builder.add_node()` 处**统一包装**节点函数，节点名取自
既有常量（路由表唯一事实源）：

```text
N_HYPOTHESIZE = "hypothesize"   N_PLAN = "plan"    N_TOOLS = "tools"
N_REEVALUATE  = "reevaluate"    N_DECIDE = "decide"
```

约束：**不改节点内部实现、不改图拓扑**（`add_edge` / `add_conditional_edges` 一行不动）。
span 记录节点名 + 轻量入参摘要（复用 `persist_service._node_input_summary` 的口径思路，
不新造摘要函数语义）。

### 4.3 LLM generation：唯一入口

`src/pra/agent/guardrails/llm_shell.py` 的 `call_structured_llm()` 是**节点调 LLM 的唯一入口**。
generation span 必须包在**内层 `backend.complete()` 外层**（不是包 `call_structured_llm` 整体）：

- 一次 `call_structured_llm` **最多 2 次真实调用**（`for attempt in (1, 2)`）：
  - **schema 修正重试**（后端成功返回但 `model_validate_json` 失败 → 回喂第 1 次非法输出原文）；
  - **transport 退避重试**（`backend.complete` 抛异常 → `_transport_backoff` 后原样重试）；
  - **截断路径不重试**（`finish_reason=length` 且校验失败 → `attempts=1` 直接降级）。
- 因此**每次真实调用单独记录一条 generation**，条数与 `LLMCallOutcome.attempts` 对齐；
  这样「第 1 次为什么失败、第 2 次改了什么」在 Langfuse 里可逐条看到。
- 每条记录：`model`（= `backend.name` 原值）/ `input`（messages）/ `output`（content）/
  `usage_details`（`resp.usage` 非 None 时才传）/ `latency_ms` + `attempt` + `truncated`
  （放 metadata）/ 错误（`record_error` → `level=ERROR` + `status_message`）。
- span 名 `llm.{node}`（如 `llm.hypothesize`），便于按节点过滤。

### 4.4 Tool span：唯一入口

`src/pra/agent/tools_node.py` 的 `await tool.call(parsed, ctx)`（含 infra 失败重试的第 2 次调用）。
**复用已有字段，不重算**：`latency_ms`（工具侧计时）、`args`、`status`（`ok` / `error`）、
`error` —— 这些字段已经写在 `tool_call_history` 的 audit record 里并落 `review_trace`
（`step_type=TOOL_CALL`），Langfuse 侧直接读同一 record，避免两处计时口径漂移。

### 4.5 Gate span：不是图节点

Gate **不是 Graph Node**：`run_decision_overlay`（`src/pra/agent/guardrails/gate.py`）
由 `decide` 节点内部调用（`src/pra/agent/nodes/decide.py` 末尾 `final = run_decision_overlay(...)`）。
因此做法是**在 decide 内加一个子 span**：

- **不新增 Graph Node、不改路由、不改 Gate 判定顺序**（`overlay_state` 只读快照）；
- span 记录：规则命中序列（overrides / 风险等级收敛结果）与 proposal → final 的差异摘要，
  用于回答「这次为什么从 LLM 提案被 Gate 改写」。

## 5. 关联与实验版本

**metadata**（`TraceContext.metadata`，至少支持）：

| 键 | 含义 | 来源 |
|---|---|---|
| `case_id` | 业务案件 ID | `ProductReviewCase.case_id` |
| `eval_case_id` | 评测案 ID（与业务 `case_id` **解耦**） | `EvalCase.eval_case_id`（如 `EC_0001`） |
| `scene` | 五类场景标签 | `EvalCase.scene` |
| `scheme` | `rule` / `single_call_llm` / `agent` | `pra.evaluation.runner.ALL_SCHEMES` |
| `experiment` | 实验名（如 `baseline` / `prompt-v2`），同时下发为 trace `version` | 调用方注入 |
| `tool_world` | `eval` / `rag` 世界 | `EvalContext.tool_world` |
| `rag_mode` | 检索模式（bm25 / vector / hybrid） | RAG 侧 |
| `source` | `http` / `evaluation` | 三个 root 埋点（`persist_service` 即 HTTP 落库路径，与 `api/service` 同为 `http`） |
| `dataset` | 数据集标识（`v1` / `v2`） | 数据路径 |
| budget limits | 预算上限快照（`max_llm_calls` / `max_tool_calls` / `max_tokens` / `max_latency_ms`） | `BudgetLimits` |

**tags**（UI 过滤用）：`env:local` / `scheme:agent` / `experiment:baseline` / `dataset:v2` /
`source:evaluation` / `source:http`。

**评测侧口径**：

- v2 数据集 **320 案**（`eval_data/v2/cases_v2.jsonl`，实测 `wc -l` = 320 行）→
  **每案一条 root trace**；
- **只给 `agent` scheme 打 trace**：`rule` / `single_call_llm` 无图执行（无节点、无
  LLM 壳、无工具循环），**不伪造空 trace**——否则 UI 上会出现一堆无内容的 trace，
  反而污染对比；
- 评测路径 `trace_id = uuid5(experiment:eval_case_id:scheme)` —— **确定性**：
  同 experiment + 同案 + 同方案重跑**落同一条 trace**（可覆盖、可对比，不产生重复）；
- `session_id` = 一次 evaluation run 的标识 → 把该 run 的 320 条 trace 聚成一个会话，
  UI 按 session 过滤即「这一轮评测的全部案件」；
- 采样：`PRA_LANGFUSE_SAMPLE`，**默认 1.0（全采）**；判定必须**确定性** ——
  `should_sample(key=trace_id, sample)` 用 `sha256(key)` 前 8 hex 落桶，
  `sample<=0` 恒 False、`sample>=1` 恒 True（`tracing.should_sample`，测试守护）。

**结论**：多实验、多轮次、多数据集的观测数据**靠 experiment / session / tag / filter
管理，不靠删数据**（清理只作为容量手段，不作为版本隔离手段）。

## 6. 版本与 API（本机实测，非网上示例）

实测环境：本机 **2026-09-09**，SDK `langfuse==4.15.1`（独立 spike venv，实测
`langfuse.__version__` = `4.15.1`）；**服务端选 v4**（官方 v4 于 2026-08-17 发布，
v3 进入维护期仅安全补丁 —— 据 Langfuse 官方 [changelog](https://langfuse.com/changelog/2026-08-17-langfuse-v4)
与 [compatibility 文档](https://langfuse.com/docs/compatibility)，**未独立复核其版本策略原文**）。

| # | 实测结论 | 证据 |
|---|---|---|
| 1 | 入口是 **client 实例**：`get_client().start_as_current_observation(...)`；**模块级同名函数不存在** | 实测 `hasattr(langfuse, "start_as_current_observation")` → `False` |
| 2 | 签名参数（`inspect.signature` 实读）：`trace_context` / `name` / `as_type` / `input` / `output` / `metadata` / `version` / `level` / `status_message` / `completion_start_time` / `model` / `model_parameters` / `usage_details` / `cost_details` / `prompt` / `end_on_exit` | 实测签名列表 |
| 3 | `trace_context={"trace_id": <32-hex>}` **实测生效**（回读 trace_id 一致）→ 可与 `review_run.run_id` 硬对齐 | S0 spike 实测 |
| 4 | 子观测靠 OTel contextvar **自动嵌套**：`await` / `asyncio.create_task` / `asyncio.gather` 三种调度**均继承同一 trace**（LangGraph 节点在独立 task 里跑，这点必须成立） | S0 spike 实测 |
| 5 | trace 级属性用 `propagate_attributes(session_id / metadata / tags / version ...)` 下发（实读参数：`user_id` / `session_id` / `metadata` / `version` / `tags` / `trace_name` / `environment` / `prompt` / `as_baggage`） | 实测签名 + 适配层使用 |
| 6 | **v4 中 trace 级 input/output 已废弃** → 整体 input/output 放 **root observation** | S0 spike 实测 |
| 7 | 观测类型 `as_type="tool"` / `"generation"` / `"span"` 受支持 | S0 spike 实测 |
| 8 | 服务端不可达时 SDK **后台**打印 `Transient error ...` 重试日志 → 因此**未启用时不构造 client**（避免噪音与重试开销） | S0 spike 实测 |

适配层落点：`langfuse_backend.py` 的 `LangfuseTracer` 全部 SDK 调用包 `try/except`
（观测旁路，异常静默）；`build_langfuse_tracer` 在 **SDK 未安装**时返回 `NullTracer`
（实测当前主 venv `import langfuse` → `ModuleNotFoundError`，故本仓库默认路径走
NullTracer 分支；要真接线需 `uv sync --extra observability`）。

> `pyproject.toml` 的 `observability` extra 自骨架期（`3e73067`）就带
> `opentelemetry-sdk` / `opentelemetry-instrumentation-fastapi`。本项目**不启用** OTel
> 自动埋点 / Collector（§11）；Langfuse SDK 内部基于 OTel 的 contextvar 机制属实现细节，
> 不引入额外基础设施。

## 7. Token / Latency 口径（诚实边界）

| 字段 | scripted 桩（默认路径） | real litellm（评测脚本） | 说明 |
|---|---|---|---|
| `model` | 后端**自报**名：`scripted-walkthrough`（默认桩）/ `eval-scripted-reviewer`（评测桩） | `litellm-deepseek/deepseek-chat`（`LiteLLMBackend.name = f"litellm-{model}"`） | 取 `backend.name` 原值，**不编模型名**；桩无真实模型是事实 |
| `input` / `output` | ✅ 可记（messages / content 文本） | ✅ 可记 | 观测域全文，不进 `review_trace` |
| `tokens` | **恒 0** | 仅 `usage.total_tokens`（input+output 合计、含 provider 缓存命中） | 桩：`scripted_llm.py` 返回 `LLMResponse(..., tokens=0)`；real：`litellm_backend.py` 只取 `usage.total_tokens` |
| tokens 拆分 | `None`（桩无真实 token） | `{"input": prompt_tokens, "output": completion_tokens, "total": total_tokens}` | 已新增可选字段 `LLMResponse.usage: dict \| None`（键名对齐 Langfuse `usage_details`）；**取不到的键不放**（不填 0 冒充）、整份拿不到 → `None`。属**扩展字段**，既有 `tokens` 口径不变 |
| `latency` | ✅ 可测（桩调用同样计时） | ✅ 可测 | **LLM 路径原本没有任何计时**：改造前 `call_structured_llm` 内无 `perf_counter`（S3 新增 `perf_counter` 计时）；业务侧只有 `review_trace.latency_ms` 的「节点到达墙钟差近似」（`persist_service.py` 注释自述「近似」）→ 新增计时属**真实新增观测数据**，不覆盖、不篡改既有字段 |
| `cost` | **空** | **不计算**（交由 Langfuse 按 model 定价自行推算；推不出就空） | 桩路径无模型 → Langfuse 算不出 cost 是正确结果 |

**红线：绝不伪造 token / cost / 模型名。** 桩路径 token 恒 0、无模型 → Langfuse 上
「token=0 / cost 空」是**真实情况**，不是埋点缺陷；不得为了图表好看填假值。

## 8. 环境变量与配置

| 变量 | 作用 | 默认 | 状态 |
|---|---|---|---|
| `LANGFUSE_PUBLIC_KEY` | 项目公钥（本地 Docker 初始化值 `pk-lf-pra-local`） | 无 → `NullTracer` | ✅ 已实现 |
| `LANGFUSE_SECRET_KEY` | 项目私钥（`sk-lf-pra-local`） | 无 → `NullTracer` | ✅ 已实现 |
| `LANGFUSE_HOST` | 服务端地址 | `http://localhost:3000` | ✅ 已实现 |
| `PRA_LANGFUSE_ENABLED` | 总开关（`0/false/no/off` 关闭） | 未设 = 启用（仍需凭据） | ✅ 已实现 |
| `PRA_LANGFUSE_EXPERIMENT` | 实验名 → metadata.experiment + trace version | 未设 | ⚠️ **规划中**：当前代码未消费（实测 `grep` 无引用），S5 接线 |
| `PRA_LANGFUSE_SAMPLE` | 采样率（确定性判定） | `1.0` | ✅ 已实现 |

**配置层陷阱（本项目真实踩过，commit `6cf763e`）**：`pra.infra.db.Settings` 是
`pydantic-settings` + **`extra="forbid"`**，且 `env_file` 指向仓库根 `.env`。
因此**往 `.env` 加任何新键都必须同步在 `Settings` 里声明字段**，否则
`Settings()` 直接抛 `ValidationError` → `get_sessionmaker → process_review` 全线失败
（历史上 `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` 就是这样把 `POST /api/v1/reviews`
打成恒 500 的）。接 Langfuse 时的正确做法：照 `deepseek_api_key` 的先例，在 `Settings`
里补 `langfuse_public_key` / `langfuse_secret_key` / `langfuse_host` 等**可选字段**
（默认 `None`，infra 不消费），**`extra="forbid"` 保持不变**（未知键仍报错，防 DSN 拼错
静默连错库）。**该同步当前尚未落地（实测 `Settings` 只声明了 `database_url` +
`deepseek_*`），属 S3/S5 必做项。**

## 9. 部署（本地 Docker 自托管）

官方 v4 compose 需要 **6 个常驻服务**（另加一次性 `minio-init` 建桶容器）：

| 服务 | 作用 |
|---|---|
| `langfuse-web` | UI + 公开 API（宿主 3000） |
| `langfuse-worker` | 异步消费/写 ClickHouse（宿主 3030） |
| `postgres` | 元数据（组织/项目/用户/密钥） |
| `redis` | 队列/缓存 |
| `clickhouse` | trace / observation 明细存储 |
| `minio` | S3 对象存储（大 payload） |
| `minio-init`（一次性） | `mc alias set` + `mc mb --ignore-existing local/langfuse` |

**本机实测约束（如实记录）**：

- Docker **29.7.2**（Docker Desktop for Mac）；
- `registry-1.docker.io` / `hub.docker.com` / `raw.githubusercontent.com` 与
  **GitHub HTTPS 全部超时**，只有 GitHub **SSH** 可用；
- Docker 已配镜像源但**大镜像极慢**：13.7MB 的 alpine 用显式镜像源 15 秒成功；
  **100MB 级镜像 10 分钟未完成** → 故改为**显式镜像源逐层拉取 + `docker tag` 还原规范名**
  （compose 里的镜像名保持规范名不变）；4 个可用镜像源
  （`docker.1panel.live` / `docker.1ms.run` / `hub.rat.dev` / `docker.m.daocloud.io`）
  实测 `redis:7-alpine`（58.7MB）均 33–34 秒成功，故 6 个镜像**分散到 4 源并行拉取**；
- 官方 compose 的 minio 用 `cgr.dev/chainguard/minio`（非 Docker Hub，本机 `403` 拉不到）
  → 替换为 **`minio/minio`**，并相应调整 `entrypoint` / `command`
  （`server --address ":9000" --console-address ":9001" /data`）/ `healthcheck`
  （`curl -f http://localhost:9000/minio/health/live`）/ 建桶方式（改用 `minio-init`）；
- **端口占用检查**（`lsof -nP -iTCP -sTCP:LISTEN` 实测全部空闲）：3000 / 3030 / 5432 /
  6379 / 8123 / 9000 / 9090 / 9091；
- 部署文件落在 `deploy/langfuse/`：`docker-compose.yml`（261 行，vendored 官方 v4 并按
  上述差异调整）+ `.env`（**gitignore，不入库**）+ `README.md`（部署/镜像源/排障全流程）。

## 10. 实施顺序与当前状态（S0~S6）

| 阶段 | 内容 | 状态 | 证据 / 说明 |
|---|---|---|---|
| S0 | SDK spike（API 形状实测） | ✅ **已完成** | 4 项实测：client 实例入口与签名 / `trace_context` 生效 / contextvar 三种调度嵌套 / `as_type` 类型（详见 §6） |
| S1 | 本地 Docker Langfuse v4 | 🔄 **镜像拉取中** | 实测 6 个 `docker pull` 在跑；已到位 `redis:7-alpine`（4 源）、`minio/minio:latest`、`alpine:3.20` 探针；`langfuse-web` / `langfuse-worker` / `clickhouse` / `postgres:17` / `redis:7` / `minio/mc` **未完成**；**容器尚未启动，服务未验证** |
| S2 | 适配层（`tracing.py` + `langfuse_backend.py`） | ✅ **已提交 `288feb8`** | 16 用例全绿（实测 `uv run pytest tests/test_observability_tracing.py -q` → `16 passed`）；**零业务改动** |
| S3 | LLM + Tool 埋点 | 🔄 **进行中（未提交、未验证）** | 埋点代码已落在**工作树**：`llm_shell.py` / `tools_node.py` 已调 `get_tracer()`；`litellm_backend.py` 已透出 `usage`（`LLMResponse.usage`）。**尚未提交、未跑测试、未端到端验证** |
| S4 | Node + Root + Gate 埋点 | ⏳ **未开始** | 埋点位置已定稿（§4.1/4.2/4.5） |
| S5 | Evaluation 接线 | ⏳ **未开始** | `PRA_LANGFUSE_EXPERIMENT` / `session_id` / `uuid5` 确定性 trace_id **均未实现**（实测 `grep` 为空）；`Settings` 字段同步亦未做（§8） |
| S6 | 文档 | 🔄 **进行中** | 本文档即其一；`deploy/langfuse/README.md` 已就绪 |

> 状态一律如实标注：**S1 服务未起、S3 未提交未验证、S4/S5 代码未接线**，任何「已接入」
> 的说法在 S4/S5 完成并实测端到端 trace 之前都不成立。
> 本节状态为**本文落笔时快照**（2026-09-09 22:30 前后）；S3 正在并行落地，后续以
> `git log` 与 `docs/07-project-status.md` 为准。

## 11. 明确不做（范围冻结）

以下项**不在本次范围**，不做也不承诺：

- OpenTelemetry 全套基础设施（Collector / 自动埋点 / 跨服务传播）；
- Phoenix、LangSmith、Coze Loop、Ragas；
- 自定义 Dashboard、Prometheus / Grafana、生产告警；
- Kubernetes 部署；
- Remote Qdrant、Reranker；
- MQ Worker、Redis 分布式锁、MySQL Checkpointer 续跑；
- Human Feedback 系统；
- **修改 Evaluation metrics**、**修改 Gate 判定**、**修改 Agent 图拓扑**。

理由：本项目解冻范围 = 「把已有的确定性 Agent 链路观测出来」，任何改动评测口径、
Gate 行为或图结构的工作都会污染既有 347 全绿基线与 v1/v2 回归 digest。

## 12. 结论边界

- 本项目观测目标是**面试项目级可演示的最小完整闭环**（root → node → generation/tool/gate
  能看到、能关联到 MySQL `run_id`），**不是生产级 Observability 平台**：无采样治理、
  无容量规划、无多租户、无告警、无 SLA。
- **默认路径（scripted 桩 + InMemory 世界）下 trace 结构完整，但 token / cost 为空**——
  这是**真实情况**（桩无模型、`tokens=0`，§7），不是缺陷，也不得填假值补齐。
- **真实 LLM 只出现在 real 评测脚本**（`scripts/run_evaluation_real.py`，注入
  `LiteLLMBackend`）；服务运行时 LLM 恒为 scripted 桩（`AGENTS.md` 口径红线）。
- real LLM 目前只跑过 **v1 35 案单次抽样**（acc 0.200 / HRR 0.771，出处
  `docs/02-evaluation.md`）：因此「real 路径的 Langfuse 观测」样本极小、非确定性，
  不可当能力证据。
- S1 服务端与端到端 trace 落库**尚未验证**（§10）；本文档所有 API 结论来自 SDK 侧
  spike 实测，**服务端 UI 回读未验证**（待 S1 起服务后确认）。

## 13. 验收命令

```bash
# 1) 全量测试（含适配层 16 用例；无需任何 Langfuse key）
uv run pytest tests/ -q

# 2) 评测确定性回归（不烧 key、不联网）
uv run python scripts/run_regression.py

# 3) 服务端健康检查（先起本地 Langfuse：cd deploy/langfuse && docker compose up -d）
curl -s http://localhost:3000/api/public/health
```

端到端发一条 trace（**待实现**，S5 后补）：

```bash
# 占位脚本：scripts/demo_langfuse_trace.py —— 尚未创建（实测 ls 不存在）
# 预期用法（待实现后确认）：
#   export LANGFUSE_PUBLIC_KEY=pk-lf-pra-local
#   export LANGFUSE_SECRET_KEY=sk-lf-pra-local
#   export LANGFUSE_HOST=http://localhost:3000
#   uv run --extra observability python scripts/demo_langfuse_trace.py --case-id <CASE>
# 预期行为（未验证）：发一条 root trace + node/llm/tool span，flush 后在
#   http://localhost:3000 按 trace_id（= run_id）查到完整 span 树。
```

启用真实 Langfuse 的可选依赖（当前主 venv 未安装，实测 `import langfuse` 失败）：

```bash
uv sync --extra observability   # 装 langfuse SDK；不装则适配层恒走 NullTracer
```
