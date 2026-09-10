# RAG Phase 2：Qdrant 向量库 + 本地 BGE Embedding（设计定稿）

> 状态：**已拍板并实施**（2026-09-09，对齐 rag-implementation-plan.md R-1/R-2 的
> Phase 2 路线：正式向量库 Qdrant + 本地语义 Embedding，经 `Embedder` provider 与
> 索引工厂替换，**不改检索上层 / 工具契约**）。
> 验证模式 = 工作区镜像开发 + patch 交付 + 主 agent 真实仓库落地（291 全绿基线 +
> 新增用例全绿 + v1/v2 回归 digest 零变化）。

## 0. 定位（勿偏移）

MVP（489c470）已证明「Agent 能经 RAG 获取政策/先例、Evaluation 可重放」，但
embedding 是**确定性 hash（词面特征，不是语义）**——检索质量是 mock。Phase 2 要
回答的唯一增量问题：

> 换上**真语义 embedding + 正式向量库**后，（1）替换缝是否真的零改动上层？
> （2）词面检索（BM25 / mock hash）检索不到的**同义改写 query**，语义检索能否命中？
> （3）BM25 / Vector / Hybrid 三路在真实 embedding 下的表现对比（Hybrid 是否更优，
> 由实验回答，不预设）。

**不是**：生产级向量库调优、分布式部署、大规模语料、reranker、query 改写。
语料仍为 Policy KB 24 条 / Case KB 67 条（R-3/R-4 不变）。

## 1. 技术选型拍板（P2 系列）

| # | 决策点 | 拍板 | 理由 / 边界 |
|---|---|---|---|
| P2-1 | 语义模型 | **`BAAI/bge-small-zh-v1.5`**（fastembed 官方支持列表，dim 512，onnx ~90MB） | 语料全中文；small 档轻量（onnxruntime 无 torch）；fastembed 是 Qdrant 官方生态配套（叙事一致）；本机实测 hf-mirror 可达、模型下载 + 中文 embedding 语义可分（仿冒/复刻 vs 正品 距离拉开） |
| P2-2 | embedding 运行时 | **fastembed（onnxruntime）**；`HF_ENDPOINT` 环境变量可切镜像 | 经 `Embedder` Protocol 注入（R-2 预留位）；`cache_dir` 参数化（默认 `<repo>/.cache/fastembed`，gitignore；无模型时 `BgeEmbedder.available()`=False → 调用方显式降级/报错，绝不静默回退 mock） |
| P2-3 | 向量库接入方式 | **qdrant-client 进程内模式**：`:memory:`（测试/演示默认）或 `path=<目录>`（本地持久索引）或 `url=`（远端 server，生产叙事位） | qdrant-client 是"真 Qdrant API"（collection/point/payload/query），无需起服务即可复现 CI；远端 server 仅换连接串，代码同路径（D-4） |
| P2-4 | Qdrant 承担的角色 | **只做「向量存储 + 余弦打分」**；元数据过滤（category/risk_type/status）与 BM25 / 融合 / Top-K 排序仍在 Python 侧复用 MVP 的确定性函数 | 保证与本地索引**同口径**（同分 tie-break、min-max 归一化、6 位取整 → score 语义可比），过滤边界逐条对齐（见 §2）；避免把过滤语义搬进 Qdrant filter 产生双实现漂移 |
| P2-5 | 默认路径 | `build_policy_index/build_case_index(backend="local")`（缺省）与 `build_tools("rag")` **逐字节不变**；Qdrant 仅经 `backend="qdrant"` 显式开启 | v1/v2 回归 digest 零变化；CI 无 qdrant-client 也全绿（新测试 `pytest.importorskip("qdrant_client")`） |
| P2-6 | 检索指令（BGE query instruction） | **v1 不引入**（doc/query 同域短句 plain 编码；bge-zh 检索指令优化排后续） | `Embedder.embed()` 单一入口不区分 doc/query，加指令需改契约——收益小、复杂度高，诚实标注为已知取舍 |
| P2-7 | 语义验证口径 | 定向 demo + 小型标注 probe 集三路对比（Recall@3），**如实并排不预设 Hybrid 最优** | 对齐 R-6 / 02 §7.2：结论边界 = 语料小、单模型、非 SOTA 声明 |

## 2. 架构与实现落点

```
src/pra/rag/
├── embedder.py        ← + BgeEmbedder（fastembed 封装：lazy init / cache_dir / available()）
├── qdrant_index.py    ← 新增 QdrantPolicyIndex / QdrantCaseIndex（实现 tools 层 Protocol）
├── index.py           （不动：本地 numpy 索引 = 默认路径）
├── factory.py         ← build_*_index 增 backend="local"|"qdrant"（缺省 local 零变化）
├── retrieval.py       （不动：过滤后打分/融合/Top-K 共用）
└── corpus/            （不动）
src/pra/tools/__init__.py   ← build_tools 增 rag_backend 透传（缺省 local）
scripts/run_rag_phase2_demo.py  ← 新增：BGE+Qdrant 检索演示 + 语义 vs mock 对比 + probe 三路表
tests/test_rag_qdrant.py        ← 新增：qdrant 索引单测 + BGE embedder 单测（离线守卫）
```

### 2.1 Qdrant 索引行为契约（与本地索引逐条对齐）

- **行/文本/过滤**：构造期 `rows → 强校验 record` + `_texts`（policy: title+text；
  case: summary）+ `BM25Index` 全部复用 MVP 逻辑；元数据候选过滤在 Python 侧完成
  （policy：`effective_only` 且 status=EFFECTIVE、category ∈ {None, 目标类目,
  全类目}、risk_type 交叠非空；case：category 精确、risk_type 交叠非空——与
  `RagPolicyIndex/RagCaseIndex.search` 同源码口径）。
- **collection**：每个 KB 一个（`pra_policy_<dim>` / `pra_case_<dim>`，前缀可配），
  建库 `cosine` + dim=embedder 维度；point id = `sha256(clause_id/case_id)` 稳定
  哈希 → 重建幂等（upsert 覆盖）。payload 存完整 record JSON（检索结果重组 hit 用，
  不进 evidence 的敏感字段由 corpus 脱敏保证，R-4 隔离不变）。
- **打分（三模式同口径）**：
  - `bm25`：`BM25Index.scores` + 候选内 min-max（复用 `retrieval`）；
  - `vector`：query embed → `client.query_points(vector=…, query_filter=point_id∈候选,
    limit=len(候选))` → 候选内余弦分（Qdrant cosine 输出即 0~1）；
  - `hybrid`：`fuse_scores(bm25_raw, qdrant_vec, weights)`（与本地完全同函数同权重）；
  - 排序：`(score 降序, 候选原序 idx 升序)` 在 Python 侧做、分取 6 位 —— **不信任
    Qdrant 对同分点的顺序**，确定性 tie-break 与本地一致。
- **同构测试**：同一 rows + 同一 embedder（MockHash）下，qdrant 与 local 两索引
  的 `search` 返回**同序同 id、score 6 位取整相等**（允许 ulp 级浮点差，
  Qdrant 余弦 vs numpy 余弦实现差异）→ 证明替换缝行为保持。
- **BGE 下语义生效判据**：同义改写 query（与 corpus 无共享关键词）在
  `mode="vector"/"hybrid"` 命中目标条款/先例且 `mode="bm25"` 漏检或显著降序
  —— demo 与 probe 呈现。

### 2.2 BgeEmbedder

- 构造：`model_name`（缺省 bge-small-zh-v1.5）、`cache_dir=None`（→ 环境变量
  `PRA_EMBED_CACHE_DIR` → fastembed 默认）；`dim` 属性 = 512；`embed(text)` 返回
  `list[float]`；批量内部走 `TextEmbedding.embed`（list 入参）再取首条。
- **懒加载 + 离线守卫**：`fastembed` 在首次 `embed()` 时才 import 与下载；
  `available() -> bool`（能否 import，不触发下载）；类级 `model_ready()`（cache_dir
  内模型文件已存在）供测试 skip 与 demo 预检。不可用时**显式抛错或由调用方降级**，
  不静默回退 mock（诚实标注：mock 与真模型不可混算维度/语义）。
- 确定性口径：同模型同文本同进程内 `embed` 逐位相等；跨进程/平台允许浮点尾差
  （真实模型非逐字节契约——诚实标注，回归基线仍以默认 mock 路径为准）。

### 2.3 装配

- `factory.build_policy_index/build_case_index` 增 `backend: Literal["local","qdrant"]="local"`、
  `qdrant_client=None`（外部注入，测试用）、`collection_prefix=None`；`backend="qdrant"`
  时延迟 import `pra.rag.qdrant_index`（无 qdrant-client 环境不 import 失败——
  仅显式开启才需要）。缺省路径与改动前逐字节一致。
- `build_tools(data_source="rag", rag_backend="local", rag_embedder=None)`：
  `rag_backend="qdrant"` 时经 factory 传 `backend="qdrant"` + `embedder=rag_embedder`
  （None → MockHash，保证 qdrant 后端也可离线单测）。
- `evaluation` 的 RAG 世界（`make_rag_world_tools`）**不动**（保持 mock 本地路径，
  评测确定性基线不变）；Phase 2 语义对照走 demo/probe 脚本，不进评测 digest。

## 3. 依赖与文件

- `pyproject.toml`：`rag = ["qdrant-client", "fastembed"]`（fastembed 新增）。
- `.gitignore`：+ `.cache/`（模型缓存与 qdrant 本地索引目录统一落 `.cache/`）。
- 新增：`docs/06-rag-phase2-qdrant-bge.md`（本文件）、
  `src/pra/rag/qdrant_index.py`、`scripts/run_rag_phase2_demo.py`、
  `tests/test_rag_qdrant.py`。
- 改动：`src/pra/rag/embedder.py`、`src/pra/rag/factory.py`、
  `src/pra/tools/__init__.py`、`pyproject.toml`、`.gitignore`。

## 4. 验收

1. 291 既有全绿 + 新增用例全绿（qdrant 单测经 `importorskip` 离线可跑；BGE 真模型
   用例 `skipif(model_ready()=False)` 不阻塞 CI）。
2. v1 35 案 + v2 320 案确定性 digest 与改动前**逐字节一致**（默认路径零变化）。
3. 同构测试：同 rows+同 embedder 下 qdrant ≡ local（同序同 id、score 6 位相等）。
4. 语义定向：同义改写 query 在 BGE+Qdrant 下命中目标（mock/BM25 漏检案例成立）。
5. probe 集三路 Recall@3 并排报告（bm25/vector/hybrid），不预设 Hybrid 最优。
6. 隔离红线不变：Case KB 只含 `RAG_CASE_` 前缀，与 eval GT 零交集（R-4）。

## 5. 结论边界（如实标注，勿当能力承诺）

- 语义质量 = 单模型（bge-small-zh-v1.5）单语料（24/67 条）的**定向演示与 probe
  观测**，非大规模评测；换模型/语料结果变。
- Qdrant 进程内模式 = 真 Qdrant API 但非分布式部署。**远端 server（`url=`）已于
  2026-09-10 实测**（`deploy/qdrant`，v1.19.0）：原「代码路径同一，**仅连接串差异**」
  的假设**被证伪** —— 进程内模式不校验 point id 上界，故 128 位 id 在本机全绿、
  只在真 server 以 400 暴露；已修（u64）并补真服务端集成测试，详见 §7。
- 跨进程/平台 embedding 浮点尾差不纳入逐字节契约；确定性回归恒以默认 mock 路径为准。

## 6. 实测记录（2026-09-09 · commit 2533825 后补 · run_rag_phase2_demo.py）

> 实证口径：BGE(bge-small-zh-v1.5) + Qdrant 进程内(:memory:) + hybrid 默认权重 0.5/0.5；
> 同输入两次运行逐行一致（可重放）；完整输出见 `scripts/run_rag_phase2_demo.py` 运行 stdout。

**语义 vs 词面（Part B，同义改写 query 与目标检索文本零/近零共享关键词，bm25.tokenize 程序化验证）**：

| 目标 | bm25 | vector | hybrid |
|---|---|---|---|
| POLICY_1.4_v1_c1（鞋靴仿名牌，token 交集 0） | 漏 | 命中@1 | 命中@1 |
| POLICY_4.2_v2_c1（改标题重上架规避，交集 0） | 漏 | 命中@2 | 命中@2 |
| RAG_CASE_0001（无标高仿女跑鞋+多次下架，交集 2 低信号 bigram） | 漏 | 命中@1 | 命中@1 |
| RAG_CASE_0011（冒用双 G logo 先例，交集 1） | 漏 | 命中@1 | 漏（如实呈现） |

→ 「BM25 词面漏检、语义检索命中」在 Policy 2/2、Case 2/2 成立——语义路（BGE）补上了词面路（BM25/mock hash）够不到的同义改写，即 Phase 2 相对 MVP 的真实增量。

**三路 Recall@3（Part C，Policy/Case 各 8 条人工标注 probe = 5 keyword + 3 同义改写）**：

| KB | bm25 | vector | hybrid |
|---|---|---|---|
| Policy (8) | 6/8 (75%) | 8/8 (100%) | 8/8 (100%) |
| Case (8) | 6/8 (75%) | 8/8 (100%) | 8/8 (100%) |

→ 本 probe 下 hybrid 与 vector 并列最优且 ≥ bm25；hybrid 增益全部来自 vector 语义路救回 bm25
漏掉的同义改写项，**无 hybrid 单独优于 vector 的案例**（N=8 小样本定向观测，不预设——R-6/P2-7 口径）。

**边界**：单模型单语料定向演示（24/67 条），非大规模评测；评测回归基线恒以默认 mock 路径为准
（BGE 浮点跨进程尾差不入逐字节契约）。

## 7. 远端 server（`url=`）实测与修复（2026-09-10 · `deploy/qdrant`）

**动机**：§5 曾把 `url=` 标注为「未实测、仅连接串差异」——用真服务端验证该假设。

**部署**：`deploy/qdrant`（`qdrant/qdrant:v1.19.0`，仅绑 `127.0.0.1:6333/6334`，
healthcheck 用 bash `/dev/tcp` 判 `/readyz`——官方镜像无 curl/wget）。镜像须走国内源
拉取后 retag（`registry-1.docker.io` 直连实测超时），命令见该目录 README §3。

**发现（假设证伪）**：`build_policy_index(backend="qdrant", location="http://127.0.0.1:6333")`
→ **400** `value … is not a valid point ID`。根因：`_point_id` 取 sha256 **前 16 字节
（128 位 int）**，而真 server 只收 **u64 或 UUID**；**qdrant-client 进程内模式不校验
id 上界** → `:memory:` / `path=` 与既有测试全绿，缺陷被「未实测」掩盖。

**修复**：`_point_id` 改为 `digest()[:8]` → u64（保留 `int` 类型，调用点零改动；
同键碰撞由既有 `len(set(ids))` 校验拦截）。id 变更同时作用于本地路径 → 既有 `path=`
持久索引需重建（本仓库 `.cache/` 无遗留索引，无迁移负担）。

**验证（真 server）**：policy KB **24 点** / case KB **67 点**全量落库成功；检索顶层序
与 `local` 后端**逐条一致**（§2.1 同构契约在真 server 上成立）；id 接受性直测通过。

**回归守护（两条互补路径）**：
- `tests/test_rag_qdrant_point_id.py` —— **不依赖 qdrant-client，任何环境恒跑**（含 CI）：
  钉住 id ∈ [0, 2⁶⁴-1] + corpus 内无碰撞 + 稳定性。**必须独立成文件**：其余 qdrant
  测试都在顶层 `importorskip("qdrant_client")`，而 **CI 只跑 `uv sync --frozen`（不装
  extra）→ 那些文件在 CI 上整文件 skip**（即 qdrant 后端此前在 CI 上零覆盖）。
- `tests/test_rag_qdrant_server.py` —— 真 server 端到端（建库 / 全量 upsert 点数 /
  与 local 同口径 / R-4 前缀 / id 接受性）；服务端不可达则 skip，`PRA_QDRANT_URL` 可覆盖。

全量测试：**439 passed, 1 skipped**（服务端不在时 **436 passed, 4 skipped**）；v1/v2 回归双 PASS。
