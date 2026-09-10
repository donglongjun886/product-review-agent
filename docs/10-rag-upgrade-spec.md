# RAG 真实化升级规格（LlamaIndex + BGE + ChromaDB + BM25 + RRF）

> **状态**：2026-09-10 拍板，待执行。本文是**唯一实施契约**（subagent 按此交付，主 agent 按此验收）。
> 目标：把 RAG 做成「**真实可运行、选型合理、面试可讲清楚**」，然后工程收尾，**不扩 RAG 能力**。
> 不扩大范围：沿用现有 **24 Policy + 67 Case**；不爬外网、不做 reranker / ES / Milvus / KG / 增量索引平台。

## 0. 拍板记录（2026-09-10）

| 编号 | 决策点 | 拍板 | 说明 |
|---|---|---|---|
| C1 | `CaseHit.similarity` 口径 | **重命名为 `retrieval_score`**；Hybrid 下允许表示 RRF 分；**不得**把 RRF 分解释成语义相似度 | **Evidence 结构不变**（仍 `weight=retrieval_score`），不牵动 Agent / Gate / Evaluation |
| C2 | LlamaIndex 边界 | **全量 LlamaIndex**（Index / Node / Retriever 全面接管） | 业务层边界仍由 `PolicyIndex` / `CaseIndex` Protocol 守护 |
| C3 | BM25 | **换成熟库 + `jieba` 中文分词**（不手搓） | ⚠️ **引擎名与 C2 冲突，见 §6-R1（未决）** |
| — | 向量库 | **ChromaDB**（Docker 服务端 + HttpClient） | Qdrant **暂不删除**（迁移定稿后再定去留） |
| — | Embedding | 真实 RAG 默认 **BGE**（`BAAI/bge-small-zh-v1.5` + fastembed）；**测试/回归仍走 `MockHashEmbedder`** | 确定性契约不变 |

## 1. 目标链路

```
Query
  → Hybrid Retrieval（Policy KB / Case KB 各自独立）
       ├── VectorRetriever : ChromaDB(cosine) + BGE 向量
       └── BM25Retriever   : 成熟 BM25 引擎 + jieba 分词
  → RRF 融合（Reciprocal Rank Fusion）
  → Top-K
  → PolicySearchTool / CaseSearchTool（现有契约，一行不改）
  → ToolResult → Evidence → Agent Reevaluate / Decision
```

## 2. 复用边界（铁律）

**`PolicySearchTool` / `CaseSearchTool` 的索引注入契约（`PolicyIndex` / `CaseIndex` Protocol）不得改变。**
这是本次改动波及面的**唯一闸门**：向量库、检索器、融合算法全部换掉，Agent / Gate / Evaluation / Evidence **零改动**。

| 模块 | 处置 |
|---|---|
| `tools/policy_search`、`tools/case_search`（除 C1 字段改名） | ✅ 复用；契约不变 |
| `rag/corpus/`（24+67 JSON + schema + R-4 隔离） | ✅ 复用，数据不动 |
| `rag/embedder.py`（`Embedder` Protocol / `MockHashEmbedder` / `BgeEmbedder`） | ✅ 复用，另加「包成 LlamaIndex `BaseEmbedding`」适配层 |
| `rag/index.py`（local numpy） | ✅ **保留**为确定性基线（默认后端不变） |
| `rag/vectors.py`、`rag/retrieval.py` | ✅ 保留（local 后端与既有单测仍用） |
| `rag/factory.py`、`build_tools(data_source, rag_backend)` | 🔧 扩展：新增 `rag_backend="chroma"` |
| `rag/qdrant_index.py` | ⏸️ **暂留不删**（`rag_backend="qdrant"` 保持可用）。**注意状态差异**：代码与 `deploy/qdrant` **保留在仓库**，但**本机容器/镜像/数据卷已卸**（恢复 = `cd deploy/qdrant && docker compose up -d`，镜像走国内源约 30s）；因此 pytest 里 3 个 Qdrant 真服务端用例当前为 skip |
| `scripts/run_rag_eval.py` | 🔧 扩展为三路/多后端对比 |
| `scripts/run_rag_phase2_demo.py` | ⏸️ 暂留（Qdrant Phase 2 演示，不删） |

## 3. 实测钉死的实现契约（**不要照抄网上示例**）

**ChromaDB 服务端（本机 `127.0.0.1:8001`，容器内 8000）**

- **`/api/v1` 已废弃**：实测 `GET /api/v1/heartbeat` → **410 Unimplemented**（"Please use /v2"）；只有 `/api/v2/*` 可用。healthcheck 用 `/api/v2/heartbeat`。
- `GET /api/v2/version` 返回 **`"1.0.0"`**（API 版本，非包版本 1.5.9）。
- 镜像 **`chromadb/chroma:1.5.9` 是 Rust 内核，内部无 `python` / `curl` / `wget`**（Debian 13，有 `bash`）→ 探针只能 bash `/dev/tcp`。
- 建库需显式 `embedding_function=None`，否则 Chroma 会启用**默认 ONNX 嵌入函数**（会去下模型）——
  我们**自带 BGE 向量**，必须显式关掉。证据：省略 EF 时落库配置为
  `{"type":"known","name":"default","config":{}}`，而 `DefaultEmbeddingFunction` 源码 docstring
  逐字写 “delegates to `ONNXMiniLM_L6_V2`”。
- 🔴 **建库必须显式指定 `space="cosine"` —— Chroma 缺省是 `l2`，此坑静默且致命**：
  用 `configuration={"hnsw": {"space": "cosine"}}`（等价旧写法 `metadata={"hnsw:space": "cosine"}`）。
  实测同一对向量 `[1,0,0]` vs `[0.9,0.1,0]`：

  | 建库方式 | 落库 space | Chroma 返回 distance | 判定 |
  |---|---|---|---|
  | 只写 `embedding_function=None`（**缺省**） | **`l2`** | **`0.020000005`** | ❌ = 2(1−cos)，口径静默错 |
  | `configuration={"hnsw":{"space":"cosine"}}` | `cosine` | **`0.006116271`** | ✅ = 1−cos |
  | `metadata={"hnsw:space":"cosine"}` | `cosine` | `0.006116271` | ✅ |

  → **`相似度 = 1 − distance` 只在 cosine space 下成立**。实现必须（a）建库时显式指定 cosine；
  （b）**运行期自检**：读 `collection.configuration_json["hnsw"]["space"] == "cosine"`，并用单位向量
  校验 `1 − distance == numpy 余弦`。**不得假设** —— L2 库不报错，只会让 §5-1「与 local 同口径」
  静默失败（本文件初稿正是漏写了这条，由 subagent 实测发现后回改）。

  **⚠️ 更隐蔽的第二层（实测，必须一并防）**：对**已存在**的 l2 collection，
  `get_or_create_collection(..., configuration={"hnsw":{"space":"cosine"}})` **与**
  `metadata={"hnsw:space":"cosine"}` **都不会改建其空间**（实测两次重取仍是 `l2`、距离仍是
  `0.020000005`）→ **只在新库写配置不足以保证语义正确，复用路径必须校验**。
  实现已在**创建与复用两条路径**都断言（不符即 `ValueError` 并给出「删除重建 / 换 prefix」指引）；
  主 agent 独立验证：埋一个 l2 库 → 同前缀构造 → **被正确拦截**，未静默沿用。

  **float32 尾差（实测）**：自距离可为 **`-1.1920929e-07`** 而非 `0.0` → 自检须用 `abs() <= 1e-6`；
  经 `1 − distance` 可能得 `1.0000001`，由取分函数夹取保证 ⊂ [0,1]（否则 `CaseHit` 的 `ge=0, le=1`
  会炸）。**另注**：自距离在 l2 空间同样为 0 → **空间判定的唯一真值来源是读 configuration**，
  自检只是第二道防线（分工已写进实现 docstring，防后人误以为自检可替代空间断言）。
- 客户端三形态：`HttpClient(host, port)`（服务端）/ `EphemeralClient()`（内存，**离线测试恒跑**）/ `PersistentClient(path=)`（本地）。

**LlamaIndex**

- ⚠️ **`QueryFusionRetriever` 必须显式 `num_queries=1`**：其默认值会**调用 LLM 生成 query 变体** → 既引入非确定性、又需要 LLM（本项目检索侧**不允许**有任何 LLM 调用）。
- 🔴 **`QueryFusionRetriever` 的融合不能用于本链路（实测，反直觉）**：它融合时**原地改写跨检索共享的
  node `.score`**（源码 `reranked_nodes[-1].score = score`）→ 实测同一 query 连查 3 次**首次结果即不同**
  （`[('RAG_CASE_0066',0.030118)]` vs `[('RAG_CASE_0061',0.016667)]` …），复用 fusion/retriever 时同一
  node 会被重复计入（分 > `1/(k+1)`）。**故 RRF 自算**（`chroma_backend._fuse_rrf`，口径与它逐条一致、
  `k=60`）；保留一个「`num_queries=1` + `MockLLM()` 守卫」的正确构造器供引用/测试。
- 🔴 **`ChromaVectorStore.query` 的分数不是 `1 − distance`**：实测为 **`exp(-distance)`**
  （同一对向量返 `0.993902` vs `1−distance = 0.9938837`）→ **向量取数必须走 Chroma 原生
  `collection.query` 的 distance** 自行换算；装配面（`TextNode` / `MetadataFilters` /
  `ChromaVectorStore`）仍走 LlamaIndex。
- **`risk_type` 无法下推**：Chroma 1.5.9 对**列表字段**无可用成员算子（`$in`/`$eq` 恒不命中、
  列表嵌 `$in` 报类型错、`$contains` 又无 LlamaIndex 算子映射）→ 向量路只下推 `category`，
  `risk_type` 与 BM25 路**共用同一 Python 谓词**（单实现，无漂移）。
- **Chroma 拒空列表 metadata**：`risk_type: []` 的行**不写该键**（67 case 中 16 条、24 policy 中 1 条），
  语义等价（缺键 == 空列表）。
- **`llama-index-retrievers-bm25 0.8.0` 无 tokenizer 注入点**：`from_defaults(tokenizer=…)` 已
  deprecated（只 warning、不生效），构造与检索两处硬编码 `bm25s.tokenize` → 只能受控替换
  `bm25s.tokenize`（jieba + 模块级锁）。**检索期也必须在同一上下文内**，否则首跑即
  `ValueError: The maximum token ID in the query (379) is higher than the number of tokens in the index.`
- 用 `llama-index-core` + 3 个具体集成包（`vector-stores-chroma` / `retrievers-bm25` / `embeddings-fastembed`），**不装 `llama-index` 伞包**（伞包会拖进 `llama-index-llms-openai` / `embeddings-openai` 等本项目不用的 OpenAI 集成）。
- 兼容性实测：`vector-stores-chroma` 要求 `chromadb>=0.5.17` → **兼容 1.5.9** ✅。
- 待验证风险：`llama-index-core` 依赖 `nltk` —— **必须实测确认离线（无网）不触发数据下载**（我们不做句子切分，理论上不触发；需给证据）。

**确定性（不可退化）**

- 检索链路**零 LLM、零随机**；排序 tie-break 规则：**分数降序 + corpus 原序（idx 升序）**，分保留 **6 位**。
- 同分竞争不依赖底层库的返回顺序（Chroma / BM25 引擎的顺序都不可信）。

**Node / Chunk 口径（不切碎）**

- Policy：**1 条款 = 1 Node**；Case：**1 先例 = 1 Node**。
- metadata 至少含：policy → `clause_id / policy_id / version / category / risk_type / status / effective_date`；case → `case_id / category / decision / risk_level / risk_type`。

**Metadata 过滤（⚠️ 有口径后果，见 §6-R2）**

- 向量路：经 LlamaIndex `MetadataFilters` 下推到 Chroma `where`（store 侧过滤）。
- BM25 路：`BM25Retriever` 无等价过滤 → **Python 侧**先过滤候选。
- → 两路过滤位置**不再统一**（原 docs/06 P2-4 刻意在 Python 侧单实现以避免双实现漂移）。**必须如实写进文档**，不得含糊。

## 4. 文件级改动

**新增**

- `src/pra/rag/chroma_backend.py` —— LlamaIndex 装配：Chroma 向量存储 + Node 构造 + 双 Retriever + RRF，实现 `PolicyIndex` / `CaseIndex` Protocol。
- `src/pra/rag/llama_embedding.py` —— `Embedder` → LlamaIndex `BaseEmbedding` 适配（Mock 与 BGE 均可注入）。
- `tests/test_rag_chroma.py` —— 离线（`EphemeralClient` / 桩嵌入）恒跑用例。
- `tests/test_rag_chroma_server.py` —— 真服务端集成（不可达自动 skip，沿用 `test_rag_qdrant_server.py` 的 skip/清理惯例）。
- `deploy/chroma/README.md` —— 部署与实测记录。

**修改**

- `src/pra/rag/factory.py`、`src/pra/tools/__init__.py` —— 新增 `backend="chroma"` 开关。
- `src/pra/tools/case_search/tool.py` —— `CaseHit.similarity` → **`retrieval_score`**（含 InMemory 种子、排序、`to_evidence` 的 `weight=` 赋值、docstring）；**`CaseHit` 之外的 `similarity`（`image_analysis` / `evidence.py` 的 IMAGE_SIMILARITY）一律不动**。
- `src/pra/rag/index.py`、`src/pra/rag/qdrant_index.py`、`src/pra/rag/retrieval.py`、`src/pra/rag/corpus/schema.py` —— 随字段改名同步（仅字面）。
- `pyproject.toml` + `uv.lock` —— rag extra 增补 LlamaIndex/BM25/jieba 依赖。
- `README.md`、`docs/06`（加「已被 docs/10 取代」标注，历史记录保留）、`docs/00`/`docs/02` 相关段。

**不改**（红线）

- Agent / Gate / Evidence / Evaluation 逻辑；`PolicySearchTool` / `CaseSearchTool` 的 args/result 形状（除 `CaseHit` 字段名）。

## 5. 验收标准

1. **同构等价——按模式分别断言**（初稿笼统写「同序同 id」是把三种模式混为一谈，实测后按下表改写，
   否则 bm25/hybrid 会被误读成未达标）：

   | 模式 | 与 `local` 的关系 | 断言 |
   |---|---|---|
   | `vector` | **同序同 id**（实测通过） | 同序同 id + 打分 6 位一致（容差 1e-6：Chroma float32 存算，实测尾差 ≤1e-6） |
   | `bm25` | **不可比**（本地 = 自写 Okapi + CJK 字符 bigram、全库 IDF；Chroma 路 = `bm25s` + jieba 真词、语料 = 候选 node） | 只断言「候选完整、可复现、R-4 隔离」，**并如实记录口径差异** |
   | `hybrid` | **不可比**（本地 = `0.5·norm(bm25)+0.5·cos`，量纲 [0,1]；Chroma 路 = **RRF** `Σ1/(60+rank)`，实测 ~0.0167–0.0331） | 同上；且 `retrieval_score` 必须标注为 RRF 分（C1） |

   实测记录（2026-09-10，主 agent 独立探针 + SA-1 探针，结论一致）：vector 模式 policy/case
   同序同 id ✅；case 分差 `[0,0,0,0,1e-6]`。
2. **确定性**：同一输入两次运行结果逐字节一致。
3. **R-4 隔离**：Case KB 命中 `case_id` 全部 `RAG_CASE_` 前缀，与 eval GT 零交集。
4. **三路并排报告**：`bm25` / `vector` / `rrf-hybrid` 的 probe `Recall@3` **并排输出，不预设 RRF 最优**（口径同 docs/06 §6）。
5. **离线恒跑兜底**：CI 只跑 `uv sync --frozen`（**不装任何 extra**）→ `chromadb` / LlamaIndex 都不在
   → **chroma 相关测试文件在 CI 上必然整文件 skip**（这是事实，必须如实写进测试 docstring，不得声称
   「CI 覆盖」）。因此在 CI 上真正跑得动的守护只有一条，且必须补上：
   **「默认路径零额外依赖」契约测试**（默认 `build_policy_index()` / `build_tools()` 之后，
   `chromadb` / `llama_index` / `jieba` / `bm25s` **不得进入 `sys.modules`**）—— 它只用默认依赖，
   恒跑，且正好守住「不装 extra 也能用」这条承诺。
6. **回归零漂移**：`uv run pytest tests/ -q` 全绿；`run_regression.py`（v1/v2）**双 PASS**（RAG 不在默认评测路径，决策序列不得变）。
7. **CI**：push 后 GitHub Actions success；`uv.lock` **必须保持规范源 `pypi.org`**（实测：用国内镜像 lock 会把 registry 写成镜像源 → 绝不允许提交，改完必须 `grep -c tuna uv.lock` = 0）。
8. **真服务端**：`deploy/chroma` 起服务后，集成测试真跑通（落库点数 == corpus 行数；检索与 local 同口径）。

## 6. 未决 / 风险（执行前必须处理）

- **R1（未决，最要紧）**：`llama-index-retrievers-bm25 0.8.0` 实测依赖 **`bm25s` + `pystemmer`**，**不是 `rank_bm25`**。→ C3「换 rank_bm25」与 C2「全量 LlamaIndex」**字面冲突**。需二选一：
  - **(a)** 用 LlamaIndex `BM25Retriever`（引擎 = `bm25s`）+ `jieba` 分词 —— 满足「不手搓 + 成熟库 + 中文分词」的**意图**，但引擎不是 rank_bm25；
  - **(b)** 坚持 `rank_bm25` → 自写 Retriever 包一层，LlamaIndex 不再「全量」。
- **R2（口径后果）**：过滤位置分裂（向量路 store 侧 / BM25 路 Python 侧），见 §3。需在文档如实标注。
- **R3（依赖体量）**：LlamaIndex 最小集成组合实测 **+35 包**（含 `nltk` / `networkx` / `banks` / `aiosqlite` / `bm25s` / `jieba` / `pystemmer`），现项目共 138 包 → 约 +25%。公开仓库需评估是否可接受。
- **R4（默认嵌入）**：真实 RAG 默认切 BGE 后，**CI / 无模型缓存环境**必须优雅降级或 skip（沿用 `BgeEmbedder.available()` 与「绝不静默回退 mock」约定）。
- **R5（R RF 不可解释为相似度）**：已由 C1 处置（字段改名 `retrieval_score`），文档与 docstring 必须同步措辞，禁止再写「相似度」。

**执行期新增（2026-09-10，SA-1 实测与主 agent 复查）**：

- **R6（BM25 分词替换的健壮性，未解决）**：`bm25s.tokenize` 的受控替换 + 模块级锁**未做并发正确性实测**；
  多线程/多进程场景下这是脆弱点（库无注入点所致）。可选出路：换用 lib 的其它注入路径、或直接包 `bm25s`
  不用 LlamaIndex 的 retriever（会削弱「全量 LlamaIndex」）。**待拍板**。
- **R7（两路检索文本不一致，未解决）**：`BM25Retriever` 内部用 `MetadataMode.EMBED` → **case 的 BM25 索引
  文本里混入 `case_id` / `category` / `decision` 等 metadata 字面值**，而向量路只 embed 正文 →
  两路文本口径不同（BM25 可能因 metadata 字面命中）。是否裁剪为「只索引正文」**待拍板**。
- **R8（hybrid 权重尺度变化对 Agent 的影响）——已查清，无影响**：RRF 分让
  `CaseHit.retrieval_score` 从 ~0.89 变成 ~0.033，进而 `Evidence.weight(CASE_PRECEDENT)` 同步变小。
  读码确认：**Agent/Gate 侧对 `CASE_PRECEDENT` 只按「类型 + `ref_id` 存在」判定**
  （`gate.CITABLE_TYPES` / `converge` 同义、`scripted_llm` 按存在性分支），**从不读其 weight**；
  `quality_filter` 与 `gate._strong_similarity`、`evidence.py` 的三处门限**全部只作用于 `IMAGE_SIMILARITY`**。
  → 无阈值破坏。**唯一残留**：weight 数值仍会出现在证据记录/prompt 中（真实 LLM 可见），
  量纲需在报告中说明。
- **R9（评测侧是否给 chroma 开开关，待拍板）**：`agent_scheme.make_rag_world_tools` 目前只走
  `local` / `qdrant`；`tool_world="rag"` 的评测是否切到 chroma，需决定（影响 RAG 侧评测口径的切分）。

## 7. Subagent 分派

| 单元 | 交付物 | 依赖 | 文件边界（避免并行冲突） |
|---|---|---|---|
| **SA-1** Chroma+LlamaIndex 向量路 | `src/pra/rag/chroma_backend.py`、`llama_embedding.py`、`factory.py`/`tools/__init__.py` 开关 | §3 契约 | 独占 `src/pra/rag/` 新增文件 |
| **SA-2** BM25 路 + RRF | BM25 Retriever 接线 + RRF + `retrieval_score` 口径 | **R1 拍板** | 独占 `src/pra/rag/rrf.py`（若需要） |
| **SA-3** 测试 | `tests/test_rag_chroma.py`、`tests/test_rag_chroma_server.py` | SA-1 接口 | 独占两个测试文件 |
| **SA-4** 部署与文档 | `deploy/chroma/README.md`、docs 同步 | 无（**可立即并行**） | 独占 `deploy/chroma/` 与文档段 |
| **SA-5** 验收脚本 | `scripts/run_rag_eval.py` 三路对比 + probe 报告 | SA-1/SA-2 | 独占该脚本 |
| **主 agent** | 字段改名落地、pyproject/lock、回归、验收、提交、CI | 全部 | 冲突合并与最终裁决 |

> 协作协议（AGENTS.md）：subagent 沙箱通常只覆盖工作区 → **镜像开发 + patch 交付，主 agent 在真实仓库落地并跑验收**；接口冻结（§3）是并行前提。
