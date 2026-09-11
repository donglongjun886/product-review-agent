# ChromaDB 部署（本地开发 · RAG `HttpClient` 服务端模式）

> 本目录只做一件事：把 **ChromaDB 服务端**跑起来，让 `pra.rag` 的
> **`HttpClient(host="127.0.0.1", port=8001)`** 装配路径可以被真实验证。
> 选型与实施契约以 [docs/10-rag-upgrade-spec.md](../../docs/10-rag-upgrade-spec.md) 为准；
> 本文件只记录**部署事实与实测结果**。

## 1. 定位（勿偏移）

- **ChromaDB 是本仓库 RAG 的向量库后端**（docs/10 §0 拍板）：`ChromaDB 1.5.9`，
  **Docker 服务端 + Python `HttpClient`**（不进程内起库，与「真服务端」叙事一致）。
- Chroma 只承担「**存向量 + 算余弦**」；元数据过滤、BM25、融合、Top-K 排序仍在
  **Python 侧**（同口径前提，docs/10 §3/§6-R2）。
- **`deploy/qdrant` 暂不删除**：Qdrant 是 docs/06 Phase 2 的向量库，`rag_backend="qdrant"`
  与 `scripts/run_rag_phase2_demo.py` 仍在（docs/10 §2「暂留不删」）。但**本机容器已卸、
  `qdrant_qdrant_storage` 卷亦已删除**（`docker ps -a` / `docker volume ls` 均无）——要复跑
  那 3 个真服务端集成用例得先 `cd deploy/qdrant && docker compose up -d` 重来。
  两套部署的端口**互不冲突**（Qdrant 6333/6334 vs Chroma 8001）；**Qdrant 暂留不删（去留另定，docs/10 §0）**，
  迁移期保留其代码与部署以便对照/复跑。
- **检索升级已实施**：本目录交付**部署**，检索升级（LlamaIndex + BGE 向量路 + BM25(jieba)
  + RRF）与 `src/pra/rag/chroma_backend.py`（docs/10 §4「新增」清单）**均已落地**，
  实施契约以 [docs/10-rag-upgrade-spec.md](../../docs/10-rag-upgrade-spec.md) 为准；
  `factory.py` 已提供 `backend="chroma"` 装配开关（见 §6）。

## 2. 端口与数据

| 项 | 值 | 说明 |
|---|---|---|
| 容器内 HTTP | `8000` | 由 `CHROMA_SERVER_HTTP_PORT=8000` 指定 |
| 宿主机映射 | **`127.0.0.1:8001`** | 仅 loopback 可达，不对外暴露 |
| 数据卷 | named volume `chroma_chroma_data` → 容器 `/data` | compose 项目名取自目录名 |
| 容器名 | `chroma-chroma-1` | 同上 |

**为什么宿主机是 8001 而不是 8000**：仓库自己的 FastAPI dev server 就占 `8000`
（README「API 演示」段），两者同时跑会**端口冲突**。故宿主机侧刻意错开为 `8001`，
容器内仍保持镜像默认的 `8000`。

- 两个端口都只绑 `127.0.0.1`（与 `deploy/langfuse` 内部服务、`deploy/qdrant` 同策略）。

## 3. 镜像拉取（本机网络约束，实测）

本机 **`registry-1.docker.io` 直连超时**，必须走国内镜像源，并在拉取后
**retag 成规范名**（compose 里只写规范名，与 `deploy/langfuse` / `deploy/qdrant` 同一做法）：

```bash
docker pull docker.1panel.live/chromadb/chroma:1.5.9
docker tag  docker.1panel.live/chromadb/chroma:1.5.9 chromadb/chroma:1.5.9
```

**实测结果**（`docker images`）：

| 镜像 | DISK USAGE | CONTENT SIZE | 结果 |
|---|---|---|---|
| `chromadb/chroma:1.5.9`（retag 后规范名） | **822MB** | **165MB** | ✅ 已就位、`Up (healthy)` |
| `docker.1panel.live/chromadb/chroma:1.5.9`（源前缀名） | 822MB | 165MB | 同一 image id（`1e0b73a187a2`），仅多一个 tag |

- `chromadb/chroma` 是 Docker Hub 官方命名空间下的仓库，**不加 `library/` 前缀**。
- 若某个源卡住，换另一个源重拉即可——已下载的层会复用（详见 `deploy/langfuse` README §5.1
  的镜像源测速表）。

## 4. 启动 / 停止 / 状态 / 验证

```bash
cd deploy/chroma
docker compose up -d            # 启动（首次会创建数据卷）
docker compose ps               # 状态（应变 healthy）
docker compose logs -f chroma   # 日志
docker compose down             # 停止（保留数据卷）
docker compose down -v          # 停止并删除数据（彻底重置）
```

**启动耗时（实测）**：轮询 `docker inspect --format '{{.State.Health.Status}}'`：

| 场景 | 结果 |
|---|---|
| `docker compose down` → `up -d`（冷启动） | **5.5 秒**转 `healthy` |
| `docker compose restart chroma`（复用容器） | **6.6 秒**转 `healthy` |

> ⚠️ 时间数值的出处澄清（避免误引）：**本目录 compose 注释并没有写「约 9 秒」** ——
> 「约 9 秒转 healthy」是 **`deploy/qdrant/README.md`** 里 **Qdrant** 的实测值；docs/10 未给数值。
> 本目录实测更快（冷启动 5.5s / restart 6.6s，本机 Docker Desktop），两者**未深究差异**
> （可能是机器负载或首次拉层解压）。**以你机器上的 `docker compose ps` 为准。**

> ⚠️ `docker compose down -v` 会**删掉整块向量库数据**（含集成测试建的 collection）——
> 这是**单卷、无备份**的部署，删了只能重灌 corpus（见 §9）。

**实测旁注**：本服务端是**共享单实例** —— 任何进程建 collection 都落在同一个
`chroma_chroma_data` 卷里，`docker compose down -v` 一删全没。排查时可用
`GET /api/v2/tenants/default_tenant/databases/default_database/collections` 看当前有哪些
collection（本机写作时除集成测试临时 collection 外，还观察到此前实测流程留下的
`probe_filters`，非本目录所建、未清理）。

host 侧探针（**实测输出逐字如下**，2026-09-10，`chroma-chroma-1 Up (healthy)`）：

| 探针 | HTTP | 响应体 | 说明 |
|---|---|---|---|
| `GET /api/v2/heartbeat` | **200** | `{"nanosecond heartbeat":1789008753155745762}` | ✅ 唯一可用于健康检查的探针 |
| `GET /api/v1/heartbeat` | **410** | `{"error":"Unimplemented","message":"The v1 API is deprecated. Please use /v2 apis"}` | ⚠️ **v1 已废弃**，见下 |
| `GET /api/v2/version` | **200** | `"1.0.0"` | **API 版本**，不是包版本 `1.5.9` |
| `GET /health` | **404** | *（空）* | **该端点不存在** |
| `GET /` | **404** | *（空）* | 无根路由 |

```bash
# 复现上面这张表
for p in /api/v2/heartbeat /api/v1/heartbeat /api/v2/version /health; do
  printf "%-22s HTTP=%s  " "$p" "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8001$p)"
  curl -s http://127.0.0.1:8001$p; echo
done
```

> **踩坑提醒**：
> 1. **`/api/v1/*` 在 Chroma 1.x 上整体不可用**（410 + `Please use /v2 apis`）。网上大量
>    示例（含各家 healthcheck 片段）仍在打 `/api/v1/heartbeat` —— 照抄会**恒判失败**。
> 2. **没有 `/health`**：这不是 Qdrant/langfuse 那种自带 `/readyz`、`/api/public/health`
>    的服务，探针必须打 `/api/v2/heartbeat`。
> 3. `/api/v2/version` 返回的 `"1.0.0"` 是**API 版本**；镜像/包的版本是 **1.5.9**，别混用。

## 5. 镜像特性（**实测**）：Rust 内核，无 python / curl / wget

```
$ docker run --rm --entrypoint sh chromadb/chroma:1.5.9 -c 'command -v python python3 curl wget bash sh'
python: MISSING    python3: MISSING   curl: MISSING   wget: MISSING
bash: /usr/bin/bash   sh: /usr/bin/sh
```

| 探测项 | 实测值 |
|---|---|
| 基础镜像 | `Debian GNU/Linux 13 (trixie)` |
| Python / curl / wget | **全部 MISSING**（Chroma 1.x 是 **Rust 内核**，不再带 Python 运行时） |
| Shell | 有 **`bash`**（`/usr/bin/bash`）；`sh` → `/usr/bin/dash` |
| CLI | `/usr/local/bin/chroma`（子命令：`run` / `browse` / `vacuum` / `db` / `login` …） |
| entrypoint | `dumb-init -- chroma run /config.yaml` |
| 镜像内 `/config.yaml` | 只有一行 `persist_path: "/data"` → 持久路径 = 卷挂载点 `/data` |

**所以 healthcheck 不能用常见的 `wget --spider` / `curl -f`**（两者都不存在，`python -c` 也没有），
只能用 **bash 的 `/dev/tcp`** 发最小 HTTP 请求判 `/api/v2/heartbeat` 是否 200（compose 里就是这么写的）：

```yaml
test: ["CMD", "bash", "-c",
       "exec 3<>/dev/tcp/127.0.0.1/8000 && printf 'GET /api/v2/heartbeat HTTP/1.0\\r\\n\\r\\n' >&3 && grep -q 200 <&3"]
```

> 注意两点：① **必须是 `bash`**，`sh`（dash）**没有 `/dev/tcp`** —— 实测
> `sh -c '… >/dev/tcp/127.0.0.1/8000'` 直接报 `cannot create …: Directory nonexistent`；
> ② 探测的是**容器内**的 `8000`（不是宿主机 `8001`），所以用 `127.0.0.1` 即可。

## 6. 客户端接线

服务端起好后，客户端形态（docs/10 §3 三形态）：

```python
import chromadb

client = chromadb.HttpClient(host="127.0.0.1", port=8001)   # 服务端（本目录）
# client = chromadb.EphemeralClient()                       # 进程内内存 → 离线测试恒跑
# client = chromadb.PersistentClient(path="…")              # 进程内本地持久
```

> ⚠️ **接线状态**：`src/pra/rag/chroma_backend.py`（LlamaIndex 装配）与
> `src/pra/rag/factory.py` 的 `backend="chroma"` 分支**均已落地**（docs/10 为实施契约）。
> 上面的片段既是**客户端契约**，也是仓库实现的接线方式（已用 `chromadb 1.5.9` 对本服务端实测通过）。
> `pyproject.toml` 的 `rag` extra 已含 `chromadb` + 三个 LlamaIndex 具体集成包。

> ⚠️ **实测提醒（docs/10 §3 未写、建库必须处理）**：Chroma 的 collection **缺省 `space` 是
> `l2` 而不是 `cosine`**（实测）。docs/10 §3 只说「建库需显式 `embedding_function=None`」，
> 但**同时必须显式设 `space="cosine"`**，否则 `api/v2` 返回的是 L2 距离、与 `local` 后端
> 口径不一致且**不会报错**。详见 §6.2。

### 6.1 建库必须显式 `embedding_function=None`

**实测**（对真服务端 `get_or_create_collection` 后读 `configuration_json`）：

| 写法 | 落库的 `embedding_function` | 后果 |
|---|---|---|
| `create_collection(name=…)`（省略） | `{"type":"known","name":"default","config":{}}` | ❌ 启用**默认 ONNX 嵌入函数**（会去下模型） |
| `create_collection(name=…, embedding_function=None)` | `null` | ✅ 只用我们**自带传入**的向量 |

> 上表第一行的 `"default"` 即 `chromadb.api.types.DefaultEmbeddingFunction`，其源码 docstring
> 写明 **"delegates to `ONNXMiniLM_L6_V2`"**（`__call__` 内部 import 并调用该 ONNX 模型，
> 即 MiniLM-L6-v2 的 ONNX 版）——这就是「会去下模型」的实证来源，不是推测。

我们**自带 BGE 向量**（docs/10 §0），必须显式关掉默认 EF —— 否则 Chroma 会尝试下载
ONNX 模型，既慢又与「默认不联网」基调冲突。

### 6.2 距离口径：`distance = 1 − cos`（**但 collection 的 `space` 必须显式设成 `cosine`**）

**实测**（`chromadb 1.5.9`；同一结果在 `EphemeralClient` 与 `HttpClient` 上一致）：

| 建库方式 | `space` | `cos([1,0,0], [0.9,0.1,0])` | `1 − cos` | **Chroma 返回的 distance** |
|---|---|---|---|---|
| `embedding_function=None`（不设 space） | **`l2`**（缺省） | 0.993883735 | 0.006116265 | **`0.020000005`** ❌ 不是余弦 |
| `configuration={"hnsw": {"space": "cosine"}}` | `cosine` | 0.993883735 | 0.006116265 | **`0.006116271`** ✅ |
| `metadata={"hnsw:space": "cosine"}`（旧写法） | `cosine` | 0.993883735 | 0.006116265 | **`0.006116271`** ✅ |

→ **结论：只要 collection 是 `cosine`，Chroma 的 distance 就是 `1 − cos`**（与 docs/10 §3 一致，
实测 `0.006116271` ↔ 计算值 `0.006116265`，差在浮点与向量归一化精度）。

**但缺省 space 是 `l2`（不是 cosine）** —— 这是本次实测最容易踩的坑：用
`embedding_function=None` 建库、**不改 space**，拿到的就是 L2 距离（实测 `0.020000005`，
在归一化向量上 = `2(1−cos)`），任何「相似度 = 1 − distance」的换算都会**静默错**。

- 建库两种等价写法（**都实测过，落库 `space` 均为 `cosine`**）：

  ```python
  client.create_collection("pra_policy_512", embedding_function=None,
                           configuration={"hnsw": {"space": "cosine"}})   # 新写法（推荐）
  client.create_collection("pra_policy_512", embedding_function=None,
                           metadata={"hnsw:space": "cosine"})             # 旧写法（兼容）
  ```

- **核验手段**（别只看代码）：`get_collection(...).configuration_json["hnsw"]["space"]` 必须是
  `"cosine"`；再对**单位向量**做一次自检（如 `[1,0,0]` vs 单位化的 `[0.9,0.1,0]` → 期望
  `0.006116271`，若得 `0.0122325…` 说明 `space` 是 `l2`）。
- 换算口径：`cos = 1 − distance`（`space="cosine"` 下），**`distance` 已含归一化** ——
  传 `[0.9,0.1,0]` 还是单位化的 `[0.9,0.1,0]/‖·‖`，`cosine` space 下返回的都是 `0.006116271`
  （两种输入实测同值），所以**不需要自己先归一化**；但**必须**确认 `space` 正确。

### 6.3 配置红线：**别把连接串写进仓库根 `.env`**

`Settings` 是 **`extra="forbid"`**：未声明的键会让配置校验**直接报错**、应用起不来
（`6cf763e` 踩过「POST 恒 500」的同类事故；qdrant 的 `QDRANT_*` 已有同样提醒）。
Chroma / Qdrant 的连接参数目前**只能程序化传入**，尚无配置项或 CLI 开关。

## 7. 安全

- 默认**不设鉴权**（Chroma 无内置 auth 时即为开放 HTTP API），但**仅绑 `127.0.0.1`**
  → 风险等价于本机进程。
- 若需跨机访问：**必须**改为非 loopback 绑定，并自行在其前面加**反向代理 + 鉴权**
  （Basic Auth / mTLS）或走受控内网；`ANONYMIZED_TELEMETRY=False` 已设（默认不联网），
  但**这不等于访问控制**。

## 8. 排障

| 现象 | 处理 |
|---|---|
| `docker compose up` 报镜像找不到 | 先按 §3 retag，`docker images \| grep chroma` 确认规范名在 |
| 探针恒失败 / 容器一直 `unhealthy` | 确认打的是 **`/api/v2/heartbeat`**（`/api/v1/*` → 410、`/health` → 404）；healthcheck 里**必须用 `bash`**（dash 无 `/dev/tcp`） |
| `curl 127.0.0.1:8001` 连不上 | `docker compose ps` 看是否 `Up`；`lsof -nP -iTCP:8001 -sTCP:LISTEN` 看端口占用 |
| 想彻底重来 | `docker compose down -v`（数据卷 `chroma_chroma_data` 一并删除） |
| 客户端建库后开始下模型 | 建库漏了 `embedding_function=None`（见 §6.1） |
| 检索分数与 `local` 后端对不上 | 先核 collection 的 `space` 是否为 `cosine`（缺省是 `l2`！），再核换算 `cos = 1 − distance`（见 §6.2） |

## 9. 结论边界（如实标注，勿当能力承诺）

- 这是**本机单机开发部署**：**单节点、未集群、未压测、未开鉴权**；数据落在**单个 named
  volume**，无备份、无高可用、无迁移方案。
- 语料仍是 **24 政策 / 67 案例**的小语料（docs/10 §0，沿用不变）→ **不构成能力声明**，
  更不代表 Chroma 在生产规模下的表现。
- 本目录**只交付部署**；检索升级（LlamaIndex + BGE 向量路 + BM25 + RRF）**已实施**，
  相关代码见 `src/pra/rag/`（`chroma_backend.py` 等；契约以 docs/10 为准，见 §1、§6）。
- 以上均为 `chromadb/chroma:1.5.9` 镜像 + `chromadb 1.5.9` Python 包（本机 `uv` 环境）
  的实测结果；镜像内 `chroma --version` 自报 **`1.4.4`**（CLI 自报版本与镜像 tag / Python
  包版本不同源，**未深究二者差异**，如需精确对齐请以 tag 与 PyPI 包为准）。
