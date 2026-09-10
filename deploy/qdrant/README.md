# Qdrant 部署（本地开发 · RAG `url=` 远端模式）

> 本目录只做一件事：把 **Qdrant 服务端**跑起来，让 `pra.rag` 里那条
> **`location="http://…"`（`url=` 远端 server）**的装配路径可以被真实验证。
> 服务端的定位、参数与技术选型以 [docs/06-rag-phase2-qdrant-bge.md](../../docs/06-rag-phase2-qdrant-bge.md)
> 为准；本文件只记录**部署事实与实测结果**。

## 1. 定位（勿偏移）

- docs/06 **P2-3** 已拍板：Qdrant 以 **qdrant-client 进程内模式**接入 ——
  `:memory:`（测试/演示）、`path=<dir>`（本地持久）、`url=`（远端 server，**生产叙事位**）。
- docs/06 **§5** 当时如实标注：**「远端 server 未实测（代码路径同一，仅连接串差异）」**。
  本目录补的就是这一条 —— 起真服务端，实测 `url=` 路径。
- **该假设实测不成立**（见 §7）：远端不是「只差连接串」，存在真实阻断缺陷。
- Qdrant 承担的角色不变（**P2-4**）：**只做「向量存储 + 余弦打分」**；元数据过滤
  （category / risk_type / status）、BM25、融合、Top-K 排序**仍在 Python 侧**复用 MVP
  的确定性函数，以保证与本地索引同口径。

## 2. 端口与数据

| 项 | 值 | 说明 |
|---|---|---|
| REST + Web UI | `127.0.0.1:6333` | `/dashboard` 为内置 UI |
| gRPC | `127.0.0.1:6334` | qdrant-client 默认走 REST，未显式启用 gRPC |
| 数据卷 | named volume `qdrant_qdrant_storage` | `docker volume rm` 即彻底重置 |
| 容器名 | `qdrant-qdrant-1` | compose 项目名取自目录名 |

- **两端口都只绑 `127.0.0.1`**，不对外暴露（与 `deploy/langfuse` 内部服务同一策略）。
- 端口冲突实测：本机已占用 `3000/3030/5432/6379/8123/9000/9090/9091`（langfuse）、
  `3306`（mysql-dev）；**`6333/6334` 空闲**，故未做端口改写。

## 3. 镜像拉取（本机网络约束，实测）

本机 **`registry-1.docker.io` 直连不通**（实测 `HTTP=000`，15s 超时），必须走国内镜像源，
并在拉取后 **retag 成规范名**（compose 里只写规范名，与 `deploy/langfuse` 同一做法）。

```bash
docker pull docker.1panel.live/qdrant/qdrant:v1.19.0
docker tag  docker.1panel.live/qdrant/qdrant:v1.19.0 qdrant/qdrant:v1.19.0
```

**镜像源实测**（2026-09-10）：

| 源 | `qdrant/qdrant` manifest 探测 | 结论 |
|---|---|---|
| `registry-1.docker.io`（直连） | `HTTP=000`，15s 超时 | ❌ 不可用 |
| `docker.1panel.live` | `HTTP=200`（1.35s），实拉成功 | ✅ 本次使用 |
| `docker.m.daocloud.io` | `HTTP=401`（需 token，属正常） | 可用（备用） |

**tag 选型**：`v1.19.0` —— 实测镜像源上**最新的真实 tag**（`v1.20.0` 起返回 `404`），
且与 `.venv` 内 `qdrant-client==1.19.0` **同 minor 版本**。镜像体积实测：
`docker images` 显示 DISK USAGE **287MB** ／ CONTENT SIZE **74.8MB**。

## 4. 启动 / 停止 / 状态

```bash
cd deploy/qdrant
docker compose up -d                 # 启动（首次会创建数据卷）
docker compose ps                    # 状态（应显示 healthy）
docker compose logs -f qdrant        # 日志
docker compose down                  # 停止（保留数据卷）
docker compose down -v               # 停止并删除数据（彻底重置）
```

首次启动实测 **约 9 秒转为 `healthy`**。

## 5. 验证（host 侧）

```bash
curl -s http://127.0.0.1:6333/            # {"title":"qdrant - vector search engine","version":"1.19.0",...}
curl -s http://127.0.0.1:6333/readyz      # all shards are ready
curl -s http://127.0.0.1:6333/collections # {"result":{"collections":[]},"status":"ok",...}
curl -sI http://127.0.0.1:6333/dashboard  # HTTP/1.1 200 OK（Web UI）
```

容器 healthcheck 用 `bash` 的 `/dev/tcp` 发最小 HTTP 请求判 `/readyz` 是否 200 ——
**官方镜像（Debian 13）内既无 `curl` 也无 `wget`**（实测 `command -v` 均 MISSING），
故不能用常见的 `wget --spider` 写法。

## 6. 与仓库接线（`url=` 模式）

`pra.rag.factory` 按 `location` 前缀分派（`qdrant_index._make_client`）——
`http(s)://` → `QdrantClient(url=…)`；`:memory:` → `location=`；其余路径 → `path=`：

```python
from pra.rag.factory import build_policy_index, build_case_index

idx = build_policy_index(backend="qdrant", location="http://127.0.0.1:6333")
# 或注入自有 client（需要 api_key / gRPC 等参数时）：
# from qdrant_client import QdrantClient
# build_policy_index(backend="qdrant", qdrant_client=QdrantClient(url="...", api_key="..."))
```

> **不要**把连接串写进仓库根 `.env`：`Settings` 是 `extra="forbid"`，未声明的键会让
> 配置校验直接报错（`6cf763e` 踩过「POST 恒 500」）。目前 `url=` **只能程序化传入**，
> 尚无配置项或 CLI 开关 —— 是否补开关属后续方案范围。

## 7. 实测发现的阻断缺陷（**已修**，保留复盘）

> 装好服务端后第一次跑 `url=` 路径即撞上：**docs/06 §5「远端 server 仅连接串差异」的假设
> 被证伪**。下面是完整复盘（§7.1 = 修复）。

**现象**（真 server + 仓库既有代码，未做任何改动）：

```
qdrant_client.http.exceptions.UnexpectedResponse: Unexpected Response: 400 (Bad Request)
{"status":{"error":"Format error in JSON body: value 281249463334913001576625440764477603211
 is not a valid point ID, valid values are either an unsigned integer or a UUID"}}
```

**根因**：`src/pra/rag/qdrant_index.py` 的 `_point_id()`

```python
return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:16], "big")  # 128 bit
```

取 sha256 **前 16 字节** → **128 位整数**；Qdrant 服务端只接受 **u64 整数或 UUID**。
qdrant-client 的**进程内模式对 id 类型宽容**（本地实现不校验上界），所以
`:memory:` / `path=` 全绿，**只有打到真 server 才暴露**。

**影响面**：仅 `url=` 远端路径。默认路径（`local` numpy / `:memory:` / `path=`）不受影响 ——
`tests/test_rag_qdrant.py` 只覆盖 `:memory:` 与 `path=`，**url/server 路径零测试**，
这正是 docs/06 §5「未实测」隐藏的缺口。

**反证（仅将 id 截为 u64，其余代码不动，内存打补丁、未改仓库）**：

| 验证项 | 结果 |
|---|---|
| 真 server 上建库 + 全量 upsert | ✅ `srvprobe_policy_256` = **24 点**、`srvprobe_case_256` = **67 点** |
| 与 `local` 后端检索同口径 | ✅ top3 逐条一致（`POLICY_1.4_v1_c1` / `POLICY_3.5_v1_c1` / `POLICY_1.1_v2_c1`） |
| 点 id 类型探测 | 128bit int ❌ 400 ／ u64 ✅ 接受 ／ UUID 字符串 ✅ 接受 |

**结论：唯一阻断就是这个 id**（同口径承诺本身成立）。

### 7.1 已修复（本次）

**修法**：`_point_id` 改为取 sha256 **前 8 字节 → u64**（`digest()[:8]`）。

选它而不是 UUID 的理由：① 改动最小且**保留 `int` 类型**（`HasIdCondition` / 点序映射
等调用点零改动，`_point_id` 的全部使用点都是内部 `list[int]`）；② u64 熵对 KB 规模
足够，且**同键碰撞由既有显式校验拦截**（`len(set(point_ids)) != len(point_ids)` →
`ValueError`），不依赖「不会撞」的假设；③ UUID 会改 id 类型（`int` → `str`），收益仅是
多 64 位熵，代价是波及面更大。

**顺带处理**：① id 变更**同时改变本地路径 id** → 既有 `path=` 持久索引需重建（本仓库
`.cache/` 下无遗留 qdrant 索引，实测无迁移负担）；② 补测试三类 ——
`tests/test_rag_qdrant_point_id.py`（**不依赖 qdrant-client，任何环境恒跑**：钉住 id 取值域
+ 无碰撞 + 稳定性。**独立成文件的原因**：其余 qdrant 测试都在顶层
`importorskip("qdrant_client")`，而 CI 只跑 `uv sync --frozen`（**不装 extra**）→ 那些文件
在 CI 上**整文件 skip**，qdrant 后端此前在 CI 上零覆盖）+ 真服务端集成
`tests/test_rag_qdrant_server.py`（建库/全量 upsert/与 local 同口径/id 接受性，
服务端不可达则 skip）。

**验证**：真 server 上 policy/case 两个 KB 全量落库成功、检索与 `local` 顶层序一致；
全量测试 **439 passed, 1 skipped**（服务端不在时 **436 passed, 4 skipped**）；
v1/v2 回归双 PASS；lint 违例数与 HEAD 持平（120 vs 120，零新增）。

**仍存在的边界**：CI 既无 Qdrant 服务端、也不装 `rag` extra → 真服务端集成测试与
`test_rag_qdrant.py` 在 CI 上均 skip（如实标注）；id 取值域由
`test_rag_qdrant_point_id.py` 在 CI 上恒跑兜底；服务端未做分布式/集群、未压测、未开鉴权。

> 结论边界：以上均为 `v1.19.0` 服务端 + `qdrant-client 1.19.0` 的实测结果；语料仍为
> 24 政策 / 67 案例的小语料，**不构成能力声明**（对齐 docs/06 §5）。

## 8. 安全

- 默认**不设 API key**（仅 `127.0.0.1` 可达，风险等价于本机进程）。
- 若需跨机访问：设 `QDRANT__SERVICE__API_KEY` 并改为非 loopback 绑定，同时客户端须
  **注入带 `api_key=` 的 client**（`factory` 的 `qdrant_client=` 参数），或自行扩展配置层。
