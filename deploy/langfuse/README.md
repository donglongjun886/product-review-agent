# Langfuse v4 自托管（本机 Docker）

在**本机用 Docker Compose 单机跑一套 Langfuse v4**，用于给 `product-review-agent` 做 LLM 链路可观测性
（trace / span / generation 落库与回看）。

- 版本：Langfuse **v4**（web + worker 双进程，存储用 Postgres + ClickHouse + Redis + MinIO）
- compose 来源：官方 `git@github.com:langfuse/langfuse.git:docker-compose.yml`（vendored，已按本机镜像源调整）
- 本目录**只放部署文件**，不改动仓库任何 Python 代码。

---

## 1. 前置条件

| 项 | 要求 |
| --- | --- |
| Docker | 已实测 **29.7.2**（Docker Desktop for Mac） |
| Docker Compose | v2（`docker compose` 子命令） |
| 磁盘 | 约 **8–10 GB**（6 个镜像 + 数据卷） |
| 内存 | 建议 ≥ 8 GB 给 Docker（ClickHouse 吃内存） |
| 网络 | 公网直连 Docker Hub / `docker.langfuse.com` / `cgr.dev` **本机不可达**，必须走国内镜像源，见第 5 节 |

**端口**（本机已确认全部空闲，故沿用官方映射；若你的机器冲突，只改 compose 里冒号**左侧**的宿主机端口）：

| 端口 | 服务 | 绑定 |
| --- | --- | --- |
| 3000 | Langfuse Web（UI + API） | 0.0.0.0 |
| 3030 | Langfuse Worker | 127.0.0.1 |
| 5432 | Postgres | 127.0.0.1 |
| 6379 | Redis | 127.0.0.1 |
| 8123 | ClickHouse HTTP | 127.0.0.1 |
| 9000 | ClickHouse native | 127.0.0.1 |
| 9090 | MinIO S3 API | 0.0.0.0 |
| 9091 | MinIO Console | 127.0.0.1 |

> 注意：`9000` 是 ClickHouse 的 native 端口（容器内），MinIO 的 S3 API 被映射到宿主机 `9090`，两者不冲突。

---

## 2. 首次使用：准备 `.env`

本目录的 `.env` **已在本机生成**（所有密钥已随机化，`.gitignore` 已忽略，不入库）——直接
`docker compose up -d` 即可。只有在**换机器 / 想重建**时才需要：

```bash
cd deploy/langfuse
cp .env.example .env
```

然后把 `.env` 里所有 `<CHANGEME_...>` 换成真随机值：

```bash
# 每个占位符执行一次，把输出粘回去
openssl rand -hex 32
```

**必须替换的 7 处**：`SALT`、`ENCRYPTION_KEY`、`NEXTAUTH_SECRET`、`POSTGRES_PASSWORD`（同时要改
`DATABASE_URL` 里的密码，两处必须一致）、`CLICKHOUSE_PASSWORD`、`REDIS_AUTH`、`MINIO_ROOT_PASSWORD`
（以及三个 `LANGFUSE_S3_*_SECRET_ACCESS_KEY`，它们必须等于 `MINIO_ROOT_PASSWORD`）。

另外 `LANGFUSE_INIT_USER_PASSWORD` 是 **UI 初始登录密码**，也要换。

两条硬规则：

1. `SALT` / `ENCRYPTION_KEY` 一旦有数据写入就**不要再改**，否则已存的 API key 等加密字段无法解密。
2. `.env` 已被 `.gitignore` 忽略，**不要提交、不要外传**。

---

## 3. 常用命令

```bash
cd deploy/langfuse

# 启动（后台）
docker compose up -d

# 看状态（等 postgres/redis/clickhouse/minio 变 healthy）
docker compose ps

# 看日志
docker compose logs -f langfuse-web
docker compose logs -f langfuse-worker
docker compose logs --tail=100 minio-init     # 建桶的一次性容器

# 健康检查（期望 200 + JSON）
curl -s http://localhost:3000/api/public/health

# 停止（保留数据）
docker compose down

# 停止并**清空全部数据**（Postgres/ClickHouse/MinIO/Redis 卷全删，慎用）
docker compose down -v
```

重启单个服务：`docker compose restart langfuse-web`。

---

## 4. UI 地址与初始账号

| 项 | 值 |
| --- | --- |
| UI | <http://localhost:3000> |
| MinIO Console | <http://localhost:9091>（账号见 `.env` 的 `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD`） |
| 登录邮箱 | `admin@local.dev`（= `.env` 的 `LANGFUSE_INIT_USER_EMAIL`） |
| 登录密码 | `.env` 的 `LANGFUSE_INIT_USER_PASSWORD` |
| 组织 / 项目 | `PRA Local` / `product-review-agent` |
| 项目公钥 `pk` | `pk-lf-pra-local` |
| 项目私钥 `sk` | `sk-lf-pra-local` |

`LANGFUSE_INIT_*` 只在**首次启动**（数据库为空时）生效，自动建好组织、项目、用户和 API 密钥，
不用手工点 UI。之后想换账号，改 `.env` 并 `docker compose down -v` 重来。

业务侧接入（`product-review-agent` 的 `.env`，**由主 agent 决定是否接线，本目录不改它**）：

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-pra-local
LANGFUSE_SECRET_KEY=sk-lf-pra-local
LANGFUSE_HOST=http://localhost:3000
```

---

## 5. 镜像拉取很慢时的应对（本机实测）

本机（macOS）**直连不可达**：`registry-1.docker.io`、`hub.docker.com`、`raw.githubusercontent.com`、
GitHub HTTPS 全部超时（只有 GitHub **SSH** 可用）；`docker.langfuse.com` 与 `cgr.dev` 也不可用
（`cgr.dev` 实测返回 `Forbidden`）。因此镜像必须走国内镜像源，且要用 `<mirror>/<namespace>/<repo>:<tag>`
的写法（Docker Hub 官方镜像补 `library/`）。

### 5.1 镜像源测速结果

测速方法：`docker pull <mirror>/library/redis:7-alpine`（58.7MB），每个源 100s 硬超时，6 源并行。

| 镜像源 | 镜像 | 体积 | 耗时 | 结果 |
| --- | --- | --- | --- | --- |
| `docker.1panel.live` | `library/redis:7-alpine` | 58.7MB | 33s | ✅ 成功 |
| `docker.1ms.run` | `library/redis:7-alpine` | 58.7MB | 34s | ✅ 成功 |
| `hub.rat.dev` | `library/redis:7-alpine` | 58.7MB | 34s | ✅ 成功 |
| `docker.m.daocloud.io` | `library/redis:7-alpine` | 58.7MB | 33s | ✅ 成功 |
| `swr.cn-north-4.myhuaweicloud.com` | `library/redis:7-alpine` | — | 2s | ❌ 失败（路径需 `ddn-k8s/docker.io/` 前缀） |
| `docker.nju.edu.cn` | `library/redis:7-alpine` | — | 1s | ❌ 失败 |
| `cgr.dev` | `chainguard/library/alpine:3.20` | — | 2s | ❌ 403 Forbidden |

**结论：4 个源都可用且速度接近（≈1.8 MB/s 解压后）**，因此把 6 个镜像**分散到 4 个源并行拉取**，
避免单源限速（实测单源拉 100MB 级镜像会长时间卡住）。

### 5.2 镜像 → 源 → 规范名 的对应关系

镜像源前缀只是为了**下载**，拉完必须 retag 成 compose 用的规范名（compose 文件因此不用改镜像地址）：

```bash
# 拉取（示例：每个镜像换一个源并行）
docker pull docker.1panel.live/langfuse/langfuse:4
docker pull docker.1ms.run/langfuse/langfuse-worker:4
docker pull hub.rat.dev/clickhouse/clickhouse-server:25.12
docker pull docker.m.daocloud.io/library/postgres:17
docker pull docker.1panel.live/library/redis:7
docker pull docker.1ms.run/minio/minio:latest

# retag 成规范名（compose 里引用的就是这些名字）
docker tag docker.1panel.live/langfuse/langfuse:4          langfuse/langfuse:4
docker tag docker.1ms.run/langfuse/langfuse-worker:4       langfuse/langfuse-worker:4
docker tag hub.rat.dev/clickhouse/clickhouse-server:25.12  clickhouse/clickhouse-server:25.12
docker tag docker.m.daocloud.io/library/postgres:17        postgres:17
docker tag docker.1panel.live/library/redis:7              redis:7
docker tag docker.1ms.run/minio/minio:latest               minio/minio:latest
```

> 共 **6 个镜像**即可。`minio-init` 建桶容器**复用 `minio/minio:latest` 本身**
> （实测该镜像内置 `/usr/bin/sh` 与 `mc RELEASE.2025-08-13`），无需再拉 `minio/mc`。

**大镜像一定要后台拉 + 看日志**，不要前台死等：

```bash
nohup docker pull docker.1panel.live/langfuse/langfuse:4 > /tmp/lf-pull-web.log 2>&1 &
# 之后：
tail -f /tmp/lf-pull-web.log
docker images
```

若某个源卡住（日志长时间不动），换另一个源重拉即可——已下载的层会被复用，不会从头开始。

---

## 6. 与官方 compose 的 4 处差异

1. **镜像地址**：`docker.langfuse.com/langfuse/langfuse:4` → `langfuse/langfuse:4`；
   `docker.io/clickhouse/clickhouse-server:25.12` → `clickhouse/clickhouse-server:25.12`；
   `docker.io/postgres:${POSTGRES_VERSION:-17}` → `postgres:17`；`docker.io/redis:7` → `redis:7`
   （`docker.io` 本就是默认 registry 前缀，去掉只为显式）。**实际下载走镜像源，retag 见 5.2。**
2. **minio**：`cgr.dev/chainguard/minio`（本机 403）→ `minio/minio:latest`。
   - chainguard 的 `entrypoint: sh` + `command: -c 'mkdir -p /data/langfuse && minio server ...'` 是它专用写法，
     已换成 minio 原生 `command: server --address ":9000" --console-address ":9001" /data`；
   - 健康检查 `mc ready local`（chainguard 镜像内有 mc）→ `curl -f http://localhost:9000/minio/health/live`；
   - 建桶不再靠 `mkdir`，改为一次性容器 **`minio-init`** 执行
     `mc alias set` + `mc mb --ignore-existing local/langfuse`，`langfuse-web` / `langfuse-worker`
     通过 `depends_on: minio-init: condition: service_completed_successfully` 等它建完桶再启动。
     `minio-init` 直接复用 `minio/minio:latest`（内置 `mc` 与 `/bin/sh`），不需要额外镜像。
3. **凭据**：官方所有 `# CHANGEME` 值改由 `.env` 注入。
4. **`LANGFUSE_INIT_*`**：填上了组织 / 项目 / 用户 / 密钥，首次启动自动初始化。

---

## 7. ⚠️ v4 写入模式：默认 `events_only`（实测踩坑，必读）

Langfuse v4 默认 `LANGFUSE_MIGRATION_V4_WRITE_MODE=events_only`，**旧的 v3 ingestion / 读取 API 全部不可用**：

| 调用 | 结果（本机实测） |
| --- | --- |
| `POST /api/public/ingestion`（`trace-create`） | **400** `Event type "trace-create" is not accepted ... when LANGFUSE_MIGRATION_V4_WRITE_MODE is events_only` |
| `GET /api/public/traces` | **不可用** `This endpoint is not available on deployments running in Langfuse v4 events_only mode` |
| `POST /api/public/otel/v1/traces`（OTLP/HTTP） | ✅ **200**，写入 MinIO `events/otel/...` → worker → ClickHouse `events_full` |

**结论：接 `product-review-agent` 必须走 v4 路径**，即 Langfuse Python SDK v3+（默认走 OTLP）或
OpenTelemetry OTLP exporter，不要用旧的 `langfuse.trace()` 直传 ingestion 的写法。

若暂时只能用旧 SDK，可在 `.env` 加一行作为过渡（web 与 worker 都要生效）：

```bash
LANGFUSE_MIGRATION_V4_WRITE_MODE=dual
```

然后 `docker compose up -d` 重建 web/worker。`dual` 会同时保留 v3 与 v4 写入路径，代价是数据双写。

**本机已实测的接入参数**：

```bash
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-pra-local
LANGFUSE_SECRET_KEY=sk-lf-pra-local
# OTLP endpoint（SDK 自动使用；手写 exporter 时用）
# http://localhost:3000/api/public/otel
```

---

## 8. 排障

| 现象 | 处理 |
| --- | --- |
| `docker compose up` 报镜像找不到 | 先按 5.2 retag，`docker images` 确认 6 个规范名都在 |
| web 一直起不来 / 重启 | `docker compose logs --tail=200 langfuse-web`；多半是 Postgres 或 ClickHouse 还没 healthy |
| `minio-init` 失败 | `docker compose logs minio-init`；确认 `MINIO_ROOT_PASSWORD` 与三个 `LANGFUSE_S3_*_SECRET_ACCESS_KEY` 一致 |
| 端口被占用 | `lsof -nP -iTCP -sTCP:LISTEN \| grep -E ':(3000\|9090)'`，改 compose 冒号左侧端口 |
| 想彻底重来 | `docker compose down -v`（数据全清，`LANGFUSE_INIT_*` 会重新生效） |
| 镜像拉取卡住 | 换镜像源重拉（见 5.2），已下载的层会复用 |
