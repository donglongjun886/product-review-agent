"""langfuse_smoke.py —— Langfuse 接线端到端自检：发一条合成 trace，再用公开 API 回读断言。

用途（把 docs/09 §13 那条「端到端发一条 trace」的占位验收命令落成可执行脚本）：

1. 用 ``pra.observability.tracing.get_tracer()`` 取当前生效 tracer —— 与业务**同一入口**，
   不绕过适配层、不直接 import SDK；
2. 在**一条**合成 trace 里覆盖四类观测（root / node / llm generation / tool），
   并用 ``asyncio.create_task`` 跑一个子 span（验证 OTel contextvar 传播，
   docs/09 §6 实测结论第 4 条 —— LangGraph 节点在独立 task 里跑，这点必须成立）；
3. ``flush`` 后用**标准库 urllib**（Basic Auth = ``base64(public_key:secret_key)``）轮询
   ``GET {host}/api/public/traces/{trace_id}``（ingestion 有延迟），逐条断言并打印缩进的
   observation 树（按 ``parentObservationId`` 组装，root 无 parent）。

**退出码约定（CI 友好）**：

| 情形 | 行为 | 退出码 |
|---|---|---|
| 无凭据 / ``PRA_LANGFUSE_ENABLED=0`` / SDK 未装（NullTracer） | 打印 ``tracing disabled (NullTracer: <reason>)`` + 启用提示 | 0 |
| 已启用 + ``--no-verify`` | 发 trace + flush，跳过回读 | 0 |
| 已启用 + 回读断言全通过 | 打印 ``SMOKE PASS`` + UI 链接 | 0 |
| 回读不到 / 任一断言失败 | 打印具体差异 | 1 |
| 参数非法（``--trace-id`` 非 32-hex） | 打印用法 | 2 |

**注意**：本脚本只读环境变量与 HTTP GET，**不安装任何依赖、不写任何文件**；
``langfuse`` SDK 属 optional extra，未安装时同样走 NullTracer 分支并退出码 0。

用法::

    python scripts/langfuse_smoke.py                       # 自动生成 trace_id
    python scripts/langfuse_smoke.py --trace-id <HEX32>    # 指定 trace_id（可复跑同一条）
    python scripts/langfuse_smoke.py --host http://localhost:3000 --timeout 60
    python scripts/langfuse_smoke.py --no-verify           # 只发不读（服务端未起时看埋点）
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any
from uuid import uuid4

from pra.observability.tracing import TraceContext, Tracer, get_tracer

__all__ = ["main"]

#: Langfuse 服务地址缺省值（与 tracing.make_tracer 一致）。
DEFAULT_HOST = "http://localhost:3000"
#: 回读轮询总时长缺省（秒）。
DEFAULT_TIMEOUT = 30.0
#: 轮询间隔（秒）。
POLL_INTERVAL_S = 5.0
#: 轮询最大次数（即使 --timeout 很大也不无限等）。
POLL_MAX_ATTEMPTS = 12

#: 合成 trace 的固定关联字段（断言用，脚本与测试共享）。
TRACE_NAME = "smoke"
SESSION_ID = "langfuse-smoke"
SOURCE_TAG = "source:smoke"
CASE_ID = "SMOKE_001"
#: 合成 generation 的 token 用量 —— **故意用非零值**：既验证 usage 真的过管道落库
#: （全 0 可能被服务端当作"无用量"丢弃），也便于回读时做逐键比对。这是**管道自检
#: 的合成数据**，与业务口径无关（真实 Agent 走 scripted 桩时 token 恒 0，docs/09 §7）。
EXPECTED_USAGE = {"input": 12, "output": 34, "total": 46}
#: 异步嵌套验证用的子 span 名（docs/09 §6 实测第 4 条）。
ASYNC_CHILD_SPAN = "plan.async_child"

#: 断言目标：必须存在的 observation 名（root / node / generation / tool）。
REQUIRED_SPANS = ("smoke", "hypothesize", "plan", "tools")
REQUIRED_GENERATIONS = ("llm.hypothesize",)
REQUIRED_TOOLS = ("ProductTool",)

#: W3C trace id 形状（32 位小写 hex）。
_HEX32 = re.compile(r"\A[0-9a-f]{32}\Z")

#: 每个 HTTP 请求的单次超时上限（秒）—— 总时长由 --timeout 控制。
_HTTP_TIMEOUT_S = 10.0


# --------------------------------------------------------------------------------------
# 纯函数：URL / 认证头（测试直接断言，不联网）
# --------------------------------------------------------------------------------------


def _trace_api_url(host: str, trace_id: str) -> str:
    """回读用公开 API URL（``{host}/api/public/traces/{trace_id}``，尾斜杠归一）。"""
    return f"{host.rstrip('/')}/api/public/traces/{trace_id}"


def _ui_url(host: str, trace_id: str) -> str:
    """Langfuse UI 中该 trace 的链接。"""
    return f"{host.rstrip('/')}/trace/{trace_id}"


def _auth_header(public_key: str, secret_key: str) -> str:
    """Basic Auth 头（**手写 base64**，不引入额外依赖）：``Basic base64(pk:sk)``。"""
    token = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode("ascii")
    return f"Basic {token}"


def _json_or_none(body: str) -> dict[str, Any] | None:
    """解析响应体；非 JSON 对象 → ``None``（观测旁路，绝不抛）。"""
    try:
        payload = json.loads(body)
    except (ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _api_get(
    host: str,
    trace_id: str,
    *,
    public_key: str,
    secret_key: str,
    timeout: float = _HTTP_TIMEOUT_S,
) -> tuple[int | None, dict[str, Any] | None, str | None]:
    """单次 ``GET {host}/api/public/traces/{trace_id}``。

    :return: ``(status, payload, error)`` —— 成功 ``(200, {...}, None)``；
        404 等 HTTP 错误 ``(code, None, "HTTP ...")``；网络错误 ``(None, None, "TypeError: ...")``。
        **绝不抛异常**（未就绪/服务端未起是预期状态，由轮询处理）。
    """
    url = _trace_api_url(host, trace_id)
    request = urllib.request.Request(
        url, headers={"Authorization": _auth_header(public_key, secret_key)}, method="GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            status = int(getattr(response, "status", 200))
            return status, _json_or_none(body), None
    except urllib.error.HTTPError as exc:
        return int(exc.code), None, f"HTTP {exc.code} {exc.reason}"
    except Exception as exc:  # noqa: BLE001 - 服务端未起/网络错误都属预期
        return None, None, f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------------------
# 发合成 trace（覆盖四类观测 + 异步嵌套）
# --------------------------------------------------------------------------------------


async def _emit_synthetic_trace(tracer: Tracer, trace_id: str) -> None:
    """在一条 trace 里发四类观测 + 一个 ``create_task`` 子 span，最后 flush。

    trace 级属性（session_id / metadata / tags / version）由 ``TraceContext`` 下发；
    整体 input/output 放 **root observation**（v4 中 trace 级 input/output 已废弃）。
    """
    ctx = TraceContext(
        trace_id=trace_id,
        name=TRACE_NAME,
        session_id=SESSION_ID,
        version="smoke",
        metadata={"case_id": CASE_ID, "source": "smoke", "experiment": "smoke"},
        tags=["env:local", SOURCE_TAG],
        input={"case_id": CASE_ID},
    )

    async def _async_child() -> None:
        """``asyncio.create_task`` 里开的子 span —— 验证 contextvar 传播（不靠 await 链）。"""
        with tracer.node_span(ASYNC_CHILD_SPAN) as child:
            child.update(
                output={"async": True, "note": "contextvar propagation via create_task"},
                metadata={"latency_ms": 1},
            )
            await asyncio.sleep(0)

    with tracer.trace_root(ctx) as root:
        # 1) node span：hypothesize → 内含一次 LLM generation
        with tracer.node_span("hypothesize"), tracer.llm_generation(
            name="llm.hypothesize",
            model="scripted",
            input=[{"role": "user", "content": f"{CASE_ID}: 该商品是否需要人工复核"}],
            model_parameters={"temperature": 0.0},
        ) as generation:
            generation.update(
                output="{}",
                usage_details=dict(EXPECTED_USAGE),
                metadata={"latency_ms": 1},
            )

        # 2) node span：tools → 内含一次 tool span
        with tracer.node_span("tools"), tracer.tool_span(
            name="ProductTool", input={"args": {"product_id": "P_88231"}}
        ) as tool:
            tool.update(
                output={"evidence": ["PRODUCT_FACT P_88231"], "status": "ok"},
                metadata={"latency_ms": 1},
            )

        # 3) node span：plan → 内含 create_task 起的子 span（异步嵌套验证）
        with tracer.node_span("plan"):
            await asyncio.create_task(_async_child())

        # root observation 的整体 output（终态决策摘要）
        root.update(output={"decision": "HUMAN_REVIEW"})

    try:
        tracer.flush()
    except Exception:  # noqa: BLE001 - 观测旁路：flush 失败不得让自检脚本崩
        print("WARN: tracer.flush() raised (ignored)")


# --------------------------------------------------------------------------------------
# 回读 + 断言
# --------------------------------------------------------------------------------------


def _fetch_trace(
    host: str,
    trace_id: str,
    *,
    public_key: str,
    secret_key: str,
    timeout: float,
) -> tuple[dict[str, Any] | None, list[str]]:
    """轮询回读 trace：最多 ``POLL_MAX_ATTEMPTS`` 次、间隔 ``POLL_INTERVAL_S`` 秒，
    总时长受 ``timeout`` 约束。

    :return: ``(payload, notes)`` —— 拿到即返回；超时返回 ``(None, 每次尝试的说明)``。
    """
    attempts = min(POLL_MAX_ATTEMPTS, max(1, int(timeout // POLL_INTERVAL_S)))
    deadline = time.monotonic() + timeout
    notes: list[str] = []
    for attempt in range(1, attempts + 1):
        remaining = deadline - time.monotonic()
        per_request = max(1.0, min(remaining, _HTTP_TIMEOUT_S)) if remaining > 0 else 1.0
        status, payload, error = _api_get(
            host,
            trace_id,
            public_key=public_key,
            secret_key=secret_key,
            timeout=per_request,
        )
        if payload is not None:
            print(f"  read back on attempt {attempt}/{attempts} (status={status})")
            return payload, notes
        note = f"attempt {attempt}/{attempts}: status={status} error={error}"
        notes.append(note)
        print(f"  not visible yet — {note}")
        if attempt < attempts:
            sleep_for = min(POLL_INTERVAL_S, max(0.0, deadline - time.monotonic()))
            if sleep_for > 0:
                time.sleep(sleep_for)
    return None, notes


def _observation_tree(observations: list[dict[str, Any]]) -> list[str]:
    """按 ``parentObservationId`` 组装缩进树（root 无 parent；父缺失按 root 处理）。

    :return: 每行一条 observation 的字符串（深度即缩进层级），供直接打印。
    """
    known_ids = {str(obs.get("id")) for obs in observations if obs.get("id")}
    children: dict[str | None, list[dict[str, Any]]] = {}
    for obs in observations:
        parent = obs.get("parentObservationId")
        parent_key = str(parent) if parent and str(parent) in known_ids else None
        children.setdefault(parent_key, []).append(obs)

    lines: list[str] = []

    def _walk(parent: str | None, depth: int) -> None:
        if depth > 12:  # 防意外环（观测数据来自外部服务，不做信任假设）
            return
        for obs in sorted(
            children.get(parent, []),
            key=lambda o: (str(o.get("startTime") or ""), str(o.get("name") or "")),
        ):
            bits = [f"{obs.get('name') or '<unnamed>'} [{obs.get('type') or '?'}]"]
            if obs.get("model"):
                bits.append(f"model={obs['model']}")
            usage = obs.get("usageDetails") or obs.get("usage")
            if usage:
                bits.append(f"usage={json.dumps(usage, ensure_ascii=False, sort_keys=True)}")
            level = str(obs.get("level") or "").upper()
            if level and level not in {"DEFAULT", "INFO"}:
                bits.append(f"level={obs['level']}")
            bits.append(f"id={obs.get('id')}")
            lines.append("  " * depth + "- " + "  ".join(bits))
            _walk(str(obs.get("id")), depth + 1)

    _walk(None, 0)
    return lines


def _verify_trace(trace: Any, trace_id: str) -> list[str]:
    """回读断言清单（对应脚本要求的 1~4 条）；返回失败说明列表（空 = 全通过）。

    1. trace 存在且 ``id == trace_id``；
    2. ``sessionId == "langfuse-smoke"`` 且 ``tags`` 含 ``source:smoke``；
    3. observations 含 ``smoke`` / ``hypothesize`` / ``plan`` / ``tools`` /
       ``llm.hypothesize``(GENERATION) / ``ProductTool``(TOOL)；
    4. ``llm.hypothesize`` 的 ``usageDetails`` 非空、``model`` 非空。
    """
    problems: list[str] = []
    if not isinstance(trace, dict):
        return [f"trace payload is not a JSON object: {type(trace).__name__}"]

    # 1) trace 存在（能走到这里即回读成功）且 id 一致
    if trace.get("id") != trace_id:
        problems.append(f"trace id mismatch: got {trace.get('id')!r}, want {trace_id!r}")

    # 2) 会话与标签
    if trace.get("sessionId") != SESSION_ID:
        problems.append(
            f"sessionId mismatch: got {trace.get('sessionId')!r}, want {SESSION_ID!r}"
        )
    tags = trace.get("tags")
    tags_list = tags if isinstance(tags, list) else []
    if SOURCE_TAG not in tags_list:
        problems.append(f"tags missing {SOURCE_TAG!r}: got {tags!r}")

    # 3) observation 名与类型
    observations = trace.get("observations")
    if not isinstance(observations, list) or not observations:
        problems.append(f"observations empty/missing: {observations!r}")
        return problems
    by_name: dict[str, list[dict[str, Any]]] = {}
    for obs in observations:
        if isinstance(obs, dict):
            by_name.setdefault(str(obs.get("name")), []).append(obs)
    for name in (*REQUIRED_SPANS, *REQUIRED_GENERATIONS, *REQUIRED_TOOLS):
        if name not in by_name:
            problems.append(
                f"missing observation {name!r} (present: {sorted(by_name)})"
            )
    for name in REQUIRED_GENERATIONS:
        for obs in by_name.get(name, []):
            obs_type = str(obs.get("type") or "").upper()
            if obs_type != "GENERATION":
                problems.append(f"{name!r} type is {obs.get('type')!r}, want 'GENERATION'")

    # 4) generation 的 usage / model（usage 逐键比对 EXPECTED_USAGE，验证真的过管道）
    for name in REQUIRED_GENERATIONS:
        for obs in by_name.get(name, []):
            usage = obs.get("usageDetails") or obs.get("usage")
            if not usage:
                problems.append(f"{name!r} usageDetails is empty: {usage!r}")
            elif isinstance(usage, dict):
                for key, want in EXPECTED_USAGE.items():
                    got = usage.get(key)
                    if got != want:
                        problems.append(
                            f"{name!r} usageDetails[{key!r}] = {got!r}, want {want!r} "
                            f"(full: {usage!r})"
                        )
            if not str(obs.get("model") or "").strip():
                problems.append(f"{name!r} model is empty: {obs.get('model')!r}")
    for name in REQUIRED_TOOLS:
        for obs in by_name.get(name, []):
            obs_type = str(obs.get("type") or "").upper()
            if obs_type != "TOOL":
                problems.append(f"{name!r} type is {obs.get('type')!r}, want 'TOOL'")

    return problems


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _print_enable_hint(host: str) -> None:
    """打印「如何启用」提示（无凭据 / SDK 未装时都要打）。"""
    print("how to enable Langfuse tracing:")
    print("  uv sync --extra observability      # 安装 langfuse SDK（optional extra）")
    print("  export LANGFUSE_PUBLIC_KEY=pk-lf-...")
    print("  export LANGFUSE_SECRET_KEY=sk-lf-...")
    print(f"  export LANGFUSE_HOST={host}")
    print("  uv run --extra observability python scripts/langfuse_smoke.py")
    print("  # 服务端：cd deploy/langfuse && docker compose up -d（docs/09 §9）")


def _build_parser() -> argparse.ArgumentParser:
    """命令行解析器（``--trace-id`` / ``--host`` / ``--timeout`` / ``--no-verify``）。"""
    parser = argparse.ArgumentParser(
        prog="langfuse_smoke.py",
        description=(
            "Langfuse 接线端到端自检：发一条合成 trace（四类观测 + 异步嵌套），"
            "flush 后经公开 API 回读并断言。无凭据 / SDK 未装时打印提示并以 0 退出。"
        ),
    )
    parser.add_argument(
        "--trace-id",
        default=None,
        help="32-hex W3C trace id（缺省 uuid4().hex；可与业务 run_id 对齐）",
    )
    parser.add_argument(
        "--host",
        default=None,
        help=f"Langfuse 服务地址（缺省 $LANGFUSE_HOST，再缺省 {DEFAULT_HOST}）",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=(
            f"回读轮询总时长秒（缺省 {DEFAULT_TIMEOUT:g}；"
            f"最多 {POLL_MAX_ATTEMPTS} 次、间隔 {POLL_INTERVAL_S:g}s）"
        ),
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="只发 trace + flush，不做回读断言（服务端未起时看埋点用）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """脚本入口：返回退出码（0 通过/跳过、1 断言失败、2 参数非法）。"""
    args = _build_parser().parse_args(argv)

    trace_id = (args.trace_id or uuid4().hex).strip().lower()
    if not _HEX32.match(trace_id):
        print(f"invalid --trace-id {args.trace_id!r}: expect 32 lowercase hex chars (W3C trace id)")
        return 2
    host = (args.host or os.environ.get("LANGFUSE_HOST") or DEFAULT_HOST).strip()

    tracer = get_tracer()
    if not getattr(tracer, "enabled", False):
        reason = getattr(tracer, "reason", "unknown")
        print(f"tracing disabled (NullTracer: {reason})")
        _print_enable_hint(host)
        return 0

    print(f"tracer enabled → emitting synthetic trace (trace_id={trace_id})")
    asyncio.run(_emit_synthetic_trace(tracer, trace_id))
    print("trace emitted + flushed")
    print(f"ui: {_ui_url(host, trace_id)}")

    if args.no_verify:
        print("--no-verify: skipped read-back verification")
        return 0

    public_key = (os.environ.get("LANGFUSE_PUBLIC_KEY") or "").strip()
    secret_key = (os.environ.get("LANGFUSE_SECRET_KEY") or "").strip()
    if not public_key or not secret_key:
        print("read-back skipped: missing LANGFUSE_PUBLIC_KEY/SECRET_KEY")
        return 0

    print(f"reading back {_trace_api_url(host, trace_id)} (timeout={args.timeout:g}s) …")
    trace, notes = _fetch_trace(
        host,
        trace_id,
        public_key=public_key,
        secret_key=secret_key,
        timeout=args.timeout,
    )
    if trace is None:
        print(f"SMOKE FAIL: trace {trace_id} not retrievable from {host}")
        for note in notes:
            print(f"  {note}")
        print(f"  hint: 服务端是否已起？curl -s {host.rstrip('/')}/api/public/health")
        return 1

    raw_observations = trace.get("observations")
    raw_observations = raw_observations if isinstance(raw_observations, list) else []
    # 外部服务返回的 payload 不做信任假设：只保留 dict 元素（非 dict 由断言清单报告）
    observations = [obs for obs in raw_observations if isinstance(obs, dict)]
    print(f"observations: {len(observations)}")
    print("observation tree:")
    tree = _observation_tree(observations)
    for line in tree or ["<empty>"]:
        print(f"  {line}")
    if not any(str(obs.get("name")) == ASYNC_CHILD_SPAN for obs in observations):
        print(
            f"NOTE: async child span {ASYNC_CHILD_SPAN!r} missing — "
            "contextvar 传播可能不成立（docs/09 §6 实测第 4 条）"
        )

    problems = _verify_trace(trace, trace_id)
    if problems:
        print(f"SMOKE FAIL: {len(problems)} assertion(s) failed")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("SMOKE PASS")
    print(f"ui: {_ui_url(host, trace_id)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
