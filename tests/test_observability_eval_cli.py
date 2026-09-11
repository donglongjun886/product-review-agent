"""评测侧 Langfuse 接线 + 演示脚本的离线用例（NullTracer 路径，无网络、无 SDK 依赖）。

1. ``run_evaluation.py --experiment prompt-v2`` 跑前把 ``PRA_LANGFUSE_EXPERIMENT`` 设成该值；
2. 不传 ``--experiment`` → 取环境变量，再缺省 ``baseline``；
3. ``flush_tracer()`` 在评测收尾恰好调用一次（成功与异常路径都只一次）；
4. 无凭据时 ``--langfuse`` 不报错、退出码 0（观测失败绝不中断评测）；
5. ``demo_langfuse_trace.py`` 无凭据路径跑完且退出码 0，stdout 含 ``tracing disabled``；
6. 加 ``--experiment`` 后报告与不加时逐字节一致（不改评测指标）。

数据集：``eval_data/v1`` 前 3 条写临时 JSONL。
"""

from __future__ import annotations

import io
import os
import runpy
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import pytest

from pra.observability import tracing as T

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_EVAL_SCRIPT = REPO_ROOT / "scripts" / "run_evaluation.py"
DEMO_SCRIPT = REPO_ROOT / "scripts" / "demo_langfuse_trace.py"
V1_DATA = REPO_ROOT / "eval_data" / "v1" / "cases_v1.jsonl"
#: 冒烟子集条数：v1 35 案的 scheme 分布在 3 条内已有 PASS/REJECT 混合。
SMOKE_CASES = 3

#: 观测相关环境变量 —— 用例前后一律清空，保证"无凭据"路径与顺序无关。
_OBS_ENV_KEYS = (
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "PRA_LANGFUSE_ENABLED",
    "PRA_LANGFUSE_SAMPLE",
    "PRA_LANGFUSE_EXPERIMENT",
    "PRA_LANGFUSE_SESSION",
)


@pytest.fixture(autouse=True)
def _restore_process_env():
    """整份还原 ``os.environ``：被测脚本直接写 EXPERIMENT / SESSION，不还原会污染后续用例。"""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture(autouse=True)
def _offline_observability(monkeypatch):
    """无凭据环境：清观测变量 + 屏蔽 .env 凭据 + 注入 NullTracer；``.env`` 也会被读，不屏蔽
    本地真实凭据会干扰「无凭据路径」（甚至联网）。"""
    real_env_or_settings = T._env_or_settings
    #: 只屏蔽"是否启用/凭据"四键；EXPERIMENT / SESSION 仍走真实实现（用例要读写它们）。
    blocked = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "PRA_LANGFUSE_ENABLED",
               "PRA_LANGFUSE_SAMPLE")

    def _no_langfuse(env_key: str, settings_attr: str) -> str | None:
        if env_key in blocked:
            return None
        return real_env_or_settings(env_key, settings_attr)

    for key in _OBS_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(T, "_env_or_settings", _no_langfuse)
    T.set_tracer(T.NullTracer("test-offline"))
    try:
        yield
    finally:
        T.set_tracer(None)


def _clean_subprocess_env() -> dict[str, str]:
    # PRA_LANGFUSE_ENABLED=0 保证即便仓库根 .env 配了真实凭据也走 NullTracer（不联网、不 import SDK）
    env = {k: v for k, v in os.environ.items() if k not in _OBS_ENV_KEYS}
    env["PRA_LANGFUSE_ENABLED"] = "0"
    return env


@pytest.fixture(scope="module")
def smoke_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """最小评测集（v1 前 3 条）：确定性、不触网。"""
    lines = V1_DATA.read_text(encoding="utf-8").splitlines()[:SMOKE_CASES]
    assert len(lines) == SMOKE_CASES, "eval_data/v1 行数不足，无法构造最小评测集"
    path = tmp_path_factory.mktemp("s5b") / "cases_smoke.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _load_run_evaluation() -> Any:
    return runpy.run_path(str(RUN_EVAL_SCRIPT), run_name="_s5b_run_evaluation")


def _run_eval(argv: list[str]) -> tuple[int, str]:
    """in-process 跑评测 CLI，返回 (退出码, stdout)。"""
    module = _load_run_evaluation()
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = module["main"](argv)
    return code, buf.getvalue()


# 1. --experiment 写进 PRA_LANGFUSE_EXPERIMENT（AgentScheme 造 trace_id 的读点）


def test_experiment_flag_sets_env(smoke_data: Path) -> None:
    code, out = _run_eval(
        ["--data", str(smoke_data), "--smoke", "--smoke-limit", str(SMOKE_CASES),
         "--experiment", "prompt-v2"]
    )

    assert code == 0
    assert os.environ["PRA_LANGFUSE_EXPERIMENT"] == "prompt-v2"
    assert "[OK] 跑分完成" in out


def test_experiment_name_reaches_trace_context(smoke_data: Path, monkeypatch) -> None:
    """``--experiment prompt-v2`` 跑完后 ``os.environ`` 与 ``experiment_name()`` 都是它。"""
    monkeypatch.setenv("PRA_LANGFUSE_ENABLED", "0")  # 即便有凭据也不联网
    code, _ = _run_eval(
        ["--data", str(smoke_data), "--smoke", "--smoke-limit", str(SMOKE_CASES),
         "--experiment", "prompt-v2"]
    )

    assert code == 0
    assert T.experiment_name() == "prompt-v2"


# 2. 不传 --experiment → 环境变量优先，未设则 baseline


def test_no_experiment_flag_uses_env_then_baseline(smoke_data: Path, monkeypatch) -> None:
    monkeypatch.setenv("PRA_LANGFUSE_EXPERIMENT", "prompt-v9")
    code, _ = _run_eval(["--data", str(smoke_data), "--smoke", "--smoke-limit", str(SMOKE_CASES)])
    assert code == 0
    assert os.environ["PRA_LANGFUSE_EXPERIMENT"] == "prompt-v9"  # 未被覆盖

    monkeypatch.delenv("PRA_LANGFUSE_EXPERIMENT", raising=False)
    code, _ = _run_eval(["--data", str(smoke_data), "--smoke", "--smoke-limit", str(SMOKE_CASES)])
    assert code == 0
    assert os.environ["PRA_LANGFUSE_EXPERIMENT"] == "baseline"
    assert T.experiment_name() == "baseline"


# 3. flush_tracer 在收尾恰好调用一次（成功路径 + 异常路径）


def test_flush_tracer_called_exactly_once(smoke_data: Path, monkeypatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(T, "flush_tracer", lambda: calls.append(1))

    code, _ = _run_eval(["--data", str(smoke_data), "--smoke", "--smoke-limit", str(SMOKE_CASES)])

    assert code == 0
    assert len(calls) == 1


def test_flush_tracer_called_once_even_on_failure(monkeypatch) -> None:
    # 数据集缺失 → ValueError，但 flush 仍必须恰好一次
    calls: list[int] = []
    monkeypatch.setattr(T, "flush_tracer", lambda: calls.append(1))

    with pytest.raises(ValueError, match="评测集文件不存在"):
        _run_eval(["--data", "eval_data/__no_such_file__.jsonl"])

    assert len(calls) == 1


# 4. 无凭据时 --langfuse 不报错、退出码 0


def test_langfuse_flag_without_credentials_runs_clean(smoke_data: Path) -> None:
    code, out = _run_eval(
        ["--data", str(smoke_data), "--smoke", "--smoke-limit", str(SMOKE_CASES), "--langfuse"]
    )

    assert code == 0
    assert "tracing disabled" in out
    assert "LANGFUSE_PUBLIC_KEY" in out
    assert "[OK] 跑分完成" in out


def test_default_path_prints_no_langfuse_noise(smoke_data: Path) -> None:
    """不传 ``--langfuse`` 且无凭据 → stdout 不出现任何观测噪音。"""
    _code, out = _run_eval(
        ["--data", str(smoke_data), "--smoke", "--smoke-limit", str(SMOKE_CASES)]
    )

    assert "langfuse" not in out.lower()


# 5. demo_langfuse_trace.py 无凭据路径：跑完、退出码 0、提示 tracing disabled


def test_demo_script_without_credentials_exits_zero() -> None:
    proc = subprocess.run(
        [sys.executable, str(DEMO_SCRIPT)],
        capture_output=True,
        text=True,
        env=_clean_subprocess_env(),
        cwd=str(REPO_ROOT),
        timeout=300,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "tracing disabled" in proc.stdout
    assert "decision:" in proc.stdout  # 业务照常：仍然产出裁决
    assert "trace_id:" in proc.stdout
    assert "/project/pra-local/traces/" in proc.stdout  # v4 UI 路由


def test_demo_script_accepts_explicit_run_id() -> None:
    """``--run-id`` 可复现同一条 trace（32-hex 时 trace_id 原样等于 run_id）。"""

    run_id = "0123456789abcdef0123456789abcdef"
    proc = subprocess.run(
        [sys.executable, str(DEMO_SCRIPT), "--run-id", run_id],
        capture_output=True,
        text=True,
        env=_clean_subprocess_env(),
        cwd=str(REPO_ROOT),
        timeout=300,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert f"run_id:              {run_id}" in proc.stdout
    assert f"trace_id:            {run_id}" in proc.stdout


# 6. 不改评测指标：加观测参数后报告输出逐字节一致


def test_observability_params_do_not_change_metrics(smoke_data: Path) -> None:
    """同一最小评测集：裸跑 vs 加 ``--experiment``/``--tag`` → 报告逐字节一致（新参数自己的
    打印行对比时截掉）。"""
    base_argv = ["--data", str(smoke_data), "--smoke", "--smoke-limit", str(SMOKE_CASES)]
    code_base, out_base = _run_eval(base_argv)
    code_obs, out_obs = _run_eval(
        base_argv + ["--experiment", "prompt-v2", "--tag", "round:1", "--tag", "llm:scripted"]
    )

    assert code_base == 0 and code_obs == 0
    # 报告正文（截掉新参数自己的打印行）逐字节一致
    assert out_obs.split("[tag]")[0] == out_base
    assert "[tag] 额外标签: round:1, llm:scripted" in out_obs
    assert "prompt-v2" not in out_base
    # 指标口径未被观测参数改写：汇总/分层/审计三块表头都在且一致
    for marker in ("总体指标", "按 scene 分层", "决策分布审计", "[normal]"):
        assert marker in out_base and marker in out_obs


def test_tag_is_printed_only_when_given(smoke_data: Path) -> None:
    """``--tag`` 只影响最终报告行（不进 EvalContext / 不改 metrics）。"""
    base_argv = ["--data", str(smoke_data), "--smoke", "--smoke-limit", str(SMOKE_CASES)]
    _code, out_base = _run_eval(base_argv)
    _code2, out_tag = _run_eval(base_argv + ["--tag", "round:1", "--tag", "llm:scripted"])

    assert "[tag] 额外标签: round:1, llm:scripted" in out_tag
    assert "[tag]" not in out_base
