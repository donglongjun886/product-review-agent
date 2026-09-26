"""评测 CLI 的思考配置入口：``--thinking`` / ``--reasoning-effort`` 必须真的落到后端。"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_evaluation.py"


def _load_script():
    """按路径加载跑分脚本（它不是包）。"""
    spec = importlib.util.spec_from_file_location("run_evaluation_mod", SCRIPT)
    assert spec and spec.loader, f"无法定位脚本: {SCRIPT}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _backend(mod, **cfg):
    return mod._make_real_backend(
        model=mod.DEFAULT_MODEL, api_key="sk-test", base_url=None, tools=[], **cfg
    )


def test_llm_config_defaults_are_not_sent() -> None:
    """默认：CLI 与后端两侧都为 None（不下发，用网关默认）。"""
    mod = _load_script()
    args = mod._parse_args([])
    assert args.thinking is None
    assert args.reasoning_effort is None

    backend = _backend(mod, thinking=args.thinking, reasoning_effort=args.reasoning_effort)
    assert backend.thinking is None
    assert backend.reasoning_effort is None


def test_llm_config_threads_cli_to_backend() -> None:
    """显式档位：CLI 解析值原样进后端属性（含 disabled 与非默认 effort）。"""
    mod = _load_script()
    args = mod._parse_args(["--thinking", "disabled", "--reasoning-effort", "low"])
    assert (args.thinking, args.reasoning_effort) == ("disabled", "low")

    backend = _backend(mod, thinking=args.thinking, reasoning_effort=args.reasoning_effort)
    assert backend.thinking == "disabled"
    assert backend.reasoning_effort == "low"


def test_llm_config_rejects_out_of_contract_values() -> None:
    """白名单外取值直接报错：只暴露官方规范档（none/low/high/max + enabled/disabled）。"""
    mod = _load_script()
    with pytest.raises(SystemExit):
        mod._parse_args(["--thinking", "off"])
    with pytest.raises(SystemExit):
        mod._parse_args(["--reasoning-effort", "medium"])
