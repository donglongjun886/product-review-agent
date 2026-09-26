"""默认路径「无凭据 = 不碰 Langfuse」契约测试 —— CI 上恒跑。"""

from __future__ import annotations

import json
import os
import subprocess
import sys

#: 需要从子进程 env 中剔除的键前缀。
_CREDENTIAL_PREFIXES = ("LANGFUSE_", "PRA_LANGFUSE_")

#: 子进程回传证据行的前缀。
_PAYLOAD_MARKER = "PRA_TRACE_DEFAULT_PATH_GUARD_JSON:"

#: 子进程脚本：调 ``trace_callbacks()`` 后回传「结果 + 进程里有没有 langfuse」。
_CHILD_SCRIPT_TEMPLATE = """
import json, sys

from pra.wiring import trace_callbacks


def main() -> dict:
    callbacks = trace_callbacks()
    return {
        "callbacks_empty": callbacks == [],
        "callbacks_len": len(callbacks),
        "sdk_imported": "langfuse" in sys.modules,
        "wiring_loaded": "pra.wiring" in sys.modules,
        "settings_loaded": "pra.infra.db" in sys.modules,
    }


print(__PAYLOAD_MARKER__ + json.dumps(main(), ensure_ascii=False))
"""

_CHILD_SCRIPT = _CHILD_SCRIPT_TEMPLATE.replace("__PAYLOAD_MARKER__", repr(_PAYLOAD_MARKER))


def _run_default_path_child() -> dict:
    child_env = {
        k: v for k, v in os.environ.items()
        if not k.startswith(_CREDENTIAL_PREFIXES)
    }
    child_env["LANGFUSE_PUBLIC_KEY"] = ""
    child_env["LANGFUSE_SECRET_KEY"] = ""
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_SCRIPT],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
        env=child_env,
    )
    assert proc.returncode == 0, (
        f"默认路径子进程失败（rc={proc.returncode}）：\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    )
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(_PAYLOAD_MARKER)]
    assert lines, f"子进程未回传证据行：\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    return json.loads(lines[-1][len(_PAYLOAD_MARKER):])


def test_trace_callbacks_without_credentials_is_empty_and_imports_no_langfuse() -> None:
    """无凭据 → ``trace_callbacks() == []`` 且 ``langfuse`` 不在 ``sys.modules``。"""
    payload = _run_default_path_child()
    assert payload["callbacks_empty"], (
        f"无凭据时 trace_callbacks() 必须返回 []，实际 len={payload['callbacks_len']}"
    )
    assert not payload["sdk_imported"], (
        "无凭据时 trace_callbacks() 不得 import langfuse（应靠「无 key 直接返回 []」保住"
        "零 import / 零 SDK warning）"
    )


def test_trace_default_path_guard_is_not_vacuous() -> None:
    """子进程必须真的 import 了 pra.wiring 并实例化过 Settings。"""
    payload = _run_default_path_child()
    assert payload["wiring_loaded"], "子进程未 import pra.wiring → 本用例会空跑（先修脚本）"
    assert payload["settings_loaded"], (
        "子进程未加载 pra.infra.db（Settings 所在模块）→ 凭据判定路径未被真正执行"
    )
