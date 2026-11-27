"""The bash tool: run commands in the workspace."""

from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from dragonfish.tools.base import Permission, Tool, ToolContext, ToolError, ToolResult

DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 600
LOG_DIR = Path(tempfile.gettempdir()) / "dragonfish-logs"
TRAILING_AMPERSAND = re.compile(r"(?<!&)&\s*$")  # a single trailing "&", not "&&"

# Refused in every mode, including yolo. Each entry: (pattern, reason shown to the model).
DENYLIST: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"\brm\s+(-\S+\s+)*(/|~|\$HOME)/?(\*)?(\s|$|;|&)"),
        "deleting the root or home folder",
    ),
    (re.compile(r"\bsudo\b"), "running commands as administrator (sudo)"),
    (re.compile(r"\bmkfs(\.\w+)?\b"), "formatting disks"),
    (re.compile(r"\bdd\b[^|;&]*\bof=/dev/"), "writing raw data to a device"),
    (re.compile(r":\(\)\s*\{.*\}\s*;\s*:"), "a fork bomb"),
    (re.compile(r"\b(shutdown|reboot|halt|poweroff)\b"), "shutting down the computer"),
    (
        re.compile(r"\b(curl|wget)\b[^|;&]*\|\s*(sudo\s+)?(ba|z|da)?sh\b"),
        "piping a download straight into a shell",
    ),
    (re.compile(r"\bchmod\s+(-\S+\s+)*777\s+/(\s|$)"), "changing permissions on the whole disk"),
    (re.compile(r">\s*/dev/(sd|disk|nvme)"), "overwriting a disk device"),
]


def check_denylist(command: str) -> None:
    for pattern, reason in DENYLIST:
        if pattern.search(command):
            raise ToolError(
                f"Blocked: this command looks like {reason}, which DragonFish never runs. "
                "Find a safer way, or ask the user to run it themselves."
            )


def build_env(workspace: Path) -> dict[str, str]:
    """The environment for commands: DragonFish's own virtualenv is hidden, the project's used."""
    env = dict(os.environ)
    path = env.get("PATH", "").split(os.pathsep)
    if sys.prefix != sys.base_prefix:  # DragonFish itself runs inside a venv: hide it
        own_bin = str(Path(sys.prefix) / "bin")
        path = [p for p in path if p != own_bin]
        env.pop("VIRTUAL_ENV", None)
    project_venv = workspace / ".venv"
    if (project_venv / "bin").is_dir():
        path.insert(0, str(project_venv / "bin"))
        env["VIRTUAL_ENV"] = str(project_venv)
    env["PATH"] = os.pathsep.join(path)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("TERM", "dumb")
    return env


def _kill(proc: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=2)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)


def run_bash(inp: dict[str, Any], ctx: ToolContext) -> ToolResult:
    command = inp["command"]
    check_denylist(command)
    env = build_env(ctx.workspace)
    if inp.get("background"):
        return _run_background(command, ctx, env)
    if TRAILING_AMPERSAND.search(command):
        # `cmd &` leaves the program holding our output pipe, so we would wait until the
        # timeout. Run it as a proper background job instead.
        return _run_background(TRAILING_AMPERSAND.sub("", command), ctx, env)

    timeout = min(inp.get("timeout_s", DEFAULT_TIMEOUT), MAX_TIMEOUT)
    proc = subprocess.Popen(
        ["bash", "-c", command],
        cwd=ctx.workspace,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # own process group, so a timeout kills child processes too
    )
    deadline = time.monotonic() + timeout
    chunks: list[bytes] = []
    while True:
        try:
            out, _ = proc.communicate(timeout=0.25)
            chunks.append(out or b"")
            break
        except subprocess.TimeoutExpired:
            if ctx.cancel.is_set():
                _kill(proc)
                return ToolResult(_decode(proc) + "\n[interrupted by the user]", is_error=True)
            if time.monotonic() > deadline:
                _kill(proc)
                partial = _decode(proc)
                return ToolResult(
                    f"{partial}\n[timed out after {timeout}s and was stopped. For programs that "
                    "keep running (servers, GUIs) use background: true]",
                    is_error=True,
                )
    output = b"".join(chunks).decode("utf-8", errors="replace").rstrip()
    code = proc.returncode
    text = f"{output}\n[exit code {code}]" if output else f"[exit code {code}, no output]"
    return ToolResult(text, is_error=code != 0)


def _decode(proc: subprocess.Popen[Any]) -> str:
    try:
        out, _ = proc.communicate(timeout=1)
    except (subprocess.TimeoutExpired, ValueError):
        return ""
    return (out or b"").decode("utf-8", errors="replace").rstrip()


def _run_background(command: str, ctx: ToolContext, env: dict[str, str]) -> ToolResult:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"bg-{int(time.time() * 1000)}.log"
    with log_path.open("wb") as log:
        proc = subprocess.Popen(
            ["bash", "-c", command],
            cwd=ctx.workspace,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # keeps running after DragonFish exits
        )
    time.sleep(1.5)  # catch programs that crash immediately
    early = log_path.read_text(errors="replace").strip()
    if proc.poll() is not None:
        return ToolResult(
            f"The background command exited right away with code {proc.returncode}.\n{early}",
            is_error=proc.returncode != 0,
        )
    tail = f"\nFirst output:\n{early[-2000:]}" if early else ""
    return ToolResult(f"Started in the background (pid {proc.pid}). Log: {log_path}{tail}")


BASH_TOOL = Tool(
    "bash",
    "Run a bash command in the workspace and return its output and exit code. Use it to run "
    "programs, tests, and package managers. Each call starts a fresh shell in the workspace "
    "(cd does not persist; chain with &&). For programs that keep running, such as GUIs or "
    "servers, set background: true. Do not use it to read or edit files; use the file tools.",
    {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout_s": {"type": "integer", "minimum": 1, "maximum": MAX_TIMEOUT},
            "background": {"type": "boolean"},
        },
        "required": ["command"],
        "additionalProperties": False,
    },
    Permission.EXEC,
    run_bash,
)
