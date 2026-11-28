"""The shell tool: run commands in the workspace.

The tool itself is OS-neutral. Which shell runs the command, and how process trees are
started and stopped, come from `dragonfish.platform`.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from dragonfish.platform.detect import current
from dragonfish.platform.process import kill_tree, release, spawn
from dragonfish.platform.shell import ShellBackend, ShellNotFound, find_shell
from dragonfish.tools.base import Permission, Tool, ToolContext, ToolError, ToolResult

DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 600
LOG_DIR = Path(tempfile.gettempdir()) / "dragonfish-logs"
TRAILING_AMPERSAND = re.compile(r"(?<!&)&\s*$")  # a single trailing "&", not "&&"

_DRIVE_ROOT = r"['\"]?[A-Za-z]:[\\/]?['\"]?"  # C:  C:\  "C:/"

# Refused in every mode, including yolo, whatever the OS or shell (a pattern for another
# OS simply never matches). Each entry: (pattern, reason shown to the model).
DENYLIST: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"\brm\s+(-\S+\s+)*(/|~|\$HOME|\$USERPROFILE|\$env:USERPROFILE|/[a-z]|/mnt/[a-z]|"
            r"[A-Za-z]:[\\/]?)/?(\*)?(\s|$|;|&)"
        ),
        "deleting the root, a drive or the home folder",
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
    # Windows (cmd and PowerShell)
    (
        re.compile(r"\bformat(\.com)?\s+[A-Za-z]:|\b(format-volume|clear-disk)\b", re.I),
        "formatting disks",
    ),
    (
        re.compile(rf"\b(rd|rmdir|del|erase)\b[^|;&]*\s{_DRIVE_ROOT}(\*(\.\*)?)?(\s|$|;|&)", re.I),
        "deleting a whole drive",
    ),
    (
        re.compile(
            rf"\b(remove-item|ri)\b[^|;&]*\s({_DRIVE_ROOT}|~|\$HOME|\$env:USERPROFILE)"
            r"(\\\*|/\*)?(\s|$|;|&)",
            re.I,
        ),
        "deleting a drive or the home folder",
    ),
    (re.compile(r"\b(diskpart|bcdedit)\b", re.I), "changing disks or boot settings"),
    (
        re.compile(r"\breg(\.exe)?\s+delete\s+HK(LM|EY_LOCAL_MACHINE)", re.I),
        "editing the system registry",
    ),
    (re.compile(r"\b(stop|restart)-computer\b", re.I), "shutting down the computer"),
    (
        re.compile(r"\brunas\b|\bstart-process\b[^|;&]*-verb\s+runas", re.I),
        "running commands as administrator",
    ),
    (
        re.compile(
            r"\b(iwr|irm|curl|wget|invoke-webrequest|invoke-restmethod)\b[^|;]*\|\s*"
            r"(iex|invoke-expression)\b",
            re.I,
        ),
        "piping a download straight into a shell",
    ),
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
    bin_dir = current().venv_bin_dir
    env = dict(os.environ)
    path = env.get("PATH", "").split(os.pathsep)
    if sys.prefix != sys.base_prefix:  # DragonFish itself runs inside a venv: hide it
        own_bin = os.path.normcase(str(Path(sys.prefix) / bin_dir))
        path = [p for p in path if os.path.normcase(p) != own_bin]
        env.pop("VIRTUAL_ENV", None)
    shim_dir = _install_shims()
    if shim_dir:
        path.insert(0, shim_dir)
    project_venv = workspace / ".venv"
    if (project_venv / bin_dir).is_dir():
        path.insert(0, str(project_venv / bin_dir))
        env["VIRTUAL_ENV"] = str(project_venv)
    env["PATH"] = os.pathsep.join(path)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONUTF8"] = "1"  # Python output is UTF-8 everywhere, as we decode it
    env.setdefault("TERM", "dumb")
    return env


SHIM_DIR = Path(tempfile.gettempdir()) / "dragonfish-shims"


def _install_shims() -> str | None:
    """Write the current shell's shims (if any) and return their folder for PATH."""
    try:
        shims = find_shell().shims
    except ShellNotFound:
        return None
    if not shims:
        return None
    SHIM_DIR.mkdir(parents=True, exist_ok=True)
    for name, script in shims:
        target = SHIM_DIR / name
        data = script.encode("utf-8")  # LF line endings: these are run by a POSIX shell
        if not target.is_file() or target.read_bytes() != data:
            target.write_bytes(data)
    return str(SHIM_DIR)


def _shell() -> ShellBackend:
    try:
        return find_shell()
    except ShellNotFound as exc:
        raise ToolError(str(exc)) from exc


def run_shell(inp: dict[str, Any], ctx: ToolContext) -> ToolResult:
    command = inp["command"]
    check_denylist(command)
    shell = _shell()
    env = build_env(ctx.workspace)
    if inp.get("background"):
        return _run_background(shell, command, ctx, env)
    if TRAILING_AMPERSAND.search(command):
        # `cmd &` leaves the program holding our output pipe, so we would wait until the
        # timeout. Run it as a proper background job instead.
        return _run_background(shell, TRAILING_AMPERSAND.sub("", command), ctx, env)

    timeout = min(inp.get("timeout_s", DEFAULT_TIMEOUT), MAX_TIMEOUT)
    proc = spawn(  # own process tree: a timeout stops children too
        shell.argv(command),
        background=False,
        cwd=ctx.workspace,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
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
                kill_tree(proc)
                return ToolResult(_decode(proc) + "\n[interrupted by the user]", is_error=True)
            if time.monotonic() > deadline:
                kill_tree(proc)
                partial = _decode(proc)
                return ToolResult(
                    f"{partial}\n[timed out after {timeout}s and was stopped. For programs that "
                    "keep running (servers, GUIs) use background: true]",
                    is_error=True,
                )
    release(proc)
    output = b"".join(chunks).decode("utf-8", errors="replace").rstrip()
    code = proc.returncode
    text = f"{output}\n[exit code {code}]" if output else f"[exit code {code}, no output]"
    return ToolResult(text, is_error=code != 0)


run_bash = run_shell  # the original name, kept for callers and tests


def _decode(proc: subprocess.Popen[Any]) -> str:
    try:
        out, _ = proc.communicate(timeout=1)
    except (subprocess.TimeoutExpired, ValueError):
        return ""
    return (out or b"").decode("utf-8", errors="replace").rstrip()


def _run_background(
    shell: ShellBackend, command: str, ctx: ToolContext, env: dict[str, str]
) -> ToolResult:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"bg-{int(time.time() * 1000)}.log"
    with log_path.open("wb") as log:
        proc = spawn(  # keeps running after DragonFish exits
            shell.argv(command),
            background=True,
            cwd=ctx.workspace,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    time.sleep(1.5)  # catch programs that crash immediately
    early = log_path.read_text(encoding="utf-8", errors="replace").strip()
    if proc.poll() is not None:
        return ToolResult(
            f"The background command exited right away with code {proc.returncode}.\n{early}",
            is_error=proc.returncode != 0,
        )
    tail = f"\nFirst output:\n{early[-2000:]}" if early else ""
    # Forward slashes work in every shell, including Git Bash.
    return ToolResult(
        f"Started in the background (pid {proc.pid}). Log: {log_path.as_posix()}{tail}"
    )


_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "string"},
        "timeout_s": {"type": "integer", "minimum": 1, "maximum": MAX_TIMEOUT},
        "background": {"type": "boolean"},
    },
    "required": ["command"],
    "additionalProperties": False,
}

_DESCRIPTION = (
    "Run a {kind} command in the workspace and return its output and exit code. Use it to run "
    "programs, tests, and package managers. Each call starts a fresh shell in the workspace "
    "(cd does not persist; chain with {chain}). For programs that keep running, such as GUIs "
    "or servers, set background: true. Do not use it to read or edit files; use the file tools."
)


def shell_tool(shell: ShellBackend | None = None) -> Tool:
    """The command tool for this machine's shell, named after it ("bash" or "powershell")."""
    if shell is None:
        try:
            shell = find_shell()
        except ShellNotFound:
            shell = None
    name = shell.name if shell else "bash"
    chain = "&&" if name == "bash" else ";"
    return Tool(
        name,
        _DESCRIPTION.format(kind=name, chain=chain),
        _SCHEMA,
        Permission.EXEC,
        run_shell,
    )


BASH_TOOL = shell_tool()
