"""Which shell runs the agent's commands, and how to call it.

Each backend module exposes `find() -> ShellBackend | None`, returning None when it does not
apply to this machine. `find_shell()` asks them in order. A future port adds a module and a
line in `FINDERS`; nothing else changes.

Override with the DRAGONFISH_SHELL environment variable: a backend name ("bash",
"powershell") or the full path to a shell executable.
"""

from __future__ import annotations

import base64
import functools
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from dragonfish.platform.detect import current


@dataclass(frozen=True)
class ShellBackend:
    name: str  # the tool name the model sees: "bash" or "powershell"
    label: str  # for people: "bash", "Git Bash", "Windows PowerShell"
    executable: str
    args: tuple[str, ...]  # placed before the command
    prompt_hint: str  # one line for the system prompt's Environment section
    wrap: str = "{command}"  # template around the command (e.g. exit-code handling)
    encoded: bool = False  # pass the script base64/UTF-16LE (PowerShell -EncodedCommand)
    # Small scripts put on PATH for this shell, by command name (e.g. a `python3` that runs
    # `python` where Windows has no python3). Placed after the project's .venv.
    shims: tuple[tuple[str, str], ...] = ()

    def argv(self, command: str) -> list[str]:
        script = self.wrap.format(command=command)
        if self.encoded:  # immune to Windows command-line quoting rules
            script = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        return [self.executable, *self.args, script]


class ShellNotFound(RuntimeError):
    """No usable shell on this machine."""


def _finders() -> list[Callable[[], ShellBackend | None]]:
    from dragonfish.platform.shell import posix_bash, windows_gitbash, windows_powershell

    # Order matters: the first backend that applies wins.
    return [posix_bash.find, windows_gitbash.find, windows_powershell.find]


def _from_override(value: str) -> ShellBackend | None:
    from dragonfish.platform.shell import posix_bash, windows_gitbash, windows_powershell

    path = Path(value)
    if path.suffix or os.sep in value or "/" in value:  # a path to an executable
        if not path.is_file():
            raise ShellNotFound(f"DRAGONFISH_SHELL points to {value}, which does not exist.")
        stem = path.stem.lower()
        if stem in ("pwsh", "powershell"):
            return windows_powershell.backend(str(path))
        if stem == "bash":
            module = windows_gitbash if current().is_windows else posix_bash
            return module.backend(str(path))
        raise ShellNotFound(f"DRAGONFISH_SHELL: unsupported shell {path.name}.")
    for find in _finders():
        found = find()
        if found and found.name == value.lower():
            return found
    raise ShellNotFound(f"DRAGONFISH_SHELL={value}: that shell was not found on this machine.")


@functools.cache
def find_shell() -> ShellBackend:
    override = os.environ.get("DRAGONFISH_SHELL", "").strip()
    if override:
        found = _from_override(override)
        if found:
            return found
    for find in _finders():
        found = find()
        if found:
            return found
    raise ShellNotFound(
        "No shell found for running commands. On Windows, install Git for Windows "
        "(https://git-scm.com/download/win), which includes Git Bash."
    )
