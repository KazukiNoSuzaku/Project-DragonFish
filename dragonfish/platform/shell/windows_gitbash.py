"""Git Bash on Windows: the preferred shell there, so the agent keeps writing bash.

Plain `bash` on a Windows PATH is usually the WSL launcher (System32\\bash.exe) or a Store
alias, which run commands inside Linux, not in the Windows workspace. Those are skipped.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from dragonfish.platform.detect import current, is_store_stub
from dragonfish.platform.shell import ShellBackend


def backend(executable: str) -> ShellBackend:
    return ShellBackend(
        name="bash",
        label="Git Bash",
        executable=executable,
        args=("-c",),
        prompt_hint=(
            "- Shell: Git Bash on Windows (the `bash` tool). It prints paths as /c/Users/...; "
            "use forward slashes. Run Python as `python` (not `python3`)."
        ),
        # Windows Python installs have no python3 (only a Store alias): send it to `python`.
        shims=(("python3", '#!/bin/sh\nexec python "$@"\n'),),
    )


def candidates() -> list[Path]:
    """Places Git for Windows puts bash.exe, most reliable first."""
    found: list[Path] = []
    git = shutil.which("git")
    if git:
        # git.exe lives in <root>\cmd, <root>\bin or <root>\mingw64\bin.
        for parent in Path(git).resolve().parents[:3]:
            found.append(parent / "bin" / "bash.exe")
    for var, sub in (
        ("ProgramFiles", "Git"),
        ("ProgramFiles(x86)", "Git"),
        ("LOCALAPPDATA", "Programs/Git"),
    ):
        base = os.environ.get(var)
        if base:
            found.append(Path(base) / sub / "bin" / "bash.exe")
    on_path = shutil.which("bash")
    if on_path:
        found.append(Path(on_path))
    return found


def is_wsl_launcher(path: Path) -> bool:
    parts = {p.lower() for p in path.parts}
    return "system32" in parts or "sysnative" in parts or is_store_stub(path)


def find() -> ShellBackend | None:
    if not current().is_windows:
        return None
    for path in candidates():
        if path.is_file() and not is_wsl_launcher(path):
            return backend(str(path))
    return None
