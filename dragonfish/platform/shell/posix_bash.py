"""bash on macOS and other POSIX systems (DragonFish's original shell)."""

from __future__ import annotations

import shutil

from dragonfish.platform.detect import current
from dragonfish.platform.shell import ShellBackend


def backend(executable: str = "bash") -> ShellBackend:
    return ShellBackend(
        name="bash",
        label="bash",
        executable=executable,
        args=("-c",),
        prompt_hint="- Shell: bash (the `bash` tool).",
    )


def find() -> ShellBackend | None:
    if current().is_windows:
        return None
    return backend(shutil.which("bash") or "bash")
