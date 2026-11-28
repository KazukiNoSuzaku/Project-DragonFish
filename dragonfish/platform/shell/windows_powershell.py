"""PowerShell on Windows: the fallback when Git Bash is not installed (always present)."""

from __future__ import annotations

import shutil
from pathlib import Path

from dragonfish.platform.detect import current
from dragonfish.platform.shell import ShellBackend

# UTF-8 output, and a real exit code: plain -Command only reports 0 or 1.
_WRAP = (
    "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8\n"
    "$global:LASTEXITCODE = 0\n"
    "{command}\n"
    "if (-not $?) {{ if ($LASTEXITCODE) {{ exit $LASTEXITCODE }} else {{ exit 1 }} }}\n"
    "exit $LASTEXITCODE"
)


def backend(executable: str) -> ShellBackend:
    edition = "PowerShell 7" if Path(executable).stem.lower() == "pwsh" else "Windows PowerShell"
    chain = "`&&` or `;`" if edition == "PowerShell 7" else "`;` (not `&&`)"
    return ShellBackend(
        name="powershell",
        label=edition,
        executable=executable,
        args=("-NoLogo", "-NoProfile", "-NonInteractive", "-EncodedCommand"),
        prompt_hint=(
            f"- Shell: {edition} (the `powershell` tool; bash is not installed). Use PowerShell "
            f"syntax: chain commands with {chain}, read variables with $env:NAME. "
            "Run Python as `python`."
        ),
        wrap=_WRAP,
        encoded=True,
    )


def find() -> ShellBackend | None:
    if not current().is_windows:
        return None
    exe = shutil.which("pwsh") or shutil.which("powershell")
    return backend(exe) if exe else None
