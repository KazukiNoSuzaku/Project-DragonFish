"""What machine are we on? OS, CPU, memory, and the OS conventions the tools depend on."""

from __future__ import annotations

import functools
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PlatformInfo:
    system: str  # "windows" | "darwin" | "linux" | other lowercase platform.system()
    machine: str  # normalised CPU architecture: "arm64" | "x86_64" | ...
    ram_gb: float  # total physical memory (0.0 if it could not be read)

    @property
    def is_windows(self) -> bool:
        return self.system == "windows"

    @property
    def is_apple_silicon(self) -> bool:
        return self.system == "darwin" and self.machine == "arm64"

    @property
    def venv_bin_dir(self) -> str:
        """The folder inside a virtualenv that holds its executables."""
        return "Scripts" if self.is_windows else "bin"

    @property
    def python_names(self) -> tuple[str, ...]:
        """Python command names to look for, most conventional first."""
        return ("python", "python3") if self.is_windows else ("python3", "python")

    def describe(self) -> str:
        return f"{platform.system()} {platform.release()} ({platform.machine()})"


_ARCH_ALIASES = {"amd64": "x86_64", "x64": "x86_64", "aarch64": "arm64"}


@functools.cache
def current() -> PlatformInfo:
    machine = platform.machine().lower()
    return PlatformInfo(
        system=_system_name(),
        machine=_ARCH_ALIASES.get(machine, machine),
        ram_gb=total_ram_gb(),
    )


def _system_name() -> str:
    if sys.platform == "win32":
        return "windows"
    return platform.system().lower()


def total_ram_gb() -> float:
    try:
        import psutil
    except ImportError:  # psutil is a dependency, but never let detection crash the app
        return 0.0
    return round(psutil.virtual_memory().total / 1024**3, 1)


@functools.cache
def nvidia_vram_gb() -> float:
    """Memory of the largest NVIDIA GPU, or 0.0 when there is none (or no driver tools)."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return 0.0
    try:
        out = subprocess.run(
            [smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return 0.0
    sizes = [int(m) for m in re.findall(r"\d+", out)]
    return round(max(sizes) / 1024, 1) if sizes else 0.0


def is_store_stub(path: str | Path) -> bool:
    """True for Windows' app-execution aliases in WindowsApps (python3.exe, bash.exe, ...).

    They exist on a fresh Windows install but only open the Microsoft Store or WSL, so a
    command found there is not really installed for our purposes.
    """
    parts = {p.lower() for p in Path(path).parts}
    return "windowsapps" in parts


def to_native_path(path: str) -> str:
    """Translate Git-Bash style drive paths (/c/Users/...) to native ones (C:/Users/...).

    Models working through Git Bash see `pwd` print /c/..., and then use that form in file
    tools. Everywhere other than Windows this returns the path unchanged.
    """
    if not current().is_windows:
        return path
    match = re.match(r"^/(?:mnt/)?([a-zA-Z])(/.*)?$", path)
    if not match:
        return path
    return f"{match.group(1).upper()}:{match.group(2) or '/'}"
