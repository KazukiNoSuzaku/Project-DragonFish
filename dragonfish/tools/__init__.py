"""DragonFish tools."""

from dragonfish.tools.base import Tool
from dragonfish.tools.fs import FS_TOOLS
from dragonfish.tools.shell import BASH_TOOL


def default_tools() -> list[Tool]:
    """The standard tool set, in a fixed order (a stable order keeps the prompt cache warm)."""
    return [*FS_TOOLS, BASH_TOOL]
