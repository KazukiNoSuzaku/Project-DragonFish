"""Everything that differs between operating systems lives in this package.

The rest of DragonFish never checks `sys.platform` or `os.name` itself: it asks this package.
A new port (Linux, Intel Mac, ...) adds a shell backend or a row in a table here, and the
callers stay unchanged.
"""

from dragonfish.platform.detect import PlatformInfo, current

__all__ = ["PlatformInfo", "current"]
