"""Terminal setup that differs per OS."""

from __future__ import annotations

import contextlib
import io
import sys


def ensure_utf8_stdio() -> None:
    """Make stdout/stderr UTF-8 so output like "❯" or model text never crashes printing.

    On Windows, output that is piped or redirected uses the legacy code page (e.g. cp1252),
    which cannot encode most of what DragonFish and the model print.
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if encoding != "utf8" and isinstance(stream, io.TextIOWrapper):
            with contextlib.suppress(ValueError, OSError):
                stream.reconfigure(encoding="utf-8", errors="replace")
