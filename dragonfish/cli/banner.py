"""The startup banner: a pixel-art dragonfish and a blocky pixel title.

Pixels are drawn with half-block characters: each terminal cell shows two pixels stacked
vertically ("▀" with the top pixel as foreground and the bottom pixel as background).
"""

from __future__ import annotations

from rich.console import Group, RenderableType
from rich.style import Style
from rich.table import Table
from rich.text import Text

from dragonfish import __version__
from dragonfish.cli import theme

PALETTE = {
    "T": theme.DEEP_TEAL,
    "t": theme.TEAL,
    "b": theme.BRIGHT_TEAL,
    "E": theme.EMBER,
    "G": theme.GOLD,
    "W": "#F8FAFC",
    "K": "#020617",
}

# A deep-sea dragonfish facing right, with a glowing lure on its chin barbel.
SPRITE = [
    "..............................",
    "......b.........TTTT..........",
    ".....bTb....TTTTttttTTT.......",
    "....bTTTb.TTtttttttttttTT.....",
    "...bTTTTTTtttbbbbtttttWKtT....",
    "....bTTTb.TttttttttttttTWEW...",
    ".....bTb...TTtttttttttTWEWE...",
    "......b.....TTTTtttttTT...E...",
    "...............TTTTTT......E..",
    "............................G.",
    "...........................GEG",
    "............................G.",
]

# 5x5 pixel font for the title.
FONT = {
    "D": ["1111.", "1...1", "1...1", "1...1", "1111."],
    "R": ["1111.", "1...1", "1111.", "1..1.", "1...1"],
    "A": [".111.", "1...1", "11111", "1...1", "1...1"],
    "G": [".1111", "1....", "1..11", "1...1", ".1111"],
    "O": [".111.", "1...1", "1...1", "1...1", ".111."],
    "N": ["1...1", "11..1", "1.1.1", "1..11", "1...1"],
    "F": ["11111", "1....", "1111.", "1....", "1...."],
    "I": ["111", ".1.", ".1.", ".1.", "111"],
    "S": [".1111", "1....", ".111.", "....1", "1111."],
    "H": ["1...1", "1...1", "11111", "1...1", "1...1"],
    " ": ["..", "..", "..", "..", ".."],
}


def _lerp(a: str, b: str, t: float) -> str:
    ca = [int(a[i : i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb, strict=True))


def render_pixels(rows: list[list[str | None]]) -> Text:
    """Render a grid of hex colors (None = transparent) using half blocks."""
    if len(rows) % 2:
        rows = [*rows, [None] * len(rows[0])]
    out = Text()
    for top, bottom in zip(rows[::2], rows[1::2], strict=True):
        for up, down in zip(top, bottom, strict=True):
            if up and down:
                out.append("▀", Style(color=up, bgcolor=down))
            elif up:
                out.append("▀", Style(color=up))
            elif down:
                out.append("▄", Style(color=down))
            else:
                out.append(" ")
        out.append("\n")
    out.rstrip()
    return out


def sprite_pixels() -> list[list[str | None]]:
    return [[PALETTE.get(ch) for ch in row] for row in SPRITE]


def title_pixels(word: str = "DRAGONFISH") -> list[list[str | None]]:
    glyph_rows: list[str] = ["" for _ in range(5)]
    for ch in word:
        for i, line in enumerate(FONT[ch]):
            glyph_rows[i] += line + "."
    width = len(glyph_rows[0])
    grid: list[list[str | None]] = []
    for line in glyph_rows:
        grid.append(
            [
                _lerp(theme.BRIGHT_TEAL, theme.EMBER, x / max(1, width - 1)) if c == "1" else None
                for x, c in enumerate(line)
            ]
        )
    return [[None] * width, *grid]  # one blank row on top so the title sits centered


def banner(
    model: str, effort: str, mode: str, workspace: str, demo: bool = False
) -> RenderableType:
    info = Text()
    info.append(f"v{__version__}", style=theme.MUTED)
    info.append("  ·  a hand-built coding agent\n\n", style=theme.MUTED)
    info.append("model     ", style=theme.MUTED)
    info.append(f"{model}" + ("  (offline demo)" if demo else "") + "\n", style=theme.TEXT)
    info.append("effort    ", style=theme.MUTED)
    info.append(f"{effort}\n", style=theme.TEXT)
    info.append("mode      ", style=theme.MUTED)
    info.append(f"{mode}\n", style=theme.TEXT)
    info.append("workspace ", style=theme.MUTED)
    info.append(workspace, style=theme.TEXT)

    right = Group(render_pixels(title_pixels()), Text(""), info)
    grid = Table.grid(padding=(0, 3))
    grid.add_column()
    grid.add_column()
    grid.add_row(render_pixels(sprite_pixels()), right)

    tips = Text()
    tips.append("\n  /help", style=f"bold {theme.BRIGHT_TEAL}")
    tips.append(" commands  ·  ", style=theme.MUTED)
    tips.append("Ctrl+C", style=f"bold {theme.BRIGHT_TEAL}")
    tips.append(" interrupt  ·  ", style=theme.MUTED)
    tips.append("Esc then Enter", style=f"bold {theme.BRIGHT_TEAL}")
    tips.append(" new line  ·  ", style=theme.MUTED)
    tips.append("/exit", style=f"bold {theme.BRIGHT_TEAL}")
    tips.append(" quit\n", style=theme.MUTED)
    return Group(Text(""), grid, tips)
