from __future__ import annotations

import io

from rich.console import Console

from dragonfish.cli.banner import banner, render_pixels, sprite_pixels
from dragonfish.cli.render import Renderer, short_args
from dragonfish.core.agent import Agent
from dragonfish.core.config import Settings
from dragonfish.core.permissions import PermissionGate
from dragonfish.providers.demo import LOOK_AROUND, DemoProvider
from dragonfish.tools.registry import Registry


def console() -> tuple[Console, io.StringIO]:
    buf = io.StringIO()
    return Console(file=buf, width=100, force_terminal=False), buf


def test_banner_renders():
    con, buf = console()
    con.print(banner("claude-opus-5-5", "high", "ask", "~/x"))
    out = buf.getvalue()
    assert "claude-opus-5-5" in out and "/help" in out
    assert "▀" in render_pixels(sprite_pixels()).plain


def test_short_args_prefers_meaningful_keys():
    assert short_args({"command": "pytest -q", "timeout_s": 5}) == "pytest -q"
    assert short_args({"x": "y" * 200}, limit=20).endswith("…")


def test_demo_session_renders_end_to_end(tmp_path):
    (tmp_path / "calc.py").write_text("")
    agent = Agent(
        DemoProvider(delay=0),
        Registry([LOOK_AROUND]),
        PermissionGate(),
        Settings(workspace=tmp_path),
        "sys",
    )
    con, buf = console()
    renderer = Renderer(con)
    renderer.start_turn()
    for event in agent.run("hi"):
        renderer.handle(event)
    renderer.end_turn()
    out = buf.getvalue()
    assert "look_around" in out and "calc.py" in out
    assert "offline demo mode" in out and "2.6k in" in out
