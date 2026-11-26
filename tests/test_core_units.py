from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from conftest import make_tool

from dragonfish.__main__ import main
from dragonfish.core.config import ConfigError, Settings, load_settings, load_system_prompt
from dragonfish.core.permissions import Decision, PermissionGate
from dragonfish.tools.base import Permission
from dragonfish.tools.registry import Registry, truncate

# --- config -------------------------------------------------------------------------------


def test_defaults():
    s = Settings()
    assert s.model == "claude-opus-5-5" and s.effort == "high" and s.permission_mode == "ask"


def test_invalid_values_rejected():
    with pytest.raises(ConfigError):
        Settings(effort="extreme")
    with pytest.raises(ConfigError):
        Settings(permission_mode="anything")


def test_cli_flags_override_config_file(tmp_path: Path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('model = "from-file"\neffort = "low"\n')
    args = argparse.Namespace(model="from-cli", effort=None, mode="yolo", workspace=tmp_path)
    s = load_settings(args, config_file=cfg)
    assert s.model == "from-cli" and s.effort == "low" and s.permission_mode == "yolo"
    assert s.workspace == tmp_path.resolve()


def test_unknown_config_key_rejected(tmp_path: Path):
    cfg = tmp_path / "config.toml"
    cfg.write_text("modle = 'typo'\n")
    with pytest.raises(ConfigError, match="modle"):
        load_settings(None, config_file=cfg)


def test_system_prompt_renders(tmp_path: Path):
    prompt = load_system_prompt(Settings(workspace=tmp_path))
    assert "DragonFish" in prompt and str(tmp_path.resolve()) in prompt
    assert "{" not in prompt  # every placeholder filled


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "DragonFish 0.1.0" in capsys.readouterr().out


# --- permissions --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "perm", "expected"),
    [
        ("ask", Permission.READ, Decision.ALLOW),
        ("ask", Permission.WRITE, Decision.ASK),
        ("ask", Permission.EXEC, Decision.ASK),
        ("auto-edit", Permission.WRITE, Decision.ALLOW),
        ("auto-edit", Permission.EXEC, Decision.ASK),
        ("yolo", Permission.EXEC, Decision.ALLOW),
    ],
)
def test_mode_table(mode, perm, expected):
    decision, _ = PermissionGate(mode).check(make_tool("t", perm), {})
    assert decision is expected


def test_taint_overrides_always_except_in_yolo():
    tool = make_tool("bash", Permission.EXEC)
    gate = PermissionGate("auto-edit")
    gate.remember("bash")
    assert gate.check(tool, {})[0] is Decision.ALLOW
    gate.tainted = True
    assert gate.check(tool, {})[0] is Decision.ASK
    gate.set_mode("yolo")
    assert gate.check(tool, {})[0] is Decision.ALLOW


# --- registry -----------------------------------------------------------------------------


def test_registry_schemas_are_stable_and_validated():
    reg = Registry([make_tool("b"), make_tool("a")])
    assert [s["name"] for s in reg.schemas()] == ["b", "a"]
    assert reg.validate("a", {"x": "ok"}) is None
    assert "Unknown tool" in reg.validate("zzz", {})
    assert "must be a JSON object" in reg.validate("a", "text")
    assert "x" in reg.validate("a", {})
    with pytest.raises(ValueError):
        reg.register(make_tool("a"))


def test_truncate_keeps_head_and_tail():
    text = "HEAD" + "x" * 100 + "TAIL"
    out = truncate(text, limit=20)
    assert out.startswith("HEAD") and out.endswith("TAIL") and "truncated" in out
    assert truncate("short", limit=20) == "short"
