"""Settings: defaults <- ~/.dragonfish/config.toml <- command-line flags."""

from __future__ import annotations

import os
import platform
import tomllib
from dataclasses import dataclass, field, fields, replace
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
PERMISSION_MODES = ("ask", "auto-edit", "yolo")


class ConfigError(ValueError):
    """Raised for invalid configuration values."""


def _default_data_dir() -> Path:
    return Path(os.environ.get("DRAGONFISH_HOME", Path.home() / ".dragonfish"))


@dataclass(frozen=True)
class Settings:
    model: str = "claude-opus-5-5"
    effort: str = "high"  # Opus 5.5 defaults to "medium"; coding work benefits from "high".
    max_tokens: int = 64000  # Safe because every request streams.
    workspace: Path = field(default_factory=Path.cwd)
    permission_mode: str = "ask"
    max_iterations: int = 60  # Hard stop for runaway loops.
    refusal_fallback: bool = True  # Server-side fallback model when a request is refused.
    data_dir: Path = field(default_factory=_default_data_dir)

    def __post_init__(self) -> None:
        if self.effort not in EFFORT_LEVELS:
            raise ConfigError(f"effort must be one of {EFFORT_LEVELS}, got {self.effort!r}")
        if self.permission_mode not in PERMISSION_MODES:
            raise ConfigError(
                f"permission mode must be one of {PERMISSION_MODES}, got {self.permission_mode!r}"
            )
        if self.max_iterations < 1:
            raise ConfigError("max_iterations must be at least 1")
        # Normalise the workspace once so every path check compares resolved paths.
        object.__setattr__(self, "workspace", Path(self.workspace).expanduser().resolve())
        object.__setattr__(self, "data_dir", Path(self.data_dir).expanduser())


_FILE_KEYS = {f.name for f in fields(Settings)}
_PATH_KEYS = {"workspace", "data_dir"}


def _read_config_file(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    unknown = set(raw) - _FILE_KEYS
    if unknown:
        raise ConfigError(f"unknown keys in {path}: {', '.join(sorted(unknown))}")
    return {k: Path(v) if k in _PATH_KEYS else v for k, v in raw.items()}


def load_settings(args: Any = None, *, config_file: Path | None = None) -> Settings:
    """Build Settings from defaults, the optional config file, and parsed CLI args."""
    base = Settings()
    load_dotenv(base.data_dir / ".env")  # global secrets
    load_dotenv()  # project/cwd .env (searched upwards); does not override existing env vars

    overrides = _read_config_file(config_file or base.data_dir / "config.toml")
    if args is not None:
        cli = {
            "model": getattr(args, "model", None),
            "effort": getattr(args, "effort", None),
            "permission_mode": getattr(args, "mode", None),
            "workspace": getattr(args, "workspace", None),
        }
        overrides.update({k: v for k, v in cli.items() if v is not None})
    return replace(base, **overrides)


def load_system_prompt(settings: Settings) -> str:
    """Render the system prompt. Kept stable for a whole session so prompt caching works."""
    template = resources.files("dragonfish.prompts").joinpath("system.md").read_text("utf-8")
    return template.format(
        workspace=settings.workspace,
        os=f"{platform.system()} {platform.release()} ({platform.machine()})",
        date=date.today().isoformat(),
    )
