"""The local-model engines, chosen by what this machine can run.

To add an engine: write a `LocalProvider` subclass and add an `Engine` row to `ENGINES`.
Engines are picked by capability, never by OS name, so a new OS that can run an existing
engine needs no change here.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from dragonfish.core.config import Settings
from dragonfish.platform.detect import PlatformInfo, current, nvidia_vram_gb
from dragonfish.providers import local_models
from dragonfish.providers.base import ProviderError
from dragonfish.providers.local_base import LocalProvider
from dragonfish.providers.local_models import ModelChoice

INSTALL_HINT = (
    "Local models need the `local` extra. Reinstall with it: "
    'uv tool install --editable ".[local]" --force   (or, in a checkout: uv sync --extra local)'
)


@dataclass(frozen=True)
class Engine:
    name: str
    weights_format: str  # "mlx" | "gguf"
    runs_here: Callable[[PlatformInfo], bool]  # can this machine run it at all?
    installed: Callable[[], bool]  # is its Python package importable?
    provider: Callable[[], type[LocalProvider]]  # imported lazily: engines are optional
    budgets: Callable[[PlatformInfo], list[tuple[str, float]]]  # memory to size the model by


def _mlx_installed() -> bool:
    from dragonfish.providers import local_mlx

    return local_mlx.available()


def _mlx_provider() -> type[LocalProvider]:
    from dragonfish.providers.local_mlx import LocalMLXProvider

    return LocalMLXProvider


def _llama_installed() -> bool:
    from dragonfish.providers import local_llama

    return local_llama.available()


def _llama_provider() -> type[LocalProvider]:
    from dragonfish.providers.local_llama import LlamaCppProvider

    return LlamaCppProvider


def _llama_budgets(info: PlatformInfo) -> list[tuple[str, float]]:
    from dragonfish.providers import local_llama

    budgets: list[tuple[str, float]] = []
    if local_llama.gpu_offload_supported():
        vram = nvidia_vram_gb()
        if vram:
            budgets.append(("gpu", vram))
    budgets.append(("cpu", info.ram_gb))
    return budgets


# In order of preference: the first one that runs here and is installed wins.
ENGINES: list[Engine] = [
    Engine(
        "mlx",
        "mlx",
        runs_here=lambda info: info.is_apple_silicon,
        installed=_mlx_installed,
        provider=_mlx_provider,
        budgets=lambda info: [("apple_gpu", info.ram_gb)],
    ),
    Engine(
        "llamacpp",
        "gguf",
        runs_here=lambda info: True,
        installed=_llama_installed,
        provider=_llama_provider,
        budgets=_llama_budgets,
    ),
]


def select_engine(settings: Settings) -> Engine:
    info = current()
    if settings.local_engine != "auto":
        engine = next(e for e in ENGINES if e.name == settings.local_engine)
        if not engine.runs_here(info):
            raise ProviderError(f"The {engine.name} engine does not run on this machine.")
        if not engine.installed():
            raise ProviderError(f"The {engine.name} engine is not installed. {INSTALL_HINT}")
        return engine
    for engine in ENGINES:
        if engine.runs_here(info) and engine.installed():
            return engine
    raise ProviderError(INSTALL_HINT)


def start_local(
    settings: Settings,
    confirm_download: Callable[[ModelChoice, Path], bool],
    loading: Callable[[str], AbstractContextManager[Any]] | None = None,
) -> tuple[LocalProvider, Settings]:
    """Pick engine and model, download if needed (after `confirm_download`), and load it.

    Returns the provider and the settings with the model choice filled in.
    """
    engine = select_engine(settings)
    choice = local_models.choose(settings, engine.weights_format, engine.budgets(current()))
    settings = replace(
        settings,
        local_engine=engine.name,
        local_model=choice.repo,
        local_model_file=choice.file,
        local_context=choice.context,
    )
    path = local_models.find_weights(settings, choice)
    if path is None:
        if not confirm_download(choice, local_models.model_dir(settings, choice.repo)):
            raise ProviderError("Local model not downloaded.")
        path = local_models.download(settings, choice)
    provider_class = engine.provider()
    status = loading or (lambda _message: nullcontext())
    with status(f"Loading {settings.display_model} into memory…"):
        return provider_class(settings, path), settings
