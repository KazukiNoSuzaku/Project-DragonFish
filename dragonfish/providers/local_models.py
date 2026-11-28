"""Which local model to run on this machine, and where its weights live.

Everything here is a table: a new model, size or weights format is a new row, not new code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dragonfish.core.config import Settings


@dataclass(frozen=True)
class Weights:
    repo: str  # Hugging Face repo id
    file: str | None  # one file inside the repo (GGUF), or None for the whole repo (MLX)
    size_gb: float  # download size


@dataclass(frozen=True)
class Tier:
    label: str
    context: int  # default context window, in tokens
    weights: dict[str, Weights]  # by format: "gguf", "mlx"


# Qwen3.5, 4-bit. Qwen3.5 mixes linear-attention layers with full attention, so its KV cache
# is small: even the 64k window of the 9B model needs only ~2 GB.
TIERS: dict[str, Tier] = {
    "0.8B": Tier(
        "0.8B",
        16384,
        {"gguf": Weights("unsloth/Qwen3.5-0.8B-GGUF", "Qwen3.5-0.8B-Q4_K_M.gguf", 0.53)},
    ),
    "2B": Tier(
        "2B",
        16384,
        {"gguf": Weights("unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q4_K_M.gguf", 1.28)},
    ),
    "4B": Tier(
        "4B",
        32768,
        {
            "gguf": Weights("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf", 2.74),
            "mlx": Weights("mlx-community/Qwen3.5-4B-MLX-4bit", None, 3.0),
        },
    ),
    "9B": Tier(
        "9B",
        65536,
        {
            "gguf": Weights("unsloth/Qwen3.5-9B-GGUF", "Qwen3.5-9B-Q4_K_M.gguf", 5.68),
            "mlx": Weights("mlx-community/Qwen3.5-9B-4bit", None, 6.0),
        },
    ),
}

# Which tier fits, by the kind of memory the model will run in. Rows: (at least N GB, tier),
# largest first; the first row that fits wins.
BUDGETS: dict[str, list[tuple[float, str]]] = {
    # Apple Silicon (MLX): unified memory, GPU speed.
    "apple_gpu": [(14, "9B"), (0, "4B")],
    # llama.cpp with layers on a dedicated GPU: by video memory.
    "gpu": [(7, "9B"), (3.5, "4B")],
    # llama.cpp on the CPU: by RAM, capped at 4B so replies stay usably fast.
    "cpu": [(11, "4B"), (7, "2B"), (0, "0.8B")],
}

DEFAULT_GGUF_PATTERN = "*Q4_K_M.gguf"  # for a --local-model repo given without a file


@dataclass(frozen=True)
class ModelChoice:
    repo: str
    file: str | None
    format: str  # "gguf" | "mlx"
    context: int
    size_gb: float | None  # None when unknown (a model the user picked)
    reason: str  # why this model, shown before downloading


def pick_tier(budget: str, memory_gb: float) -> str | None:
    for minimum, label in BUDGETS[budget]:
        if memory_gb >= minimum:
            return label
    return None


def choose(
    settings: Settings, fmt: str, budgets: list[tuple[str, float]]
) -> ModelChoice:
    """The model to run. `budgets`: (kind, GB) pairs to try in order, e.g. [("gpu", 8)]."""
    if settings.local_model:  # the user chose
        known = _find_known(settings.local_model)
        tier_context = TIERS[known[0]].context if known else 32768
        file = settings.local_model_file or (known[1].file if known else None)
        return ModelChoice(
            repo=settings.local_model,
            file=file,
            format=fmt,
            context=settings.local_context or tier_context,
            size_gb=known[1].size_gb if known else None,
            reason="chosen in your settings",
        )
    for kind, memory_gb in budgets:
        label = pick_tier(kind, memory_gb)
        if label and fmt in TIERS[label].weights:
            weights = TIERS[label].weights[fmt]
            where = {"gpu": "video memory", "apple_gpu": "memory"}.get(kind, "RAM")
            return ModelChoice(
                repo=weights.repo,
                file=weights.file,
                format=fmt,
                context=settings.local_context or TIERS[label].context,
                size_gb=weights.size_gb,
                reason=f"picked for {memory_gb:g} GB of {where}",
            )
    smallest = next(t for t in TIERS.values() if fmt in t.weights)
    weights = smallest.weights[fmt]
    return ModelChoice(
        weights.repo,
        weights.file,
        fmt,
        settings.local_context or smallest.context,
        weights.size_gb,
        "the smallest model available",
    )


def _find_known(repo: str) -> tuple[str, Weights] | None:
    for label, tier in TIERS.items():
        for weights in tier.weights.values():
            if weights.repo.lower() == repo.lower():
                return label, weights
    return None


# ------------------------------------------------------------------------------ weights


def model_dir(settings: Settings, repo: str) -> Path:
    return settings.data_dir / "models" / repo.split("/")[-1]


def find_weights(settings: Settings, choice: ModelChoice) -> Path | None:
    """The path to load (a GGUF file, or an MLX model folder), or None if not downloaded."""
    folder = model_dir(settings, choice.repo)
    if choice.format == "mlx":
        complete = (folder / "config.json").is_file() and any(folder.glob("*.safetensors"))
        return folder if complete else None
    if choice.file:
        path = folder / choice.file
        return path if path.is_file() else None
    ggufs = sorted(p for p in folder.glob("*.gguf") if not p.name.startswith("mmproj"))
    return ggufs[0] if ggufs else None


def download(settings: Settings, choice: ModelChoice) -> Path:
    """Download (resuming if interrupted) and return the path to load."""
    from huggingface_hub import hf_hub_download, snapshot_download

    folder = model_dir(settings, choice.repo)
    if choice.format == "gguf" and choice.file:
        hf_hub_download(choice.repo, choice.file, local_dir=str(folder))
    elif choice.format == "gguf":
        snapshot_download(
            choice.repo, local_dir=str(folder), allow_patterns=[DEFAULT_GGUF_PATTERN]
        )
    else:
        snapshot_download(choice.repo, local_dir=str(folder))
    path = find_weights(settings, choice)
    if path is None:
        raise FileNotFoundError(
            f"No model weights found in {choice.repo}"
            + ("" if choice.file else f" matching {DEFAULT_GGUF_PATTERN}")
        )
    return path
