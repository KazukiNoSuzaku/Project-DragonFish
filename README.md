# DragonFish

A small, hand-built AI coding agent for the terminal. It controls the computer (shell + files),
and builds working software from a single prompt. It runs on **Claude** through the API, or on a
**local open model (Qwen) entirely offline** on Apple Silicon.

> Work in progress. See [docs/PLAN.md](docs/PLAN.md) and [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md).

## Quick start

```bash
uv sync
uv run dragonfish --demo          # try the interface offline with a scripted model
```

### With Claude

```bash
cp .env.example .env              # then set ANTHROPIC_API_KEY in .env
uv run dragonfish                 # interactive
uv run dragonfish -p "hello"      # one prompt, headless
```

### With a local model (Apple Silicon, no API key, $0)

```bash
uv sync --extra local             # installs MLX
uv run dragonfish --local         # first run offers to download Qwen3.5-9B (≈6 GB)
uv run dragonfish --local --effort medium   # faster: turns off Qwen's thinking
```

The model runs inside the DragonFish process with [MLX](https://github.com/ml-explore/mlx);
no Ollama or server is needed. DragonFish converts its conversation to Qwen's chat format,
parses Qwen's XML tool calls itself, and reuses the KV cache between steps of a tool loop.

## Tools

| Tool | What it does | Needs approval (default mode) |
|---|---|---|
| `read_file`, `list_dir`, `grep` | inspect the workspace | no |
| `write_file`, `edit_file` | change files (diff shown before approval) | yes |
| `bash` | run commands; `background: true` for GUIs/servers | yes |

Safety: every path is confined to the workspace; dangerous commands (`sudo`, `rm -rf ~`,
`curl … | sh`, …) are refused in every mode; permission modes are `ask`, `auto-edit`, `yolo`.

## Development

```bash
uv run pytest
uv run ruff check .
uv run mypy dragonfish
uv run python scripts/smoke_test.py   # live test against Claude (needs a key)
```
