# DragonFish

A small, hand-built AI coding agent for your terminal. You describe a task and DragonFish
plans it, writes the files, runs the code and tests, reads the errors, and fixes them.

DragonFish runs an open model (**Qwen**) **entirely on your own Mac**: no API key, no account,
no internet connection once the model is downloaded, and no cost per request.

> Status: early, work in progress. See [docs/PLAN.md](docs/PLAN.md) for the roadmap.

---

## Requirements

| | Needed |
|---|---|
| Computer | A Mac with Apple Silicon (M1 or newer) |
| Operating system | macOS 14 (Sonoma) or newer |
| Memory | 16 GB recommended for the default model; 8 GB works with the smaller model (see below) |
| Disk space | About 7 GB free (6 GB for the default model) |
| Tools | [`uv`](https://docs.astral.sh/uv/) (installs Python for you) and `git` |

The local model runs on [MLX](https://github.com/ml-explore/mlx), Apple's machine-learning
framework, so it does not work on Intel Macs, Windows or Linux yet.

---

## Setup

### 1. Install `uv`

`uv` manages Python and DragonFish's dependencies. You don't need to install Python yourself:
`uv` downloads Python 3.12 automatically.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or, with Homebrew: `brew install uv`. Open a new terminal window afterwards.

### 2. Get the code

```bash
git clone https://github.com/KazukiNoSuzaku/Project-DragonFish.git
cd Project-DragonFish
```

### 3. Install the `dragonfish` command

```bash
uv tool install --editable ".[local]"
```

This installs DragonFish with its local-model support, and puts a `dragonfish` command in
`~/.local/bin` so you can run it from any folder. `--editable` means changes you make to
the code take effect immediately, without reinstalling.

If your terminal then says `dragonfish: command not found`, run `uv tool update-shell` and
open a new terminal window.

### 4. Download the model (first run only)

Start DragonFish once:

```bash
dragonfish --local
```

On the first run it asks to download **Qwen3.5-9B** (about 6 GB) from Hugging Face. Answer
`Y`. The download is saved to `~/.dragonfish/models/` and happens only once. If it gets
interrupted, run the same command again and it resumes.

When the download finishes, the model loads (a few seconds) and the DragonFish prompt `❯` appears.

---

## Using DragonFish

DragonFish works on **the folder you start it in**. Make or open a project folder and start
it there:

```bash
mkdir ~/my-calculator && cd ~/my-calculator
dragonfish --local
```

Then type a task, for example:

```
❯ Build a Python calculator with a Tkinter GUI and pytest tests, then run the tests.
```

By default DragonFish **asks before it changes a file or runs a command**, and shows you the
exact change first. Answer `y` (yes), `n` (no) or `a` (always allow this tool for the session).

### Command-line options

| Option | What it does |
|---|---|
| `--local` | Run the local Qwen model (needed for now) |
| `--effort medium` | Faster replies: turns off the model's step-by-step thinking |
| `--local-model <id>` | Use a different MLX model from Hugging Face (see below) |
| `--mode ask` / `auto-edit` / `yolo` | How much to ask before acting (see below) |
| `--workspace <folder>` | Work on a different folder without `cd`-ing into it |
| `-p "<task>"` | Run one task without the interactive prompt, then exit |
| `--demo` | Try the interface with a scripted fake model (no download needed) |
| `--version`, `--help` | Version and full option list |

### Inside DragonFish

| Type | To |
|---|---|
| `/help` | list commands |
| `/clear` | start a fresh conversation |
| `/mode auto-edit` | change the permission mode |
| `/tools` | list the tools the agent can use |
| `/cost` | show token usage for the session |
| `/exit` or Ctrl+D | quit |
| Ctrl+C | stop the current task (press again at an empty prompt to quit) |
| Esc, then Enter | new line in your message |

### Permission modes

| Mode | Reading files | Changing files | Running commands |
|---|---|---|---|
| `ask` (default) | allowed | asks you | asks you |
| `auto-edit` | allowed | allowed | asks you |
| `yolo` | allowed | allowed | allowed |

Use `yolo` only in a throwaway folder.

### Safety

- The agent can only read and write files **inside the folder it was started in**.
- Dangerous commands are refused in every mode, including `yolo`: for example `sudo`,
  deleting your home folder, formatting disks, or piping a download into a shell.
- Commands run in a fresh shell in your project folder, with a time limit.

---

## Choosing a model

| Model | Download | Best for |
|---|---|---|
| `mlx-community/Qwen3.5-9B-4bit` (default) | 6 GB | 16 GB Macs; best results |
| `mlx-community/Qwen3.5-4B-MLX-4bit` | 3 GB | 8 GB Macs, or when you want it lighter and faster |

```bash
dragonfish --local --local-model mlx-community/Qwen3.5-4B-MLX-4bit
```

Any model is downloaded on first use, the same way as the default.

---

## Performance and memory

- **While DragonFish is open** in local mode, the model uses about 6–8 GB of memory (about
  3–4 GB with the 4B model). While it is generating, your Mac's GPU is busy and other apps
  may feel slower, especially on a 16 GB Mac with many apps open.
- **When you quit DragonFish**, all of that memory is freed immediately. Nothing keeps
  running in the background: there is no server or background service.
- To make it lighter: use `--effort medium`, use the 4B model, or close memory-heavy apps.
- Expect a small local model to be slower and make more mistakes than large cloud models.
  It reads its own errors and retries, so give it a few steps.

---

## How it works

DragonFish does not use Ollama or any model server. The model runs **inside the DragonFish
process** using MLX:

1. The conversation is converted to Qwen's chat format, including the list of tools.
2. Qwen generates a reply token by token, streamed to your screen.
3. DragonFish parses Qwen's tool calls (written as XML, such as `<function=bash>…`) and runs them.
4. The results go back to the model, and the loop repeats until the task is done.

Within a task, DragonFish reuses the model's cache of the conversation, so each step only
processes the new text. When the conversation gets long, older tool output is trimmed to make room.

The code is organised as:

```
dragonfish/
  core/       the agent loop, permissions, settings, events
  providers/  model back ends: local_mlx.py (Qwen via MLX), qwen_format.py (prompt + parsing)
  tools/      file tools (fs.py) and the shell tool (shell.py)
  cli/        the terminal interface: banner, rendering, prompt, slash commands
```

---

## Where DragonFish keeps its files

| Path | Contents |
|---|---|
| `~/.dragonfish/models/` | downloaded models |
| `~/.dragonfish/history` | your prompt history |
| `~/.dragonfish/config.toml` | optional default settings (for example `effort = "medium"`) |

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `dragonfish: command not found` | Run `uv tool update-shell`, then open a new terminal. |
| `Local models need the MLX extra` | Reinstall with the extra: `uv tool install --editable ".[local]" --force` |
| The download stopped partway | Run `dragonfish --local` again; it resumes. |
| `The conversation is too long for the local model` | Type `/clear` to start a fresh conversation. |
| It is slow, or the Mac feels sluggish | Use `--effort medium` or the 4B model, and close other heavy apps. |
| The pixel banner looks garbled | Use a terminal with true-color support: iTerm2, Ghostty, or the VS Code terminal. |

---

## Uninstall

```bash
uv tool uninstall dragonfish
rm -rf ~/.dragonfish          # removes downloaded models, history and settings
```

---

## For developers

```bash
uv sync --extra local               # set up a development environment
uv run pytest                       # run the test suite
uv run ruff check . && uv run mypy dragonfish
uv run python scripts/local_smoke.py   # live end-to-end test with the local model
```
