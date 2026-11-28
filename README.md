# DragonFish

A small, hand-built AI coding agent for your terminal. You describe a task and DragonFish
plans it, writes the files, runs the code and tests, reads the errors, and fixes them.

DragonFish runs an open model (**Qwen**) **entirely on your own computer**, Windows or Mac:
no API key, no account, no internet connection once the model is downloaded, and no cost per
request. It picks a model size that fits your machine, from a 4 GB laptop up to a gaming PC.

> Status: early, work in progress. See [docs/PLAN.md](docs/PLAN.md) for the roadmap.

---

## Requirements

| | Windows | Mac |
|---|---|---|
| Computer | Any 64-bit PC (a GPU is optional, see [Using your GPU](#using-your-gpu-on-windows)) | Apple Silicon (M1 or newer) |
| Operating system | Windows 10 or 11 | macOS 14 (Sonoma) or newer |
| Memory | 4 GB minimum; 8 GB or more for the larger models | 8 GB minimum; 16 GB recommended |
| Disk space | 0.5–3 GB for the model DragonFish picks, plus about 0.3 GB for the program | About 7 GB |
| Tools | [`uv`](https://docs.astral.sh/uv/); [Git for Windows](https://git-scm.com/download/win) recommended | [`uv`](https://docs.astral.sh/uv/) and `git` |

The model runs inside DragonFish: with [llama.cpp](https://github.com/ggml-org/llama.cpp) on
Windows (on the CPU by default) and with [MLX](https://github.com/ml-explore/mlx), Apple's
machine-learning framework, on Apple Silicon Macs. Linux and Intel Macs are not supported yet.

On Windows, DragonFish runs the agent's commands in **Git Bash** (part of Git for Windows).
Without it, it falls back to PowerShell, which works, but small models write bash more reliably.

---

## Setup

### 1. Install `uv`

`uv` manages Python and DragonFish's dependencies. You don't need to install Python yourself:
`uv` downloads Python 3.12 automatically.

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Mac:

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

This installs DragonFish with its local-model support (llama.cpp on Windows, MLX on Apple
Silicon; both prebuilt, so no compiler is needed), and puts a `dragonfish` command in
`~/.local/bin` (`%USERPROFILE%\.local\bin` on Windows) so you can run it from any folder. `--editable` means changes you make to
the code take effect immediately, without reinstalling.

If your terminal then says `dragonfish: command not found`, run `uv tool update-shell` and
open a new terminal window.

### 4. Download the model (first run only)

Start DragonFish once:

```bash
dragonfish --local
```

On the first run it picks a Qwen3.5 model that fits your computer (see
[Choosing a model](#choosing-a-model)), shows its size, and asks to download it from Hugging
Face. Answer `Y`. The download is saved to `~/.dragonfish/models/` and happens only once. If it gets
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

By default DragonFish picks the largest Qwen3.5 model that runs well on your machine:

| Your machine | Model | Download | Context |
|---|---|---|---|
| Windows, up to 6 GB RAM | `unsloth/Qwen3.5-0.8B-GGUF` (Q4_K_M) | 0.5 GB | 16k tokens |
| Windows, 7–10 GB RAM | `unsloth/Qwen3.5-2B-GGUF` | 1.3 GB | 16k |
| Windows, 11 GB RAM or more | `unsloth/Qwen3.5-4B-GGUF` | 2.7 GB | 32k |
| Windows, GPU build and an NVIDIA GPU with 7 GB+ | `unsloth/Qwen3.5-9B-GGUF` | 5.7 GB | 64k |
| Mac, 8 GB | `mlx-community/Qwen3.5-4B-MLX-4bit` | 3 GB | 32k |
| Mac, 16 GB or more | `mlx-community/Qwen3.5-9B-4bit` | 6 GB | 64k |

On the CPU, DragonFish stops at 4B so replies stay quick. To pick a model yourself:

```bash
dragonfish --local --local-model unsloth/Qwen3.5-9B-GGUF            # Windows: a GGUF repo
dragonfish --local --local-model mlx-community/Qwen3.5-4B-MLX-4bit  # Mac: an MLX repo
```

Or set it in `~/.dragonfish/config.toml`:

```toml
local_model = "unsloth/Qwen3.5-2B-GGUF"
local_model_file = "Qwen3.5-2B-Q8_0.gguf"  # optional; the default is the Q4_K_M file
local_context = 16384                      # optional
```

Any model is downloaded on first use, the same way as the default.

### Using your GPU on Windows

The default install runs on the CPU, so it works on every PC. With an NVIDIA GPU, install the
CUDA build instead (for any other GPU, replace `cu124` with `vulkan`):

```powershell
uv tool install --editable ".[local]" --force --reinstall-package llama-cpp-python --index llama-cpp-cpu=https://abetlen.github.io/llama-cpp-python/whl/cu124
```

DragonFish then puts the whole model on the GPU, and picks a bigger model if the GPU has room.

---

## Performance and memory

- **While DragonFish is open** in local mode, the model uses about its download size plus
  0.5–2 GB of memory (about 1 GB for 0.8B, 6–8 GB for 9B). While it is generating, the CPU
  or GPU is busy and other apps may feel slower.
- **When you quit DragonFish**, all of that memory is freed immediately. Nothing keeps
  running in the background: there is no server or background service.
- To make it lighter: use `--effort medium`, a smaller model, or close memory-heavy apps.
- Expect a small local model to be slower and make more mistakes than large cloud models.
  It reads its own errors and retries, so give it a few steps.

---

## How it works

DragonFish does not use Ollama or any model server. The model runs **inside the DragonFish
process**, using llama.cpp (Windows) or MLX (Apple Silicon):

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
  platform/   everything that differs per OS: detection, shells (bash, Git Bash, PowerShell),
              starting and stopping processes
  providers/  model back ends: local_base.py (shared local logic), local_llama.py (llama.cpp),
              local_mlx.py (MLX), local_engines.py and local_models.py (which engine and
              model fit this machine), qwen_format.py (prompt + parsing)
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

On Windows, `~` is your user folder (`C:\Users\<you>`).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `dragonfish: command not found` | Run `uv tool update-shell`, then open a new terminal. |
| `Local models need the local extra` | Reinstall with the extra: `uv tool install --editable ".[local]" --force` |
| Commands run in PowerShell instead of bash (Windows) | Install [Git for Windows](https://git-scm.com/download/win), or set `DRAGONFISH_SHELL` to the full path of your `bash.exe`. |
| `Python was not found; ... Microsoft Store` (Windows) | Install Python from python.org, or run `uv python install`. |
| The download stopped partway | Run `dragonfish --local` again; it resumes. |
| `The conversation is too long for the local model` | Type `/clear` to start a fresh conversation. |
| It is slow, or the Mac feels sluggish | Use `--effort medium` or the 4B model, and close other heavy apps. |
| The pixel banner looks garbled | Use a terminal with true-color support: Windows Terminal, iTerm2, Ghostty, or the VS Code terminal. |

---

## Uninstall

```bash
uv tool uninstall dragonfish
rm -rf ~/.dragonfish          # removes downloaded models, history and settings
```

On Windows (PowerShell), the second line is `Remove-Item -Recurse -Force ~\.dragonfish`.

---

## For developers

```bash
uv sync --extra local               # set up a development environment
uv run pytest                       # run the test suite
uv run ruff check . && uv run mypy dragonfish
uv run python scripts/local_smoke.py   # live end-to-end test with the local model
```
