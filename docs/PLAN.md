# DragonFish — Project Plan

> A small, hand-built AI coding agent for the terminal. It can control the computer (shell + files),
> surf the web, and build working software from a single prompt — shown off by building a Python
> calculator end to end.

## 1. Key decisions

| Decision | Choice | Why |
|---|---|---|
| Language | **Python 3.12** (managed with `uv`) | Fastest path to a polished terminal UI. The `anthropic` SDK needs Python 3.10+. |
| LLM layer | **Raw `anthropic` SDK + a hand-written agent loop** (not the Claude Agent SDK) | The point of the project is owning the loop: tool dispatch, context, permissions. It also makes swapping in another model API a one-file change. |
| Provider abstraction | `Provider` interface → `AnthropicProvider` first | A future API is added as one new class; nothing else changes. |
| Model | `claude-opus-5-5`, `effort: high`, adaptive thinking | Opus 5.5 defaults to `medium` effort, so it is set explicitly. |
| Web tools | **Own client-side tools** (search + fetch), Anthropic server tools as an optional toggle | Server tools disappear when switching providers; own tools are portable and more impressive. |
| Terminal UI | **Rich** (rendering) + **prompt_toolkit** (input) | Streaming markdown, spinners, syntax-highlighted diffs, styled prompt. |

## 2. Architecture

```
┌──────────── DragonFish CLI (Rich + prompt_toolkit) ────────────┐
│  pixel banner · streaming output · tool cards · approve y/n    │
└───────────────┬────────────────────────────────────────────────┘
                │ user message / events
        ┌───────▼────────┐      ┌───────────────────────────┐
        │  Agent Loop    │◄────►│ Provider (interface)      │
        │  (core/agent)  │      │  ├ AnthropicProvider      │
        │  while tool_use│      │  └ YourAPIProvider (later)│
        └───────┬────────┘      └───────────────────────────┘
                │ tool calls
        ┌───────▼─────────────────────────────────────────┐
        │ Tool Registry  (schema + permission level + fn) │
        │  fs: read/write/edit/list/grep                  │
        │  shell: bash (sandboxed to workspace, timeout)  │
        │  web: web_search, web_fetch, (browser)          │
        │  computer: screenshot/click/type (optional)     │
        └─────────────────────────────────────────────────┘
                │
        Permission gate · Trace logger (JSONL) · Token/cost meter
```

**Loop:** send messages + tools → stream response → execute every `tool_use` block → return **all**
results in one user message → repeat until `stop_reason == "end_turn"`. Also handle `max_tokens`,
`refusal`, and `pause_turn`.

## 3. Tools

**Computer control — shell + files (core, Phases 1–2)**
- `bash` — run commands in the workspace, with timeout and output truncation
- `read_file`, `write_file`, `edit_file` (string replace + diff preview), `list_dir`, `grep`

**Web surfing (Phase 3)**
- `web_search` — DuckDuckGo, no API key needed
- `web_fetch` — `httpx` + `trafilatura` readable-text extraction
- Stretch: `browser` via Playwright (open / click / read)

**GUI computer use (Phase 5, optional, behind `--computer` flag, off by default)**
- Claude's `computer_toolset_20260801` (the only computer-use version Opus 5.5 accepts on the Claude API);
  screenshots/clicks executed locally via `mss` + `pyautogui`.

## 4. Terminal look — original pixel style

- Startup banner: a **pixel-art dragonfish** drawn with half-block characters (`▀ ▄ █`) in truecolor,
  "DRAGONFISH" in a blocky pixel font, model, cwd, tips. Own sprite and palette — **deep teal + ember orange**.
- Streaming assistant text rendered as markdown.
- Tool cards: `◆ bash(python calc.py)` with bordered, truncated output.
- Approval prompt: `Allow? [y]es / [n]o / [a]lways` for writes and shell.
- Spinner status line ("Swimming…", "Breathing fire…") with elapsed time + tokens.
- Status bar: model · tokens · $ cost · context %.
- Slash commands: `/help /clear /model /cost /tools /yolo /save /exit`. Ctrl+C interrupts a turn, not the app.

## 5. Repo layout

```
dragonfish/
  __main__.py               # `dragonfish` entrypoint
  cli/        app.py banner.py render.py prompt.py commands.py
  core/       agent.py context.py permissions.py trace.py config.py
  providers/  base.py anthropic_provider.py
  tools/      base.py registry.py fs.py shell.py web.py browser.py computer.py
  prompts/    system.md
evals/        tasks/*.yaml  run_evals.py
tests/        unit tests for tools + loop (mock provider)
pyproject.toml  README.md  .env.example  demo.gif
```

## 6. Milestones

| Phase | Deliverable | Done when |
|---|---|---|
| **0** | Scaffold: `uv`, `pyproject`, `.env`, `dragonfish` command | `dragonfish --version` works |
| **1** | Provider + agent loop + streaming, tested with a mock provider | Plain chat works in the terminal |
| **2** | Shell + file tools, permission prompts, workspace sandbox | **Calculator demo passes** (§7) |
| **3** | Web tools | "Look up the latest Python version and put it in the README" works |
| **4** | Pixel UI polish, slash commands, cost meter, session save | Demo-GIF ready |
| **5** | Evals, trace replay, Playwright browser, optional GUI computer use | Published success-rate numbers |

## 7. Showcase acceptance test — the calculator

Prompt: *"Build a working Python calculator with a GUI and tests."*

DragonFish must:
1. Plan the work.
2. Write `calculator/` — Tkinter GUI + a safe expression parser (no `eval()`).
3. Write `test_calculator.py`.
4. Run `pytest`, read failures, fix, re-run until green.
5. Launch the app.

The self-correction (fail → fix → re-run) is the money shot for the demo recording.

## 8. Owner to-dos

1. **Anthropic API key** from platform.claude.com, with credits ($10–20 covers dev + demos). Put it in `.env`.
2. **Truecolor terminal** — iTerm2, Ghostty, or the VS Code terminal (Terminal.app renders pixel art poorly).
3. **Public GitHub repo** with a strong README when ready.
4. *(Phase 5 GUI control only)* grant the terminal Accessibility + Screen Recording in
   System Settings → Privacy & Security.
5. *(Later, own API)* provide its request/response format, tool-calling support, and streaming support.
6. **Record the demo** with `vhs` or `asciinema`; put the GIF at the top of the README.

## 9. Concepts to know (interview material)

- **Agent loop & `stop_reason`** — `tool_use` → keep looping, `end_turn` → done, `max_tokens` → truncated, `refusal` → declined.
- **Parallel tool calls** — one response can hold several `tool_use` blocks; all results go back in one message; errors use `is_error: true`.
- **Thinking blocks** — Opus 5.5 always thinks; its thinking blocks must be passed back unchanged. Treat history as append-only.
- **Prompt caching** — keep the system prompt + tool list byte-identical across calls for big cost savings; surface cache hits in the status bar.
- **Context management** — truncate large tool outputs; use compaction or a summarizer for long sessions.
- **Safety** — workspace jail, dangerous-command denylist, human approval. **Prompt injection**: fetched web pages may contain instructions; treat tool output as data and keep approval on for shell after web fetches.
- **Evals** — measure task success rate, turns, cost, and time per task.

## 10. CV boosters

- `evals/` with 10–15 tasks and a results table in the README ("solves 12/15 at $0.31 avg").
- JSONL trace logs + `dragonfish replay <session>`.
- Model-swap story: same evals on Claude vs. your own API, compared side by side.
- README with architecture diagram and a "design decisions" section.
- Tests, type hints, `ruff`, GitHub Actions CI badge.

## 11. Resolved defaults

- "Computer control" = shell + files now; GUI mouse/keyboard control is optional Phase 5.
- Calculator = Tkinter GUI + tested pure-Python core.
- Palette = deep teal + ember orange.
