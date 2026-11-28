"""File tools, all confined to the workspace."""

from __future__ import annotations

import difflib
import fnmatch
import os
import re
from pathlib import Path
from typing import Any

from dragonfish.platform.detect import to_native_path
from dragonfish.tools.base import Permission, Tool, ToolContext, ToolError, ToolResult

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".dragonfish",
    ".idea",
    ".vscode",
    "dist",
    "build",
}
MAX_READ_BYTES = 5_000_000
MAX_LINE_CHARS = 2_000
SKIP_FILES = {".DS_Store", "Thumbs.db", "desktop.ini"}  # OS metadata files


def resolve_in_workspace(workspace: Path, path: str) -> Path:
    """Resolve `path` against the workspace; refuse anything that ends up outside it."""
    root = workspace.resolve()
    candidate = Path(to_native_path(path)).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()  # follows symlinks, collapses ".."
    if resolved != root and not resolved.is_relative_to(root):
        hint = Path(path).name or "file.txt"
        raise ToolError(
            f"Path {path!r} is outside the workspace ({root}). Use a path relative to the "
            f"workspace, for example {hint!r}."
        )
    return resolved


def _rel(workspace: Path, path: Path) -> str:
    try:
        # Forward slashes on every OS: one form for the model, the UI and diffs.
        return path.relative_to(workspace.resolve()).as_posix()
    except ValueError:
        return str(path)


def _read_text(path: Path) -> str:
    if not path.exists():
        raise ToolError(f"File not found: {path.name}. Use list_dir to see what exists.")
    if path.is_dir():
        raise ToolError(f"{path.name} is a directory. Use list_dir instead.")
    if path.stat().st_size > MAX_READ_BYTES:
        raise ToolError(f"{path.name} is too large to read ({path.stat().st_size:,} bytes).")
    data = path.read_bytes()
    if b"\0" in data[:8192]:
        raise ToolError(f"{path.name} looks like a binary file; it cannot be read as text.")
    return data.decode("utf-8", errors="replace")


def make_diff(rel: str, before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
            n=3,
        )
    )


def _diff_result(message: str, rel: str, before: str, after: str) -> ToolResult:
    return ToolResult(message, display={"kind": "diff", "diff": make_diff(rel, before, after)})


# ---------------------------------------------------------------------------------- tools


def read_file(inp: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = resolve_in_workspace(ctx.workspace, inp["path"])
    lines = _read_text(path).splitlines()
    if not lines:
        return ToolResult("(empty file)")
    start = max(1, inp.get("offset", 1))
    limit = inp.get("limit", 2000)
    chunk = lines[start - 1 : start - 1 + limit]
    out = [
        f"{n:>6}\t{line[:MAX_LINE_CHARS]}{'…' if len(line) > MAX_LINE_CHARS else ''}"
        for n, line in enumerate(chunk, start)
    ]
    remaining = len(lines) - (start - 1 + len(chunk))
    if remaining > 0:
        out.append(f"… {remaining} more lines. Read on with offset={start + len(chunk)}.")
    return ToolResult(
        "\n".join(out) if chunk else f"offset {start} is past the end ({len(lines)} lines)."
    )


def write_file(inp: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = resolve_in_workspace(ctx.workspace, inp["path"])
    if path.is_dir():
        raise ToolError(f"{inp['path']} is a directory.")
    rel = _rel(ctx.workspace, path)
    before = _read_text(path) if path.exists() else ""
    existed = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = inp["content"]
    path.write_text(content, encoding="utf-8", newline="")  # exactly these bytes, on any OS
    n = len(content.splitlines())
    verb = "Updated" if existed else "Created"
    return _diff_result(f"{verb} {rel} ({n} lines).", rel, before, content)


def edit_file(inp: dict[str, Any], ctx: ToolContext) -> ToolResult:
    path = resolve_in_workspace(ctx.workspace, inp["path"])
    rel = _rel(ctx.workspace, path)
    old, new = inp["old_string"], inp["new_string"]
    if old == new:
        raise ToolError("old_string and new_string are identical; nothing to change.")
    if not old:
        raise ToolError("old_string is empty. Use write_file to create or replace a whole file.")
    before = _read_text(path)
    count = before.count(old)
    if count == 0 and "\r\n" in before and "\n" in old:
        # The model writes \n; a Windows-style file has \r\n. Match in the file's style.
        old, new = _crlf(old), _crlf(new)
        count = before.count(old)
    if count == 0:
        raise ToolError(
            f"old_string was not found in {rel}. Read the file first and copy the exact text, "
            "including indentation and whitespace."
        )
    if count > 1 and not inp.get("replace_all", False):
        raise ToolError(
            f"old_string appears {count} times in {rel}. Include more surrounding lines to make "
            "it unique, or set replace_all to true."
        )
    after = before.replace(old, new)
    path.write_text(after, encoding="utf-8", newline="")
    return _diff_result(
        f"Edited {rel} ({count} replacement{'s' * (count > 1)}).", rel, before, after
    )


def _crlf(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\n", "\r\n")


def list_dir(inp: dict[str, Any], ctx: ToolContext) -> ToolResult:
    root = resolve_in_workspace(ctx.workspace, inp.get("path", "."))
    if not root.is_dir():
        raise ToolError(f"{inp.get('path')} is not a directory.")
    depth = inp.get("depth", 2)
    lines: list[str] = [f"{_rel(ctx.workspace, root)}/"]
    limit = 500

    def walk(folder: Path, level: int, prefix: str) -> None:
        try:
            entries = sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return
        entries = [e for e in entries if e.name not in SKIP_DIRS and e.name not in SKIP_FILES]
        for entry in entries:
            if len(lines) >= limit:
                return
            if entry.is_dir():
                lines.append(f"{prefix}{entry.name}/")
                if level < depth:
                    walk(entry, level + 1, prefix + "  ")
            else:
                lines.append(f"{prefix}{entry.name}")

    walk(root, 1, "  ")
    if len(lines) >= limit:
        lines.append(f"… stopped at {limit} entries; list a subfolder for more.")
    if len(lines) == 1:
        lines.append("  (empty)")
    return ToolResult("\n".join(lines))


def grep(inp: dict[str, Any], ctx: ToolContext) -> ToolResult:
    root = resolve_in_workspace(ctx.workspace, inp.get("path", "."))
    try:
        regex = re.compile(inp["pattern"])
    except re.error as exc:
        raise ToolError(f"Invalid regular expression: {exc}") from exc
    glob = inp.get("glob")
    hits: list[str] = []
    limit = 200
    files = [root] if root.is_file() else _walk_files(root)
    for file in files:
        if glob and not fnmatch.fnmatch(file.name, glob):
            continue
        try:
            if file.stat().st_size > 2_000_000:
                continue
            data = file.read_bytes()
        except OSError:
            continue
        if b"\0" in data[:8192]:
            continue
        for n, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
            if regex.search(line):
                hits.append(f"{_rel(ctx.workspace, file)}:{n}: {line.strip()[:300]}")
                if len(hits) >= limit:
                    hits.append(f"… stopped at {limit} matches; narrow the pattern or path.")
                    return ToolResult("\n".join(hits))
    return ToolResult("\n".join(hits) if hits else "No matches.")


def _walk_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        out.extend(Path(dirpath) / f for f in sorted(filenames))
    return out


# ----------------------------------------------------------------------------- schemas


def _schema(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


PATH = {"type": "string", "description": "Path relative to the workspace."}

FS_TOOLS = [
    Tool(
        "read_file",
        "Read a text file. Returns numbered lines. For large files, page with offset and limit.",
        _schema(
            {
                "path": PATH,
                "offset": {"type": "integer", "minimum": 1, "description": "First line (1-based)."},
                "limit": {"type": "integer", "minimum": 1, "maximum": 5000},
            },
            ["path"],
        ),
        Permission.READ,
        read_file,
    ),
    Tool(
        "write_file",
        "Create a file, or replace a file's entire contents. Creates parent folders. "
        "For small changes to an existing file, prefer edit_file.",
        _schema({"path": PATH, "content": {"type": "string"}}, ["path", "content"]),
        Permission.WRITE,
        write_file,
    ),
    Tool(
        "edit_file",
        "Replace exact text in a file. old_string must match the file exactly (including "
        "indentation) and be unique unless replace_all is true. Read the file first.",
        _schema(
            {
                "path": PATH,
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            ["path", "old_string", "new_string"],
        ),
        Permission.WRITE,
        edit_file,
    ),
    Tool(
        "list_dir",
        "List files and folders as a tree (skips .git, .venv, caches).",
        _schema(
            {"path": PATH, "depth": {"type": "integer", "minimum": 1, "maximum": 5}},
            [],
        ),
        Permission.READ,
        list_dir,
    ),
    Tool(
        "grep",
        "Search file contents with a regular expression. Returns file:line: text.",
        _schema(
            {
                "pattern": {"type": "string", "description": "Python regular expression."},
                "path": PATH,
                "glob": {"type": "string", "description": "Filename filter, e.g. *.py"},
            },
            ["pattern"],
        ),
        Permission.READ,
        grep,
    ),
]
