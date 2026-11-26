"""The tool registry: lookup, schemas, and input validation."""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator

from dragonfish.tools.base import Tool

OUTPUT_LIMIT = 30_000


def truncate(text: str, limit: int = OUTPUT_LIMIT) -> str:
    """Keep the head and tail of long output so neither the command nor its error is lost."""
    if len(text) <= limit:
        return text
    half = limit // 2
    dropped = len(text) - 2 * half
    return f"{text[:half]}\n…[{dropped} characters truncated]…\n{text[-half:]}"


class Registry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        self._validators: dict[str, Draft202012Validator] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        Draft202012Validator.check_schema(tool.input_schema)
        self._tools[tool.name] = tool
        self._validators[tool.name] = Draft202012Validator(tool.input_schema)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """Tool definitions in registration order. A stable order keeps the prompt cache warm."""
        return [tool.schema() for tool in self._tools.values()]

    def validate(self, name: str, tool_input: Any) -> str | None:
        """Return an error message the model can act on, or None if the input is valid."""
        validator = self._validators.get(name)
        if validator is None:
            return f"Unknown tool {name!r}. Available tools: {', '.join(self._tools) or 'none'}."
        if not isinstance(tool_input, dict):
            return f"Tool input must be a JSON object, got {type(tool_input).__name__}."
        errors = sorted(validator.iter_errors(tool_input), key=lambda e: list(e.path))
        if not errors:
            return None
        details = "; ".join(
            f"{'/'.join(map(str, e.path)) or '(root)'}: {e.message}" for e in errors[:5]
        )
        return f"Invalid input for {name}: {details}"
