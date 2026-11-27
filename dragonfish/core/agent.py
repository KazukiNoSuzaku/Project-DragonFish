"""The agent loop: ask the model, run the tools it calls, feed results back, repeat."""

from __future__ import annotations

import threading
from collections.abc import Callable, Generator, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from dragonfish.core.config import Settings
from dragonfish.core.events import (
    AgentEvent,
    AgentNotice,
    PermissionAnswer,
    PermissionRequest,
    ToolCallReady,
    ToolResultEvent,
    TurnFinished,
    TurnUsage,
    Usage,
)
from dragonfish.core.permissions import Decision, PermissionGate
from dragonfish.providers.base import (
    Cancelled,
    Completed,
    Provider,
    ProviderError,
    ToolCall,
    ToolInputDelta,
)
from dragonfish.tools.base import Permission, Tool, ToolContext, ToolError, ToolResult
from dragonfish.tools.registry import Registry, truncate

PermissionCallback = Callable[[PermissionRequest], PermissionAnswer]

INTERRUPTED = "Interrupted by the user before this tool ran."
DENIED = "The user denied this action. Do not retry it; ask how they would like to proceed."
TRUNCATED_CALL = (
    "This tool call was cut off because the response hit the output token limit, so it was "
    "not run. Split the work into smaller steps (for example, write a large file in parts)."
)

# Tools that only look at things can safely run at the same time.
_PARALLEL_SAFE = {Permission.READ, Permission.NETWORK}


def _deny_all(_: PermissionRequest) -> PermissionAnswer:
    return "no"


@dataclass
class _Pending:
    call: ToolCall
    tool: Tool | None = None
    result: ToolResult | None = None  # set early when the call is rejected before running


class Agent:
    def __init__(
        self,
        provider: Provider,
        registry: Registry,
        gate: PermissionGate,
        settings: Settings,
        system_prompt: str,
        ask_permission: PermissionCallback = _deny_all,
    ):
        self.provider = provider
        self.registry = registry
        self.gate = gate
        self.settings = settings
        self.system_prompt = system_prompt
        self.ask_permission = ask_permission
        self.messages: list[dict[str, Any]] = []
        self.total_usage = Usage()

    # ------------------------------------------------------------------ public

    def run(self, user_text: str, cancel: threading.Event | None = None) -> Iterator[AgentEvent]:
        """Handle one user message, yielding events until the agent is done or interrupted."""
        cancel = cancel or threading.Event()
        self._append_user_text(user_text)
        tools = self.registry.schemas()

        for _ in range(self.settings.max_iterations):
            try:
                completed = yield from self._stream_one_request(tools, cancel)
            except Cancelled:
                yield AgentNotice("info", "Interrupted.")
                yield TurnFinished("cancelled")
                return
            except ProviderError as exc:
                yield AgentNotice("error", str(exc))
                yield TurnFinished("error")
                return

            # Append the assistant turn exactly as returned: thinking blocks must not be altered.
            self.messages.append({"role": "assistant", "content": completed.content})
            self.total_usage += completed.usage
            yield TurnUsage(completed.usage)

            reason = completed.stop_reason
            calls = completed.tool_calls

            if reason == "pause_turn":  # a server-side tool paused; send the turn back as-is
                continue
            if reason == "refusal":
                yield AgentNotice("warning", "The model declined this request.")
                yield TurnFinished(reason)
                return
            if reason == "max_tokens":
                if not calls:
                    yield AgentNotice("warning", "The reply hit the output token limit.")
                    yield TurnFinished(reason)
                    return
                # A truncated tool input can still parse as valid JSON, so never run it.
                yield AgentNotice("warning", "A tool call was cut off; asking the model to retry.")
                self._append_tool_results(
                    [_result_block(c.id, TRUNCATED_CALL, is_error=True) for c in calls]
                )
                continue
            if not calls:
                yield TurnFinished(reason)
                return

            blocks = yield from self._execute_all(calls, cancel)
            self._append_tool_results(blocks)
            if cancel.is_set():
                yield AgentNotice("info", "Interrupted.")
                yield TurnFinished("cancelled")
                return

        yield AgentNotice(
            "warning",
            f"Stopped after {self.settings.max_iterations} steps. Send a message to continue.",
        )
        yield TurnFinished("max_iterations")

    def reset(self) -> None:
        self.messages.clear()

    # ----------------------------------------------------------------- helpers

    def _stream_one_request(
        self, tools: list[dict[str, Any]], cancel: threading.Event
    ) -> Generator[AgentEvent, None, Completed]:
        completed: Completed | None = None
        stream = self.provider.stream_turn(
            system=self.system_prompt, messages=self.messages, tools=tools, cancel=cancel
        )
        for event in stream:
            if isinstance(event, Completed):
                completed = event
            elif isinstance(event, ToolInputDelta):
                continue  # raw input fragments are not shown
            else:
                yield event
        if completed is None:
            raise ProviderError("The model stream ended without a final message.")
        return completed

    def _append_user_text(self, text: str) -> None:
        block = {"type": "text", "text": text}
        last = self.messages[-1] if self.messages else None
        if last is not None and last["role"] == "user":
            # The previous request was interrupted (or ended on tool results). Merge into that
            # user turn so roles keep alternating and every tool_use keeps its tool_result.
            content = last["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            last["content"] = [*content, block]
        else:
            self.messages.append({"role": "user", "content": [block]})

    def _append_tool_results(self, blocks: list[dict[str, Any]]) -> None:
        # All results for one assistant turn go back in ONE user message.
        self.messages.append({"role": "user", "content": blocks})

    def _execute_all(
        self, calls: list[ToolCall], cancel: threading.Event
    ) -> Generator[AgentEvent, None, list[dict[str, Any]]]:
        pending = [self._prepare(call) for call in calls]

        # 1. Ask for any approvals first, in the order the model made the calls.
        for item in pending:
            if item.result is not None or item.tool is None:
                continue
            decision, reason = self.gate.check(item.tool, item.call.input)
            if decision is Decision.DENY:
                item.result = ToolResult(reason or DENIED, is_error=True)
            elif decision is Decision.ASK:
                if cancel.is_set():
                    item.result = ToolResult(INTERRUPTED, is_error=True)
                    continue
                answer = self.ask_permission(
                    PermissionRequest(item.call.id, item.call.name, item.call.input, reason)
                )
                if answer == "always":
                    self.gate.remember(item.tool.name)
                elif answer != "yes":
                    item.result = ToolResult(DENIED, is_error=True)

        runnable = [p for p in pending if p.result is None]
        for item in runnable:
            yield ToolCallReady(item.call.id, item.call.name, item.call.input)

        # 2. Read-only tools run concurrently; tools with side effects run one by one, in order.
        ctx = ToolContext(self.settings.workspace, cancel)
        parallel = [p for p in runnable if p.tool and p.tool.permission in _PARALLEL_SAFE]
        if parallel and not cancel.is_set():
            with ThreadPoolExecutor(max_workers=min(8, len(parallel))) as pool:
                futures = [(p, pool.submit(_run_tool, p.tool, p.call.input, ctx)) for p in parallel]
                for item, future in futures:
                    item.result = future.result()
        for item in runnable:
            if item.result is not None:
                continue
            if cancel.is_set():
                item.result = ToolResult(INTERRUPTED, is_error=True)
                continue
            assert item.tool is not None
            item.result = _run_tool(item.tool, item.call.input, ctx)

        if any(p.tool and p.tool.permission is Permission.NETWORK for p in runnable):
            self.gate.tainted = True

        blocks = []
        for item in pending:
            result = item.result or ToolResult(INTERRUPTED, is_error=True)
            output = truncate(result.output, self.settings.tool_output_limit)
            yield ToolResultEvent(
                item.call.id, item.call.name, output, result.is_error, result.display
            )
            blocks.append(_result_block(item.call.id, output, result.is_error))
        return blocks

    def _prepare(self, call: ToolCall) -> _Pending:
        tool = self.registry.get(call.name)
        error = self.registry.validate(call.name, call.input)
        if error:
            return _Pending(call, tool, ToolResult(error, is_error=True))
        return _Pending(call, tool)


def _run_tool(tool: Tool | None, tool_input: dict[str, Any], ctx: ToolContext) -> ToolResult:
    assert tool is not None
    try:
        return tool.run(tool_input, ctx)
    except ToolError as exc:  # an expected failure with a message written for the model
        return ToolResult(str(exc), is_error=True)
    except Exception as exc:  # A broken tool must never crash the agent.
        return ToolResult(f"Tool crashed: {type(exc).__name__}: {exc}", is_error=True)


def _result_block(tool_use_id: str, output: str, is_error: bool = False) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "tool_result", "tool_use_id": tool_use_id, "content": output}
    if is_error:
        block["is_error"] = True
    return block
