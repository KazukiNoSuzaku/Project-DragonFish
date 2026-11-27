from __future__ import annotations

from dragonfish.core.events import TextDelta, ToolCallStarted
from dragonfish.providers.qwen_format import (
    MALFORMED,
    RAW_BLOCK,
    QwenStreamParser,
    parse_tool_calls,
    to_qwen_messages,
    to_qwen_tools,
)
from dragonfish.tools import default_tools

TOOLS = [t.schema() for t in default_tools()]

CALL = """<tool_call>
<function=write_file>
<parameter=path>
calc.py
</parameter>
<parameter=content>
def add(a, b):
    return a + b
</parameter>
</function>
</tool_call>"""


def test_parse_single_call_keeps_code_formatting():
    [call] = parse_tool_calls(CALL, TOOLS)
    assert call.name == "write_file"
    assert call.input == {"path": "calc.py", "content": "def add(a, b):\n    return a + b"}


def test_parse_coerces_types_from_schema():
    text = (
        "<tool_call>\n<function=bash>\n<parameter=command>\nsleep 1\n</parameter>\n"
        "<parameter=timeout_s>\n30\n</parameter>\n<parameter=background>\ntrue\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    [call] = parse_tool_calls(text, TOOLS)
    assert call.input == {"command": "sleep 1", "timeout_s": 30, "background": True}


def test_parse_multiple_and_missing_closing_tags():
    text = (
        CALL
        + "\n<tool_call>\n<function=list_dir>\n</function>\n</tool_call>"
        + ("\n<tool_call>\n<function=read_file>\n<parameter=path>\nx.py")
    )
    calls = parse_tool_calls(text, TOOLS)
    assert [c.name for c in calls] == ["write_file", "list_dir", "read_file"]
    assert calls[1].input == {} and calls[2].input == {"path": "x.py"}
    assert len({c.id for c in calls}) == 3


def test_parse_malformed():
    [call] = parse_tool_calls('<tool_call>{"name": "bash"}</tool_call>', TOOLS)
    assert call.name == MALFORMED and "bash" in call.input["raw"]


def test_stream_parser_with_thinking_split_across_chunks():
    p = QwenStreamParser(thinking=True)
    out = "I should write the file.\n</think>\n\nCreating it now." + CALL
    events = []
    for i in range(0, len(out), 3):  # tiny chunks split every tag
        events += p.feed(out[i : i + 3])
    events += p.finish()
    assert p.reasoning.strip() == "I should write the file."
    assert "".join(e.text for e in events if isinstance(e, TextDelta)) == "Creating it now."
    assert [e.name for e in events if isinstance(e, ToolCallStarted)] == ["write_file"]
    assert p.tool_markup.startswith("<tool_call>")
    assert parse_tool_calls(p.tool_markup, TOOLS)[0].name == "write_file"


def test_stream_parser_without_thinking_plain_answer():
    p = QwenStreamParser(thinking=False)
    events = p.feed("  Hello <b>world</b>") + p.finish()
    assert "".join(e.text for e in events) == "Hello <b>world</b>"
    assert p.tool_markup == ""


def test_history_conversion():
    history = [
        {"role": "user", "content": [{"type": "text", "text": "build it"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "ok"},
                {"type": "tool_use", "id": "t1", "name": "list_dir", "input": {}},
                {"type": RAW_BLOCK, "text": "raw output"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "a.py"},
                {"type": "text", "text": "also add tests"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "t2", "name": "bash", "input": {"command": "ls"}}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "t2", "content": "boom", "is_error": True}
            ],
        },
    ]
    msgs = to_qwen_messages("SYS", history)
    assert [m["role"] for m in msgs] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
        "assistant",
        "tool",
    ]
    assert msgs[2] == {"role": "assistant", "content": "raw output"}  # replayed verbatim
    assert msgs[5]["tool_calls"][0]["function"] == {"name": "bash", "arguments": {"command": "ls"}}
    assert msgs[6]["content"] == "Error: boom"


def test_tool_conversion():
    [first, *_] = to_qwen_tools(TOOLS)
    assert first["type"] == "function" and first["function"]["name"] == "read_file"
    assert first["function"]["parameters"]["type"] == "object"


def test_trimming_replaces_oldest_tool_results_only():
    from dragonfish.providers.qwen_format import TRIMMED, count_tool_results

    history = [{"role": "user", "content": [{"type": "text", "text": "go"}]}]
    for i in range(3):
        history.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": f"t{i}", "name": "bash", "input": {"command": "ls"}}
                ],
            }
        )
        history.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": f"t{i}", "content": f"output {i}"}
                ],
            }
        )
    assert count_tool_results(history) == 3
    tool_msgs = [m for m in to_qwen_messages("S", history, trim_first=2) if m["role"] == "tool"]
    assert [m["content"] for m in tool_msgs] == [TRIMMED, TRIMMED, "output 2"]
    assert history[2]["content"][0]["content"] == "output 0"  # stored history untouched
