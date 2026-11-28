"""Render a model's own Jinja chat template, the way Hugging Face's apply_chat_template does.

GGUF files carry the template as metadata (`tokenizer.chat_template`), so engines without a
Hugging Face tokenizer can still build exactly the prompt the model was trained on.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from typing import Any


class ChatTemplateError(Exception):
    """The template is missing or raised an error while rendering."""


def compile_template(source: str | None) -> Callable[..., str]:
    if not source:
        raise ChatTemplateError("The model file has no chat template.")
    import jinja2
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message: str) -> None:
        raise ChatTemplateError(message)

    def tojson(
        value: Any,
        indent: int | None = None,
        separators: tuple[str, str] | None = None,
        sort_keys: bool = False,
        ensure_ascii: bool = False,
    ) -> str:
        return json.dumps(
            value,
            indent=indent,
            separators=separators,
            sort_keys=sort_keys,
            ensure_ascii=ensure_ascii,
        )

    # The same environment settings Hugging Face transformers uses for chat templates.
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
    )
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
    try:
        template = env.from_string(source)
    except jinja2.TemplateError as exc:
        raise ChatTemplateError(f"Invalid chat template: {exc}") from exc

    def render(**variables: Any) -> str:
        try:
            return template.render(**variables)
        except jinja2.TemplateError as exc:
            raise ChatTemplateError(str(exc)) from exc

    return render
