"""Turning raw Codeforces rows into chat examples, and reading code back out.

The prompt format here is the contract between training and eval: `ladder eval`
builds prompts with the exact same function, so a formatting change can never
silently create a train/eval mismatch.
"""

from __future__ import annotations

import re
from typing import Any

SYSTEM_PROMPT = (
    "You are an expert competitive programmer. You will be given a problem statement, "
    "test case constraints and example test inputs and outputs. Reason about the "
    "problem, settle on an approach that fits the constraints, then give a complete "
    "solution.\n\n"
    "Put your final solution in a single Python code block that reads from standard "
    "input and writes to standard output. Do not print anything except the required "
    "output."
)

_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)(?:```|\Z)", re.DOTALL)


def _fmt_limit(value: Any, unit: str) -> str | None:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    text = f"{num:.1f}".rstrip("0").rstrip(".")
    return f"{text} {unit}"


def build_problem_prompt(row: dict[str, Any]) -> str:
    """Render one dataset row into the user turn of a chat example.

    Sections that are null for a given problem (interaction format, notes) are
    dropped entirely rather than emitted with an empty body, so the model does
    not learn that a heading can be followed by nothing.
    """
    parts: list[str] = []

    title = row.get("title")
    parts.append(f"# Problem\n\n{title}" if title else "# Problem")

    limits = []
    if (tl := _fmt_limit(row.get("time_limit"), "seconds")) is not None:
        limits.append(f"Time limit per test: {tl}")
    if (ml := _fmt_limit(row.get("memory_limit"), "megabytes")) is not None:
        limits.append(f"Memory limit per test: {ml}")
    if limits:
        parts.append("\n".join(limits))

    for heading, key in (
        ("## Problem Statement", "description"),
        ("## Input Format", "input_format"),
        ("## Output Format", "output_format"),
        ("## Interaction Format", "interaction_format"),
        ("## Notes", "note"),
    ):
        value = row.get(key)
        if value:
            parts.append(f"{heading}\n\n{value.strip()}")

    examples = row.get("examples") or []
    if examples:
        rendered = []
        for i, ex in enumerate(examples, start=1):
            inp = (ex.get("input") or "").rstrip("\n")
            out = (ex.get("output") or "").rstrip("\n")
            rendered.append(f"Example {i}\n\nInput:\n```\n{inp}\n```\n\nOutput:\n```\n{out}\n```")
        parts.append("## Examples\n\n" + "\n\n".join(rendered))

    return "\n\n".join(parts).strip()


def to_chat_example(row: dict[str, Any]) -> dict[str, Any] | None:
    """Build a `{"messages": [...]}` example, or None if the row is unusable.

    Prefers the dataset's own `messages` field when present; otherwise assembles
    the turns from `prompt`/`generation`. Either way the system prompt is ours,
    so every example in the mix shares one instruction.
    """
    user = build_problem_prompt(row)
    if not user:
        return None

    assistant = None
    messages = row.get("messages")
    if messages:
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                assistant = msg["content"]
                break
    if assistant is None:
        assistant = row.get("generation")
    if not assistant or not assistant.strip():
        return None

    return {
        "problem_id": row.get("id"),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant.strip()},
        ],
    }


def extract_code(text: str) -> str | None:
    """Pull the submittable program out of a model response.

    Takes the *last* fenced block: reasoning traces routinely sketch a wrong or
    partial approach in an earlier block before committing to the final one. An
    unterminated trailing fence still counts, since a response cut off by the
    token budget usually has its real answer in there.
    """
    if not text:
        return None
    blocks = [m.group(1) for m in _CODE_BLOCK.finditer(text)]
    blocks = [b for b in blocks if b.strip()]
    if not blocks:
        return None
    return blocks[-1].strip("\n")
