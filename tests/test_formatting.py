from ladder.data.formatting import (
    SYSTEM_PROMPT,
    build_problem_prompt,
    extract_code,
    to_chat_example,
)

ROW = {
    "id": "1234A",
    "title": "Sum of Two",
    "time_limit": 2.0,
    "memory_limit": 256.0,
    "description": "Given a and b, print a+b.",
    "input_format": "Two integers a and b.",
    "output_format": "One integer.",
    "interaction_format": None,
    "note": None,
    "examples": [{"input": "1 2", "output": "3"}],
    "generation": "Reasoning...\n```python\nprint(sum(map(int, input().split())))\n```",
    "finish_reason": "stop",
}


def test_prompt_includes_all_present_sections():
    prompt = build_problem_prompt(ROW)
    assert "Sum of Two" in prompt
    assert "Time limit per test: 2 seconds" in prompt
    assert "Memory limit per test: 256 megabytes" in prompt
    assert "## Problem Statement" in prompt
    assert "## Input Format" in prompt
    assert "Example 1" in prompt


def test_prompt_omits_null_sections():
    prompt = build_problem_prompt(ROW)
    assert "Interaction Format" not in prompt
    assert "## Notes" not in prompt


def test_prompt_survives_a_row_with_almost_nothing():
    assert build_problem_prompt({"id": "x"}) == "# Problem"


def test_chat_example_shape():
    example = to_chat_example(ROW)
    roles = [m["role"] for m in example["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert example["messages"][0]["content"] == SYSTEM_PROMPT
    assert example["problem_id"] == "1234A"


def test_chat_example_prefers_messages_field_over_generation():
    row = dict(ROW, messages=[{"role": "assistant", "content": "from the messages field"}])
    assert to_chat_example(row)["messages"][-1]["content"] == "from the messages field"


def test_chat_example_is_none_without_an_assistant_turn():
    assert to_chat_example(dict(ROW, generation="", messages=None)) is None
    assert to_chat_example(dict(ROW, generation="   ", messages=None)) is None


def test_extract_code_takes_the_last_block():
    text = "First try:\n```python\nwrong()\n```\nActually:\n```python\nright()\n```"
    assert extract_code(text) == "right()"


def test_extract_code_handles_unterminated_fence():
    # A response truncated by the token budget still has its answer in there.
    assert extract_code("thinking\n```python\nprint(1)\nprint(2)\n") == "print(1)\nprint(2)"


def test_extract_code_ignores_empty_blocks():
    assert extract_code("```python\nreal()\n```\n```\n\n```") == "real()"


def test_extract_code_returns_none_when_there_is_no_block():
    assert extract_code("I could not solve this problem.") is None
    assert extract_code("") is None
