import pytest

from ladder.config import DataConfig, EvalConfig, RunConfig
from ladder.eval.runner import evaluate, pass_at_k
from ladder.eval.tasks import EvalProblem, collect_tests


def test_collect_tests_reads_the_dict_of_lists_shape():
    row = {"public_tests": {"input": ["1", "2"], "output": ["a", "b"]}}
    assert collect_tests(row) == [("1", "a"), ("2", "b")]


def test_collect_tests_reads_the_list_of_records_shape():
    row = {"public_tests_ms": [{"input": "1", "output": "a"}]}
    assert collect_tests(row) == [("1", "a")]


def test_collect_tests_prefers_official_tests_and_dedupes():
    row = {
        "public_tests": {"input": ["1"], "output": ["a"]},
        "private_tests": {"input": ["1", "2"], "output": ["a", "b"]},
        "generated_tests": {"input": ["3"], "output": ["c"]},
    }
    assert collect_tests(row) == [("1", "a"), ("2", "b"), ("3", "c")]


def test_collect_tests_respects_the_limit():
    row = {"generated_tests": {"input": [str(i) for i in range(50)], "output": ["x"] * 50}}
    assert len(collect_tests(row, limit=5)) == 5


def test_collect_tests_on_a_row_with_none():
    assert collect_tests({"public_tests": None, "private_tests": None}) == []


def test_pass_at_k_matches_the_closed_form():
    assert pass_at_k(1, 1, 1) == 1.0
    assert pass_at_k(1, 0, 1) == 0.0
    assert pass_at_k(10, 10, 5) == 1.0
    assert pass_at_k(10, 0, 5) == 0.0
    # 5 of 10 correct: pass@1 is the hit rate, pass@5 is much higher.
    assert pass_at_k(10, 5, 1) == pytest.approx(0.5)
    assert pass_at_k(10, 5, 5) == pytest.approx(1 - (5 * 4 * 3 * 2 * 1) / (10 * 9 * 8 * 7 * 6))


def test_pass_at_k_rejects_k_larger_than_n():
    with pytest.raises(ValueError):
        pass_at_k(2, 1, 5)


def _cfg(tmp_path, samples=1):
    cfg = RunConfig(
        name="test",
        data=DataConfig(),
        eval=EvalConfig(
            samples_per_problem=samples,
            timeout_seconds=10,
            results_path=str(tmp_path / "results.json"),
        ),
    )
    return cfg


ADD_PROBLEM = EvalProblem(
    problem_id="1A",
    title="Add",
    prompt="add two numbers",
    tests=[("1 2\n", "3\n"), ("4 5\n", "9\n")],
)


def test_evaluate_scores_a_correct_model(tmp_path):
    def generate(prompt, n):
        return ["```python\nprint(sum(map(int, input().split())))\n```"] * n

    summary = evaluate(generate, _cfg(tmp_path), [ADD_PROBLEM])
    assert summary["metrics"]["pass@1"] == 1.0
    assert summary["verdicts"] == {"accepted": 1}


def test_evaluate_scores_a_wrong_model(tmp_path):
    def generate(prompt, n):
        return ["```python\nprint(0)\n```"] * n

    summary = evaluate(generate, _cfg(tmp_path), [ADD_PROBLEM])
    assert summary["metrics"]["pass@1"] == 0.0
    assert summary["verdicts"] == {"wrong_answer": 1}


def test_evaluate_records_a_response_with_no_code(tmp_path):
    summary = evaluate(lambda p, n: ["I give up."] * n, _cfg(tmp_path), [ADD_PROBLEM])
    assert summary["verdicts"] == {"no_code": 1}
    assert summary["problems"][0]["verdicts"] == ["no_code"]


def test_evaluate_writes_results_to_disk(tmp_path):
    import json

    evaluate(lambda p, n: ["```python\nprint(0)\n```"] * n, _cfg(tmp_path), [ADD_PROBLEM])
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["n_problems"] == 1
    assert saved["problems"][0]["problem_id"] == "1A"


def test_evaluate_reports_pass_at_k_for_multiple_samples(tmp_path):
    calls = {"n": 0}

    def generate(prompt, n):
        # One correct sample out of four.
        calls["n"] += 1
        return ["```python\nprint(sum(map(int, input().split())))\n```"] + [
            "```python\nprint(0)\n```"
        ] * (n - 1)

    summary = evaluate(generate, _cfg(tmp_path, samples=4), [ADD_PROBLEM])
    assert summary["metrics"]["pass@1"] == pytest.approx(0.25)
    assert "pass@5" not in summary["metrics"]


def test_evaluate_refuses_an_empty_problem_set(tmp_path):
    with pytest.raises(RuntimeError, match="no eval problems"):
        evaluate(lambda p, n: [""], _cfg(tmp_path), [])


def test_progress_logging_survives_a_console_that_cannot_encode_the_title(tmp_path, capsys):
    # A cp1252 console raises on the first "n ≤ 10^9" in a problem title.
    # Losing a character in a progress line is fine; losing the eval is not.
    problem = EvalProblem(
        problem_id="1B",
        title="Constraints n ≤ 10⁹ and α → β",
        prompt="p",
        tests=[("1\n", "1\n")],
    )
    summary = evaluate(
        lambda p, n: ["```python\nprint(input())\n```"] * n, _cfg(tmp_path), [problem]
    )
    assert summary["metrics"]["pass@1"] == 1.0
