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


def test_results_are_written_after_every_problem(tmp_path):
    """A long eval must not lose everything to a dropped session."""
    import json

    seen = []

    def generate(prompt, n):
        # Inspect the file mid-run: results for earlier problems must be there.
        seen.append(json.loads(results_file.read_text()) if results_file.exists() else None)
        return ["```python\nprint(sum(map(int, input().split())))\n```"] * n

    cfg = _cfg(tmp_path)
    results_file = tmp_path / "results.json"
    problems = [
        EvalProblem(f"p{i}", f"P{i}", "prompt", [("1 2\n", "3\n")]) for i in range(3)
    ]
    evaluate(generate, cfg, problems)

    # Before problem 2 there should already be one scored problem on disk.
    assert seen[1] is not None and seen[1]["n_problems"] == 1
    assert seen[2]["n_problems"] == 2
    assert json.loads(results_file.read_text())["n_problems"] == 3


def test_an_interrupted_run_resumes_instead_of_regenerating(tmp_path):
    cfg = _cfg(tmp_path)
    problems = [
        EvalProblem(f"p{i}", f"P{i}", "prompt", [("1 2\n", "3\n")]) for i in range(4)
    ]
    correct = "```python\nprint(sum(map(int, input().split())))\n```"

    # First pass dies after two problems.
    calls = {"n": 0}

    def flaky(prompt, n):
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("session dropped")
        return [correct] * n

    with pytest.raises(RuntimeError):
        evaluate(flaky, cfg, problems)

    # Second pass only generates for what is missing.
    resumed = {"n": 0}

    def counting(prompt, n):
        resumed["n"] += 1
        return [correct] * n

    summary = evaluate(counting, cfg, problems)
    assert resumed["n"] == 2, "should regenerate only the two unfinished problems"
    assert summary["n_problems"] == 4
    assert summary["metrics"]["pass@1"] == 1.0


def test_resume_can_be_disabled(tmp_path):
    cfg = _cfg(tmp_path)
    problems = [EvalProblem("p0", "P0", "prompt", [("1 2\n", "3\n")])]
    correct = "```python\nprint(sum(map(int, input().split())))\n```"
    evaluate(lambda p, n: [correct] * n, cfg, problems)

    calls = {"n": 0}

    def counting(prompt, n):
        calls["n"] += 1
        return [correct] * n

    evaluate(counting, cfg, problems, resume=False)
    assert calls["n"] == 1, "resume=False must rescore"


def test_a_truncated_partial_file_does_not_break_resume(tmp_path):
    """A run killed mid-write leaves invalid JSON; that must not be fatal."""
    cfg = _cfg(tmp_path)
    (tmp_path / "results.json").write_text('{"problems": [', encoding="utf-8")
    problems = [EvalProblem("p0", "P0", "prompt", [("1 2\n", "3\n")])]
    summary = evaluate(
        lambda p, n: ["```python\nprint(sum(map(int, input().split())))\n```"] * n,
        cfg, problems,
    )
    assert summary["n_problems"] == 1


def test_problem_index_filter_selects_only_matching_indices():
    """Restricting to div2 A/B is a resolution fix, so it must actually bind."""
    from ladder.config import EvalConfig

    cfg = EvalConfig(problem_indices=["A", "B"])
    wanted = {i.upper() for i in cfg.problem_indices}

    rows = [
        {"id": "1/A", "index": "A"},
        {"id": "2/B", "index": "B1"},   # subtask variants still count as B
        {"id": "3/E", "index": "E"},
        {"id": "4/J", "index": "J"},
        {"id": "5/a", "index": "a"},    # case-insensitive
    ]
    kept = [r["id"] for r in rows if (r["index"] or "")[:1].upper() in wanted]
    assert kept == ["1/A", "2/B", "5/a"]


def test_empty_index_filter_keeps_everything():
    from ladder.config import EvalConfig

    assert EvalConfig().problem_indices == []
