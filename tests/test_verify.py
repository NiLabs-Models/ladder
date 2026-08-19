from ladder.config import DataConfig
from ladder.data.filters import FilterStats
from ladder.data.verify import check_pair, verify_pairs

CORRECT = "```python\nprint(sum(map(int, input().split())))\n```"
WRONG = "```python\nprint(0)\n```"

ROW = {
    "id": "1A",
    "problem_type": "diff",
    "public_tests": {"input": ["1 2\n", "4 5\n"], "output": ["3\n", "9\n"]},
}


def pair(trace, **row_extra):
    row = {**ROW, **row_extra}
    return row, {"problem_id": row["id"], "messages": [{"role": "assistant", "content": trace}]}


CFG = DataConfig(verify_solutions=True, verify_max_tests=5, verify_timeout=10.0)


def test_a_correct_solution_is_kept():
    assert check_pair(pair(CORRECT), CFG) == ""


def test_a_wrong_solution_is_dropped_with_its_verdict():
    assert check_pair(pair(WRONG), CFG) == "failed_verification:wrong_answer"


def test_a_trace_with_no_code_is_dropped():
    assert check_pair(pair("I could not solve this."), CFG) == "no_code_in_trace"


def test_a_crashing_solution_is_dropped():
    reason = check_pair(pair("```python\nraise SystemError\n```"), CFG)
    assert reason.startswith("failed_verification:")


def test_checker_problems_are_kept_not_failed():
    # Many valid outputs -- failing these would discard correct traces for
    # being right in a different way than the expected-output file.
    assert check_pair(pair(WRONG, problem_type="checker"), CFG) == ""


def test_checker_problems_can_be_dropped_instead():
    cfg = DataConfig(verify_solutions=True, verify_keep_unverifiable=False, verify_timeout=10.0)
    assert check_pair(pair(WRONG, problem_type="checker"), cfg) == "unverifiable_problem_type"


def test_problems_without_tests_are_kept_by_default():
    assert check_pair(pair(CORRECT, public_tests=None), CFG) == ""


def test_verify_pairs_filters_a_stream_and_records_why():
    stats = FilterStats()
    pairs = [pair(CORRECT), pair(WRONG), pair(CORRECT), pair("no code here")]
    kept = list(verify_pairs(pairs, CFG, stats))
    assert len(kept) == 2
    assert stats.dropped["failed_verification:wrong_answer"] == 1
    assert stats.dropped["no_code_in_trace"] == 1


def test_verify_pairs_on_an_empty_stream():
    assert list(verify_pairs([], CFG, FilterStats())) == []


def test_threaded_verification_does_not_stall():
    """Verification under threads must not deadlock or serialize pathologically.

    This is the regression that cost a 12-hour Kaggle GPU session. The judge
    applied rlimits through `preexec_fn`, which runs between fork and exec and
    is documented as unsafe in the presence of threads -- and this code path
    calls it from a ThreadPoolExecutor. On Linux it turned 0.2s/trace into
    93s/trace; locally it was invisible because the preexec_fn was only
    installed on POSIX, so Windows never ran it.

    The assertion is a wall-clock budget, deliberately loose enough not to flake
    on a loaded CI box but far tighter than a stall.
    """
    import time

    n = 24
    pairs = [pair(CORRECT) for _ in range(n)]
    cfg = DataConfig(
        verify_solutions=True,
        verify_max_tests=2,
        verify_timeout=5.0,
        verify_workers=8,
        dedup_by_problem=False,
    )

    started = time.monotonic()
    kept = list(verify_pairs(pairs, cfg, FilterStats()))
    elapsed = time.monotonic() - started

    assert len(kept) == n
    assert elapsed < n * 2.0, (
        f"threaded verification took {elapsed:.1f}s for {n} traces "
        f"({elapsed / n:.1f}s each) -- expected well under 2s each. "
        "A fork-time callback deadlocking against the thread pool looks exactly "
        "like this."
    )


def test_a_runaway_program_is_killed_and_does_not_hang_the_pool():
    """A timing-out solution must cost its timeout, not block the pool."""
    import time

    spin = "```python\nwhile True:\n    pass\n```"
    cfg = DataConfig(
        verify_solutions=True, verify_max_tests=1, verify_timeout=2.0,
        verify_workers=4, dedup_by_problem=False,
    )
    pairs = [pair(spin) for _ in range(4)]

    started = time.monotonic()
    kept = list(verify_pairs(pairs, cfg, FilterStats()))
    elapsed = time.monotonic() - started

    assert kept == []
    # Four 2s timeouts across four workers should overlap, not queue up.
    assert elapsed < 12.0, f"took {elapsed:.1f}s; timeouts are not running concurrently"
