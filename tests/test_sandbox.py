import pytest

from ladder.eval.sandbox import Verdict, judge, normalize_output, outputs_match, run_program


def test_normalize_ignores_trailing_whitespace_and_newlines():
    assert normalize_output("3 \n\n") == normalize_output("3")
    assert normalize_output("a\r\nb\r\n") == "a\nb"


def test_outputs_match_is_float_tolerant():
    assert outputs_match("0.5", "0.500000")
    assert outputs_match("1 2.0", "1 2")
    assert not outputs_match("0.5", "0.6")


def test_outputs_match_rejects_a_different_token_count():
    assert not outputs_match("1 2", "1 2 3")


def test_outputs_match_does_not_treat_words_as_numbers():
    assert not outputs_match("YES", "NO")
    assert outputs_match("YES", "YES\n")


def test_accepted_program():
    result = run_program("print(sum(map(int, input().split())))", "1 2\n", timeout_seconds=10)
    assert result.verdict is Verdict.ACCEPTED
    assert result.stdout.strip() == "3"


def test_runtime_error_is_reported_with_stderr():
    result = run_program("raise ValueError('boom')", "", timeout_seconds=10)
    assert result.verdict is Verdict.RUNTIME_ERROR
    assert "boom" in result.stderr


def test_infinite_loop_hits_the_timeout():
    result = run_program("while True: pass", "", timeout_seconds=2)
    assert result.verdict is Verdict.TIMEOUT


def test_empty_code_is_not_an_execution():
    assert run_program("", "").verdict is Verdict.NO_CODE
    assert run_program("   \n", "").verdict is Verdict.NO_CODE


def test_judge_reports_first_failure_and_tests_passed():
    code = "n = int(input())\nprint(n if n < 3 else 999)"
    tests = [("1\n", "1\n"), ("2\n", "2\n"), ("3\n", "3\n"), ("4\n", "4\n")]
    verdict, passed = judge(code, tests, timeout_seconds=10)
    assert verdict is Verdict.WRONG_ANSWER
    assert passed == 2


def test_judge_accepts_a_fully_correct_program():
    code = "print(int(input()) * 2)"
    tests = [("1\n", "2\n"), ("5\n", "10\n")]
    verdict, passed = judge(code, tests, timeout_seconds=10)
    assert verdict is Verdict.ACCEPTED
    assert passed == 2


def test_generated_code_cannot_import_the_harness():
    # -I keeps the child off the parent's sys.path, so a model emitting
    # `import ladder` gets an ImportError instead of reaching into the harness.
    result = run_program("import ladder", "", timeout_seconds=10)
    assert result.verdict is Verdict.RUNTIME_ERROR
    assert "ladder" in result.stderr


@pytest.mark.skipif("os.name != 'posix'", reason="rlimits are POSIX only")
def test_memory_limit_is_enforced_on_posix():
    result = run_program("x = bytearray(4 * 1024**3)", "", timeout_seconds=20, memory_limit_mb=256)
    assert result.verdict is Verdict.RUNTIME_ERROR


def test_exit_is_available_to_generated_code():
    # site.py defines exit()/quit(). Competitive-programming solutions bail out
    # with exit() constantly; disabling site turned correct programs into
    # runtime errors, so the sandbox must not pass -S.
    result = run_program("print('yes')\nexit()\nprint('unreachable')", "", timeout_seconds=10)
    assert result.verdict is Verdict.ACCEPTED
    assert result.stdout.strip() == "yes"


def test_sys_exit_with_a_nonzero_code_is_a_runtime_error():
    assert run_program("import sys; sys.exit(1)", "").verdict is Verdict.RUNTIME_ERROR


def test_yes_no_case_is_accepted_by_default():
    # Codeforces treats these as interchangeable, and reference solutions really
    # do print a different case than the expected-output file.
    assert outputs_match("YES", "Yes")
    assert outputs_match("NO", "no")
    assert outputs_match("Alice", "ALICE")


def test_case_insensitivity_does_not_reach_non_alphabetic_tokens():
    # A grid line carries meaning in its case; "C.C." is not alphabetic, so it
    # stays an exact comparison.
    assert not outputs_match("C.C.", "c.c.")


def test_case_sensitive_mode_rejects_a_case_difference():
    assert not outputs_match("YES", "Yes", case_sensitive=True)


def test_case_insensitivity_still_rejects_a_different_word():
    assert not outputs_match("YES", "nope")


def test_judge_honours_case_sensitivity():
    code = "print('Yes')"
    assert judge(code, [("", "YES\n")], timeout_seconds=10)[0] is Verdict.ACCEPTED
    assert (
        judge(code, [("", "YES\n")], timeout_seconds=10, case_sensitive=True)[0]
        is Verdict.WRONG_ANSWER
    )
