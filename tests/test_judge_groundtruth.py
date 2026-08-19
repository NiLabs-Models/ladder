"""The judge, checked against solutions that are known to be correct.

Unit tests did not catch any of the three real judge bugs found so far -- each
one passed every test in test_sandbox.py while silently reporting correct
solutions as wrong. What catches that class is running real solutions that are
known to work and asserting the judge agrees.

The fixture is 30 real Codeforces problems with reference solutions, every one
verified accepted when the fixture was built. A failure here is a judge
regression, and it invalidates every number the project publishes.
"""

import json
from pathlib import Path

import pytest

from ladder.eval.sandbox import Verdict, judge

FIXTURE = Path(__file__).parent / "fixtures" / "judge_groundtruth.json"


def load():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["problems"]


def test_fixture_is_present_and_substantial():
    problems = load()
    assert len(problems) >= 25, "fixture too small to catch much"
    for p in problems:
        assert p["code"].strip()
        assert p["tests"], f"{p['id']} has no tests"


@pytest.mark.parametrize("problem", load(), ids=lambda p: p["id"].replace("/", "_"))
def test_known_good_solution_is_accepted(problem):
    tests = [(t["input"], t["output"]) for t in problem["tests"]]
    verdict, passed = judge(problem["code"], tests, timeout_seconds=20)
    assert verdict is Verdict.ACCEPTED, (
        f"{problem['id']} ({problem['title']}) is a known-correct solution but the "
        f"judge returned {verdict.value} after {passed}/{len(tests)} tests. "
        f"This is a judge regression -- it affects every published number."
    )


def test_fixture_covers_the_exit_builtin_regression():
    """`-S` on the child interpreter removed exit(); solutions using it broke.

    Guarding this explicitly so nobody re-adds -S for 'isolation' and quietly
    turns a chunk of correct solutions into runtime errors.
    """
    import re

    using_exit = [p["id"] for p in load() if re.search(r"(?<!sys\.)\bexit\(\)", p["code"])]
    assert using_exit, "fixture no longer covers bare exit(); the -S regression is unguarded"


def test_fixture_covers_the_output_case_regression():
    """Codeforces accepts Yes/yes/YES interchangeably; exact match did not.

    Requires a genuine case *mismatch* -- a solution printing "Yes" against an
    expected file saying "YES". Merely having uppercase output is not enough:
    the first version of this fixture had that and still failed to catch the
    regression when case-sensitivity was turned back on.
    """
    from ladder.eval.sandbox import outputs_match, run_program

    caught = []
    for problem in load():
        for t in problem["tests"]:
            got = run_program(problem["code"], t["input"], timeout_seconds=20).stdout
            insensitive = outputs_match(t["output"], got)
            sensitive = outputs_match(t["output"], got, case_sensitive=True)
            if insensitive and not sensitive:
                caught.append(problem["id"])
                break

    assert caught, (
        "fixture contains no solution whose output differs from the expected "
        "file only by case, so the case-insensitivity behaviour is unguarded"
    )
