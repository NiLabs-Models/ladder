"""Check the judge against solutions that are known to be correct.

Every number this project publishes depends on the judge being right, and a
judge bug is silent: it does not crash, it just reports correct solutions as
wrong and drags pass@k down. Unit tests do not catch this class -- the three
real bugs found so far all passed the unit tests:

  * `-S` on the child interpreter removed the `exit()` builtin, so any solution
    that bailed out early became a runtime error
  * `checker` problems have many valid outputs and cannot be scored by exact match
  * Codeforces accepts Yes/yes/YES interchangeably; exact match did not

What catches them is running solutions that are known to work and checking that
the judge agrees.

    python scripts/validate_judge.py              # offline, committed fixture
    python scripts/validate_judge.py --live 100   # stream fresh rows from HF

The offline mode is deterministic, needs no network, and runs in CI. Use the
live mode when changing the judge, and put the agreement rate in the PR.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ladder.eval.sandbox import Verdict, judge  # noqa: E402

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "judge_groundtruth.json"


def load_fixture() -> list[dict]:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return data["problems"]


def load_live(limit: int) -> list[dict]:
    """Stream fresh rows and extract the same shape the fixture uses."""
    from datasets import load_dataset

    from ladder.data.formatting import extract_code, to_chat_example
    from ladder.eval.tasks import collect_tests

    rows = load_dataset(
        "open-r1/codeforces-cots", "solutions_py_decontaminated", split="train", streaming=True
    )
    out = []
    for row in rows:
        # Only exact-diff problems are gradeable this way; see EvalConfig.problem_types.
        if row.get("problem_type") != "diff":
            continue
        tests = collect_tests(row, 6)
        example = to_chat_example(row)
        if not tests or example is None:
            continue
        code = extract_code(example["messages"][-1]["content"])
        if not code:
            continue
        out.append({
            "id": row["id"],
            "title": row.get("title", ""),
            "code": code,
            "tests": [{"input": i, "output": o} for i, o in tests],
        })
        if len(out) >= limit:
            break
    return out


def check(problems: list[dict], timeout: float) -> tuple[int, list[tuple[str, str, int, int]]]:
    accepted, failures = 0, []
    for p in problems:
        tests = [(t["input"], t["output"]) for t in p["tests"]]
        verdict, passed = judge(p["code"], tests, timeout_seconds=timeout)
        if verdict is Verdict.ACCEPTED:
            accepted += 1
        else:
            failures.append((p["id"], verdict.value, passed, len(tests)))
    return accepted, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", type=int, metavar="N", help="stream N fresh rows instead")
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()

    if args.live:
        problems = load_live(args.live)
        # Live traces are model-generated, so some are genuinely wrong. The
        # fixture is pre-verified; a live run is a rate, not a pass/fail.
        expectation = "expect well under 100% -- live traces include wrong solutions"
    else:
        problems = load_fixture()
        expectation = "expect 100% -- every fixture solution was verified accepted"

    if not problems:
        print("no problems loaded", file=sys.stderr)
        return 1

    accepted, failures = check(problems, args.timeout)
    total = len(problems)
    print(f"judge accepted {accepted}/{total} = {accepted / total:.1%}  ({expectation})")

    for pid, verdict, passed, n in failures:
        print(f"  MISS {pid:<10} {verdict:<14} {passed}/{n} tests")

    if args.live:
        return 0
    if failures:
        print(
            "\nJUDGE REGRESSION: these solutions are known-correct and the judge "
            "now rejects them. Every published number is affected.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
