## What this changes

<!-- One or two sentences. -->

## Why

<!-- What was wrong, or what this makes possible. -->

## Checks

- [ ] `pytest -q` passes
- [ ] `ruff check .` is clean

## If this touches the judge or the prompt format

`build_problem_prompt` and `eval/sandbox.py` are contracts. Changing either
invalidates existing checkpoints' numbers, and a judge bug silently distorts
every result the project publishes.

- [ ] Re-ran the ground-truth check: dataset reference solutions judged against
      their own tests, pass rate reported below
- [ ] Noted that published numbers need regenerating

<!--
Ground-truth pass rate before / after:
-->
