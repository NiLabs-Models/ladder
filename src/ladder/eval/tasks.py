"""Assembling the held-out problem set and its test cases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ladder.config import DataConfig, EvalConfig
from ladder.data.filters import split_key
from ladder.data.formatting import build_problem_prompt


@dataclass
class EvalProblem:
    problem_id: str
    title: str
    prompt: str
    tests: list[tuple[str, str]] = field(default_factory=list)
    time_limit: float | None = None


def _paired(field_value: Any) -> list[tuple[str, str]]:
    """Normalize the two shapes tests come in across the Codeforces datasets.

    Some columns are a dict of parallel lists (`{"input": [...], "output": [...]}`),
    others are a list of records (`[{"input": ..., "output": ...}]`).
    """
    if not field_value:
        return []
    if isinstance(field_value, dict):
        inputs = field_value.get("input") or []
        outputs = field_value.get("output") or []
        # strict=False on purpose: a malformed row with ragged columns should yield
        # the pairs it does have rather than blowing up the whole eval load.
        pairs = zip(inputs, outputs, strict=False)
        return [(i, o) for i, o in pairs if i is not None and o is not None]
    if isinstance(field_value, list):
        out = []
        for item in field_value:
            if isinstance(item, dict) and item.get("input") is not None:
                out.append((item["input"], item.get("output") or ""))
        return out
    return []


def collect_tests(row: dict[str, Any], limit: int = 20) -> list[tuple[str, str]]:
    """Gather test cases, best-quality first, deduplicated, capped at `limit`.

    Order matters because judging stops at the first failure: official tests
    catch real bugs, generated tests are the fallback when a problem ships with
    only its samples.
    """
    tests: list[tuple[str, str]] = []
    seen: set[str] = set()
    for key in ("public_tests", "public_tests_ms", "private_tests", "generated_tests"):
        for stdin, expected in _paired(row.get(key)):
            if stdin in seen:
                continue
            seen.add(stdin)
            tests.append((stdin, expected))
            if len(tests) >= limit:
                return tests
    return tests


def load_problems(cfg: EvalConfig, data_cfg: DataConfig) -> list[EvalProblem]:
    """Load held-out problems that have at least one usable test case.

    Selection reuses `split_key` with the training seed, so eval problems are
    exactly the ones the SFT build routed to validation -- the model is never
    scored on a problem it was trained on. Problems that exact-match scoring
    cannot grade are filtered out here; see `EvalConfig.problem_types`.
    """
    from datasets import load_dataset

    rows = load_dataset(cfg.dataset, cfg.config_name, split=cfg.split, streaming=True)

    problems: list[EvalProblem] = []
    seen_ids: set[str] = set()
    for row in rows:
        pid = row.get("id")
        if pid is None or pid in seen_ids:
            continue
        if data_cfg.val_fraction > 0 and split_key(pid, data_cfg.seed) >= data_cfg.val_fraction:
            continue

        if cfg.problem_types and row.get("problem_type") not in cfg.problem_types:
            continue

        tests = collect_tests(row, cfg.max_tests_per_problem)
        if not tests:
            continue
        prompt = build_problem_prompt(row)
        if not prompt:
            continue

        seen_ids.add(pid)
        problems.append(
            EvalProblem(
                problem_id=pid,
                title=row.get("title") or pid,
                prompt=prompt,
                tests=tests,
                time_limit=row.get("time_limit"),
            )
        )
        if len(problems) >= cfg.num_problems:
            break

    return problems
