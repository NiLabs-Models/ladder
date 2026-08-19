"""Drop training traces whose solution does not actually pass the problem's tests.

The reasoning traces in `codeforces-cots` are *model-generated*. Measured against
the problems' own test cases, they are essentially always correct on easy
problems and roughly 70% correct on div1 D/E/G/H -- so training on the raw corpus
teaches confident, well-structured, wrong solutions on exactly the problems that
are hardest.

Verification is the same judge the eval uses, run over the training set: extract
the trace's code, run it, keep the trace only if it passes. It is expensive
(every kept row costs a few subprocess runs) but it is CPU work, so it belongs in
data prep where it does not consume GPU hours.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ladder.config import DataConfig
from ladder.data.filters import FilterStats
from ladder.data.formatting import extract_code
from ladder.eval.sandbox import Verdict, judge
from ladder.eval.tasks import collect_tests

Pair = tuple[dict[str, Any], dict[str, Any]]


def _chunks(items: Iterable[Pair], size: int) -> Iterator[list[Pair]]:
    batch: list[Pair] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def check_pair(pair: Pair, cfg: DataConfig) -> str:
    """Return the reason to drop this trace, or "" to keep it."""
    row, example = pair

    # Only exact-diff problems can be verified this way; a `checker` problem has
    # many valid outputs and would be failed for being right in a different way.
    if row.get("problem_type") != "diff":
        return "" if cfg.verify_keep_unverifiable else "unverifiable_problem_type"

    tests = collect_tests(row, cfg.verify_max_tests)
    if not tests:
        return "" if cfg.verify_keep_unverifiable else "no_tests"

    code = extract_code(example["messages"][-1]["content"])
    if code is None:
        return "no_code_in_trace"

    verdict, _ = judge(
        code,
        tests,
        timeout_seconds=cfg.verify_timeout,
        memory_limit_mb=cfg.verify_memory_limit_mb,
    )
    return "" if verdict is Verdict.ACCEPTED else f"failed_verification:{verdict.value}"


def verify_pairs(
    pairs: Iterable[Pair],
    cfg: DataConfig,
    stats: FilterStats | None = None,
) -> Iterator[Pair]:
    """Yield only the pairs whose solution passes the problem's own tests.

    Threads rather than processes: the work is `subprocess.run`, which releases
    the GIL, and threads keep this a streaming pass instead of forcing the whole
    corpus into memory to be scattered across workers.
    """
    stats = stats if stats is not None else FilterStats()
    workers = cfg.verify_workers or 8

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for batch in _chunks(pairs, workers * 4):
            reasons = list(pool.map(lambda p: check_pair(p, cfg), batch))
            for pair, reason in zip(batch, reasons, strict=True):
                if reason:
                    stats.drop(reason)
                else:
                    yield pair
