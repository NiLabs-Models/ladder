"""Row filtering and the train/val split.

Filtering is where most of the quality comes from in an SFT run, so every drop
is counted and reported: a build that silently discards 90% of the corpus should
be obvious from the log, not from a suspiciously fast epoch.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

from ladder.config import DataConfig


@dataclass
class FilterStats:
    seen: int = 0
    kept: int = 0
    dropped: dict[str, int] = field(default_factory=dict)

    def drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1

    def summary(self) -> str:
        lines = [f"seen={self.seen} kept={self.kept}"]
        for reason, count in sorted(self.dropped.items(), key=lambda kv: -kv[1]):
            lines.append(f"  dropped[{reason}] = {count}")
        return "\n".join(lines)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate for when no tokenizer is loaded.

    ~3.6 chars/token is a reasonable average for Python-heavy English text under
    the Qwen tokenizer. Only used for logging and coarse filtering; the training
    script passes the real tokenizer.
    """
    return max(1, int(len(text) / 3.6))


def _example_text(example: dict[str, Any]) -> str:
    return "\n".join(m["content"] for m in example["messages"])


def _rating_of(row: dict[str, Any]) -> int | None:
    """Codeforces difficulty rating, when the source dataset carries one.

    `open-r1/codeforces-cots` has no rating column, so rating bounds are a no-op
    there and the rows fall through untouched.
    """
    value = row.get("rating")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def apply_filters(
    pairs: Iterable[tuple[dict[str, Any], dict[str, Any]]],
    cfg: DataConfig,
    count_tokens: Callable[[str], int] | None = None,
    stats: FilterStats | None = None,
) -> Iterator[dict[str, Any]]:
    """Filter `(raw_row, chat_example)` pairs, yielding the examples that survive.

    Streams rather than materializing: the full CoT corpus does not need to be
    resident in a 13GB Kaggle notebook just to be filtered.
    """
    count_tokens = count_tokens or estimate_tokens
    stats = stats if stats is not None else FilterStats()
    seen_problems: set[str] = set()

    for row, example in pairs:
        stats.seen += 1

        if cfg.require_finish_reason is not None:
            finish = row.get("finish_reason")
            if finish is not None and finish != cfg.require_finish_reason:
                stats.drop("finish_reason")
                continue

        rating = _rating_of(row)
        if rating is not None:
            if cfg.min_rating is not None and rating < cfg.min_rating:
                stats.drop("rating_too_low")
                continue
            if cfg.max_rating is not None and rating > cfg.max_rating:
                stats.drop("rating_too_high")
                continue

        if cfg.dedup_by_problem:
            pid = example.get("problem_id") or row.get("id")
            if pid is not None:
                if pid in seen_problems:
                    stats.drop("duplicate_problem")
                    continue
                seen_problems.add(pid)

        n_tokens = count_tokens(_example_text(example))
        if n_tokens < cfg.min_tokens:
            stats.drop("too_short")
            continue
        if n_tokens > cfg.max_tokens:
            stats.drop("too_long")
            continue

        example = dict(example)
        example["n_tokens"] = n_tokens
        stats.kept += 1
        yield example

        if cfg.max_samples and stats.kept >= cfg.max_samples:
            return


def split_key(problem_id: Any, seed: int) -> float:
    """Deterministic [0,1) hash of a problem id.

    Hashing instead of shuffling means the split is stable across runs, across
    machines, and as new rows are added upstream -- rerunning the build after a
    dataset refresh will not quietly move a held-out problem into training.
    """
    digest = hashlib.sha256(f"{seed}:{problem_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def is_validation(example: dict[str, Any], cfg: DataConfig) -> bool:
    if cfg.val_fraction <= 0:
        return False
    pid = example.get("problem_id")
    if pid is None:
        return False
    return split_key(pid, cfg.seed) < cfg.val_fraction
