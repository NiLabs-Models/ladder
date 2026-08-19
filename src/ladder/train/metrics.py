"""Deriving throughput from a finished training run.

Run sizing is currently guesswork: the Kaggle config's step count was chosen
against an estimated 800-1200 tok/s for a 3B in 4-bit on a T4, a range wide
enough to move the total by hours. Everything needed to replace that estimate
with a measurement already exists -- the dataset build records a token count per
example, and the trainer reports `train_runtime` -- it just was never computed.

Kept free of torch so it can be unit tested on any machine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def count_trained_tokens(
    jsonl_path: str | Path,
    max_steps: int,
    effective_batch: int,
) -> tuple[int, int]:
    """Total tokens actually fed to the optimizer, and how many examples had token counts.

    A capped run (`max_steps`) consumes only the first `max_steps * batch`
    examples, so summing the whole file would overstate throughput by however
    much of the corpus went untouched. Examples missing `n_tokens` are ignored in
    the returned example count.
    """
    limit = max_steps * effective_batch if max_steps else None

    total = 0
    counted = 0
    consumed = 0
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            if limit is not None and consumed >= limit:
                break
            record = json.loads(line)
            consumed += 1
            n = record.get("n_tokens")
            if n is None:
                continue
            total += n
            counted += 1
    return total, counted


def summarize(
    train_runtime_seconds: float,
    total_tokens: int,
    n_examples: int,
    peak_vram_bytes: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build the throughput record saved alongside a run's adapter."""
    if train_runtime_seconds <= 0:
        raise ValueError(f"train_runtime must be positive, got {train_runtime_seconds}")

    summary: dict[str, Any] = {
        "train_runtime_seconds": round(train_runtime_seconds, 1),
        "train_runtime_hours": round(train_runtime_seconds / 3600, 3),
        "n_examples": n_examples,
        "total_tokens": total_tokens,
        "tokens_per_second": round(total_tokens / train_runtime_seconds, 1),
        "seconds_per_example": round(train_runtime_seconds / n_examples, 3) if n_examples else None,
        "mean_tokens_per_example": round(total_tokens / n_examples) if n_examples else None,
    }
    if peak_vram_bytes is not None:
        summary["peak_vram_gb"] = round(peak_vram_bytes / 1024**3, 2)
    summary.update(extra)
    return summary


def project_runtime(tokens_per_second: float, total_tokens: int) -> float:
    """Hours a run of `total_tokens` would take at a measured rate.

    The point of recording throughput: sizing the next run by arithmetic rather
    than by guessing a range and trimming for safety.
    """
    if tokens_per_second <= 0:
        raise ValueError("tokens_per_second must be positive")
    return total_tokens / tokens_per_second / 3600
