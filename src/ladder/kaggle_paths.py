"""Finding the artifacts Kaggle mounts for a kernel.

Kernel outputs get mounted under /kaggle/input, but the exact layout is Kaggle's
to decide. Hardcoding `/kaggle/input/<slug>/data/sft` was wrong and cost a run,
so these search instead -- and they live here, in the package, rather than
inline in the kernel script, because a kernel script cannot be imported or
tested and this is precisely the logic worth testing.
"""

from __future__ import annotations

import glob
import os

KAGGLE_INPUT = "/kaggle/input"


def _find(filename: str, root: str) -> str | None:
    """Directory of the first `filename` found anywhere under `root`."""
    hits = sorted(glob.glob(os.path.join(root, "**", filename), recursive=True))
    return os.path.dirname(hits[0]) if hits else None


def find_prepared_data(root: str = KAGGLE_INPUT) -> str | None:
    """Directory holding train.jsonl from the data-build kernel."""
    return _find("train.jsonl", root)


def find_adapter(root: str = KAGGLE_INPUT) -> str | None:
    """Directory holding a trained LoRA adapter from the training kernel.

    Keys on adapter_config.json rather than the weights file, because the
    weights extension varies (safetensors, bin) while the config does not.
    Checkpoint directories are skipped: `checkpoint-100` is a mid-training
    snapshot, and scoring one instead of the final adapter would silently
    report the wrong model.
    """
    hits = sorted(glob.glob(os.path.join(root, "**", "adapter_config.json"), recursive=True))
    finals = [h for h in hits if "checkpoint-" not in h.replace("\\", "/")]
    chosen = finals or hits
    return os.path.dirname(chosen[0]) if chosen else None


def describe_mounts(root: str = KAGGLE_INPUT, max_depth: int = 3) -> str:
    """A short listing of what is mounted, for error messages.

    A failure to find an artifact is nearly always a kernel_sources wiring
    problem, and showing what IS there names which one immediately instead of
    costing another round trip.
    """
    if not os.path.isdir(root):
        return f"  ({root} does not exist)"
    lines = []
    base_depth = root.rstrip("/").count("/")
    for current, dirs, files in os.walk(root):
        depth = current.count("/") - base_depth
        if depth > max_depth:
            dirs[:] = []
            continue
        lines.append(f"  {current}: {sorted(files)[:6]}")
    return "\n".join(lines) or f"  (nothing mounted under {root})"
