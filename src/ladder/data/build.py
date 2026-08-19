"""Build the SFT train/val JSONL files from the source dataset.

Kept separate from training on purpose: data prep is CPU-bound and can run on a
laptop or a free CPU notebook, so you do not burn GPU-hours streaming and
filtering a dataset before the first optimizer step.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ladder.config import RunConfig
from ladder.data.filters import FilterStats, apply_filters, is_validation
from ladder.data.formatting import to_chat_example


def _iter_pairs(rows):
    for row in rows:
        example = to_chat_example(row)
        if example is None:
            continue
        yield row, example


def build(cfg: RunConfig, out_dir: str | Path, tokenizer=None) -> dict[str, int]:
    """Write train.jsonl / val.jsonl under `out_dir`; return row counts."""
    from datasets import load_dataset

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count_tokens = None
    if tokenizer is not None:
        def count_tokens(text: str) -> int:
            return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    dcfg = cfg.data
    print(f"loading {dcfg.dataset}:{dcfg.config_name}:{dcfg.split} (streaming)", file=sys.stderr)
    rows = load_dataset(dcfg.dataset, dcfg.config_name, split=dcfg.split, streaming=True)

    stats = FilterStats()
    counts = {"train": 0, "val": 0}
    train_path, val_path = out_dir / "train.jsonl", out_dir / "val.jsonl"

    pairs = _iter_pairs(rows)
    if dcfg.verify_solutions:
        # Verification executes model-written code. Same caveat as the eval:
        # container or disposable VM only.
        from ladder.data.verify import verify_pairs

        print("verifying solutions against problem tests (cpu-bound)", file=sys.stderr)
        pairs = verify_pairs(pairs, dcfg, stats)

    with open(train_path, "w", encoding="utf-8") as ftrain, \
         open(val_path, "w", encoding="utf-8") as fval:
        for example in apply_filters(pairs, dcfg, count_tokens, stats):
            split = "val" if is_validation(example, dcfg) else "train"
            handle = fval if split == "val" else ftrain
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
            counts[split] += 1
            if stats.kept % 500 == 0:
                print(f"  kept {stats.kept} ...", file=sys.stderr)

    print(stats.summary(), file=sys.stderr)
    print(f"wrote {counts['train']} -> {train_path}", file=sys.stderr)
    print(f"wrote {counts['val']} -> {val_path}", file=sys.stderr)

    (out_dir / "build_meta.json").write_text(
        json.dumps(
            {"config": cfg.to_dict(), "counts": counts, "filter_stats": stats.dropped},
            indent=2,
        ),
        encoding="utf-8",
    )
    return counts
