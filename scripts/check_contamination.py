"""Verify no evaluated problem appears in the training data.

The hash split makes this true by construction, but "by construction" is an
argument. This checks the artifacts a run actually produced, which catches the
failure the argument cannot: a build and an eval configured with different seeds
or a different val_fraction, which silently produces a great-looking pass@1 on
problems the model was trained on.

    python scripts/check_contamination.py --config configs/ladder-3b-kaggle.yaml \
        --data data/sft

Exits non-zero on any overlap.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ladder.config import load_config  # noqa: E402
from ladder.data.contamination import check  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default="data/sft", help="directory holding train.jsonl")
    parser.add_argument(
        "--against",
        choices=["val", "eval"],
        default="eval",
        help="compare training against the val split file, or against the problems "
        "the eval harness would actually load (needs the dataset)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(args.data)

    train_path = data_dir / "train.jsonl"
    if not train_path.exists():
        sys.exit(f"{train_path} not found -- run `ladder build-data` first")
    train = read_jsonl(train_path)

    if args.against == "val":
        val_path = data_dir / "val.jsonl"
        if not val_path.exists():
            sys.exit(f"{val_path} not found")
        held_out = read_jsonl(val_path)
        source = str(val_path)
    else:
        from ladder.eval.tasks import load_problems

        problems = load_problems(cfg.eval, cfg.data)
        held_out = [{"problem_id": p.problem_id} for p in problems]
        source = f"{cfg.eval.dataset} (as the eval harness would load it)"

    print(f"train: {train_path}")
    print(f"held out: {source}\n")

    report = check(train, held_out)
    print(report.summary())

    if not report.clean:
        print(
            "\nCONTAMINATION: the model would be scored on problems it trained on. "
            "Check that build-data and eval used the same seed and val_fraction.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
