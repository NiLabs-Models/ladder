"""Record a finished run's artifacts under results/ so its numbers are auditable.

Refuses to record a run whose base and tuned evals do not cover the same
problems. Comparing pass@1 across two different problem sets is the easiest way
to publish a number that means nothing, and it is invisible once the numbers are
in a table.

    python scripts/record_run.py --name ladder-3b-kaggle --from outputs/kaggle
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"

WANTED = ["eval-base.json", "eval-tuned.json", "throughput.json", "run_config.json"]


def find(root: Path, name: str) -> Path | None:
    """Kaggle output nests directories, so search rather than assume a layout."""
    direct = root / name
    if direct.exists():
        return direct
    return next(iter(sorted(root.rglob(name))), None)


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def problem_ids(results_path: Path) -> set[str]:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    return {p["problem_id"] for p in data.get("problems", [])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="run name, becomes the directory")
    parser.add_argument("--from", dest="src", required=True, help="directory to pull from")
    parser.add_argument("--gpu", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="record even if base and tuned evals cover different problems",
    )
    args = parser.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        sys.exit(f"{src} is not a directory")

    found = {name: find(src, name) for name in WANTED}
    missing = [n for n, p in found.items() if p is None]
    for name, path in found.items():
        print(f"  {'ok ' if path else 'MISSING'} {name}" + (f"  <- {path}" if path else ""))
    if found["eval-base.json"] is None or found["eval-tuned.json"] is None:
        sys.exit("\nboth eval-base.json and eval-tuned.json are required")

    base_ids = problem_ids(found["eval-base.json"])
    tuned_ids = problem_ids(found["eval-tuned.json"])
    if base_ids != tuned_ids:
        only_base, only_tuned = base_ids - tuned_ids, tuned_ids - base_ids
        message = (
            f"\nbase and tuned evals cover different problems: "
            f"{len(only_base)} only in base, {len(only_tuned)} only in tuned. "
            "Their pass@1 values are not comparable."
        )
        if not args.allow_mismatch:
            sys.exit(message + "\nPass --allow-mismatch to record anyway.")
        print(message + "\nrecording anyway (--allow-mismatch)")

    out = RESULTS / args.name
    out.mkdir(parents=True, exist_ok=True)
    for name, path in found.items():
        if path:
            shutil.copy2(path, out / name)

    base = json.loads((out / "eval-base.json").read_text(encoding="utf-8"))
    tuned = json.loads((out / "eval-tuned.json").read_text(encoding="utf-8"))
    b, t = base["metrics"]["pass@1"], tuned["metrics"]["pass@1"]

    notes = [
        f"# {args.name}",
        "",
        f"- commit: `{git_sha()}`",
        f"- gpu: {args.gpu or 'unrecorded'}",
        f"- problems: {len(tuned_ids)}",
        "",
        "| model | pass@1 |",
        "| --- | --- |",
        f"| base | {b:.3f} |",
        f"| tuned | {t:.3f} |",
        f"| delta | {t - b:+.3f} |",
    ]
    if missing:
        notes += ["", f"Missing artifacts: {', '.join(missing)}"]
    if args.notes:
        notes += ["", args.notes]
    (out / "notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")

    print(f"\nrecorded -> {out}")
    print(f"base {b:.3f} -> tuned {t:.3f} ({t - b:+.3f}) on {len(tuned_ids)} problems")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
