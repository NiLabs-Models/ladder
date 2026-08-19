"""Publish a trained adapter to the Hugging Face Hub with a filled-in model card.

Takes the adapter directory and the two eval result files produced by a run, and
writes a model card whose numbers come from those files rather than from
whatever someone remembers. A model card with hand-typed metrics is how published
results drift from what was actually measured.

    python scripts/push_to_hub.py \
        --adapter outputs/kaggle/outputs/ladder-3b-kaggle \
        --base-results outputs/kaggle/outputs/eval-base.json \
        --tuned-results outputs/kaggle/outputs/eval-tuned.json \
        --repo NiLabs-Models/Ladder-3B

Needs an HF token with write access: `huggingface-cli login`, or HF_TOKEN.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "docs" / "model-card-template.md"


def load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def build_card(args, base: dict, tuned: dict, run_config: dict) -> str:
    model = run_config.get("model", {})
    train = run_config.get("train", {})

    effective_batch = train.get("per_device_train_batch_size", 1) * train.get(
        "gradient_accumulation_steps", 1
    )
    fields = {
        "MODEL_NAME": args.repo.split("/")[-1],
        "N_PROBLEMS": str(tuned.get("n_problems", "?")),
        "BASE_PASS1": f"{base['metrics']['pass@1']:.3f}",
        "TUNED_PASS1": f"{tuned['metrics']['pass@1']:.3f}",
        "LORA_R": str(model.get("lora_r", "?")),
        "LORA_ALPHA": str(model.get("lora_alpha", "?")),
        "MAX_SEQ_LEN": str(model.get("max_seq_length", "?")),
        "MAX_STEPS": str(train.get("max_steps", "?")),
        "EFFECTIVE_BATCH": str(effective_batch),
        "LR": str(train.get("learning_rate", "?")),
        "GPU": args.gpu,
        "HOURS": args.hours,
        "N_TRAIN": args.n_train,
    }

    card = TEMPLATE.read_text(encoding="utf-8")
    for key, value in fields.items():
        card = card.replace("{" + key + "}", value)

    leftover = [k for k in fields if "{" + k + "}" in card]
    if leftover:
        sys.exit(f"model card still has unfilled placeholders: {leftover}")
    return card


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, help="directory holding the LoRA adapter")
    parser.add_argument("--base-results", required=True)
    parser.add_argument("--tuned-results", required=True)
    parser.add_argument("--repo", required=True, help="e.g. NiLabs-Models/Ladder-3B")
    parser.add_argument("--gpu", default="T4 16GB")
    parser.add_argument("--hours", default="?")
    parser.add_argument("--n-train", default="?")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="write the card, do not upload")
    args = parser.parse_args()

    adapter = Path(args.adapter)
    if not adapter.is_dir():
        sys.exit(f"{adapter} is not a directory")
    # An adapter directory without weights means the run saved somewhere else.
    if not any(adapter.glob("adapter_model*")):
        sys.exit(f"no adapter_model* in {adapter}; is this the right directory?")

    base, tuned = load_json(args.base_results), load_json(args.tuned_results)

    run_config_path = adapter / "run_config.json"
    run_config = load_json(run_config_path) if run_config_path.exists() else {}
    if not run_config:
        print(f"warning: no run_config.json in {adapter}; card will have '?' fields")

    card = build_card(args, base, tuned, run_config)
    card_path = adapter / "README.md"
    card_path.write_text(card, encoding="utf-8")
    print(f"wrote model card -> {card_path}")

    delta = tuned["metrics"]["pass@1"] - base["metrics"]["pass@1"]
    b, t = base["metrics"]["pass@1"], tuned["metrics"]["pass@1"]
    print(f"base pass@1 {b:.3f} -> tuned {t:.3f} ({delta:+.3f})")

    if args.dry_run:
        print("dry run; not uploading")
        return 0

    if not (os.environ.get("HF_TOKEN") or (Path.home() / ".cache/huggingface/token").exists()):
        sys.exit("no HF credentials: run `huggingface-cli login` or set HF_TOKEN")

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=str(adapter),
        repo_id=args.repo,
        repo_type="model",
        # The eval JSON and checkpoints are run artifacts, not part of the model.
        ignore_patterns=["checkpoint-*", "*.json.tmp", "runs/*"],
    )
    print(f"pushed -> https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
