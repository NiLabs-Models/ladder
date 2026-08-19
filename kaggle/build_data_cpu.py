"""CPU-only Kaggle kernel that builds the SFT dataset.

This exists because the first real run put data prep inside the GPU session and
spent all twelve hours of it there, with the accelerator idle the whole time.
The README always said data prep is CPU-only and should be done anywhere else;
the kernel simply did not follow its own advice.

Kaggle CPU kernels do not consume the weekly GPU quota, and a GPU kernel can
read another kernel's output directly, so the training kernel declares this one
in `kernel_sources` and starts at step one with the data already built.
"""

import json
import os
import subprocess
import sys
import time

REPO = "/tmp/ladder"
OUT_DIR = "/kaggle/working/data/sft"
CONFIG = f"{REPO}/configs/ladder-3b-kaggle.yaml"

subprocess.run("pip install -q 'datasets>=2.19'", shell=True, check=False)
if not os.path.isdir(REPO):
    subprocess.run(
        f"git clone -q https://github.com/NiLabs-Models/ladder.git {REPO}",
        shell=True, check=True,
    )
sys.path.insert(0, f"{REPO}/src")

from ladder.config import load_config  # noqa: E402
from ladder.data.build import build  # noqa: E402

cfg = load_config(CONFIG)
print(f"building {cfg.data.max_samples} examples "
      f"(verify={cfg.data.verify_solutions}, workers={cfg.data.verify_workers})",
      flush=True)

started = time.monotonic()
counts = build(cfg, OUT_DIR)
elapsed = time.monotonic() - started

summary = {
    "counts": counts,
    "seconds": round(elapsed, 1),
    "seconds_per_example": round(elapsed / max(1, sum(counts.values())), 2),
    "config": cfg.to_dict(),
}
with open("/kaggle/working/build_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)

print(f"\nbuilt {counts} in {elapsed / 60:.1f} min "
      f"({summary['seconds_per_example']}s each)", flush=True)

# The training kernel reads these two files and nothing else.
for name in ("train.jsonl", "val.jsonl"):
    path = os.path.join(OUT_DIR, name)
    print(f"{name}: {os.path.getsize(path) / 1e6:.1f} MB", flush=True)
