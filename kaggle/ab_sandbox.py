"""A/B the old and new sandbox on one Kaggle GPU box, same data, same session.

The failed 12-hour run and the fast 13-minute diagnostic differed in exactly one
thing: the sandbox rewrite that removed preexec_fn. Everything else measured
fine, and two earlier hypotheses died under testing, so this settles it by
running both versions back to back on identical rows rather than reasoning about
them.

The earlier CPU diagnostic did test the old preexec_fn -- but serially, ten
spawns in a loop. The suspect combination is preexec_fn called from a thread
pool, which that never exercised. This does.

Runs the literal old file from git, not a reimplementation.
"""

import importlib
import json
import os
import subprocess
import sys
import time

OLD_SHA = "bfe5708"          # before the sandbox rewrite
SANDBOX = "src/ladder/eval/sandbox.py"
REPO = "/kaggle/working/ladder"
OUT = "/kaggle/working/ab_result.json"
N = 60

report = {}


def save():
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2), flush=True)


subprocess.run("pip install -q 'datasets>=2.19'", shell=True, check=False)
if not os.path.isdir(REPO):
    subprocess.run(
        f"git clone -q https://github.com/NiLabs-Models/ladder.git {REPO}",
        shell=True, check=True,
    )
sys.path.insert(0, f"{REPO}/src")

gpu = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader",
                     shell=True, capture_output=True, text=True)
report["gpu"] = gpu.stdout.strip()
report["cpu_count"] = os.cpu_count()

from datasets import load_dataset  # noqa: E402

from ladder.config import DataConfig  # noqa: E402
from ladder.data.formatting import to_chat_example  # noqa: E402

print("fetching rows once, so both arms judge identical work", flush=True)
stream = load_dataset("open-r1/codeforces-cots", "solutions_py_decontaminated",
                      split="train", streaming=True)
rows = []
for row in stream:
    example = to_chat_example(row)
    if example is not None:
        rows.append((row, example))
    if len(rows) >= N:
        break
report["rows"] = len(rows)
save()

# Exactly the config the failed run used.
cfg = DataConfig(verify_solutions=True, verify_max_tests=4, verify_timeout=5.0,
                 verify_workers=8, dedup_by_problem=False)


def measure(label):
    """Re-import the sandbox and time threaded verification over the same rows."""
    import ladder.data.verify as verify_mod
    import ladder.eval.sandbox as sandbox_mod
    import ladder.eval.tasks as tasks_mod

    importlib.reload(sandbox_mod)
    importlib.reload(tasks_mod)
    importlib.reload(verify_mod)

    from ladder.data.filters import FilterStats

    started = time.monotonic()
    kept = list(verify_mod.verify_pairs(iter(rows), cfg, FilterStats()))
    elapsed = time.monotonic() - started

    report[label] = {
        "seconds_total": round(elapsed, 1),
        "seconds_per_trace": round(elapsed / len(rows), 2),
        "kept": len(kept),
        "uses_preexec": "preexec_fn" in open(f"{REPO}/{SANDBOX}").read(),
    }
    save()
    return elapsed


print("\n=== ARM 1: current sandbox (no preexec_fn) ===", flush=True)
measure("new_sandbox")

print(f"\n=== ARM 2: sandbox from {OLD_SHA} (preexec_fn) ===", flush=True)
subprocess.run(f"cd {REPO} && git checkout {OLD_SHA} -- {SANDBOX}",
               shell=True, check=True)
measure("old_sandbox")

subprocess.run(f"cd {REPO} && git checkout HEAD -- {SANDBOX}", shell=True, check=False)

new = report["new_sandbox"]["seconds_per_trace"]
old = report["old_sandbox"]["seconds_per_trace"]
report["slowdown_factor"] = round(old / new, 1) if new else None
print("\n=== VERDICT ===", flush=True)
print(f"new sandbox: {new}s/trace", flush=True)
print(f"old sandbox: {old}s/trace", flush=True)
print(f"old is {report['slowdown_factor']}x slower", flush=True)
save()
