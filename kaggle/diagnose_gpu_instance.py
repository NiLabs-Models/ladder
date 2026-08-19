"""GPU-instance diagnostic: reproduce the data-prep stall where it happened.

Every component measured fine on a Kaggle CPU kernel -- 0.21s per example
projected against 93s observed. The remaining untested variable is the GPU
instance itself, which differs from a CPU kernel in ways that matter here:
roughly 13GB of RAM instead of 30GB, and fewer usable cores.

So this runs the real pipeline on a GPU box, instrumented, under a hard 20
minute budget. It reports the rate as it goes rather than only at the end,
because a constant slow rate and a rate that degrades point at completely
different causes: a constant rate means the work is simply expensive here, while
degradation means something is accumulating -- memory pressure, a growing
structure, or swap.

Budgeted to ~25 minutes of GPU quota rather than another 12-hour session.
"""

import json
import os
import resource
import subprocess
import sys
import time

OUT = "/kaggle/working/gpu_diagnosis.json"
BUDGET_SECONDS = 20 * 60
report = {"samples": []}


def save():
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=2)


def rss_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def mem_available_mb():
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    return None


print("== environment ==", flush=True)
report["cpu_count"] = os.cpu_count()
report["mem_available_mb_start"] = mem_available_mb()
gpu = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader",
                     shell=True, capture_output=True, text=True)
report["gpu"] = gpu.stdout.strip()
try:
    with open("/proc/meminfo") as fh:
        report["mem_total_mb"] = round(int(fh.readline().split()[1]) / 1024, 1)
except OSError:
    pass
print(json.dumps(report, indent=2), flush=True)
save()

subprocess.run("pip install -q 'datasets>=2.19'", shell=True, check=False)
if not os.path.isdir("/tmp/ladder"):
    subprocess.run("git clone -q https://github.com/NiLabs-Models/ladder.git "
                   "/tmp/ladder", shell=True, check=True)
sys.path.insert(0, "/tmp/ladder/src")

from datasets import load_dataset  # noqa: E402

from ladder.config import load_config  # noqa: E402
from ladder.data.filters import FilterStats, apply_filters  # noqa: E402
from ladder.data.formatting import to_chat_example  # noqa: E402
from ladder.data.verify import verify_pairs  # noqa: E402

cfg = load_config("/tmp/ladder/configs/ladder-3b-kaggle.yaml")
dcfg = cfg.data
print(f"\n== running the real pipeline: verify={dcfg.verify_solutions} "
      f"workers={dcfg.verify_workers} tests={dcfg.verify_max_tests} "
      f"timeout={dcfg.verify_timeout} ==", flush=True)

rows = load_dataset(dcfg.dataset, dcfg.config_name, split=dcfg.split, streaming=True)


def pairs_iter():
    for row in rows:
        example = to_chat_example(row)
        if example is not None:
            yield row, example


stream = pairs_iter()
if dcfg.verify_solutions:
    stream = verify_pairs(stream, dcfg, FilterStats())

stats = FilterStats()
started = time.monotonic()
last_mark = started
kept = 0

for _example in apply_filters(stream, dcfg, None, stats):
    kept += 1
    if kept % 25 == 0:
        now = time.monotonic()
        sample = {
            "kept": kept,
            "elapsed_s": round(now - started, 1),
            "seconds_per_example_overall": round((now - started) / kept, 2),
            "seconds_per_example_recent": round((now - last_mark) / 25, 2),
            "rss_mb": rss_mb(),
            "mem_available_mb": mem_available_mb(),
        }
        report["samples"].append(sample)
        print(json.dumps(sample), flush=True)
        save()
        last_mark = now
    if time.monotonic() - started > BUDGET_SECONDS:
        print("budget reached, stopping", flush=True)
        break

elapsed = time.monotonic() - started
report["total_kept"] = kept
report["total_seconds"] = round(elapsed, 1)
report["seconds_per_example"] = round(elapsed / kept, 2) if kept else None
report["observed_on_failed_run"] = 93
report["drop_reasons"] = stats.dropped
save()

print("\n== VERDICT ==", flush=True)
print(f"kept {kept} in {elapsed / 60:.1f} min "
      f"= {report['seconds_per_example']}s each (failed run: ~93s)", flush=True)
if report["samples"]:
    first = report["samples"][0]["seconds_per_example_recent"]
    last = report["samples"][-1]["seconds_per_example_recent"]
    print(f"rate first->last: {first}s -> {last}s per example", flush=True)
    print("degrading" if last > first * 2 else "stable", flush=True)
