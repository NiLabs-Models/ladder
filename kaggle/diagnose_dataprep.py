"""CPU-only Kaggle kernel: find out why data prep took 12 hours here.

Measured elsewhere, the same work is fast: ~0.2s/row streaming and ~0.2s/trace
verifying on a Windows laptop, and threaded verification stays inside its
wall-clock budget on GitHub's Linux runners. On Kaggle the first real run
managed 446 examples in twelve hours -- about 93s each -- and never reached
training.

So the cause is specific to this environment, and guessing at it has already
been wrong once. This times each stage separately, on Kaggle, on CPU, where it
costs nothing but wall clock.

Hypotheses, each with its own timing below:
  H1  streaming from HF is slow here, and verification is irrelevant
  H2  verification is slow because subprocess spawning is slow on this box
  H3  the old preexec_fn rlimit callback stalls against the thread pool
  H4  RLIMIT_NPROC(64) is a per-*user* limit, and this container's user already
      has more than 64 processes, so every fork under it fails
"""

import json
import os
import resource
import subprocess
import sys
import time

OUT = "/kaggle/working/diagnosis.json"
results = {}


def save():
    with open(OUT, "w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps(results, indent=2), flush=True)


def section(name):
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}", flush=True)


# --------------------------------------------------------------------------
section("environment")
results["cpu_count"] = os.cpu_count()
try:
    with open("/proc/loadavg") as fh:
        results["loadavg"] = fh.read().strip()
except OSError:
    pass

# H4: how many processes does this user already have, versus the limit the old
# code tried to impose?
try:
    out = subprocess.run(["ps", "-u", str(os.getuid()), "--no-headers"],
                         capture_output=True, text=True)
    results["user_process_count"] = len(out.stdout.strip().split("\n"))
except Exception as exc:
    results["user_process_count"] = f"error: {exc}"
results["rlimit_nproc"] = resource.getrlimit(resource.RLIMIT_NPROC)
results["rlimit_as"] = resource.getrlimit(resource.RLIMIT_AS)
save()

# --------------------------------------------------------------------------
section("H2: bare subprocess spawn cost")
t0 = time.monotonic()
N = 20
for _ in range(N):
    subprocess.run([sys.executable, "-I", "-c", "pass"], capture_output=True)
results["spawn_seconds_each"] = round((time.monotonic() - t0) / N, 3)
save()

# --------------------------------------------------------------------------
section("H4: spawn with RLIMIT_NPROC(64), the old preexec_fn")


def old_preexec():
    resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024,) * 2)
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    os.setsid()


t0 = time.monotonic()
errors = 0
for _ in range(10):
    try:
        subprocess.run([sys.executable, "-I", "-c", "pass"],
                       capture_output=True, preexec_fn=old_preexec, timeout=30)
    except Exception as exc:
        errors += 1
        results.setdefault("nproc_error_example", f"{type(exc).__name__}: {exc}")
results["old_preexec_seconds_each"] = round((time.monotonic() - t0) / 10, 3)
results["old_preexec_errors"] = errors
save()

# --------------------------------------------------------------------------
section("install ladder + datasets")
subprocess.run("pip install -q 'datasets>=2.19'", shell=True, check=False)
if not os.path.isdir("/kaggle/working/ladder"):
    subprocess.run("git clone -q https://github.com/NiLabs-Models/ladder.git "
                   "/kaggle/working/ladder", shell=True, check=True)
sys.path.insert(0, "/kaggle/working/ladder/src")

# --------------------------------------------------------------------------
section("H1: streaming cost")
import itertools  # noqa: E402

from datasets import load_dataset  # noqa: E402

t0 = time.monotonic()
stream = load_dataset("open-r1/codeforces-cots", "solutions_py_decontaminated",
                      split="train", streaming=True)
results["stream_open_seconds"] = round(time.monotonic() - t0, 1)

t0 = time.monotonic()
rows = list(itertools.islice(stream, 60))
elapsed = time.monotonic() - t0
results["stream_seconds_per_row"] = round(elapsed / len(rows), 3)
results["stream_rows"] = len(rows)
save()

# --------------------------------------------------------------------------
section("H2/H3: verification cost, serial then threaded")
from ladder.config import DataConfig  # noqa: E402
from ladder.data.filters import FilterStats  # noqa: E402
from ladder.data.formatting import to_chat_example  # noqa: E402
from ladder.data.verify import check_pair, verify_pairs  # noqa: E402

pairs = [(r, e) for r in rows if (e := to_chat_example(r)) is not None]
cfg = DataConfig(verify_solutions=True, verify_max_tests=4, verify_timeout=5.0,
                 verify_workers=8, dedup_by_problem=False)

t0 = time.monotonic()
serial = [check_pair(p, cfg) for p in pairs[:20]]
results["verify_serial_seconds_each"] = round((time.monotonic() - t0) / 20, 2)
save()

t0 = time.monotonic()
kept = list(verify_pairs(pairs, cfg, FilterStats()))
elapsed = time.monotonic() - t0
results["verify_threaded_seconds_each"] = round(elapsed / len(pairs), 2)
results["verify_threaded_kept"] = len(kept)
results["verify_threaded_total_seconds"] = round(elapsed, 1)
save()

section("VERDICT")
per_example = (results["stream_seconds_per_row"] + results["verify_threaded_seconds_each"])
print(f"projected seconds per kept example: {per_example:.2f}")
print("observed on the failed run:          ~93")
results["projected_seconds_per_example"] = round(per_example, 2)
results["observed_seconds_per_example"] = 93
save()
