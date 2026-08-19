"""Push the Ladder kernel to Kaggle, wait for it, and print the numbers.

Kaggle's free tier is the target compute, and it has a real API, so the whole run
is scriptable: push, poll, fetch output. No notebook babysitting.

    python scripts/kaggle_run.py --user <kaggle-username>
    python scripts/kaggle_run.py --user <kaggle-username> --status
    python scripts/kaggle_run.py --user <kaggle-username> --fetch

Needs credentials in ~/.kaggle/kaggle.json (or KAGGLE_USERNAME/KAGGLE_KEY).
Get them at https://www.kaggle.com/settings -> API -> Create New Token.

GPU quota is 30 h/week and a kernel session is capped at 12 h. The shipped
config is sized to land well inside that, including both evals.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SLUG = "ladder-qlora-codeforces"
SMOKE_SLUG = "ladder-smoke"
# Training and evaluation run as separate sessions: measured on a T4 they are
# 5.4h and 5.7h, and 11.2h against a 12h cap is the margin that lost the
# first run. Splitting also banks the adapter before evaluation is attempted.
STAGE_SLUGS = {"train": "ladder-train", "eval": "ladder-eval"}
BUILD_DATA_SLUG = "ladder-build-data"
POLL_SECONDS = 300


def kaggle(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "kaggle", *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def require_credentials() -> None:
    """Accept any of the three shapes Kaggle credentials come in.

    Newer tokens are a single `KGAT_...` string in ~/.kaggle/access_token; older
    ones are a {username, key} pair in ~/.kaggle/kaggle.json. Checking only for
    kaggle.json rejects a perfectly good modern token.
    """
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return
    kaggle_dir = Path.home() / ".kaggle"
    if (kaggle_dir / "kaggle.json").exists() or (kaggle_dir / "access_token").exists():
        return
    sys.exit(
        "No Kaggle credentials found.\n"
        "  https://www.kaggle.com/settings -> API -> Create New Token\n"
        "  Save kaggle.json to ~/.kaggle/kaggle.json, or an access token\n"
        "  to ~/.kaggle/access_token.\n"
        "GPU also requires a phone-verified Kaggle account."
    )


def slug_for(smoke: bool, stage: str) -> str:
    return SMOKE_SLUG if smoke else STAGE_SLUGS.get(stage, SLUG)


def sources_for(user: str, stage: str) -> list:
    """What this stage needs mounted.

    Training needs the prepared dataset. Evaluation needs the adapter the
    training kernel produced; it reads its problems from the dataset itself,
    so it does not need train.jsonl at all.
    """
    if stage == "eval":
        return [user + "/" + STAGE_SLUGS["train"]]
    return [user + "/" + BUILD_DATA_SLUG]


def staging_dir(user: str, smoke: bool = False, stage: str = "all") -> Path:
    """Kaggle pushes a directory: one code file plus its metadata."""
    out = REPO_ROOT / "kaggle" / "_push"
    out.mkdir(parents=True, exist_ok=True)

    source = (REPO_ROOT / "kaggle" / "ladder_kernel.py").read_text(encoding="utf-8")
    # Kaggle has no way to pass parameters to a kernel, so flags are set by a
    # prelude prepended to the same source the real run uses. Running the same
    # file is the point -- a smoke test of different code proves nothing about
    # the run it is meant to de-risk.
    prelude = []
    if stage != "all":
        prelude.append('os.environ["LADDER_STAGE"] = ' + '"' + stage + '"')
    if smoke:
        prelude.append('os.environ["LADDER_SMOKE"] = "1"')
    if prelude:
        source = '\n'.join(["import os", *prelude, "", source])

    (out / "ladder_kernel.py").write_text(source, encoding="utf-8")
    metadata = {
        "id": f"{user}/{slug_for(smoke, stage)}",
        "title": slug_for(smoke, stage).replace("-", " ").title(),
        "code_file": "ladder_kernel.py",
        "language": "python",
        "kernel_type": "script",
        # Private. Kaggle rejects a public kernel (403 on SaveKernel) unless the
        # account is verified for it, and nothing here needs to be public -- the
        # code already lives in a public repo; this is just the runner.
        "is_private": True,
        "enable_gpu": True,
        # Ask for a T4 explicitly. Kaggle otherwise hands out P100s, which are
        # Pascal (sm_60), and current torch builds ship no kernels for them:
        # loading the model dies with "no kernel image is available for
        # execution on the device". T4 is Turing (sm_75) and is supported.
        "machine_shape": "NvidiaTeslaT4",
        # The kernel pip-installs unsloth and clones the repo.
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        # The prepared SFT data, built by the CPU kernel. Attaching it is what
        # keeps data prep out of the GPU session; without it the training kernel
        # refuses to run rather than quietly rebuilding the dataset here.
        "kernel_sources": sources_for(user, stage),
    }
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out


def push(user: str, smoke: bool = False, stage: str = "all") -> None:
    out = staging_dir(user, smoke, stage)
    slug = slug_for(smoke, stage)
    print(f"pushing {user}/{slug} (T4, internet on)")
    result = kaggle("kernels", "push", "-p", str(out), check=False)
    print(result.stdout.strip() or result.stderr.strip())
    if result.returncode != 0:
        sys.exit(result.returncode)
    print(f"\nrunning at https://www.kaggle.com/code/{user}/{slug}")


def status(user: str, slug: str = SLUG) -> str:
    result = kaggle("kernels", "status", f"{user}/{slug}", check=False)
    text = (result.stdout + result.stderr).strip()
    for state in ("complete", "error", "cancelAcknowledged", "running", "queued"):
        if state in text:
            return state
    return text


def fetch(user: str, slug: str = SLUG) -> int:
    """Download kernel output and print the result table."""
    out_dir = REPO_ROOT / "outputs" / "kaggle"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = kaggle("kernels", "output", f"{user}/{slug}", "-p", str(out_dir), check=False)
    if result.returncode != 0:
        print(result.stderr.strip())
        return 1

    status_file = out_dir / "status.json"
    if not status_file.exists():
        print(f"no status.json in kernel output; files: {[p.name for p in out_dir.iterdir()]}")
        return 1

    data = json.loads(status_file.read_text(encoding="utf-8"))
    print(f"\nelapsed: {data.get('elapsed_hours')} h")
    for name, info in data.get("stages", {}).items():
        mark = "ok " if info.get("ok") else "FAIL"
        print(f"  [{mark}] {name:<12} {info.get('minutes')} min  {info.get('error', '')}")

    base, tuned = data.get("base_metrics"), data.get("tuned_metrics")
    if base and tuned:
        print(f"\n{'model':<28} {'pass@1':>8}")
        print(f"{'Qwen2.5-Coder-3B (base)':<28} {base['pass@1']:>8.3f}")
        print(f"{'Ladder-3B':<28} {tuned['pass@1']:>8.3f}")
        print(f"{'delta':<28} {tuned['pass@1'] - base['pass@1']:>+8.3f}")
        return 0

    print("\nrun did not produce both evals -- see the stage table above")
    return 1


def wait(user: str, timeout_hours: float, slug: str = SLUG) -> int:
    deadline = time.time() + timeout_hours * 3600
    while time.time() < deadline:
        state = status(user, slug)
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp}] {state}", flush=True)
        if state == "complete":
            return fetch(user, slug)
        if state in ("error", "cancelAcknowledged"):
            fetch(user, slug)
            return 1
        time.sleep(POLL_SECONDS)
    print("timed out waiting; the kernel may still be running on Kaggle")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True, help="your Kaggle username")
    parser.add_argument("--status", action="store_true", help="print status and exit")
    parser.add_argument("--fetch", action="store_true", help="download results and exit")
    parser.add_argument("--no-wait", action="store_true", help="push and exit")
    parser.add_argument("--stage", choices=["train", "eval", "all"], default="all",
                        help="train and eval fit better as separate sessions")
    parser.add_argument("--smoke", action="store_true",
                        help="20 steps and 3 eval problems, to its own slug")
    parser.add_argument("--timeout-hours", type=float, default=13.0)
    args = parser.parse_args()

    require_credentials()

    slug = slug_for(args.smoke, args.stage)

    if args.status:
        print(status(args.user, slug))
        return 0
    if args.fetch:
        return fetch(args.user, slug)

    push(args.user, args.smoke, args.stage)
    if args.no_wait:
        return 0
    return wait(args.user, args.timeout_hours, slug)


if __name__ == "__main__":
    raise SystemExit(main())
