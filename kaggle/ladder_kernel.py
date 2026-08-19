"""Kaggle kernel entry point: preflight -> data -> base eval -> train -> tuned eval.

Runs unattended in one kernel session, so it is written to survive being killed:
every stage writes its result to /kaggle/working/status.json the moment it
finishes, and the adapter is checkpointed as it trains. A session that dies
during the tuned eval still leaves the base number and the adapter behind.

Stage order is deliberate. The base eval runs BEFORE training because a tuned
number with nothing to compare it to is not a result, and the base eval is the
cheapest stage to lose if the clock runs out.
"""

import json
import os
import subprocess
import sys
import time
import traceback

REPO = "https://github.com/NiLabs-Models/ladder.git"
# Cloned outside /kaggle/working on purpose: everything under working becomes
# kernel output, so cloning there put the whole repo in every download and made
# fetching a 2KB status.json take minutes. Output should be results only.
REPO_DIR = "/tmp/ladder"
WORK = "/kaggle/working"
CONFIG = f"{REPO_DIR}/configs/ladder-3b-kaggle.yaml"
STATUS_PATH = f"{WORK}/status.json"

# Data comes from the CPU kernel declared in kernel_sources, already built and
# verified. Building it here is what cost the first run its entire session: all
# twelve hours went to CPU work with the GPU idle, and it still did not finish.
#
# The mount path is searched rather than assumed. Guessing it as
# /kaggle/input/<slug>/data/sft was wrong and cost a run, and the exact layout
# is Kaggle's to decide, not ours.
# Which half of the pipeline this session runs. Measured on a T4, training is
# ~5.4h and the two evals are ~5.7h; together that is 11.2h against a 12h cap,
# which is the same thin margin that lost the first run. Splitting them puts
# each comfortably inside the cap and, more importantly, banks the adapter
# before evaluation is attempted -- a slow eval can no longer take the trained
# model down with it.
#
#   train  attach data, train, save the adapter
#   eval   attach a trained adapter, score base and tuned
#   all    everything in one session (what the smoke run uses)
STAGE = os.environ.get("LADDER_STAGE", "all")


# Mount discovery lives in ladder.kaggle_paths, which is importable and tested.
# It is resolved after setup() clones the repo, not at module import.
DATA_DIR = None

STATUS = {"stages": {}, "started": time.time()}


def save_status(**kw):
    STATUS.update(kw)
    STATUS["elapsed_hours"] = round((time.time() - STATUS["started"]) / 3600, 3)
    with open(STATUS_PATH, "w") as fh:
        json.dump(STATUS, fh, indent=2)
    print(f"[status] {json.dumps(STATUS['stages'])}", flush=True)


def run_stage(name, fn, required=True):
    """Run one stage, recording its outcome.

    A plain function, not a decorator: the decorator version rebound each stage
    name to its return value, so the following call tried to invoke the result.

    `required` stages abort the run. Continuing past a failed setup just burns a
    session slot failing every later stage for the same reason.
    """
    started = time.time()
    print(f"\n{'=' * 70}\n[{name}] starting\n{'=' * 70}", flush=True)
    try:
        result = fn()
        STATUS["stages"][name] = {"ok": True, "minutes": round((time.time() - started) / 60, 1)}
        save_status()
        return result
    except Exception as exc:
        STATUS["stages"][name] = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "minutes": round((time.time() - started) / 60, 1),
        }
        save_status()
        traceback.print_exc()
        if required:
            print(f"\n[{name}] is required -- aborting instead of burning the session.", flush=True)
            sys.exit(1)
        return None


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


# --------------------------------------------------------------------------
# 0. preflight -- fail in seconds, not after a 4-minute pip timeout
# --------------------------------------------------------------------------
def preflight():
    """Check the two entitlements this run cannot proceed without.

    Kaggle silently downgrades `enable_internet` and `enable_gpu` when the
    account is not verified for them, so the kernel starts and then dies four
    minutes later inside pip with a DNS error. Check both up front and say
    plainly which one is missing.
    """
    import socket

    try:
        socket.setdefaulttimeout(10)
        socket.getaddrinfo("pypi.org", 443)
        net = True
    except OSError as exc:
        net = False
        print(f"no internet: {exc}", flush=True)

    gpu = subprocess.run(
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
        shell=True, capture_output=True, text=True,
    )
    gpu_name = gpu.stdout.strip() if gpu.returncode == 0 else ""
    print(f"internet: {net} | gpu: {gpu_name or 'NONE'}", flush=True)
    save_status(internet=net, gpu=gpu_name)

    if not net:
        raise RuntimeError(
            "kernel has no internet. Enable it in the notebook settings; it "
            "requires a phone-verified Kaggle account, and Kaggle downgrades the "
            "flag silently when the account is not verified."
        )
    if not gpu_name:
        raise RuntimeError("kernel has no GPU. Set the accelerator to GPU T4 x2.")
    return gpu_name


run_stage("preflight", preflight)


# --------------------------------------------------------------------------
# 1. environment
# --------------------------------------------------------------------------
# Keep library caches out of /kaggle/working: everything there becomes kernel
# output, and HF tokenizer + unsloth compile caches are tens of MB that make
# fetching a small status.json slow. Results only.
# Assigned, not setdefault: Kaggle sets these itself, pointing into
# /kaggle/working, so setdefault was a no-op and the training run still shipped
# 800MB of tokenizer cache as "output".
os.environ["HF_HOME"] = "/tmp/hf"
os.environ["HF_HUB_CACHE"] = "/tmp/hf/hub"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/hf/transformers"
os.environ["HF_DATASETS_CACHE"] = "/tmp/hf/datasets"
os.environ["UNSLOTH_CACHE_DIR"] = "/tmp/unsloth"


def setup():
    # Unpinned on purpose. Pinning unsloth==2025.9.1 broke a run outright:
    # Kaggle ships transformers 5.0.0, which that release predates badly enough
    # that it cannot import. Let unsloth resolve its own stack, and pin trl or
    # transformers only against a combination actually observed working.
    sh("pip install -q -U unsloth unsloth_zoo")
    sh("pip install -q -U bitsandbytes")
    if not os.path.isdir(REPO_DIR):
        sh(f"git clone -q {REPO} {REPO_DIR}")
    return True


run_stage("setup", setup)
sys.path.insert(0, f"{REPO_DIR}/src")

from ladder.kaggle_paths import (  # noqa: E402
    describe_mounts,
    find_adapter,
    find_prepared_data,
)

DATA_DIR = find_prepared_data() or f"{WORK}/data/sft"

from ladder.config import load_config  # noqa: E402

cfg = load_config(CONFIG)
cfg.train.output_dir = f"{WORK}/outputs/ladder-3b-kaggle"

# Smoke mode runs the real model on the real data for a handful of steps. It
# exists to prove the chain end to end -- masking, training, saving, loading the
# adapter back, judging -- before a full session is committed to it. Every
# failure so far was found in the first minutes of a run; this makes those
# minutes cheap to buy on purpose.
if os.environ.get("LADDER_SMOKE") == "1":
    cfg.name = "smoke"
    cfg.train.max_steps = 20
    cfg.train.save_steps = 20
    cfg.train.logging_steps = 1
    cfg.eval.num_problems = 3
    cfg.eval.max_new_tokens = 512
    cfg.train.output_dir = f"{WORK}/outputs/smoke"
    print("SMOKE MODE: 20 steps, 3 eval problems", flush=True)

print(f"stage: {STAGE}", flush=True)
save_status(config=cfg.to_dict(), stage=STAGE,
            smoke=os.environ.get("LADDER_SMOKE") == "1")


# --------------------------------------------------------------------------
# 2. data -- attached, not built
# --------------------------------------------------------------------------
def use_prepared_data():
    """Use the prepared dataset, and refuse to build it here if it is missing.

    Building costs hours of CPU work, and this session's clock is GPU time.
    Failing loudly with instructions is strictly better than silently spending
    the whole session on data prep -- which is exactly what happened the first
    time.
    """
    train_file = os.path.join(DATA_DIR, "train.jsonl")
    if not os.path.exists(train_file):
        # Show what IS mounted. A failure here is nearly always a kernel_sources
        # wiring problem, and the listing says immediately which.
        listing = describe_mounts()
        raise RuntimeError(
            "no prepared data found. Looked for train.jsonl under "
            f"/kaggle/input and fell back to {DATA_DIR}.\n"
            f"Mounted:\n{listing}\n"
            "Run the CPU kernel natedemoss/ladder-build-data and add it to "
            "this kernel's kernel_sources. Data prep does not belong in a "
            "GPU session."
        )
    with open(train_file, encoding="utf-8") as fh:
        n_train = sum(1 for _ in fh)
    val_file = os.path.join(DATA_DIR, "val.jsonl")
    n_val = 0
    if os.path.exists(val_file):
        with open(val_file, encoding="utf-8") as fh:
            n_val = sum(1 for _ in fh)
    print(f"using prepared data: {n_train} train / {n_val} val from {DATA_DIR}", flush=True)
    return {"train": n_train, "val": n_val}


if STAGE in ("train", "all"):
    counts = run_stage("attach_data", use_prepared_data)
    save_status(data_counts=counts)
else:
    # Evaluation reads its problems from the dataset, not from train.jsonl.
    print("eval stage: no training data needed", flush=True)


# --------------------------------------------------------------------------
# 3/5. evals -- same problems, same prompt, only the adapter differs
# --------------------------------------------------------------------------
def run_eval(adapter, label):
    import gc

    import torch

    from ladder.eval.runner import evaluate
    from ladder.infer import load_for_inference, make_generator

    cfg.eval.results_path = f"{WORK}/outputs/eval-{label}.json"
    model, tok = load_for_inference(cfg, adapter)
    try:
        return evaluate(make_generator(model, tok, cfg), cfg)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


# Not required: if the base eval dies we still want the trained adapter out of
# this session, and the base number can be recomputed on CPU-cheap hardware later.
base = None
if STAGE in ("eval", "all"):
    base = run_stage("eval_base", lambda: run_eval(None, "base"), required=False)
if base:
    save_status(base_metrics=base["metrics"], base_verdicts=base["verdicts"])


# --------------------------------------------------------------------------
# 4. train
# --------------------------------------------------------------------------
def train_model():
    from ladder.train.sft import train

    return train(cfg, DATA_DIR)


adapter_dir = None
if STAGE in ("train", "all"):
    adapter_dir = run_stage("train", train_model)
else:
    adapter_dir = find_adapter()
    if adapter_dir is None:
        raise SystemExit(
            "eval stage needs a trained adapter. Run the train kernel first "
            "and add it to this kernel's kernel_sources."
        )
    print(f"using adapter from {adapter_dir}", flush=True)
    save_status(adapter_dir=adapter_dir)

tuned = None
if STAGE in ("eval", "all"):
    tuned = run_stage("eval_tuned", lambda: run_eval(adapter_dir, "tuned"), required=False)
if tuned:
    save_status(tuned_metrics=tuned["metrics"], tuned_verdicts=tuned["verdicts"])


# --------------------------------------------------------------------------
# 6. the answer
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"LADDER RESULTS (stage: {STAGE})")
print("=" * 70)
if STAGE == "train":
    print(f"adapter -> {adapter_dir}")
    print("now run the eval kernel with this one in its kernel_sources")
if base and tuned:
    b, t = base["metrics"]["pass@1"], tuned["metrics"]["pass@1"]
    print(f"problems      : {base['n_problems']}")
    print(f"base  pass@1  : {b:.3f}")
    print(f"ladder pass@1 : {t:.3f}")
    print(f"delta         : {t - b:+.3f}")
    print(f"base verdicts : {base['verdicts']}")
    print(f"tuned verdicts: {tuned['verdicts']}")
else:
    print("incomplete -- see status.json for which stage failed")
save_status(finished=True)
print("=" * 70)
