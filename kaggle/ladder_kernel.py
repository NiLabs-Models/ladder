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
REPO_DIR = "/kaggle/working/ladder"
WORK = "/kaggle/working"
CONFIG = f"{REPO_DIR}/configs/ladder-3b-kaggle.yaml"
STATUS_PATH = f"{WORK}/status.json"

# Data comes from the CPU kernel declared in kernel_sources, already built and
# verified. Building it here is what cost the first run its entire session: all
# twelve hours went to CPU work with the GPU idle, and it still did not finish.
PREPARED_DATA = "/kaggle/input/ladder-build-data/data/sft"
DATA_DIR = PREPARED_DATA if os.path.isdir(PREPARED_DATA) else f"{WORK}/data/sft"

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

from ladder.config import load_config  # noqa: E402

cfg = load_config(CONFIG)
cfg.train.output_dir = f"{WORK}/outputs/ladder-3b-kaggle"
save_status(config=cfg.to_dict())


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
        raise RuntimeError(
            f"no prepared data at {DATA_DIR}. Run the CPU kernel "
            "natedemoss/ladder-build-data first and add it to this kernel's "
            "kernel_sources. Data prep does not belong in a GPU session."
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


counts = run_stage("attach_data", use_prepared_data)
save_status(data_counts=counts)


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
base = run_stage("eval_base", lambda: run_eval(None, "base"), required=False)
if base:
    save_status(base_metrics=base["metrics"], base_verdicts=base["verdicts"])


# --------------------------------------------------------------------------
# 4. train
# --------------------------------------------------------------------------
def train_model():
    from ladder.train.sft import train

    return train(cfg, DATA_DIR)


adapter_dir = run_stage("train", train_model)

tuned = run_stage("eval_tuned", lambda: run_eval(adapter_dir, "tuned"), required=False)
if tuned:
    save_status(tuned_metrics=tuned["metrics"], tuned_verdicts=tuned["verdicts"])


# --------------------------------------------------------------------------
# 6. the answer
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
print("LADDER RESULTS")
print("=" * 70)
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
