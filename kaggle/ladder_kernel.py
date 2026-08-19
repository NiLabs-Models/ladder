"""Kaggle kernel entry point: data -> base eval -> train -> tuned eval.

Runs unattended in one kernel session, so it is written to survive being killed:
every stage writes its result to /kaggle/working the moment it finishes, and the
adapter is checkpointed along the way. A session that dies during the tuned eval
still leaves the base number and the trained adapter behind.

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
DATA_DIR = f"{WORK}/data/sft"
STATUS_PATH = f"{WORK}/status.json"

STATUS = {"stages": {}, "started": time.time()}


def save_status(**kw):
    STATUS.update(kw)
    STATUS["elapsed_hours"] = round((time.time() - STATUS["started"]) / 3600, 3)
    with open(STATUS_PATH, "w") as fh:
        json.dump(STATUS, fh, indent=2)
    print(f"[status] {json.dumps(STATUS['stages'])}", flush=True)


def stage(name):
    """Run a stage, recording its outcome without letting a failure kill the rest."""
    def wrap(fn):
        started = time.time()
        print(f"\n{'=' * 70}\n[{name}] starting\n{'=' * 70}", flush=True)
        try:
            result = fn()
            STATUS["stages"][name] = {
                "ok": True,
                "minutes": round((time.time() - started) / 60, 1),
            }
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
            return None
    return wrap


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


# --------------------------------------------------------------------------
# 0. environment
# --------------------------------------------------------------------------
@stage("setup")
def _setup():
    sh("pip install -q -U 'unsloth==2025.9.1' 'unsloth_zoo==2025.9.1'")
    sh("pip install -q 'trl>=0.9.6,<0.12' 'peft>=0.12.0' 'bitsandbytes>=0.43.0'")
    if not os.path.isdir(REPO_DIR):
        sh(f"git clone -q {REPO} {REPO_DIR}")
    sys.path.insert(0, f"{REPO_DIR}/src")
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
    return True


_setup()
sys.path.insert(0, f"{REPO_DIR}/src")

from ladder.config import load_config  # noqa: E402

cfg = load_config(CONFIG)
cfg.train.output_dir = f"{WORK}/outputs/ladder-3b-kaggle"
save_status(config=cfg.to_dict())


# --------------------------------------------------------------------------
# 1. data
# --------------------------------------------------------------------------
@stage("build_data")
def _build():
    from ladder.data.build import build

    return build(cfg, DATA_DIR)


counts = _build()
save_status(data_counts=counts)


# --------------------------------------------------------------------------
# 2/4. evals -- same problems, same prompt, only the adapter differs
# --------------------------------------------------------------------------
def run_eval(adapter, label):
    import gc

    import torch

    from ladder.eval.runner import evaluate
    from ladder.infer import load_for_inference, make_generator

    cfg.eval.results_path = f"{WORK}/outputs/eval-{label}.json"
    model, tok = load_for_inference(cfg, adapter)
    try:
        summary = evaluate(make_generator(model, tok, cfg), cfg)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return summary


@stage("eval_base")
def _eval_base():
    return run_eval(None, "base")


base = _eval_base()
if base:
    save_status(base_metrics=base["metrics"], base_verdicts=base["verdicts"])


# --------------------------------------------------------------------------
# 3. train
# --------------------------------------------------------------------------
@stage("train")
def _train():
    from ladder.train.sft import train

    return train(cfg, DATA_DIR)


adapter_dir = _train()


@stage("eval_tuned")
def _eval_tuned():
    if not adapter_dir:
        raise RuntimeError("training did not produce an adapter")
    return run_eval(adapter_dir, "tuned")


tuned = _eval_tuned()
if tuned:
    save_status(tuned_metrics=tuned["metrics"], tuned_verdicts=tuned["verdicts"])


# --------------------------------------------------------------------------
# 5. the answer
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
