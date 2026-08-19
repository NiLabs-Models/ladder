"""Report which training APIs actually exist on a Kaggle GPU box.

The training run died with `NameError: PreTrainedConfig`, which is a version
conflict: the kernel pinned `trl<0.12` to keep trl's completion-only collator,
and that pin drags transformers back to something unsloth 2025.9.1 cannot use.

Guessing at compatible version sets has a poor record in this project, and each
guess costs a GPU session to disprove. This installs the stack and reports what
is present, so the fix can be written once against facts.
"""

import json
import subprocess
import sys

report = {}


def note(key, value):
    report[key] = value
    print(f"{key}: {value}", flush=True)


# Round 1 pinned unsloth==2025.9.1 and found transformers 5.0.0 already on the
# image, which that unsloth predates -- it cannot even import. The pin was the
# problem: a version picked without checking whether it matches the environment.
# Round 2 asks for the current unsloth and lets it resolve its own stack.
print("installing: latest unsloth, letting it resolve transformers/trl/peft", flush=True)
subprocess.run("pip install -q -U unsloth unsloth_zoo", shell=True, check=False)
subprocess.run("pip install -q -U bitsandbytes", shell=True, check=False)

for mod in ("torch", "transformers", "trl", "peft", "unsloth", "accelerate", "datasets"):
    try:
        m = __import__(mod)
        note(f"version.{mod}", getattr(m, "__version__", "unknown"))
    except Exception as exc:
        note(f"version.{mod}", f"IMPORT FAILED: {type(exc).__name__}: {exc}")

# Which trainer API? Newer trl folds the training args into SFTConfig.
try:
    import trl
    note("has.SFTConfig", hasattr(trl, "SFTConfig"))
    note("has.SFTTrainer", hasattr(trl, "SFTTrainer"))
    note("has.DataCollatorForCompletionOnlyLM", hasattr(trl, "DataCollatorForCompletionOnlyLM"))
    import inspect
    if hasattr(trl, "SFTTrainer"):
        sig = list(inspect.signature(trl.SFTTrainer.__init__).parameters)
        note("SFTTrainer.params", sig)
    if hasattr(trl, "SFTConfig"):
        sig = list(inspect.signature(trl.SFTConfig.__init__).parameters)
        note("SFTConfig.has_dataset_text_field", "dataset_text_field" in sig)
        note("SFTConfig.has_max_seq_length", "max_seq_length" in sig)
        note("SFTConfig.has_max_length", "max_length" in sig)
except Exception as exc:
    note("trl.inspect", f"FAILED: {type(exc).__name__}: {exc}")

# Unsloth's own masking helper is the supported replacement for the collator.
try:
    from unsloth.chat_templates import train_on_responses_only  # noqa: F401
    note("has.train_on_responses_only", True)
except Exception as exc:
    note("has.train_on_responses_only", f"NO: {type(exc).__name__}: {exc}")

# And can the actual base model load at the context this project needs?
try:
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit",
        max_seq_length=8192, load_in_4bit=True, dtype=None,
    )
    note("model_loads", True)
    note("chat_template_present", bool(getattr(tok, "chat_template", None)))
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}],
        tokenize=False,
    )
    note("rendered_sample", rendered[:200])
    import torch
    note("vram_after_load_gb", round(torch.cuda.max_memory_allocated() / 1024**3, 2))
except Exception as exc:
    note("model_loads", f"FAILED: {type(exc).__name__}: {exc}")

with open("/kaggle/working/deps.json", "w") as fh:
    json.dump(report, fh, indent=2)
print("\nwrote /kaggle/working/deps.json", flush=True)
sys.stdout.flush()
