---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-3B-Instruct
library_name: peft
pipeline_tag: text-generation
tags:
  - competitive-programming
  - codeforces
  - qlora
  - code
datasets:
  - open-r1/codeforces-cots
language:
  - en
---

# {MODEL_NAME}

A QLoRA adapter for `Qwen2.5-Coder-3B-Instruct`, fine-tuned on verified
Codeforces reasoning traces. Trained and evaluated with
[Ladder](https://github.com/NiLabs-Models/ladder).

This is an **adapter**, not a merged model. Load it on top of the base.

## Results

Measured by executing generated programs against the problems' real test cases —
not by similarity to a reference solution, and not by a model judge.

| Model | Problems | pass@1 |
| --- | --- | --- |
| Qwen2.5-Coder-3B-Instruct (base) | {N_PROBLEMS} | {BASE_PASS1} |
| {MODEL_NAME} | {N_PROBLEMS} | {TUNED_PASS1} |

Held-out problems come from a hash-based split of `open-r1/codeforces-cots`,
so no evaluated problem appears in training. Greedy decoding.
Codeforces `checker` and interactive problems are excluded, since exact-match
scoring cannot grade them.

Reproduce:

```bash
git clone https://github.com/NiLabs-Models/ladder
ladder eval --config configs/ladder-3b-kaggle.yaml                    # base
ladder eval --config configs/ladder-3b-kaggle.yaml --adapter <this>   # tuned
```

## Training

| | |
| --- | --- |
| Method | QLoRA, 4-bit, via Unsloth |
| LoRA rank / alpha | {LORA_R} / {LORA_ALPHA} |
| Context length | {MAX_SEQ_LEN} |
| Optimizer steps | {MAX_STEPS} |
| Effective batch | {EFFECTIVE_BATCH} |
| Learning rate | {LR}, cosine |
| Hardware | {GPU} |
| Wall clock | {HOURS} h |

Loss is computed on the assistant turn only; prompt tokens are masked.

## Training data

`open-r1/codeforces-cots`, config `solutions_py_decontaminated`, filtered:

- Deduplicated by problem id
- Traces that hit the generation token ceiling dropped, not truncated
- **Solutions verified by execution** — each trace's program is run against its
  own problem's test cases and dropped if it fails. The traces are
  model-generated and are wrong a meaningful fraction of the time on hard
  problems; without this filter, training teaches confident wrong answers.

{N_TRAIN} traces after filtering.

## Limitations

- **Python only.** The training config uses the Python solution split; the model
  is not tuned for C++, which is what most competitive programmers actually use.
- **Not evaluated on `checker` or interactive problems**, which are a real slice
  of Codeforces.
- Small model, short run. This is a demonstration that the pipeline moves the
  number, not a state-of-the-art system.
- Generated code is untrusted. Run it in a sandbox.

## License

Apache-2.0, matching the base model. Training data is ODC-BY.
