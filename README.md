# Ladder

Small open models fine-tuned for competitive programming, trained on free GPUs.

Ladder is a QLoRA fine-tuning and evaluation pipeline for Codeforces-style
problems. It is built around one constraint: **everything has to run on a free
Kaggle or Colab T4.** That rules out full fine-tuning, bf16, and flash-attention,
and it drives most of the design decisions below.

The eval is verifiable. Models are not scored by another model, by BLEU against a
reference solution, or by whether the output looks like code. A generated program
is run against the problem's actual test cases, and it either passes them or it
does not.

> **Status:** pipeline complete and tested; no trained checkpoint published yet.
> The results table below is empty on purpose. It gets filled in from a real run,
> not from an estimate.

## What it does

```
open-r1/codeforces-cots  ──build-data──▶  train.jsonl / val.jsonl
                                                │
                                                ├──train──▶  LoRA adapter
                                                │
held-out problems + tests  ─────────eval────────┴──▶  pass@k
```

- **Data** — [`open-r1/codeforces-cots`](https://huggingface.co/datasets/open-r1/codeforces-cots),
  the `solutions_py_decontaminated` config: Codeforces problems with reasoning
  traces, decontaminated against common benchmarks. Each row also carries public,
  private, and generated test cases, which is what makes the eval verifiable
  without a second dataset — and what makes the filtering below possible.
- **Verified training data** — the traces are model-generated, so some of them
  are wrong. Every trace's solution is run against its own problem's tests during
  data prep, and the failures are dropped. See below.
- **Training** — 4-bit QLoRA via [Unsloth](https://github.com/unslothai/unsloth)
  on `Qwen2.5-Coder-3B-Instruct`. Loss is computed on the assistant turn only.
- **Eval** — generate a program, run it against up to 20 test cases in a
  subprocess with time and memory limits, report pass@k.

## The training data is not all correct

The reasoning traces in `codeforces-cots` are model output, not verified
solutions. Running them against the problems' own test cases:

| Sample | Traces whose solution actually passes |
| --- | --- |
| Easy problems (div2 A/B) | 16/16 |
| Mixed sample including div1 D/E/G/H | 36/50 |

Every failure was a wrong answer on a hard problem, not a crash or a timeout —
these are confident, well-structured, incorrect solutions, and they are
concentrated on exactly the problems worth learning from.

`verify_solutions: true` (on by default in both shipped configs) runs the eval's
judge across the training set during data prep and drops the traces that fail.
It costs about eight minutes of CPU for the full corpus and no GPU time at all,
since data prep runs before you ever start a GPU session.

Problems where exact-match judging cannot decide correctness — Codeforces
`checker` problems with many valid outputs, and problems shipping no tests — are
kept rather than discarded, since failing them would throw away correct traces
for being right in a different way. Set `verify_keep_unverifiable: false` to
train on the verified subset only.

## Quickstart

Data prep is CPU-only, so do it first and do it anywhere:

```bash
pip install -e .
ladder build-data --config configs/smoke-1.5b.yaml --out data/smoke
```

Then on a GPU box:

```bash
pip install -r requirements-train.txt
ladder train --config configs/smoke-1.5b.yaml --data data/smoke
ladder eval  --config configs/smoke-1.5b.yaml --adapter outputs/smoke-1.5b
```

`configs/smoke-1.5b.yaml` is a 20-step run on a 1.5B base. It exists to prove the
whole pipeline works end to end in about fifteen minutes, **before** you spend
free GPU hours on `configs/ladder-3b-t4.yaml`. Run it first.

For the real thing, [`notebooks/kaggle_ladder.ipynb`](notebooks/kaggle_ladder.ipynb)
is a ready-to-run Kaggle notebook.

## Commands

| Command | GPU | Notes |
| --- | --- | --- |
| `ladder build-data` | no | Streams and filters the source dataset into JSONL |
| `ladder train` | yes | QLoRA SFT; `--max-steps` for a smoke run |
| `ladder eval` | yes | Generate and judge; omit `--adapter` to score the base model |
| `ladder judge` | no | Rescore saved generations without reloading a model |
| `ladder show-config` | no | Print the fully resolved config |

`build-data` is where verification happens, which is why it is the slow CPU step
and why it is worth running once and reusing the JSONL across training runs.

Every knob lives in `configs/*.yaml`, and an unrecognized key is an error rather
than a silent no-op — a typo'd `learning_rat` should not cost you a four-hour run.

## Results

Baseline and fine-tune, same 100 held-out problems, same prompt, greedy decoding.

| Model | pass@1 | pass@5 |
| --- | --- | --- |
| Qwen2.5-Coder-3B-Instruct (base) | — | — |
| Ladder-3B | — | — |

Reproduce with:

```bash
ladder eval --config configs/ladder-3b-t4.yaml                                  # base
ladder eval --config configs/ladder-3b-t4.yaml --adapter outputs/ladder-3b-t4   # tuned
```

Held-out problems are selected by hashing the problem id with the training seed,
so the eval set is exactly the split that data prep withheld. It is not a
separate sample that might overlap with training.

## Design notes

**Why 3B.** A 16GB T4 fits a 3B base in 4-bit with an 8192-token context and room
for activations. 7B fits only by cutting context to about 2048, which truncates
most reasoning traces in this dataset — the wrong trade for a task whose training
signal *is* the reasoning.

**Why traces get dropped, not truncated.** A trace longer than `max_tokens` is
discarded. Truncating it would teach the model to stop mid-thought, which is a
worse failure than never seeing the example.

**Why the split is a hash.** `split_key(problem_id, seed)` is stable across runs,
machines, and upstream dataset refreshes. Shuffling with a seed is not: add rows
upstream and a held-out problem can quietly migrate into training.

**Why dedup by problem.** The corpus has several traces per problem. Without
dedup, the same problem lands in both sides of the split and the eval reports a
number that is partly memorization.

**Why the eval skips `checker` problems.** Their output is not unique — `no` and
`NO` are both right, and so is any valid arrangement. Exact-match scoring reports
correct solutions as wrong on them, which would drag every pass@k number down for
a reason that has nothing to do with the model. `EvalConfig.problem_types`
controls this.

**Why output comparison is case-insensitive.** Codeforces accepts `Yes`, `yes`
and `YES` interchangeably, and reference solutions genuinely do print a different
case than the expected-output file. The tolerance is limited to purely alphabetic
tokens, so a grid line like `C.C.C` is still compared exactly.

**Why fp16 and not bf16.** T4 and P100 are pre-Ampere and have no bf16 units.
`train/sft.py` picks based on `torch.cuda.is_bf16_supported()`, so the same
config runs on a T4 and on an A100.

## Security

`eval/sandbox.py` executes model-generated code. It runs each program in a
subprocess with a wall-clock timeout, POSIX rlimits on address space, process
count, and file size, in an isolated interpreter (`-I -S`) and a temp directory.

**This is a robustness boundary, not a security boundary.** It stops an
accidental infinite loop or a 20GB allocation. It does not stop deliberately
hostile code, which can still reach the filesystem and the network. Run evals
inside a container or a disposable VM. Kaggle and Colab already give you one,
which is the environment this is designed for.

## License

Apache-2.0. The upstream dataset and base models carry their own licenses —
`open-r1/codeforces-cots` is ODC-BY, `Qwen2.5-Coder` is Apache-2.0.
