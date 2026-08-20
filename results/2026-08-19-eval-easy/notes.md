# 2026-08-19 — first base vs tuned evaluation (inconclusive)

| model | pass@1 |
| --- | --- |
| Qwen2.5-Coder-3B-Instruct (base) | 0.050 |
| Ladder-3B | 0.050 |
| delta | +0.000 |

**This does not show that training had no effect.** The two models were not
measured on comparable terms, and the delta above should not be quoted.

| verdict | base | tuned |
| --- | --- | --- |
| `no_code` | **0** | **28** |
| `wrong_answer` | 31 | 10 |
| `runtime_error` | 7 | 0 |
| `accepted` | 2 | 2 |

The tuned model produced no code at all on **70%** of problems. The base model
on none. Same problems, same prompt, same decoding, same 4,096-token budget.

## Why

The training data is the cause. Traces in `codeforces-cots` have a median of
~13,770 estimated tokens, so the fine-tune was taught to reason at length — and
was then given 4,096 tokens to do it in. It is cut off mid-reasoning before
reaching a solution, and scores as producing nothing.

This was predicted before the run and under-corrected: the budget was raised
from 3,072 to 4,096 on exactly this reasoning, without checking what the trained
model's outputs actually needed. A 3× shortfall was papered over with a 33%
increase.

## What can and cannot be said

- **Cannot**: whether the fine-tune is better or worse than the base model. The
  measurement does not support a comparison in either direction.
- **Can**: training made the model substantially more verbose, and at this
  budget verbosity dominates everything else.
- **Cannot**: that the tuned model is "really" at 2/12 = 0.167. Those 12 are the
  problems needing least reasoning — the subset is selected by the very thing
  that biases it.

## Setup

| | |
| --- | --- |
| Problems | 40 held-out div2 A/B (`configs/eval-easy.yaml`) |
| Decoding | greedy, `max_new_tokens` 4,096 |
| Judged | generated programs run against the problems' real test cases |
| Hardware | Colab T4 |

The div2 A/B restriction was itself a fix: a uniform Codeforces sample gave a 3B
no measurable score at all. A pass@1 here is not comparable to one over the full
difficulty range.

## Next

Re-run with a budget matched to what the tuned model actually emits, measured
rather than guessed. Until then the README results table stays empty, because
there is no defensible number to put in it.
