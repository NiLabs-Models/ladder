# 2026-08-18 — first GPU run, failed

No model. Recorded because a project whose whole point is auditable numbers
should not quietly drop the run that produced none.

| | |
| --- | --- |
| Kernel | `natedemoss/ladder-qlora-codeforces` |
| Hardware | Tesla P100-PCIE-16GB, 4 CPU, 32GB RAM |
| Config | `configs/ladder-3b-kaggle.yaml` @ `66cbee8` |
| Wall clock | ~12h, killed at Kaggle's session cap |
| Reached | data prep only — never trained, never evaluated |
| Produced | 446 of a targeted 4500 examples |

## What went wrong

**1. Data prep ran inside the GPU session.** The README already said data prep is
CPU-only and should be done anywhere else; the kernel did it at line 146 of its
own script. Twelve hours of GPU quota went to CPU work with the accelerator
idle. This alone would have failed the run.

**2. The length bound was set from a biased sample.** `max_tokens: 5120` came
from 25 rows read at offset 0 — easy early problems, median ~2,700 estimated
tokens. The corpus median is 13,770 and p90 is 23,400, so the bound kept 17% of
the data and `max_samples: 4500` was never reachable at all: the whole dataset
yields ~1,350 rows at that bound.

**3. Something hung.** 446 examples were written and then nothing for twelve
hours. The rate was normal and went to zero.

## Diagnosis of the hang

Reported as "93 seconds per example" at first. That was wrong — it is 12h ÷ 446,
a total divided by a count, not a measured rate.

Measured afterwards, all on Kaggle unless noted:

| Test | Result |
| --- | --- |
| CPU kernel, all components | 0.21 s/example projected |
| **GPU kernel, full pipeline, 20 min** | **0.59 s/example, stable, 29GB free** |
| GPU kernel, old vs new sandbox, 60 traces | 0.12 vs 0.08 s/trace |
| GitHub Linux CI, threaded verification | within budget |
| `preexec_fn` serial spawn ×10 | 0.107s, same as a bare spawn |
| `RLIMIT_NPROC` on Kaggle | unlimited; container user has 3 processes |

Every steady-state explanation is dead, including three I proposed and one I
pushed a commit asserting before testing it. The GPU instance runs the same
pipeline fine.

The mechanism that fits is an unbounded hang. `subprocess.Popen.__init__` blocks
reading the child's exec-status pipe with no timeout. `preexec_fn` runs in the
child between fork and exec, where it can deadlock on a lock another thread held
at fork time — the documented reason it is unsafe with threads. The parent then
blocks *before* `subprocess.run`'s timeout is reachable, so the timeout never
fires. Probabilistic, which is why runs of 60–2,000 traces never reproduced it.

**Not proven.** A probabilistic hang cannot be confirmed by a run that does not
hang. See #7.

## Changes this produced

- `preexec_fn` removed; rlimits applied by a bootstrap inside the child,
  isolation via `start_new_session`, timeout kills the process group
- `RLIMIT_NPROC` dropped — a per-*user* limit is wrong inside a shared container
- Data prep moved to a CPU kernel; the training kernel now refuses to build data
- `max_tokens` raised to 8192 from the measured distribution, not a sample
- Training records `throughput.json`, so the next run is sized by arithmetic

## Cost

~12h of a 30h weekly GPU quota, for no model.
