# 2026-08-19 — training run (adapter produced, not yet evaluated)

The first Ladder adapter. **No pass@1 yet**: the GPU quota ran out before the
evaluation session could start, so there is nothing here to compare against the
base model. The results table in the README stays empty until there is.

| | |
| --- | --- |
| Kernel | `natedemoss/ladder-train` |
| Hardware | Tesla T4 16GB |
| Base | `unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit` |
| Data | 2,271 train / 85 val, verified by execution |
| Steps | 150 (effective batch 16 → 2,400 sequences, ~1 epoch) |
| Context | 8,192 |
| Wall clock | **5.05 h** |

## Measured throughput

| | |
| --- | --- |
| Tokens/sec | **594.9** |
| Total tokens | 10.8M |
| Mean tokens/example | 4,762 |
| Seconds/example | 8.0 |
| Peak VRAM | **6.78 GB of 15.3** |

This is the number the project spent the whole of the previous day guessing at.
The estimate in use before this run was 800–1200 tok/s, wrong by up to 2×, and
every time budget quoted from it was wrong in proportion. Sized from the smoke
run's 573 tok/s, this run was projected at 5.4h and took 5.05h.

Peak VRAM is under half the card. A larger batch or a longer context both have
room, and 8,192 was chosen as a guess at the ceiling rather than a measurement
of it — see #10, where the length bound costs 70% of the corpus.

## What stopped it here

GPU quota. The account allows **6 h/week**, not the 30 h quoted from Kaggle's
advertised free tier throughout this project, and Kaggle checks quota at session
*start* — so the earlier 12-hour failed run was able to begin with quota
available and then overrun it, consuming the week. Quota refreshes 2026-08-22.

`scripts/kaggle_run.py` now queries the quota API before pushing and warns when
a stage does not fit. That API was available the whole time.

## Reproduce

```bash
python scripts/kaggle_run.py --user <you> --stage train   # ~5h on a T4
python scripts/kaggle_run.py --user <you> --stage eval    # ~5.7h, needs the above
```
