# Results

One directory per run, holding the artifacts the run actually produced. A pass@1
in a README table is a claim; these are the evidence behind it.

```
results/<run-name>/
  eval-base.json      # untuned base model, same problems, same prompt
  eval-tuned.json     # the fine-tune
  throughput.json     # tokens/sec, peak VRAM, wall clock
  run_config.json     # the fully resolved config, including the git SHA
  notes.md            # hardware, anything unusual
```

Committed on purpose. They are small (tens of KB), they make a published number
auditable by someone who was not there, and they accumulate into a record of how
the pipeline behaves across hardware — which is the thing this project is
shortest on.

Add a run with `scripts/record_run.py`, which pulls the files out of a Kaggle
kernel's output and refuses to record a run whose base and tuned evals do not
cover the same problem set. Comparing two different problem sets is the easiest
way to publish a number that means nothing.

## Runs

_None yet. The first is training._
