# Contributing

## Setup

```bash
pip install -e ".[dev]"
pytest -q
ruff check .
```

The test suite is CPU-only and takes a few seconds. Nothing in `tests/` loads a
model or downloads a dataset, and it should stay that way: the point is that you
can verify a change to the data pipeline or the judge without a GPU.

## Layout

| Path | Needs a GPU | What it does |
| --- | --- | --- |
| `src/ladder/config.py` | no | Typed run config, loaded from `configs/*.yaml` |
| `src/ladder/data/` | no | Prompt formatting, filtering, JSONL build |
| `src/ladder/train/` | yes | Unsloth QLoRA SFT |
| `src/ladder/eval/` | judging is CPU | Test-case execution, verdicts, pass@k |
| `src/ladder/infer.py` | yes | Adapter loading, generation |

## Things worth knowing before you change something

**`build_problem_prompt` is a contract.** Training and eval both call it, so
changing the prompt format invalidates every existing checkpoint's numbers.
If you change it, say so in the PR and re-run the eval.

**The sandbox is not a security boundary.** It stops runaway loops and
allocations, not hostile code. Do not add a claim to the docs that it does more
than that, and do not run evals outside a container or a disposable VM.

**Judging normalizes like a real judge.** Trailing whitespace and float
formatting are not wrong answers. If you tighten `outputs_match`, you will start
reporting correct solutions as failures.

## Adding a base model

Ship a config in `configs/` rather than changing defaults. If the model is not
Qwen/ChatML, `train_on_completions_only` needs that family's turn markers in
`train/sft.py` -- the current ones are `<|im_start|>user` and
`<|im_start|>assistant`.

## Reporting results

Include the config, the git SHA, the GPU, and `outputs/**/eval/results.json`.
A pass@1 number without the problem set it was measured on is not a result.

## The judge has to be right before the model matters

Every number this project reports depends on the judge being correct, so changes
to `eval/sandbox.py` need evidence, not just green tests. The check that catches
real bugs is running the dataset's own trace solutions through the judge and
confirming the pass rate is what you expect (near-100% on easy problems). Three
harness bugs were found that way, each of which would have silently under-
reported every result:

- Passing `-S` to the child interpreter removed the `exit()` builtin, so any
  solution that bailed out early became a runtime error.
- `checker` problems have many valid outputs and cannot be scored by exact match.
- Codeforces accepts `Yes`/`yes`/`YES` interchangeably; exact match did not.

If you change the judge, run that check and put the numbers in the PR.
