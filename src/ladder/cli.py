"""`ladder` command line entry point.

Subcommands split along the lines of what needs a GPU:
  build-data  CPU only
  train       GPU
  eval        GPU to generate, CPU to judge
  judge       CPU only -- rescore saved generations without reloading a model
"""

from __future__ import annotations

import argparse
import sys

from ladder.config import load_config


def _cmd_build_data(args) -> int:
    from ladder.data.build import build

    cfg = load_config(args.config)
    tokenizer = None
    if args.use_tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.model.base_model)
    build(cfg, args.out, tokenizer=tokenizer)
    return 0


def _cmd_train(args) -> int:
    from ladder.train.sft import train

    cfg = load_config(args.config)
    if args.max_steps:
        cfg.train.max_steps = args.max_steps
    if args.output_dir:
        cfg.train.output_dir = args.output_dir
    train(cfg, args.data)
    return 0


def _cmd_eval(args) -> int:
    from ladder.eval.runner import evaluate
    from ladder.infer import load_for_inference, make_generator

    cfg = load_config(args.config)
    if args.num_problems:
        cfg.eval.num_problems = args.num_problems
    if args.samples:
        cfg.eval.samples_per_problem = args.samples
    if args.results:
        cfg.eval.results_path = args.results

    model, tokenizer = load_for_inference(cfg, args.adapter)
    evaluate(make_generator(model, tokenizer, cfg), cfg)
    return 0


def _cmd_judge(args) -> int:
    """Rescore a JSONL of `{"problem_id", "completion"}` against the test suite."""
    import json

    from ladder.data.formatting import extract_code
    from ladder.eval.sandbox import Verdict, judge
    from ladder.eval.tasks import load_problems

    cfg = load_config(args.config)
    problems = {p.problem_id: p for p in load_problems(cfg.eval, cfg.data)}

    solved = total = 0
    with open(args.generations, encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            problem = problems.get(record["problem_id"])
            if problem is None:
                print(f"skip: {record['problem_id']} not in eval set", file=sys.stderr)
                continue
            total += 1
            code = extract_code(record.get("completion", ""))
            verdict = Verdict.NO_CODE
            if code:
                verdict, _ = judge(
                    code,
                    problem.tests,
                    cfg.eval.timeout_seconds,
                    cfg.eval.memory_limit_mb,
                )
            solved += verdict is Verdict.ACCEPTED
            print(f"{record['problem_id']}\t{verdict.value}")

    if total:
        print(f"\nsolved {solved}/{total} = {solved / total:.3f}", file=sys.stderr)
    return 0


def _cmd_show_config(args) -> int:
    import json

    print(json.dumps(load_config(args.config).to_dict(), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ladder", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-data", help="build train/val JSONL from the source dataset")
    p.add_argument("--config", required=True)
    p.add_argument("--out", default="data/sft")
    p.add_argument(
        "--use-tokenizer",
        action="store_true",
        help="count real tokens instead of estimating (slower, needs transformers)",
    )
    p.set_defaults(func=_cmd_build_data)

    p = sub.add_parser("train", help="run QLoRA SFT")
    p.add_argument("--config", required=True)
    p.add_argument("--data", default="data/sft")
    p.add_argument("--output-dir")
    p.add_argument("--max-steps", type=int, help="override for a smoke run")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("eval", help="generate solutions and judge them")
    p.add_argument("--config", required=True)
    p.add_argument("--adapter", help="adapter dir; omit to score the untuned base model")
    p.add_argument("--num-problems", type=int)
    p.add_argument("--samples", type=int)
    p.add_argument("--results")
    p.set_defaults(func=_cmd_eval)

    p = sub.add_parser("judge", help="rescore saved generations, no GPU needed")
    p.add_argument("--config", required=True)
    p.add_argument("--generations", required=True, help="JSONL of {problem_id, completion}")
    p.set_defaults(func=_cmd_judge)

    p = sub.add_parser("show-config", help="print the fully resolved config")
    p.add_argument("--config", required=True)
    p.set_defaults(func=_cmd_show_config)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
