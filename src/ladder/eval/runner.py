"""Generate solutions, judge them, report pass@k."""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ladder.config import RunConfig
from ladder.data.formatting import extract_code
from ladder.eval.sandbox import Verdict, judge
from ladder.eval.tasks import EvalProblem, load_problems


def _log(message: str) -> None:
    """Write progress to stderr without dying on the console's encoding.

    Problem titles carry math symbols, and a Windows console defaulting to
    cp1252 raises UnicodeEncodeError on the first one. Losing a character in a
    progress line is fine; losing a multi-hour eval to it is not.
    """
    encoding = getattr(sys.stderr, "encoding", None) or "utf-8"
    safe = message.encode(encoding, errors='replace').decode(encoding)
    print(safe, file=sys.stderr)


@dataclass
class ProblemResult:
    problem_id: str
    title: str
    verdicts: list[str] = field(default_factory=list)
    tests_passed: list[int] = field(default_factory=list)
    n_tests: int = 0

    @property
    def solved(self) -> bool:
        return Verdict.ACCEPTED.value in self.verdicts


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from Chen et al. 2021 (the Codex paper).

    `n` samples drawn, `c` of them correct. Computed as 1 - C(n-c,k)/C(n,k) via a
    product so it stays numerically stable, instead of the naive c/n which
    over-reports for k > 1.
    """
    if k > n:
        raise ValueError(f"pass@{k} needs at least {k} samples, got {n}")
    if n - c < k:
        return 1.0
    prob = 1.0
    for i in range(n - c + 1, n + 1):
        prob *= 1.0 - k / i
    return 1.0 - prob


def evaluate(
    generate,
    cfg: RunConfig,
    problems: list[EvalProblem] | None = None,
) -> dict:
    """Run the eval loop.

    `generate(prompt, n) -> list[str]` is the only model dependency, so the same
    harness scores a local Unsloth model, a vLLM server, or a hosted API without
    changes -- and the tests can drive it with a stub.
    """
    ecfg = cfg.eval
    if problems is None:
        problems = load_problems(ecfg, cfg.data)
    if not problems:
        raise RuntimeError("no eval problems loaded -- check dataset config and val_fraction")

    results: list[ProblemResult] = []
    verdict_counts: Counter[str] = Counter()

    for idx, problem in enumerate(problems, start=1):
        completions = generate(problem.prompt, ecfg.samples_per_problem)
        record = ProblemResult(
            problem_id=problem.problem_id,
            title=problem.title,
            n_tests=len(problem.tests),
        )

        for completion in completions:
            code = extract_code(completion)
            if code is None:
                record.verdicts.append(Verdict.NO_CODE.value)
                record.tests_passed.append(0)
                verdict_counts[Verdict.NO_CODE.value] += 1
                continue
            verdict, passed = judge(
                code,
                problem.tests,
                timeout_seconds=ecfg.timeout_seconds,
                memory_limit_mb=ecfg.memory_limit_mb,
                case_sensitive=ecfg.case_sensitive,
            )
            record.verdicts.append(verdict.value)
            record.tests_passed.append(passed)
            verdict_counts[verdict.value] += 1

        results.append(record)
        mark = "AC" if record.solved else record.verdicts[0][:2].upper()
        _log(f"[{idx}/{len(problems)}] {mark:>2}  {problem.problem_id}  {problem.title[:50]}")

    n = ecfg.samples_per_problem
    metrics = {}
    for k in (1, 5, 10):
        if k <= n:
            scores = [
                pass_at_k(n, sum(v == Verdict.ACCEPTED.value for v in r.verdicts), k)
                for r in results
            ]
            metrics[f"pass@{k}"] = sum(scores) / len(scores)

    summary = {
        "run": cfg.name,
        "model": cfg.model.base_model,
        "n_problems": len(results),
        "samples_per_problem": n,
        "metrics": metrics,
        "verdicts": dict(verdict_counts),
        "problems": [asdict(r) for r in results],
    }

    out_path = Path(ecfg.results_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _log("\n=== results ===")
    for key, value in metrics.items():
        _log(f"{key}: {value:.3f}")
    for verdict, count in verdict_counts.most_common():
        _log(f"  {verdict}: {count}")
    _log(f"wrote {out_path}")

    return summary
