"""Run a generated program against one test case, under time and memory limits.

SECURITY: this is a *robustness* boundary, not a security boundary. It stops a
model's accidental infinite loop or 20GB allocation from taking down the box; it
does not stop deliberately hostile code, which can still touch the filesystem and
the network. Run evals inside a container or a throwaway notebook VM. Kaggle and
Colab already give you a disposable VM, which is why this is tolerable there.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Verdict(str, Enum):
    ACCEPTED = "accepted"
    WRONG_ANSWER = "wrong_answer"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    NO_CODE = "no_code"


@dataclass
class RunResult:
    verdict: Verdict
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0


def normalize_output(text: str) -> str:
    """Compare the way a competitive-programming judge does.

    Trailing whitespace on a line and a missing final newline are not wrong
    answers, so both sides get stripped per line and re-joined. Anything
    stricter would report a solved problem as failed.
    """
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def outputs_match(expected: str, actual: str, case_sensitive: bool = False) -> bool:
    """Compare a program's output against the expected output, token by token.

    Two tolerances, both matching what a real Codeforces checker does:

    * **Floats** within 1e-6 relative error. An exact string compare would fail a
      correct solution over "0.5" vs "0.500000".
    * **Case**, for purely alphabetic tokens. Codeforces accepts "Yes", "yes" and
      "YES" interchangeably, and observed reference solutions really do print a
      different case than the expected-output file. Restricting this to alphabetic
      tokens keeps it away from anything where case could carry meaning -- a grid
      line like "C.C.C" is not alphabetic, so it is still compared exactly.
    """
    if normalize_output(expected) == normalize_output(actual):
        return True

    exp_toks, act_toks = normalize_output(expected).split(), normalize_output(actual).split()
    if len(exp_toks) != len(act_toks):
        return False
    for e, a in zip(exp_toks, act_toks, strict=True):  # lengths checked above
        if e == a:
            continue
        if not case_sensitive and e.isalpha() and a.isalpha() and e.lower() == a.lower():
            continue
        try:
            ef, af = float(e), float(a)
        except ValueError:
            return False
        if abs(ef - af) > 1e-6 * max(1.0, abs(ef)):
            return False
    return True


# Rlimits are applied by a bootstrap *inside* the child rather than by
# preexec_fn. preexec_fn runs between fork and exec and is documented as unsafe
# in the presence of threads, and verification calls this from a
# ThreadPoolExecutor -- so the combination was removed on principle.
#
# It is NOT established that this caused the 12-hour Kaggle data-prep stall.
# Reintroducing the exact original code on a branch passed Linux CI, so
# preexec_fn plus threads does not stall on its own. See
# kaggle/diagnose_dataprep.py, which measures the candidates on Kaggle itself.
#
# RLIMIT_NPROC is gone for a separate and concrete reason: it is a per-*user*
# limit on Linux, so in a shared container whose user already has more than 64
# processes, setting it to 64 breaks every subsequent fork. Runaway spawning is
# contained instead by start_new_session plus killing the process group on
# timeout.
_BOOTSTRAP = """import resource, runpy, sys
_limit = {mem_bytes}
resource.setrlimit(resource.RLIMIT_AS, (_limit, _limit))
_fsize = {fsize_bytes}
resource.setrlimit(resource.RLIMIT_FSIZE, (_fsize, _fsize))
sys.argv = ["solution.py"]
runpy.run_path("solution.py", run_name="__main__")
"""


def _kill_process_group(proc) -> None:
    """Kill the child and anything it spawned."""
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def run_program(
    code: str,
    stdin: str,
    timeout_seconds: float = 10.0,
    memory_limit_mb: int = 1024,
) -> RunResult:
    """Execute `code` as a Python program with `stdin` piped in."""
    import time

    if not code or not code.strip():
        return RunResult(Verdict.NO_CODE)

    with tempfile.TemporaryDirectory(prefix="ladder-eval-") as tmpdir:
        tmp = Path(tmpdir)
        (tmp / "solution.py").write_text(code, encoding="utf-8")

        # -I (isolated): the program cannot import from the eval harness's own
        # directory or inherit PYTHONPATH from the parent.
        #
        # Deliberately NOT -S. Disabling site.py also removes the `exit`/`quit`
        # builtins, and competitive-programming solutions call exit() to bail
        # out early constantly -- it turned correct programs into runtime errors.
        if os.name == "posix":
            (tmp / "_bootstrap.py").write_text(
                _BOOTSTRAP.format(
                    mem_bytes=memory_limit_mb * 1024 * 1024,
                    fsize_bytes=64 * 1024 * 1024,
                ),
                encoding="utf-8",
            )
            target = "_bootstrap.py"
        else:
            target = "solution.py"

        cmd = [sys.executable, "-I", target]
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"}

        started = time.monotonic()
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=tmpdir,
                env=env,
                # Own session, so a runaway program's children die with it.
                # Thread-safe, unlike the preexec_fn this replaced.
                start_new_session=(os.name == "posix"),
            )
        except OSError as exc:
            return RunResult(Verdict.RUNTIME_ERROR, stderr=str(exc))

        try:
            out, err = proc.communicate(stdin, timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            _kill_process_group(proc)
            try:
                proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            return RunResult(Verdict.TIMEOUT, duration=timeout_seconds)
        duration = time.monotonic() - started

    if proc.returncode != 0:
        return RunResult(
            Verdict.RUNTIME_ERROR,
            stdout=out,
            stderr=err[-4000:],
            duration=duration,
        )
    return RunResult(Verdict.ACCEPTED, stdout=out, stderr=err, duration=duration)



def judge(
    code: str,
    tests: list[tuple[str, str]],
    timeout_seconds: float = 10.0,
    memory_limit_mb: int = 1024,
    case_sensitive: bool = False,
) -> tuple[Verdict, int]:
    """Run every test case; return the first failing verdict and tests passed.

    Stops at the first failure the way a real judge does -- there is no extra
    signal in test 40 once test 3 is wrong, and eval time is the bottleneck.
    """
    passed = 0
    for stdin, expected in tests:
        result = run_program(code, stdin, timeout_seconds, memory_limit_mb)
        if result.verdict is not Verdict.ACCEPTED:
            return result.verdict, passed
        if not outputs_match(expected, result.stdout, case_sensitive):
            return Verdict.WRONG_ANSWER, passed
        passed += 1
    return Verdict.ACCEPTED, passed
