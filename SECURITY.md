# Security

## The sandbox is not a security boundary

`src/ladder/eval/sandbox.py` executes model-generated Python. So does data prep
when `verify_solutions` is on. This is inherent to the project: the entire point
is scoring a program by running it.

What the sandbox actually does:

| Control | Mechanism | Platform |
| --- | --- | --- |
| Wall-clock limit | `subprocess.run(timeout=...)` | all |
| Address space cap | `RLIMIT_AS` | POSIX only |
| Process count cap | `RLIMIT_NPROC` | POSIX only |
| File write cap | `RLIMIT_FSIZE` | POSIX only |
| Own process group | `os.setsid()` | POSIX only |
| No harness imports | `python -I`, temp cwd | all |

What it does **not** do: block network access, block filesystem reads or writes
outside the size cap, block subprocess spawning within the process cap, or
isolate the filesystem namespace. On Windows the rlimits do not apply at all and
only the timeout and interpreter isolation remain.

It stops an accidental infinite loop or a 20GB allocation. **It does not stop
deliberately hostile code.**

## How to run this safely

Run evaluation and verification inside a disposable environment: a container, a
VM, or a hosted notebook. Kaggle and Colab both give you a throwaway VM, which is
the environment this project is designed for and why the shipped configs are
comfortable executing generated code.

Do not run `ladder eval`, `ladder judge`, or `build-data` with
`verify_solutions: true` directly on a machine you care about, and do not run
them against model outputs or datasets from a source you do not trust.

## Reporting a vulnerability

Open an issue at https://github.com/NiLabs-Models/ladder/issues for anything
affecting the harness itself.

Please do **not** file "the sandbox can read files" or "the sandbox can reach the
network" — those are documented above, not vulnerabilities. What *is* worth
reporting: a way for generated code to affect the *judging* result (fake an
`accepted` verdict, escape the timeout, corrupt another problem's run, or
influence the harness process), since that would silently invalidate every number
the project publishes.
