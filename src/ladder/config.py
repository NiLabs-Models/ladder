"""Typed run configuration, loaded from the YAML files in configs/.

Every knob that changes what a run produces lives here, so a run is reproducible
from `configs/<name>.yaml` + a git SHA. Nothing reads os.environ except for
secrets (HF token), which never belong in a config file.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    """Which slice of Codeforces data to train on, and how to filter it."""

    dataset: str = "open-r1/codeforces-cots"
    config_name: str = "solutions_py_decontaminated"
    split: str = "train"

    # Reasoning traces are long. Anything past max_tokens gets dropped rather
    # than truncated -- a truncated trace teaches the model to stop mid-thought.
    max_tokens: int = 8192
    min_tokens: int = 128

    # Contest difficulty band (Codeforces rating). None disables the bound.
    min_rating: int | None = None
    max_rating: int | None = None

    # Drop samples whose generation hit the provider's token ceiling.
    require_finish_reason: str | None = "stop"

    # Deduplicate by problem id, keeping the first trace seen per problem.
    dedup_by_problem: bool = True

    # Run each trace's solution against the problem's own tests and drop the
    # ones that fail. The traces are model-generated and are wrong maybe 30% of
    # the time on hard problems; without this, SFT teaches confident wrong
    # answers. Costs CPU during data prep, never GPU. See data/verify.py.
    verify_solutions: bool = False
    verify_max_tests: int = 5
    verify_timeout: float = 6.0
    verify_memory_limit_mb: int = 1024
    verify_workers: int = 0  # 0 => a sensible default
    # Keep traces that cannot be verified (checker problems, problems shipping
    # no tests) rather than throwing away a third of the corpus for a reason
    # that says nothing about whether the solution is correct.
    verify_keep_unverifiable: bool = True

    # Hold out this fraction for validation loss. Split is by problem id, not by
    # row, so two traces for the same problem cannot straddle the split.
    val_fraction: float = 0.02
    seed: int = 17

    # 0 = use everything. Set small for smoke tests.
    max_samples: int = 0


@dataclass
class ModelConfig:
    """Base checkpoint and LoRA shape."""

    base_model: str = "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit"
    max_seq_length: int = 8192
    load_in_4bit: bool = True

    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    use_gradient_checkpointing: str = "unsloth"


@dataclass
class TrainConfig:
    """SFT hyperparameters, sized for a single 16GB T4."""

    output_dir: str = "outputs/ladder-3b"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    num_train_epochs: float = 1.0
    max_steps: int = 0  # non-zero overrides num_train_epochs

    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    optim: str = "adamw_8bit"
    max_grad_norm: float = 0.5

    logging_steps: int = 5
    save_steps: int = 100
    save_total_limit: int = 2
    eval_steps: int = 100

    seed: int = 17

    # Train on the assistant turn only. Prompt tokens are masked out, so the
    # model is never scored on reproducing the problem statement.
    train_on_completions_only: bool = True

    # Optional push of the adapter (never the merged base) to the Hub.
    hub_model_id: str | None = None
    push_to_hub: bool = False


@dataclass
class EvalConfig:
    """Verifiable eval: generate a program, run it against the problem's tests."""

    dataset: str = "open-r1/codeforces-cots"
    config_name: str = "solutions_py_decontaminated"
    split: str = "train"

    num_problems: int = 100
    samples_per_problem: int = 1  # >1 gives you pass@k

    # Only judge problems whose output is uniquely determined. Codeforces marks
    # the rest `checker` (many valid answers -- "no" vs "NO", any valid
    # arrangement) or `interactive` (the solution talks to a judge process).
    # Exact-match scoring cannot grade either, and including them would report
    # correct solutions as wrong and drag every pass@k number down.
    problem_types: list[str] = field(default_factory=lambda: ["diff"])
    max_new_tokens: int = 4096
    temperature: float = 0.2
    top_p: float = 0.95
    seed: int = 17

    # Per-test-case limits for the generated program.
    timeout_seconds: float = 10.0
    memory_limit_mb: int = 1024
    max_tests_per_problem: int = 20

    # Codeforces accepts "Yes"/"yes"/"YES" interchangeably. Leave this off unless
    # you are judging a problem set where letter case genuinely carries meaning.
    case_sensitive: bool = False

    results_path: str = "outputs/eval/results.json"


@dataclass
class RunConfig:
    name: str = "ladder"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls: type, raw: dict[str, Any] | None, path: str) -> Any:
    """Instantiate a dataclass from a dict, rejecting unknown keys.

    Silently ignoring a typo'd key is how you discover after a four-hour run
    that `learning_rate_` did nothing.
    """
    raw = raw or {}
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"unknown key(s) in {path}: {sorted(unknown)}. valid keys: {sorted(known)}"
        )
    return cls(**raw)


def load_config(path: str | Path) -> RunConfig:
    """Load a run config from YAML, filling defaults for anything omitted."""
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")

    sections = {"data", "model", "train", "eval"}
    unknown = set(raw) - sections - {"name"}
    if unknown:
        raise ValueError(f"unknown top-level section(s) in {path}: {sorted(unknown)}")

    return RunConfig(
        name=raw.get("name", path.stem),
        data=_build(DataConfig, raw.get("data"), f"{path}:data"),
        model=_build(ModelConfig, raw.get("model"), f"{path}:model"),
        train=_build(TrainConfig, raw.get("train"), f"{path}:train"),
        eval=_build(EvalConfig, raw.get("eval"), f"{path}:eval"),
    )
