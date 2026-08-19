"""QLoRA supervised fine-tuning on top of Unsloth.

Sized for one 16GB T4 -- the GPU you actually get for free on Kaggle and Colab.
Everything heavy is imported inside the functions so that `import ladder` stays
usable on a CPU-only box for data prep, eval scoring, and tests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ladder.config import RunConfig
from ladder.train.metrics import count_trained_tokens, summarize


def load_model(cfg: RunConfig):
    """Load the 4-bit base and attach LoRA adapters."""
    from unsloth import FastLanguageModel

    mcfg = cfg.model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=mcfg.base_model,
        max_seq_length=mcfg.max_seq_length,
        load_in_4bit=mcfg.load_in_4bit,
        dtype=None,  # None => fp16 on T4/P100, bf16 on Ampere and newer.
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=mcfg.lora_r,
        lora_alpha=mcfg.lora_alpha,
        lora_dropout=mcfg.lora_dropout,
        target_modules=mcfg.target_modules,
        bias="none",
        use_gradient_checkpointing=mcfg.use_gradient_checkpointing,
        random_state=cfg.train.seed,
    )
    return model, tokenizer


def _render(tokenizer, examples):
    """Apply the base model's chat template to a batch of message lists."""
    return {
        "text": [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in examples["messages"]
        ]
    }


def train(cfg: RunConfig, data_dir: str | Path) -> str:
    """Run SFT and save the adapter. Returns the output directory."""
    import torch
    from datasets import load_dataset
    from transformers import TrainingArguments

    data_dir = Path(data_dir)
    train_file = data_dir / "train.jsonl"
    val_file = data_dir / "val.jsonl"
    if not train_file.exists():
        raise FileNotFoundError(f"{train_file} not found -- run `ladder build-data` first")

    model, tokenizer = load_model(cfg)

    files = {"train": str(train_file)}
    if val_file.exists() and val_file.stat().st_size > 0:
        files["validation"] = str(val_file)
    raw = load_dataset("json", data_files=files)
    dataset = raw.map(
        lambda ex: _render(tokenizer, ex),
        batched=True,
        remove_columns=raw["train"].column_names,
    )

    tcfg = cfg.train
    args = TrainingArguments(
        output_dir=tcfg.output_dir,
        per_device_train_batch_size=tcfg.per_device_train_batch_size,
        gradient_accumulation_steps=tcfg.gradient_accumulation_steps,
        num_train_epochs=tcfg.num_train_epochs,
        max_steps=tcfg.max_steps or -1,
        learning_rate=tcfg.learning_rate,
        warmup_ratio=tcfg.warmup_ratio,
        lr_scheduler_type=tcfg.lr_scheduler_type,
        weight_decay=tcfg.weight_decay,
        optim=tcfg.optim,
        max_grad_norm=tcfg.max_grad_norm,
        logging_steps=tcfg.logging_steps,
        save_steps=tcfg.save_steps,
        save_total_limit=tcfg.save_total_limit,
        seed=tcfg.seed,
        # T4 and P100 have no bf16 units; picking the wrong one here is an
        # instant crash on a free-tier GPU.
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
        save_safetensors=True,
    )

    # trl's API moved: SFTConfig now carries what used to be TrainingArguments
    # plus the dataset/length settings, SFTTrainer takes `processing_class`
    # rather than `tokenizer`, and DataCollatorForCompletionOnlyLM is gone.
    # Built by capability detection rather than a version pin -- pinning a
    # version chosen without checking it against the target environment is what
    # broke an earlier run outright.
    import inspect

    from trl import SFTConfig, SFTTrainer

    sft_fields = set(inspect.signature(SFTConfig.__init__).parameters)
    length_kwargs = {}
    if "max_length" in sft_fields:
        length_kwargs["max_length"] = cfg.model.max_seq_length
    elif "max_seq_length" in sft_fields:
        length_kwargs["max_seq_length"] = cfg.model.max_seq_length
    if "dataset_text_field" in sft_fields:
        length_kwargs["dataset_text_field"] = "text"
    if "packing" in sft_fields:
        # Packing would blend two problems into one window.
        length_kwargs["packing"] = False

    sft_args = SFTConfig(**{**vars(args), **length_kwargs}) if sft_fields else args

    trainer_fields = set(inspect.signature(SFTTrainer.__init__).parameters)
    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset["train"],
        "args": sft_args,
    }
    trainer_kwargs["processing_class" if "processing_class" in trainer_fields
                   else "tokenizer"] = tokenizer
    if "validation" in dataset:
        trainer_kwargs["eval_dataset"] = dataset["validation"]

    trainer = SFTTrainer(**trainer_kwargs)

    if tcfg.train_on_completions_only:
        # Mask the prompt so loss is computed only over the model's own turn.
        # Unsloth's helper replaces trl's removed collator. If neither exists,
        # say so loudly -- training on the full sequence silently produces a
        # worse model with nothing in the logs to point at.
        try:
            from unsloth.chat_templates import train_on_responses_only

            trainer = train_on_responses_only(
                trainer,
                instruction_part="<|im_start|>user\n",
                response_part="<|im_start|>assistant\n",
            )
            print("loss masked to the assistant turn", file=sys.stderr)
        except ImportError as exc:
            raise RuntimeError(
                "train_on_completions_only is set but no masking helper is "
                f"available ({exc}). Training would silently compute loss over "
                "the problem statement too. Install a current unsloth, or set "
                "train_on_completions_only: false to accept full-sequence loss."
            ) from exc

    stats = trainer.train()

    out_dir = Path(tcfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    (out_dir / "run_config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")

    # Throughput, so the next run can be sized by arithmetic instead of a guess.
    runtime = stats.metrics.get("train_runtime")
    if runtime is not None:
        effective_batch = tcfg.per_device_train_batch_size * tcfg.gradient_accumulation_steps
        total_tokens, n_examples = count_trained_tokens(
            train_file, tcfg.max_steps, effective_batch
        )
        extra = {
            "max_seq_length": cfg.model.max_seq_length,
            "effective_batch": effective_batch,
            "max_steps": tcfg.max_steps,
        }
        peak_vram_bytes = None
        if torch.cuda.is_available():
            peak_vram_bytes = torch.cuda.max_memory_allocated()
            extra["gpu"] = torch.cuda.get_device_name(0)

        throughput = summarize(
            runtime, total_tokens, n_examples, peak_vram_bytes=peak_vram_bytes, **extra
        )
        (out_dir / "throughput.json").write_text(
            json.dumps(throughput, indent=2), encoding="utf-8"
        )
        print(
            f"throughput: {throughput['tokens_per_second']} tok/s"
            f" | peak vram {throughput.get('peak_vram_gb')} GB"
            f" | {throughput['train_runtime_hours']} h",
            file=sys.stderr,
        )

    print(f"saved adapter -> {out_dir}", file=sys.stderr)

    if tcfg.push_to_hub and tcfg.hub_model_id:
        model.push_to_hub(tcfg.hub_model_id)
        tokenizer.push_to_hub(tcfg.hub_model_id)
        print(f"pushed -> https://huggingface.co/{tcfg.hub_model_id}", file=sys.stderr)

    return str(out_dir)
