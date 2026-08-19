"""Loading a trained adapter and generating solutions.

Produces the `generate(prompt, n) -> list[str]` callable that `eval.runner`
expects, so scoring the fine-tune and scoring the untouched base model differ by
one command-line flag.
"""

from __future__ import annotations

from ladder.config import RunConfig
from ladder.data.formatting import SYSTEM_PROMPT


def load_for_inference(cfg: RunConfig, adapter_dir: str | None = None):
    """Load base + optional adapter in inference mode.

    Passing `adapter_dir=None` loads the untouched base model, which is how you
    get the before-number for the README table.
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_dir or cfg.model.base_model,
        max_seq_length=cfg.model.max_seq_length,
        load_in_4bit=cfg.model.load_in_4bit,
        dtype=None,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def make_generator(model, tokenizer, cfg: RunConfig):
    """Build the `generate(prompt, n)` callable used by the eval harness."""
    import torch

    ecfg = cfg.eval

    def generate(prompt: str, n: int = 1) -> list[str]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=ecfg.max_new_tokens,
                # Greedy for n=1: the eval is a capability measurement, and
                # sampling noise would show up as run-to-run drift in pass@1.
                do_sample=n > 1 or ecfg.temperature > 0,
                temperature=ecfg.temperature if ecfg.temperature > 0 else None,
                top_p=ecfg.top_p,
                num_return_sequences=n,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[-1]
        return [tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in outputs]

    return generate
