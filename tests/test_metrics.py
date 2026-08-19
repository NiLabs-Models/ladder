import json

import pytest

from ladder.train.metrics import count_trained_tokens, project_runtime, summarize


def write_jsonl(tmp_path, token_counts):
    path = tmp_path / "train.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for n in token_counts:
            fh.write(json.dumps({"problem_id": f"p{n}", "n_tokens": n}) + "\n")
    return path


def test_counts_every_example_when_the_run_is_uncapped(tmp_path):
    path = write_jsonl(tmp_path, [100, 200, 300])
    assert count_trained_tokens(path, max_steps=0, effective_batch=16) == (600, 3)


def test_a_capped_run_only_counts_what_it_consumed(tmp_path):
    # 2 steps x batch 4 = 8 examples, so the last 12 are never seen. Summing the
    # whole file would overstate throughput by the untouched remainder.
    path = write_jsonl(tmp_path, [10] * 20)
    assert count_trained_tokens(path, max_steps=2, effective_batch=4) == (80, 8)


def test_cap_larger_than_the_file_is_not_an_error(tmp_path):
    path = write_jsonl(tmp_path, [10, 10])
    assert count_trained_tokens(path, max_steps=100, effective_batch=16) == (20, 2)


def test_examples_without_a_token_count_are_skipped(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps({"n_tokens": 50}) + "\n" + json.dumps({"problem_id": "x"}) + "\n",
        encoding="utf-8",
    )
    assert count_trained_tokens(path, max_steps=0, effective_batch=1) == (50, 1)


def test_summary_computes_throughput():
    s = summarize(train_runtime_seconds=100.0, total_tokens=50_000, n_examples=20)
    assert s["tokens_per_second"] == 500.0
    assert s["mean_tokens_per_example"] == 2500
    assert s["seconds_per_example"] == 5.0
    assert s["train_runtime_hours"] == pytest.approx(0.028, abs=1e-3)


def test_summary_records_peak_vram_in_gb():
    s = summarize(60.0, 1000, 10, peak_vram_bytes=8 * 1024**3)
    assert s["peak_vram_gb"] == 8.0


def test_summary_omits_vram_when_it_was_not_measured():
    assert "peak_vram_gb" not in summarize(60.0, 1000, 10)


def test_summary_carries_extra_context_through():
    s = summarize(60.0, 1000, 10, gpu="Tesla T4", max_steps=250)
    assert s["gpu"] == "Tesla T4"
    assert s["max_steps"] == 250


def test_summary_survives_a_run_with_no_examples():
    s = summarize(60.0, 0, 0)
    assert s["seconds_per_example"] is None
    assert s["mean_tokens_per_example"] is None


def test_summary_rejects_a_nonsense_runtime():
    with pytest.raises(ValueError):
        summarize(0.0, 1000, 10)


def test_projection_answers_the_sizing_question():
    # The whole point: at a measured rate, how long would the full corpus take?
    assert project_runtime(1000.0, 3_600_000) == pytest.approx(1.0)


def test_projection_rejects_a_zero_rate():
    with pytest.raises(ValueError):
        project_runtime(0.0, 1000)
