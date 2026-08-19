import pytest
import yaml

from ladder.config import load_config


def write(tmp_path, data):
    path = tmp_path / "run.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_defaults_fill_in_omitted_sections(tmp_path):
    cfg = load_config(write(tmp_path, {"name": "bare"}))
    assert cfg.name == "bare"
    assert cfg.model.base_model.startswith("unsloth/")
    assert cfg.data.dataset == "open-r1/codeforces-cots"


def test_values_override_defaults(tmp_path):
    cfg = load_config(write(tmp_path, {"train": {"learning_rate": 5e-5, "max_steps": 42}}))
    assert cfg.train.learning_rate == 5e-5
    assert cfg.train.max_steps == 42
    assert cfg.train.optim == "adamw_8bit"  # untouched default


def test_name_falls_back_to_the_filename(tmp_path):
    assert load_config(write(tmp_path, {"train": {}})).name == "run"


def test_typo_in_a_key_is_an_error_not_a_silent_noop(tmp_path):
    # The whole point: a four-hour run must not start with a dead hyperparameter.
    with pytest.raises(ValueError, match="learning_rat"):
        load_config(write(tmp_path, {"train": {"learning_rat": 1e-4}}))


def test_unknown_top_level_section_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="trian"):
        load_config(write(tmp_path, {"trian": {}}))


def test_non_mapping_yaml_is_rejected(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


def test_shipped_configs_are_valid():
    from pathlib import Path

    configs = sorted(Path(__file__).resolve().parents[1].glob("configs/*.yaml"))
    assert configs, "no configs found"
    for path in configs:
        cfg = load_config(path)
        # A LoRA context longer than what data prep keeps would waste memory.
        assert cfg.data.max_tokens <= cfg.model.max_seq_length
