"""Tests for mount discovery.

The kernel once hardcoded /kaggle/input/<slug>/data/sft, guessed wrong, and lost
a run to it. These cover the same class of mistake for the eval stage, which
mounts an adapter and has not been exercised at full scale yet.
"""

from ladder.kaggle_paths import describe_mounts, find_adapter, find_prepared_data


def test_finds_data_at_any_depth(tmp_path):
    nested = tmp_path / "ladder-build-data" / "data" / "sft"
    nested.mkdir(parents=True)
    (nested / "train.jsonl").write_text("{}\n", encoding="utf-8")
    assert find_prepared_data(str(tmp_path)) == str(nested)


def test_finds_data_at_the_root(tmp_path):
    (tmp_path / "train.jsonl").write_text("{}\n", encoding="utf-8")
    assert find_prepared_data(str(tmp_path)) == str(tmp_path)


def test_no_data_returns_none(tmp_path):
    assert find_prepared_data(str(tmp_path)) is None


def test_finds_adapter(tmp_path):
    d = tmp_path / "ladder-train" / "outputs" / "ladder-3b-kaggle"
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    (d / "adapter_model.safetensors").write_bytes(b"\0")
    assert find_adapter(str(tmp_path)) == str(d)


def test_adapter_search_ignores_checkpoint_dirs(tmp_path):
    """A checkpoint is a mid-training snapshot; scoring it reports the wrong model."""
    root = tmp_path / "outputs" / "run"
    ckpt = root / "checkpoint-100"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")
    root.mkdir(exist_ok=True)
    (root / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert find_adapter(str(tmp_path)) == str(root)


def test_checkpoint_used_only_when_it_is_all_there_is(tmp_path):
    ckpt = tmp_path / "outputs" / "run" / "checkpoint-100"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert find_adapter(str(tmp_path)) == str(ckpt)


def test_no_adapter_returns_none(tmp_path):
    assert find_adapter(str(tmp_path)) is None


def test_describe_mounts_lists_what_is_there(tmp_path):
    d = tmp_path / "some-kernel"
    d.mkdir()
    (d / "train.jsonl").write_text("{}", encoding="utf-8")
    out = describe_mounts(str(tmp_path))
    assert "train.jsonl" in out


def test_describe_mounts_on_a_missing_root():
    assert "does not exist" in describe_mounts("/definitely/not/here")


def test_describe_mounts_on_an_empty_root(tmp_path):
    out = describe_mounts(str(tmp_path))
    assert "nothing mounted" in out or str(tmp_path) in out
