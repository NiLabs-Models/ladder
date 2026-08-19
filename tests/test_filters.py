from ladder.config import DataConfig
from ladder.data.filters import FilterStats, apply_filters, is_validation, split_key


def make_example(pid, text="x" * 4000):
    return {"problem_id": pid, "messages": [{"role": "assistant", "content": text}]}


def pair(pid, **row_extra):
    return ({"id": pid, "finish_reason": "stop", **row_extra}, make_example(pid))


def run(pairs, cfg=None, **kw):
    cfg = cfg or DataConfig(min_tokens=1, max_tokens=10_000)
    stats = FilterStats()
    kept = list(apply_filters(pairs, cfg, stats=stats, **kw))
    return kept, stats


def test_keeps_a_clean_row():
    kept, stats = run([pair("a")])
    assert len(kept) == 1
    assert stats.kept == 1 and stats.seen == 1
    assert kept[0]["n_tokens"] > 0


def test_drops_truncated_generations():
    kept, stats = run([pair("a", finish_reason="length")])
    assert kept == []
    assert stats.dropped["finish_reason"] == 1


def test_missing_finish_reason_is_not_a_drop():
    # Not every source dataset carries the column; absence must not empty the corpus.
    kept, _ = run([({"id": "a"}, make_example("a"))])
    assert len(kept) == 1


def test_length_bounds():
    cfg = DataConfig(min_tokens=100, max_tokens=200)
    kept, stats = run([pair("a", ), pair("b")], cfg, count_tokens=lambda t: 50)
    assert kept == [] and stats.dropped["too_short"] == 2

    kept, stats = run([pair("a")], cfg, count_tokens=lambda t: 5000)
    assert kept == [] and stats.dropped["too_long"] == 1


def test_dedup_by_problem_keeps_the_first_trace():
    kept, stats = run([pair("a"), pair("a"), pair("b")])
    assert [e["problem_id"] for e in kept] == ["a", "b"]
    assert stats.dropped["duplicate_problem"] == 1


def test_dedup_can_be_disabled():
    cfg = DataConfig(min_tokens=1, max_tokens=10_000, dedup_by_problem=False)
    kept, _ = run([pair("a"), pair("a")], cfg)
    assert len(kept) == 2


def test_rating_bounds_apply_when_the_column_exists():
    cfg = DataConfig(min_tokens=1, max_tokens=10_000, min_rating=1200, max_rating=1800)
    kept, stats = run([pair("a", rating=800), pair("b", rating=1500), pair("c", rating=2400)], cfg)
    assert [e["problem_id"] for e in kept] == ["b"]
    assert stats.dropped["rating_too_low"] == 1
    assert stats.dropped["rating_too_high"] == 1


def test_rating_bounds_are_a_noop_without_the_column():
    cfg = DataConfig(min_tokens=1, max_tokens=10_000, min_rating=1200)
    kept, _ = run([pair("a"), pair("b")], cfg)
    assert len(kept) == 2


def test_max_samples_stops_early():
    cfg = DataConfig(min_tokens=1, max_tokens=10_000, max_samples=2)
    kept, stats = run([pair(str(i)) for i in range(50)], cfg)
    assert len(kept) == 2
    assert stats.seen == 2  # streamed: it stopped pulling rows, not just dropping them


def test_split_is_deterministic_and_seed_dependent():
    assert split_key("1234A", 17) == split_key("1234A", 17)
    assert split_key("1234A", 17) != split_key("1234A", 18)
    assert 0.0 <= split_key("1234A", 17) < 1.0


def test_split_is_roughly_the_requested_fraction():
    cfg = DataConfig(val_fraction=0.1, seed=17)
    n = 4000
    val = sum(is_validation({"problem_id": f"p{i}"}, cfg) for i in range(n))
    assert 0.08 < val / n < 0.12


def test_no_validation_when_fraction_is_zero():
    cfg = DataConfig(val_fraction=0.0)
    assert not is_validation({"problem_id": "p1"}, cfg)
