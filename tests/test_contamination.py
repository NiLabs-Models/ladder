import pytest

from ladder.config import DataConfig
from ladder.data.contamination import check
from ladder.data.filters import is_validation


def rec(pid, aliases=None):
    return {"problem_id": pid, "aliases": aliases or []}


def test_disjoint_sets_are_clean():
    report = check([rec("1/A"), rec("2/B")], [rec("3/C")])
    assert report.clean
    assert report.n_train == 2 and report.n_eval == 1
    assert "clean" in report.summary()


def test_direct_overlap_is_caught():
    report = check([rec("1/A"), rec("2/B")], [rec("2/B")])
    assert not report.clean
    assert report.direct_overlap == ["2/B"]
    assert "DIRECT OVERLAP" in report.summary()


def test_alias_overlap_is_caught():
    # 1149/C and 1150/E are the same problem cross-posted between divisions.
    # Ids differ, so a direct comparison misses it entirely.
    report = check([rec("1149/C", ["1150/E"])], [rec("1150/E")])
    assert not report.clean
    assert report.direct_overlap == []
    assert report.alias_overlap == [("1149/C", "1150/E")]
    assert "ALIAS OVERLAP" in report.summary()


def test_alias_overlap_detected_from_either_side():
    report = check([rec("1150/E")], [rec("1149/C", ["1150/E"])])
    assert report.alias_overlap == [("1150/E", "1149/C")]


def test_shared_alias_between_two_different_primaries():
    report = check([rec("100/A", ["999/Z"])], [rec("200/B", ["999/Z"])])
    assert report.alias_overlap == [("100/A", "200/B")]


def test_a_direct_overlap_is_not_double_reported_as_an_alias_overlap():
    report = check([rec("1/A", ["9/Z"])], [rec("1/A", ["9/Z"])])
    assert report.direct_overlap == ["1/A"]
    assert report.alias_overlap == []


def test_unrelated_aliases_do_not_trigger():
    report = check([rec("1/A", ["8/X"])], [rec("2/B", ["9/Y"])])
    assert report.clean


def test_records_without_an_id_are_ignored():
    report = check([rec("1/A"), {"messages": []}], [rec("2/B")])
    assert report.clean and report.n_train == 1


def test_empty_inputs():
    report = check([], [])
    assert report.clean and report.n_train == 0


@pytest.mark.parametrize("val_fraction", [0.02, 0.05, 0.2])
def test_the_hash_split_itself_never_puts_a_problem_on_both_sides(val_fraction):
    """The property the whole scheme rests on, asserted rather than argued."""
    cfg = DataConfig(val_fraction=val_fraction, seed=17)
    ids = [f"{n}/{c}" for n in range(400) for c in "ABC"]
    train = [i for i in ids if not is_validation({"problem_id": i}, cfg)]
    held = [i for i in ids if is_validation({"problem_id": i}, cfg)]

    assert train and held, "split degenerate at this fraction"
    assert check([rec(i) for i in train], [rec(i) for i in held]).clean


def test_a_seed_mismatch_between_build_and_eval_is_caught():
    """The realistic failure: two stages configured with different seeds.

    Structurally the split is sound; this is what actually goes wrong.
    """
    ids = [f"{n}/{c}" for n in range(400) for c in "ABC"]
    build_cfg = DataConfig(val_fraction=0.1, seed=17)
    eval_cfg = DataConfig(val_fraction=0.1, seed=99)

    train = [i for i in ids if not is_validation({"problem_id": i}, build_cfg)]
    held = [i for i in ids if is_validation({"problem_id": i}, eval_cfg)]

    report = check([rec(i) for i in train], [rec(i) for i in held])
    assert not report.clean, "a seed mismatch must not look clean"
