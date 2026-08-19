"""Checking that no evaluated problem was trained on.

The split is structurally sound -- `split_key` hashes the problem id, and eval
selects exactly the ids the build routed to validation -- but "structurally
sound" is an argument, not a check. This turns it into one that runs against the
artifacts a run actually produced.

Two vectors are checked:

**Direct id overlap.** The one the hash split is supposed to prevent. A failure
here means the build and the eval disagreed about the split, usually because
they were run with different seeds or `val_fraction`.

**Alias overlap.** Codeforces cross-posts a problem between div1 and div2 rounds,
so the same problem has two ids (`1149/C` is also `1150/E`). `split_key` hashes
the id, so aliases of one problem hash independently and can land on opposite
sides -- 11 of 225 sampled rows had exactly that property. In
`solutions_py_decontaminated` this is harmless, because the dataset carries one
row per problem and records the other ids as aliases rather than repeating them:
1000 sampled rows had 1000 unique ids and zero alias ids present as their own
row. It is checked anyway, because that is a property of this dataset rather
than of the pipeline, and a different config or source could break it silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContaminationReport:
    n_train: int = 0
    n_eval: int = 0
    direct_overlap: list[str] = field(default_factory=list)
    alias_overlap: list[tuple[str, str]] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.direct_overlap and not self.alias_overlap

    def summary(self) -> str:
        lines = [f"train problems: {self.n_train}", f"eval problems:  {self.n_eval}"]
        if self.direct_overlap:
            lines.append(f"DIRECT OVERLAP ({len(self.direct_overlap)}):")
            lines += [f"  {pid}" for pid in self.direct_overlap[:20]]
        if self.alias_overlap:
            lines.append(f"ALIAS OVERLAP ({len(self.alias_overlap)}):")
            lines += [f"  train {a} is an alias of eval {b}" for a, b in self.alias_overlap[:20]]
        if self.clean:
            lines.append("clean: no evaluated problem appears in training")
        return "\n".join(lines)


def _ids_and_aliases(records: list[dict[str, Any]]) -> tuple[set[str], dict[str, set[str]]]:
    """Map each record to its primary id and the full set of ids naming it."""
    ids: set[str] = set()
    alias_of: dict[str, set[str]] = {}
    for record in records:
        pid = record.get("problem_id") or record.get("id")
        if pid is None:
            continue
        ids.add(pid)
        names = {pid} | {a for a in (record.get("aliases") or []) if a}
        alias_of[pid] = names
    return ids, alias_of


def check(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
) -> ContaminationReport:
    """Report any problem that appears on both sides of the split."""
    train_ids, train_alias = _ids_and_aliases(train_records)
    eval_ids, eval_alias = _ids_and_aliases(eval_records)

    report = ContaminationReport(n_train=len(train_ids), n_eval=len(eval_ids))
    report.direct_overlap = sorted(train_ids & eval_ids)

    # Any shared name means the same problem, even when the primary ids differ.
    direct = set(report.direct_overlap)
    seen: set[tuple[str, str]] = set()
    for train_pid, train_names in train_alias.items():
        if train_pid in direct:
            continue
        for eval_pid, eval_names in eval_alias.items():
            if eval_pid in direct:
                continue
            if train_names & eval_names and (train_pid, eval_pid) not in seen:
                seen.add((train_pid, eval_pid))
                report.alias_overlap.append((train_pid, eval_pid))

    report.alias_overlap.sort()
    return report
