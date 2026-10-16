"""Result-file helpers shared by the experiment CLIs (resume logic)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_results(path: str | Path, resume: bool = True) -> list[dict[str, Any]]:
    """Records of a previous run at ``path`` (``[]`` if absent or ``resume`` is off).

    Resuming skips every run whose identifying keys (task, method, seed, sweep,
    ...) already have a record, *regardless of the budget* (epochs, sample
    counts) it was produced with; a warning says so. With ``resume=False`` the
    file is ignored and overwritten by the new run.
    """
    path = Path(path)
    if not path.exists():
        return []
    if not resume:
        print(f"--no-resume: ignoring and overwriting existing {path}\n")
        return []
    with open(path) as f:
        records = json.load(f)
    print(
        f"Resumed: loaded {len(records)} previous results from {path}\n"
        "WARNING: runs that already have a record are SKIPPED without checking that they "
        "used the same budget (epochs, n-train, ...); pass --no-resume to rerun everything.\n"
    )
    return records


def save_results(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Write ``records`` as indented JSON."""
    with open(path, "w") as f:
        json.dump(records, f, indent=2)
