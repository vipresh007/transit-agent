"""Every file this project reads or writes, in one place.

Paths used to be bare strings scattered across nine modules — `"transit.db"`
in tools/transit.py, again in constraints.py, again in evals.py and
optimize_db.py. All relative, so they resolved against the CURRENT WORKING
DIRECTORY, which meant the tests had to `os.chdir(ROOT)` before importing
anything or half the suite would report "transit.db not found".

Two fixes here, and the second is the one that matters:

  1. One definition per file, so moving a database is a one-line change.
  2. Paths are ABSOLUTE, anchored to the repo root rather than to wherever
     you happened to run python from. `python plan.py` and
     `python tests/run_all.py` and an editor's test runner now all find the
     same database.

Every path takes an environment override, which is what makes the test suite
able to point memory.db at a temp file instead of clobbering real preferences.
"""

from __future__ import annotations

import os
from pathlib import Path

# transit/paths.py -> transit/ -> repo root
ROOT = Path(__file__).resolve().parent.parent

DATA = Path(os.getenv("DATA_DIR", ROOT / "data"))
TRACES = Path(os.getenv("TRACE_DIR", ROOT / "traces"))
CACHE = Path(os.getenv("CACHE_DIR", ROOT / ".cache"))


def _db(env: str, name: str) -> Path:
    return Path(os.getenv(env) or DATA / name)


# Built from source data. Delete and rebuild freely.
TRANSIT_DB = _db("TRANSIT_DB", "transit.db")      # scripts/load_gtfs.py
GUIDES_DB = _db("GUIDES_DB", "guides.db")         # scripts/load_guides.py

# NOT rebuildable — memory.db holds the traveller's standing preferences and
# graph.db holds resumable runs. Both are gitignored, so a fresh clone starts
# empty; that's expected, but deleting them locally does lose something.
MEMORY_DB = _db("MEMORY_DB", "memory.db")
GRAPH_DB = _db("GRAPH_DB", "graph.db")

# Source data and download caches.
GTFS_ZIP = Path(os.getenv("GTFS_ZIP") or DATA / "ttc_gtfs.zip")
GUIDES_RAW = Path(os.getenv("GUIDES_RAW") or DATA / "guides_raw.json")


def ensure_dirs() -> None:
    """Create the directories writers need. Readers must not call this —
    a missing database should report itself, not be masked by a new folder."""
    for d in (DATA, TRACES, CACHE):
        d.mkdir(parents=True, exist_ok=True)


def readonly_uri(path: Path) -> str:
    """SQLite read-only URI. Used everywhere a query must not mutate.

    `as_posix()` is not cosmetic. These paths became absolute when they moved
    here, and on Windows that means `C:\\Users\\...\\transit.db`. Interpolated
    straight into a `file:` URI the backslashes are escape characters, so the
    connection either fails or — worse — silently resolves somewhere else.
    Forward slashes are valid on Windows and unambiguous in a URI.
    """
    return f"file:{Path(path).as_posix()}?mode=ro"
