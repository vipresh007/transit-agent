"""Run every offline test. No API key, no network, no quota.

    python tests/run_all.py

Seven suites:
    test_tools     tool logic + SQL against transit.db
    test_agent     loop mechanics with a scripted fake model
    test_grounding whether an answer's specifics trace to its sources
    test_constraints whether an itinerary is actually possible
    test_memory    what persists between sessions, and what must not
    test_crew      decomposition, concurrent subagents, synthesis
    evals          --selftest, i.e. do the eval checkers themselves work

Deliberately excluded: smoke_test.py (hits live APIs) and the eval suite
proper (spends model quota). Run those on purpose, not by habit.
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

SUITES = [
    ("tool logic", [sys.executable, str(HERE / "test_tools.py")], HERE),
    ("agent loop", [sys.executable, str(HERE / "test_agent.py")], HERE),
    ("grounding", [sys.executable, str(HERE / "test_grounding.py")], HERE),
    ("constraints", [sys.executable, str(HERE / "test_constraints.py")], HERE),
    ("memory", [sys.executable, str(HERE / "test_memory.py")], HERE),
    ("crew", [sys.executable, str(HERE / "test_crew.py")], HERE),
    ("eval checkers", [sys.executable, str(ROOT / "evals.py"), "--selftest"], ROOT),
]


def main() -> None:
    failed = []
    for name, cmd, cwd in SUITES:
        print(f"\n{'=' * 62}\n{name}\n{'=' * 62}")
        result = subprocess.run(cmd, cwd=cwd)
        if result.returncode != 0:
            failed.append(name)

    print(f"\n{'=' * 62}")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        sys.exit(1)
    print(f"All {len(SUITES)} suites passed.")


if __name__ == "__main__":
    main()
