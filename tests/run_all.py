"""Run every offline test. No API key, no network, no quota.

    python tests/run_all.py

Eleven suites:
    test_imports   static: do all imports resolve, no dangling references
    test_tools     tool logic + SQL against transit.db
    test_agent     loop mechanics with a scripted fake model
    test_grounding whether an answer's specifics trace to its sources
    test_constraints whether an itinerary is actually possible
    test_memory    what persists between sessions, and what must not
    test_crew      decomposition, concurrent subagents, synthesis
    test_streaming live observers + the plan()/UI split
    test_realtime  GTFS-RT decoding against saved bytes, and the predictions
                   we deliberately refuse to make
    test_graph     the LangGraph port — skips if langgraph isn't installed
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
    # Static first: it costs 0.2s and catches the refactor breakage that
    # every behavioural suite below would sail past.
    ("imports", [sys.executable, str(HERE / "test_imports.py")], HERE),
    ("tool logic", [sys.executable, str(HERE / "test_tools.py")], HERE),
    ("agent loop", [sys.executable, str(HERE / "test_agent.py")], HERE),
    ("grounding", [sys.executable, str(HERE / "test_grounding.py")], HERE),
    ("constraints", [sys.executable, str(HERE / "test_constraints.py")], HERE),
    ("memory", [sys.executable, str(HERE / "test_memory.py")], HERE),
    ("crew", [sys.executable, str(HERE / "test_crew.py")], HERE),
    ("streaming/UI", [sys.executable, str(HERE / "test_streaming.py")], HERE),
    ("realtime", [sys.executable, str(HERE / "test_realtime.py")], HERE),
    ("langgraph port", [sys.executable, str(HERE / "test_graph.py")], HERE),
    ("eval checkers", [sys.executable, "-m", "transit.pipeline.evals", "--selftest"], ROOT),
]


# A suite exits 3 to say "I did not run" — a missing optional dependency, not
# a pass. Reporting a skip as success is how a broken import in test_graph.py
# survived: the runner printed the same thing either way.
SKIPPED = 3


def main() -> None:
    failed, skipped = [], []
    for name, cmd, cwd in SUITES:
        print(f"\n{'=' * 62}\n{name}\n{'=' * 62}")
        code = subprocess.run(cmd, cwd=cwd).returncode
        if code == SKIPPED:
            skipped.append(name)
        elif code != 0:
            failed.append(name)

    print(f"\n{'=' * 62}")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        sys.exit(1)
    ran = len(SUITES) - len(skipped)
    line = f"All {ran} suites passed."
    if skipped:
        line += f"  SKIPPED (did not run): {', '.join(skipped)}"
    print(line)


if __name__ == "__main__":
    main()
