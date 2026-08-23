"""Where did the run's time go?

    python scripts/timing.py                    # the last run
    python scripts/timing.py traces/2026....json
    python scripts/timing.py --compare a.json b.json

Reads what trace.py already records. The point is to make an optimisation
falsifiable: change one thing, re-run, compare. Without a before-and-after you
are guessing, and every "that felt faster" is unfalsifiable.
"""

# Run either way: `python scripts/timing.py` or `python -m scripts.timing`.
# The first puts scripts/ on sys.path rather than the repo root, so `transit`
# would not be importable without this.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import json
import sys
from pathlib import Path

from transit import paths


def load(arg: str | None):
    path = Path(arg) if arg else paths.TRACES / "latest.json"
    if not path.exists():
        sys.exit(f"No trace at {path}. Run something first.")
    return path, json.loads(path.read_text(encoding="utf-8"))


def bar(seconds: float, total: float, width: int = 34) -> str:
    filled = 0 if total <= 0 else round(width * seconds / total)
    return "#" * filled + "." * (width - filled)


def report(path: Path, trace: dict) -> None:
    t = trace.get("timing") or {}
    if not t:
        sys.exit(f"{path.name} predates timing instrumentation.")

    wall = t.get("wall_seconds", 0.0)
    cache = trace.get("cache") or {}
    hits = cache.get("hits", 0)

    print(f"\n{path.name}   {trace.get('model', '?')}   {wall:.1f}s wall")

    # A cached replay and a fast run look identical in the totals, and this
    # reporter said "tools are 95%, index the database" about a 3.8s run that
    # never called a model at all. Same failure the test runner had counting
    # skips as passes: two very different states rendered the same way.
    replay = t.get("calls", 0) == 0 and hits
    if replay:
        print(f"\n  CACHE REPLAY — {hits} hits, 0 model calls.")
        print("  This is not a speed measurement. Every model response came")
        print("  from .cache/, so the only real work was the tools.")
        print("  For a true reading: set CACHE=0, or change the question.")
    elif hits:
        print(f"  ({hits} of {hits + t.get('calls', 0)} responses served from cache)")
    print()

    rows = [
        ("generating", t.get("model_seconds", 0), "smaller prompt, fewer tools, THINKING_BUDGET=0"),
        ("waiting (rate limits)", t.get("wait_seconds", 0), "pin a provider, or come back later"),
        ("pacing (ours)", t.get("throttle_seconds", 0), "lower min_interval in providers.py"),
        ("rejected requests", t.get("failed_seconds", 0),
         "these are 429s bouncing — pin a provider or wait for quota"),
        ("tools", t.get("tool_seconds", 0), "index the database"),
        ("our own python", t.get("unaccounted_seconds", 0), "should be ~0"),
    ]

    # Subagents run concurrently, so the buckets legitimately add up to more
    # than the wall clock. Scaling bars against `wall` would overflow them and
    # print shares like 214%. Scale against the larger of the two, and say why.
    spent = sum(max(0, s) for _, s, _ in rows)
    concurrent = t.get("concurrent") or spent > wall * 1.2
    scale = max(wall, spent) if concurrent else wall

    for label, seconds, fix in rows:
        if seconds < 0.05:
            continue
        share = seconds / scale if scale else 0
        print(f"  {label:<22} {seconds:6.1f}s {share:5.0%}  {bar(seconds, scale)}")
        # A share alone is not a reason to act. 95% of 3.8 seconds is nothing
        # worth optimising, and advising it teaches you to distrust the tool.
        if share > 0.25 and seconds > 3 and not replay:
            print(f"  {'':<22} {'':>6}        -> {fix}")

    if concurrent and wall:
        print(f"\n  {spent:.0f}s of work in {wall:.0f}s of wall clock "
              f"— {spent / wall:.1f}x parallelism.")
        print("  Shares are of total work, not of elapsed time. Subagents ran")
        print("  at the same time, so the buckets sum past the clock.")

    print(f"\n  {t.get('calls', 0)} model calls   "
          f"median {t.get('median_call', 0):.1f}s   "
          f"slowest {t.get('slowest_call', 0):.1f}s")

    gaps = t.get("step_gaps") or []
    if gaps:
        print("\n  per-step latency (does it grow?)")
        widest = max((g["gap_seconds"] for g in gaps), default=1) or 1
        for g in gaps:
            print(f"    step {str(g['step']):<3} {g['gap_seconds']:6.1f}s  "
                  f"{bar(g['gap_seconds'], widest, 24)}  {g['before']}")
        # The first event has nothing before it, so its gap is 0 — comparing
        # against that would report growth on every run ever.
        real = [g["gap_seconds"] for g in gaps if g["gap_seconds"] > 0.5]
        first, last = (real[0], real[-1]) if real else (0, 0)
        if last > first * 2 and last > 5:
            print(f"\n  Latency grew {first:.0f}s -> {last:.0f}s across the run. "
                  f"That's the\n  conversation getting longer, not the provider "
                  f"getting slower —\n  every turn resends everything before it.")

    usage = trace.get("usage") or {}
    if usage.get("n"):
        print(f"\n  {usage['n']} requests, {usage.get('prompt_tokens', 0):,} prompt tokens "
              f"({usage.get('prompt_tokens', 0) // usage['n']:,} per call average)")


def compare(a: Path, ta: dict, b: Path, tb: dict) -> None:
    print(f"\n{'':<24}{a.name[:18]:>18}{b.name[:18]:>19}   change")
    for key, label in (("wall_seconds", "wall clock"),
                       ("model_seconds", "generating"),
                       ("wait_seconds", "waiting"),
                       ("median_call", "median call"),
                       ("slowest_call", "slowest call")):
        x = (ta.get("timing") or {}).get(key, 0)
        y = (tb.get("timing") or {}).get(key, 0)
        delta = f"{(y - x) / x:+.0%}" if x else "—"
        print(f"  {label:<22}{x:>16.1f}s{y:>17.1f}s   {delta:>7}")
    for key, label in (("n", "requests"), ("prompt_tokens", "prompt tokens")):
        x = (ta.get("usage") or {}).get(key, 0)
        y = (tb.get("usage") or {}).get(key, 0)
        delta = f"{(y - x) / x:+.0%}" if x else "—"
        print(f"  {label:<22}{x:>17,}{y:>18,}   {delta:>7}")
    print()


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--compare":
        if len(args) != 3:
            sys.exit("usage: --compare BEFORE.json AFTER.json")
        pa, ta = load(args[1])
        pb, tb = load(args[2])
        compare(pa, ta, pb, tb)
        return
    path, trace = load(args[0] if args else None)
    report(path, trace)
    print()


if __name__ == "__main__":
    main()
