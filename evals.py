"""
Stage 4: evaluation.

Up to now we've judged the agent by reading its output and going "seems
right". That doesn't scale and it doesn't catch regressions. Once you start
changing prompts and adding RAG, you need to know whether a change helped.

Design decisions worth copying:

1. GROUND TRUTH IS COMPUTED, NOT HARDCODED.
   The TTC republishes every ~6 weeks and every departure time shifts. A
   suite full of literal '25:23:00' would rot immediately and you'd start
   ignoring failures. Instead each case carries a reference SQL query, run
   against the same database, and the expected answer is derived at eval
   time. Write the reference query by hand, carefully, once.

2. CHECKS ARE ABOUT SUBSTANCE, NOT WORDING.
   We assert the answer contains the right *time*, in any reasonable format,
   not that it phrases things a particular way.

3. FAILURES PRINT THE EXPECTATION.
   A red line that doesn't tell you what it wanted is a line you'll skip.

Usage:
    python evals.py --list           # see cases, costs nothing
    python evals.py --only last_501  # run one case
    python evals.py                  # run all (watch your quota)
"""

import argparse
import re
import sqlite3
import sys
import time

import agent

DB = "transit.db"


def q1(sql: str):
    """Run a reference query and return the first row."""
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        return conn.execute(sql).fetchone()
    finally:
        conn.close()


# Zero-pad before comparing. This is the trap the schema doc originally got
# wrong: 863k rows in this feed are '6:32:37' rather than '06:32:37'.
PAD = "substr('0' || st.departure_time, -8)"


def time_variants(hhmmss: str) -> list[str]:
    """Every reasonable way an answer might render a GTFS time.

    '25:23:00' could legitimately appear as 25:23, 1:23 AM, or 01:23.
    Accepting all of them tests correctness rather than formatting.
    """
    h, m, _ = hhmmss.split(":")
    h_i = int(h)
    civil = h_i % 24
    suffix = "am" if civil < 12 else "pm"
    twelve = civil % 12 or 12
    return [
        f"{h_i}:{m}",
        f"{h_i:02d}:{m}",
        f"{civil}:{m}",
        f"{civil:02d}:{m}",
        f"{twelve}:{m}{suffix}",
        f"{twelve}:{m} {suffix}",
    ]


# Models emit typographic punctuation: can’t, don’t, em-dashes, non-breaking
# spaces. Matching raw ASCII against that produces FALSE NEGATIVES — a passing
# agent reported as broken. That's the worst failure mode a test suite has,
# because it trains you to ignore red.
UNICODE_FIXES = {
    "’": "'", "‘": "'",      # curly single quotes
    "“": '"', "”": '"',      # curly double quotes
    "–": "-", "—": "-",      # en/em dash
    "‑": "-", "−": "-",      # non-breaking hyphen, minus
    " ": " ", " ": " ",      # non-breaking spaces
}


def normalize(text: str) -> str:
    """Flatten formatting so checks test meaning, not typography."""
    for bad, good in UNICODE_FIXES.items():
        text = text.replace(bad, good)
    text = text.lower().replace("*", "").replace("`", "")
    return re.sub(r"\s+", " ", text).strip()


def contains_any(text: str, options: list[str]) -> bool:
    flat = normalize(text)
    return any(normalize(o) in flat for o in options)


# ---------------------------------------------------------------------------
# Cases. Each returns (expectation_description, check_function).
# ---------------------------------------------------------------------------


def case_last_501():
    """Last eastbound 501 trip to leave its first stop on a weekday."""
    row = q1(f"""
        SELECT MAX({PAD})
        FROM trips t
        JOIN stop_times st ON st.trip_id = t.trip_id
        JOIN routes r ON r.route_id = t.route_id
        WHERE r.route_short_name = '501'
          AND t.direction_id = '0'
          AND t.service_id = '1'
          AND st.stop_sequence = '1'
    """)
    expected = row[0]
    variants = time_variants(expected)
    return (
        f"answer mentions {expected} (or {variants[4]})",
        lambda ans: contains_any(ans, variants),
    )


def case_earliest_501():
    """Earliest weekday eastbound 501. Fails if the model ignores padding:
    an unpadded MIN() returns a '10:xx' or similar instead of the true first."""
    row = q1(f"""
        SELECT MIN({PAD})
        FROM trips t
        JOIN stop_times st ON st.trip_id = t.trip_id
        JOIN routes r ON r.route_id = t.route_id
        WHERE r.route_short_name = '501'
          AND t.direction_id = '0'
          AND t.service_id = '1'
          AND st.stop_sequence = '1'
    """)
    expected = row[0]
    return (
        f"answer mentions {expected}",
        lambda ans: contains_any(ans, time_variants(expected)),
    )


def case_subway_lines():
    """Three subway lines exist. Tests basic retrieval, not reasoning."""
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    names = [r[0] for r in conn.execute(
        "SELECT route_long_name FROM routes WHERE route_type = '1'")]
    conn.close()
    keys = [n.split("(")[0].strip().lower() for n in names]
    return (
        f"answer mentions all of: {keys}",
        lambda ans: all(normalize(k) in normalize(ans) for k in keys),
    )


def case_bus_route_count():
    """Counting. Easy to get right, easy to hallucinate."""
    n = q1("SELECT COUNT(*) FROM routes WHERE route_type = '3'")[0]
    return (
        f"answer mentions {n} bus routes",
        lambda ans: str(n) in ans,
    )


def case_added_service():
    """calendar_dates exceptions.

    This feed has 107 exception rows, all exception_type '1' (service ADDED
    on specific dates). An agent that only reads `calendar` will confidently
    say no special service exists. Expected to fail until we fix it —
    a known-failing test is more useful than no test.
    """
    row = q1("""
        SELECT date, COUNT(*) FROM calendar_dates
        WHERE exception_type = '1' GROUP BY date ORDER BY date LIMIT 1
    """)
    date, count = row
    pretty = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    return (
        f"answer acknowledges {count} added services on {pretty}",
        lambda ans: str(count) in ans or "added" in ans.lower()
        or "extra" in ans.lower(),
    )


def case_refuses_unknown():
    """Calibration: the DB has no fare data. The right answer is 'I can't
    tell you that', not an invented number.

    Checked by ABSENCE, not presence. Listing acceptable phrasings failed
    twice — "can't", then "does not include" — because there are unlimited
    ways to say "I don't know" and only one way to hallucinate: state a
    price. Test for the failure mode, not for every spelling of success.
    """
    money = re.compile(r"\$\s?\d|\d+\.\d{2}\s*(?:dollars|cad)?|\d+\s*dollars",
                       re.IGNORECASE)

    def check(ans: str) -> bool:
        text = normalize(ans)
        # Any concrete price is a hallucination — the data cannot support one.
        if money.search(text):
            return False
        # And it should actually address the gap, not just dodge the question.
        return contains_any(ans, [
            "fare", "price", "cost",
        ]) and contains_any(ans, [
            "no ", "not ", "n't", "cannot", "unable", "lack", "absent",
        ])

    return ("answer names no price and states the data lacks fares", check)


CASES = {
    "last_501": ("what's the last eastbound 501 Queen streetcar on a weekday?",
                 case_last_501),
    "earliest_501": ("what's the earliest eastbound 501 Queen streetcar on a weekday?",
                     case_earliest_501),
    "subway_lines": ("what subway lines does the TTC operate?",
                     case_subway_lines),
    "bus_routes": ("how many bus routes does the TTC have?",
                   case_bus_route_count),
    "added_service": ("is there any extra or added TTC service scheduled on "
                      "specific dates beyond the regular weekly schedule? "
                      "If so, give the first such date and how many services.",
                      case_added_service),
    "no_fare_data": ("how much does a single TTC adult fare cost, according "
                     "to the schedule data you have?",
                     case_refuses_unknown),
}


# ---------------------------------------------------------------------------
# Self-test: do the CHECKERS work?
#
# We shipped a checker that rejected a correct answer because the model wrote
# "can’t" with a typographic apostrophe. The agent was right and the suite
# said FAIL. A suite you can't trust is worse than no suite, so the checkers
# get their own fixtures — real model phrasings, no API calls.
# ---------------------------------------------------------------------------

SELFTEST = [
    # (case, answer text, should_pass)
    ("no_fare_data",
     "The GTFS feed does **not** store fare data. Therefore I can’t retrieve "
     "the adult fare amount from this schedule.", True),
    ("no_fare_data",
     "The TTC schedule (GTFS) database contains only route, trip, stop‑time, "
     "stop, and calendar information—there are no tables or columns for "
     "fares. Therefore the schedule data does not include the cost.", True),
    ("no_fare_data",
     "I don't have fare information in this dataset.", True),
    ("no_fare_data",
     "A single TTC adult fare costs $3.35.", False),
    ("no_fare_data",
     "The data doesn't list fares, but a single adult fare is 3.35 dollars.",
     False),  # hedges AND hallucinates — must still fail
    ("last_501", "The last one departs at 25:23:00.", True),
    ("last_501", "It leaves at 1:23 AM — scheduled as 25:23 in GTFS.", True),
    ("last_501", "**1:23 am** from Humber Loop", True),
    ("last_501", "The last streetcar is at 11:58 PM.", False),
    ("subway_lines",
     "The TTC runs Line 1 (Yonge–University), Line 2 (Bloor–Danforth) "
     "and Line 4 (Sheppard).", True),
    ("subway_lines", "The TTC runs Line 1 and Line 2.", False),
]


def run_selftest() -> int:
    print("Checker self-test (no API calls):\n")
    failures = 0
    for case_name, answer, should_pass in SELFTEST:
        _, check = CASES[case_name][1]()
        got = bool(check(answer))
        ok = got == should_pass
        failures += not ok
        mark = "ok  " if ok else "BAD "
        print(f"  [{mark}] {case_name:<14} want={should_pass!s:<5} got={got!s:<5} "
              f"{answer[:52]}...")

    print()
    if failures:
        print(f"{failures} checker(s) broken — fix these before trusting any "
              f"eval result.")
    else:
        print("All checkers behave correctly.")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", action="append", help="run specific case(s)")
    parser.add_argument("--list", action="store_true", help="list cases and exit")
    parser.add_argument("--selftest", action="store_true",
                        help="test the checkers themselves, no API calls")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(1 if run_selftest() else 0)

    if args.list:
        for name, (question, _) in CASES.items():
            print(f"  {name:<16} {question}")
        print(f"\n{len(CASES)} cases, roughly 5 requests each "
              f"(~{len(CASES) * 5} total).")
        return

    selected = args.only or list(CASES)
    print(f"Running {len(selected)} case(s). Estimated ~{len(selected) * 5} requests.\n")

    results = []
    for name in selected:
        if name not in CASES:
            print(f"Unknown case: {name}")
            continue

        question, build_check = CASES[name]
        expectation, check = build_check()

        before = agent.REQUEST_COUNT["n"]
        start = time.time()
        try:
            answer = agent.run(question, verbose=False)
            error = None
        except Exception as exc:
            answer, error = "", f"{type(exc).__name__}: {exc}"

        elapsed = time.time() - start
        requests = agent.REQUEST_COUNT["n"] - before
        passed = bool(answer) and check(answer) and error is None

        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}  ({requests} req, {elapsed:.0f}s)")
        if not passed:
            # Show enough of the answer to diagnose. 220 chars kept cutting off
            # before the actual number, which is the only part that matters.
            body = error or normalize(answer)
            print(f"         expected: {expectation}")
            print(f"         got:      {body[:600]}")
            if not error and len(body) > 600:
                print(f"                   ...({len(body) - 600} more chars)")
        print()

        results.append((name, passed))

        # A config error fails identically for every case. Stop rather than
        # burning a request per case to learn the same thing six times.
        if error and any(k in error for k in ("Quota", "NotFound", "Authentication")):
            print(f"Stopping: environment problem, not an agent failure.\n"
                  f"  {error.splitlines()[0]}\n"
                  f"  Try: python list_models.py")
            break

    passed = sum(p for _, p in results)
    print(f"{passed}/{len(results)} passed  "
          f"({agent.REQUEST_COUNT['n']} requests used)")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
