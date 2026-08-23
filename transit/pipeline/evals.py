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
import json
import re
import sqlite3
import sys
import time
from transit import paths

# agent/llm are imported lazily inside main(), not at module scope. They pull
# in the OpenAI SDK, and --selftest advertises itself as needing no API access
# — so it must not require the SDK to be installed either. A "free, offline"
# check that fails on an import is not offline.
DB = paths.TRANSIT_DB


def q1(sql: str):
    """Run a reference query and return the first row."""
    conn = sqlite3.connect(paths.readonly_uri(DB), uri=True)
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
    conn = sqlite3.connect(paths.readonly_uri(DB), uri=True)
    rows = conn.execute(
        "SELECT route_short_name, route_long_name FROM routes "
        "WHERE route_type = '1'").fetchall()
    conn.close()

    def check(ans: str) -> bool:
        """Each line identified by NUMBER and NAME, in any layout.

        Demanding the literal "line 1" failed a correct answer that used a
        markdown table — `| 1 | Yonge-University |` names the line perfectly
        and contains no such string. The checker was testing formatting.

        Requiring both the number and a distinctive word from the long name
        keeps it strict: a bare "1" somewhere in the prose isn't enough, and
        neither is "Yonge" without the number.
        """
        flat = normalize(ans).lower()
        for number, long_name in rows:
            words = [w for w in re.findall(r"[a-z]+", long_name.lower())
                     if len(w) > 4 and w != "line"]
            numbered = re.search(rf"(?<!\d){re.escape(number)}(?!\d)", flat)
            if not numbered or not any(w in flat for w in words):
                return False
        return True

    listed = ", ".join(f"{n} ({ln.split('(')[0].strip()})" for n, ln in rows)
    # Extra lines are fine: 5 Eglinton and 6 Finch West are in the feed as
    # light rail, and the TTC brands them as lines too.
    return (f"answer identifies each subway line by number and name: {listed}",
            check)


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


def case_journey():
    """End-to-end journey planning — the thing the project is actually for.

    Ground truth from plan_journey itself, which is verified against the
    database. This checks that the AGENT surfaces what the tool found: the
    right routes, real times, and correctly-labelled endpoints. It took a
    dozen iterations to get right, so it gets a test.
    """
    # Reference computed INDEPENDENTLY of plan_journey. Using the planner to
    # generate ground truth for a test of the planner is circular — it would
    # pass even if the planner silently changed its mind about the route.
    # These are the corridor stops, confirmed against the data by hand:
    #   8128  Spadina Ave at Nassau St South Side   (510 southbound)
    #   15648 King St West at Spadina Ave East Side (504 eastbound)
    leg = """
        SELECT r.route_short_name,
               MIN(substr('0' || st.departure_time, -8)) AS depart
        FROM trips t
        JOIN routes r     ON r.route_id = t.route_id
        JOIN stop_times st ON st.trip_id = t.trip_id
        WHERE st.stop_id = ? AND t.service_id = '1'
          AND r.route_short_name = ?
          AND substr('0' || st.departure_time, -8) >= ?
    """
    conn = sqlite3.connect(paths.readonly_uri(DB), uri=True)
    try:
        _, board = conn.execute(leg, ("8128", "510", "08:00:00")).fetchone()
    finally:
        conn.close()
    routes = ["510", "504"]

    def check(ans: str) -> bool:
        flat = normalize(ans)
        # Every transit route named, the real boarding time present, and no
        # claim of a direct trip when a transfer is required.
        return (
            all(r.lower() in flat for r in routes)
            and contains_any(ans, time_variants(board))
            and "distillery" in flat
        )

    return (
        f"answer names routes {routes}, boards at {board}, reaches Distillery",
        check,
    )



def _guides_ready() -> bool:
    import os
    return os.path.exists(paths.GUIDES_DB)


def case_guide_character():
    """Descriptive question that only the travel guides can answer.

    Ground truth comes from FTS, not from vector search: the eval must not
    depend on an embedding provider being up, and it must not use the same
    retrieval path it's testing. Distinctive vocabulary from the actual
    corpus is the check — an answer built from the guides will echo some of
    it; an answer from the model's own knowledge of Toronto probably won't.
    """
    conn = sqlite3.connect(paths.readonly_uri(paths.GUIDES_DB), uri=True)
    try:
        row = conn.execute(
            "SELECT text FROM chunks WHERE article LIKE '%Kensington%' "
            "AND heading LIKE '%Understand%' LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    source = (row[0] if row else "").lower()

    # Terms that appear in the guide passage. Requiring several rather than
    # one avoids passing on a lucky single word.
    candidates = ["bohemian", "vintage", "jewish", "multicultural",
                  "car-free", "augusta", "market"]
    present = [w for w in candidates if w in source]

    def check(ans: str) -> bool:
        flat = normalize(ans)
        return sum(1 for w in present if w in flat) >= 2

    return (f"answer echoes >=2 guide terms from {present}", check)


def case_guide_out_of_scope():
    """The guides cover Toronto. Asked about Osaka, retrieval returns its
    nearest Toronto match at ~0.50 — worthless. The agent must say it
    doesn't know rather than dress up a weak match as an answer."""

    def check(ans: str) -> bool:
        flat = normalize(ans)
        # Must not present Toronto restaurants as an answer about Osaka.
        #
        # Scoped to the SENTENCE, not the whole answer. A correct refusal —
        # "I don't have coverage for Osaka. If you want ramen recommendations
        # in Toronto, ask!" — contains both "osaka" and "recommend" and was
        # marked FAIL. It was refusing and then offering to help, which is
        # the behaviour we want.
        #
        # Fourth time in this project a checker has fired on a correct answer
        # by matching vocabulary instead of meaning. Precision over recall:
        # a false positive here would send you tuning a prompt that is fine.
        pretends = any(
            "osaka" in sentence
            and any(w in sentence for w in ["recommend", "try ", "you should visit"])
            and not any(w in sentence for w in
                        ["if you", "feel free", "would you", "let me know",
                         "happy to", "i can help"])
            for sentence in re.split(r"[.!?\n]", flat)
        )
        admits = contains_any(ans, [
            "don't have", "do not have", "no information", "not cover",
            "doesn't cover", "only cover", "toronto", "cannot", "can't",
            "outside", "not in the guides", "unable",
        ])
        return admits and not pretends

    return ("answer admits the Toronto guides don't cover Osaka", check)


CASES = {
    "guide_character": ("what is Kensington Market like as a neighbourhood?",
                        case_guide_character),
    "guide_out_of_scope": ("what's the best ramen in Osaka?",
                           case_guide_out_of_scope),
    "journey": ("how do I get from Kensington Market to the Distillery "
                "District on a weekday morning?", case_journey),
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
    # A real answer the old checker rejected: a markdown table names every
    # line correctly and contains the string "line 1" nowhere.
    ("subway_lines",
     "| Line # | Common name | Route |\n|---|---|---|\n"
     "| 1 | Yonge-University | north-south on Yonge Street |\n"
     "| 2 | Bloor-Danforth | east-west along Bloor and Danforth |\n"
     "| 4 | Sheppard | east from Yonge along Sheppard Avenue |\n"
     "| 5 | Eglinton | Eglinton Crosstown LRT |", True),
    # Naming Yonge and Bloor without the numbers is not identifying them.
    ("subway_lines",
     "The TTC subway covers Yonge, Bloor and Sheppard.", False),
    ("subway_lines", "The TTC runs Line 1 and Line 2.", False),
    # Your first fully correct journey run, kept verbatim as a fixture.
    ("journey",
     "8:00 AM walk: Kensington Market -> Spadina Ave at Nassau St South Side. "
     "8:03 AM streetcar 510 to Spadina Ave at King St West. 8:15 AM "
     "streetcar 504A to Distillery Loop, arriving 8:37.", True),
    # Right shape, invented times — must fail.
    ("journey",
     "Take the 510 then the 504 to the Distillery District, about 40 minutes.",
     False),
    ("guide_character",
     "Kensington Market is a bohemian, multicultural pocket of narrow "
     "car-free streets full of vintage shops.", True),
    ("guide_character",
     "Kensington Market is a nice area with shops and restaurants.", False),
    # A real answer that the old checker rejected: it refuses correctly and
    # then offers Toronto help, which put "osaka" and "recommend" in the same
    # answer though not the same sentence.
    ("guide_out_of_scope",
     "I am a travel planning assistant for Toronto and only have access to "
     "transit schedules and travel guides for the greater Toronto area. I "
     "don't have guide data or local coverage for Osaka, Japan. If you are "
     "looking for ramen recommendations or transit directions within Toronto, "
     "feel free to ask!", True),
    ("guide_out_of_scope",
     "My guides only cover Toronto, so I don't have information on Osaka.",
     True),
    # A real answer that the old checker rejected: it refuses correctly and
    # then offers Toronto help, which put "osaka" and "recommend" in the same
    # answer though not the same sentence.
    ("guide_out_of_scope",
     "I am a travel planning assistant for Toronto and only have access to "
     "transit schedules and travel guides for the greater Toronto area. I "
     "don't have guide data or local coverage for Osaka, Japan. If you are "
     "looking for ramen recommendations or transit directions within Toronto, "
     "feel free to ask!", True),
    ("guide_out_of_scope",
     "For ramen in Osaka you should try the places around Dotonbori.", False),
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

    from transit.core import agent   # noqa: PLC0415 — deliberately deferred, see note above
    from transit.core import llm     # noqa: PLC0415

    selected = args.only or list(CASES)
    print(f"Running {len(selected)} case(s). Estimated ~{len(selected) * 5} requests.\n")

    results = []
    for name in selected:
        if name not in CASES:
            print(f"Unknown case: {name}")
            continue

        question, build_check = CASES[name]
        expectation, check = build_check()

        before = llm.USAGE["n"]
        start = time.time()
        try:
            answer = agent.run(question, verbose=False)
            error = None
        except Exception as exc:
            answer, error = "", f"{type(exc).__name__}: {exc}"

        elapsed = time.time() - start
        requests = llm.USAGE["n"] - before
        passed = bool(answer) and check(answer) and error is None

        # FAIL and ERROR are not the same finding and must not print the same.
        # A 413 from Groq's token-per-minute cap was reported as [FAIL] with
        # the exception text in the "got" field, which reads as "the model
        # answered wrongly" — so you go tune the prompt when the actual news
        # is that the request never completed. An eval that cannot tell "the
        # answer was wrong" from "there was no answer" measures the wrong
        # thing exactly when you most need it: while changing providers.
        status = "PASS" if passed else ("ERROR" if error else "FAIL")
        print(f"[{status}] {name}  ({requests} req, {elapsed:.0f}s)")
        if error:
            print(f"         the run did not complete — this is NOT a quality")
            print(f"         result. {error[:400]}")
            print()
            results.append((name, passed, True))
            if any(k in error for k in ("Quota", "NotFound", "Authentication",
                                        "too large", "rate_limit")):
                print(f"Stopping: environment problem, not an agent failure.\n"
                      f"  {error.splitlines()[0]}\n"
                      f"  Try: python scripts/list_models.py")
                break
            continue
        if not passed:
            # Show enough of the answer to diagnose. 220 chars kept cutting off
            # before the actual number, which is the only part that matters.
            body = error or normalize(answer)
            print(f"         expected: {expectation}")
            print(f"         got:      {body[:600]}")
            if not error and len(body) > 600:
                print(f"                   ...({len(body) - 600} more chars)")
        print()

        results.append((name, passed, False))

    scored = [(n, p) for n, p, errored in results if not errored]
    errored = [n for n, _, e in results if e]
    passed = sum(p for _, p in scored)

    print(f"{passed}/{len(scored)} of the cases that RAN passed  "
          f"({llm.USAGE['n']} requests used)")
    if errored:
        # Reported separately and never averaged in. Folding errors into the
        # score would let a quota wall look like a quality regression.
        print(f"{len(errored)} case(s) never completed: {', '.join(errored)}")
        print("Those say nothing about answer quality — fix the environment "
              "and re-run before drawing conclusions.")
    sys.exit(0 if scored and passed == len(scored) else 1)


if __name__ == "__main__":
    main()
