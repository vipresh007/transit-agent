"""Grounding checks: do an answer's specifics trace to the retrieved text?

    python tests/test_grounding.py

The motivating case is real. A Kensington Market answer had eleven of twelve
specifics grounded and asserted "street art" from the model's own knowledge of
Toronto. Plausible, probably true, unsupported by anything it was given.
"""

import sys
from pathlib import Path

from _harness import check, section

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import grounding  # noqa: E402

SOURCE = [
    "Kensington Market is, first and foremost, a market: bohemian, "
    "multicultural, centred on Augusta Ave and Baldwin St. Vintage shops "
    "abound and many vendors prefer cash.",
    '[{"route": "510", "depart": "08:03:31", "from": "Spadina Ave at Nassau St"}]',
]


def test_supported_and_invented():
    section("supported vs invented specifics")

    r = grounding.check(
        "Head to Kensington Market around Augusta Ave for vintage shops; "
        "the 510 leaves Spadina Ave at Nassau St at 08:03:31.", SOURCE)
    check("a fully grounded answer scores 100%", r["coverage"], 1.0)
    check("nothing flagged", r["unsupported"], [])

    r = grounding.check(
        "Take the 504 from Dundas Station at 09:15:00 to Yorkville.", SOURCE)
    check("invented specifics are caught", sorted(r["unsupported"]),
          ["09:15:00", "504", "Dundas Station", "Yorkville"])

    # The exact failure this module exists for.
    r = grounding.check(
        "Kensington Market is bohemian and known for its Graffiti Alley "
        "murals.", SOURCE)
    check("an embellishment among true statements is caught",
          "Graffiti Alley" in r["unsupported"])


def test_false_positive_guards():
    section("things that look like claims but aren't")

    # Markdown headers and table furniture. These produced seven false
    # positives on a real answer before being filtered.
    r = grounding.check(
        "**Getting there**\n\n| Departure | Arrival | Mode |\n"
        "| 08:03:31 | 08:27 | Streetcar |", SOURCE)
    check("headers and table cells aren't treated as claims",
          [c for c in r["unsupported"] if not c[0].isdigit()], [])

    # U+202F narrow no-break space: models emit it, and a raw comparison
    # reports a grounded claim as invented. Same bug class as the curly
    # apostrophe that broke the eval checkers.
    r = grounding.check("Board at Spadina Ave at Nassau St.", SOURCE)
    check("typographic whitespace doesn't break matching",
          r["unsupported"], [])

    r = grounding.check("Kensington’s vintage shops are worth a look.", SOURCE)
    check("curly apostrophes don't break matching", r["unsupported"], [])

    # Partial-name matching: the corpus may mention the words separately.
    r = grounding.check("The Baldwin Street shops open late.", SOURCE)
    check("multi-word names match on their parts", r["unsupported"], [])


def test_markdown_is_not_a_claim():
    section("document structure is not a factual claim")

    # A real run flagged ten markdown headings as invented facts. The agent
    # complied by DELETING content — including the walking legs of an
    # itinerary — to satisfy a complaint about its own formatting. A
    # false-positive guard made a correct answer unusable.
    formatted = """## Journey Overview
**Recommended Journey Option**

| Origin Stop | Scheduled Departure | Interchange Walk |
|---|---|---|
| Spadina Ave at Nassau St | 09:01 | 2 min |

Take the **Spadina Streetcar** south, then the King Streetcar east."""
    r = grounding.check(formatted, SOURCE + [
        '{"from":"Spadina Ave at Nassau St","depart":"09:01","route":"510"}'])
    check("headings aren't claims", "Journey Overview" not in r["unsupported"])
    check("table labels aren't claims",
          "Scheduled Departure" not in r["unsupported"])
    check("bold-only lines aren't claims",
          "Recommended Journey Option" not in r["unsupported"])
    # "Spadina Streetcar" is supported by a source naming Spadina; demanding
    # the word "streetcar" too would flag a true statement.
    check("generic words don't need grounding",
          "Spadina Streetcar" not in r["unsupported"])

    # ...but a real invented venue inside formatted text is still caught.
    r = grounding.check(formatted + "\n\nStop at Casa Loma on the way.", SOURCE)
    check("a genuine invention survives the filter",
          "Casa Loma" in r["unsupported"])


def test_derived_values():
    section("derived and rounded numbers")

    # Rounding 129m to 130m, or summing legs into a 32-minute total, produces
    # numbers that appear in no source. Flagging them is correct — they're
    # computed, not retrieved — but it's advisory noise, not an error.
    r = grounding.check("The walk is about 130 m and the trip takes 32 minutes.",
                        ["walk of 129 metres"])
    check("computed numbers are surfaced", sorted(r["unsupported"]),
          ["130", "32"])


def test_edges():
    section("edges")

    r = grounding.check("It's a nice area with a good vibe.", SOURCE)
    check("prose with no specifics is vacuously grounded", r["coverage"], 1.0)
    check("and reports no claims", r["claims"], 0)
    check("summary says so", grounding.summary(r), "no checkable claims")

    r = grounding.check("Kensington Market.", [])
    check("no sources means nothing can be supported", r["coverage"], 0.0)


if __name__ == "__main__":
    for fn in (test_supported_and_invented, test_false_positive_guards,
               test_markdown_is_not_a_claim, test_derived_values, test_edges):
        fn()
    from _harness import PASSED
    print(f"\n{PASSED['n']} checks passed")
