"""Does the answer actually follow from what was retrieved?

RAG improves the material a model works from; it does not stop the model
drawing on its own priors. A real run here produced a Kensington Market
description where eleven of twelve specific claims traced back to retrieved
text — and "street art" did not. Plausible, probably true, entirely
unsupported by anything the agent was given.

That's the normal case, and nothing else in this project would flag it.

WHAT THIS CHECKS, AND WHAT IT DOESN'T.
It extracts claim-like tokens — proper nouns, numbers, times, prices — and
asks whether each appears in the retrieved text. That catches invented
specifics, which is the failure mode that matters: a wrong street name or an
invented price is actionable and harmful, whereas an unsupported adjective
("lively") mostly isn't.

It cannot catch a claim that recombines retrieved facts wrongly ("the 510
runs to Kensington" when the corpus says the 506 does). Detecting that needs
an LLM judge, which costs a request per check and brings its own error rate.
This is the cheap 80% — deterministic, free, and honest about its limits.
"""

import re

# Multi-word proper nouns first ("Kensington Market"), then standalone
# capitalised words, numbers, times and prices.
PROPER_NOUN = re.compile(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*)\b")
NUMERIC = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b|\$\s?\d+(?:\.\d{2})?|\b\d{2,}\b")

# Words that get capitalised for reasons other than being a claim.
STOPWORDS = {
    "the", "this", "that", "there", "here", "you", "your", "it", "its",
    "and", "but", "for", "with", "from", "into", "about", "then", "than",
    "what", "when", "where", "which", "while", "who", "why", "how",
    "many", "most", "some", "any", "all", "both", "each", "few", "more",
    "note", "tip", "step", "total", "walk", "take", "get", "see", "eat",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "morning", "afternoon", "evening", "night", "today",
    # Document structure and domain-generic vocabulary. A phrase made up
    # ENTIRELY of these is the model's own formatting, not a claim about the
    # world: "Journey Overview", "Scheduled Departure", "Interchange Walk".
    # Missing these, the checker flagged ten markdown headings as invented
    # facts, and the agent responded by deleting real content to comply — a
    # false-positive guard actively made the answer worse.
    "departure", "arrival", "mode", "route", "leg", "legs", "time", "times",
    "summary", "caveats", "atmosphere", "food", "getting", "total",
    "journey", "overview", "routing", "notes", "recommended", "option",
    "options", "available", "origin", "destination", "scheduled", "estimated",
    "interchange", "transfer", "next", "first", "last", "final", "start",
    "streetcar", "subway", "bus", "train", "station", "stop", "stops",
    "line", "service", "platform", "direction", "duration", "distance",
    "northbound", "southbound", "eastbound", "westbound", "inbound",
    "plan", "steps", "details", "info", "information", "overall",
    "toronto",  # in every chunk; matching it proves nothing
}


# Models emit typographic whitespace and punctuation. "Union Station"
# uses U+202F (narrow no-break space) and will never match "Union Station" in
# the source, so a grounded claim gets reported as invented. This is the same
# bug that made the eval checkers reject a correct answer over a curly
# apostrophe — normalise before comparing, always.
UNICODE_SPACES = dict.fromkeys(
    map(ord, "       "), " "
)
UNICODE_PUNCT = {"’": "'", "‘": "'", "“": '"', "”": '"',
                 "–": "-", "—": "-", "‑": "-", "−": "-"}


# Street types are written both ways constantly — the guides say "Baldwin St",
# an answer says "Baldwin Street", and a literal comparison calls a real place
# invented. Collapse both to the short form before matching. Domain-specific,
# and worth it: nearly every claim in this project is an address.
STREET_TYPES = {
    "street": "st", "avenue": "ave", "road": "rd", "boulevard": "blvd",
    "drive": "dr", "crescent": "cres", "court": "crt", "place": "pl",
    "square": "sq", "parkway": "pkwy", "highway": "hwy", "lane": "ln",
    "east": "e", "west": "w", "north": "n", "south": "s",
}
_STREET_RE = re.compile(
    r"\b(" + "|".join(STREET_TYPES) + r")\b", re.IGNORECASE
)


def normalize(text: str) -> str:
    text = text.translate(UNICODE_SPACES)
    for bad, good in UNICODE_PUNCT.items():
        text = text.replace(bad, good)
    text = _STREET_RE.sub(lambda m: STREET_TYPES[m.group(1).lower()], text)
    return re.sub(r"[ \t]+", " ", text)


# A capital at the start of a sentence, line, list item or table cell means
# nothing — "Departure", "Walking", "Getting" are formatting, not facts. Only
# count a phrase as a claim if it appears capitalised somewhere it didn't have
# to be. This removed most of the false positives on a real answer.
SENTENCE_START = re.compile(r"(?:^|[.!?:]\s+|\n\s*|[|#>*_\-]+\s*)$")


# Markdown headings and bold-only lines are document structure. Enumerating
# their vocabulary ("Journey Overview", "Alternative Journey Option",
# "Slightly Later Departure") is unbounded — every answer invents new ones.
# Removing the STRUCTURE is a rule; listing the words is whack-a-mole.
HEADING_LINE = re.compile(r"^\s*(?:#{1,6}\s+.*|\*\*[^*]+\*\*:?\s*)$", re.M)
TABLE_DIVIDER = re.compile(r"^\s*\|[\s|:-]+\|\s*$", re.M)


def strip_formatting(text: str) -> str:
    """Drop headings and table scaffolding, keep prose and data rows."""
    text = HEADING_LINE.sub("", text)
    text = TABLE_DIVIDER.sub("", text)
    return text


def claims(text: str) -> list[str]:
    """Claim-like tokens worth checking against a source."""
    text = strip_formatting(normalize(text))
    found = []

    for m in PROPER_NOUN.finditer(text):
        words = m.group(1).split()

        # Trim leading stopwords. "**What Kensington Market is like**" made the
        # regex swallow the heading's "What" into the phrase, so a grounded
        # name was reported as an invented one. The capitalisation belongs to
        # the markdown, not to the claim.
        while words and words[0].lower() in STOPWORDS:
            words.pop(0)
        if not words:
            continue

        phrase = " ".join(words)
        # Multi-word proper nouns are claims wherever they appear; a lone
        # capitalised word only counts mid-sentence.
        if len(words) == 1 and SENTENCE_START.search(text[:m.start()][-24:]):
            continue
        found.append(phrase)

    found += NUMERIC.findall(text)
    # Preserve order, drop duplicates.
    seen, out = set(), []
    for c in found:
        key = c.lower()
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def check(answer: str, sources: list[str]) -> dict:
    """Compare an answer's specifics against the text it was given.

    `sources` should be everything the agent actually saw — retrieved guide
    passages AND tool results like journey legs, since a departure time is
    grounded by the schedule tool, not by the guides.
    """
    haystack = normalize(" ".join(sources)).lower()
    supported, unsupported = [], []

    for claim in claims(answer):
        low = claim.lower()
        # A multi-word phrase counts as supported if the whole phrase appears,
        # or if every word does — guides may say "Kensington" and "Market"
        # separately, and calling that invented would be a false alarm.
        whole = low in haystack
        # Only the DISTINCTIVE words need grounding. "Spadina Streetcar" is
        # supported by a source mentioning Spadina — demanding the corpus also
        # contain the word "streetcar" flags a true statement as invented.
        distinctive = [w for w in low.split() if w not in STOPWORDS]
        parts = bool(distinctive) and all(w in haystack for w in distinctive)
        (supported if (whole or parts) else unsupported).append(claim)

    total = len(supported) + len(unsupported)
    return {
        "claims": total,
        "supported": len(supported),
        "unsupported": unsupported,
        "coverage": round(len(supported) / total, 3) if total else 1.0,
    }


def summary(result: dict) -> str:
    if not result["claims"]:
        return "no checkable claims"
    line = (f"{result['supported']}/{result['claims']} specifics grounded "
            f"({result['coverage']:.0%})")
    if result["unsupported"]:
        shown = ", ".join(result["unsupported"][:5])
        extra = "" if len(result["unsupported"]) <= 5 else ", ..."
        line += f" — unsupported: {shown}{extra}"
    return line
