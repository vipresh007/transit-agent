"""The agent loop.

No framework. Send the conversation plus tool schemas, get back either text
(done) or tool calls (not done); run the tools, append results, repeat. That
is the whole idea — everything else in agent frameworks is ergonomics,
persistence and error handling layered on top.

What lives elsewhere, so this file stays readable:
    providers.py  which model, failover, pacing
    llm.py        retries, rate limits, quota, token accounting
    cache.py      replaying identical requests
    trace.py      the JSON record of a run
    tools/        the tools themselves

What's left here is the loop and the guardrails around it, each of which
exists because of a specific failure:
    step budget + nudge      it explored forever and got guillotined
    forced final answer      it returned "I gave up" holding good data
    duplicate blocking       five find_pois calls at different radii
    barren-result blocking   it retried tools that kept returning nothing
    times pushback           it answered in 2 steps with invented times
    LAST_RUN flags           a clean-looking run that verified nothing

Run:  python agent.py "what's the last eastbound 501 on a weekday?"
"""

import json
import os
import re
import sys
import time
from datetime import date

from dotenv import load_dotenv
from openai import NotFoundError

load_dotenv()

from transit.core import cache
from transit.verify import grounding
from transit.core import llm
from transit.core import providers
from transit.core import trace
from transit.core.llm import DailyQuotaExhausted, call_model
from transit.core.threadstate import ThreadLocalDict
from transit.tools import SCHEDULE_TOOLS, TOOL_FUNCTIONS

MAX_STEPS = providers.MAX_STEPS


TODAY = date.today()

SYSTEM_PROMPT = f"""You are a travel planning assistant for Toronto.

Today's date is {TODAY:%A, %d %B %Y} ({TODAY:%Y%m%d} in GTFS format).

Tools available: geocoding, weather, points of interest, a SQL database of
the real TTC schedule, and Wikivoyage travel guides.

Use them rather than guessing. You do not know today's weather, whether a
museum is open, or when the last streetcar runs, and inventing those details
makes you useless.

MEMORY. Call recall_preferences at the start of a planning task to see what
this traveller has told you before.

When they state a preference, decide whether it is durable before saving it:
  "I'd rather walk than take a bus"   -> remember(scope='standing')
  "I never travel before 9am"          -> remember(scope='standing')
  "I need to be there by 3pm"          -> scope='trip'; do NOT persist
  "today I'm in a hurry"               -> scope='trip'
A one-off saved as standing quietly constrains every future journey, and
nobody will connect next month's odd answer to today's throwaway remark.
When in doubt, use 'trip'.

CHOOSING BETWEEN THE SCHEDULE AND THE GUIDES — get this right first:
  "when", "how do I get to", "what time", "which route"  -> plan_journey /
      query_transit. The schedule is authoritative and the guides are not.
  "what's it like", "worth seeing", "where should I eat", "is it walkable"
      -> search_guides. The database has no opinions, only timetables.
  Questions with both ("rainy afternoon near Kensington, how do I get
      there?") need both tools. Retrieve, then route.

Do not call search_guides for a departure time — it returns prose, and prose
that sounds confident about a time is exactly how wrong answers happen.

REACT TO RETRIEVAL QUALITY. search_guides reports a "quality" field:
  strong    use the passages, move on
  moderate  usable; don't search again just to feel more certain
  weak      your wording probably isn't in the guides. Search ONCE more using
            the "suggested_terms" it returns — those are real section
            headings from the corpus. Then work with whatever you get.
  nothing relevant  say so. Do not dress a weak match up as an answer.

Rewrite toward the corpus, not toward synonyms you like. "Rainy day" appears
nowhere in these guides; "indoor", "museum" and "gallery" appear constantly.
Two searches is the limit — a third means the corpus doesn't cover it.

GROUND YOUR PROSE. State specifics — street names, venues, prices — only if
they appeared in a tool result. General impressions are fine; invented
particulars are not, and they're the ones a reader will act on.

For a journey between two places, do exactly this:
  1. geocode each place to get coordinates
  2. if the traveller avoids a mode, check_mode_feasibility on the destination
  3. plan_journey(origin_lat, origin_lon, dest_lat, dest_lon, after_time)

plan_journey searches nearby stops at both ends, finds a direct ride or
computes a single-transfer option with a real interchange, and returns
verified times for every leg. Those few calls answer the whole question.

If check_mode_feasibility says the trip is impossible without an avoided mode,
SAY SO and stop. Do not find another way, and do not use a route you remember:
routes close. Line 3 RT shut in 2023 and is not in this feed. An itinerary
built on a line that no longer exists is worse than telling the traveller
their preference conflicts with reality.

Do NOT hand-pick stops, guess an interchange, or write journey SQL yourself.
Every attempt at that produced empty results that were then misread as "no
service". find_direct_trips and query_transit exist for follow-ups
plan_journey doesn't cover — not as a substitute for it.

A stop_id is ONE PLATFORM serving ONE direction. College St at Augusta Ave
is stop 809 eastbound and stop 12338 westbound. Check the 'serves' headsigns
from find_nearby_stops to pick the platform pointing your way — querying the
wrong one returns zero rows, which reads exactly like "no service".

Never search the stops table by neighbourhood or landmark name. Stop names
are intersections ('College St at Augusta Ave'); 'Kensington Market' and
'Distillery District' do not appear in the data at all. Coordinates are the
only bridge between a place and its stops.

ONLY pair a route with stops that find_nearby_stops listed as serving it.
That tool returns a "routes" field per stop. Querying route 504 at a stop
whose routes are ["306","506"] returns nothing, and the empty result looks
identical to "no service" — you will misread it as a scheduling fact.

Do NOT filter on calendar.start_date or calendar.end_date. This feed has one
active service period, and a date outside it silently returns zero rows.
Selecting the right service_id is sufficient.

Vague times need a concrete departure time before you can query anything.
Use these unless the user gives one, and say which you assumed:
    morning 08:00 | midday 12:00 | afternoon 14:00
    evening 19:00 | night 22:00  | "now" the current time
Then ask for departures AFTER that time, not MIN() over the whole day —
MIN() gives you the 05:29 first run, which nobody means by "morning".

If a query returns an error, read it and write a corrected query rather than
giving up. If a query returns zero rows, suspect your filters before
concluding the service does not exist.

MINIMUM WORK BEFORE ANSWERING — this comes first:
Never state a departure or arrival time you did not retrieve from a tool.
For a journey question you MUST have called find_direct_trips (or
query_transit) and received actual clock times before you answer. Stopping
early and filling in plausible times is a failure, not efficiency — an
itinerary of invented times is worse than no itinerary.

If find_direct_trips reports that a transfer is required, that is the middle
of the job, not the end: pick an interchange and call it again for each leg.

BUDGET AND STOPPING — once you have real data:
Don't explore aimlessly. Before each ADDITIONAL query, ask: "will this change
my answer?" If it only adds confidence or detail, skip it. Never keep
querying to remove the last bit of uncertainty — state it as a caveat.

If a detail is ambiguous, pick the most reasonable reading, say which one
you picked, and move on. Do not run queries to resolve ambiguity.

Answer concisely and concretely, with real place names and real times. If
the data doesn't support a confident answer, say so plainly instead of
filling the gap."""


# --- per-run state ---------------------------------------------------------
# Reset at the top of every run(). These flags are how callers tell a run that
# finished from a run that actually established something — different
# questions, and only the second one licenses trusting the answer.
def _fresh_run_state() -> dict:
    return {
    "truncated": False,
    "steps": 0,
    "repeats": 0,
    "productive": set(),   # tools that returned real data
    "barren": set(),       # tools that only ever errored or came back empty
    "times_retrieved": 0,  # tool results that actually contained clock times
    "searches": 0,         # how many times the guides were queried
    "best_retrieval": 0.0, # best similarity seen across those searches
    "rewrote_query": False,# did it re-query after a weak result?
    "grounding": None,     # coverage report for the final answer
    }


# Thread-local, because stage 9 runs agents concurrently. Kept as a
# dict-shaped object so every existing LAST_RUN["..."] call site is unchanged.
LAST_RUN = ThreadLocalDict(_fresh_run_state)

# "Did query_transit return rows?" turned out to be the wrong question. A run
# that only looked up `SELECT DISTINCT service_id FROM calendar` satisfies it
# while establishing no schedule whatsoever -- and then reports invented
# departure times with no warning. What we actually care about is whether a
# CLOCK TIME ever came back from the data. Matches '8:03:00' and '25:23:17'.
GTFS_TIME_IN_RESULT = re.compile(r"\b\d{1,2}:[0-5]\d:[0-5]\d\b")

# Result prefixes that mean "you learned nothing". Kept as a list rather than
# inlined so it's obvious what counts as progress and easy to adjust.
BARREN_MARKERS = (
    "error",
    "sql error",
    "no location found",
    "query returned no rows",
    "not found",
    "unknown category",
)


# Every tool result stays in the conversation for the rest of the run, and
# every subsequent request resends the whole thing. A 50-row JSON result isn't
# expensive once -- it's expensive eight more times. Token budgets (Groq bills
# 200k tokens/day) run out long before request counts do, so this is the lever
# that actually matters.
MAX_RESULT_CHARS = int(os.getenv("MAX_RESULT_CHARS", "2500"))


def clip(result: str) -> str:
    """Truncate an oversized tool result, telling the model it was truncated.

    Silent truncation is dangerous -- the model would treat a cut-off list as
    complete. Saying so lets it narrow the query instead.
    """
    if len(result) <= MAX_RESULT_CHARS:
        return result
    kept = result[:MAX_RESULT_CHARS]
    return (
        f"{kept}\n\n[TRUNCATED: {len(result) - MAX_RESULT_CHARS} more chars. "
        f"This result was too large. Re-run with a tighter WHERE clause, "
        f"fewer columns, or a smaller LIMIT to see the rest.]"
    )


def _is_barren(result: str) -> bool:
    low = result.strip().lower()
    return (
        not low
        or low.startswith(BARREN_MARKERS)
        or low.startswith("no ")          # "No museum found within 1500m."
    )


# Phrases that claim provenance. An answer may cheerfully invent venues; the
# genuinely dangerous move is telling the reader those inventions came from a
# source. One real run listed nine venues found nowhere in the corpus and
# closed with "all of the venues above are listed in the Wikivoyage guides,
# so you can trust the opening-hour details".
PROVENANCE_CLAIM = re.compile(
    r"\b(?:listed in|according to|from|per|sourced from|based on|found in)\b"
    r"[^.]{0,40}\b(?:the )?(?:guides?|wikivoyage|schedule|data|sources?)\b",
    re.IGNORECASE,
)

# Below this fraction of grounded specifics, an answer is substantially the
# model's own knowledge wearing a retrieved-looking costume.
MIN_GROUNDING = float(os.getenv("MIN_GROUNDING", "0.85"))


def run(user_message: str, verbose: bool = True, require_times: bool = False,
        require_grounding: bool = False) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    LAST_RUN.update(
        truncated=False, steps=0, repeats=0, productive=set(), barren=set(),
        times_retrieved=0, searches=0, best_retrieval=0.0, rewrote_query=False,
        grounding=None,
    )
    trace.reset()
    seen_calls: dict[str, int] = {}   # exact (tool, args) -> times requested
    barren: dict[str, int] = {}       # tool -> consecutive useless results
    pushed_back = False               # allow exactly one "go do the work"
    grounding_pushed = False          # and exactly one "cite what you retrieved"
    empty_retried = False             # and exactly one "you said nothing"

    for step in range(MAX_STEPS):
        remaining = MAX_STEPS - step

        # The model has no idea how many steps it has left, so it explores as
        # if the budget were infinite and gets truncated mid-thought. Telling
        # it the budget converts a hard cutoff into a deadline it can plan for.
        if remaining == 3:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You have 2 tool calls left. Stop exploring. Make at "
                        "most one more query if it's essential, then give your "
                        "final answer with whatever you have."
                    ),
                }
            )

        response = call_model(messages, verbose=verbose)
        message = response.choices[0].message

        # No tool calls means the model thinks it's finished.
        if not message.tool_calls:
            # ...but "finished" and "did the work" are different claims.
            # Prompting alone didn't hold: after the anti-thrash guidance told
            # it to answer quickly, it started stopping after two steps and
            # inventing times. A precondition checked in code is a guarantee;
            # a precondition described in a prompt is a suggestion.
            # One pushback only — if it still won't, the caller gets the
            # honest unverified answer rather than an infinite loop.
            if (
                require_times
                and LAST_RUN["times_retrieved"] == 0
                and not pushed_back
                and step < MAX_STEPS - 2
            ):
                pushed_back = True
                if verbose:
                    print(
                        "  [!] tried to answer with no retrieved times — "
                        "pushing back",
                        file=sys.stderr,
                    )
                trace.event("pushback", step=step, draft=message.content)
                messages.append(message.model_dump(exclude_none=True))
                messages.append({
                    "role": "user",
                    "content": (
                        "Stop. You have not retrieved a single scheduled time "
                        "from the data, so every time in that answer is "
                        "invented. Use find_direct_trips with the stop_ids you "
                        "already have. If it reports a transfer is required, "
                        "pick an interchange and call it once per leg. Do not "
                        "answer again until you have real times."
                    ),
                })
                continue

            # Grounding pushback. Same shape as the times check: verify a
            # precondition in CODE before accepting an answer, because the
            # prompt already asked for this and the model complied only
            # partially. Fires once, then we accept what we get.
            draft = message.content or ""
            if require_grounding and not grounding_pushed and draft:
                sources = [e["result"] for e in trace.EVENTS
                           if e["kind"] == "tool_call"]
                report = grounding.check(draft, sources)
                LAST_RUN["grounding"] = report
                claims_provenance = bool(PROVENANCE_CLAIM.search(draft))
                thin = report["claims"] >= 5 and report["coverage"] < MIN_GROUNDING

                if sources and (thin or (claims_provenance and report["unsupported"])):
                    grounding_pushed = True
                    if verbose:
                        print(f"  [!] grounding {report['coverage']:.0%} — "
                              f"pushing back on {len(report['unsupported'])} "
                              f"unsupported specifics", file=sys.stderr)
                    trace.event("grounding_pushback", step=step,
                                coverage=report["coverage"],
                                unsupported=report["unsupported"])
                    messages.append(message.model_dump(exclude_none=True))
                    messages.append({
                        "role": "user",
                        "content": (
                            "Stop. These specifics appear nowhere in the tool "
                            "results you were given:\n\n  "
                            + ", ".join(report["unsupported"][:15])
                            + "\n\nThey came from your own knowledge, not from "
                            "the retrieved sources. Rewrite so that every named "
                            "venue, street, price and time is one you actually "
                            "retrieved. Drop the rest — a shorter sourced "
                            "answer is worth more than a long one that mixes "
                            "retrieval with recall.\n\n"
                            "And do NOT claim the guides are the source for "
                            "anything you did not retrieve from them. If you "
                            "add general knowledge, label it as such."
                        ),
                    })
                    continue

            # A model that finishes its tool calls and then says nothing has
            # not answered. One run saved two preferences and returned an
            # empty string. Ask once, without tools, for the actual reply.
            if not draft.strip() and not empty_retried:
                empty_retried = True
                if verbose:
                    print("  [!] empty final answer — asking again",
                          file=sys.stderr)
                messages.append({
                    "role": "user",
                    "content": ("You returned nothing. Summarise what you did "
                                "and answer the question in plain prose."),
                })
                response = call_model(messages, verbose=verbose, use_tools=False)
                draft = response.choices[0].message.content or ""

            trace.event("final", step=step, content=draft)
            return draft or "(empty response)"

        # The assistant turn must go into history before the tool results,
        # or the next request will 400 on a dangling tool_call_id.
        #
        # model_dump() rather than a hand-built dict: it round-trips provider
        # extras like Gemini's thought_signature, which a manual rebuild
        # silently drops. See the THINKING_BUDGET note above.
        messages.append(message.model_dump(exclude_none=True))

        LAST_RUN["steps"] = step + 1
        for tool_call in message.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                args = {}

            # Anti-thrash. A model that gets an unhelpful result often retries
            # the same tool with a trivially different argument -- five
            # find_pois calls at radii 500/800/1000/1500/2000 -- burning the
            # budget without gaining information. Detect the pattern and say
            # so, instead of dutifully executing the same lookup again.
            signature = f"{name}:{json.dumps(args, sort_keys=True)}"
            seen_calls[signature] = seen_calls.get(signature, 0) + 1

            if seen_calls[signature] > 1:
                LAST_RUN["repeats"] += 1
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": (
                        f"You already called {name} with these exact arguments "
                        f"and got a result. Repeating it will not help. Use "
                        f"what you have, or try a different tool."
                    ),
                })
                if verbose:
                    print(f"  [{step}] {name}(...) — duplicate, blocked",
                          file=sys.stderr)
                continue

            # Throttle on FRUITLESS calls only. Counting every call punished
            # query_transit -- the workhorse -- for doing legitimate stepwise
            # exploration, which is worse than the thrashing it was meant to
            # prevent. What actually signals a stuck agent is repeatedly
            # getting nothing back, so that's what we count.
            if barren.get(name, 0) >= 3:
                LAST_RUN["repeats"] += 1
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": (
                        f"{name} has returned no useful result {barren[name]} "
                        f"times in a row. Varying the arguments further is "
                        f"unlikely to help. Use a different tool, or proceed "
                        f"with what you have and note the gap."
                    ),
                })
                if verbose:
                    print(f"  [{step}] {name}(...) — {barren[name]} barren "
                          f"results, blocked", file=sys.stderr)
                continue

            if verbose:
                print(f"  [{step}] {name}({args})", file=sys.stderr)

            # Notify, not event: a live view wants to show "running" while a
            # 2.5s journey search happens, but recording a start/finish pair
            # would halve every per-step gap the timing report derives.
            trace.notify("tool_start", step=step, tool=name, args=args)

            started = time.time()
            fn = TOOL_FUNCTIONS.get(name)
            if fn is None:
                result = f"Error: no tool named {name!r}"
            else:
                try:
                    result = fn(**args)
                except Exception as exc:
                    # Handing the error back to the model instead of crashing is
                    # what lets it self-correct. This one line is most of the
                    # difference between a demo and something that survives use.
                    result = f"Error calling {name}: {type(exc).__name__}: {exc}"

            # Did that call teach us anything? Errors and empty results are
            # how a stuck agent looks from the outside. Consecutive, so one
            # bad query among good ones doesn't count against the tool.
            if _is_barren(str(result)):
                barren[name] = barren.get(name, 0) + 1
                LAST_RUN["barren"].add(name)
            else:
                barren[name] = 0
                LAST_RUN["productive"].add(name)
                # Any schedule tool counts, not just query_transit. When
                # find_direct_trips was added it became the primary source of
                # times and this check silently stopped seeing them — the
                # hazard of naming a specific tool instead of a capability.
                if name in SCHEDULE_TOOLS and GTFS_TIME_IN_RESULT.search(str(result)):
                    LAST_RUN["times_retrieved"] += 1

            # Retrieval telemetry. Stage 6 is about the agent REACTING to weak
            # results, and "did it react?" is only answerable if we record
            # both the quality it saw and whether it searched again.
            if name == "search_guides":
                if LAST_RUN["searches"]:
                    LAST_RUN["rewrote_query"] = True
                LAST_RUN["searches"] += 1
                try:
                    payload = json.loads(str(result))
                    # Must be a dict: "nothing relevant" is prose, and older
                    # result shapes were bare lists. Telemetry crashing the
                    # run would be a spectacular own goal — it exists to
                    # observe the run, not to end it.
                    if isinstance(payload, dict):
                        LAST_RUN["best_retrieval"] = max(
                            LAST_RUN["best_retrieval"],
                            float(payload.get("best_score", 0.0)),
                        )
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

            trace.event(
                "tool_call",
                step=step,
                tool=name,
                args=args,
                seconds=round(time.time() - started, 2),
                barren=_is_barren(str(result)),
                # Full result, not the clipped version: the whole point of a
                # trace is seeing what the model was actually looking at.
                result=str(result),
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": clip(str(result)),
                }
            )

    # Out of steps. Never return "I gave up" to a user when we're holding a
    # pile of successfully gathered data — make one final call with no tools
    # available, so the only thing the model can produce is an answer.
    LAST_RUN["truncated"] = True
    if verbose:
        print(f"  [!] step budget spent — forcing a final answer", file=sys.stderr)

    messages.append(
        {
            "role": "user",
            "content": (
                "You are out of tool calls. Answer the original question now "
                "using only what you have already gathered.\n\n"
                "CRITICAL: you must not invent any value you did not retrieve. "
                "If you never queried a departure time, do not state one — say "
                "explicitly that times were not retrieved. An answer that "
                "admits a gap is useful; an answer with plausible invented "
                "numbers is worse than no answer at all."
            ),
        }
    )
    final = call_model(messages, verbose=verbose, use_tools=False)
    return final.choices[0].message.content or (
        f"Stopped after {MAX_STEPS} steps without reaching an answer."
    )


def write_trace(question: str, answer: str = "", extra: dict | None = None):
    """Thin wrapper so callers don't have to assemble the metadata."""
    return trace.write(
        question,
        answer,
        provider=providers.current()["name"],
        model=providers.model(),
        usage=llm.USAGE,
        cache_stats=cache.STATS,
        flags=LAST_RUN,
        extra=extra,
    )


def main() -> None:
    question = " ".join(sys.argv[1:]) or "What should I do in Toronto tomorrow?"
    answer = ""
    try:
        answer = run(question, require_grounding=True)
        print(answer)
    except DailyQuotaExhausted as exc:
        print(f"\n{exc}", file=sys.stderr)
        sys.exit(1)
    except NotFoundError as exc:
        # Models get retired for new users without warning.
        print(
            f"\nModel {MODEL!r} is not available to your key.\n"
            f"Run `python scripts/list_models.py` to see what is, then update MODEL "
            f"in .env.\n\nOriginal error: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)
    finally:
        # A trace is written even when the run failed — a crashed run is
        # exactly the one you want a record of.
        path = write_trace(question, answer)
        print(f"\n[{llm.usage_line()}]", file=sys.stderr)
        print(f"[trace: {path}]", file=sys.stderr)


if __name__ == "__main__":
    main()
