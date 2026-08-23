"""
Stage 3: produce a validated Itinerary instead of prose.

    python plan.py "how do I get from Kensington Market to the Distillery District?"

Two-phase design, and the split is the point:

  Phase 1 (research)  the normal agent loop, with tools, gathering facts.
  Phase 2 (structure) no tools, one job: emit JSON matching the schema.

Why separate them? A model juggling "which SQL do I write next" and "what
shape must my output be" does both worse. Splitting gives each phase one
job. It costs one extra request and buys a large reliability gain.

Phase 2 retries against Pydantic's validation errors. This is self-correction
with a perfect grader: unlike a SQL error, a schema violation is unambiguous
and the error message names the exact field. It's the cleanest example of the
pattern you'll build in this project.
"""

import json
import re
import sys
import time
from dataclasses import dataclass, field

from pydantic import ValidationError

# providers exposes accessors (current(), model(), describe()) rather than
# module-level values, because failover reassigns the active provider at
# runtime. `from agent import provider` used to bind the old value forever and
# this footer reported "gemini" after failing over to Groq. Functions can't go
# stale, so that class of bug is now structurally impossible.
from transit.core import agent
from transit.verify import constraints
from transit.verify import grounding
from transit.core import llm
from transit.tools import memory
from transit.core import providers
from transit.core.agent import run
from transit.core.llm import call_model
from transit.verify.schemas import Itinerary

STRUCTURE_PROMPT = """\
Convert the research below into a single JSON object matching this schema.

{schema}

Rules:
- Output ONLY the JSON object. No prose, no markdown fences.
- Times must be GTFS format "HH:MM:SS". After midnight use 24+ notation:
  1:23am on the next day is "25:23:00", never "1:23 AM" and never "01:23:00".
- Every non-walk leg needs a route.
- If NO journey satisfies the constraints, set feasible=false, leave legs
  EMPTY, and put the reason in infeasible_reason. Do NOT invent a placeholder
  leg. "This cannot be done, here is why" is a valid and useful answer.
- Legs must be in chronological order and must not overlap.
- Put anything you could not verify into caveats. An empty caveats list is a
  claim that everything is confirmed — only make it if that's true.
- Do NOT invent times. Every depart/arrive must trace to a value in the
  research. If the research never established a real departure time, say so
  in caveats as the FIRST entry, in those words.
{truncation_warning}
RESEARCH:
{research}
"""

TRUNCATION_WARNING = """\
- IMPORTANT: the research phase ran out of tool calls before finishing. Some
  values below are likely estimates rather than retrieved data. Your FIRST
  caveat must state plainly which values were not verified against the
  schedule database.
"""

# Stronger than the truncation warning, because this case is worse: the run
# finished cleanly, so nothing looks wrong, but no schedule data was ever
# retrieved. Left alone, the model writes plausible times AND a caveat
# claiming they came from the database.
NO_DATA_WARNING = """\
- CRITICAL: no successful query against the schedule database was made during
  research. Every attempt failed or returned nothing. You therefore have NO
  real departure or arrival times.
  Your FIRST caveat must be exactly:
    "No schedule data was retrieved; all times below are estimates, not
     verified departures."
  Do NOT write any caveat claiming a time was extracted, confirmed, or
  verified from the database. That would be false.
"""

RESEARCH_SUFFIX = """

Research this as a concrete journey. You are NOT done until find_direct_trips
has returned real clock times for every leg. Establish:
- each leg: mode, route, origin stop_id, destination stop_id
- the actual scheduled departure and arrival for each leg, from the tools
- anything you could not verify

If find_direct_trips says a transfer is required, find an interchange stop
and call it once per leg. Do not substitute estimated times for a leg you
failed to look up — go and look it up.

Report your findings as plain notes. Do not format them as JSON."""


def extract_json(text: str) -> str:
    """Models wrap JSON in prose or fences no matter how firmly you ask."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1)
    braced = re.search(r"\{.*\}", text, re.S)
    return braced.group(0) if braced else text


def structure(
    research: str,
    attempts: int = 3,
    verbose: bool = True,
    truncated: bool = False,
    no_schedule_data: bool = False,
) -> Itinerary:
    """Turn research notes into a validated Itinerary, retrying on error."""
    warning = ""
    if no_schedule_data:
        warning = NO_DATA_WARNING
    elif truncated:
        warning = TRUNCATION_WARNING

    schema = json.dumps(Itinerary.model_json_schema(), indent=2)
    messages = [
        {
            "role": "user",
            "content": STRUCTURE_PROMPT.format(
                schema=schema, research=research, truncation_warning=warning
            ),
        }
    ]

    last_error = None
    for attempt in range(attempts):
        response = call_model(messages, verbose=verbose, use_tools=False)
        raw = response.choices[0].message.content or ""

        try:
            return Itinerary.model_validate_json(extract_json(raw))
        except (ValidationError, ValueError) as exc:
            last_error = exc
            if verbose:
                first = str(exc).splitlines()[0]
                print(f"  ! invalid output (attempt {attempt + 1}): {first}",
                      file=sys.stderr)
            if attempt == attempts - 1:
                break
            # Hand back the model's own output plus the precise complaint.
            # Naming the field is what makes the next attempt likely to work.
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (
                    f"That failed validation:\n\n{exc}\n\n"
                    "Fix ONLY the fields named above and return the corrected "
                    "JSON object. No prose, no fences."
                ),
            })

    raise ValueError(f"Could not produce a valid Itinerary: {last_error}")


REPAIR_PROMPT = """\
The itinerary below is scheduled-valid in shape but violates real-world
constraints, checked against the TTC database:

{itinerary}

VIOLATIONS
{violations}

Fix them using your tools. Each violation says what to change — follow it.
Specifically:
- For a tight transfer or an unscheduled departure, call plan_journey or
  find_direct_trips again and use a departure that actually exists. Do not
  adjust a time by hand; a plausible-looking time you invented is exactly
  what created this problem.
- For a walk that's too fast, allow the stated duration and shift later legs.
- If a constraint genuinely cannot be satisfied (no service that late, no
  route without a transfer), say so plainly instead of producing something
  that looks compliant.

Report the corrected journey as plain notes, with real retrieved times."""


def repair(itinerary, violations, prefs, rounds: int = 2, verbose: bool = True,
           collected_sources: list | None = None):
    """Hand violations back to the agent and re-verify. Bounded rounds.

    The repair goes through the full agent loop WITH TOOLS, not through the
    structuring pass. Fixing "only 1 minute to make the 504" requires looking
    up a later 504 — no amount of reformatting the existing JSON will do it.
    A repair loop that can't gather new information can only produce
    plausible-looking edits, which is how you get an itinerary that passes
    validation by having its numbers quietly changed.
    """
    for attempt in range(rounds):
        if verbose:
            print(f"\n  [repair {attempt + 1}/{rounds}] "
                  f"{len(violations)} violation(s)", file=sys.stderr)
            for v in violations:
                print(f"      {v}", file=sys.stderr)

        notes = run(
            REPAIR_PROMPT.format(
                itinerary=itinerary.render(),
                violations="\n".join(f"- {v}" for v in violations),
            ),
            verbose=verbose,
            require_times=True,
        )
        # run() resets trace.EVENTS, so harvest this round's tool results
        # before the next call wipes them. Without this the grounding check
        # ends up comparing the final answer against only the last repair's
        # sources, and reports genuine facts from the research phase as
        # invented.
        if collected_sources is not None:
            collected_sources.extend(
                e["result"] for e in agent.trace.EVENTS if e["kind"] == "tool_call"
            )
        try:
            fixed = structure(notes, verbose=verbose)
        except ValueError as exc:
            if verbose:
                print(f"  [repair] could not restructure: {exc}", file=sys.stderr)
            return itinerary, violations

        remaining = constraints.verify(fixed, prefs)
        # Only accept a repair that actually improves things. A "fix" that
        # trades two violations for three is a regression, and without this
        # check the loop would happily accept it and call itself done.
        if len(remaining) < len(violations):
            itinerary, violations = fixed, remaining
            if not violations:
                return itinerary, violations
        else:
            if verbose:
                print(f"  [repair] no improvement "
                      f"({len(violations)} -> {len(remaining)}), keeping original",
                      file=sys.stderr)
            return itinerary, violations

    return itinerary, violations

@dataclass
class PlanResult:
    """Everything a caller needs, with nothing printed.

    _plan() used to be one 140-line function that computed and printed in the
    same breath, which was fine while stdout was the only consumer. A second
    consumer — the Streamlit UI — could only get at the results by
    reimplementing the pipeline, and a reimplementation drifts. Every fix
    would have to be made twice, and the second copy would silently rot.

    So: one pipeline that returns, two presentations that print. `flags` is
    computed here rather than by each caller for the same reason — the rule
    for when an itinerary counts as UNVERIFIED is a property of the result,
    not of how you happen to be displaying it.
    """

    question: str
    research: str
    prefs: constraints.Preferences
    remembered: list
    itinerary: object | None = None
    violations: list = field(default_factory=list)
    grounding: dict = field(default_factory=dict)
    sources: list = field(default_factory=list)
    truncated: bool = False
    no_schedule_data: bool = False
    steps: int = 0
    repeats: int = 0
    trace_path: object | None = None
    error: str | None = None

    @property
    def flags(self) -> list:
        out = []
        if self.error:
            out.append("FAILED")
        elif self.itinerary is not None and not self.itinerary.feasible:
            out.append("NO ROUTE")
        elif self.no_schedule_data:
            out.append("UNVERIFIED TIMES")
        if self.truncated:
            out.append("TRUNCATED")
        if self.repeats:
            out.append(f"{self.repeats} blocked repeats")
        return out


def plan(question: str, prefs: constraints.Preferences | None = None,
         verbose: bool = True, record: bool = True) -> PlanResult:
    """Research, structure, verify, repair. Returns; does not print."""
    # Zero the per-run counters. Harmless on the command line, essential in
    # Streamlit: that process survives many questions, and without this the
    # second one reports the first one's tokens and seconds too.
    llm.reset_run()
    started = time.perf_counter()

    if prefs is None:
        # Environment first, then memory fills the gaps. Precedence matters:
        # what the traveller says NOW must beat what they said last week, or
        # stored memory becomes impossible to escape without editing a database.
        prefs = constraints.Preferences.from_env()
    prefs, remembered = memory.apply_to(prefs)

    # State the constraints up front as well as checking them afterwards.
    # Verification alone would work, but every violation costs a repair round,
    # and a repair round costs a full agent run.
    research = run(
        question + RESEARCH_SUFFIX
        + f"\n\nThe traveller's constraints: {prefs.describe()}.",
        require_times=True, require_grounding=True, verbose=verbose,
    )

    result = PlanResult(
        question=question, research=research, prefs=prefs,
        remembered=remembered,
        truncated=agent.LAST_RUN["truncated"],
        # The load-bearing check: did a real CLOCK TIME ever come back from
        # the database? "Did query_transit return rows" was too weak — a run
        # that only looked up a service_id passed it while inventing every
        # departure. Check for the thing you need, not a proxy for it.
        no_schedule_data=agent.LAST_RUN["times_retrieved"] == 0,
        steps=agent.LAST_RUN["steps"],
        repeats=agent.LAST_RUN["repeats"],
    )

    try:
        result.itinerary = structure(
            research, truncated=result.truncated,
            no_schedule_data=result.no_schedule_data, verbose=verbose,
        )
    except ValueError as exc:
        result.error = str(exc)
        return result

    # Harvest the research phase's tool results NOW: repair rounds call run()
    # again, which resets the trace.
    result.sources = [e["result"] for e in agent.trace.EVENTS
                      if e["kind"] == "tool_call"]

    # An infeasible result has no times, so "no schedule data was retrieved"
    # and an UNVERIFIED TIMES flag are noise — they describe a missing
    # verification of something that was never claimed. Warnings about absent
    # data only make sense when data was supposed to be there.
    if not result.itinerary.feasible:
        result.itinerary.caveats = [
            c for c in result.itinerary.caveats
            if "no schedule data" not in c.lower() and "estimates" not in c.lower()
        ]
        result.no_schedule_data = False

    result.violations = ([] if not result.itinerary.feasible
                         else constraints.verify(result.itinerary, prefs))
    if result.violations:
        result.itinerary, result.violations = repair(
            result.itinerary, result.violations, prefs,
            collected_sources=result.sources, verbose=verbose,
        )

    # Surviving violations go INTO the itinerary, not just onto stderr. A run
    # left an invented 08:10:26 departure in the printed output with the
    # warning on a stream the reader may never see. Detecting a problem and
    # then presenting the output as fact anyway is the same failure as not
    # detecting it — worse, because we knew.
    if result.violations:
        result.itinerary.caveats = [
            f"UNVERIFIED: {v.detail}" for v in result.violations
        ] + list(result.itinerary.caveats)

    # Grounding: do the answer's specifics trace to what the tools returned?
    # RAG improves the material a model works from; it does not stop it
    # embellishing.
    result.grounding = grounding.check(result.itinerary.render(), result.sources)

    if record:
        # Our own wall clock, not one derived from trace events. agent.run()
        # resets the trace on every call, so by the time we write, EVENTS holds
        # only the LAST run — the structuring and repair passes fall outside
        # it. Deriving the wall clock from that span made the totals exceed it
        # and the reporter announced 1.7x parallelism on a sequential run.
        result.trace_path = agent.write_trace(
            question,
            answer=result.itinerary.render(),
            wall_seconds=time.perf_counter() - started,
            extra={
                "research_notes": research,
                "itinerary": result.itinerary.model_dump(),
                "connection_gaps_min": result.itinerary.connection_gaps(),
                "phase": "plan",
                "grounding": result.grounding,
                "constraints": {
                    "preferences": prefs.describe(),
                    "from_memory": remembered,
                    "violations": [str(v) for v in result.violations],
                },
            },
        )
    return result


def _plan() -> None:
    """The command line. Printing only — the pipeline is plan()."""
    question = " ".join(sys.argv[1:]) or (
        "How do I get from Kensington Market to the Distillery District "
        "on a weekday morning?"
    )
    prefs = constraints.Preferences.from_env()
    merged, remembered = memory.apply_to(prefs)

    print(f"Researching: {question}", file=sys.stderr)
    print(f"Constraints: {merged.describe()}", file=sys.stderr)
    if remembered:
        print(f"  (from memory: {', '.join(remembered)})", file=sys.stderr)
    print(file=sys.stderr)

    result = plan(question, prefs=prefs)

    if result.no_schedule_data:
        print("\n  [!] NO successful schedule query — every time in this "
              "itinerary is an estimate", file=sys.stderr)
    elif result.truncated:
        print("\n  [!] research was truncated — the itinerary will be marked "
              "as partly unverified", file=sys.stderr)

    if result.error:
        print(f"\n{result.error}", file=sys.stderr)
        print("\nFalling back to the research notes:\n")
        print(result.research)
        sys.exit(1)

    print(result.itinerary.render())

    if result.violations:
        print(f"\n  [!] {constraints.report(result.violations)}", file=sys.stderr)
    else:
        print("\n  [ok] all constraints satisfied", file=sys.stderr)

    if result.grounding.get("unsupported"):
        print(f"\n  [grounding] {grounding.summary(result.grounding)}",
              file=sys.stderr)

    suffix = f" | {', '.join(result.flags)}" if result.flags else ""
    print(f"\n[{llm.usage_line()} | steps: {result.steps}{suffix}]",
          file=sys.stderr)
    print(f"[trace: {result.trace_path}]", file=sys.stderr)


def main() -> None:
    """Entry point, wrapping the run so a crash still leaves a record.

    This used to live under `if __name__ == "__main__":`. Adding the root
    launcher moved the entry point to `from transit.pipeline.plan import main`,
    so the guard stopped running and the crash trace silently disappeared —
    discovered when a 400 killed a run five requests in and left nothing
    behind. Logic in a `__main__` guard is unreachable the moment anything
    imports the module; tests/test_imports.py now checks for it.
    """
    try:
        _plan()
    except llm.DailyQuotaExhausted as exc:
        # Not a crash: an expected, explained end state. A traceback here
        # implies something is broken and invites debugging, when the correct
        # response is "come back tomorrow or change provider".
        print(f"\n{exc}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        import traceback
        # A crash is precisely when the trace is most useful, and precisely
        # when the happy-path write at the end of _plan() never runs.
        try:
            path = agent.write_trace(
                " ".join(sys.argv[1:]) or "(default question)",
                answer="",
                extra={"phase": "crashed", "traceback": traceback.format_exc()},
            )
            print(f"\n[crash trace: {path}]", file=sys.stderr)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
