"""A browser front end for the planner.

    pip install streamlit
    streamlit run ui.py

WHY A UI AT ALL. A plan takes about a minute, and 94% of that is waiting on
the model. Nothing makes that minute shorter, but a blank screen makes it feel
much longer than a list of tool calls appearing one by one. This buys
perceived speed, not speed — worth being honest that they're different.

THE ONE STRUCTURAL PROBLEM. Streamlit reruns this script top to bottom on
every interaction, and the agent is a blocking 60-second call. Running it
inline freezes the page until it finishes, which is the blank screen again
with extra steps. So the agent runs on a worker thread and posts events to a
queue that the main thread drains and renders.

That means the pipeline needs no idea a UI exists. It publishes events; this
subscribes. `plan.py` on the command line is unchanged and unaware.
"""

from __future__ import annotations

import io
import queue
import threading
import time
import traceback as tb
from contextlib import redirect_stderr

import streamlit as st

st.set_page_config(page_title="Toronto transit agent", page_icon="🚋",
                   layout="centered")

# PAINT BEFORE IMPORTING. The project imports below pull in the whole agent —
# providers, tool registry, SQLite. If any of that raises or hangs, and it
# happens before the first st.* call, the browser gets a blank page and the
# terminal gets nothing: the server is fine, the script just never finished.
# A blank page is the least debuggable failure there is, so make it impossible.
st.title("🚋 Toronto transit agent")
_loading = st.empty()
_loading.caption("loading the agent…")

# EVERY entry point must do this. plan.py, crew.py and graph.py all call
# load_dotenv(); this file didn't, so no API key reached the environment and
# providers.py exited during import — silently, because SystemExit is not an
# Exception. "Load the config" is a per-entry-point responsibility that is
# invisible until the one entry point that forgets it.

try:
    from transit import paths
    from transit.core import llm, trace
    from transit.pipeline import view
    from transit.tools import memory
    from transit.verify import constraints
# BaseException, not Exception. A module calling sys.exit() raises SystemExit,
# which `except Exception` sails straight past — and Streamlit then swallows it
# and leaves the page hanging on this caption with a clean terminal. Catching
# the narrower type is what made this bug invisible for three rounds.
except BaseException as exc:                                # noqa: BLE001
    _loading.empty()
    if isinstance(exc, SystemExit):
        st.error(f"A module called sys.exit() while importing: {exc}")
        st.caption("A library should raise, not exit — only an entry point "
                   "gets to end the process.")
    else:
        st.error("The agent failed to import. This is a code or environment "
                 "problem, not a Streamlit one.")
        st.code(tb.format_exc())
    st.stop()

_loading.empty()

# Tools whose progress is worth narrating. The rest are fast enough that a
# spinner for them is noise.
FRIENDLY = {
    "recall_preferences": "checking what you've told me before",
    "geocode": "finding the place",
    "get_weather": "checking the forecast",
    "find_pois": "looking for places nearby",
    "find_nearby_stops": "finding nearby stops",
    "check_mode_feasibility": "can we get there without buses?",
    "plan_journey": "searching the timetable",
    "find_direct_trips": "checking departures",
    "query_transit": "querying the schedule",
    "describe_transit_schema": "reading the schedule layout",
    "search_guides": "searching the travel guides",
    "remember": "saving a preference",
    "forget_preference": "forgetting a preference",
}


# ---------------------------------------------------------------------------
# Running the agent off the main thread
# ---------------------------------------------------------------------------

class StderrPump(io.TextIOBase):
    """Forward the pipeline's existing stderr logging into the queue.

    Everything interesting is ALREADY logged — rate limits, provider failover,
    grounding pushback, blocked duplicates. Re-instrumenting all of it for the
    UI would mean two sources of truth that drift. Capturing the stream we
    already have costs nothing and can't fall behind.
    """

    def __init__(self, out: queue.Queue):
        self.out = out

    def write(self, text: str) -> int:
        line = text.strip()
        if line:
            self.out.put({"kind": "log", "line": line})
        return len(text)


def run_plan(question: str, out: queue.Queue) -> None:
    """Worker thread: run the pipeline, push every event onto the queue."""
    def observer(event: dict) -> None:
        out.put(event)

    trace.subscribe(observer)
    try:
        with redirect_stderr(StderrPump(out)):
            # ONE pipeline. The first version of this file re-ran research,
            # structuring and verification by hand, which meant every fix to
            # plan.py would have had to be made twice — and the copy here
            # would have quietly rotted. plan.plan() returns; _plan() prints;
            # this renders. Nothing is duplicated.
            from transit.pipeline.plan import plan

            prefs = constraints.Preferences.from_env()
            merged, remembered = memory.apply_to(prefs)
            out.put({"kind": "prefs", "describe": merged.describe(),
                     "remembered": remembered})

            result = plan(question, prefs=prefs)
            out.put({"kind": "done", "result": result})
    except Exception as exc:                                # noqa: BLE001
        # A crash must reach the page. Swallowing it here would leave the
        # spinner turning forever, which is the worst of both worlds.
        out.put({"kind": "error", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        trace.unsubscribe(observer)
        out.put({"kind": "finished"})


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_itinerary(itinerary) -> None:
    if not itinerary.feasible:
        st.error(f"**No route found.** {itinerary.infeasible_reason}")
        return

    st.dataframe(view.leg_rows(itinerary), hide_index=True,
                 use_container_width=True)
    st.caption(f"{itinerary.total_min} min total, "
               f"{itinerary.transfers} transfer(s)")

    for index, gap in itinerary.risky_connections():
        st.warning(f"Tight transfer after leg {index + 1}: {gap} min to make "
                   f"the {itinerary.legs[index + 1].route}")


def render_badges(result) -> None:
    for column, (label, value) in zip(st.columns(3),
                                      view.badge_values(result).items()):
        column.metric(label, value)

    if result.flags:
        st.warning(" · ".join(result.flags))

    if result.violations:
        with st.expander(f"{len(result.violations)} constraint violation(s)"):
            for violation in result.violations:
                st.write(f"**{violation.kind}** — {violation.detail}")
                st.caption(f"Fix: {violation.fix}")

    unsupported = result.grounding.get("unsupported")
    if unsupported:
        with st.expander(f"{len(unsupported)} unsupported specific(s)"):
            st.caption("Present in the answer, absent from every tool result. "
                       "Some are false positives — see grounding.py.")
            st.write(", ".join(unsupported))


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

with st.sidebar:
    st.subheader("Standing preferences")
    st.caption("Saved in memory.db and enforced as constraints, not "
               "suggestions. Cleared only when you clear them.")
    try:
        rows = view.remembered_rows()
    except Exception:                                       # noqa: BLE001
        # Reading memory.db must never cost you the whole page.
        st.error("Could not read stored preferences.")
        st.code(tb.format_exc())
        rows = []
    if rows:
        for label, value, forgettable in rows:
            col_a, col_b = st.columns([3, 1])
            col_a.write(f"**{label}**: {value}" if forgettable
                        else f"_{value}_")
            # Notes get no forget button: they are shown to the model but
            # never enforced, so offering to "forget" one would imply it had
            # been changing your journeys.
            if forgettable and col_b.button("forget", key=f"forget-{label}"):
                memory.forget(label)
                st.rerun()
    else:
        st.caption("_nothing remembered yet_")

    st.divider()
    st.caption(f"transit.db {'found' if paths.TRANSIT_DB.exists() else 'MISSING'}"
               f" · guides.db {'found' if paths.GUIDES_DB.exists() else 'MISSING'}")

question = st.text_input(
    "Where are you going?",
    placeholder="how do I get from Kensington Market to the Distillery District?",
)

if st.button("Plan it", type="primary", disabled=not question):
    events: queue.Queue = queue.Queue()
    worker = threading.Thread(target=run_plan, args=(question, events),
                              daemon=True)
    worker.start()

    result = None
    started = time.time()
    running: dict[str, float] = {}

    with st.status("Researching…", expanded=True) as status:
        while True:
            try:
                event = events.get(timeout=0.2)
            except queue.Empty:
                if not worker.is_alive():
                    break
                continue

            kind = event.get("kind")
            if kind == "finished":
                break
            elif kind == "prefs":
                if event["remembered"]:
                    st.write(f"Applying: {event['describe']}")
            elif kind == "tool_start":
                tool = event["tool"]
                running[tool] = time.time()
                st.write(f"⏳ {FRIENDLY.get(tool, tool)}…")
            elif kind == "tool_call":
                tool = event["tool"]
                mark = "⚠️" if event.get("barren") else "✅"
                st.write(f"{mark} {FRIENDLY.get(tool, tool)} "
                         f"— {event.get('seconds', 0):.1f}s")
            elif kind == "phase":
                status.update(label="Building the itinerary…")
            elif kind == "log":
                line = event["line"]
                # Only the lines that explain a delay or a correction. The
                # per-tool chatter is already shown above, better formatted.
                if line.startswith(("~", "!", ".")):
                    st.caption(line)
            elif kind == "done":
                result = event
            elif kind == "error":
                result = event
            elif kind == "final":
                status.update(label="Building the itinerary…")

        elapsed = time.time() - started
        if result and result.get("kind") == "error":
            status.update(label=f"Failed after {elapsed:.0f}s", state="error")
        else:
            status.update(label=f"Done in {elapsed:.0f}s", state="complete")

    if result is None:
        st.error("The run ended without producing anything. Check the terminal.")
    elif result.get("kind") == "error":
        st.error(result["error"])
    else:
        plan_result = result["result"]
        if plan_result.error:
            st.error(plan_result.error)
            with st.expander("Research notes (the itinerary failed to parse)"):
                st.write(plan_result.research)
        else:
            render_itinerary(plan_result.itinerary)
            render_badges(plan_result)
            if plan_result.itinerary.caveats:
                with st.expander("Caveats"):
                    for caveat in plan_result.itinerary.caveats:
                        st.write(f"- {caveat}")
        st.caption(llm.usage_line())
