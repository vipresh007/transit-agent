"""Stage 10: crew.py rewritten as a LangGraph state graph.

    pip install langgraph langgraph-checkpoint-sqlite

    python graph.py --draw
    python graph.py "plan me a Saturday in Toronto: morning, lunch, afternoon"
    python graph.py --approve "..."      # pause for approval before spending
    python graph.py --resume <thread_id>
    python graph.py --history <thread_id>

THIS IS A COMPARISON, NOT AN UPGRADE. crew.py still works and still does the
same job. The point of this file is to answer one question honestly: having
hand-written decomposition, parallel execution, partial-failure handling and
synthesis, what does a framework actually add?

WHAT IS IDENTICAL
  Same three phases. Same prompts, imported from crew.py rather than copied,
  so the two cannot drift. Same agent.run() underneath, untouched. If this
  file produced different answers, the comparison would be worthless.

WHAT THE FRAMEWORK ADDS, concretely:

  1. CHECKPOINTING. State is written to graph.db after every superstep. The
     run that died on a 422 at subtask 3 resumes at subtask 3 instead of
     paying for 1 and 2 again. crew.py loses everything.

  2. INTERRUPTS. `interrupt()` stops the graph mid-run, hands control back,
     and resumes later with an injected value. That's how --approve shows you
     the plan and its estimated cost before spending 26 requests on it.
     crew.py has no way to express "stop here and ask".

  3. TIME TRAVEL. Every checkpoint is kept, so --history shows each state the
     run passed through, and you can resume from any of them.

  4. A DIAGRAM. --draw emits Mermaid straight from the graph, so the picture
     can never disagree with the code.

WHAT IT COSTS

  Control flow becomes data. In crew.py you read main() top to bottom and see
  the whole shape. Here the shape lives in add_node/add_edge calls and you
  reconstruct it in your head — which is exactly why --draw exists.

  And a real trap, demonstrated below: RESUMING RE-RUNS THE INTERRUPTED NODE
  FROM ITS FIRST LINE. Put `interrupt()` after a paid API call in the same
  node and you pay for that call twice. Durable execution is replay-based, so
  nodes must be safe to re-run. That's why planning and approval are two
  nodes here rather than one — see the comment on approve().
"""

from __future__ import annotations

import argparse
import hashlib
import operator
import os
import sqlite3
import sys
import time
from typing import Annotated, TypedDict

from dotenv import load_dotenv

load_dotenv()

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, Send, interrupt
except ImportError:                                        # pragma: no cover
    print("LangGraph is not installed. Run:\n"
          "  pip install langgraph langgraph-checkpoint-sqlite",
          file=sys.stderr)
    raise

from transit.core import agent          # noqa: E402
from transit.pipeline import crew           # noqa: E402  — prompts and helpers, reused not copied
from transit.core import llm            # noqa: E402
from transit.core import providers      # noqa: E402
from transit.core import trace          # noqa: E402
from transit import paths

CHECKPOINT_DB = paths.GRAPH_DB

# Rough, but enough to decide whether to approve: ~6 requests per subtask
# plus one to plan and one to synthesise.
REQUESTS_PER_SUBTASK = 6


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class CrewState(TypedDict, total=False):
    """What flows through the graph.

    `results` carries the reducer. Several research nodes finish in the same
    superstep and every one of them returns {"results": [...]}, so without
    `operator.add` the last writer would win and the rest would vanish
    silently. A reducer is the framework's answer to the same problem
    threadstate.py solved for LAST_RUN: concurrent writers to one container.
    """

    question: str
    tasks: list[str]
    results: Annotated[list[dict], operator.add]
    answer: str
    approved: bool
    # Which graph wrote this state. Checkpoints outlive the code that made
    # them, and LangGraph will happily resume a half-finished run into a
    # different topology without saying so — see thread_id_for().
    shape: str


class Subtask(TypedDict):
    """A Send payload. The research node sees THIS, not CrewState.

    That is the isolation crew.py achieves with a fresh thread — here it's
    structural: the node is handed only its own subtask and physically cannot
    read another branch's work.
    """

    index: int
    task: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def make_plan(shape: str):
    """One model call to split the question. Identical to crew.decompose.

    Stamped with the graph's shape so a later resume can tell whether the
    code has changed underneath the checkpoint.
    """
    def plan(state: CrewState) -> dict:
        tasks = crew.decompose(state["question"])
        print(f"Decomposed into {len(tasks)} subtask(s):", file=sys.stderr)
        for i, t in enumerate(tasks, 1):
            print(f"  {i}. {t}", file=sys.stderr)
        return {"tasks": tasks, "shape": shape}

    return plan


def approve(state: CrewState) -> dict:
    """Pause and show the cost before committing to it.

    THIS NODE IS DELIBERATELY SEPARATE FROM plan(). Resuming from an interrupt
    replays the interrupted node from its first statement — so if interrupt()
    lived at the end of plan(), every resume would re-run decompose() and pay
    for another model call. This node does no I/O, so replaying it is free.

    The general rule: everything before an interrupt() in the same node runs
    again on resume. Keep interrupts in nodes that are cheap to repeat.
    """
    estimate = len(state["tasks"]) * REQUESTS_PER_SUBTASK + 2
    decision = interrupt({
        "tasks": state["tasks"],
        "estimated_requests": estimate,
        "prompt": "Approve this plan? Resume with 'yes' or 'no'.",
    })
    return {"approved": str(decision).strip().lower() in {"y", "yes", "ok"}}


def fan_out(state: CrewState) -> list[Send] | str:
    """Conditional edge: one research branch per subtask.

    Send() is what makes the width dynamic. A plain edge is fixed at build
    time; here the number of parallel branches is decided at runtime from the
    planner's output, which is the whole reason a graph beats a flowchart.
    """
    if state.get("approved") is False:
        return "cancelled"
    return [Send("research", {"index": i, "task": t})
            for i, t in enumerate(state["tasks"])]


def research(payload: Subtask) -> dict:
    """One subtask, one agent, fresh context. Delegates to crew.research_one.

    Sync LangGraph runs a superstep's nodes on a thread pool, so the
    thread-local LAST_RUN and trace.EVENTS from stage 9 are still doing real
    work here — the framework did not remove that problem, it just happens to
    parallelise the same way. Worth knowing: had those still been module
    globals, this graph would interleave them exactly as crew.py did.
    """
    result = crew.research_one(payload["index"], payload["task"], verbose=True)
    status = "failed" if result["error"] else "done"
    print(f"  [subtask {payload['index'] + 1}] {status} "
          f"in {result['seconds']}s", file=sys.stderr)
    return {"results": [result]}


def synthesize(state: CrewState) -> dict:
    """Merge. No tools — synthesis must not invent."""
    results = sorted(state["results"], key=lambda r: r["index"])
    print("\nSynthesising...\n", file=sys.stderr)
    return {"answer": crew.synthesize(state["question"], results)}


def cancelled(state: CrewState) -> dict:
    return {"answer": "Cancelled before any research was run. "
                      "No requests were spent on subtasks."}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build(checkpointer=None, with_approval: bool = False):
    """Assemble the graph. This function IS the architecture diagram."""
    nodes = ["plan", "research", "synthesize"]
    if with_approval:
        nodes += ["approve", "cancelled"]
    shape = ",".join(sorted(nodes))

    b = StateGraph(CrewState)
    b.add_node("plan", make_plan(shape))
    b.add_node("research", research)
    b.add_node("synthesize", synthesize)

    if with_approval:
        b.add_node("approve", approve)
        b.add_node("cancelled", cancelled)
        b.add_edge(START, "plan")
        b.add_edge("plan", "approve")
        b.add_conditional_edges("approve", fan_out, ["research", "cancelled"])
        b.add_edge("cancelled", END)
    else:
        b.add_edge(START, "plan")
        b.add_conditional_edges("plan", fan_out, ["research"])

    # Fan-in. LangGraph waits for every branch of the superstep before
    # running synthesize, which is what as_completed + sorted() did by hand.
    b.add_edge("research", "synthesize")
    b.add_edge("synthesize", END)
    return b.compile(checkpointer=checkpointer)


def saver() -> tuple[SqliteSaver, sqlite3.Connection]:
    """SQLite checkpointer.

    check_same_thread=False is required, not optional: LangGraph runs a
    superstep's nodes on a thread pool and each one writes its checkpoint from
    whichever thread it landed on.
    """
    conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    return SqliteSaver(conn), conn


def pending_interrupt(snap) -> bool:
    """Is this run paused at an interrupt, as opposed to crashed mid-flight?

    Read defensively. Where the snapshot exposes it has moved between
    LangGraph versions, and guessing wrong here would send Command(resume=...)
    into a graph that never asked for it.
    """
    if getattr(snap, "interrupts", None):
        return True
    return any(getattr(t, "interrupts", None) for t in getattr(snap, "tasks", ()))


def is_finished(snap) -> bool:
    """Has this thread already run to completion?"""
    return bool(getattr(snap, "values", None)) and not snap.next


def shape_of(graph) -> str:
    """A fingerprint of the graph's topology, for keying threads."""
    return ",".join(sorted(n for n in graph.get_graph().nodes
                           if n not in ("__start__", "__end__")))


def shape_matches(snap, graph) -> bool:
    """Was this checkpoint written by a graph of the same shape?

    Unstamped state (from before this check existed) is allowed through
    rather than treated as a mismatch — refusing to resume old threads would
    be a worse failure than the one being prevented.
    """
    recorded = (getattr(snap, "values", None) or {}).get("shape")
    return not recorded or recorded == shape_of(graph)


def thread_id_for(graph, question: str) -> str:
    """Same question AND same graph shape, same thread.

    Two separate hazards, both of which produced real wrong output:

    FINISHED THREADS. `results` has an `operator.add` reducer, so state is
    APPEND-ONLY. Reusing a completed thread doesn't restart it — the old
    run's three results are still there, the new run appends three more, and
    synthesis merges six sections with three from the previous answer. No
    error, no warning, just a longer answer that looks fine.

    CHANGED TOPOLOGY. A checkpoint is NOT bound to the graph that wrote it.
    `--approve` paused a run at the `approve` node; the next command built the
    graph without that node, and LangGraph happily recomputed what came next
    against the NEW shape and carried on into research. The pause silently
    stopped existing. It got the right answer by luck — the general case is a
    run half-executed under one topology and finished under another, with the
    checkpoint offering no clue that happened.

    Both are the same underlying point: durable state outlives the code that
    created it, so anything the state assumes has to be part of its key.
    """
    base = hashlib.sha256(
        f"{question}\n{shape_of(graph)}".encode()).hexdigest()[:12]
    for n in range(100):
        tid = base if n == 0 else f"{base}-{n}"
        snap = graph.get_state({"configurable": {"thread_id": tid}})
        if not is_finished(snap):
            return tid
    return f"{base}-{int(time.time())}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def report(question: str, state: dict, tid: str, elapsed: float) -> None:
    answer = state.get("answer", "")
    print(answer)

    results = sorted(state.get("results", []), key=lambda r: r["index"])
    if results:
        # Same two-level audit as crew.py, imported rather than reimplemented.
        print(file=sys.stderr)
        ground = crew.audit(answer, results)
        crew.report_grounding(ground)
    else:
        ground = {}

    print(f"\n[{llm.usage_line()} | {len(results)} subtasks | {elapsed:.0f}s "
          f"| thread {tid}]", file=sys.stderr)

    # Same thread-local merge crew.py needs — LangGraph runs the research
    # nodes on a pool, so each one's events live on a different thread.
    merged = sorted((e for r in results for e in r["events"]),
                    key=lambda e: float(e["t"]))
    path = trace.write(
        question, answer,
        provider=providers.current()["name"], model=providers.model(),
        usage=llm.USAGE, cache_stats={}, flags={"subtasks": len(results)},
        events=merged, wall_seconds=elapsed,
        extra={
            "phase": "graph",
            "thread_id": tid,
            "subtasks": [{k: v for k, v in r.items() if k != "events"}
                         for r in results],
            "grounding": ground,
        },
    )
    print(f"[trace: {path}]", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("question", nargs="*", help="the request to plan")
    ap.add_argument("--draw", action="store_true", help="print the Mermaid diagram")
    ap.add_argument("--approve", action="store_true",
                    help="pause for approval after planning")
    ap.add_argument("--resume", metavar="THREAD_ID",
                    help="continue an interrupted or crashed run")
    ap.add_argument("--history", metavar="THREAD_ID",
                    help="list every checkpoint of a run")
    args = ap.parse_args()

    if args.draw:
        print("%% default\n" + build().get_graph().draw_mermaid())
        print("\n%% --approve\n"
              + build(with_approval=True).get_graph().draw_mermaid())
        return

    cp, conn = saver()
    try:
        if args.history:
            config = {"configurable": {"thread_id": args.history}}
            for snap in build(cp).get_state_history(config):
                nxt = ", ".join(snap.next) or "END"
                done = len(snap.values.get("results", []))
                print(f"  next={nxt:<12} subtasks_done={done}  "
                      f"{snap.config['configurable']['checkpoint_id']}")
            return

        if args.resume:
            tid = args.resume
            config = {"configurable": {"thread_id": tid}}
            # Pick the topology this thread was actually written by, rather
            # than assuming. Building the wrong one lets LangGraph recompute
            # `next` against a shape the run never used.
            graph = build(cp, with_approval=True)
            if not shape_matches(graph.get_state(config), graph):
                graph = build(cp, with_approval=False)
            snap = graph.get_state(config)
            if not shape_matches(snap, graph):
                print(f"Thread {tid} was written by a different graph "
                      f"({snap.values.get('shape')}). Refusing to resume it "
                      f"into this one.", file=sys.stderr)
                sys.exit(1)
            if not snap.next:
                print(f"Thread {tid} already finished.", file=sys.stderr)
                print(snap.values.get("answer", ""))
                return
            question = snap.values.get("question", "")
            print(f"Resuming {tid} at: {', '.join(snap.next)}", file=sys.stderr)
            if pending_interrupt(snap):
                reply = input("Approve? [y/N] ")
                payload = Command(resume=reply)
            else:
                # A crash, not a pause. Passing None means "carry on from the
                # last checkpoint" — the completed subtasks are already in
                # state and will not be researched or paid for again.
                payload = None
            t0 = time.time()
            state = graph.invoke(payload, config)
            report(question, state, tid, time.time() - t0)
            return

        question = " ".join(args.question) or (
            "Plan me a Saturday in Toronto: somewhere to spend the morning, "
            "lunch nearby, and how to get between them."
        )
        graph = build(cp, with_approval=args.approve)
        tid = thread_id_for(graph, question)
        config = {"configurable": {"thread_id": tid}}
        snap = graph.get_state(config)

        print(f"Question: {question}", file=sys.stderr)
        print(f"Thread:   {tid}", file=sys.stderr)

        # Asking the same question again after a crash or a pause picks up
        # where it stopped instead of paying for the finished subtasks twice.
        if pending_interrupt(snap):
            # Don't answer the approval question on the traveller's behalf
            # just because they typed the command again.
            print(f"\nThis run is paused waiting for approval.\n"
                  f"  python graph.py --resume {tid}", file=sys.stderr)
            return
        if snap.next:
            done = len(snap.values.get("results", []))
            print(f"Resuming an unfinished run at {', '.join(snap.next)} "
                  f"({done} subtask(s) already done)", file=sys.stderr)
            payload = None
        else:
            payload = {"question": question}
        print(file=sys.stderr)

        t0 = time.time()
        state = graph.invoke(payload, config)

        if "__interrupt__" in state:
            payload = state["__interrupt__"][0].value
            print(f"\n  PAUSED — {len(payload['tasks'])} subtasks, roughly "
                  f"{payload['estimated_requests']} requests.", file=sys.stderr)
            print(f"  Resume with: python graph.py --resume {tid}",
                  file=sys.stderr)
            return

        report(question, state, tid, time.time() - t0)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
