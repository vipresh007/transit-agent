"""The LangGraph port: does it behave like crew.py, and does it buy anything?

    python tests/test_graph.py

Offline. No API key, no network. crew.decompose / research_one / synthesize
are replaced with counting stubs, so what's under test is the GRAPH — the
fan-out width, the reducer, failure tolerance, and the checkpointing that is
the only real reason to prefer this file over crew.py.

Skips cleanly if langgraph isn't installed, because stage 10 is optional and
the other suites must not start failing because of a missing dependency.
"""

import sqlite3
import sys
import tempfile
from pathlib import Path

from _harness import check, clean_env, install_fake_openai, section

install_fake_openai()
clean_env()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import graph as G                                      # noqa: E402
    from langgraph.types import Command, Send              # noqa: E402
    HAVE_LANGGRAPH = True
except ImportError as exc:
    print(f"  (skipped: {exc})")
    print("  install with: pip install langgraph langgraph-checkpoint-sqlite")
    HAVE_LANGGRAPH = False

if HAVE_LANGGRAPH:
    import crew                                            # noqa: E402


CALLS = {"decompose": 0, "research": [], "synthesize": 0}


def stub(tasks, fail_on=(), answer="MERGED"):
    """Replace the three model-calling helpers with counting fakes."""
    CALLS["decompose"] = 0
    CALLS["research"] = []
    CALLS["synthesize"] = 0

    def decompose(question, verbose=True):
        CALLS["decompose"] += 1
        return list(tasks)

    def research_one(index, task, verbose):
        CALLS["research"].append(index)
        if index in fail_on:
            raise RuntimeError(f"subtask {index} exploded")
        return {"index": index, "task": task, "answer": f"A{index}",
                "error": None, "seconds": 0.0, "flags": {}, "events": []}

    def synthesize(question, results):
        CALLS["synthesize"] += 1
        return answer + "".join(r["answer"] for r in results)

    crew.decompose = decompose
    crew.research_one = research_one
    crew.synthesize = synthesize


def saver(path):
    conn = sqlite3.connect(path, check_same_thread=False)
    return G.SqliteSaver(conn), conn


def run(graph, question="Q", tid="t1", payload=None):
    config = {"configurable": {"thread_id": tid}}
    return graph.invoke(payload if payload is not None
                        else {"question": question}, config), config


# ---------------------------------------------------------------------------

def test_shape():
    section("the graph's shape")

    mermaid = G.build().get_graph().draw_mermaid()
    for node in ("plan", "research", "synthesize"):
        check(f"{node} is a node", node in mermaid)
    check("approval isn't there unless asked for", "approve" not in mermaid)

    approving = G.build(with_approval=True).get_graph().draw_mermaid()
    check("--approve adds the pause", "approve" in approving)
    check("--approve adds a cancel path", "cancelled" in approving)


def test_fan_out_width():
    section("fan-out is decided at runtime, not build time")

    for n in (1, 3, 4):
        sends = G.fan_out({"tasks": [f"t{i}" for i in range(n)]})
        check(f"{n} tasks produce {n} branches", len(sends), n)
        check("each branch is a Send to research",
              all(isinstance(s, Send) and s.node == "research" for s in sends))

    # A branch sees only its own payload — that IS the context isolation, and
    # here it's structural rather than a convention.
    one = G.fan_out({"tasks": ["only"]})[0]
    check("a branch carries just its subtask", sorted(one.arg), ["index", "task"])


def test_reducer_keeps_every_branch():
    section("concurrent writers to one list")

    stub(["a", "b", "c"])
    cp, conn = saver(":memory:")
    try:
        state, _ = run(G.build(cp))
        # Without Annotated[..., operator.add] the last branch to finish would
        # overwrite the other two, silently, with no error anywhere.
        check("all three results survive", len(state["results"]), 3)
        check("every subtask ran once", sorted(CALLS["research"]), [0, 1, 2])
        check("synthesis ran once", CALLS["synthesize"], 1)
        check("the answer is the merged one", state["answer"].startswith("MERGED"))
    finally:
        conn.close()


def test_one_failure_does_not_lose_the_rest():
    section("partial failure")

    # crew.research_one catches its own exceptions, so simulate the real
    # thing: an error recorded in the result, not raised out of the node.
    stub(["a", "b"])
    original = crew.research_one

    def flaky(index, task, verbose):
        if index == 1:
            return {"index": 1, "task": task, "answer": "", "error": "boom",
                    "seconds": 0.0, "flags": {}, "events": []}
        return original(index, task, verbose)

    crew.research_one = flaky
    cp, conn = saver(":memory:")
    try:
        state, _ = run(G.build(cp))
        check("both branches reported", len(state["results"]), 2)
        errors = [r for r in state["results"] if r["error"]]
        check("the failure is recorded, not swallowed", len(errors), 1)
        check("synthesis still ran", CALLS["synthesize"], 1)
    finally:
        conn.close()


def test_resume_does_not_repay_for_finished_work():
    section("checkpointing — the actual reason this file exists")

    # THIS IS THE CLAIM UNDER TEST. crew.py loses an entire run to one
    # exception; the pitch for LangGraph is that finished branches are
    # already durably recorded and are not researched again on resume.
    # If this check fails, the port is fine and the pitch is wrong — which
    # is worth knowing either way.
    with tempfile.TemporaryDirectory() as tmp:
        db = str(Path(tmp) / "cp.db")

        stub(["a", "b", "c"], fail_on={1})
        cp, conn = saver(db)
        try:
            crashed = False
            try:
                run(G.build(cp), tid="resume-me")
            except Exception:                              # noqa: BLE001
                crashed = True
            check("an unhandled subtask error stops the run", crashed)
            first_pass = list(CALLS["research"])
        finally:
            conn.close()

        # Second attempt: nothing fails this time.
        stub(["a", "b", "c"])
        cp, conn = saver(db)
        try:
            graph = G.build(cp)
            config = {"configurable": {"thread_id": "resume-me"}}
            snap = graph.get_state(config)
            check("the run is resumable", bool(snap.next))
            check("it isn't waiting on a human", G.pending_interrupt(snap), False)

            state = graph.invoke(None, config)
            check("the run completes on resume", len(state["results"]), 3)

            redone = [i for i in CALLS["research"] if i in first_pass and i != 1]
            check("subtasks that already succeeded are not re-run", redone, [])
            check("the one that failed is retried", 1 in CALLS["research"])
        finally:
            conn.close()


def test_rerunning_a_finished_question_starts_clean():
    section("re-asking a question that already finished")

    # The bug this pins: `results` has an operator.add reducer, so state is
    # append-only. Reusing a COMPLETED thread doesn't restart it — it appends,
    # and synthesis then merges the old run's sections alongside the new ones.
    # Six results, no error, no warning, just a wrong answer.
    stub(["a", "b", "c"])
    cp, conn = saver(":memory:")
    try:
        graph = G.build(cp)

        first = G.thread_id_for(graph, "same question")
        state, _ = run(graph, tid=first)
        check("the first run produces three results", len(state["results"]), 3)
        check("and the thread is finished",
              G.is_finished(graph.get_state(
                  {"configurable": {"thread_id": first}})))

        second = G.thread_id_for(graph, "same question")
        check("re-asking gets a different thread", second != first)

        stub(["a", "b", "c"])
        state, _ = run(graph, tid=second)
        check("the second run is three results, not six",
              len(state["results"]), 3)

        # An UNfinished thread must still be reused, or crash-resume by
        # re-asking stops working — which is the whole reason the id is
        # derived from the question rather than random.
        approving = G.build(cp, with_approval=True)
        paused_id = G.thread_id_for(approving, "pause here")
        approving.invoke({"question": "pause here"},
                         {"configurable": {"thread_id": paused_id}})
        check("a paused thread is offered again",
              G.thread_id_for(approving, "pause here"), paused_id)
    finally:
        conn.close()


def test_topology_change_cannot_hijack_a_thread():
    section("a checkpoint outliving the code that wrote it")

    # What actually happened on a real run: --approve paused at `approve`,
    # then the next command built the graph WITHOUT that node. LangGraph
    # recomputed `next` against the new shape and walked straight into
    # research. The pause silently ceased to exist. A checkpoint is not bound
    # to the graph that produced it, so the shape has to be part of the key.
    stub(["a", "b", "c"])
    cp, conn = saver(":memory:")
    try:
        approving = G.build(cp, with_approval=True)
        plain = G.build(cp)
        check("the two graphs have different shapes",
              G.shape_of(approving) != G.shape_of(plain))

        paused = G.thread_id_for(approving, "same words")
        approving.invoke({"question": "same words"},
                         {"configurable": {"thread_id": paused}})
        check("the approving run paused before researching", CALLS["research"], [])

        # The bug: the plain graph must not be handed the paused thread.
        plain_tid = G.thread_id_for(plain, "same words")
        check("the same question on a different shape gets its own thread",
              plain_tid != paused)

        snap = approving.get_state({"configurable": {"thread_id": paused}})
        check("the state records which graph wrote it",
              snap.values.get("shape"), G.shape_of(approving))
        check("and that state is refused by the other graph",
              G.shape_matches(snap, plain), False)
        check("while its own graph still accepts it",
              G.shape_matches(snap, approving))
    finally:
        conn.close()


def test_approval_can_stop_the_spending():
    section("interrupts")

    stub(["a", "b", "c"])
    cp, conn = saver(":memory:")
    try:
        graph = G.build(cp, with_approval=True)
        config = {"configurable": {"thread_id": "approve-me"}}
        state = graph.invoke({"question": "Q"}, config)

        check("the graph paused", "__interrupt__" in state)
        payload = state["__interrupt__"][0].value
        check("it says how many subtasks", len(payload["tasks"]), 3)
        check("and roughly what they'll cost",
              payload["estimated_requests"], 3 * G.REQUESTS_PER_SUBTASK + 2)
        check("nothing was researched before asking", CALLS["research"], [])
        check("the pause is detectable from the snapshot",
              G.pending_interrupt(graph.get_state(config)))

        # Declining must cost nothing beyond the one planning call.
        final = graph.invoke(Command(resume="no"), config)
        check("declining runs no subtasks", CALLS["research"], [])
        check("and says so", "Cancelled" in final["answer"])
        # The trap this file's docstring warns about: if interrupt() lived in
        # plan(), resuming would replay it and pay for a second model call.
        check("resuming did not re-run the planner", CALLS["decompose"], 1)
    finally:
        conn.close()


def test_approval_yes_proceeds():
    section("interrupts — the happy path")

    stub(["a", "b"])
    cp, conn = saver(":memory:")
    try:
        graph = G.build(cp, with_approval=True)
        config = {"configurable": {"thread_id": "yes-please"}}
        graph.invoke({"question": "Q"}, config)
        final = graph.invoke(Command(resume="yes"), config)
        check("approving runs every subtask", sorted(CALLS["research"]), [0, 1])
        check("and produces an answer", final["answer"].startswith("MERGED"))
        check("still only one planning call", CALLS["decompose"], 1)
    finally:
        conn.close()


if __name__ == "__main__":
    if not HAVE_LANGGRAPH:
        sys.exit(0)
    for fn in (test_shape, test_fan_out_width, test_reducer_keeps_every_branch,
               test_one_failure_does_not_lose_the_rest,
               test_resume_does_not_repay_for_finished_work,
               test_rerunning_a_finished_question_starts_clean,
               test_topology_change_cannot_hijack_a_thread,
               test_approval_can_stop_the_spending, test_approval_yes_proceeds):
        fn()
    from _harness import PASSED
    print(f"\n{PASSED['n']} checks passed")
