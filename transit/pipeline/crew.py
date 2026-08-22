"""Stage 9: several agents on one question.

    python crew.py "plan me a Saturday in Toronto: morning, lunch, afternoon"

Three phases, and each exists for a reason:

  PLAN       one model call decomposes the question into independent subtasks
  RESEARCH   each subtask runs as its own agent, in its own thread, with its
             own fresh context
  SYNTHESIS  one model call merges the results into a single answer

WHY THIS IS WORTH ANYTHING: context isolation. A single agent planning a whole
day accumulates every tool result from every part of it, and by the afternoon
its context is mostly morning. Each subagent here starts clean and sees only
what its own subtask needed. Parallelism is a bonus; isolation is the point.

WHAT IT COSTS: N times the requests, plus two more for planning and synthesis.
A four-part question that a single agent could answer in 8 calls becomes
roughly 4x6 + 2 = 26. Multi-agent is not a general upgrade — it's a trade of
tokens for context cleanliness, and it only pays when the parts are genuinely
independent.

WHEN NOT TO USE IT: if subtask 2 needs subtask 1's answer, this design is
wrong. "Where should I eat, and how do I get there from the museum?" is
sequential — the route depends on which restaurant. Decomposition into
PARALLEL parts requires the parts to be parallel, and the planner is
instructed to refuse rather than fake it.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

load_dotenv()

from transit.core import agent          # noqa: E402
from transit.verify import constraints    # noqa: E402
from transit.verify import grounding      # noqa: E402
from transit.core import llm            # noqa: E402
from transit.tools import memory         # noqa: E402
from transit.core import providers      # noqa: E402
from transit.core import trace          # noqa: E402

# Each subtask is a full agent run. Four is already ~26 requests.
MAX_SUBTASKS = int(os.getenv("MAX_SUBTASKS", "4"))

PLAN_PROMPT = """\
Break this request into INDEPENDENT subtasks that can be researched in
parallel, each by a separate assistant that will not see the others' work.

REQUEST: {question}

Rules:
- Each subtask must be answerable ALONE. If subtask B needs subtask A's
  answer, they are not independent — merge them into one subtask.
- Between 1 and {max_subtasks} subtasks. One is a perfectly good answer for a
  simple question; splitting a simple question wastes requests for nothing.
- Each subtask should be a complete question, with the city and any time
  constraints restated, because the assistant answering it sees no other
  context.

Return ONLY a JSON array of strings. No prose.
Example: ["What are the best museums in Toronto?", "Where can I eat lunch
downtown Toronto around 1pm?"]
"""

SYNTHESIS_PROMPT = """\
Combine these independently-researched answers into ONE answer to the
original request.

ORIGINAL REQUEST: {question}

{sections}

Your job is to CHOOSE, not to concatenate. Each section researched its part
without seeing the others, so each returned a list of candidates. The reply
must be a single recommendation.

Rules:
- Pick ONE option per part of the request. Say in one line why you picked it,
  using only what the sections report. List the rest under "Other options"
  as bare names, no detail.
- Prefer choices that fit together. If the sections give locations, favour
  ones that are near each other, and say when they are not.
- Reproduce any detail you DO keep exactly as researched — times, routes,
  addresses, prices. Never round or tidy a time. Dropping a candidate is
  fine; altering one is not.
- Add NO facts of your own. You have no tools here. That includes joining
  material: do not state which streetcar connects two places, which direction
  one is from another, or how long anything takes, unless a section says so.
  Invented connective tissue is the most common error at this step.
- If two sections disagree, say so rather than silently picking one.
- Preserve caveats. A section that flagged something unverified must keep
  that flag.
"""


def decompose(question: str, verbose: bool = True) -> list[str]:
    """One model call to split the question. Falls back to not splitting."""
    response = llm.call_model(
        [{"role": "user", "content": PLAN_PROMPT.format(
            question=question, max_subtasks=MAX_SUBTASKS)}],
        verbose=verbose, use_tools=False,
    )
    raw = response.choices[0].message.content or ""
    match = re.search(r"\[.*\]", raw, re.S)
    if not match:
        return [question]

    try:
        tasks = json.loads(match.group(0))
    except json.JSONDecodeError:
        return [question]

    tasks = [str(t).strip() for t in tasks if str(t).strip()][:MAX_SUBTASKS]
    # A planner that returns nothing usable must not silently produce an empty
    # crew — fall back to answering the question directly.
    return tasks or [question]


def research_one(index: int, task: str, verbose: bool) -> dict:
    """Run one subtask as a fully independent agent.

    Thread-local state makes this safe: LAST_RUN and trace.EVENTS belong to
    this thread alone. Before that change, concurrent agents wrote into the
    same dict and list, producing a trace that mixed conversations and flags
    that belonged to neither.
    """
    started = time.time()
    trace.reset()
    agent.LAST_RUN.reset()

    try:
        answer = agent.run(task, verbose=verbose, require_grounding=True)
        error = None
    except Exception as exc:                      # noqa: BLE001
        answer, error = "", f"{type(exc).__name__}: {exc}"

    return {
        "index": index,
        "task": task,
        "answer": answer,
        "error": error,
        "seconds": round(time.time() - started, 1),
        "flags": agent.LAST_RUN.snapshot(),
        "events": trace.EVENTS.snapshot(),
    }


def research(tasks: list[str], verbose: bool = True) -> list[dict]:
    """Run every subtask concurrently, tolerating individual failures."""
    if len(tasks) == 1:
        return [research_one(0, tasks[0], verbose)]

    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(len(tasks), 4)) as pool:
        futures = {
            pool.submit(research_one, i, t, verbose): i
            for i, t in enumerate(tasks)
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if verbose:
                status = "failed" if result["error"] else "done"
                print(f"  [subtask {result['index'] + 1}] {status} "
                      f"in {result['seconds']}s", file=sys.stderr)

    # One subtask failing must not lose the others. Partial results with a
    # note beat an exception that discards work already paid for.
    return sorted(results, key=lambda r: r["index"])


def synthesize(question: str, results: list[dict], verbose: bool = True) -> str:
    """Merge subtask answers. No tools — synthesis must not invent."""
    usable = [r for r in results if r["answer"] and not r["error"]]
    if not usable:
        return "Every subtask failed:\n" + "\n".join(
            f"  - {r['task']}: {r['error']}" for r in results)

    if len(usable) == 1:
        return usable[0]["answer"]

    sections = "\n\n".join(
        f"--- SECTION {r['index'] + 1}: {r['task']}\n{r['answer']}"
        for r in usable
    )
    failed = [r for r in results if r["error"]]
    if failed:
        sections += "\n\n--- NOT RESEARCHED (say so in the answer):\n" + "\n".join(
            f"  - {r['task']}" for r in failed)

    response = llm.call_model(
        [{"role": "user", "content": SYNTHESIS_PROMPT.format(
            question=question, sections=sections)}],
        verbose=verbose, use_tools=False,
    )
    return response.choices[0].message.content or sections


def audit(answer: str, results: list[dict]) -> dict:
    """Ground the merged answer at two levels, because they catch different lies.

    RESEARCH grounding compares the answer to the raw tool results. Every
    subagent already does this on its own output, so at the crew level it is
    mostly redundant — and it is also too forgiving. The union of three
    subtasks' tool output is a large haystack, so a sentence assembled from
    words scattered across three unrelated sources scores as supported.

    SYNTHESIS grounding compares the merged answer to the SECTION ANSWERS.
    That is the check that was missing. The synthesiser has no tools, so
    anything in the final text that no section said is invention — and it is
    invention of a specific, plausible kind: joining material. A real run
    produced "walk north under the Gardiner to reach the Distillery District"
    (it is east) and "509 or 510 from Union Station" (the 510 does not serve
    Union). Neither claim came from a section. Both passed research grounding
    at 87%, which instead flagged 'Saturdays', 'Group' and 'African'.

    Two checks against different haystacks, because "did the research support
    this" and "did the merge stay faithful to the research" are not the same
    question and fail in different ways.
    """
    sources = [e["result"] for r in results for e in r["events"]
               if e["kind"] == "tool_call"]
    sections = [r["answer"] for r in results if r["answer"]]
    return {
        "research": grounding.check(answer, sources),
        "synthesis": grounding.check(answer, sections),
    }


def report_grounding(audited: dict) -> None:
    for level in ("research", "synthesis"):
        result = audited[level]
        if result["unsupported"]:
            print(f"  [grounding/{level}] {grounding.summary(result)}",
                  file=sys.stderr)


def main() -> None:
    question = " ".join(sys.argv[1:]) or (
        "Plan me a Saturday in Toronto: somewhere to spend the morning, "
        "lunch nearby, and how to get between them."
    )

    print(f"Question: {question}\n", file=sys.stderr)

    t0 = time.time()
    tasks = decompose(question)
    print(f"Decomposed into {len(tasks)} subtask(s):", file=sys.stderr)
    for i, t in enumerate(tasks, 1):
        print(f"  {i}. {t}", file=sys.stderr)
    print(file=sys.stderr)

    results = research(tasks)
    print("\nSynthesising...\n", file=sys.stderr)
    answer = synthesize(question, results)
    print(answer)

    print(file=sys.stderr)
    ground = audit(answer, results)
    report_grounding(ground)

    elapsed = time.time() - t0
    print(f"\n[{llm.usage_line()} | {len(tasks)} subtasks | {elapsed:.0f}s]",
          file=sys.stderr)

    path = trace.write(
        question, answer,
        provider=providers.current()["name"], model=providers.model(),
        usage=llm.USAGE, cache_stats={}, flags={"subtasks": len(tasks)},
        extra={
            "phase": "crew",
            "subtasks": [
                {k: v for k, v in r.items() if k != "events"} for r in results
            ],
            "grounding": ground,
        },
    )
    print(f"[trace: {path}]", file=sys.stderr)


if __name__ == "__main__":
    main()
