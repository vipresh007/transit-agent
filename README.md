# Toronto transit & trip planning agent

A learning project: an agent that plans a day in Toronto using real transit
schedules, opening hours and weather — on entirely free services.

The point isn't the trip planner. Each stage teaches one concept and produces
something that runs.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate.bat
pip install -r requirements.txt
cp .env.example .env                 # paste your Gemini key in

python scripts/load_gtfs.py          # TTC schedule -> data/transit.db  (~3 min)
python scripts/optimize_db.py        # indexes the journey planner needs
ollama pull nomic-embed-text         # local embedding model
python scripts/load_guides.py        # Wikivoyage -> data/guides.db     (~6 min)
python scripts/load_shapes.py        # optional: real track geometry for the map (~25s)
python tests/run_all.py              # verify, free
```

Or skip all of that and let the container build its own database:

```bash
docker build -t transit-agent .      # ~6 min, downloads the feed itself
docker run --rm -p 8000:8000 --env-file .env \
  -v "$PWD/traces:/app/traces:ro" transit-agent
```

Get a Gemini key at https://aistudio.google.com/apikey — no card needed.

**Budget the free tier before you plan your day around it.** Gemini's free
tier allowed **20 requests/day** for `gemini-3.6-flash`, and one `plan.py` run
costs 16-21 — so roughly one run per day, per model. Check what your key
actually serves with `python scripts/list_models.py`; a smaller model often
has a far larger daily allowance. Set `CACHE=1` and pin `PROVIDER=` while
iterating so repeated questions cost nothing.

Free tiers move constantly; three model names here broke within a week. Run
`python scripts/list_models.py` to see what your keys actually serve, and avoid
`-latest` aliases — they resolve to the newest model, which has the smallest
quota. Providers are tried in order and fail over on a quota wall;
`PROVIDER=groq` pins one (and makes the response cache usable).

## Which script do I run?

Layers, not alternatives — each calls the one below. Pick one per question.

| Run this | When | What it adds |
|---|---|---|
| `python agent.py "..."` | debugging a tool | the raw loop, prose answer |
| `python plan.py "..."` | one A-to-B journey | typed itinerary, constraint checks, repair, memory |
| `python crew.py "..."` | several *independent* questions | parallel subagents + synthesis |
| `python graph.py "..."` | same as crew, via LangGraph | checkpointing, approval pauses |
| `python serve.py` | the web app | live tool calls, timeline, journey map, free replay |
| `streamlit run ui.py` | the older front end | same data, Streamlit's chrome |

`crew.py` and `graph.py` call `agent.run()` directly, so they skip the typing
and schedule verification `plan.py` adds — their answers are grounded but not
checked against the timetable.

`python scripts/timing.py` breaks the last run down by where the wall clock
went; `--compare before.json after.json` makes an optimisation falsifiable.

Both front ends replay any saved trace — timeline, map, badges — with **zero
model requests**. On a 20-a-day quota, looking at the UI again shouldn't cost a
sixteenth of your budget.

**Why two front ends exist.** Streamlit owns its page: the header, hamburger
and Deploy button are the framework, and hiding them with CSS is fighting the
tool. `serve.py` is FastAPI plus hand-written HTML — full control, no chrome.
Both call `plan.plan()` and render `view.result_to_dict()`, so neither can
disagree with the other about whether an itinerary is trustworthy.

Everything else in `scripts/` runs once at setup. `tests/run_all.py` runs after code
changes. `python -m transit.pipeline.evals` spends quota; run it on purpose.

## Layout

```
agent.py  plan.py  crew.py  graph.py   three-line launchers, nothing else
transit/
  paths.py      every file the project reads or writes, absolute
  core/         agent, llm, providers, cache, trace, threadstate, embeddings,
                realtime (GTFS-RT vehicle positions)
  tools/        geo, transit, journey, guides, memory, registry
  verify/       constraints, grounding, schemas, gtfstime
  pipeline/     plan, crew, graph, evals, view (all render logic, testable)
serve.py        the web app  ->  http://127.0.0.1:8000
ui.py           the earlier Streamlit front end (optional)
transit/web/    FastAPI: 6 endpoints, SSE for live tool calls
assets/web/     index.html, app.css, app.js — hand-written, no build step
scripts/        load_gtfs, load_shapes, load_guides, optimize_db, list_models,
                timing, probe_rt (does the realtime feed join to our data?)
Dockerfile      two stages: build the database, then ship it
data/           databases and downloads (gitignored)
tests/
```

Start at `transit/core/agent.py` — the loop and its guardrails.

**Dependencies point one way:** `pipeline → verify → tools → core`, with two
deferred imports inside functions as documented exceptions. `memory.py` lives
in `tools/` because the agent calls `remember`/`recall` as tools; in
`pipeline/` it inverted the layering.

## Tests

```bash
python tests/run_all.py     # eleven suites, offline, no API key, no quota
```

`_harness.py` stubs the OpenAI SDK before any project module loads, so the loop
runs against a scripted fake model — the only way to test "the model emitted
malformed tool-call JSON" or "the provider hit its daily quota mid-conversation"
on demand. Every check in `test_agent.py` matches a bug that actually happened;
the comments say which.

A suite that can't run exits **3**, and the runner reports it as SKIPPED
rather than counting it as a pass — `test_graph.py` was silently skipping in
an environment without langgraph while a broken import sat in it.

`test_imports.py` is static and runs first. It exists because the package
reorganisation left `journey.py` calling `paths.readonly_uri()` without
importing `paths`. Every other suite at the time passed — the bug sat inside a
function body on the one path they skip, since exercising it needs a 4.2M-row
database. It surfaced as a live run that spent 36 requests inventing a midnight
departure. **Behavioural tests only cover code they run, and the expensive
paths are the ones they skip.**

`test_realtime.py` decodes the protobuf feed bytes saved by `probe_rt.py`, so
it exercises real data with no network. Its most useful checks assert what the
code *won't* do: `Vehicle` has no time field and never will, and the module
never decodes `stop_time_update`. See stage 11 for why.

`test_docker.py` is static text analysis of the Dockerfile. A build takes six
minutes to tell you a `COPY` path is wrong; this takes 0.2s. It also catches
the two mistakes that are invisible in a build log — a secret that could enter
a layer, and a `chown -R` after a `COPY`.

Cost money, so excluded: `tests/smoke_test.py` (hits the free APIs) and
`python -m transit.pipeline.evals` (~35 model requests).

**Never run live: `crew.py`.** Its tests use a scripted fake model, so they
prove the mechanics — decomposition, concurrent subagents, reassembly — and
not that a real model decomposes sensibly. Three concurrent agents against a
20-a-day quota is why.

## Stages

**1 — the agent loop.** Tool calling, dispatch, feeding errors back, a step cap.

**2 — GTFS in SQLite.** `describe_transit_schema` + `query_transit` (read-only).
The agent writes its own SQL, gets errors back as text, and retries.

**4 — evaluation** *(built early, out of order, and it should have been)*.
Ground truth computed from the database at run time, not hardcoded, so the
suite survives the TTC's six-week republish. Includes a hallucination check and
a known-failing case.

**3 — structured output.** Research with tools, then a toolless pass emitting
JSON validated against `Itinerary`. Validation errors go back with the
offending field named — self-correction against a perfect grader.

**5 — RAG over Wikivoyage.** Chunks on section boundaries, embeds locally,
hybrid dense + FTS5 fused with Reciprocal Rank Fusion, with a relevance floor so
out-of-corpus questions return "nothing here". Measured, not assumed:
`tests/check_retrieval.py` runs an ablation — the two retrievers disagree on 5
of 9 probes, and dense alone answered "somewhere to eat late at night" with a
bed & breakfast section.

**6 — agentic RAG.** Retrieval reports a quality band and `suggested_terms`, so
the agent can react to a bad result instead of treating 0.56 like 0.83.
Grounding checks whether specifics trace back to tool output.

Honest finding: query rewriting earned little — models expand queries natively.
Grounding earned a lot: one run named nine venues found nowhere in the retrieved
text and closed with "all of the venues above are listed in the guides, so you
can trust the opening hours". Inventing a venue is bad; asserting a source for
the invention is worse.

**7 — replanning against constraints.** Does that departure exist, is one minute
enough to change vehicles, can a person walk 1km in 2 minutes. Violations go back
to the agent *with tools*, because fixing "only 1 min to make the 504" needs a
new lookup. A repair is only accepted if it reduces the violation count.

**8 — memory.** Standing preferences merged into `Preferences` at plan time, so
a remembered "avoid buses" becomes an enforced constraint rather than more
prompt text. `scope` is required: `standing` persists, `trip` refuses loudly —
saving "I need to be there by 3pm" means every future journey silently inherits
a deadline nothing explains. Environment beats memory, always.

**9 — multi-agent.** Planner decomposes into independent subtasks, each runs as
its own agent with fresh context, a synthesiser merges them. **Context isolation
is the point; parallelism is a bonus.** Not a general upgrade: four subtasks cost
~26 requests where one agent might use 8. Forced `threadstate.py`, because two
subagents writing to the same module-level dict produced no crash — just a trace
mixing conversations.

**Tool scoping.** Every schema is resent on every call — 13 of them is ~1,910
tokens riding along 16-20 times a run, and a point-to-point question never
touches the weather. `agent.run(tools="journey")` offers 8 instead of 13,
saving ~740 tokens per call (~13,000 per run). `plan.py` uses it for both the
research and repair passes.

The trade is real: a tool the agent can't call is a move it can't make, so the
default stays everything and the evals decide whether narrowing cost anything.
The cache key hashes the schemas ACTUALLY sent — keying a narrowed request the
same as a full one would return the wrong answer confidently, which is worse
than a miss.

**10 — the same thing in LangGraph.** `pip install langgraph
langgraph-checkpoint-sqlite`, then `python graph.py --draw`.

```
START → plan ──fan_out──▶ research ×N ──▶ synthesize → END
                │
     (--approve) └─▶ approve ─┬─▶ research ×N
                              └─▶ cancelled → END
```

Buys: checkpointing (a run that dies at subtask 3 resumes at subtask 3),
`interrupt()` for approval before spending, state history, a generated diagram.
Costs: control flow becomes data, which is why `--draw` exists.

Two bugs it caused, both the same lesson — *durable state outlives the code that
made it, so anything the state assumes has to be part of its key*: an
`operator.add` reducer makes state append-only, so re-running a finished thread
merged six sections instead of three; and a checkpoint isn't bound to the graph
that wrote it, so a run paused at `approve` was resumed by a graph without that
node and the pause silently stopped existing.

Verdict: checkpointing is real. Nothing else here would improve the agent, and
the constraint checking, grounding and evals that make it trustworthy come from
none of it. Port something you've hand-written if you want to judge a framework.

**11 — real-time, and the number that killed most of it.** TTC publishes
GTFS-RT free, no key: vehicle positions, trip updates, service alerts. Before
building anything, `scripts/probe_rt.py` measured whether the feed joins to
`transit.db`:

| join | rate | verdict |
|---|---|---|
| `trip_id` | 0% | synthetic negative hashes — no fix exists |
| `stop_id` | 59.3% | **1.1% real.** The rest are integer collisions |
| `route_id` | 99.4% | genuine; 807/807 vehicles resolved |

That 59.3% is the whole lesson. A set-membership test answers *is this string
present*; it cannot answer *does this string mean the same thing*. Feed route
23 is Dawes Rd, and its "matched" stops resolved to Bathurst St. Built on that
join, every bus leg would show a live, precise, confidently wrong delay — and
a wrong number looks exactly like a right one.

So the feature is positions only. `Vehicle` has no time field, a test asserts
it never will, and another greps the module to ensure `stop_time_update` is
never decoded. "23 streetcars are on the 504, here is where they are" needs
route + coordinates and cannot be wrong about a time it never states.

The map reports per route, never summed: *11 on 510 · nothing running on 304*.
A total hid the zero, and the zero was the useful half.

**Containerising.** Two stages: build the database from the 34MB feed in a
container that gets thrown away, copy out the one finished file. It is built
rather than copied because 589MB doesn't belong in a build context, because
copying would freeze one laptop's schedule into every image, and mostly
because it proves the loaders run somewhere that isn't your machine.

Three things it found, none of which any test or log would have shown:

- `chown -R` after `COPY` **duplicated the 488MB database**. Overlay
  filesystems can't change a lower layer's metadata in place, so touching a
  file's owner copies the whole thing up — and layers are append-only. Use
  `COPY --chown=` and create the user first. 1.45GB → 849MB.
- The image never ran `ANALYZE`, so the container planned queries without
  `sqlite_stat1` while the laptop had it. Nothing fails; it's just slower,
  invisibly.
- `load_gtfs.py` was building three indexes no query plan could reach —
  each a strict prefix of a composite added later to fix a slow join.
  **129MB, 22% of the database.** `EXPLAIN QUERY PLAN` confirmed all four
  query shapes still use an index without them.

## Gotchas hit so far

**`400 Function call is missing a thought_signature`** — Gemini signs its
reasoning into `tool_calls[N].extra_content`, a field the OpenAI schema doesn't
know. Rebuilding the assistant message by hand drops it. Append
`message.model_dump()` instead. "OpenAI-compatible" is not drop-in.

**GTFS times aren't zero-padded** — 863,539 of 4.2M rows store `8:03:31`, so
comparing against `'08:03:31'` silently misses them. Use
`substr('0' || departure_time, -8)`.

**Whitelist provider fields** — Groq returns a `reasoning` field that 422s
Mistral on failover. A blacklist is always one release behind.

**Compute from mutable state at the point of use** — three bugs, one cause:
`from agent import provider` bound a stale value, `trace.EVENTS` was read after
a reset, and `sanitize()` ran once before the retry loop and resent a
Gemini-shaped payload to Mistral.

**A guard's precision matters more than its recall** — grounding flagged
markdown headings, and the agent complied by *deleting the walking legs*. A
checker that fires on correct answers is worse than none, because the agent
obeys it.

**Make absence representable** — zero rows vs "no service"; finished vs
established something; rate limit vs dead quota. `legs` had `min_length=1`, so
asked for a bus-free route to a bus-only destination the model emitted a
zero-minute walk and wrote *"a placeholder to satisfy schema requirements"*.

**Normalise before comparing strings** — Unicode broke comparison three times:
curly apostrophes, em-dashes, U+202F narrow no-break space.

**A disclaimer followed by the number is worse than either half** — asked
what a TTC fare costs, the agent said "the schedule has no fare data" and then
added "$3.35 (as of 2023)". Readers keep the number and drop the caveat. The
eval caught it; the fix is a prompt rule saying stop at the gap rather than
filling it from memory.

**With an exception hierarchy, handler order IS the logic** — Python picks
the first matching `except`, not the most specific. `RateLimitError`,
`BadRequestError` and `InternalServerError` all subclass `APIStatusError`, so
a 413 handler placed early swallowed 400s that should resample and 503s that
should retry. Catching a base class is only safe at the bottom. The test
harness stubbed these as flat `Exception` subclasses and passed either way —
**a double simpler than the real thing stops testing the part that's hard.**

**A retry delay that stops shrinking is a cap, not a queue** — a rolling
window drains, so each wait leaves less to wait. Gemini answered "retry in
57s" four times running against a 20/day quota; we slept nearly four minutes
to learn nothing. Watching whether the delay decreases beats parsing each
vendor's quota vocabulary, and works for vendors you haven't met.

**A library raises; only an entry point exits** — `providers.py` called
`sys.exit()` at import when no API key was set. `SystemExit` is a
`BaseException`, so `except Exception` guards sail past it. The Streamlit UI
imported `llm` (which reaches `providers`) before `agent` (which loaded
`.env`), so no key was set, the import exited, and the page hung on "loading…"
with a clean terminal. `.env` now loads in `transit/__init__.py` — config that
everything depends on can't be loaded by one of the things that depends on it.

**A default that's wrong doesn't announce itself** — `leg_shape()` defaulted
to `service_id="1"`, weekday service, so a leg on the 304 (all service 3) drew
a straight line while 1,917 shape points sat unused in the table. Rails don't
move on Sundays; filtering geometry by service day was never meaningful, only
accidental. Third time here a default has posed as a fact.

**Ask a tool to attribute a number to a cause** — `docker history` found
490MB spent on one word, `scripts/timing.py` found 52s of rejected requests
filed under "our own Python", and `probe_rt.py` found that a 59% match rate
was 1% agreement. Each number looked fine until something made it explain
itself.

**Windows** — `.ps1` does nothing in `cmd.exe` (use `activate.bat`); PowerShell's
`del` wants commas, not spaces; the Store Python alias can shadow the venv;
`winget`-installed `ollama` needs a fresh shell.

## Free data sources

| Source | Use | Key |
|---|---|---|
| [Nominatim](https://nominatim.openstreetmap.org) | Geocoding | No (1 req/sec, set a User-Agent) |
| [Open-Meteo](https://open-meteo.com) | Weather | No |
| [Overpass](https://overpass-api.de) | POIs, opening hours | No |
| [TTC GTFS](https://open.toronto.ca/dataset/ttc-routes-and-schedules/) | Toronto schedules | No |
| [TTC GTFS-RT](https://open.toronto.ca/dataset/ttc-bustime-real-time-next-vehicle-arrival-nvas/) | Live vehicle positions (surface only) | No |
| [Transitland](https://www.transit.land) | Feeds for other cities | Free key |
| [Wikivoyage dumps](https://dumps.wikimedia.org/enwikivoyage/) | RAG corpus | No |

Nominatim and Overpass are volunteer-run. Cache aggressively.

## Notes

- Keep `.env` out of git. If a key lands in a chat, screenshot or commit,
  **rotate it** — removing it later doesn't un-disclose it.
- `langgraph` is optional. Only `graph.py` imports it, and `test_graph.py`
  skips itself without it.
- Everything in `data/` is gitignored. `transit.db` and `guides.db` rebuild from
  the scripts; `memory.db` and `graph.db` do not.
