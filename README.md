# Toronto transit & trip planning agent

A learning project. Build an agent that plans a day in Toronto using real
transit schedules, real opening hours, and real weather — on entirely free
data, with no paid services anywhere in the stack.

The point isn't the trip planner. The point is that each stage forces you to
learn one concept properly, and every stage produces something that runs.

## Setup

```bash
git clone <your-repo-url>
cd transit-agent

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then paste your Gemini key into .env
python agent.py "what should I do in Toronto tomorrow?"
```

You should see tool calls printed to stderr, then an answer. If the agent
answers without calling any tools, that's the first thing worth debugging —
usually a tool description in `tools/registry.py` isn't specific enough.

Full setup, in order:

```bash
python load_gtfs.py                 # TTC schedule -> transit.db  (~3 min)
python optimize_db.py               # indexes the journey planner needs
ollama pull nomic-embed-text        # local embedding model
python load_guides.py               # Wikivoyage -> guides.db     (~6 min)
python tests/run_all.py             # verify, free
```

## Getting a Gemini API key (free, no card)

1. Go to https://aistudio.google.com/apikey and sign in with a Google account.
2. Click **Create API key**. Pick an existing Google Cloud project or let it
   make one. You do not need to enable billing.
3. Copy the key into `.env`.

Free tiers move constantly — three model names in this project broke within a
week. Run `python list_models.py` to see what your keys actually serve, and
avoid `-latest` aliases: they resolve to the newest model, which is the one
with the smallest free quota.

Providers are configured in `.env` and tried in order, so a quota wall fails
over instead of ending the run. `PROVIDER=groq` pins one (and makes the
response cache usable, since the model name is part of the cache key).

## Files

| File | What's in it |
|---|---|
| `agent.py` | The loop and its guardrails. Start here. |
| `providers.py` | Which model, failover order, pacing. Accessors, not globals. |
| `llm.py` | Retries, rate limits, quota, token accounting. |
| `cache.py` | Replaying identical requests during iteration. |
| `trace.py` | The JSON record written to `traces/` on every run. |
| `tools/geo.py` | Geocoding, weather, POIs (external APIs). |
| `tools/transit.py` | GTFS schema access and stop/trip lookups. |
| `tools/journey.py` | End-to-end journey planning with transfers. |
| `tools/guides.py` | Hybrid retrieval over the Wikivoyage guides. |
| `embeddings.py` | Provider-agnostic embedding (Ollama, Mistral, Gemini). |
| `tools/registry.py` | Tool schemas + dispatch table. |
| `schemas.py` | The typed `Itinerary` and its validators. |
| `plan.py` | Two-phase research → structured itinerary. |
| `constraints.py` | Is the itinerary possible? Checked against the DB. |
| `grounding.py` | Do the answer's specifics trace to retrieved text? |
| `evals.py` | Test suite, plus `--selftest` for the checkers. |

Scripts: `load_gtfs.py` (build the transit DB), `load_guides.py` (build the
guide index), `optimize_db.py` (add indexes),
`list_models.py` (what your keys can actually use).

## Tests

```bash
python tests/run_all.py     # everything offline: no API key, no quota
```

Three suites, all free:

| Suite | Covers |
|---|---|
| `tests/test_tools.py` | tool logic, GTFS time arithmetic, SQL guardrails |
| `tests/test_agent.py` | retries, failover, throttling, the loop's guardrails |
| `evals.py --selftest` | whether the eval checkers themselves work |

`tests/_harness.py` stubs the OpenAI SDK before any project module loads, so
the loop runs against a scripted fake model. That's the only way to test
things like "the model emitted malformed tool-call JSON" or "the provider ran
out of daily quota mid-conversation" on demand.

Every check in `test_agent.py` corresponds to a bug that actually happened.
The comments say which.

Not included, because they cost something:

```bash
python tests/smoke_test.py   # hits Nominatim/Open-Meteo/Overpass
python evals.py              # spends model quota (~35 requests)
```

**Why this shape.** `agent.py` and `tools.py` both hit ~950 lines doing five
jobs each. The split isn't cosmetic: `providers.py` exposes `current()` and
`model()` as *functions* because failover reassigns the active provider at
runtime, and the old module-level `provider` variable produced a real bug —
`from agent import provider` bound the stale value and the footer reported
the wrong model for a whole run. Functions can't go stale.

## Free data sources

| Source | Use | Key needed |
|---|---|---|
| [Nominatim](https://nominatim.openstreetmap.org) | Geocoding | No (1 req/sec, set a User-Agent) |
| [Open-Meteo](https://open-meteo.com) | Weather forecast | No |
| [Overpass](https://overpass-api.de) | POIs, opening hours | No (few hundred queries/day) |
| [TTC GTFS](https://open.toronto.ca/dataset/ttc-routes-and-schedules/) | Toronto schedules | No — a zip of CSVs |
| [TTC GTFS-RT](https://open.toronto.ca/dataset/ttc-gtfs-realtime-gtfs-rt/) | Live alerts & delays | No |
| [Transitland](https://www.transit.land) | Feeds for other cities | Free key |
| [Wikivoyage dumps](https://dumps.wikimedia.org/enwikivoyage/) | Travel guide text (RAG corpus) | No |

## Roadmap

Each stage is roughly a sitting, and each one teaches exactly one thing.

**Stage 1 — the agent loop** ✅ *(this code)*
Tool calling, dispatch, feeding errors back to the model, a step cap.
Read `agent.py` top to bottom before moving on.

**Stage 2 — GTFS in SQLite** ✅
`python load_gtfs.py` pulls the current TTC feed and loads it into
`transit.db`. The agent gets two new tools: `describe_transit_schema` and
`query_transit` (read-only SELECT). It writes its own SQL, gets errors back as
text, and retries — self-correction with a real feedback signal.

**Stage 4 — evaluation** ✅ *(done early, out of order — and it should have been)*
`python evals.py`. Six cases with ground truth computed from the database at
run time, not hardcoded, so the suite survives the TTC's six-week republish.
Includes a hallucination check (asks for fare data the feed doesn't contain)
and a known-failing case (`calendar_dates` exceptions).

**Stage 3 — structured output** ✅
`python plan.py "how do I get from A to B?"`. Two phases: research with tools,
then a toolless structuring pass that emits JSON validated against the
`Itinerary` model in `schemas.py`. Validation errors are fed back with the
offending field named — self-correction against a perfect grader.

**Stage 5 — RAG over Wikivoyage** ✅
`ollama pull nomic-embed-text` then `python load_guides.py`. Chunks on section
boundaries (not character counts), embeds locally, stores vectors as blobs in
SQLite. `search_guides` does hybrid retrieval — dense + FTS5, fused with
Reciprocal Rank Fusion — with a relevance floor so out-of-corpus questions
return "nothing here" instead of the nearest weak match.

Measured, not assumed: `python tests/check_retrieval.py` runs probes and an
ablation. On this corpus the two retrievers pick different top results on 5 of
9 probes — dense alone answered "somewhere to eat late at night" with a bed &
breakfast section, having latched onto "night".

**Stage 6 — agentic RAG** ✅
`search_guides` now reports a quality band (strong / moderate / weak /
nothing-relevant) plus `suggested_terms` drawn from real section headings, so
the agent can react to bad retrieval instead of treating a 0.56 match like a
0.83 one. `grounding.py` checks whether an answer's specifics trace back to
what the tools returned, and `agent.py` pushes back once if they don't.

The honest finding: query rewriting earned less than expected. Asked about a
"rainy day" — a phrase appearing zero times in the corpus — the model expanded
to "museum gallery indoor attractions" on its own and scored 0.74 first try.
Modern models do query expansion natively.

Grounding earned much more. That same run named nine venues found nowhere in
the retrieved text (Eaton Centre, Mirvish, Yorkdale...) and closed with "all
of the venues above are listed in the Wikivoyage guides, so you can trust the
opening-hour details". Inventing venues is bad; asserting a source for the
invention is worse, and it's why `PROVENANCE_CLAIM` exists.

**Stage 7 — replanning against real constraints** ✅
`constraints.py` verifies an itinerary against the schedule and the
traveller's preferences: does that departure exist, is one minute enough to
change vehicles, can a person walk 1km in 2 minutes, does the route run today.
Violations go back to the agent WITH TOOLS (`repair()` in `plan.py`), because
fixing "only 1 min to make the 504" needs a new lookup, not a reworded JSON.

The repair loop only accepts a fix that reduces the violation count — without
that, a "repair" trading two problems for three passes as progress.

Preferences come from `.env`: `PREF_EARLIEST`, `PREF_LATEST`,
`PREF_MIN_TRANSFER`, `PREF_MAX_TRANSFERS`, `PREF_AVOID`. They're stated in the
prompt *and* checked afterwards — stating them avoids repair rounds, checking
them catches the times stating didn't work.

**Stage 8 — memory**
Persist preferences across sessions. "No early mornings", "vegetarian",
"I'd rather walk than take a bus."

**Stage 9 — multi-agent**
One researcher per day of the trip running in parallel, a synthesizer merging
them into one itinerary. Now compare hand-rolled vs LangGraph and you'll have
earned an opinion.

## Gotchas hit so far

**`400 Function call is missing a thought_signature`**
Gemini's thinking models sign their reasoning and attach it to tool calls in
`tool_calls[N].extra_content.google.thought_signature` — a field the OpenAI
schema knows nothing about. Rebuilding the assistant message by hand drops it,
and the next turn is rejected. Fix: append `message.model_dump()` instead, so
provider extras round-trip. Escape hatch: `THINKING_BUDGET=0` in `.env`.

This is the general shape of the tradeoff with compatibility layers — portable
code, but provider-specific features leak. Worth remembering before you assume
"OpenAI-compatible" means drop-in.

## Notes

- Nominatim and Overpass are volunteer-run. Cache aggressively, don't hammer them.
- GTFS times can exceed 24:00:00 (a 25:30:00 departure is 1:30am the next
  service day). This will bite you in stage 2.
- Keep `.env` out of git. The `.gitignore` handles it, but check `git status`
  before your first push anyway.
